# -*- coding: utf-8 -*-
# Copyright (c) 2025 University Medical Center Göttingen, Germany.
# All rights reserved.
#
# Patent Pending: DE 10 2024 112 939.5
# SPDX-License-Identifier: LicenseRef-Proprietary-See-LICENSE
#
# This software is licensed under a custom license. See the LICENSE file
# in the root directory for full details.
#
# **Commercial use is prohibited without a separate license.**
# Contact MBM ScienceBridge GmbH (https://sciencebridge.de/en/) for licensing.

"""2D full-field sarcomere tracking (flow-predict + detection-snap).

Complements the 1D LOI-based tracker in :mod:`sarcasm.motion` by following every
sarcomere vector across the whole image automatically. The underlying dense
optical flow field (computed from two-channel distance transforms of the Z-band
and M-band masks) is also exposed as a first-class output, giving
tracking-independent motion quantification.

The tracker does **not** track M-bands as a separate entity. Instead each
"query point" represents one sarcomere center and is carried forward as follows:

1. **Flow engine.** Binary Z-band and M-band masks are converted to distance
   transforms. Farneback flow is computed on each channel and averaged,
   producing a dense `(H, W, 2)` displacement field per frame pair.
2. **Lagrangian prediction.** Each query point's pixel position is advected
   frame-to-frame by the flow.
3. **Snap-to-detection.** At every frame, each query point snaps to the
   nearest sarcomere detection that is *consistent* with its prediction —
   inside the anisotropic along-/perpendicular-to-sarcomere ellipse and with
   orientation compatible modulo π. If no consistent detection is found, the
   query point keeps its predicted position but records NaN for slen and
   orientation that frame.

Anti-convergence is enforced by the snap: detections sit at physical sarcomere
centres (~1 sarcomere apart), so snapping keeps neighbouring query points
anchored to different detections and they cannot collapse onto each other.

Soft assignment is allowed — multiple query points may snap to the same
detection. Unclaimed detections spawn new query points (appearance).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numba import njit
from scipy import ndimage
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angular_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a − b, wrapped to (−π/2, π/2].

    Sarcomere orientations are undirected axes (θ and θ+π describe the same
    line), so we wrap modulo π. Returns a value with magnitude at most π/2.
    """
    d = (a - b) % np.pi
    if d > np.pi / 2:
        d -= np.pi
    return float(d)


def _axial_similarity(a: float, b: float) -> float:
    """Axial similarity `cos(2·(a − b))` — the standard scalar for comparing
    orientations modulo π. Returns 1 for aligned, −1 for perpendicular.
    """
    return float(np.cos(2.0 * (a - b)))


# ---------------------------------------------------------------------------
# Flow engine
# ---------------------------------------------------------------------------

def _normalize_dt(dt: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """Clip + rescale a distance transform to uint8 for Farneback input."""
    x = np.clip(dt, 0.0, clip) / clip
    return (x * 255.0).astype(np.uint8)


def build_dt_channels(
    zbands_mask: np.ndarray,
    mbands_mask: np.ndarray,
    threshold: float = 0.5,
    clip: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the two-channel distance-transform representation used for flow.

    Returns ``(dt_z, dt_m)`` as uint8 arrays — each is 0 on the structure and
    grows with distance from it, clipped at ``clip`` pixels.
    """
    if zbands_mask.dtype != np.bool_:
        z = zbands_mask > threshold
    else:
        z = zbands_mask
    if mbands_mask.dtype != np.bool_:
        m = mbands_mask > threshold
    else:
        m = mbands_mask
    dt_z = ndimage.distance_transform_edt(~z)
    dt_m = ndimage.distance_transform_edt(~m)
    return _normalize_dt(dt_z, clip), _normalize_dt(dt_m, clip)


def compute_flow_farneback(
    dt_t: np.ndarray,
    dt_t1: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 4,
    winsize: int = 21,
    iterations: int = 3,
    poly_n: int = 7,
    poly_sigma: float = 1.5,
) -> np.ndarray:
    """Farneback flow on a single uint8 channel. Returns ``(H, W, 2)`` float32
    ``[dx, dy]`` in pixels (OpenCV convention)."""
    flow = cv2.calcOpticalFlowFarneback(
        dt_t, dt_t1, None,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma,
        flags=0,
    )
    return flow.astype(np.float32, copy=False)


def compute_flow_pair(
    zbands_t: np.ndarray,
    mbands_t: np.ndarray,
    zbands_t1: np.ndarray,
    mbands_t1: np.ndarray,
    threshold: float = 0.5,
    clip: float = 20.0,
    farneback_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Flow for one frame pair, averaged across the two DT channels.

    Returns ``(H, W, 2)`` float32, ``[dy, dx]`` in pixels (numpy row/col
    convention).
    """
    kw = farneback_kwargs or {}
    dt_z_t, dt_m_t = build_dt_channels(zbands_t, mbands_t, threshold, clip)
    dt_z_t1, dt_m_t1 = build_dt_channels(zbands_t1, mbands_t1, threshold, clip)
    flow_z = compute_flow_farneback(dt_z_t, dt_z_t1, **kw)  # [dx, dy]
    flow_m = compute_flow_farneback(dt_m_t, dt_m_t1, **kw)
    flow_xy = 0.5 * (flow_z + flow_m)
    flow = np.empty_like(flow_xy)
    flow[..., 0] = flow_xy[..., 1]  # dy
    flow[..., 1] = flow_xy[..., 0]  # dx
    return flow


def compute_flow_sequence(
    zbands_stack: np.ndarray,
    mbands_stack: np.ndarray,
    threshold: float = 0.5,
    clip: float = 20.0,
    farneback_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Flow for a full sequence. Returns ``(T-1, H, W, 2)`` float32 ``[dy,dx]``."""
    T = len(zbands_stack)
    if T < 2:
        raise ValueError("Need at least 2 frames to compute flow.")
    H, W = zbands_stack.shape[-2:]
    flows = np.empty((T - 1, H, W, 2), dtype=np.float32)
    for t in range(T - 1):
        flows[t] = compute_flow_pair(
            zbands_stack[t], mbands_stack[t],
            zbands_stack[t + 1], mbands_stack[t + 1],
            threshold=threshold, clip=clip,
            farneback_kwargs=farneback_kwargs,
        )
    return flows


# ---------------------------------------------------------------------------
# Motion-field sampling
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _sample_bilinear(flow: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear-interpolated flow lookup at subpixel (y, x) positions."""
    N = ys.shape[0]
    H = flow.shape[0]
    W = flow.shape[1]
    out = np.empty((N, 2), dtype=np.float32)
    for i in range(N):
        y = ys[i]
        x = xs[i]
        if y < 0 or x < 0 or y > H - 1 or x > W - 1:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            continue
        y0 = int(y)
        x0 = int(x)
        y1 = y0 + 1 if y0 < H - 1 else y0
        x1 = x0 + 1 if x0 < W - 1 else x0
        fy = y - y0
        fx = x - x0
        for c in range(2):
            v00 = flow[y0, x0, c]
            v01 = flow[y0, x1, c]
            v10 = flow[y1, x0, c]
            v11 = flow[y1, x1, c]
            out[i, c] = (
                v00 * (1 - fy) * (1 - fx)
                + v01 * (1 - fy) * fx
                + v10 * fy * (1 - fx)
                + v11 * fy * fx
            )
    return out


def sample_flow_bilinear(flow: np.ndarray, positions_px: np.ndarray) -> np.ndarray:
    """Bilinear-interpolated flow lookup. Positions in pixels (row, col)."""
    if positions_px.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ys = np.ascontiguousarray(positions_px[:, 0], dtype=np.float32)
    xs = np.ascontiguousarray(positions_px[:, 1], dtype=np.float32)
    return _sample_bilinear(flow, ys, xs)


def sample_flow_at_structures(
    flows: np.ndarray,
    positions_per_frame: List[np.ndarray],
    pixelsize: float,
) -> List[np.ndarray]:
    """Sample flow at per-frame structure positions. Returns per-frame
    displacement in µm; last frame is zero-filled (no outgoing flow)."""
    T = len(positions_per_frame)
    out: List[np.ndarray] = []
    for t in range(T):
        pos = positions_per_frame[t]
        if pos is None or len(pos) == 0:
            out.append(np.zeros((0, 2), dtype=np.float32))
            continue
        if t < T - 1:
            disp_px = sample_flow_bilinear(flows[t], np.asarray(pos, dtype=np.float32))
            out.append(disp_px * pixelsize)
        else:
            out.append(np.zeros((len(pos), 2), dtype=np.float32))
    return out


def decompose_along_perpendicular(
    displacement: np.ndarray,
    orientations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project displacement onto sarcomere orientation (along) and its
    perpendicular. Sarcomere axis = (sin θ, cos θ) in (row, col)."""
    if displacement.size == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    s = np.sin(orientations)
    c = np.cos(orientations)
    along = displacement[..., 0] * s + displacement[..., 1] * c
    perp = -displacement[..., 0] * c + displacement[..., 1] * s
    return along.astype(np.float32), perp.astype(np.float32)


def compute_motion_field_stats(
    flow_at_vectors: List[np.ndarray],
    orientations_per_frame: List[np.ndarray],
    frametime: Optional[float] = None,
) -> Dict[str, List[np.ndarray]]:
    """Per-frame motion-field summaries (magnitude + along/perp decomposition)."""
    mag: List[np.ndarray] = []
    along: List[np.ndarray] = []
    perp: List[np.ndarray] = []
    vel: List[np.ndarray] = []
    for disp, ori in zip(flow_at_vectors, orientations_per_frame):
        if disp is None or len(disp) == 0:
            empty = np.zeros(0, np.float32)
            mag.append(empty); along.append(empty); perp.append(empty); vel.append(empty)
            continue
        m = np.linalg.norm(disp, axis=1).astype(np.float32)
        a, p = decompose_along_perpendicular(disp, np.asarray(ori, np.float32))
        mag.append(m); along.append(a); perp.append(p)
        vel.append((m / frametime).astype(np.float32) if frametime else m.copy())
    return {
        'displacement_magnitude': mag,
        'displacement_along_sarcomere': along,
        'displacement_perpendicular': perp,
        'velocity_magnitude': vel,
    }


# ---------------------------------------------------------------------------
# Query-point tracker
# ---------------------------------------------------------------------------

def _anisotropic_snap(
    query_pos: Tuple[float, float],
    query_ori: float,
    detections: np.ndarray,          # (N, 2) row/col
    det_oris: np.ndarray,            # (N,)
    candidate_indices: np.ndarray,   # output of kdtree radius query
    max_along: float,
    max_perp: float,
    ori_tol_rad: float,
) -> int:
    """Return the index of the closest detection (by Euclidean distance) that
    passes both the anisotropic displacement and orientation gates, or -1."""
    if len(candidate_indices) == 0:
        return -1
    s = np.sin(query_ori)
    c = np.cos(query_ori)
    best = -1
    best_d2 = np.inf
    max_along2 = max_along * max_along
    max_perp2 = max_perp * max_perp
    qy, qx = query_pos
    for idx in candidate_indices:
        cy, cx = detections[idx]
        dy = cy - qy
        dx = cx - qx
        along = dy * s + dx * c
        perp = -dy * c + dx * s
        if along * along > max_along2 or perp * perp > max_perp2:
            continue
        if np.isfinite(det_oris[idx]) and ori_tol_rad < np.pi:
            d_ori = abs(_angular_diff(float(det_oris[idx]), query_ori))
            if d_ori > ori_tol_rad:
                continue
        d2 = dy * dy + dx * dx
        if d2 < best_d2:
            best_d2 = d2
            best = int(idx)
    return best


@njit(cache=True)
def _greedy_claim(
    qp_sorted: np.ndarray,
    det_sorted: np.ndarray,
    claimed_qp: np.ndarray,
    claimed_det: np.ndarray,
) -> np.ndarray:
    """Greedy assignment on (qp_idx, det_idx) pairs pre-sorted by ascending
    distance. Mutates ``claimed_qp`` / ``claimed_det`` in place and returns the
    indices (into the sorted pair arrays) of the accepted pairs.
    """
    n = qp_sorted.shape[0]
    out = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        qi = qp_sorted[i]
        dj = det_sorted[i]
        if claimed_qp[qi] or claimed_det[dj]:
            continue
        claimed_qp[qi] = True
        claimed_det[dj] = True
        out[k] = i
        k += 1
    return out[:k]


def _as_arr_padded(x, n: int, dtype=np.float32, fill=np.nan) -> np.ndarray:
    """Return an array of length ``n`` from ``x``, padding missing entries with
    ``fill``. Used so per-detection attribute arrays always align with the
    detections array even when the caller provides shorter lists."""
    if x is None or len(x) == 0:
        return np.full(n, fill, dtype=dtype)
    a = np.asarray(x, dtype=dtype)
    if len(a) >= n:
        return a[:n].copy() if a.dtype != dtype else a[:n]
    out = np.full(n, fill, dtype=dtype)
    out[:len(a)] = a
    return out


def track_sarcomere_vectors(
    zbands_stack: np.ndarray,
    mbands_stack: np.ndarray,
    pos_vectors_px_all: List[np.ndarray],
    midline_ids_all: List[np.ndarray],   # retained for API compat; not used here
    sarcomere_lengths_all: List[np.ndarray],
    orientations_all: List[np.ndarray],
    pixelsize: float,
    frametime: Optional[float] = None,
    threshold_mbands: float = 0.25,
    threshold_zbands: float = 0.5,
    dt_clip: float = 20.0,
    max_disp_along_px: float = 15.0,
    max_disp_perp_px: float = 2.0,
    ori_tol_deg: float = 45.0,
    memory: int = 5,
    min_track_length: int = 5,
    max_gap_interpolation: int = 5,
    compute_motion_field: bool = True,
    store_flow_fields: bool = False,
    farneback_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    """Run the 2D full-field tracker on a stack.

    The tracker does not persist M-band identity. Instead every query point is
    a sarcomere-centre marker that flow-advects and snaps to a consistent
    detection each frame. Outputs are dense ``(n_tracks, T)`` arrays.
    """
    T = len(zbands_stack)
    if T < 2:
        raise ValueError("Need at least 2 frames.")
    H, W = zbands_stack.shape[-2:]
    ori_tol_rad = float(np.deg2rad(ori_tol_deg))
    ori_similarity_threshold = float(np.cos(2.0 * ori_tol_rad))
    max_along2 = float(max_disp_along_px * max_disp_along_px)
    max_perp2 = float(max_disp_perp_px * max_disp_perp_px)
    max_radius = float(max_disp_along_px)

    logger.info("Computing optical flow sequence…")
    flows = compute_flow_sequence(
        zbands_stack, mbands_stack,
        threshold=max(threshold_zbands, threshold_mbands),
        clip=dt_clip,
        farneback_kwargs=farneback_kwargs,
    )

    # --- struct-of-arrays track state ---
    # History arrays (grow in chunks when capacity exceeded). Live state
    # (last_* / frames_since_snap / alive) mirrors only the tracks that are
    # still alive, but is kept at the same size as the history arrays so that
    # indexing by absolute slot is always valid.
    pos0_raw = pos_vectors_px_all[0]
    n0 = 0 if pos0_raw is None else len(pos0_raw)
    capacity = max(256, n0 * 4)

    positions_px = np.full((capacity, T, 2), np.nan, dtype=np.float32)
    tracks_slen = np.full((capacity, T), np.nan, dtype=np.float32)
    tracks_ori = np.full((capacity, T), np.nan, dtype=np.float32)
    tracks_snapped = np.zeros((capacity, T), dtype=bool)
    start_frame_arr = np.zeros(capacity, dtype=np.int32)
    last_y = np.full(capacity, np.nan, dtype=np.float32)
    last_x = np.full(capacity, np.nan, dtype=np.float32)
    last_ori = np.zeros(capacity, dtype=np.float32)
    frames_since_snap = np.zeros(capacity, dtype=np.int32)
    alive = np.zeros(capacity, dtype=bool)

    def _grow(needed: int):
        """Double capacity until >= ``needed`` and resize all SoA arrays."""
        nonlocal capacity, positions_px, tracks_slen, tracks_ori, tracks_snapped
        nonlocal start_frame_arr, last_y, last_x, last_ori, frames_since_snap, alive
        if needed <= capacity:
            return
        new_cap = capacity
        while new_cap < needed:
            new_cap *= 2

        def _resize(arr, fill):
            shape = (new_cap,) + arr.shape[1:]
            out = np.full(shape, fill, dtype=arr.dtype) if fill is not None else np.empty(shape, dtype=arr.dtype)
            out[:capacity] = arr
            return out

        positions_px = _resize(positions_px, np.nan)
        tracks_slen = _resize(tracks_slen, np.nan)
        tracks_ori = _resize(tracks_ori, np.nan)
        tracks_snapped = _resize(tracks_snapped, False)
        start_frame_arr = _resize(start_frame_arr, 0)
        last_y = _resize(last_y, np.nan)
        last_x = _resize(last_x, np.nan)
        last_ori = _resize(last_ori, 0.0)
        frames_since_snap = _resize(frames_since_snap, 0)
        alive = _resize(alive, False)
        capacity = new_cap

    n_tracks = 0

    # --- seed query points from frame 0 ---
    logger.info("Seeding query points from frame 0…")
    if n0 > 0:
        pos0 = np.asarray(pos0_raw, dtype=np.float32)
        slen0 = _as_arr_padded(sarcomere_lengths_all[0], n0, np.float32, np.nan)
        ori0_raw = _as_arr_padded(orientations_all[0], n0, np.float32, np.nan)
        # History/live orientation defaults to 0.0 for NaN/missing oris, matching
        # the original per-track seeding behaviour.
        ori0 = np.where(np.isfinite(ori0_raw), ori0_raw, 0.0).astype(np.float32)

        _grow(n0)
        sl = slice(0, n0)
        positions_px[sl, 0, 0] = pos0[:, 0]
        positions_px[sl, 0, 1] = pos0[:, 1]
        tracks_slen[sl, 0] = slen0
        tracks_ori[sl, 0] = ori0
        tracks_snapped[sl, 0] = True
        start_frame_arr[sl] = 0
        last_y[sl] = pos0[:, 0]
        last_x[sl] = pos0[:, 1]
        last_ori[sl] = ori0
        alive[sl] = True
        n_tracks = n0

    # --- frame-to-frame: advect, snap, spawn, close ---
    logger.info("Tracking frames…")
    for t in range(T - 1):
        flow = flows[t]

        # 1. advect every live track along its sarcomere axis only. Motion
        # perpendicular to the axis can only come from the snap residual,
        # hard-capped at max_disp_perp_px (anti-perpendicular-jump guarantee).
        live = np.flatnonzero(alive[:n_tracks])
        if live.size > 0:
            ys = last_y[live]
            xs = last_x[live]
            disp = _sample_bilinear(flow, ys, xs)
            ori_live = last_ori[live]
            s_live = np.sin(ori_live)
            c_live = np.cos(ori_live)
            along_live = disp[:, 0] * s_live + disp[:, 1] * c_live
            last_y[live] = ys + along_live * s_live
            last_x[live] = xs + along_live * c_live

        # 2. prepare detections for frame t+1
        dets_raw = pos_vectors_px_all[t + 1]
        dets = np.asarray(
            dets_raw if dets_raw is not None else np.zeros((0, 2)),
            dtype=np.float32,
        )
        n_det = len(dets)
        det_oris = _as_arr_padded(orientations_all[t + 1], n_det, np.float32, np.nan)
        det_slens = _as_arr_padded(sarcomere_lengths_all[t + 1], n_det, np.float32, np.nan)

        # Bool masks over the current track slots / detections so we can do
        # claim bookkeeping in pure numpy without Python sets.
        claimed_qp_mask = np.zeros(n_tracks, dtype=bool)
        claimed_det_mask = np.zeros(n_det, dtype=bool)

        # 3. build kd-tree and vectorized gate on all (qp, candidate) pairs.
        # Hard greedy assignment prevents query-point convergence — two qps
        # cannot snap onto the same detection and collapse.
        if n_det > 0 and live.size > 0:
            tree = cKDTree(dets)
            live_pos = np.column_stack((last_y[live], last_x[live]))
            neighbors = tree.query_ball_point(live_pos, r=max_radius)

            counts = np.fromiter(
                (len(n) for n in neighbors),
                dtype=np.int64,
                count=len(neighbors),
            )
            total = int(counts.sum())

            if total > 0:
                det_flat = np.empty(total, dtype=np.int64)
                offset = 0
                for n_list in neighbors:
                    k = len(n_list)
                    if k:
                        det_flat[offset:offset + k] = n_list
                        offset += k
                qp_flat_rel = np.repeat(
                    np.arange(live.size, dtype=np.int64), counts,
                )
                qp_abs = live[qp_flat_rel]

                qy = last_y[qp_abs]
                qx = last_x[qp_abs]
                qo = last_ori[qp_abs]
                cy = dets[det_flat, 0]
                cx = dets[det_flat, 1]
                co = det_oris[det_flat]

                finite_co = np.isfinite(co)
                # Orientation gate (axial similarity). NaN det ori ⇒ pass.
                sim = np.cos(2.0 * (co - qo))
                pass_ori = (~finite_co) | (sim >= ori_similarity_threshold)

                # Axial-average reference orientation via double-angle
                # arithmetic; fall back to qo where co is NaN.
                co_safe = np.where(finite_co, co, qo)
                s2 = np.sin(2.0 * qo) + np.sin(2.0 * co_safe)
                c2 = np.cos(2.0 * qo) + np.cos(2.0 * co_safe)
                axial_avg = 0.5 * np.arctan2(s2, c2)
                ref_ori = np.where(finite_co, axial_avg, qo)

                sref = np.sin(ref_ori)
                cref = np.cos(ref_ori)
                dy = cy - qy
                dx = cx - qx
                along_p = dy * sref + dx * cref
                perp_p = -dy * cref + dx * sref
                pass_pos = (along_p * along_p <= max_along2) & (perp_p * perp_p <= max_perp2)
                mask = pass_ori & pass_pos

                keep_pairs = np.flatnonzero(mask)
                if keep_pairs.size > 0:
                    dy_k = dy[keep_pairs]
                    dx_k = dx[keep_pairs]
                    d2 = (dy_k * dy_k + dx_k * dx_k).astype(np.float32)
                    qp_kept = qp_abs[keep_pairs].astype(np.int64)
                    det_kept = det_flat[keep_pairs].astype(np.int64)

                    order = np.argsort(d2, kind='stable')
                    qp_sorted = np.ascontiguousarray(qp_kept[order])
                    det_sorted = np.ascontiguousarray(det_kept[order])

                    accepted_idx = _greedy_claim(
                        qp_sorted, det_sorted,
                        claimed_qp_mask, claimed_det_mask,
                    )
                    qp_acc = qp_sorted[accepted_idx]
                    det_acc = det_sorted[accepted_idx]

                    # 4. write claimed tracks in a single vectorized shot
                    cy_acc = dets[det_acc, 0]
                    cx_acc = dets[det_acc, 1]
                    co_acc = det_oris[det_acc]
                    sl_acc = det_slens[det_acc]

                    positions_px[qp_acc, t + 1, 0] = cy_acc
                    positions_px[qp_acc, t + 1, 1] = cx_acc
                    tracks_slen[qp_acc, t + 1] = sl_acc
                    tracks_ori[qp_acc, t + 1] = co_acc
                    tracks_snapped[qp_acc, t + 1] = True
                    last_y[qp_acc] = cy_acc
                    last_x[qp_acc] = cx_acc
                    # Only overwrite last_orientation when detection ori is finite.
                    finite_det = np.isfinite(co_acc)
                    if finite_det.any():
                        last_ori[qp_acc[finite_det]] = co_acc[finite_det]
                    frames_since_snap[qp_acc] = 0

        # 5. unmatched live tracks → gap frame
        if live.size > 0:
            unclaimed_live_mask = ~claimed_qp_mask[live]
            unclaimed_live = live[unclaimed_live_mask]
            if unclaimed_live.size > 0:
                positions_px[unclaimed_live, t + 1, 0] = last_y[unclaimed_live]
                positions_px[unclaimed_live, t + 1, 1] = last_x[unclaimed_live]
                # tracks_slen / tracks_ori already NaN; tracks_snapped already False.
                frames_since_snap[unclaimed_live] += 1
                died_local = frames_since_snap[unclaimed_live] > memory
                if died_local.any():
                    alive[unclaimed_live[died_local]] = False

        # 6. unclaimed detections → new tracks (appearance)
        if n_det > 0:
            unclaimed_det = np.flatnonzero(~claimed_det_mask)
            n_new = unclaimed_det.size
            if n_new > 0:
                _grow(n_tracks + n_new)
                new_slots = np.arange(n_tracks, n_tracks + n_new, dtype=np.int64)
                new_y = dets[unclaimed_det, 0]
                new_x = dets[unclaimed_det, 1]
                new_slen = det_slens[unclaimed_det]
                new_ori_raw = det_oris[unclaimed_det]
                new_ori = np.where(
                    np.isfinite(new_ori_raw), new_ori_raw, 0.0,
                ).astype(np.float32)

                positions_px[new_slots, t + 1, 0] = new_y
                positions_px[new_slots, t + 1, 1] = new_x
                tracks_slen[new_slots, t + 1] = new_slen
                tracks_ori[new_slots, t + 1] = new_ori
                tracks_snapped[new_slots, t + 1] = True
                start_frame_arr[new_slots] = t + 1
                last_y[new_slots] = new_y
                last_x[new_slots] = new_x
                last_ori[new_slots] = new_ori
                frames_since_snap[new_slots] = 0
                alive[new_slots] = True
                n_tracks += n_new

    # --- filter short tracks (count of actual snaps) ---
    logger.info("Filtering short tracks…")
    snapped_counts = tracks_snapped[:n_tracks].sum(axis=1)
    keep_mask = snapped_counts >= int(min_track_length)
    kept = np.flatnonzero(keep_mask)
    n = int(kept.size)

    out_positions_px = positions_px[kept].copy()
    out_positions_um = (out_positions_px * pixelsize).astype(np.float32)
    out_tracks_slen = tracks_slen[kept].copy()
    out_tracks_ori = tracks_ori[kept].copy()
    out_tracks_snapped = tracks_snapped[kept].copy()
    out_start_frame = start_frame_arr[kept].astype(np.int64)
    out_track_ids = kept.astype(np.int64)
    out_track_lengths = snapped_counts[kept].astype(np.int64)

    result: Dict[str, object] = {
        'n_tracks': n,
        'track_ids': out_track_ids,
        'track_start_frame': out_start_frame,
        'track_lengths': out_track_lengths,
        'tracks_positions_um': out_positions_um,
        'tracks_positions_px': out_positions_px,
        'tracks_slen': out_tracks_slen,
        'tracks_orientations': out_tracks_ori,
        'tracks_snapped': out_tracks_snapped,
    }

    if compute_motion_field:
        pos_px_lists: List[np.ndarray] = []
        for p in pos_vectors_px_all:
            if p is None or len(p) == 0:
                pos_px_lists.append(np.zeros((0, 2), np.float32))
            else:
                pos_px_lists.append(np.asarray(p, dtype=np.float32))
        flow_at_vectors = sample_flow_at_structures(flows, pos_px_lists, pixelsize)
        stats = compute_motion_field_stats(
            flow_at_vectors,
            [o if o is not None else np.zeros(0, np.float32) for o in orientations_all],
            frametime=frametime,
        )
        result['flow_at_vectors'] = flow_at_vectors
        result.update(stats)

    if store_flow_fields:
        scales = np.zeros(len(flows), dtype=np.float32)
        quant = np.empty(flows.shape, dtype=np.int16)
        for i, f in enumerate(flows):
            m = float(np.max(np.abs(f))) if f.size else 0.0
            sc = (m / 32000.0) if m > 0 else 1.0
            scales[i] = sc
            quant[i] = np.clip(np.round(f / sc), -32000, 32000).astype(np.int16)
        result['flow_fields_int16'] = quant
        result['flow_fields_scales'] = scales

    return result


def compute_motion_field(
    zbands_stack: np.ndarray,
    mbands_stack: np.ndarray,
    pos_vectors_px_all: List[np.ndarray],
    orientations_all: List[np.ndarray],
    pixelsize: float,
    frametime: Optional[float] = None,
    threshold: float = 0.5,
    dt_clip: float = 20.0,
    farneback_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    """Flow + sampling without tracking. Useful for quick motion assessment."""
    flows = compute_flow_sequence(
        zbands_stack, mbands_stack,
        threshold=threshold, clip=dt_clip,
        farneback_kwargs=farneback_kwargs,
    )
    pos_px_lists: List[np.ndarray] = []
    for p in pos_vectors_px_all:
        if p is None or len(p) == 0:
            pos_px_lists.append(np.zeros((0, 2), np.float32))
        else:
            pos_px_lists.append(np.asarray(p, dtype=np.float32))
    flow_at_vectors = sample_flow_at_structures(flows, pos_px_lists, pixelsize)
    stats = compute_motion_field_stats(
        flow_at_vectors,
        [o if o is not None else np.zeros(0, np.float32) for o in orientations_all],
        frametime=frametime,
    )
    out: Dict[str, object] = {'flow_at_vectors': flow_at_vectors}
    out.update(stats)
    return out
