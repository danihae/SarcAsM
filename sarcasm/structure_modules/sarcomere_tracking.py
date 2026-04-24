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
from dataclasses import dataclass, field
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

@dataclass
class QueryPoint:
    """One sarcomere tracked across time via flow-predict + snap-to-detection.

    Positions are recorded for every frame the query point is alive (snapped
    or, during brief gaps, flow-predicted). ``sarcomere_lengths`` and
    ``orientations`` are NaN in gap frames; ``snapped`` records which frames
    had a real detection.
    """
    track_id: int
    start_frame: int
    positions_px: List[Tuple[float, float]] = field(default_factory=list)
    positions_um: List[Tuple[float, float]] = field(default_factory=list)
    sarcomere_lengths: List[float] = field(default_factory=list)
    orientations: List[float] = field(default_factory=list)
    snapped: List[bool] = field(default_factory=list)
    last_pos_px: Tuple[float, float] = (0.0, 0.0)
    last_orientation: float = 0.0
    frames_since_snap: int = 0
    alive: bool = True


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

    logger.info("Computing optical flow sequence…")
    flows = compute_flow_sequence(
        zbands_stack, mbands_stack,
        threshold=max(threshold_zbands, threshold_mbands),
        clip=dt_clip,
        farneback_kwargs=farneback_kwargs,
    )

    def _as_arr(x, dtype):
        if x is None or len(x) == 0:
            return np.zeros(0 if dtype != np.float32 else 0, dtype)
        return np.asarray(x, dtype=dtype)

    # --- seed query points from frame 0 ---
    logger.info("Seeding query points from frame 0…")
    query_points: List[QueryPoint] = []
    pos0 = np.asarray(pos_vectors_px_all[0] if pos_vectors_px_all[0] is not None else np.zeros((0, 2)),
                      dtype=np.float32)
    slen0 = _as_arr(sarcomere_lengths_all[0], np.float32)
    ori0 = _as_arr(orientations_all[0], np.float32)
    for i in range(len(pos0)):
        y = float(pos0[i, 0]); x = float(pos0[i, 1])
        o = float(ori0[i]) if i < len(ori0) and np.isfinite(ori0[i]) else 0.0
        s = float(slen0[i]) if i < len(slen0) else float('nan')
        qp = QueryPoint(track_id=len(query_points), start_frame=0)
        qp.positions_px.append((y, x))
        qp.positions_um.append((y * pixelsize, x * pixelsize))
        qp.sarcomere_lengths.append(s)
        qp.orientations.append(o)
        qp.snapped.append(True)
        qp.last_pos_px = (y, x)
        qp.last_orientation = o
        query_points.append(qp)

    # --- frame-to-frame: advect, snap, spawn, close ---
    logger.info("Tracking frames…")
    max_radius = float(max_disp_along_px)  # kd-tree query radius — use the larger of the two
    for t in range(T - 1):
        flow = flows[t]

        # 1. advect every live query point. Project the flow onto the track's
        # sarcomere orientation — only the along-sarcomere component moves the
        # query point. Motion perpendicular to the sarcomere axis can only
        # come from the snap residual, which is hard-capped at
        # max_disp_perp_px. This is the anti-perpendicular-jump guarantee:
        # flow cannot drag a track sideways even if tissue translates.
        live_idx = [i for i, qp in enumerate(query_points) if qp.alive]
        if live_idx:
            ys = np.array([query_points[i].last_pos_px[0] for i in live_idx], dtype=np.float32)
            xs = np.array([query_points[i].last_pos_px[1] for i in live_idx], dtype=np.float32)
            disp = _sample_bilinear(flow, ys, xs)
            for k, i in enumerate(live_idx):
                qp = query_points[i]
                dy = float(disp[k, 0]); dx = float(disp[k, 1])
                # Project flow onto sarcomere axis = (sin θ, cos θ).
                s = np.sin(qp.last_orientation)
                c = np.cos(qp.last_orientation)
                along = dy * s + dx * c
                # Advect by the along component only; zero out perpendicular.
                ny = float(ys[k] + along * s)
                nx = float(xs[k] + along * c)
                qp.last_pos_px = (ny, nx)

        # 2. build kd-tree on frame t+1 detections
        dets = np.asarray(pos_vectors_px_all[t + 1] if pos_vectors_px_all[t + 1] is not None else np.zeros((0, 2)),
                          dtype=np.float32)
        det_oris = _as_arr(orientations_all[t + 1], np.float32)
        det_slens = _as_arr(sarcomere_lengths_all[t + 1], np.float32)
        if len(dets) == 0:
            # no detections — every live qp goes into a gap frame
            for i in live_idx:
                qp = query_points[i]
                qp.positions_px.append(qp.last_pos_px)
                qp.positions_um.append((qp.last_pos_px[0] * pixelsize, qp.last_pos_px[1] * pixelsize))
                qp.sarcomere_lengths.append(float('nan'))
                qp.orientations.append(float('nan'))
                qp.snapped.append(False)
                qp.frames_since_snap += 1
                if qp.frames_since_snap > memory:
                    qp.alive = False
            continue
        tree = cKDTree(dets)

        # 3. collect all (qp_idx, det_idx, dist_sq) candidate pairs passing the
        # anisotropic + orientation gates, then greedy-assign by ascending
        # distance so each detection is claimed by at most one query point per
        # frame. Hard assignment prevents query-point convergence — two qps
        # cannot snap onto the same detection and collapse.
        all_pairs: List[Tuple[float, int, int]] = []  # (dist², qp_i, det_j)
        max_along2 = max_disp_along_px * max_disp_along_px
        max_perp2 = max_disp_perp_px * max_disp_perp_px
        ori_similarity_threshold = float(np.cos(2.0 * ori_tol_rad))
        for i in live_idx:
            qp = query_points[i]
            neigh = tree.query_ball_point(qp.last_pos_px, r=max_radius)
            if not neigh:
                continue
            qy, qx = qp.last_pos_px
            for idx in neigh:
                # Axial orientation gate: cos(2·Δφ) >= cos(2·ori_tol). This is
                # the proper similarity measure for undirected axes (mod π).
                det_ori_valid = idx < len(det_oris) and np.isfinite(det_oris[idx])
                if det_ori_valid:
                    sim = float(np.cos(2.0 * (float(det_oris[idx]) - qp.last_orientation)))
                    if sim < ori_similarity_threshold:
                        continue
                # Anisotropic position gate. Decompose the residual using the
                # axial average of query + detection orientation (more physical
                # than the stale query orientation alone).
                if det_ori_valid:
                    # Axial average via double-angle arithmetic.
                    s2 = (np.sin(2.0 * qp.last_orientation) +
                          np.sin(2.0 * float(det_oris[idx])))
                    c2 = (np.cos(2.0 * qp.last_orientation) +
                          np.cos(2.0 * float(det_oris[idx])))
                    ref_ori = 0.5 * np.arctan2(s2, c2)
                else:
                    ref_ori = qp.last_orientation
                s = np.sin(ref_ori)
                c = np.cos(ref_ori)
                cy = dets[idx, 0]; cx = dets[idx, 1]
                dy = cy - qy; dx = cx - qx
                along = dy * s + dx * c
                perp = -dy * c + dx * s
                if along * along > max_along2 or perp * perp > max_perp2:
                    continue
                all_pairs.append((float(dy * dy + dx * dx), i, int(idx)))

        all_pairs.sort(key=lambda p: p[0])
        claimed_qp: set = set()
        claimed_det: set = set()
        for _, i, j in all_pairs:
            if i in claimed_qp or j in claimed_det:
                continue
            qp = query_points[i]
            cy, cx = float(dets[j, 0]), float(dets[j, 1])
            qp.last_pos_px = (cy, cx)
            if j < len(det_oris) and np.isfinite(det_oris[j]):
                qp.last_orientation = float(det_oris[j])
            qp.positions_px.append((cy, cx))
            qp.positions_um.append((cy * pixelsize, cx * pixelsize))
            qp.sarcomere_lengths.append(float(det_slens[j]) if j < len(det_slens) else float('nan'))
            qp.orientations.append(float(det_oris[j]) if j < len(det_oris) else float('nan'))
            qp.snapped.append(True)
            qp.frames_since_snap = 0
            claimed_qp.add(i)
            claimed_det.add(j)

        # Unmatched live qps go into a gap frame.
        for i in live_idx:
            if i in claimed_qp:
                continue
            qp = query_points[i]
            qp.positions_px.append(qp.last_pos_px)
            qp.positions_um.append((qp.last_pos_px[0] * pixelsize, qp.last_pos_px[1] * pixelsize))
            qp.sarcomere_lengths.append(float('nan'))
            qp.orientations.append(float('nan'))
            qp.snapped.append(False)
            qp.frames_since_snap += 1
            if qp.frames_since_snap > memory:
                qp.alive = False
        claimed = claimed_det

        # 4. unclaimed detections → new query points (appearance)
        for k in range(len(dets)):
            if k in claimed:
                continue
            y = float(dets[k, 0]); x = float(dets[k, 1])
            o = float(det_oris[k]) if k < len(det_oris) and np.isfinite(det_oris[k]) else 0.0
            s = float(det_slens[k]) if k < len(det_slens) else float('nan')
            qp = QueryPoint(track_id=len(query_points), start_frame=t + 1)
            qp.positions_px.append((y, x))
            qp.positions_um.append((y * pixelsize, x * pixelsize))
            qp.sarcomere_lengths.append(s)
            qp.orientations.append(o)
            qp.snapped.append(True)
            qp.last_pos_px = (y, x)
            qp.last_orientation = o
            query_points.append(qp)

    # --- filter short tracks (count of actual snaps) ---
    logger.info("Filtering short tracks…")
    keep = [qp for qp in query_points if sum(qp.snapped) >= min_track_length]

    # --- build dense outputs (n_tracks, T) with NaN padding before start_frame ---
    n = len(keep)
    positions_um = np.full((n, T, 2), np.nan, dtype=np.float32)
    positions_px = np.full((n, T, 2), np.nan, dtype=np.float32)
    tracks_slen = np.full((n, T), np.nan, dtype=np.float32)
    tracks_ori = np.full((n, T), np.nan, dtype=np.float32)
    tracks_snapped = np.zeros((n, T), dtype=bool)
    track_lengths = np.zeros(n, dtype=np.int64)
    for row, qp in enumerate(keep):
        span = len(qp.positions_px)
        s = qp.start_frame
        for k in range(span):
            f = s + k
            if f >= T:
                break
            positions_px[row, f, 0] = qp.positions_px[k][0]
            positions_px[row, f, 1] = qp.positions_px[k][1]
            positions_um[row, f, 0] = qp.positions_um[k][0]
            positions_um[row, f, 1] = qp.positions_um[k][1]
            tracks_slen[row, f] = qp.sarcomere_lengths[k]
            tracks_ori[row, f] = qp.orientations[k]
            tracks_snapped[row, f] = qp.snapped[k]
        track_lengths[row] = int(sum(qp.snapped))

    result: Dict[str, object] = {
        'n_tracks': n,
        'track_ids': np.array([qp.track_id for qp in keep], dtype=np.int64),
        'track_start_frame': np.array([qp.start_frame for qp in keep], dtype=np.int64),
        'track_lengths': track_lengths,
        'tracks_positions_um': positions_um,
        'tracks_positions_px': positions_px,
        'tracks_slen': tracks_slen,
        'tracks_orientations': tracks_ori,
        'tracks_snapped': tracks_snapped,
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
