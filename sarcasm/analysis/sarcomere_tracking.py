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

"""2D full-field sarcomere tracking (per-frame optimal assignment).

Complements the 1D LOI-based tracker in :mod:`sarcasm.motion` by following every
sarcomere vector across the whole image automatically. The tracker works purely
on the per-frame sarcomere vectors (position, length, orientation) — it never
reads a pixel, and needs no optical flow: prediction comes from the coherent
motion of neighbouring tracks, which is both cheaper and far more accurate than a
dense flow field on segmentation masks.

The tracker does **not** track M-bands as a separate entity. Instead each
"query point" represents one sarcomere vector (position, length, orientation)
and is carried forward as follows:

1. **Prediction.** A query point that was observed last frame holds its fresh
   position; a point that did *not* is advected by the local coherent motion of
   the tracks around it (median step of the nearby tracks observed in both of
   the last two frames), projected onto its own sarcomere axis. Perpendicular
   motion can therefore only come from the match residual — how far the matched
   detection sits off the prediction — which is hard-capped, the
   anti-perpendicular-jump guarantee. The advection is load-bearing: without
   it a query point that misses several frames is left behind by a moving field
   and is far more likely to drift away from its neighbourhood.
2. **Gating.** Each query point's candidate detections are those inside the
   anisotropic along-/perpendicular-to-sarcomere ellipse with orientation
   compatible modulo π.
3. **Assignment.** Candidate (query point, detection) pairs form a bipartite
   graph. Its **connected components are solved exactly** — a minimum-cost
   maximum-cardinality assignment per component — with the **gate-normalised
   anisotropic cost** ``along²/along_budget + perp²/perp_budget``. This matters
   because sarcomere vectors are a *dense* sampling of the M-band midlines
   (~1 px apart along the midline, i.e. ~60 vectors per midline), so the
   perpendicular gate spans several lateral neighbours and the components are
   effectively the midline rows. Ranking such candidates by raw Euclidean
   distance would let a lateral neighbour 1 px away outrank the correct detection
   2 px along the axis; and a one-sided greedy claim orphans a track whenever a
   row shifts, which then spawns a duplicate. Solving each row jointly removes
   both effects. Because each detection is consumed by at most one query point,
   two query points can never collapse onto one detection.
4. **Gaps and identity.** A query point that finds no consistent detection
   records an honest gap frame: position = its prediction, ``observed=False``, and
   slen/orientation NaN unless a short interior gap is interpolated
   (``max_gap_interpolation_s``, which never sets ``observed``). It keeps its identity
   and re-enters the assignment on later frames, so a dropout of *any* length does
   not end a trajectory. Tracks therefore do not retire by default
   (``retire_after_s=None``), and no post-hoc fragment stitching is needed.

A query point's **trailing coast** — the frames after its *last observed* frame — is
blanked to NaN (position, slen, orientation) in the output: a lost track does not
freeze in place at its last position. Interior gaps are anchored on both sides,
keep their predicted position, and have their slen/orientation interpolated when
the gap is short (``max_gap_interpolation_s``); ``tracks_observed`` still marks which
frames are real observations, so no metric counts an interpolated frame.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Match gates are capped at these fractions of the median sarcomere length so a
# single-frame match can never reach a neighbouring sarcomere at any pixel size.
_ALONG_SLEN_FRAC = 0.6
_PERP_SLEN_FRAC = 0.25

# Components above this size use a greedy claim instead of the O(k³) exact solve.
_LAP_MAX_COMPONENT = 1500

# Frame-count fallbacks for the seconds-valued horizons when frametime is unknown.
_MIN_OBSERVATIONS_FALLBACK = 5
_MAX_GAP_FRAMES_FALLBACK = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median_slen_px(sarcomere_lengths_all, pixelsize: float) -> Optional[float]:
    """Median finite sarcomere length across all frames, converted to pixels.

    ``sarcomere_lengths_all`` is the per-frame list of detection slens in µm.
    Returns ``None`` if no finite slen exists or pixelsize is non-positive — the
    caller then keeps the raw pixel gates unchanged.
    """
    if pixelsize is None or pixelsize <= 0 or sarcomere_lengths_all is None:
        return None
    vals = [np.asarray(s, dtype=np.float64) for s in sarcomere_lengths_all
            if s is not None and len(s) > 0]
    if not vals:
        return None
    cat = np.concatenate(vals)
    cat = cat[np.isfinite(cat)]
    if cat.size == 0:
        return None
    return float(np.median(cat)) / float(pixelsize)

def _neighbor_displacement(query_yx: np.ndarray, ref_yx: np.ndarray,
                           ref_disp: np.ndarray, radius: float,
                           k: int = 16) -> np.ndarray:
    """Local coherent displacement at each query point, from nearby tracks.

    The median frame-to-frame displacement of the ``k`` nearest reference tracks
    within ``radius``: how the tissue around a query point just moved. Used to
    carry a coasting track forward with its neighbourhood, so its anchor cannot go
    stale and let the neighbouring sarcomere into the re-acquisition gate. Median
    rather than mean, so a few mismatched neighbours cannot drag the estimate.

    Parameters
    ----------
    query_yx : np.ndarray
        ``(Q, 2)`` positions to estimate the displacement at (px).
    ref_yx : np.ndarray
        ``(R, 2)`` positions of the reference tracks at the current frame (px).
    ref_disp : np.ndarray
        ``(R, 2)`` those tracks' displacement over the previous frame step (px).
    radius : float
        Neighbourhood radius in px; reference tracks further away are ignored.
    k : int, optional
        Number of nearest references considered per query point. Default is 16.

    Returns
    -------
    np.ndarray
        ``(Q, 2)`` estimated displacement, zero where nothing is in range.
    """
    q = np.asarray(query_yx, dtype=np.float64).reshape(-1, 2)
    out = np.zeros((len(q), 2), dtype=np.float32)
    ref_yx = np.asarray(ref_yx, dtype=np.float64).reshape(-1, 2)
    if len(q) == 0 or len(ref_yx) == 0:
        return out
    kk = int(min(k, len(ref_yx)))
    dist, idx = cKDTree(ref_yx).query(q, k=kk)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    within = np.isfinite(dist) & (dist <= radius)
    d = np.where(within[:, :, None], np.asarray(ref_disp, dtype=np.float64)[idx], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(d, axis=1)
    # Points with no reference inside the radius fall back to the global median
    # (a rigid drift of the whole field is still better than assuming no motion).
    isolated = ~np.isfinite(med).all(axis=1)
    if isolated.any():
        med[isolated] = np.median(ref_disp, axis=0)
    return np.nan_to_num(med).astype(np.float32)


def compute_track_drift(positions_px: np.ndarray, observed: np.ndarray,
                        median_slen_px: Optional[float],
                        pixelsize: float, n_segments: int = 8) -> np.ndarray:
    """Per-track drift away from the coherent motion of its neighbours, in µm.

    A sarcomere moves with the tissue around it, so a track departing its
    neighbourhood by ~one sarcomere length has almost certainly changed identity.
    The reference is the *local* median displacement, so genuine (non-rigid)
    contraction is not flagged. Accumulated over ``n_segments`` time windows,
    which keeps the reference local in time and lets partial tracks be scored.

    Parameters
    ----------
    positions_px : np.ndarray
        ``(N, T, 2)`` track positions in px.
    observed : np.ndarray
        ``(N, T)`` bool, True where the track was matched to a real detection.
    median_slen_px : float or None
        Median sarcomere length in px; sets the neighbourhood radius (3 slen).
    pixelsize : float
        µm per px, to return the drift in µm.
    n_segments : int, optional
        Number of time windows. Default is 8.

    Returns
    -------
    np.ndarray
        ``(N,)`` accumulated drift in µm; NaN for tracks that never span a full
        window (too short to score).
    """
    positions_px = np.asarray(positions_px, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    N, T = observed.shape
    resid = np.zeros((N, 2))
    scored = np.zeros(N, dtype=bool)
    if N == 0 or T < 2:
        return np.full(N, np.nan)
    radius = 3.0 * median_slen_px if (median_slen_px and median_slen_px > 0) else np.inf
    edges = np.linspace(0, T - 1, int(n_segments) + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        both = observed[:, a] & observed[:, b]
        idx = np.flatnonzero(both)
        if idx.size < 6:          # too few references to define a local motion
            continue
        p0 = positions_px[idx, a]
        disp = positions_px[idx, b] - p0
        good = np.isfinite(p0).all(1) & np.isfinite(disp).all(1)
        idx, p0, disp = idx[good], p0[good], disp[good]
        if idx.size < 6:
            continue
        tree = cKDTree(p0)
        kk = int(min(17, idx.size))       # self + up to 16 neighbours
        dist, nb = tree.query(p0, k=kk)
        within = np.isfinite(dist) & (dist <= radius)
        within[:, 0] = False              # drop self from its own reference set
        d = np.where(within[:, :, None], disp[nb], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            local = np.nanmedian(d, axis=1)
        ok = np.isfinite(local).all(axis=1)
        resid[idx[ok]] += disp[ok] - local[ok]
        scored[idx[ok]] = True
    out = np.linalg.norm(resid, axis=1) * float(pixelsize)
    out[~scored] = np.nan
    return out


def _median_detections_per_frame(pos_vectors_px_all) -> float:
    """Median number of sarcomere vectors per frame.

    This is the denominator of the fragmentation ratio: with one track per vector
    the track count equals it. Note it is *not* a count of physical sarcomeres —
    the vectors are a dense (~1 px) sampling along each M-band midline, so one
    midline contributes tens of them.
    """
    counts = [0 if p is None else len(p) for p in pos_vectors_px_all]
    return float(np.median(counts)) if counts else 0.0

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
# Query-point tracker
# ---------------------------------------------------------------------------

def _assign_optimal(
    cost: np.ndarray,
    qp: np.ndarray,
    det: np.ndarray,
    n_qp: int,
    n_det: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost maximum-cardinality assignment, solved per graph component.

    ``qp``/``det`` are the gated candidate pairs and ``cost`` their
    gate-normalised anisotropic costs (each ≤ 2 by construction, since both
    squared residuals are ≤ their budget). The bipartite graph over
    ``[query points | detections]`` is split into connected components — for
    densely sampled M-band rows these components *are* the rows — and each is
    solved exactly with :func:`scipy.optimize.linear_sum_assignment`.

    Unmatchable slots are padded with a sentinel of ``2·min(k_qp,k_det)+1``, which
    exceeds the total cost of any complete set of real pairs; the optimum is
    therefore maximum-cardinality among the minimum-cost solutions, i.e. it never
    leaves a pair unmatched merely to lower the total cost.

    Parameters
    ----------
    cost : np.ndarray
        ``(P,)`` cost of each candidate pair.
    qp : np.ndarray
        ``(P,)`` query-point indices, in ``[0, n_qp)``.
    det : np.ndarray
        ``(P,)`` detection indices, in ``[0, n_det)``.
    n_qp, n_det : int
        Number of query points / detections in this frame.

    Returns
    -------
    tuple of np.ndarray
        Accepted ``(qp_indices, det_indices)``.
    """
    if len(qp) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    n = n_qp + n_det
    adj = coo_matrix((np.ones(len(qp)), (qp, det + n_qp)), shape=(n, n))
    _, labels = connected_components(adj + adj.T, directed=False)
    order = np.argsort(labels[qp], kind='stable')
    qp, det, cost = qp[order], det[order], cost[order]
    bounds = np.flatnonzero(np.diff(labels[qp])) + 1
    out_q: List[np.ndarray] = []
    out_d: List[np.ndarray] = []
    for sl in np.split(np.arange(len(qp)), bounds):
        q_ids, q_inv = np.unique(qp[sl], return_inverse=True)
        d_ids, d_inv = np.unique(det[sl], return_inverse=True)
        k_q, k_d = len(q_ids), len(d_ids)
        if max(k_q, k_d) > _LAP_MAX_COMPONENT:
            # Safety valve: an unusually large component would make the dense
            # O(k³) solve dominate. Fall back to a greedy claim by cost.
            claimed_q = np.zeros(k_q, bool)
            claimed_d = np.zeros(k_d, bool)
            for i in np.argsort(cost[sl], kind='stable'):
                a, b = q_inv[i], d_inv[i]
                if claimed_q[a] or claimed_d[b]:
                    continue
                claimed_q[a] = claimed_d[b] = True
                out_q.append(q_ids[a:a + 1])
                out_d.append(d_ids[b:b + 1])
            logger.debug(
                f"Assignment component of size {max(k_q, k_d)} exceeded "
                f"_LAP_MAX_COMPONENT={_LAP_MAX_COMPONENT}; used greedy fallback.")
            continue
        sentinel = 2.0 * min(k_q, k_d) + 1.0
        mat = np.full((k_q, k_d), sentinel)
        np.minimum.at(mat, (q_inv, d_inv), cost[sl])
        rows, cols = linear_sum_assignment(mat)
        ok = mat[rows, cols] < sentinel
        out_q.append(q_ids[rows[ok]])
        out_d.append(d_ids[cols[ok]])
    if not out_q:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return (np.concatenate(out_q).astype(np.int64),
            np.concatenate(out_d).astype(np.int64))


def _interpolate_short_gaps(
    n_tracks: int,
    tracks_slen: np.ndarray,
    tracks_ori: np.ndarray,
    tracks_observed: np.ndarray,
    max_gap: int,
) -> int:
    """Fill sarcomere length / orientation across short INTERIOR gaps, in place.

    A gap frame carries a predicted position but no detection, so its slen and
    orientation are NaN. Those holes break downstream traces (contraction
    detection, per-group time series) even when the gap is a single flickered
    frame, so runs of at most ``max_gap`` consecutive gap frames that are
    anchored by a real observation on *both* sides are filled by interpolation.

    Only interior runs are touched: leading/trailing frames have no second anchor,
    and extrapolating there would invent data at the ends of a trajectory.
    ``tracks_observed`` is deliberately left False on filled frames, so every
    real-observation metric, coverage figure and QC guard still counts only
    genuine detections — the fill is a convenience for continuous traces, never
    evidence.

    Orientation is interpolated in the double-angle representation
    (``sin 2θ``/``cos 2θ``), since sarcomere orientations are undirected axes and
    a plain linear interpolation would break across the ±π/2 wrap.

    Parameters
    ----------
    n_tracks : int
        Number of active tracks in the SoA arrays.
    tracks_slen, tracks_ori, tracks_observed : np.ndarray
        ``(capacity, T)`` track state; slen/ori are modified in place.
    max_gap : int
        Longest run of gap frames to fill. ``<= 0`` disables the pass.

    Returns
    -------
    int
        Number of (track, frame) entries whose sarcomere length was filled.
    """
    if n_tracks == 0 or int(max_gap) <= 0:
        return 0
    filled = 0
    for i in range(n_tracks):
        fr = np.flatnonzero(tracks_observed[i])
        if fr.size < 2:
            continue
        gaps = np.diff(fr)
        for a, b in zip(fr[:-1][gaps > 1], fr[1:][gaps > 1]):
            span = int(b - a)
            if span - 1 > int(max_gap):
                continue
            w = np.arange(1, span) / float(span)
            # An observed anchor can still carry a NaN length (its detection had
            # none), in which case there is nothing to interpolate between.
            s_a, s_b = tracks_slen[i, a], tracks_slen[i, b]
            if np.isfinite(s_a) and np.isfinite(s_b):
                tracks_slen[i, a + 1:b] = (1.0 - w) * s_a + w * s_b
                filled += span - 1
            o_a, o_b = tracks_ori[i, a], tracks_ori[i, b]
            if np.isfinite(o_a) and np.isfinite(o_b):
                s2 = (1.0 - w) * np.sin(2.0 * o_a) + w * np.sin(2.0 * o_b)
                c2 = (1.0 - w) * np.cos(2.0 * o_a) + w * np.cos(2.0 * o_b)
                tracks_ori[i, a + 1:b] = 0.5 * np.arctan2(s2, c2)
    return filled


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
    pos_vectors_px_all: List[np.ndarray],
    midline_ids_all: List[np.ndarray],   # per-frame midline id of each detection; -1 if absent
    sarcomere_lengths_all: List[np.ndarray],
    orientations_all: List[np.ndarray],
    pixelsize: float,
    frametime: Optional[float] = None,
    max_disp_along_um: float = 1.0,
    max_disp_perp_um: float = 0.2,
    ori_tol_deg: float = 45.0,
    retire_after_s: Optional[float] = None,
    min_track_duration_s: float = 0.08,
    max_gap_interpolation_s: float = 0.05,
    progress_notifier=None,
) -> Dict[str, object]:
    """Run the 2D full-field sarcomere-vector tracker on a detection sequence.

    The tracker does not persist M-band identity, and reads no image data: every
    query point is a sarcomere-vector marker that is predicted from the coherent
    motion of its neighbours and matched to a consistent detection each frame.
    Outputs are dense ``(n_tracks, T)`` arrays. Match gates are capped relative to
    the sarcomere length so the tracker stays scale-invariant across pixel size /
    frame time.

    Continuity rests on two properties:

    1. **Exact assignment per graph component** with a gate-normalised
       anisotropic cost (:func:`_assign_optimal`). Sarcomere vectors are a dense
       ~1 px sampling of each M-band midline, so the perpendicular gate spans
       several lateral neighbours and the components are effectively the midline
       rows; solving each row jointly is what stops a shifting row from orphaning
       tracks and spawning duplicates. Each detection is consumed by at most one
       query point — the anti-convergence guarantee.
    2. **Identity that survives a gap of any length.** An unmatched track records
       an honest gap frame (``observed=False``, slen/orientation NaN) and re-enters
       the assignment later, so detection dropout does not end a trajectory and
       no post-hoc fragment stitching is required.

    Each (track, frame) also records ``tracks_detection_id`` (index of the
    matched detection into ``pos_vectors_px_all[frame]``) and
    ``tracks_midline_id`` (that detection's entry in ``midline_ids_all``); both
    are ``-1`` on gap/interpolated frames, giving an exact join back to the
    per-frame vector / domain / myofibril analyses.

    Parameters
    ----------
    pos_vectors_px_all : list of np.ndarray
        Per-frame ``(N_t, 2)`` sarcomere-centre detections in px (row, col).
    midline_ids_all : list of np.ndarray
        Per-frame midline (M-band) id of each detection; -1 if absent.
    sarcomere_lengths_all : list of np.ndarray
        Per-frame detection sarcomere lengths in µm.
    orientations_all : list of np.ndarray
        Per-frame detection orientations in radians.
    pixelsize : float
        Pixel size in µm.
    frametime : float or None, optional
        Frame time in s; converts the seconds-valued horizons below to frames.
        Default is None.
    max_disp_along_um : float, optional
        Match gate along the sarcomere axis, in µm — i.e. the maximum distance a
        track may move along its axis between consecutive frames. This is the
        per-frame "max step": at the default 1.0 µm a track can never jump more
        than ~1 µm regardless of pixel size. Default is 1.0.
    max_disp_perp_um : float, optional
        Match gate perpendicular to the sarcomere axis, in µm; kept far tighter
        than the along gate (perpendicular jumps onto a neighbouring myofibril
        are the dangerous swap). Default is 0.2.
    ori_tol_deg : float, optional
        Orientation tolerance for matching, in degrees. Default is 45.0.
    retire_after_s : float or None, optional
        Time a track may go unmatched before it is closed, in seconds. ``None``
        (default) means tracks never retire: because an unmatched track is carried
        by the coherent motion of its neighbourhood, its identity stays valid
        through a dropout of any length, and retiring it merely fragments the
        trajectory. Set a value (e.g. 5.0) for very long recordings where
        sarcomeres genuinely appear and disappear, to bound the track count.
    min_track_duration_s : float, optional
        Minimum accumulated real observation time to keep a track, in seconds
        (converted via ``frametime``; falls back to
        ``_MIN_OBSERVATIONS_FALLBACK`` real observations when ``frametime`` is
        None). Default is 0.08.
    max_gap_interpolation_s : float, optional
        Longest gap, in seconds, whose sarcomere length / orientation is filled by
        interpolation between the real observations on either side, so short
        detection flicker does not punch holes in the per-track traces (converted
        via ``frametime``; ``_MAX_GAP_FRAMES_FALLBACK`` frames when ``frametime`` is
        None). Interior gaps only, and ``tracks_observed`` stays False there, so no
        real-observation metric is affected. Set to 0 to keep every gap frame NaN.
        Kept deliberately short (default 0.05 s): a longer fill would start to span
        a contraction and invent length dynamics.

    Returns
    -------
    dict
        Result dictionary with keys: ``'motion.tracks.n'``, ``'motion.tracks.ids'``,
        ``'motion.tracks.start_frame'``, ``'motion.tracks.n_frames'``, ``'motion.tracks.positions_um'``,
        ``'motion.tracks.positions_px'``, ``'motion.tracks.slen'`` (µm),
        ``'motion.tracks.orientations'`` (rad), ``'motion.tracks.observed'`` (bool),
        ``'motion.tracks.detection_id'``, ``'motion.tracks.midline_id'``,
        ``'motion.tracks.fragmentation_ratio'`` (tracks per median detections-per-frame; ideal
        1.0 — the headline continuity QC number) and ``'motion.tracks.n_retired'``.
        Track arrays are dense ``(n_tracks, T)`` (positions ``(n_tracks, T, 2)``).
        ``'motion.tracks.n_interpolated_gap_frames'`` counts the entries filled by
        ``max_gap_interpolation_s``.
    """
    T = len(pos_vectors_px_all)
    if T < 2:
        raise ValueError("Need at least 2 frames.")

    # Seconds-valued horizons -> frame counts, so the same physical duration is
    # used at any frame rate (frametime scale-invariance).
    def _to_frames(seconds, fallback_frames):
        if frametime and frametime > 0:
            return max(1, int(round(float(seconds) / float(frametime))))
        return int(fallback_frames)
    min_observations = _to_frames(min_track_duration_s, _MIN_OBSERVATIONS_FALLBACK)
    retire_frames = (np.inf if retire_after_s is None
                     else _to_frames(retire_after_s, 10 ** 9))
    max_gap_frames = (0 if not max_gap_interpolation_s or max_gap_interpolation_s <= 0
                      else _to_frames(max_gap_interpolation_s, _MAX_GAP_FRAMES_FALLBACK))
    logger.info(
        f"Tracking: min_track_duration={min_track_duration_s} s ({min_observations} observations), "
        f"retire_after={retire_after_s} s (frametime={frametime}).")

    if not pixelsize or pixelsize <= 0:
        raise ValueError(
            f"pixelsize must be > 0 for tracking (gates are specified in µm), "
            f"got {pixelsize!r}.")
    ori_tol_rad = float(np.deg2rad(ori_tol_deg))
    ori_similarity_threshold = float(np.cos(2.0 * ori_tol_rad))

    # Pixel-size invariance: all public gates are specified in micrometres so the
    # same defaults track correctly at any pixel size out of the box. Convert the
    # physical gates to pixels here, once, with the calibration.
    px = float(pixelsize)
    max_disp_along_px = float(max_disp_along_um) / px
    max_disp_perp_px = float(max_disp_perp_um) / px

    # cap the gates relative to the measured sarcomere length (no-op at the defaults)
    median_slen_px = _median_slen_px(sarcomere_lengths_all, pixelsize)
    if median_slen_px is not None and median_slen_px > 0:
        along_cap = _ALONG_SLEN_FRAC * median_slen_px
        perp_cap = _PERP_SLEN_FRAC * median_slen_px
        if along_cap < max_disp_along_px or perp_cap < max_disp_perp_px:
            logger.info(
                f"Scale-aware gate cap (median slen ≈ {median_slen_px:.1f} px): "
                f"along {max_disp_along_px:.1f}→{min(max_disp_along_px, along_cap):.1f} px, "
                f"perp {max_disp_perp_px:.1f}→{min(max_disp_perp_px, perp_cap):.1f} px."
            )
        max_disp_along_px = min(float(max_disp_along_px), along_cap)
        max_disp_perp_px = min(float(max_disp_perp_px), perp_cap)

    max_along2 = float(max_disp_along_px * max_disp_along_px)
    max_perp2 = float(max_disp_perp_px * max_disp_perp_px)
    max_radius = float(max_disp_along_px)

    # Neighbourhood radius for the local-motion estimate: a few sarcomeres, so the
    # estimate is local to one fibre bundle yet has enough members to be robust.
    neighbor_radius_px = 3.0 * median_slen_px if (
        median_slen_px is not None and median_slen_px > 0) else 3.0 * max_disp_along_px

    # struct-of-arrays track state: history arrays grow in chunks; live state
    # (last_* / frames_since_observation / alive) is indexed by the same slot
    pos0_raw = pos_vectors_px_all[0]
    n0 = 0 if pos0_raw is None else len(pos0_raw)
    capacity = max(256, n0 * 4)

    positions_px = np.full((capacity, T, 2), np.nan, dtype=np.float32)
    tracks_slen = np.full((capacity, T), np.nan, dtype=np.float32)
    tracks_ori = np.full((capacity, T), np.nan, dtype=np.float32)
    tracks_observed = np.zeros((capacity, T), dtype=bool)
    # Per-(track, frame) provenance: index of the matched detection into
    # pos_vectors_px_all[frame], and that detection's midline id. -1 = no match.
    tracks_detection_id = np.full((capacity, T), -1, dtype=np.int32)
    tracks_midline_id = np.full((capacity, T), -1, dtype=np.int32)
    start_frame_arr = np.zeros(capacity, dtype=np.int32)
    last_y = np.full(capacity, np.nan, dtype=np.float32)
    last_x = np.full(capacity, np.nan, dtype=np.float32)
    last_ori = np.zeros(capacity, dtype=np.float32)
    frames_since_observation = np.zeros(capacity, dtype=np.int32)
    alive = np.zeros(capacity, dtype=bool)

    def _grow(needed: int):
        """Double capacity until >= ``needed`` and resize all SoA arrays."""
        nonlocal capacity, positions_px, tracks_slen, tracks_ori, tracks_observed
        nonlocal tracks_detection_id, tracks_midline_id
        nonlocal start_frame_arr, last_y, last_x, last_ori, frames_since_observation, alive
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
        tracks_observed = _resize(tracks_observed, False)
        tracks_detection_id = _resize(tracks_detection_id, -1)
        tracks_midline_id = _resize(tracks_midline_id, -1)
        start_frame_arr = _resize(start_frame_arr, 0)
        last_y = _resize(last_y, np.nan)
        last_x = _resize(last_x, np.nan)
        last_ori = _resize(last_ori, 0.0)
        frames_since_observation = _resize(frames_since_observation, 0)
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
        tracks_observed[sl, 0] = True
        # Frame-0 seeds map 1:1 onto frame-0 detections (slot i == detection i).
        tracks_detection_id[sl, 0] = np.arange(n0, dtype=np.int32)
        tracks_midline_id[sl, 0] = _as_arr_padded(
            midline_ids_all[0] if (midline_ids_all is not None and len(midline_ids_all) > 0)
            else None, n0, np.int32, -1,
        )
        start_frame_arr[sl] = 0
        last_y[sl] = pos0[:, 0]
        last_x[sl] = pos0[:, 1]
        last_ori[sl] = ori0
        alive[sl] = True
        n_tracks = n0

    # --- frame-to-frame: advect, match, spawn ---
    logger.info("Tracking frames…")
    _frames = range(T - 1)
    if progress_notifier is not None:
        _frames = progress_notifier.iterator(_frames)
    for t in _frames:
        # 1. advect unmatched tracks along their sarcomere axis by the median step
        #    of the neighbours observed in both of the last two frames
        live = np.flatnonzero(alive[:n_tracks])
        if live.size > 0:
            ys = last_y[live]
            xs = last_x[live]
            coasting = np.flatnonzero(frames_since_observation[live] > 0)
            ref = (np.flatnonzero(tracks_observed[:n_tracks, t]
                                  & tracks_observed[:n_tracks, t - 1])
                   if t >= 1 else np.empty(0, dtype=np.int64))
            if coasting.size and ref.size:
                disp = _neighbor_displacement(
                    np.column_stack((ys[coasting], xs[coasting])),
                    positions_px[ref, t],
                    positions_px[ref, t] - positions_px[ref, t - 1],
                    radius=neighbor_radius_px)
                ci = live[coasting]
                s_live = np.sin(last_ori[ci])
                c_live = np.cos(last_ori[ci])
                along_live = disp[:, 0] * s_live + disp[:, 1] * c_live
                last_y[ci] = ys[coasting] + along_live * s_live
                last_x[ci] = xs[coasting] + along_live * c_live

        # 2. prepare detections for frame t+1
        dets_raw = pos_vectors_px_all[t + 1]
        dets = np.asarray(
            dets_raw if dets_raw is not None else np.zeros((0, 2)),
            dtype=np.float32,
        )
        n_det = len(dets)
        det_oris = _as_arr_padded(orientations_all[t + 1], n_det, np.float32, np.nan)
        det_slens = _as_arr_padded(sarcomere_lengths_all[t + 1], n_det, np.float32, np.nan)
        det_mids = _as_arr_padded(
            midline_ids_all[t + 1] if (midline_ids_all is not None and (t + 1) < len(midline_ids_all))
            else None, n_det, np.int32, -1,
        )

        # Bool masks over the current track slots / detections so we can do
        # claim bookkeeping in pure numpy without Python sets.
        claimed_qp_mask = np.zeros(n_tracks, dtype=bool)
        claimed_det_mask = np.zeros(n_det, dtype=bool)

        # 3. kd-tree + vectorized gate on all (qp, candidate) pairs; the gate does
        #    not widen with gap length (that would trade identity for fragmentation)
        if n_det > 0 and live.size > 0:
            tree = cKDTree(dets)
            live_pos = np.column_stack((last_y[live], last_x[live]))
            # kd-tree radius must cover the along gate, the larger axis.
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
                along_budget = max_along2
                perp_budget = max_perp2

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
                pass_pos = (along_p * along_p <= along_budget) & (perp_p * perp_p <= perp_budget)
                mask = pass_ori & pass_pos

                keep_pairs = np.flatnonzero(mask)
                if keep_pairs.size > 0:
                    # cost = fraction of each (anisotropic) gate budget used
                    cost = (along_p[keep_pairs] ** 2 / along_budget
                            + perp_p[keep_pairs] ** 2 / perp_budget)
                    qp_rel_kept = qp_flat_rel[keep_pairs].astype(np.int64)
                    det_kept = det_flat[keep_pairs].astype(np.int64)

                    qp_rel_acc, det_acc = _assign_optimal(
                        cost, qp_rel_kept, det_kept, live.size, n_det)
                    qp_acc = live[qp_rel_acc]
                    claimed_qp_mask[qp_acc] = True
                    claimed_det_mask[det_acc] = True

                    # 4. write claimed tracks in a single vectorized shot
                    cy_acc = dets[det_acc, 0]
                    cx_acc = dets[det_acc, 1]
                    co_acc = det_oris[det_acc]
                    sl_acc = det_slens[det_acc]

                    positions_px[qp_acc, t + 1, 0] = cy_acc
                    positions_px[qp_acc, t + 1, 1] = cx_acc
                    tracks_slen[qp_acc, t + 1] = sl_acc
                    tracks_ori[qp_acc, t + 1] = co_acc
                    tracks_observed[qp_acc, t + 1] = True
                    tracks_detection_id[qp_acc, t + 1] = det_acc.astype(np.int32)
                    tracks_midline_id[qp_acc, t + 1] = det_mids[det_acc]
                    last_y[qp_acc] = cy_acc
                    last_x[qp_acc] = cx_acc
                    # Only overwrite last_orientation when detection ori is finite.
                    finite_det = np.isfinite(co_acc)
                    if finite_det.any():
                        last_ori[qp_acc[finite_det]] = co_acc[finite_det]
                    frames_since_observation[qp_acc] = 0

        # 5. unmatched live tracks → gap frame
        if live.size > 0:
            unclaimed_live_mask = ~claimed_qp_mask[live]
            unclaimed_live = live[unclaimed_live_mask]
            if unclaimed_live.size > 0:
                positions_px[unclaimed_live, t + 1, 0] = last_y[unclaimed_live]
                positions_px[unclaimed_live, t + 1, 1] = last_x[unclaimed_live]
                # tracks_slen / tracks_ori already NaN; tracks_observed already False.
                frames_since_observation[unclaimed_live] += 1
                if np.isfinite(retire_frames):
                    died_local = frames_since_observation[unclaimed_live] > retire_frames
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
                tracks_observed[new_slots, t + 1] = True
                tracks_detection_id[new_slots, t + 1] = unclaimed_det.astype(np.int32)
                tracks_midline_id[new_slots, t + 1] = det_mids[unclaimed_det]
                start_frame_arr[new_slots] = t + 1
                last_y[new_slots] = new_y
                last_x[new_slots] = new_x
                last_ori[new_slots] = new_ori
                frames_since_observation[new_slots] = 0
                alive[new_slots] = True
                n_tracks += n_new

    # blank the trailing coast after a track's last observation (interior gaps stay)
    if n_tracks > 0:
        observed_view = tracks_observed[:n_tracks]
        has_observation = observed_view.any(axis=1)
        # Index of the last True per row (all-False rows → -1 → whole row blanked;
        # those have 0 observations and are dropped by the short-track filter anyway).
        last_observed_idx = np.where(
            has_observation,
            (T - 1) - np.argmax(observed_view[:, ::-1], axis=1),
            -1,
        )
        trailing = np.arange(T)[None, :] > last_observed_idx[:, None]
        positions_px[:n_tracks][trailing] = np.nan
        tracks_slen[:n_tracks][trailing] = np.nan
        tracks_ori[:n_tracks][trailing] = np.nan

    # --- interpolate slen / orientation across SHORT interior gaps ---
    _interpolate_short_gaps(
        n_tracks, tracks_slen, tracks_ori, tracks_observed, max_gap_frames)

    # --- filter short tracks (count of actual observations) ---
    logger.info("Filtering short tracks…")
    observed_counts = tracks_observed[:n_tracks].sum(axis=1)
    keep_mask = observed_counts >= int(min_observations)
    kept = np.flatnonzero(keep_mask)
    n = int(kept.size)

    out_positions_px = positions_px[kept].copy()
    out_positions_um = (out_positions_px * pixelsize).astype(np.float32)
    out_tracks_slen = tracks_slen[kept].copy()
    out_tracks_ori = tracks_ori[kept].copy()
    out_tracks_observed = tracks_observed[kept].copy()
    out_tracks_detection_id = tracks_detection_id[kept].copy()
    out_tracks_midline_id = tracks_midline_id[kept].copy()
    out_start_frame = start_frame_arr[kept].astype(np.int64)
    out_track_ids = kept.astype(np.int64)
    out_track_lengths = observed_counts[kept].astype(np.int64)

    # Counted on the KEPT tracks, so it matches the returned arrays exactly:
    # a finite slen on a gap frame can only come from the interpolation.
    n_interpolated = int(np.isfinite(out_tracks_slen[~out_tracks_observed]).sum())
    if n_interpolated:
        logger.info(
            f"Interpolated sarcomere length / orientation on {n_interpolated} gap "
            f"frames (gaps of at most {max_gap_frames} frames = {max_gap_interpolation_s} s); "
            f"'motion.tracks.observed' stays False there, so no coverage metric counts them.")

    # Headline continuity QC: tracks per median detections-per-frame. 1.0 means one
    # track per sarcomere vector over the whole recording; larger means the same
    # vector was split into that many trajectories.
    med_det = _median_detections_per_frame(pos_vectors_px_all)
    frag_ratio = float(n) / med_det if med_det else float('nan')
    logger.info(
        f"{n} tracks for ~{med_det:.0f} detections/frame "
        f"(fragmentation ratio {frag_ratio:.2f}, ideal 1.0).")

    track_drift_um = compute_track_drift(
        out_positions_px, out_tracks_observed, median_slen_px, pixelsize)
    n_drifted = int(np.count_nonzero(
        np.isfinite(track_drift_um) & (track_drift_um > (median_slen_px or 0) * pixelsize)))
    if n_drifted:
        logger.info(
            f"{n_drifted}/{n} tracks drift more than one sarcomere length away from "
            f"their neighbours (see 'motion.tracks.drift_um'); chain groupings "
            f"(myofibril/loi) drop them by default.")

    result: Dict[str, object] = {
        'motion.tracks.n': n,
        'motion.tracks.ids': out_track_ids,
        'motion.tracks.start_frame': out_start_frame,
        'motion.tracks.n_frames': out_track_lengths,
        'motion.tracks.drift_um': track_drift_um,
        'motion.tracks.positions_um': out_positions_um,
        'motion.tracks.positions_px': out_positions_px,
        'motion.tracks.slen': out_tracks_slen,
        'motion.tracks.orientations': out_tracks_ori,
        'motion.tracks.observed': out_tracks_observed,
        'motion.tracks.detection_id': out_tracks_detection_id,
        'motion.tracks.midline_id': out_tracks_midline_id,
        'motion.tracks.fragmentation_ratio': frag_ratio,
        'motion.tracks.n_retired': int(np.count_nonzero(~alive[:n_tracks])),
        'motion.tracks.n_interpolated_gap_frames': int(n_interpolated),
    }

    return result
