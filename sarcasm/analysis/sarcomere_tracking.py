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
on the per-frame sarcomere vectors (position, length, orientation) — it reads no
pixel itself. By default prediction comes from the coherent motion of neighbouring
tracks, which at ordinary frame rates is cheaper and more accurate than a dense
flow field. When the frame rate is coarse relative to the motion — one contraction
moves a sarcomere a sizeable fraction of its length between consecutive frames —
the hold-position prediction falls outside the gate; an optional image-flow
``predictor`` (:mod:`sarcasm.analysis.optical_flow`) then carries every query
point along with the tissue before gating.

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
   With a ``predictor`` (image optical flow), *every* live query point is moved
   by the predicted displacement instead: its along-axis component always, its
   perpendicular component only where it exceeds ``_PREDICT_PERP_KEEP_FRAC`` of
   the perpendicular gate — the flow's lateral component is dominated by the
   aperture problem on striations: at high frame rates its noise exceeds the true
   lateral motion by more than an order of magnitude and would shuffle identities
   along the dense midline rows. With this projection the predictor reproduces
   the default at high frame rates and removes the identity swaps and
   fragmentation of the hold-position prediction at coarse ones.
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
keep their predicted position (unless :func:`fill_track_gaps` measured the frame), and have their slen/orientation interpolated when
the gap is short (``max_gap_interpolation_s``); ``motion.tracks.observed`` still marks which
frames are real observations, so no metric counts an interpolated frame.

5. **Rest position.** Between beats the tissue mostly returns to where it was, but during
   a fast beat the stripes can shift by about a sarcomere within a few frames, and a
   track then lands on a neighbouring sarcomere. So on rest frames
   (:func:`rest_frames_from_trace`) a track is predicted at the position it holds at
   rest, a running mean of its earlier rest positions, and its along-axis gate narrows
   to at most half a sarcomere (``max_disp_along_rest_um``). So a track returns to its
   sarcomere at the next rest. If the tissue did not return, far fewer detections lie at the
   rest positions than where the tracks are now: then the tracks keep following, and
   their rest positions catch up.

:func:`fill_track_gaps` works on the finished tracks: it measures the gap frames of well-covered
tracks at the track's own position, in a separate array.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Match gates are capped at these fractions of the median sarcomere length so a
# match can never reach a neighbouring sarcomere from the track's prediction, at any pixel size.
_ALONG_SLEN_FRAC = 0.6
_PERP_SLEN_FRAC = 0.25
# On rest frames the along-axis gate stays within half a sarcomere, so a track cannot reach the next M-band.
_ALONG_REST_SLEN_FRAC = 0.5
# Time constant of a track's rest position: slow enough to ignore pixel jitter, fast enough to follow slow drift.
_REST_POSITION_TAU_S = 0.2
# A position is occupied if a detection lies within this distance of it, in µm (at least 1 px).
_REST_DETECTION_RADIUS_UM = 0.4
# Rest frames: the cell trace stays within this fraction of its largest excursion from the rest level ...
_REST_BAND_FRAC = 0.15
# ... and rest is only told from beats if the beat amplitude exceeds this many times the frame-to-frame noise.
_REST_MIN_SNR = 15.0

# A filled frame is kept only for a clean Z-band pair (see Utils.grade_profile_pairs), and at rest only
# within this distance of the length interpolated from the track's real frames.
_FILL_MAX_ASYMMETRY = 0.3
_FILL_MAX_THIRD_PEAK = 0.5
_FILL_REST_TOL_UM = 0.2

# Components above this size use a greedy claim instead of the O(k³) exact solve.
_LAP_MAX_COMPONENT = 1500

# Frame-count fallbacks for the seconds-valued horizons when frametime is unknown.
_MIN_OBSERVATIONS_FALLBACK = 5
_MAX_GAP_FRAMES_FALLBACK = 3
# image-flow predictor: keep the (neighbourhood-median) perpendicular component of the
# predicted step only where it exceeds this fraction of the perpendicular gate, i.e. only
# where the hold-position prediction could not reach the detection at all. Below the gate
# the residual is absorbed by the gate itself, and the flow's lateral component is mostly
# aperture noise there (at high frame rates its error is >10x the true lateral motion).
# Ablation on a high-frame-rate reference movie, fraction of the default tracker's links
# reproduced: 0.5 -> 94 %, 0.75 -> 98 %, 1.0 -> 99 %.
_PREDICT_PERP_KEEP_FRAC = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _with_progress(frames: range, progress_notifier, label: str):
    """``frames`` wrapped in the notifier's progress iterator (as is without one), labelled where the iterator
    takes a label: tracking reports two passes, so a bare console bar would look like one task run twice."""
    if progress_notifier is None:
        return frames
    iterator = progress_notifier.iterator(frames, total=len(frames))
    if hasattr(iterator, 'set_description'):              # tqdm; the GUI notifier has no label
        iterator.set_description(label)
    return iterator


@njit(cache=True, parallel=True)
def _ball_pairs_sorted(det_y, det_x, order, row_start, row0, query, radius):
    """Detections within ``radius`` of each query point, by scanning the image rows the disk spans.

    ``order`` sorts the detections by ``floor(det_y)``; ``row_start[k]`` is the first of them in row
    ``row0 + k``. Returns ``(counts, det)``: the hits of query ``p`` are ``det[sum(counts[:p]):][:counts[p]]``,
    ascending, the output of ``cKDTree(dets).query_ball_point(query, radius)`` flattened."""
    n_query, n_rows = query.shape[0], row_start.shape[0] - 1
    r2 = radius * radius
    counts = np.zeros(n_query, dtype=np.int64)
    for p in prange(n_query):
        y, x = query[p, 0], query[p, 1]
        lo = max(int(np.floor(y - radius)) - row0, 0)
        hi = min(int(np.floor(y + radius)) - row0, n_rows - 1)
        c = 0
        for k in range(lo, hi + 1):
            for s in range(row_start[k], row_start[k + 1]):
                dy, dx = det_y[order[s]] - y, det_x[order[s]] - x
                if dy * dy + dx * dx <= r2:
                    c += 1
        counts[p] = c
    offsets = np.zeros(n_query + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    det = np.empty(offsets[-1], dtype=np.int64)
    for p in prange(n_query):
        y, x = query[p, 0], query[p, 1]
        lo = max(int(np.floor(y - radius)) - row0, 0)
        hi = min(int(np.floor(y + radius)) - row0, n_rows - 1)
        o = offsets[p]
        for k in range(lo, hi + 1):
            for s in range(row_start[k], row_start[k + 1]):
                dy, dx = det_y[order[s]] - y, det_x[order[s]] - x
                if dy * dy + dx * dx <= r2:
                    det[o] = order[s]
                    o += 1
        det[offsets[p]:o].sort()
    return counts, det


def _candidate_pairs(dets: np.ndarray, query: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(counts, det)`` of the detections within ``radius`` of each query point, exactly as
    ``cKDTree(dets).query_ball_point(query, radius)`` (sorted per query) flattened, without one Python
    list per query point. The detections are M-band skeleton pixels, so bucketing them by image row is
    an exact and cheap spatial index."""
    dets = np.asarray(dets, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    rows = np.floor(dets[:, 0]).astype(np.int64)
    order = np.argsort(rows, kind='stable')
    row0 = int(rows.min())
    row_start = np.searchsorted(rows[order], np.arange(row0, int(rows.max()) + 2))
    return _ball_pairs_sorted(dets[:, 0].copy(), dets[:, 1].copy(), order, row_start, row0, query, float(radius))


@njit(cache=True)
def _find_root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _component_cost_matrices(cost, qp, det, n_qp, n_det, max_component):
    """Split the bipartite candidate graph into connected components and build each one's dense cost matrix
    for :func:`_assign_optimal`: rows = its query points ascending, columns = its detections ascending,
    unmatchable cells = the sentinel ``2 min(k_q, k_d) + 1``, each pair's cell = the lowest cost given for it.

    Returns per component (in order of their first pair): the query ids and detection ids (concatenated, with
    offsets) and the matrix (flattened row-major, with offsets; empty for components above ``max_component``,
    which are solved greedily), plus each pair's component."""
    n_pairs = qp.shape[0]
    parent = np.arange(n_qp + n_det)
    for p in range(n_pairs):
        a, b = _find_root(parent, qp[p]), _find_root(parent, n_qp + det[p])
        if a != b:
            parent[max(a, b)] = min(a, b)
    comp_of_root = np.full(n_qp + n_det, -1, dtype=np.int64)
    comp = np.empty(n_pairs, dtype=np.int64)
    n_comp = 0
    for p in range(n_pairs):
        r = _find_root(parent, qp[p])
        if comp_of_root[r] < 0:
            comp_of_root[r] = n_comp
            n_comp += 1
        comp[p] = comp_of_root[r]
    # pairs grouped by component, original order within a component
    pair_start = np.zeros(n_comp + 1, dtype=np.int64)
    for p in range(n_pairs):
        pair_start[comp[p] + 1] += 1
    pair_start = np.cumsum(pair_start)
    fill = pair_start[:-1].copy()
    by_comp = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        by_comp[fill[comp[p]]] = p
        fill[comp[p]] += 1
    k_q = np.empty(n_comp, dtype=np.int64)
    k_d = np.empty(n_comp, dtype=np.int64)
    for c in range(n_comp):
        members = by_comp[pair_start[c]:pair_start[c + 1]]
        k_q[c] = np.unique(qp[members]).shape[0]
        k_d[c] = np.unique(det[members]).shape[0]
    q_start = np.zeros(n_comp + 1, dtype=np.int64)
    d_start = np.zeros(n_comp + 1, dtype=np.int64)
    m_start = np.zeros(n_comp + 1, dtype=np.int64)
    for c in range(n_comp):
        q_start[c + 1] = q_start[c] + k_q[c]
        d_start[c + 1] = d_start[c] + k_d[c]
        small = max(k_q[c], k_d[c]) <= max_component
        m_start[c + 1] = m_start[c] + (k_q[c] * k_d[c] if small else 0)
    q_ids = np.empty(q_start[-1], dtype=np.int64)
    d_ids = np.empty(d_start[-1], dtype=np.int64)
    mats = np.empty(m_start[-1], dtype=np.float64)
    for c in range(n_comp):
        members = by_comp[pair_start[c]:pair_start[c + 1]]
        uq = np.unique(qp[members])
        ud = np.unique(det[members])
        q_ids[q_start[c]:q_start[c + 1]] = uq
        d_ids[d_start[c]:d_start[c + 1]] = ud
        if max(k_q[c], k_d[c]) > max_component:
            continue
        mat = mats[m_start[c]:m_start[c + 1]]
        mat[:] = 2.0 * min(k_q[c], k_d[c]) + 1.0
        for p in members:
            cell = np.searchsorted(uq, qp[p]) * k_d[c] + np.searchsorted(ud, det[p])
            mat[cell] = min(mat[cell], np.float64(cost[p]))
    return q_ids, q_start, d_ids, d_start, mats, m_start, comp


@njit(cache=True, parallel=True)
def _nan_skipping_median(values):
    """``np.nanmedian(values, axis=1)`` for float64 ``(n, k, m)``: per row and channel the middle of the
    finite values, or ``(low + high) / 2`` for an even count; NaN without any."""
    n, k, m = values.shape
    out = np.empty((n, m))
    for i in prange(n):
        buf = np.empty(k)
        for j in range(m):
            c = 0
            for q in range(k):
                v = values[i, q, j]
                if not np.isnan(v):
                    buf[c] = v
                    c += 1
            if c == 0:
                out[i, j] = np.nan
                continue
            s = np.sort(buf[:c])
            out[i, j] = s[c // 2] if c % 2 else (s[c // 2 - 1] + s[c // 2]) / 2
    return out


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
    med = _nan_skipping_median(d)
    # Points with no reference inside the radius fall back to the global median
    # (a rigid drift of the whole field is still better than assuming no motion).
    isolated = ~np.isfinite(med).all(axis=1)
    if isolated.any():
        med[isolated] = np.median(ref_disp, axis=0)
    return np.nan_to_num(med).astype(np.float32)


def _coherent_component(points_yx: np.ndarray, values: np.ndarray, radius: float,
                        k: int = 32) -> np.ndarray:
    """Median of ``values`` over each point's neighbours within ``radius`` px.

    Used on the perpendicular component of the image-flow prediction: the flow's
    lateral component on striations is dominated by the aperture problem, whose
    error varies from block to block, whereas a real lateral motion is shared by
    the whole neighbourhood. ``k`` nearest neighbours are considered (including
    the point itself); a point with no neighbour in range keeps its own value.
    """
    pts = np.asarray(points_yx, dtype=np.float64).reshape(-1, 2)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = pts.shape[0]
    if n < 2 or radius <= 0:
        return v.astype(np.float32)
    kk = int(min(k, n))
    dist, idx = cKDTree(pts).query(pts, k=kk)
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    neigh = np.where(dist <= radius, v[idx], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(neigh, axis=1)
    return np.where(np.isfinite(med), med, v).astype(np.float32)


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


def rest_frames_from_trace(trace: np.ndarray) -> np.ndarray:
    """Frames in which a cell is at rest, from its average sarcomere length over time.

    The rest level is the most occupied level of the trace, not its median: a cell that
    contracts for half of each cycle has its median between rest and peak contraction.
    A frame is at rest if the trace and both neighbouring frames stay within
    15 % of the excursion from that level (its 99.5th percentile), and within at least twice the noise.
    If the beat amplitude is below 15 times the frame-to-frame noise, read from the lower quartile of the
    steps, rest cannot be told from beats and no frame counts as rest. The most occupied level can be
    the contracted end, if a cell spends most of its time contracted.

    Parameters
    ----------
    trace : np.ndarray
        ``(T,)`` average sarcomere length per frame; NaN where unknown.

    Returns
    -------
    np.ndarray
        ``(T,)`` bool.
    """
    trace = np.asarray(trace, dtype=float)
    finite = trace[np.isfinite(trace)]
    rest = np.zeros(trace.shape, dtype=bool)
    if finite.size < 3:
        return rest
    amplitude = float(finite.max() - finite.min())
    # the lower quartile reads the quiet steps, even if beats fill most frames; 0.4506 for Gaussian noise
    noise = np.nanpercentile(np.abs(np.diff(trace)), 25) / 0.4506
    if not np.isfinite(noise) or amplitude <= 0 or amplitude < _REST_MIN_SNR * noise:
        return rest
    width = max(noise, amplitude / 60)
    counts, edges = np.histogram(finite, np.arange(finite.min() - width, finite.max() + 2 * width, width))
    k = int(np.convolve(counts, [1, 2, 1], 'same').argmax())
    deviation = np.abs(trace - (edges[k] + edges[k + 1]) / 2)
    with np.errstate(invalid='ignore'):
        rest = deviation < max(_REST_BAND_FRAC * np.nanpercentile(deviation, 99.5), 2 * noise)
    return rest & np.r_[rest[1:], True] & np.r_[True, rest[:-1]]


def _median_detections_per_frame(pos_vectors_px_all) -> float:
    """Median number of sarcomere vectors per frame.

    This is the denominator of the fragmentation ratio: with one track per vector
    the track count equals it. Note it is *not* a count of physical sarcomeres —
    the vectors are a dense (~1 px) sampling along each M-band midline, so one
    midline contributes tens of them.
    """
    counts = [0 if p is None else len(p) for p in pos_vectors_px_all]
    return float(np.median(counts)) if counts else 0.0



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
    # components and their cost matrices in one compiled pass; scipy solves each matrix
    q_ids_all, q_start, d_ids_all, d_start, mats, m_start, comp = _component_cost_matrices(
        np.ascontiguousarray(cost), np.ascontiguousarray(qp, dtype=np.int64),
        np.ascontiguousarray(det, dtype=np.int64), int(n_qp), int(n_det), _LAP_MAX_COMPONENT)
    out_q: List[np.ndarray] = []
    out_d: List[np.ndarray] = []
    for c in range(len(q_start) - 1):
        q_ids, d_ids = q_ids_all[q_start[c]:q_start[c + 1]], d_ids_all[d_start[c]:d_start[c + 1]]
        k_q, k_d = len(q_ids), len(d_ids)
        if max(k_q, k_d) > _LAP_MAX_COMPONENT:
            # Safety valve: an unusually large component would make the dense
            # O(k³) solve dominate. Fall back to a greedy claim by cost.
            sl = np.flatnonzero(comp == c)
            q_inv = np.searchsorted(q_ids, qp[sl])
            d_inv = np.searchsorted(d_ids, det[sl])
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
        mat = mats[m_start[c]:m_start[c + 1]].reshape(k_q, k_d)
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
    # every interior gap of every track at once: consecutive observations of one track, 2..max_gap+1 apart
    track, frame = np.nonzero(tracks_observed[:n_tracks])
    span = np.diff(frame)
    is_gap = (track[1:] == track[:-1]) & (span > 1) & (span - 1 <= int(max_gap))
    i, a, span = track[1:][is_gap], frame[:-1][is_gap], span[is_gap]
    if i.size == 0:
        return 0
    b = a + span
    # one entry per gap frame: k = 1 .. span - 1, at weight k / span
    n_fill = span - 1
    gap = np.repeat(np.arange(i.size), n_fill)
    k = np.arange(gap.size) - np.repeat(np.cumsum(n_fill) - n_fill, n_fill) + 1
    w = k / span[gap].astype(np.float64)
    rows, cols = i[gap], a[gap] + k
    # An observed anchor can still carry a NaN length (its detection had
    # none), in which case there is nothing to interpolate between.
    s_a, s_b = tracks_slen[i, a], tracks_slen[i, b]
    fill_s = (np.isfinite(s_a) & np.isfinite(s_b))[gap]
    tracks_slen[rows[fill_s], cols[fill_s]] = ((1.0 - w) * s_a[gap] + w * s_b[gap])[fill_s]
    o_a, o_b = tracks_ori[i, a], tracks_ori[i, b]
    fill_o = (np.isfinite(o_a) & np.isfinite(o_b))[gap]
    s2 = (1.0 - w) * np.sin(2.0 * o_a)[gap] + w * np.sin(2.0 * o_b)[gap]
    c2 = (1.0 - w) * np.cos(2.0 * o_a)[gap] + w * np.cos(2.0 * o_b)[gap]
    tracks_ori[rows[fill_o], cols[fill_o]] = (0.5 * np.arctan2(s2, c2))[fill_o]
    return int(fill_s.sum())


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
    predictor: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    max_disp_along_rest_um: Optional[float] = 0.8,
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
        Match gate along the sarcomere axis, in µm, measured from the track's prediction,
        regardless of pixel size. On a rest frame the prediction can be the rest position, so
        the step from the previous frame can be larger. Default is 1.0.
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
        None). Interior gaps only, and ``motion.tracks.observed`` stays False there, so no
        real-observation metric is affected. Set to 0 to keep every gap frame NaN.
        Kept deliberately short (default 0.05 s): a longer fill would start to span
        a contraction and invent length dynamics.

    predictor : callable or None, optional
        ``predictor(t, yx) -> (Q, 2)``: displacement in px from frame index ``t`` to
        ``t + 1`` of the tracked sequence at the ``(Q, 2)`` positions ``yx``, e.g.
        :class:`sarcasm.analysis.optical_flow.ImageFlowPredictor`. When given, every
        live query point is moved by it before gating (along-axis component always,
        perpendicular component only above ``_PREDICT_PERP_KEEP_FRAC`` of the
        perpendicular gate). None (default) keeps the hold-position / neighbour-advection
        prediction. Use it when the per-frame motion is a sizeable fraction of a
        sarcomere length (coarse frame rates); at ≥ ~30 fps it changes nothing.
    max_disp_along_rest_um : float or None, optional
        Match gate along the sarcomere axis on rest frames, in µm, capped at half a sarcomere
        length. On these frames a track is also predicted at its rest position, so that a beat
        cannot move it to another sarcomere for good, unless under half as many rest positions as
        current positions hold a detection: then the tissue did not return, and the tracks keep
        following. None switches both off. A movie in which no rest can be detected tracks as with
        None. Default is 0.8.

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
        ``max_gap_interpolation_s``, and ``'motion.tracks.rest_frames'`` holds the ``(T,)`` rest
        frames found from the median length of the tracks observed in at least 30 % of the frames, or
        from the detections if no track reaches that, and none if the level found is the contracted end.
        The tracker itself uses those from the detections.
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

    # Rest frames for the tracker itself come from the detections: no tracks exist yet.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        rest_frames = rest_frames_from_trace(np.array(
            [np.nanmedian(s) if s is not None and len(s) > 0 else np.nan for s in sarcomere_lengths_all]))
    anchor_at_rest = max_disp_along_rest_um is not None and bool(rest_frames.any())
    rest_along2 = max_along2
    if anchor_at_rest:
        rest_along_px = float(max_disp_along_rest_um) / px
        if median_slen_px is not None and median_slen_px > 0:
            rest_along_px = min(rest_along_px, _ALONG_REST_SLEN_FRAC * median_slen_px)
        rest_along2 = min(max_along2, rest_along_px * rest_along_px)
    rest_alpha = min(1.0, float(frametime) / _REST_POSITION_TAU_S) if frametime and frametime > 0 else 0.05
    rest_radius_px = max(_REST_DETECTION_RADIUS_UM / px, 1.0)
    max_radius = float(max_disp_along_px)

    # Neighbourhood radius for the local-motion estimate: a few sarcomeres, so the
    # estimate is local to one myofibril bundle yet has enough members to be robust.
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
    rest_y = np.full(capacity, np.nan, dtype=np.float32)
    rest_x = np.full(capacity, np.nan, dtype=np.float32)

    def _grow(needed: int):
        """Double capacity until >= ``needed`` and resize all SoA arrays."""
        nonlocal capacity, positions_px, tracks_slen, tracks_ori, tracks_observed
        nonlocal tracks_detection_id, tracks_midline_id
        nonlocal start_frame_arr, last_y, last_x, last_ori, frames_since_observation, alive, rest_y, rest_x
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
        rest_y = _resize(rest_y, np.nan)
        rest_x = _resize(rest_x, np.nan)
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
        if anchor_at_rest and rest_frames[0]:
            rest_y[sl] = pos0[:, 0]
            rest_x[sl] = pos0[:, 1]

    # --- frame-to-frame: advect, match, spawn ---
    logger.info("Tracking frames…")
    _frames = _with_progress(range(T - 1), progress_notifier, 'Tracking')
    for t in _frames:
        # 1. advect unmatched tracks along their sarcomere axis by the median step
        #    of the neighbours observed in both of the last two frames
        live = np.flatnonzero(alive[:n_tracks])
        if live.size > 0 and predictor is not None:
            # image-flow prediction for every live query point: along-axis component
            # always, perpendicular component only where it is large (see module docstring)
            ys = last_y[live]
            xs = last_x[live]
            disp = np.asarray(predictor(t, np.column_stack((ys, xs))), dtype=np.float32).reshape(-1, 2)
            if disp.shape[0] != live.size:
                raise ValueError(
                    f"predictor returned {disp.shape[0]} displacements for {live.size} query points at frame {t}.")
            s_live = np.sin(last_ori[live])
            c_live = np.cos(last_ori[live])
            along = disp[:, 0] * s_live + disp[:, 1] * c_live
            perp = -disp[:, 0] * c_live + disp[:, 1] * s_live
            # real lateral motion is coherent over a sarcomere length, aperture noise is
            # not: take the neighbourhood median before deciding whether to keep it
            perp = _coherent_component(np.column_stack((ys, xs)), perp,
                                       radius=median_slen_px if median_slen_px else max_disp_along_px)
            perp = np.where(np.abs(perp) > _PREDICT_PERP_KEEP_FRAC * max_disp_perp_px, perp, 0.0)
            last_y[live] = ys + along * s_live - perp * c_live
            last_x[live] = xs + along * c_live + perp * s_live
        elif live.size > 0:
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

        at_rest = anchor_at_rest and bool(rest_frames[t + 1])

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

        # on a rest frame the tracks go back to their rest positions, whatever the last beat did to them. If far fewer
        # detections lie there than where the tracks are now, the tissue did not return: the tracks keep following.
        # Flicker empties both alike, so it does not count.
        known = live[np.isfinite(rest_y[live])] if at_rest and n_det > 0 else live[:0]
        if known.size > 0:
            tree = cKDTree(dets)
            at_home = tree.query(np.column_stack((rest_y[known], rest_x[known])), distance_upper_bound=rest_radius_px)[0]
            at_now = tree.query(np.column_stack((last_y[known], last_x[known])), distance_upper_bound=rest_radius_px)[0]
            if np.isfinite(at_home).sum() >= 0.5 * np.isfinite(at_now).sum():
                last_y[known] = rest_y[known]
                last_x[known] = rest_x[known]

        # Bool masks over the current track slots / detections so we can do
        # claim bookkeeping in pure numpy without Python sets.
        claimed_qp_mask = np.zeros(n_tracks, dtype=bool)
        claimed_det_mask = np.zeros(n_det, dtype=bool)

        # 3. kd-tree + vectorized gate on all (qp, candidate) pairs; the gate does
        #    not widen with gap length (that would trade identity for fragmentation)
        if n_det > 0 and live.size > 0:
            live_pos = np.column_stack((last_y[live], last_x[live]))
            # the search radius must cover the along gate, the larger axis.
            counts, det_flat = _candidate_pairs(dets, live_pos, max_radius)
            total = int(det_flat.size)

            if total > 0:
                qp_flat_rel = np.repeat(
                    np.arange(live.size, dtype=np.int64), counts,
                )
                qp_abs = live[qp_flat_rel]
                along_budget = rest_along2 if at_rest else max_along2
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
                    if at_rest:
                        first = ~np.isfinite(rest_y[qp_acc])
                        rest_y[qp_acc] = np.where(first, cy_acc, rest_y[qp_acc] + rest_alpha * (cy_acc - rest_y[qp_acc]))
                        rest_x[qp_acc] = np.where(first, cx_acc, rest_x[qp_acc] + rest_alpha * (cx_acc - rest_x[qp_acc]))

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
                if at_rest:
                    rest_y[new_slots] = new_y
                    rest_x[new_slots] = new_x
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

    # Everything after tracking reads rest from the tracked lengths: all detections, including the
    # flickering ones, can split a rest period into short runs on noisy movies.
    covered = out_tracks_observed.mean(axis=1) >= 0.3
    if covered.any():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            trace = np.nanmedian(np.where(out_tracks_observed[covered], out_tracks_slen[covered], np.nan), axis=0)
            rest_frames = rest_frames_from_trace(trace)
            # Far more other frames above that level than below it: it is the contracted end, not rest.
            # Detection medians cannot tell: low-count movies read long in a beat.
            offset = trace[~rest_frames] - np.nanmedian(trace[rest_frames])
            if rest_frames.any() and np.count_nonzero(offset > 0) > 3 * np.count_nonzero(offset < 0):
                rest_frames = np.zeros_like(rest_frames)

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
        'motion.tracks.rest_frames': rest_frames,
    }

    return result


# ---------------------------------------------------------------------------
# Length refinement along tracks
# ---------------------------------------------------------------------------

def track_axis_orientations(
    orientations: np.ndarray,
    observed: np.ndarray,
    window_frames: Optional[int] = None,
) -> np.ndarray:
    """The sarcomere axis of every track: its detected orientations without their fast changes.

    Sarcomeres do not turn during a contraction. Their Z-bands tilt, because
    neighbouring myofibrils shorten by different amounts (a shear), and the
    per-frame orientation of a sarcomere vector, the normal of the Z-bands at its
    M-band pixel in that frame, follows that tilt and jitters on top of it. On a
    61-fps movie the Z-bands tilted by 5° (SD) during contractions, while the axis
    of the sarcomeres, from the tracked displacement field, turned by less than 1°.
    So each track's axis is taken as the median of its detected orientations, over
    the whole track by default or, for long recordings in which the tissue slowly
    turns, over a running window.

    The median runs on the double-angle components (``cos 2θ``, ``sin 2θ``), since
    orientations are axial; it ignores the contraction tilt as long as contractions
    fill less than half of the frames it covers. Unobserved frames are bridged by
    linear interpolation for a running window only and stay NaN in the output.

    Parameters
    ----------
    orientations : np.ndarray
        ``(n_tracks, T)`` orientations in radians.
    observed : np.ndarray
        ``(n_tracks, T)`` bool, True where a detection was matched.
    window_frames : int, optional
        None (default) gives every track one orientation; an odd number of frames
        gives a running median over that window; ``1`` returns the input.

    Returns
    -------
    np.ndarray
        ``(n_tracks, T)`` orientations in [0, π), NaN on unobserved frames.
    """
    from scipy.ndimage import median_filter
    ori = np.asarray(orientations, dtype=float)
    ok = np.asarray(observed, dtype=bool) & np.isfinite(ori)
    out = np.where(ok, np.mod(ori, np.pi), np.nan)
    if window_frames is not None:
        window_frames = int(window_frames)
        if window_frames % 2 == 0:
            window_frames += 1
        if window_frames <= 1:
            return out
    idx = np.arange(ori.shape[1])
    for i in range(ori.shape[0]):
        m = ok[i]
        if not m.any():
            continue
        c, s = np.cos(2.0 * ori[i, m]), np.sin(2.0 * ori[i, m])
        if window_frames is None:
            out[i, m] = np.mod(0.5 * np.arctan2(np.median(s), np.median(c)), np.pi)
        else:
            c = median_filter(np.interp(idx, idx[m], c), window_frames, mode='nearest')
            s = median_filter(np.interp(idx, idx[m], s), window_frames, mode='nearest')
            out[i, m] = np.mod(0.5 * np.arctan2(s, c), np.pi)[m]
    return out


def refine_track_lengths(
    positions_px: np.ndarray,
    orientations: np.ndarray,
    observed: np.ndarray,
    read_zbands: Callable[[int], np.ndarray],
    pixelsize: float,
    window_frames: Optional[int] = None,
    slen_lims: Tuple[float, float] = (1, 3),
    interp_factor: int = 4,
    linewidth: float = 0.3,
    interpolation_method: str = 'akima',
    peak_prominence: float = 0.4,
    progress_notifier=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-measure every observed track frame's length along the track's sarcomere axis and move
    its position to the midpoint between its two Z-bands.

    The axis of each track is its detected orientation without the fast changes
    (:func:`track_axis_orientations`), then the Z-band profile is sampled again at
    the tracked M-band pixel along that axis and the length extracted with the same
    estimator and settings as
    :func:`sarcasm.analysis.sarcomere_vectors.get_sarcomere_vectors`. Along the
    detection's orientation, i.e. normal to Z-bands that a shear has tilted by φ,
    the spacing is shorter than along the filaments by cos φ; along the axis it is
    not. With the detection orientations unchanged (``window_frames=1``) this
    reproduces the detection's length exactly.

    Parameters
    ----------
    positions_px : np.ndarray
        ``(n_tracks, T, 2)`` tracked M-band pixels (row, col).
    orientations, observed : np.ndarray
        ``(n_tracks, T)`` per-frame orientations (radians) and the observed mask.
    read_zbands : callable
        ``read_zbands(k)`` returns the 2D Z-band map of tracker frame ``k``, the map
        the vectors were measured on.
    pixelsize : float
        Pixel size in µm.
    window_frames : int, optional
        None (default): one axis per track; an odd number of frames: a running
        median over that window; ``1`` re-measures along the detection orientations
        unchanged (see :func:`track_axis_orientations`).
    slen_lims, interp_factor, linewidth, interpolation_method, peak_prominence
        Length-estimator settings, as used by ``analyze_sarcomere_vectors``.
    progress_notifier : ProgressNotifier, optional
        Reports per-frame progress.

    Returns
    -------
    slen : np.ndarray
        ``(n_tracks, T)`` lengths in µm; NaN on unobserved frames and where the
        profile along the axis has no valid pair of Z-bands.
    orientations_axis : np.ndarray
        ``(n_tracks, T)`` axis orientations, NaN on unobserved frames.
    positions_px : np.ndarray
        ``(n_tracks, T, 2)`` the input positions, each measured frame moved along the
        axis to the midpoint between the two Z-band peaks of its profile. The tracker
        follows integer M-band skeleton pixels, so its raw positions carry a +-0.5 px
        quantisation that would otherwise appear as jitter in the position and Z-band
        trajectories; the midpoint is the sub-pixel M-band position, as
        :func:`~sarcasm.analysis.sarcomere_vectors.get_sarcomere_vectors` uses for the
        detections. Frames without a valid pair of Z-bands keep their input position.
    """
    from ..utils import Utils, _profile_lines
    positions_px = np.asarray(positions_px, dtype=float)
    ori_s = track_axis_orientations(orientations, observed, window_frames)
    n, T = ori_s.shape
    slen = np.full((n, T), np.nan, dtype=np.float32)
    positions_out = positions_px.copy()
    linewidth_px = max(int(round(linewidth / pixelsize, 0)), 1)
    half_length = (slen_lims[1] * 1.3) / 2 / pixelsize
    frames = _with_progress(range(T), progress_notifier, 'Refining lengths')
    for k in frames:
        rows = np.flatnonzero(np.isfinite(ori_s[:, k]))
        if rows.size == 0:
            continue
        theta = ori_s[rows, k]
        centres = positions_px[rows, k].T
        direction = np.stack((np.sin(theta) * half_length, np.cos(theta) * half_length), axis=0)
        profiles = _profile_lines(np.asarray(read_zbands(k)), centres + direction, centres - direction,
                                  linewidth=linewidth_px)
        lengths, offsets = Utils.process_profiles_batch(
            profiles, pixelsize, slen_lims=slen_lims, interp_factor=interp_factor,
            interpolation_method=interpolation_method, prominence=peak_prominence)
        slen[rows, k] = lengths
        # the profile runs from centre + direction to centre - direction, so a positive offset of the
        # Z-band midpoint lies against the direction (the convention of get_sarcomere_vectors)
        ok = np.isfinite(lengths) & np.isfinite(offsets)
        shift = (offsets[ok] / pixelsize)[:, None] * np.stack((np.sin(theta[ok]), np.cos(theta[ok])), axis=1)
        positions_out[rows[ok], k] -= shift
    return slen, ori_s, positions_out


def fill_track_gaps(
    positions_px: np.ndarray,
    axis: np.ndarray,
    observed: np.ndarray,
    slen: np.ndarray,
    positions_refined_px: np.ndarray,
    rest_frames: np.ndarray,
    read_zbands: Callable[[int], np.ndarray],
    pixelsize: float,
    min_coverage: float = 0.5,
    slen_lims: Tuple[float, float] = (1, 3),
    interp_factor: int = 4,
    linewidth: float = 0.3,
    interpolation_method: str = 'akima',
    peak_prominence: float = 0.4,
    progress_notifier=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure the interior gap frames of well-covered tracks at the track's own position.

    A gap mostly means that the M-band pixel flickered, often at the end of an M-band, not that
    the sarcomere left: its Z-bands are usually still in the map. The position in a gap is the tracker's
    prediction (the rest position on rest frames, otherwise carried with the neighbouring tracks or by image
    flow), corrected linearly so that it meets the track's next observation. A fill never feeds back into
    the tracker; only the stored position of a kept frame changes, to the measured Z-band midpoint.

    A measured frame is kept if its Z-band profile is clean
    (:meth:`sarcasm.utils.Utils.grade_profile_pairs`), and on rest frames only if it also lies
    within 0.2 µm of the length interpolated from the track's real frames.

    Filled frames follow a contraction less reliably than real ones (against the real frames, slope
    0.6-1.1 on all but one of 14 recordings with a detectable rest), so their lengths are returned apart from
    the real ones, for display and coverage only.

    Parameters
    ----------
    positions_px : np.ndarray
        ``(n_tracks, T, 2)`` tracker positions before refinement: detection pixels on observed frames,
        predictions on gap frames.
    axis : np.ndarray
        ``(n_tracks, T)`` sarcomere axes from :func:`refine_track_lengths`; NaN on unobserved frames.
    observed : np.ndarray
        ``(n_tracks, T)`` real observations.
    slen : np.ndarray
        ``(n_tracks, T)`` lengths in µm; only observed frames are read.
    positions_refined_px : np.ndarray
        ``(n_tracks, T, 2)`` refined positions. Kept frames get their measured Z-band midpoint here.
    rest_frames : np.ndarray
        ``(T,)`` rest frames.
    read_zbands : callable
        ``read_zbands(k)`` returns the 2D Z-band map of tracker frame ``k``.
    pixelsize : float
        Pixel size in µm.
    min_coverage : float, optional
        Tracks with fewer real observations than this fraction of the frames are left alone.
        Default is 0.5.
    slen_lims, interp_factor, linewidth, interpolation_method, peak_prominence
        Length-estimator settings, as in :func:`refine_track_lengths`.
    progress_notifier : ProgressNotifier, optional
        Reports per-frame progress.

    Returns
    -------
    slen_filled : np.ndarray
        ``(n_tracks, T)`` real lengths plus kept filled frames, NaN elsewhere.
    filled : np.ndarray
        ``(n_tracks, T)`` bool, the kept filled frames.
    positions_out_px : np.ndarray
        ``positions_refined_px`` with the Z-band midpoints of the kept frames.
    """
    from ..utils import Utils, _profile_lines
    observed = np.asarray(observed, dtype=bool)
    n, T = observed.shape
    real = np.where(observed, np.asarray(slen, dtype=float), np.nan)
    slen_filled = real.astype(np.float32)
    filled = np.zeros((n, T), dtype=bool)
    positions_out = np.array(positions_refined_px, dtype=np.float32)
    rows = np.flatnonzero(observed.mean(axis=1) >= min_coverage)
    if rows.size == 0:
        return slen_filled, filled, positions_out

    positions_px = np.asarray(positions_px, dtype=float)
    measured = np.isfinite(real[rows])
    frame_ids = np.arange(T)
    before = np.maximum.accumulate(np.where(measured, frame_ids, -1), axis=1)
    after = np.minimum.accumulate(np.where(measured, frame_ids, T)[:, ::-1], axis=1)[:, ::-1]
    row_rel, frame = np.nonzero(~observed[rows] & (before >= 0) & (after < T))
    if frame.size == 0:
        return slen_filled, filled, positions_out
    a, b, track = before[row_rel, frame], after[row_rel, frame], rows[row_rel]
    weight = (frame - a) / (b - a)
    reference = positions_px[track, frame] + weight[:, None] * (positions_px[track, b] - positions_px[track, b - 1])
    expected = real[track, a] + weight * (real[track, b] - real[track, a])
    theta = np.asarray(axis, dtype=float)[track, a]

    order = np.argsort(frame, kind='stable')
    starts = np.searchsorted(frame[order], frame_ids)
    stops = np.searchsorted(frame[order], frame_ids, side='right')
    linewidth_px = max(int(round(linewidth / pixelsize, 0)), 1)
    # half-integer, so that rounding cannot make the lines differ in samples: the profiles have to stack
    half_length = np.floor((slen_lims[1] * 1.3) / 2 / pixelsize) + 0.5
    for k in _with_progress(range(T), progress_notifier, 'Filling gaps'):
        idx = order[starts[k]:stops[k]]
        if idx.size == 0:
            continue
        ok_axis = np.isfinite(theta[idx])
        idx = idx[ok_axis]
        if idx.size == 0:
            continue
        centres = reference[idx].T
        direction = np.stack((np.sin(theta[idx]) * half_length, np.cos(theta[idx]) * half_length), axis=0)
        profiles = np.asarray(_profile_lines(np.asarray(read_zbands(k)), centres + direction, centres - direction,
                                             linewidth=linewidth_px))
        lengths, offsets = Utils.process_profiles_batch(
            profiles, pixelsize, slen_lims=slen_lims, interp_factor=interp_factor,
            interpolation_method=interpolation_method, prominence=peak_prominence)
        asymmetry, third = Utils.grade_profile_pairs(profiles, pixelsize, lengths, offsets, interp_factor=interp_factor)
        keep = np.isfinite(lengths) & (asymmetry <= _FILL_MAX_ASYMMETRY) & (third <= _FILL_MAX_THIRD_PEAK)
        if rest_frames[k]:
            keep &= np.abs(lengths - expected[idx]) <= _FILL_REST_TOL_UM
        chosen = idx[keep]
        slen_filled[track[chosen], k] = lengths[keep]
        filled[track[chosen], k] = True
        shift = (offsets[keep] / pixelsize)[:, None] * np.stack((np.sin(theta[chosen]), np.cos(theta[chosen])), axis=1)
        positions_out[track[chosen], k] = reference[chosen] - shift
    return slen_filled, filled, positions_out
