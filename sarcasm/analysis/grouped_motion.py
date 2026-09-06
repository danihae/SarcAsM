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

"""Grouping-blind contraction analysis over per-track sarcomere lengths.

The 2D tracker (:func:`sarcasm.analysis.sarcomere_tracking.track_sarcomere_vectors`)
already produces a per-sarcomere length time-series ``tracks_slen`` of shape
``(n_tracks, T)``. Every post-tracking "method" — pool, M-band, myofibril, domain,
custom — is just a *grouping* of those tracks. This module turns a grouping
(a per-track integer label) into:

1. a per-group aggregated length time-series ``(n_groups, T)`` (:func:`aggregate_group_slen`), and
2. per-group contraction cycles + kinematics (:func:`run_cycle_engine`),

both built on the ContractionNet engine
(:mod:`sarcasm.analysis.contraction_analysis`), whose entry points are pure
functions of a ``(n_groups, T)`` length matrix plus ``frametime``.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

from sarcasm.analysis import contraction_analysis
from sarcasm.analysis.contraction_analysis import equilibrium_over_quiet  # noqa: F401  (public here too)


def aggregate_group_slen(
    tracks_slen: np.ndarray,
    group_id: np.ndarray,
    n_groups: int,
    aggregate: str = 'mean',
    slen_lims: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate per-track ``slen(t)`` into per-group ``slen(t)``.

    Parameters
    ----------
    tracks_slen : np.ndarray
        Per-track sarcomere length over time, shape ``(n_tracks, T)``; NaN on
        gap frames / before-start / after-close.
    group_id : np.ndarray
        Per-track group label, shape ``(n_tracks,)``. ``-1`` marks unassigned
        tracks (excluded from every group).
    n_groups : int
        Number of groups (labels ``0 .. n_groups-1``).
    aggregate : {'mean', 'median'}, optional
        How the members' lengths are combined into the group's ``slen`` (the
        signal fed to the contraction engine), over the members with a finite
        length in each frame. The per-frame spread (median/std/q25/q75) is
        returned regardless of this choice. Default is 'mean'.
    slen_lims : (float, float) or None, optional
        If given, member lengths outside ``[lo, hi]`` (µm) are set to NaN before
        aggregation. Default is None.

    Returns
    -------
    dict
        ``slen`` (the chosen aggregate), ``slen_median``,
        ``slen_std``, ``slen_q25``, ``slen_q75``,
        and ``n_members`` (int count of members with a finite length
        per frame), each shape ``(n_groups, T)``.
    """
    tracks_slen = np.asarray(tracks_slen, dtype=float)
    if tracks_slen.ndim != 2:
        tracks_slen = tracks_slen.reshape(len(group_id), -1)
    T = tracks_slen.shape[1]
    group_id = np.asarray(group_id).reshape(-1)

    if aggregate not in ('mean', 'median'):
        raise ValueError(f"aggregate must be 'mean' or 'median', got {aggregate!r}")
    agg_fn = np.nanmean if aggregate == 'mean' else np.nanmedian

    slen_ts = np.full((n_groups, T), np.nan)
    median_ts = np.full((n_groups, T), np.nan)
    std_ts = np.full((n_groups, T), np.nan)
    q25_ts = np.full((n_groups, T), np.nan)
    q75_ts = np.full((n_groups, T), np.nan)
    n_members = np.zeros((n_groups, T), dtype=np.int32)

    for g in range(n_groups):
        mask = group_id == g
        if not mask.any():
            continue
        sub = tracks_slen[mask]  # (k, T)
        if slen_lims is not None:
            lo, hi = float(slen_lims[0]), float(slen_lims[1])
            sub = np.where((sub >= lo) & (sub <= hi), sub, np.nan)
        finite = ~np.isnan(sub)
        n_members[g] = finite.sum(axis=0)
        # All-NaN columns must stay NaN; suppress the resulting RuntimeWarnings.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            slen_ts[g] = agg_fn(sub, axis=0)
            median_ts[g] = np.nanmedian(sub, axis=0)
            std_ts[g] = np.nanstd(sub, axis=0)
            q25_ts[g] = np.nanpercentile(sub, 25, axis=0)
            q75_ts[g] = np.nanpercentile(sub, 75, axis=0)

    return {
        'slen': slen_ts,
        'slen_median': median_ts,
        'slen_std': std_ts,
        'slen_q25': q25_ts,
        'slen_q75': q75_ts,
        'n_members': n_members,
    }


def run_cycle_engine(
    group_slen: np.ndarray,
    frametime: float,
    model_path: str,
    threshold: Optional[float] = None,
    contr_time_min: float = 0.2,
    merge_time_max: float = 0.05,
    buffer_frames: int = 3,
    min_valid_frames: float = 0.5,
    filter_params: Tuple[int, int] = (13, 5),
    group_label: str = "Domain",
    id_offset: int = 0,
) -> Dict[str, np.ndarray]:
    """Grouping-agnostic contraction analysis on a ``(n_groups, T)`` length matrix.

    Thin wrapper that runs the ContractionNet engine
    (:func:`contraction_analysis.detect_contractions` and
    :func:`contraction_analysis.analyze_contraction_parameters`) and returns
    their combined output dict (keys prefixed ``domain_*`` — the caller remaps
    the prefix to the grouping kind).

    Parameters
    ----------
    group_slen : np.ndarray
        Per-group sarcomere length over time, shape ``(n_groups, T)``.
    frametime : float
        Seconds per frame.
    model_path : str
        Path to the ContractionNet model used for contraction detection.
    threshold : float or None, optional
        Detection probability threshold. None (the default) uses the operating point the
        model was tuned for, read from its checkpoint. The right value is a property of the
        model; a threshold carried over from a different one is not meaningful.
    contr_time_min : float, optional
        Minimum contraction duration in seconds. Default is 0.2.
    merge_time_max : float, optional
        Maximum gap in seconds between contractions that are merged. Default is 0.05.
    buffer_frames : int, optional
        Frames from either end within which a cycle counts as incomplete. Incomplete
        cycles stay in the contraction mask but their duration-dependent metrics are
        NaN (see :func:`contraction_analysis.detect_contractions`). Default is 3.
    min_valid_frames : float, optional
        Minimum fraction of valid frames required per group. Default is 0.5.
    filter_params : (int, int), optional
        Savitzky-Golay (window_length, polyorder) for the velocity filter.
        Default is (13, 5).
    group_label : str, optional
        Label used only in the engine's per-group log messages. Default is "Domain".
    id_offset : int, optional
        Offset added to group ids in the engine's per-group log messages.
        Default is 0.

    Returns
    -------
    dict
        Combined ContractionNet output (keys prefixed ``domain_*``). Empty
        ``(0, T)`` / ``(0,)`` arrays when ``n_groups == 0``.
    """
    group_slen = np.asarray(group_slen, dtype=float)
    n_groups, T = group_slen.shape
    if n_groups == 0:
        return {
            'contr': np.zeros((0, T), dtype=bool),
            'n_contr': np.zeros(0, dtype=np.int32),
            'n_contr_complete': np.zeros(0, dtype=np.int32),
            'contr_complete': np.zeros((0, 1)),
            'labels_contr': np.zeros((0, T), dtype=np.int32),
            'beating_rate': np.zeros(0),
            'beating_rate_variability': np.zeros(0),
            'equ': np.zeros(0),
            'contr_max': np.zeros((0, 1)),
            'elong_max': np.zeros((0, 1)),
            'vel_contr_max': np.zeros((0, 1)),
            'vel_elong_max': np.zeros((0, 1)),
            'time_to_peak': np.zeros((0, 1)),
            'time_to_relax': np.zeros((0, 1)),
            'time_contr': np.zeros((0, 1)),
        }

    contr = contraction_analysis.detect_contractions(
        group_slen=group_slen,
        frametime=frametime,
        model_path=model_path,
        threshold=threshold,
        contr_time_min=contr_time_min,
        merge_time_max=merge_time_max,
        buffer_frames=buffer_frames,
        min_valid_frames=min_valid_frames,
        group_label=group_label,
        id_offset=id_offset,
    )
    params = contraction_analysis.analyze_contraction_parameters(
        group_slen=group_slen,
        group_labels_contr=contr['labels_contr'],
        domain_n_contr=contr['n_contr'],
        frametime=frametime,
        filter_params=filter_params,
        buffer_frames=buffer_frames,
    )
    return {**contr, **params}


def _interp_nan_1d(a: np.ndarray, max_gap: Optional[int] = None) -> np.ndarray:
    """Linear-interpolate the *interior* NaNs of a 1D array, up to ``max_gap`` long.

    Interior NaNs — those anchored by a finite value on both sides — are filled
    by linear interpolation. Leading and trailing NaNs (frames before the track
    first appears or after it is lost) are left as NaN: a track must NEVER carry
    a constant, fabricated length where it has no observation, as a held-constant
    edge corrupts every downstream contraction metric (equilibrium, delta-slen,
    velocity). An all-NaN input is returned unchanged (still all NaN).

    ``max_gap`` bounds how long a run may be and still be filled. Interpolating a
    long dropout invents a straight line across it, which at a typical beat can
    span an entire contraction and silently smooth it away; runs longer than
    ``max_gap`` are therefore left as NaN. ``None`` (or ``<= 0``) fills every
    interior run regardless of length."""
    a = np.asarray(a, dtype=float).copy()
    mask = np.isnan(a)
    if not mask.any() or mask.all():
        return a
    idx = np.arange(a.shape[0])
    finite = ~mask
    first, last = idx[finite][0], idx[finite][-1]
    # Interior = NaN strictly between the first and last finite sample.
    interior = mask & (idx > first) & (idx < last)
    if max_gap is not None and max_gap > 0 and interior.any():
        # Drop runs longer than max_gap from the fill set: a long dropout is a real
        # absence of data, not flicker to be bridged.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], interior.view(np.int8), [0]))))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start > max_gap:
                interior[start:stop] = False
    if interior.any():
        a[interior] = np.interp(idx[interior], idx[finite], a[finite])
    return a


def _chain_anchor_positions(member_pos: np.ndarray, ref_idx: int) -> np.ndarray:
    """``(K, 2)`` anchor position of each member, taken at the reference frame.

    The reference frame is the one the grouping and the head-to-tail ordering were
    built from, so anchoring there is consistent with the chain order by
    construction. A time-median anchor is **not** equivalent: two tracks that drift
    past each other over the movie collapse onto the same anchor even though they
    are a full sarcomere apart at the reference frame, which fabricates duplicate
    members in the chain. Members missing at the reference frame fall back to their
    first observed position, then to interpolation from their neighbours.
    """
    K, T, _ = member_pos.shape
    ref = int(np.clip(ref_idx, 0, max(T - 1, 0)))
    centre = member_pos[:, ref, :].copy()
    missing = ~np.isfinite(centre).all(axis=1)
    for k in np.flatnonzero(missing):
        finite = np.flatnonzero(np.isfinite(member_pos[k]).all(axis=1))
        if finite.size:
            centre[k] = member_pos[k, finite[np.argmin(np.abs(finite - ref))]]
    # A member never observed at all still has no anchor; interpolate from its
    # neighbours so the chain's arc coordinate stays continuous.
    for c in range(2):
        centre[:, c] = _interp_nan_1d(centre[:, c])
    return centre


def _chain_arc_coordinates(member_pos: np.ndarray, ref_idx: int = 0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-member fibre anchor and unit tangent from ordered member positions.

    Parameters
    ----------
    member_pos : np.ndarray
        ``(K, T, 2)`` member centre positions (µm) ordered head-to-tail, NaN on
        gap frames.
    ref_idx : int, optional
        Frame the chain geometry is anchored on. Default is 0.

    Returns
    -------
    anchor_arc : np.ndarray
        ``(K,)`` base arc coordinate of each member: the cumulative distance along
        the polyline through the members' reference-frame positions. Following the
        polyline (rather than projecting on one straight axis) keeps a curved
        fibre's coordinate monotone.
    tangent : np.ndarray
        ``(K, 2)`` unit tangent at each member (central difference of the
        neighbouring anchors, one-sided at the ends).
    centre : np.ndarray
        ``(K, 2)`` the anchor positions themselves.
    """
    centre = _chain_anchor_positions(member_pos, ref_idx)
    K = member_pos.shape[0]
    step = np.zeros(K)
    if K > 1:
        step[1:] = np.linalg.norm(np.diff(centre, axis=0), axis=1)
    anchor_arc = np.cumsum(step)
    tangent = np.zeros((K, 2))
    if K > 1:
        tangent[1:-1] = centre[2:] - centre[:-2]
        tangent[0] = centre[1] - centre[0]
        tangent[-1] = centre[-1] - centre[-2]
    else:
        tangent[0] = (1.0, 0.0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 0)
    return anchor_arc, tangent, centre


_INTERP_GAP_SECONDS = 0.05
"""Longest member dropout (s) still bridged by interpolation in a synthesized chain.

A physical duration, so the same real dropout is bridged at any frame rate. Kept
short deliberately: it exists to close brief detection flicker, not to invent a
straight line across a gap that can span a whole contraction. Matches the
tracker's ``_MERGE_GAP_SECONDS`` horizon for the same reason.
"""


def synthesize_loi_chain(member_slen: np.ndarray, frametime: float,
                         member_pos: Optional[np.ndarray] = None,
                         ref_idx: int = 0,
                         max_interp_seconds: Optional[float] = _INTERP_GAP_SECONDS):
    """Build an LOI-style ``(z_pos, slen, time)`` triple from an ordered chain of tracks.

    Turns K member sarcomere-length series ordered head-to-tail into the triple
    the :mod:`sarcasm.motion` LOI engine consumes, so it runs unmodified on a
    myofibril built from tracks.

    Each Z-band boundary is placed from **its own member's measured position**, so
    a member without an observation blanks only its own row. Accumulating the
    boundaries instead (``cumsum`` of the lengths) would let a single undefined
    member blank every boundary below it.

    Parameters
    ----------
    member_slen : np.ndarray
        ``(K, T)`` member sarcomere lengths ordered head-to-tail along the fibre,
        NaN on gap frames.
    frametime : float
        Seconds per frame.
    member_pos : np.ndarray or None, optional
        ``(K, T, 2)`` member centre positions in µm, same order as
        ``member_slen``. When given, ``z_pos`` is reconstructed from these
        measured positions. When None the boundaries are accumulated from the
        lengths instead, in which case a missing member propagates into every
        boundary below it.
    ref_idx : int, optional
        Frame index the chain geometry is anchored on — pass the reference frame
        the grouping/ordering was built from, so the arc coordinate agrees with
        the member order. Used only together with ``member_pos``. Default is 0.
    max_interp_seconds : float or None, optional
        Longest member dropout (s) still bridged by interpolation; longer gaps stay
        NaN in both ``slen`` and ``z_pos``. ``None``/0 bridges any interior gap.
        Default is :data:`_INTERP_GAP_SECONDS`.

    Returns
    -------
    z_pos : np.ndarray
        ``(K+1, T)`` arc position of the K+1 Z-band boundaries along the fibre
        (µm). With ``member_pos``, row ``k`` is member ``k``'s leading edge
        (``centre - slen/2``) and the last row is the final member's trailing
        edge, each depending only on that member. Note ``np.diff(z_pos)`` then
        equals ``slen`` only up to measurement noise: K sarcomeres yield 2K
        Z-band observations reconciled onto K+1 boundaries, so exact recovery and
        per-member independence cannot both hold. **``slen`` is the authoritative
        per-member series.**
    slen : np.ndarray
        ``(K, T)`` member lengths with interior gaps interpolated and
        leading/trailing gaps left as NaN (never held constant). This is the
        honest per-member length series: ``slen[k]`` is NaN exactly where member
        ``k`` has no observation, independent of the other members.
    time : np.ndarray
        ``(T,)`` time axis in seconds.
    """
    member_slen = np.asarray(member_slen, dtype=float)
    if member_slen.ndim != 2:
        raise ValueError('member_slen must be 2D (K, T).')
    K, T = member_slen.shape
    max_gap = (None if not max_interp_seconds or not frametime
               else max(1, int(round(float(max_interp_seconds) / float(frametime)))))
    slen = np.empty((K, T), dtype=float)
    for k in range(K):
        slen[k] = _interp_nan_1d(member_slen[k], max_gap=max_gap)
    time = np.arange(T) * frametime

    if member_pos is None or K == 0:
        z_pos = np.zeros((K + 1, T), dtype=float)
        if K > 0:
            np.cumsum(slen, axis=0, out=z_pos[1:])
        return z_pos, slen, time

    member_pos = np.asarray(member_pos, dtype=float)
    if member_pos.shape != (K, T, 2):
        raise ValueError(f'member_pos must have shape {(K, T, 2)}, got {member_pos.shape}.')
    anchor_arc, tangent, anchor_pos = _chain_arc_coordinates(member_pos, ref_idx)
    # Each member's centre along the fibre: its anchor plus the component of its
    # measured displacement along the local fibre direction. Both terms use the
    # same reference-frame anchor, so the chain coordinate matches the order.
    disp = member_pos - anchor_pos[:, None, :]
    centre_arc = anchor_arc[:, None] + np.einsum('ktc,kc->kt', disp, tangent)  # (K, T)
    z_pos = np.full((K + 1, T), np.nan, dtype=float)
    z_pos[:K] = centre_arc - 0.5 * slen          # leading edge of each member
    z_pos[K] = centre_arc[-1] + 0.5 * slen[-1]   # trailing edge of the last member
    # shift so the fibre starts at 0 (LOI convention; plots assume z_pos >= 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        origin = np.nanmin(z_pos)
    if np.isfinite(origin):
        z_pos -= origin
    return z_pos, slen, time
