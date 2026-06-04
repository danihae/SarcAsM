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

reusing the existing, battle-tested ContractionNet engine
(:mod:`sarcasm.analysis.contraction_analysis`) verbatim — those
functions are pure functions of a ``(n_groups, T)`` length matrix plus
``frametime``.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

from sarcasm.analysis import contraction_analysis


def aggregate_group_slen(
    tracks_slen: np.ndarray,
    group_id: np.ndarray,
    n_groups: int,
    aggregate: str = 'nanmedian',
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
    aggregate : {'nanmedian', 'nanmean'}
        Reduction used for the primary ``slen_timeseries`` (the signal fed to
        the contraction engine). The full distribution (median/std/q25/q75) is
        always returned for plotting regardless of this choice.
    slen_lims : tuple(float, float), optional
        If given, member lengths outside ``[lo, hi]`` (µm) are set to NaN before
        aggregation.

    Returns
    -------
    dict
        ``slen_timeseries`` (the chosen aggregate), ``slen_median_timeseries``,
        ``slen_std_timeseries``, ``slen_q25_timeseries``, ``slen_q75_timeseries``,
        and ``n_members_timeseries`` (int count of members with a finite length
        per frame), each shape ``(n_groups, T)``.
    """
    tracks_slen = np.asarray(tracks_slen, dtype=float)
    if tracks_slen.ndim != 2:
        tracks_slen = tracks_slen.reshape(len(group_id), -1)
    T = tracks_slen.shape[1]
    group_id = np.asarray(group_id).reshape(-1)

    agg_fn = np.nanmean if aggregate == 'nanmean' else np.nanmedian

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
        'slen_timeseries': slen_ts,
        'slen_median_timeseries': median_ts,
        'slen_std_timeseries': std_ts,
        'slen_q25_timeseries': q25_ts,
        'slen_q75_timeseries': q75_ts,
        'n_members_timeseries': n_members,
    }


def run_cycle_engine(
    group_slen: np.ndarray,
    frametime: float,
    model_path: str,
    threshold: float = 0.3,
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
    the prefix to the grouping kind). ``group_label`` / ``id_offset`` only label
    the engine's per-group log messages. Returns empty ``(0, T)`` / ``(0,)``
    arrays when ``n_groups == 0``.
    """
    group_slen = np.asarray(group_slen, dtype=float)
    n_groups, T = group_slen.shape
    if n_groups == 0:
        return {
            'domain_contr': np.zeros((0, T), dtype=bool),
            'domain_n_contr': np.zeros(0, dtype=np.int32),
            'domain_labels_contr': np.zeros((0, T), dtype=np.int32),
            'domain_beating_rate': np.zeros(0),
            'domain_beating_rate_variability': np.zeros(0),
            'domain_equ': np.zeros(0),
            'domain_contr_max': np.zeros((0, 1)),
            'domain_elong_max': np.zeros((0, 1)),
            'domain_vel_contr_max': np.zeros((0, 1)),
            'domain_vel_elong_max': np.zeros((0, 1)),
            'domain_time_to_peak': np.zeros((0, 1)),
            'domain_time_to_relax': np.zeros((0, 1)),
            'domain_time_contr': np.zeros((0, 1)),
        }

    contr = contraction_analysis.detect_contractions(
        domain_slen_timeseries=group_slen,
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
        domain_slen_timeseries=group_slen,
        domain_labels_contr=contr['domain_labels_contr'],
        domain_n_contr=contr['domain_n_contr'],
        frametime=frametime,
        filter_params=filter_params,
    )
    return {**contr, **params}


def _interp_nan_1d(a: np.ndarray) -> np.ndarray:
    """Linear-interpolate interior NaNs of a 1D array; edges held constant.
    An all-NaN input returns zeros (a member with no usable data)."""
    a = np.asarray(a, dtype=float).copy()
    mask = np.isnan(a)
    if mask.all():
        return np.zeros_like(a)
    if mask.any():
        idx = np.arange(a.shape[0])
        a[mask] = np.interp(idx[mask], idx[~mask], a[~mask])
    return a


def synthesize_loi_chain(member_slen: np.ndarray, frametime: float):
    """Turn an ordered chain of K member sarcomere-length series into an LOI-style
    ``(z_pos, slen, time)`` triple so the :mod:`sarcasm.motion` LOI engine runs
    unmodified on a myofibril built from tracks.

    Parameters
    ----------
    member_slen : np.ndarray
        ``(K, T)`` member sarcomere lengths ordered head-to-tail along the fibre,
        NaN on gap frames.
    frametime : float
        Seconds per frame.

    Returns
    -------
    z_pos : np.ndarray
        ``(K+1, T)`` cumulative arc-length of the K+1 Z-band boundaries (µm). NaNs
        are interpolated per sarcomere across time *before* the cumulative sum so a
        single bad member cannot poison the tail. By construction
        ``np.diff(z_pos, axis=0) == slen``.
    slen : np.ndarray
        ``(K, T)`` NaN-filled member lengths (the diff of ``z_pos``).
    time : np.ndarray
        ``(T,)`` time axis in seconds.
    """
    member_slen = np.asarray(member_slen, dtype=float)
    if member_slen.ndim != 2:
        raise ValueError('member_slen must be 2D (K, T).')
    K, T = member_slen.shape
    slen = np.empty((K, T), dtype=float)
    for k in range(K):
        slen[k] = _interp_nan_1d(member_slen[k])
    z_pos = np.zeros((K + 1, T), dtype=float)
    if K > 0:
        np.cumsum(slen, axis=0, out=z_pos[1:])
    time = np.arange(T) * frametime
    return z_pos, slen, time
