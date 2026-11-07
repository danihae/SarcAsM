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

"""Heterogeneity of sarcomere dynamics within a group of tracked sarcomeres.

Analyses over a member matrix ``(N, T)`` of per-sarcomere traces, run by
:meth:`sarcasm.motion.MotionAnalysis.analyze` for every group of every grouping kind:

* **Serial / mutual correlation** (Haertter et al., *bioRxiv* 2024, eq. 1).
  ``r(i, j, k, l)`` is the Pearson correlation of sarcomere *i* in contraction
  cycle *k* with sarcomere *j* in cycle *l*, over the frames of one cycle.
  The *serial* correlation ``r_s`` averages ``r(i, i, k, l)`` over ``k != l``
  (same sarcomere, different cycles: cycle-to-cycle consistency); the *mutual*
  correlation ``r_m`` averages ``r(i, j, k, k)`` over ``i != j`` (different
  sarcomeres, same cycle: synchrony). ``R = r_m / r_s``: ``R < 1`` means the
  sarcomeres differ *consistently* (static heterogeneity), ``R ≈ 1`` that they
  differ *randomly* from beat to beat (stochastic heterogeneity).
* **Oscillation spectrum**: continuous-wavelet magnitude spectrum of the group's
  mean ΔSL and of the individual sarcomeres over the contracting frames, and
  the low-frequency (beating) and high-frequency (single-sarcomere) peaks.
* **Popping and beat start/end states**: which sarcomeres pop, which are still short
  when the group's cycle ends, start a cycle still lengthened, or lengthen first
  (:func:`popping_events`, :func:`window_states`).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from pywt import cwt
from scipy.signal import savgol_filter

from sarcasm.analysis.contraction_analysis import _interp_nans, equilibrium_over_quiet

#: Members used for the per-sarcomere oscillation spectra of large groups.
#: The wavelet transform is the only O(N) cost here; a seeded subset of this
#: size gives the same mean spectrum as a pool of thousands.
OSCILLATION_MAX_MEMBERS = 200


# ---------------------------------------------------------------------------
# cycle windows
# ---------------------------------------------------------------------------

def cycle_windows(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """Onset frame of every contraction cycle and the common window length.

    Parameters
    ----------
    labels : np.ndarray
        Contraction cycle labels per frame, shape ``(T,)``: ``0`` = quiet,
        ``1..K`` = cycle index.

    Returns
    -------
    onsets : np.ndarray
        Onset frame of each cycle whose window ``[onset, onset + cycle_len)``
        lies inside the recording, in cycle order.
    cycle_len : int
        Median duration (frames) of the cycles that touch neither recording edge;
        ``0`` when there is no such cycle.
    """
    labels = np.asarray(labels).reshape(-1)
    T = labels.size
    ids = np.unique(labels[labels > 0])
    if ids.size == 0:
        return np.zeros(0, dtype=int), 0
    onsets, durations, complete = [], [], []
    for k in ids:
        idx = np.flatnonzero(labels == k)
        onsets.append(int(idx[0]))
        durations.append(int(idx.size))
        complete.append(idx[0] > 0 and idx[-1] < T - 1)
    complete = np.asarray(complete)
    if not complete.any():
        return np.zeros(0, dtype=int), 0
    cycle_len = int(np.median(np.asarray(durations)[complete]))
    if cycle_len < 2:
        return np.zeros(0, dtype=int), 0
    onsets = np.asarray([o for o in onsets if o + cycle_len <= T], dtype=int)
    return onsets, cycle_len


# ---------------------------------------------------------------------------
# member kinematics
# ---------------------------------------------------------------------------

def member_kinematics(member_slen: np.ndarray, contr: np.ndarray, frametime: float,
                      filter_params: Tuple[int, int] = (13, 5),
                      slen_lims: Tuple[float, float] = (1.2, 3.0),
                      ) -> Dict[str, np.ndarray]:
    """Per-member ΔSL and velocity from raw sarcomere-length traces.

    Parameters
    ----------
    member_slen : np.ndarray
        Sarcomere length per member and frame, shape ``(N, T)`` (µm).
    contr : np.ndarray
        Contraction state of the group per frame, shape ``(T,)``; the equilibrium
        length of each member is its median over the non-contracting frames.
    frametime : float
        Frame interval in seconds.
    filter_params : (int, int), optional
        Savitzky-Golay ``(window_length, polyorder)`` used to smooth the length
        before differentiating. Default is (13, 5).
    slen_lims : (float, float), optional
        Lengths outside this range (µm) are treated as missing. Default is (1.2, 3.0).

    Returns
    -------
    dict
        ``'slen'`` ``(N, T)`` the lengths with out-of-range values set to NaN (µm),
        ``'delta_slen'`` ``(N, T)`` length change from equilibrium (µm),
        ``'vel'`` ``(N, T)`` velocity (µm/s), ``'equ'`` ``(N,)`` equilibrium lengths.
    """
    slen = np.asarray(member_slen, dtype=float).copy()
    slen[(slen < slen_lims[0]) | (slen > slen_lims[1])] = np.nan
    contr = np.asarray(contr, dtype=bool).reshape(-1)
    N, T = slen.shape
    equ = equilibrium_over_quiet(slen, contr)
    vel = np.full((N, T), np.nan)
    window, poly = filter_params
    if window % 2 == 0:
        window += 1
    for i in range(N):
        finite = np.isfinite(slen[i])
        if finite.sum() < 2:
            continue
        filled = _interp_nans(slen[i])
        first, last = np.flatnonzero(finite)[[0, -1]]
        seg = filled[first:last + 1]
        if seg.size >= window:
            seg = savgol_filter(seg, window, poly)
        vel[i, first:last + 1] = np.gradient(seg, frametime)
    delta = slen - equ[:, None]
    vel[~np.isfinite(slen)] = np.nan
    return {'slen': slen, 'delta_slen': delta, 'vel': vel, 'equ': equ}


# ---------------------------------------------------------------------------
# serial / mutual correlation
# ---------------------------------------------------------------------------

def _cycle_zscores(x: np.ndarray, onsets: np.ndarray, cycle_len: int) -> np.ndarray:
    """``Z[i, k, :]`` = z-scored window of member *i* in cycle *k*; NaN rows where
    the window is incomplete or constant."""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    K = onsets.size
    Z = np.full((N, K, cycle_len), np.nan)
    for k, o in enumerate(onsets):
        Z[:, k, :] = x[:, o:o + cycle_len]
    valid = np.all(np.isfinite(Z), axis=2)
    mean = Z.mean(axis=2, keepdims=True)
    std = Z.std(axis=2, keepdims=True)
    valid &= std[..., 0] > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        Z = (Z - mean) / std
    Z[~valid] = np.nan
    return Z



def serial_mutual_correlation(x: np.ndarray, onsets: np.ndarray, cycle_len: int) -> Dict[str, float]:
    """Serial and mutual Pearson correlation of per-member traces across cycles.

    Parameters
    ----------
    x : np.ndarray
        Per-member trace, shape ``(N, T)`` (ΔSL or velocity).
    onsets : np.ndarray
        Cycle onset frames, shape ``(K,)`` (see :func:`cycle_windows`).
    cycle_len : int
        Window length in frames; every cycle is compared over its first
        ``cycle_len`` frames.

    Returns
    -------
    dict
        ``'serial'`` (mean of ``r(i, i, k, l)``, ``k != l``), ``'mutual'`` (mean of
        ``r(i, j, k, k)``, ``i != j``), ``'ratio_mutual_serial'``, ``'n_members'``
        and ``'n_cycles'``. NaN where fewer than two members or cycles are usable.

    Notes
    -----
    The averages are computed from per-cycle and per-member sums of the z-scored
    windows, so the ``N² K²`` correlation matrix is never formed: for a cycle *k*
    with valid members *V*, ``sum_{i != j} r(i, j, k, k) = (|sum_i Z_ik|² - |V| L) / L``.

    ``ratio_mutual_serial`` is NaN when ``serial <= 0``. Read it on groups whose
    members are distinct sarcomeres (a chain along a line, a domain, the pool): the
    members of an M-band group are pixel-adjacent vectors of the same few
    sarcomeres, so their shared measurement inflates the mutual correlation.
    """
    onsets = np.asarray(onsets, dtype=int).reshape(-1)
    out = {'serial': np.nan, 'mutual': np.nan, 'ratio_mutual_serial': np.nan,
           'n_members': int(np.asarray(x).shape[0]), 'n_cycles': int(onsets.size)}
    if onsets.size == 0 or cycle_len < 2 or out['n_members'] == 0:
        return out
    Z = _cycle_zscores(x, onsets, cycle_len)
    valid = np.isfinite(Z[..., 0])                      # (N, K)
    L = float(cycle_len)
    Zf = np.where(valid[..., None], Z, 0.0)

    # mutual: different members, same cycle
    S = Zf.sum(axis=0)                                  # (K, L)
    n_k = valid.sum(axis=0).astype(float)               # (K,)
    num_m = ((S ** 2).sum(axis=1) - n_k * L) / L
    pairs_m = (n_k * (n_k - 1)).sum()
    if pairs_m > 0:
        out['mutual'] = float(num_m.sum() / pairs_m)

    # serial: same member, different cycles
    R = Zf.sum(axis=1)                                  # (N, L)
    k_i = valid.sum(axis=1).astype(float)               # (N,)
    num_s = ((R ** 2).sum(axis=1) - k_i * L) / L
    pairs_s = (k_i * (k_i - 1)).sum()
    if pairs_s > 0:
        out['serial'] = float(num_s.sum() / pairs_s)

    # R is only meaningful against a positive cycle-to-cycle consistency
    if np.isfinite(out['mutual']) and np.isfinite(out['serial']) and out['serial'] > 0:
        out['ratio_mutual_serial'] = float(out['mutual'] / out['serial'])
    return out


# ---------------------------------------------------------------------------
# oscillation spectrum
# ---------------------------------------------------------------------------

def wavelet_spectrum(data: np.ndarray, frametime: float, min_scale: float = 6,
                     max_scale: float = 180, num_scales: int = 60, wavelet: str = 'morl',
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous wavelet transform along the last axis.

    Returns ``(cfs, frequencies)`` with ``cfs`` of shape ``(num_scales, *data.shape)``
    and ``frequencies`` in Hz for logarithmically spaced scales.
    """
    scales = np.geomspace(min_scale, max_scale, num=num_scales)
    cfs, frequencies = cwt(np.asarray(data, dtype=float), scales, wavelet,
                           sampling_period=frametime, axis=-1, method='fft')
    return cfs, frequencies


def oscillation_spectrum(member_delta: np.ndarray, contr: np.ndarray, frametime: float,
                         beating_rate: float, min_scale: float = 6, max_scale: float = 180,
                         num_scales: int = 60, wavelet: str = 'morl', freq_thres: float = 2.0,
                         max_members: Optional[int] = OSCILLATION_MAX_MEMBERS,
                         random_seed: int = 0) -> Dict[str, object]:
    """Wavelet magnitude spectra of the mean and the individual ΔSL traces.

    Parameters
    ----------
    member_delta : np.ndarray
        ΔSL per member and frame, shape ``(N, T)`` (µm). NaN frames are treated
        as zero displacement.
    contr : np.ndarray
        Contraction state per frame, shape ``(T,)``; magnitudes are averaged over
        the contracting frames (over all frames when none is contracting).
    frametime : float
        Frame interval in seconds.
    beating_rate : float
        Beating rate (Hz); the low/high-frequency split is at least ``2.1 x`` it.
    min_scale, max_scale, num_scales, wavelet
        Wavelet scales (logarithmically spaced) and mother wavelet.
    freq_thres : float, optional
        Lower bound (Hz) of the frequency separating beating from high-frequency
        single-sarcomere oscillations. Default is 2.0.
    max_members : int or None, optional
        Cap on the members whose individual spectra are computed (a seeded random
        subset when exceeded); None uses all. Default is 200.
    random_seed : int, optional
        Seed for the member subset. Default is 0.

    Returns
    -------
    dict
        ``'frequencies'`` ``(num_scales,)`` Hz; ``'magnitudes_avg'`` spectrum of the
        mean trace; ``'magnitudes_single'`` mean spectrum over members and
        ``'magnitudes_single_std'``; ``'peak_avg'`` / ``'amp_avg'`` frequency and
        magnitude of the strongest component of the mean trace; ``'peak_1_single'``
        / ``'amp_1_single'`` the beating peak of the single-sarcomere spectrum and
        ``'peak_2_single'`` / ``'amp_2_single'`` its high-frequency peak — the
        strongest local maximum above the split (NaN when the spectrum only decays
        there); ``'freq_thres'`` the split used.
    """
    member_delta = np.asarray(member_delta, dtype=float)
    N, T = member_delta.shape
    contr = np.asarray(contr, dtype=bool).reshape(-1)
    mask = contr if contr.any() else np.ones(T, dtype=bool)
    filled = np.nan_to_num(member_delta, nan=0.0)
    avg = np.nan_to_num(np.nanmean(member_delta, axis=0) if N else np.zeros(T), nan=0.0)

    cfs_avg, frequencies = wavelet_spectrum(avg, frametime, min_scale, max_scale, num_scales, wavelet)
    mag_avg = np.abs(cfs_avg[:, mask]).mean(axis=1)

    idx = np.arange(N)
    if max_members is not None and N > max_members:
        idx = np.sort(np.random.default_rng(random_seed).choice(N, max_members, replace=False))
    mags = np.full((idx.size, num_scales), np.nan)
    chunk = 64
    for start in range(0, idx.size, chunk):
        sel = idx[start:start + chunk]
        cfs, _ = wavelet_spectrum(filled[sel], frametime, min_scale, max_scale, num_scales, wavelet)
        mags[start:start + sel.size] = np.abs(cfs[:, :, mask]).mean(axis=2).T
    mag_single = mags.mean(axis=0) if idx.size else np.full(num_scales, np.nan)
    mag_single_std = mags.std(axis=0) if idx.size else np.full(num_scales, np.nan)

    thres = freq_thres if not np.isfinite(beating_rate) else max(freq_thres, 2.1 * beating_rate)
    out = {'frequencies': frequencies, 'magnitudes_avg': mag_avg,
           'magnitudes_single': mag_single, 'magnitudes_single_std': mag_single_std,
           'freq_thres': float(thres),
           'peak_avg': float(frequencies[np.argmax(mag_avg)]), 'amp_avg': float(mag_avg.max()),
           'peak_1_single': np.nan, 'amp_1_single': np.nan,
           'peak_2_single': np.nan, 'amp_2_single': np.nan}
    if not np.all(np.isfinite(mag_single)):
        return out
    low = frequencies <= thres
    if low.any():
        j = np.flatnonzero(low)[np.argmax(mag_single[low])]
        out['peak_1_single'], out['amp_1_single'] = float(frequencies[j]), float(mag_single[j])
    # high-frequency peak: the strongest local maximum above the split, so the
    # beat's harmonics leaking across the boundary are not mistaken for one
    interior = np.zeros_like(low)
    interior[1:-1] = (mag_single[1:-1] > mag_single[:-2]) & (mag_single[1:-1] > mag_single[2:])
    cand = np.flatnonzero(~low & interior)
    if cand.size:
        j = cand[np.argmax(mag_single[cand])]
        out['peak_2_single'], out['amp_2_single'] = float(frequencies[j]), float(mag_single[j])
    return out


# ---------------------------------------------------------------------------
# per-group driver
# ---------------------------------------------------------------------------

#: Result keys written per group by :func:`analyze_groups`, in output order.
GROUP_KEYS = ('equ_std',
              'corr_delta_slen_serial', 'corr_delta_slen_mutual', 'ratio_delta_slen_mutual_serial',
              'corr_vel_serial', 'corr_vel_mutual', 'ratio_vel_mutual_serial', 'corr_n_cycles',
              'oscill_frequencies', 'oscill_magnitudes_avg', 'oscill_magnitudes_single',
              'oscill_peak_avg', 'oscill_amp_avg', 'oscill_peak_1_single', 'oscill_amp_1_single',
              'oscill_peak_2_single', 'oscill_amp_2_single',
              'popping_rate', 'popping_rate_contr', 'track_popping',
              'still_short_rate', 'starts_long_rate', 'lengthens_first_rate')


def popping_events(delta_slen: np.ndarray, labels_contr: np.ndarray, complete: np.ndarray,
                   thres_popping: float = 0.25) -> np.ndarray:
    """Which member sarcomere pops in which contraction cycle.

    A sarcomere pops in a cycle when it lengthens more than ``thres_popping`` beyond its
    resting length while the group contracts (its ΔSL exceeds the threshold during the cycle).
    Popping is physiological, not a tracking error.

    Parameters
    ----------
    delta_slen : np.ndarray
        Length change of each member from its resting length (µm), shape ``(n_members, T)``.
    labels_contr : np.ndarray
        Contraction cycle labels of the group, shape ``(T,)`` (0 = quiet, k = cycle k).
    complete : np.ndarray
        Whether each cycle lies fully inside the recording, shape ``(n_cycles,)``.
    thres_popping : float, optional
        Elongation (µm) beyond the resting length above which a sarcomere pops. Default is 0.25.

    Returns
    -------
    np.ndarray
        ``(n_members, n_cycles)``: 1.0 popped, 0.0 did not, NaN where the cycle is incomplete or
        the member has no length during it.
    """
    delta_slen = np.asarray(delta_slen, dtype=float)
    labels_contr = np.asarray(labels_contr).reshape(-1)
    n_cycles = len(complete)
    events = np.full((delta_slen.shape[0], n_cycles), np.nan)
    for k in range(n_cycles):
        if not complete[k]:
            continue
        in_cycle = labels_contr == k + 1
        with np.errstate(invalid='ignore'):
            elong_max = np.nanmax(np.where(in_cycle, delta_slen, np.nan), axis=1) \
                if in_cycle.any() else np.full(delta_slen.shape[0], np.nan)
        observed = np.isfinite(elong_max)
        events[observed, k] = (elong_max[observed] > thres_popping).astype(float)
    return events


def window_states(delta_slen: np.ndarray, labels_contr: np.ndarray, complete: np.ndarray,
                  thres_shift: float = 0.10, min_observed: float = 0.9) -> Dict[str, float]:
    """What the member sarcomeres do at the start and end of the group's contraction cycles.

    Three fractions over member sarcomeres and single-beat cycles:

    * ``still_short_rate``: at the last frame of the cycle the sarcomere is still at least
      ``thres_shift`` shorter than its resting length; it has not relaxed when the group has.
    * ``starts_long_rate``: at the first frame of the cycle it is at least ``thres_shift`` longer
      than its resting length, still lengthened from the previous beat.
    * ``lengthens_first_rate``: its first move of at least ``thres_shift`` away from its length at
      the first frame of the cycle is a lengthening and comes in the first half of the cycle, so a
      small shortening followed by a late pop does not count.

    Only cycles that hold one beat are scored: the group's mean ΔSL has a single dip (between the
    first and last frame within 10 % of its deepest point it does not rise back above half that
    depth), and the cycle lasts at most 1.5 x the median complete cycle. A sarcomere counts in a
    cycle when it has a length in at least ``min_observed`` of the cycle's frames; first and last
    frame mean its first and last measured one.

    Parameters
    ----------
    delta_slen : np.ndarray
        Length change of each member from its resting length (µm), shape ``(n_members, T)``.
    labels_contr : np.ndarray
        Contraction cycle labels of the group, shape ``(T,)`` (0 = quiet, k = cycle k).
    complete : np.ndarray
        Whether each cycle lies fully inside the recording, shape ``(n_cycles,)``.
    thres_shift : float, optional
        Length change (µm) from rest, or from the cycle's first frame, that counts. Default is 0.10.
    min_observed : float, optional
        Fraction of a cycle's frames a member must be measured in. Default is 0.9.

    Returns
    -------
    dict
        ``'still_short_rate'``, ``'starts_long_rate'``, ``'lengthens_first_rate'``; NaN when no
        single-beat cycle has a measured member.
    """
    delta_slen = np.asarray(delta_slen, dtype=float)
    labels_contr = np.asarray(labels_contr).reshape(-1)
    out = {'still_short_rate': np.nan, 'starts_long_rate': np.nan, 'lengthens_first_rate': np.nan}
    spans = []
    for k in range(len(complete)):
        idx = np.flatnonzero(labels_contr == k + 1)
        if complete[k] and idx.size >= 5:
            spans.append((int(idx[0]), int(idx[-1]) + 1))
    if not spans or delta_slen.shape[0] == 0:
        return out
    max_len = 1.5 * np.median([b - a for a, b in spans])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)      # frames without any member
        mean = np.nanmean(delta_slen, axis=0)
    short = starts_long = first_long = n = 0
    for a, b in spans:
        L = b - a
        if L > max_len:
            continue
        m = mean[a:b]
        if np.isfinite(m).sum() >= 5 and np.nanmin(m) < 0:
            depth = np.nanmin(m)
            near = np.flatnonzero(m <= 0.9 * depth)
            if np.nanmax(m[near[0]:near[-1] + 1]) > 0.5 * depth:
                continue                                      # two dips: more than one beat
        x = delta_slen[:, a:b]
        obs = np.isfinite(x)
        use = obs.sum(axis=1) >= min_observed * L
        if not use.any():
            continue
        x, obs = x[use], obs[use]
        rows = np.arange(x.shape[0])
        first = x[rows, obs.argmax(axis=1)]
        last = x[rows, L - 1 - obs[:, ::-1].argmax(axis=1)]
        with np.errstate(invalid='ignore'):
            moved = np.abs(x - first[:, None]) >= thres_shift
        j = moved.argmax(axis=1)
        lengthens = moved.any(axis=1) & (j / max(L - 1, 1) < 0.5) & (x[rows, j] > first)
        short += int(np.sum(last <= -thres_shift))
        starts_long += int(np.sum(first >= thres_shift))
        first_long += int(np.sum(lengthens))
        n += x.shape[0]
    if n:
        out = {'still_short_rate': short / n, 'starts_long_rate': starts_long / n,
               'lengthens_first_rate': first_long / n}
    return out


def analyze_groups(tracks_slen: np.ndarray, group_id: np.ndarray, n_groups: int,
                   contr: np.ndarray, labels_contr: np.ndarray, beating_rate: np.ndarray,
                   frametime: float, filter_params: Tuple[int, int] = (13, 5),
                   slen_lims: Tuple[float, float] = (1.2, 3.0), num_scales: int = 60,
                   contr_complete: Optional[np.ndarray] = None, thres_popping: float = 0.25,
                   thres_shift: float = 0.10, **oscillation_kwargs) -> Dict[str, np.ndarray]:
    """Serial/mutual correlation, oscillation spectrum, popping and beat start/end states for every track group.

    Parameters
    ----------
    tracks_slen : np.ndarray
        Sarcomere length per track and frame, shape ``(n_tracks, T)``.
    group_id : np.ndarray
        Group of each track, shape ``(n_tracks,)``; ``-1`` = unassigned.
    n_groups : int
        Number of groups.
    contr, labels_contr : np.ndarray
        Per-group contraction state / cycle labels, shape ``(n_groups, T)``.
    beating_rate : np.ndarray
        Per-group beating rate (Hz), shape ``(n_groups,)``.
    frametime : float
        Frame interval in seconds.
    filter_params, slen_lims
        Forwarded to :func:`member_kinematics`.
    num_scales : int, optional
        Number of wavelet scales. Default is 60.
    contr_complete : np.ndarray or None, optional
        Whether each cycle of each group lies fully inside the recording, shape
        ``(n_groups, max_n_contr)`` (1.0 / 0.0, NaN padding). None counts every cycle as complete.
        Default is None.
    thres_popping : float, optional
        See :func:`popping_events`. Default is 0.25.
    thres_shift : float, optional
        See :func:`window_states`. Default is 0.10.
    **oscillation_kwargs
        Forwarded to :func:`oscillation_spectrum`.

    Returns
    -------
    dict
        One entry per name in :data:`GROUP_KEYS`: scalars as ``(n_groups,)``
        arrays, spectra as ``(n_groups, num_scales)``; ``oscill_frequencies`` is
        shared, ``(num_scales,)``. ``equ_std`` is the standard deviation of the
        members' resting lengths (µm): how much the sarcomeres of a group differ
        in resting length, the static counterpart of the correlations.
        ``track_popping`` ``(n_tracks, max_n_contr)`` holds :func:`popping_events` of each track's
        group (NaN for unassigned tracks); ``popping_rate_contr`` ``(n_groups, max_n_contr)`` is the
        fraction of members popping in each cycle and ``popping_rate`` ``(n_groups,)`` over all cycles.
        ``still_short_rate``, ``starts_long_rate`` and ``lengthens_first_rate`` ``(n_groups,)`` are
        :func:`window_states` of each group.
    """
    tracks_slen = np.asarray(tracks_slen, dtype=float)
    group_id = np.asarray(group_id).reshape(-1)
    scalars = {k: np.full(n_groups, np.nan) for k in GROUP_KEYS
               if not k.startswith('oscill_magnitudes') and k not in ('oscill_frequencies', 'popping_rate_contr',
                                                                       'track_popping')}
    n_cycles_max = max(int(np.max(labels_contr)) if np.size(labels_contr) else 0, 1)
    track_popping = np.full((tracks_slen.shape[0], n_cycles_max), np.nan)
    popping_rate_contr = np.full((n_groups, n_cycles_max), np.nan)
    mag_avg = np.full((n_groups, num_scales), np.nan)
    mag_single = np.full((n_groups, num_scales), np.nan)
    frequencies = None
    for g in range(n_groups):
        members = np.flatnonzero(group_id == g)
        if members.size == 0:
            continue
        contr_g = np.asarray(contr[g], dtype=bool)
        kin = member_kinematics(tracks_slen[members], contr_g, frametime, filter_params, slen_lims)
        # static heterogeneity of length: spread of the members' resting lengths
        equ = kin['equ'][np.isfinite(kin['equ'])]
        scalars['equ_std'][g] = float(np.std(equ)) if equ.size > 1 else np.nan
        onsets, L = cycle_windows(labels_contr[g])
        for name in ('delta_slen', 'vel'):
            r = serial_mutual_correlation(kin[name], onsets, L)
            scalars[f'corr_{name}_serial'][g] = r['serial']
            scalars[f'corr_{name}_mutual'][g] = r['mutual']
            scalars[f'ratio_{name}_mutual_serial'][g] = r['ratio_mutual_serial']
        scalars['corr_n_cycles'][g] = onsets.size
        n_cycles = int(np.max(labels_contr[g])) if np.size(labels_contr[g]) else 0
        if n_cycles:
            complete = (np.ones(n_cycles, bool) if contr_complete is None
                        else np.asarray(contr_complete[g][:n_cycles]) == 1)
            events = popping_events(kin['delta_slen'], labels_contr[g], complete, thres_popping)
            track_popping[members, :n_cycles] = events
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)      # all-NaN cycles stay NaN
                popping_rate_contr[g, :n_cycles] = np.nanmean(events, axis=0)
                scalars['popping_rate'][g] = np.nanmean(events)
            for key, value in window_states(kin['delta_slen'], labels_contr[g], complete, thres_shift).items():
                scalars[key][g] = value
        osc = oscillation_spectrum(kin['delta_slen'], contr_g, frametime, float(beating_rate[g]),
                                   num_scales=num_scales, **oscillation_kwargs)
        frequencies = osc['frequencies']
        mag_avg[g] = osc['magnitudes_avg']
        mag_single[g] = osc['magnitudes_single']
        for key in ('peak_avg', 'amp_avg', 'peak_1_single', 'amp_1_single',
                    'peak_2_single', 'amp_2_single'):
            scalars[f'oscill_{key}'][g] = osc[key]
    if frequencies is None:
        frequencies = wavelet_spectrum(np.zeros(8), frametime, num_scales=num_scales)[1]
    return {**scalars, 'oscill_frequencies': frequencies,
            'oscill_magnitudes_avg': mag_avg, 'oscill_magnitudes_single': mag_single,
            'popping_rate_contr': popping_rate_contr, 'track_popping': track_popping}
