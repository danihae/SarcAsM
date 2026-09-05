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

Two analyses over a member matrix ``(N, T)`` of per-sarcomere traces, shared by
:meth:`sarcasm.SarcAsM.analyze_track_motion` (per group, every grouping kind) and
by :class:`sarcasm.Motion` for a fibre chain:

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
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from pywt import cwt
from scipy.signal import savgol_filter

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

def _interp_interior_nans(row: np.ndarray) -> np.ndarray:
    row = row.astype(float, copy=True)
    finite = np.isfinite(row)
    if finite.sum() < 2:
        return row
    first, last = np.flatnonzero(finite)[[0, -1]]
    inner = np.arange(first, last + 1)
    gaps = inner[~finite[first:last + 1]]
    if gaps.size:
        row[gaps] = np.interp(gaps, np.flatnonzero(finite), row[finite])
    return row


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
    quiet = ~contr
    N, T = slen.shape
    equ = np.full(N, np.nan)
    vel = np.full((N, T), np.nan)
    window, poly = filter_params
    if window % 2 == 0:
        window += 1
    for i in range(N):
        vals = slen[i, quiet] if np.any(np.isfinite(slen[i, quiet])) else slen[i]
        if np.any(np.isfinite(vals)):
            equ[i] = np.nanmedian(vals)
        finite = np.isfinite(slen[i])
        if finite.sum() < 2:
            continue
        filled = _interp_interior_nans(slen[i])
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
    members are distinct sarcomeres (a fibre chain, a domain, the pool): the
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
                           sampling_period=frametime, axis=-1)
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
GROUP_KEYS = ('corr_delta_slen_serial', 'corr_delta_slen_mutual', 'ratio_delta_slen_mutual_serial',
              'corr_vel_serial', 'corr_vel_mutual', 'ratio_vel_mutual_serial', 'corr_n_cycles',
              'oscill_frequencies', 'oscill_magnitudes_avg', 'oscill_magnitudes_single',
              'oscill_peak_avg', 'oscill_amp_avg', 'oscill_peak_1_single', 'oscill_amp_1_single',
              'oscill_peak_2_single', 'oscill_amp_2_single')


def analyze_groups(tracks_slen: np.ndarray, group_id: np.ndarray, n_groups: int,
                   contr: np.ndarray, labels_contr: np.ndarray, beating_rate: np.ndarray,
                   frametime: float, filter_params: Tuple[int, int] = (13, 5),
                   slen_lims: Tuple[float, float] = (1.2, 3.0), num_scales: int = 60,
                   **oscillation_kwargs) -> Dict[str, np.ndarray]:
    """Serial/mutual correlation and oscillation spectrum for every track group.

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
    **oscillation_kwargs
        Forwarded to :func:`oscillation_spectrum`.

    Returns
    -------
    dict
        One entry per name in :data:`GROUP_KEYS`: scalars as ``(n_groups,)``
        arrays, spectra as ``(n_groups, num_scales)``; ``oscill_frequencies`` is
        shared, ``(num_scales,)``.
    """
    tracks_slen = np.asarray(tracks_slen, dtype=float)
    group_id = np.asarray(group_id).reshape(-1)
    scalars = {k: np.full(n_groups, np.nan) for k in GROUP_KEYS
               if not k.startswith('oscill_magnitudes') and k != 'oscill_frequencies'}
    mag_avg = np.full((n_groups, num_scales), np.nan)
    mag_single = np.full((n_groups, num_scales), np.nan)
    frequencies = None
    for g in range(n_groups):
        members = np.flatnonzero(group_id == g)
        if members.size == 0:
            continue
        contr_g = np.asarray(contr[g], dtype=bool)
        kin = member_kinematics(tracks_slen[members], contr_g, frametime, filter_params, slen_lims)
        onsets, L = cycle_windows(labels_contr[g])
        for name in ('delta_slen', 'vel'):
            r = serial_mutual_correlation(kin[name], onsets, L)
            scalars[f'corr_{name}_serial'][g] = r['serial']
            scalars[f'corr_{name}_mutual'][g] = r['mutual']
            scalars[f'ratio_{name}_mutual_serial'][g] = r['ratio_mutual_serial']
        scalars['corr_n_cycles'][g] = onsets.size
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
            'oscill_magnitudes_avg': mag_avg, 'oscill_magnitudes_single': mag_single}
