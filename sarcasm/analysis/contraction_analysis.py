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

"""Contraction-cycle analysis engine.

The grouping-agnostic core that turns an aggregated ``(n_groups, T)`` sarcomere-
length matrix into contraction cycles (via ContractionNet) and per-group
contraction parameters (beating rate, amplitude, velocity, time-to-peak, …).

It operates on already-aggregated per-group signals, NOT on individual tracks
(that aggregation is :mod:`sarcasm.analysis.grouped_motion`) and is not
domain-specific — it is the shared engine behind every grouping kind (pool,
m-band, myofibril, LOI, domain) via :func:`grouped_motion.run_cycle_engine`.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import label
from scipy.signal import savgol_filter

from contraction_net.decoding import decode_contractions, intervals_to_mask, recommended_decoding
from contraction_net.prediction import polarity_rule, predict_contractions

logger = logging.getLogger(__name__)



def _length_polarity(model_path: str) -> dict:
    """Keyword arguments of ``predict_contractions`` for a sarcomere length.

    A sarcomere shortens when it contracts, so its length rests at the long side
    (``polarity=1``). Only checkpoints that read after the polarity rule (``'q90'``) take a
    declared polarity; a ``'symmetric'`` one reads both signs itself and gets none."""
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    return {'polarity': 1} if state.get('input_convention', 'q90') == 'q90' else {}


def cycle_truncation_flags(labels: np.ndarray, n_cycles: int, buffer_frames: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cycle truncation flags for a labelled 1D contraction (or quiescence) mask.

    A cycle is *truncated at the start* when its first frame lies within
    ``buffer_frames`` of the beginning of the recording, and *truncated at the end*
    when its last frame lies within ``buffer_frames`` of the end — i.e. the onset
    (resp. offset) happened outside the recorded window, so any quantity that needs
    it is unobservable. A cycle truncated on neither side is **complete**.

    Truncated cycles are kept and only flagged, so they still appear in the mask
    (plots, quiet/equilibrium baseline, beating-rate onsets) while their
    duration-dependent metrics can be set to NaN.

    Parameters
    ----------
    labels : np.ndarray
        1D cycle-label array as returned by :func:`scipy.ndimage.label`
        (0 = background, ``1 .. n_cycles`` = cycles).
    n_cycles : int
        Number of labelled cycles.
    buffer_frames : int
        Frames from either end within which a cycle counts as truncated.

    Returns
    -------
    tuple of np.ndarray
        ``(trunc_start, trunc_end)``, boolean, each of shape ``(n_cycles,)``.
    """
    trunc_start = np.zeros(int(n_cycles), dtype=bool)
    trunc_end = np.zeros(int(n_cycles), dtype=bool)
    if n_cycles == 0:
        return trunc_start, trunc_end
    n_frames = labels.shape[0]
    for i in range(1, int(n_cycles) + 1):
        idx = np.flatnonzero(labels == i)
        if idx.size == 0:
            continue
        trunc_start[i - 1] = idx[0] <= buffer_frames
        trunc_end[i - 1] = idx[-1] >= n_frames - 1 - buffer_frames
    return trunc_start, trunc_end


def detect_contractions(
    group_slen: np.ndarray,
    frametime: float,
    model_path: str,
    threshold: Optional[float] = None,
    contr_time_min: float = 0.2,
    buffer_frames: int = 3,
    min_valid_frames: float = 0.5,
    group_label: str = "Group",
    id_offset: int = 0,
    group_prob: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Detect contraction cycles from per-group sarcomere length time-series via ContractionNet.

    ContractionNet reads each row's mean sarcomere length with rest declared at the long
    side, since a sarcomere shortens when it contracts (``polarity=1`` in
    :func:`contraction_net.prediction.predict_contractions`; a ``'symmetric'`` checkpoint
    reads both signs itself and takes no declaration), and
    :func:`contraction_net.decoding.decode_contractions` turns the probability into
    contraction cycles: runs of frames above the threshold that reach the model's confidence
    peak and last at least ``contr_time_min``. Gaps between cycles are not closed, so
    overlapping twitches stay separate cycles. A row whose length dwells at its short side
    with brief lengthenings, as in a cell that is contracted most of the time (or, by chance,
    in a noisy trace), is still read with rest at the long side; its polarity score is
    negative, and the rows of this kind with contractions are named in a warning.
    ``group_label`` and ``id_offset`` affect only log messages: each row ``i`` is named
    ``f"{group_label} {i + id_offset}"`` so the shared engine matches the caller's numbering
    (0-based group id for most kinds; 1-based mask label for domains, where group id = mask
    label - 1).

    Parameters
    ----------
    group_slen : np.ndarray
        Per-group mean sarcomere length time-series, shape ``(n_groups, n_frames)`` (µm).
    frametime : float
        Time between frames in s.
    model_path : str
        Path to the ContractionNet model weights (.pt file).
    threshold : float or None, optional
        Probability a frame must exceed to be contracting. None (the default) uses the
        operating point the model was tuned for, read from its checkpoint. The right value
        is a property of the model; a threshold carried over from a different one is not
        meaningful. The confidence peak is always the model's own
        (:func:`contraction_net.decoding.recommended_decoding`).
    contr_time_min : float, optional
        Minimal contraction duration in s, converted to frames with ``round()``; shorter
        contractions are removed. Default is 0.2.
    buffer_frames : int, optional
        Frames from either end within which a contraction cycle counts as
        **incomplete**: its onset (or offset) lies outside the recorded window.
        Such cycles are *kept* in the mask — so they are plotted, excluded from the
        quiet/equilibrium baseline and contribute their onset to the beating rate —
        but are flagged via ``group_contr_complete`` so that
        :func:`analyze_contraction_parameters` can NaN the metrics they cannot
        support. Default is 3.
    min_valid_frames : float, optional
        Minimum fraction of valid (non-NaN) frames required to analyze a group. Default is 0.5.
    group_label : str, optional
        Label used to name rows in log messages. Default is "Group".
    id_offset : int, optional
        Offset added to the row index when naming rows in log messages. Default is 0.
    group_prob : dict or None, optional
        Precomputed contraction probabilities,
        :func:`sarcasm.analysis.grouped_motion.track_contraction_prob` (``prob``,
        ``polarity_score``, ``too_short``). If given, each row is decoded from ``prob`` (frames
        without a value count as quiet) and ``group_slen`` is not read by the network; a row
        needs ``min_valid_frames`` of frames with a value. Default is None.

    Returns
    -------
    dict
        Per-group contraction detection results (``max_n_contr`` is the max cycle
        count across groups):

        - 'contr' : np.ndarray ``(n_groups, n_frames)``, binary contraction state
        - 'n_contr' : np.ndarray ``(n_groups,)``, number of contraction cycles
          detected per group, **including** incomplete ones at the recording edges
        - 'n_contr_complete' : np.ndarray ``(n_groups,)``, number of complete cycles
        - 'contr_complete' : np.ndarray ``(n_groups, max_n_contr)``, 1.0 for a
          complete cycle, 0.0 for an incomplete one, NaN padding. Its per-group mean is
          the fraction of complete cycles.
        - 'labels_contr' : np.ndarray ``(n_groups, n_frames)``, contraction cycle labels
        - 'beating_rate' : np.ndarray ``(n_groups,)``, beating rate (Hz)
        - 'beating_rate_variability' : np.ndarray ``(n_groups,)``, std of inter-beat interval (s)
        - 'polarity_score' : np.ndarray ``(n_groups,)``, polarity score of the group's sarcomere
          length (:func:`contraction_net.polarity.polarity_score`), positive when it rests at its
          long side; below -0.1, the rule's margin, it dwells at its short side. NaN where the row
          was not read (too few valid frames, a flat trace, or a recording too short for the rule)
    """
    decoding = recommended_decoding(model_path)
    if threshold is not None:
        decoding['threshold'] = float(threshold)

    n_groups, n_frames = group_slen.shape
    
    # Initialize output arrays
    group_contr = np.zeros((n_groups, n_frames), dtype=bool)
    group_n_contr = np.zeros(n_groups, dtype=np.int32)
    group_labels_contr = np.zeros((n_groups, n_frames), dtype=np.int32)
    group_n_contr_complete = np.zeros(n_groups, dtype=np.int32)
    group_beating_rate = np.full(n_groups, np.nan)
    group_beating_rate_var = np.full(n_groups, np.nan)
    group_polarity_score = np.full(n_groups, np.nan)
    complete_per_group: List[Optional[np.ndarray]] = [None] * n_groups

    declared = _length_polarity(model_path)
    margin = polarity_rule(model_path).margin if group_prob is not None else None
    few_cycles = []                  # (group id, n cycles) of groups without a beating rate
    rests_short = []                 # ids of groups with contractions whose length dwells at its short side
    too_short = bool(group_prob['too_short']) if group_prob is not None else False

    # Process each group
    for group_idx in range(n_groups):
        if group_prob is not None:
            # per-track median probability: decoded as it is, frames without a track count as quiet
            row = np.asarray(group_prob['prob'][group_idx], dtype=float)
            valid_fraction = np.sum(~np.isnan(row)) / n_frames
            if valid_fraction < min_valid_frames:
                logger.debug(f"{group_label} {group_idx + id_offset} has too few frames with a readable "
                             f"track ({valid_fraction:.1%}), skipping.")
                continue
            prob = np.nan_to_num(row, nan=0.0)
            score = float(group_prob['polarity_score'][group_idx])
            group_polarity_score[group_idx] = score
            disagree = score < -margin
        else:
            slen_timeseries = group_slen[group_idx]

            # Check if the group has enough valid data
            valid_fraction = np.sum(~np.isnan(slen_timeseries)) / n_frames
            if valid_fraction < min_valid_frames:
                logger.debug(f"{group_label} {group_idx + id_offset} has insufficient valid data ({valid_fraction:.1%}), skipping.")
                continue

            # Interpolate NaN values for prediction
            slen_interp = _interp_nans(slen_timeseries, edges=True)

            if np.all(np.isnan(slen_interp)):
                continue

            # Predict contractions using ContractionNet; a sarcomere shortens when it contracts, so its
            # length rests at the long side
            try:
                contr_pred, decision = predict_contractions(slen_interp, model_path, frametime=frametime,
                                                            return_decision=True, **declared)
            except Exception as e:
                logger.warning(f"ContractionNet prediction failed for {group_label.lower()} {group_idx + id_offset}: {e}")
                continue
            group_polarity_score[group_idx] = decision.score
            too_short = too_short or decision.reason == 'too_short'
            prob = contr_pred[0]
            disagree = decision.reason == 'declared, data disagree'

        # cycles at the recording edges are kept and flagged incomplete below
        contr = intervals_to_mask(decode_contractions(prob, frametime, min_duration_s=contr_time_min,
                                                      border_frames=0, **decoding), n_frames)
        group_contr[group_idx] = contr
        if disagree and contr.any():
            rests_short.append(group_idx + id_offset)

        # Label contraction cycles and flag the incomplete ones
        labels, n_contr = label(contr)
        group_labels_contr[group_idx] = labels
        group_n_contr[group_idx] = n_contr
        trunc_start, trunc_end = cycle_truncation_flags(labels, n_contr, buffer_frames)
        complete = ~(trunc_start | trunc_end)
        complete_per_group[group_idx] = complete
        group_n_contr_complete[group_idx] = int(complete.sum())

        # Calculate beating rate
        if n_contr > 1:
            start_frames = np.where(np.diff(contr.astype('float32')) > 0.5)[0]
            if len(start_frames) > 1:
                inter_beat_intervals = np.diff(start_frames) * frametime
                group_beating_rate[group_idx] = 1 / np.mean(inter_beat_intervals)
                group_beating_rate_var[group_idx] = np.std(inter_beat_intervals)
        else:
            few_cycles.append((group_idx + id_offset, n_contr))

    # one line for all groups, not one per group (a movie has hundreds of myofibrils)
    if len(few_cycles) == 1:
        gid, n = few_cycles[0]
        logger.warning(f"{group_label} {gid}: only {n} contraction cycle(s) detected, so no beating rate "
                       f"(it needs at least 2 cycles).")
    elif few_cycles:
        ids = ', '.join(str(g) for g, _ in few_cycles[:10]) + (', …' if len(few_cycles) > 10 else '')
        logger.warning(f"No beating rate for {len(few_cycles)} of {n_groups} groups ({group_label.lower()} "
                       f"{ids}): fewer than 2 contraction cycles detected.")
    if too_short:
        rule = polarity_rule(model_path)
        logger.warning(f"The recording is too short to detect contractions ({n_frames} frames, "
                       f"{n_frames * frametime:.2f} s): ContractionNet needs at least {rule.min_frames} frames "
                       f"and {rule.min_seconds:g} s.")
    if rests_short:
        ids = ', '.join(str(g) for g in rests_short[:10]) + (', …' if len(rests_short) > 10 else '')
        logger.warning(f"{len(rests_short)} of {n_groups} groups ({group_label.lower()} {ids}): the sarcomere "
                       f"length dwells at its short side, as when it is contracted most of the time or noisy. The "
                       f"contractions are read with rest at the long side, as for every sarcomere length; check "
                       f"these groups.")
    
    # Per-cycle completeness, NaN-padded to the widest group (like every other
    # per-cycle array). Float, so its per-group nanmean is the complete fraction.
    max_n_contr = int(group_n_contr.max()) if group_n_contr.size and group_n_contr.max() > 0 else 1
    group_contr_complete = np.full((n_groups, max_n_contr), np.nan)
    for group_idx, complete in enumerate(complete_per_group):
        if complete is not None and complete.size:
            group_contr_complete[group_idx, :complete.size] = complete.astype(float)

    n_incomplete = int(group_n_contr.sum() - group_n_contr_complete.sum())
    if n_incomplete:
        logger.info(
            f"{n_incomplete}/{int(group_n_contr.sum())} contraction cycles are incomplete "
            f"(within {buffer_frames} frames of the recording start/end); they are kept in the "
            f"contraction mask but their duration-dependent metrics are NaN.")

    return {
        'contr': group_contr,
        'n_contr': group_n_contr,
        'n_contr_complete': group_n_contr_complete,
        'contr_complete': group_contr_complete,
        'labels_contr': group_labels_contr,
        'beating_rate': group_beating_rate,
        'beating_rate_variability': group_beating_rate_var,
        'polarity_score': group_polarity_score,
    }


def equilibrium_over_quiet(slen: np.ndarray, contr: np.ndarray) -> np.ndarray:
    """Equilibrium (resting) sarcomere length over the non-contracting frames.

    The equilibrium is the median sarcomere length wherever the contraction
    state is ``0`` (the trace is *not* contracting), matching the
    :meth:`sarcasm.motion.Motion.get_trajectories` semantics and the value stored
    as ``equ`` by :func:`analyze_contraction_parameters`. Traces with no quiet
    frame fall back to the median over all frames.

    Parameters
    ----------
    slen : np.ndarray
        Sarcomere length trace(s). A 1D ``(T,)`` array returns a scalar; a 2D
        ``(k, T)`` stack returns one value per row, shape ``(k,)``.
    contr : np.ndarray
        Boolean contraction state per frame, shape ``(T,)``; ``True`` marks a
        contracting frame, which is excluded from the equilibrium.

    Returns
    -------
    float or np.ndarray
        Equilibrium length: a scalar for a 1D input, or shape ``(k,)`` for a 2D
        input. NaN where no finite length is available.
    """
    slen = np.asarray(slen, dtype=float)
    quiet = ~np.asarray(contr, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if slen.ndim == 1:
            vals = slen[quiet]
            if not np.any(np.isfinite(vals)):
                vals = slen
            return float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan
        equ = np.full(slen.shape[0], np.nan)
        for i in range(slen.shape[0]):
            vals = slen[i, quiet]
            if not np.any(np.isfinite(vals)):
                vals = slen[i]
            if np.any(np.isfinite(vals)):
                equ[i] = np.nanmedian(vals)
        return equ


def analyze_contraction_parameters(
    group_slen: np.ndarray,
    group_labels_contr: np.ndarray,
    group_n_contr: np.ndarray,
    frametime: float,
    filter_params: Tuple[int, int] = (13, 5),
    buffer_frames: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Analyze per-cycle contraction parameters for per-group sarcomere length trajectories.

    Computes per-group, per-contraction-cycle parameters: maximum contraction and
    elongation, peak velocities, and timing parameters.

    Parameters
    ----------
    group_slen : np.ndarray
        Per-group mean sarcomere length time-series, shape ``(n_groups, n_frames)`` (µm).
    group_labels_contr : np.ndarray
        Per-group contraction cycle labels, shape ``(n_groups, n_frames)``.
    group_n_contr : np.ndarray
        Number of contractions per group, shape ``(n_groups,)``.
    frametime : float
        Time between frames in s.
    filter_params : tuple of int, optional
        Savitzky-Golay filter parameters ``(window_length, polyorder)`` for velocity
        smoothing. Default is (13, 5).
    buffer_frames : int, optional
        Frames from either end within which a cycle counts as incomplete (must match
        the value passed to :func:`detect_contractions`). Metrics an incomplete cycle
        cannot support are set to NaN rather than computed from a truncated window:
        ``time_contr`` whenever either edge is missing; ``time_to_peak`` and the
        shortening-phase percentile times when the onset is missing; ``time_to_relax``
        when the offset is missing; and an amplitude/velocity extremum whenever it falls
        *on* the truncated boundary, because the true extremum then lies outside the
        recording. Default is 3.

    Returns
    -------
    dict
        Per-group contraction parameters (``max_n_contr`` is the max cycle count
        across groups). Entries for cycles that cannot support a given metric are NaN:

        - 'equ' : np.ndarray ``(n_groups,)``, equilibrium/resting sarcomere length (µm),
          the median over the non-contracting frames
        - 'contr_max' : np.ndarray ``(n_groups, max_n_contr)``, max contraction per cycle (µm)
        - 'elong_max' : np.ndarray ``(n_groups, max_n_contr)``, max elongation per cycle (µm)
        - 'vel_contr_max' : np.ndarray ``(n_groups, max_n_contr)``, max shortening velocity (µm/s)
        - 'vel_elong_max' : np.ndarray ``(n_groups, max_n_contr)``, max elongation velocity (µm/s)
        - 'time_to_peak' : np.ndarray ``(n_groups, max_n_contr)``, time to maximal contraction (s)
        - 'time_to_relax' : np.ndarray ``(n_groups, max_n_contr)``, time from peak to relaxation (s)
        - 'time_contr' : np.ndarray ``(n_groups, max_n_contr)``, contraction duration (s)
        - 'time_to_50' : np.ndarray ``(n_groups, max_n_contr)``, time from the onset of the cycle to 50 %
          of the shortening (s)
        - 'time_contr_10_90' : np.ndarray ``(n_groups, max_n_contr)``, time from 10 % to 90 % of the
          shortening (s)
        - 'time_relax_50' : np.ndarray ``(n_groups, max_n_contr)``, time from the peak to 50 % relaxation (s)
        - 'time_relax_90' : np.ndarray ``(n_groups, max_n_contr)``, time from the peak to 90 % relaxation (s)

        The percentile times are measured at the first crossing of the fraction of the cycle's peak
        shortening ``contr_max`` (relative to ``equ``), interpolated linearly between frames. Relaxation is
        followed past the end of the contraction window up to the onset of the next cycle; a cycle without
        shortening, or whose trace never reaches the level, gives NaN.
    """
    n_groups = group_slen.shape[0]
    max_n_contr = int(np.max(group_n_contr)) if np.max(group_n_contr) > 0 else 1
    
    # Initialize output arrays
    group_equ = np.full(n_groups, np.nan)
    group_contr_max = np.full((n_groups, max_n_contr), np.nan)
    group_elong_max = np.full((n_groups, max_n_contr), np.nan)
    group_vel_contr_max = np.full((n_groups, max_n_contr), np.nan)
    group_vel_elong_max = np.full((n_groups, max_n_contr), np.nan)
    group_time_to_peak = np.full((n_groups, max_n_contr), np.nan)
    group_time_to_relax = np.full((n_groups, max_n_contr), np.nan)
    group_time_contr = np.full((n_groups, max_n_contr), np.nan)
    group_time_to_50, group_time_contr_10_90, group_time_relax_50, group_time_relax_90 = (
        np.full((n_groups, max_n_contr), np.nan) for _ in range(4))
    
    window_length, polyorder = filter_params
    
    for group_idx in range(n_groups):
        slen = group_slen[group_idx]
        labels = group_labels_contr[group_idx]
        n_contr = group_n_contr[group_idx]
        
        if n_contr == 0 or np.all(np.isnan(slen)):
            continue
        
        # Equilibrium = median over the non-contracting frames, so it is not pulled
        # down by the contracted frames (the plots use the same definition).
        group_equ[group_idx] = equilibrium_over_quiet(slen, labels != 0)
        if not np.isfinite(group_equ[group_idx]):
            continue
        
        # Calculate velocity using Savitzky-Golay filter
        slen_interp = _interp_nans(slen, edges=True)
        if len(slen_interp) >= window_length:
            vel = savgol_filter(slen_interp, window_length, polyorder, deriv=1, delta=frametime)
        else:
            vel = np.gradient(slen_interp, frametime)
        
        # Calculate delta (change from equilibrium)
        delta_slen = slen_interp - group_equ[group_idx]
        
        # Cycles touching the recording edges are incomplete: their onset/offset
        # happened outside the window, so the metrics that need it stay NaN.
        trunc_start, trunc_end = cycle_truncation_flags(labels, n_contr, buffer_frames)

        # Analyze each contraction cycle
        for contr_idx in range(n_contr):
            cycle_mask = labels == (contr_idx + 1)
            if not np.any(cycle_mask):
                continue

            delta_cycle = delta_slen[cycle_mask]
            vel_cycle = vel[cycle_mask]
            cut_start = bool(trunc_start[contr_idx])
            cut_end = bool(trunc_end[contr_idx])

            def _extremum_observed(arg_idx: int, n: int) -> bool:
                """False when the extremum sits on a truncated edge of the cycle, i.e.
                the trace never turned around inside the recording."""
                return not ((cut_start and arg_idx == 0) or (cut_end and arg_idx == n - 1))

            # Contraction duration — needs both the onset and the offset
            if not (cut_start or cut_end):
                group_time_contr[group_idx, contr_idx] = np.sum(cycle_mask) * frametime

            # Max contraction (most negative delta) and elongation (most positive delta)
            if len(delta_cycle) > 0 and not np.all(np.isnan(delta_cycle)):
                n_cyc = len(delta_cycle)
                peak_idx = int(np.nanargmin(delta_cycle))
                elong_idx = int(np.nanargmax(delta_cycle))
                peak_observed = _extremum_observed(peak_idx, n_cyc)
                if peak_observed:
                    group_contr_max[group_idx, contr_idx] = np.nanmin(delta_cycle)
                if _extremum_observed(elong_idx, n_cyc):
                    group_elong_max[group_idx, contr_idx] = np.nanmax(delta_cycle)

                # Max velocities — tested against their own arg-extremum, which need
                # not coincide with the length extremum.
                if not np.all(np.isnan(vel_cycle)):
                    if _extremum_observed(int(np.nanargmin(vel_cycle)), n_cyc):
                        group_vel_contr_max[group_idx, contr_idx] = np.nanmin(vel_cycle)
                    if _extremum_observed(int(np.nanargmax(vel_cycle)), n_cyc):
                        group_vel_elong_max[group_idx, contr_idx] = np.nanmax(vel_cycle)

                # Timing relative to the peak: onset->peak needs the onset,
                # peak->offset needs the offset, and both need an observed peak.
                if peak_observed:
                    if not cut_start:
                        group_time_to_peak[group_idx, contr_idx] = peak_idx * frametime
                    if not cut_end:
                        group_time_to_relax[group_idx, contr_idx] = (n_cyc - peak_idx) * frametime

                    # percentile times: first crossings of fractions of the peak shortening
                    amp = delta_cycle[peak_idx]
                    if amp < 0:
                        if not cut_start:
                            rise = delta_cycle[:peak_idx + 1]
                            t10, t50, t90 = (_first_crossing(rise, p * amp, falling=True) for p in (0.1, 0.5, 0.9))
                            group_time_to_50[group_idx, contr_idx] = t50 * frametime
                            group_time_contr_10_90[group_idx, contr_idx] = (t90 - t10) * frametime
                        # relaxation runs on past the end of the window, up to the onset of the next cycle
                        frames_cycle = np.flatnonzero(cycle_mask)
                        later = np.flatnonzero((labels > 0) & (labels != contr_idx + 1))
                        later = later[later > frames_cycle[-1]]
                        stop = int(later[0]) if later.size else len(delta_slen)
                        decay = delta_slen[frames_cycle[peak_idx]:stop]
                        group_time_relax_50[group_idx, contr_idx] = _first_crossing(
                            decay, 0.5 * amp, falling=False) * frametime
                        group_time_relax_90[group_idx, contr_idx] = _first_crossing(
                            decay, 0.1 * amp, falling=False) * frametime
    
    return {
        'equ': group_equ,
        'contr_max': group_contr_max,
        'elong_max': group_elong_max,
        'vel_contr_max': group_vel_contr_max,
        'vel_elong_max': group_vel_elong_max,
        'time_to_peak': group_time_to_peak,
        'time_to_relax': group_time_to_relax,
        'time_contr': group_time_contr,
        'time_to_50': group_time_to_50,
        'time_contr_10_90': group_time_contr_10_90,
        'time_relax_50': group_time_relax_50,
        'time_relax_90': group_time_relax_90,
    }


def _first_crossing(x: np.ndarray, level: float, falling: bool) -> float:
    """
    Fractional index of the first crossing of ``level`` in ``x``, interpolated linearly between samples.

    Parameters
    ----------
    x : np.ndarray
        1D trace.
    level : float
        Level to cross.
    falling : bool
        True for the first sample at or below ``level``, False for the first at or above it.

    Returns
    -------
    float
        Fractional sample index of the crossing; NaN if ``x`` never reaches ``level`` or is already past it at
        its first sample (the crossing happened before the segment).
    """
    hit = np.flatnonzero(x <= level) if falling else np.flatnonzero(x >= level)
    if hit.size == 0 or hit[0] == 0:
        return np.nan
    j = int(hit[0])
    if not np.isfinite(x[j - 1]):
        return float(j)
    return (j - 1) + (x[j - 1] - level) / (x[j - 1] - x[j])


def _interp_nans(a: np.ndarray, *, edges: bool = False, max_gap: Optional[int] = None) -> np.ndarray:
    """Linearly interpolate the NaNs of a 1D array.

    Interior NaNs (finite values on both sides) are filled. Runs longer than ``max_gap``
    frames are left NaN: interpolating a long dropout invents a straight line that can span a
    whole contraction (``None`` or ``<= 0``: no limit). Leading and trailing NaNs stay NaN, as
    a track must never carry a fabricated constant length where it has no observation, unless
    ``edges=True``, which holds the first and last finite value there (for network input,
    which takes no NaN). An all-NaN input is returned unchanged.
    """
    a = np.asarray(a, dtype=float).copy()
    mask = np.isnan(a)
    if not mask.any() or mask.all():
        return a
    idx = np.arange(a.shape[0])
    finite = ~mask
    first, last = idx[finite][0], idx[finite][-1]
    fill = mask & (idx > first) & (idx < last)
    if max_gap is not None and max_gap > 0 and fill.any():
        runs = np.flatnonzero(np.diff(np.concatenate(([0], fill.view(np.int8), [0]))))
        for start, stop in zip(runs[::2], runs[1::2]):
            if stop - start > max_gap:
                fill[start:stop] = False
    if edges:
        fill |= mask & ((idx < first) | (idx > last))
    if fill.any():
        a[fill] = np.interp(idx[fill], idx[finite], a[finite])
    return a
