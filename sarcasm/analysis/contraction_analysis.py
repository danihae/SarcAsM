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
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label
from scipy.signal import savgol_filter

from contraction_net.prediction import predict_contractions, recommended_threshold

logger = logging.getLogger(__name__)


def cycle_truncation_flags(labels: np.ndarray, n_cycles: int, buffer_frames: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cycle truncation flags for a labelled 1D contraction (or quiescence) mask.

    A cycle is *truncated at the start* when its first frame lies within
    ``buffer_frames`` of the beginning of the recording, and *truncated at the end*
    when its last frame lies within ``buffer_frames`` of the end — i.e. the onset
    (resp. offset) happened outside the recorded window, so any quantity that needs
    it is unobservable. A cycle truncated on neither side is **complete**.

    This reproduces the classification that ``skimage.segmentation.clear_border``
    used to *delete*; here the cycles are kept and only flagged, so they still
    appear in the mask (plots, quiet/equilibrium baseline, beating-rate onsets)
    while their duration-dependent metrics can be set to NaN.

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
    domain_slen_timeseries: np.ndarray,
    frametime: float,
    model_path: str,
    threshold: Optional[float] = None,
    contr_time_min: float = 0.2,
    merge_time_max: float = 0.05,
    buffer_frames: int = 3,
    min_valid_frames: float = 0.5,
    group_label: str = "Domain",
    id_offset: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Detect contraction cycles from per-group sarcomere length time-series via ContractionNet.

    Predicts contraction states with the ContractionNet network for each row's
    mean sarcomere length signal, then cleans the predictions with morphological
    operations. ``group_label`` and ``id_offset`` affect only log messages: each
    row ``i`` is named ``f"{group_label} {i + id_offset}"`` so the shared engine
    matches the caller's numbering (0-based group id for most kinds; 1-based mask
    label for domains, where group id = mask label - 1).

    Parameters
    ----------
    domain_slen_timeseries : np.ndarray
        Per-group mean sarcomere length time-series, shape ``(n_domains, n_frames)`` (µm).
    frametime : float
        Time between frames in s.
    model_path : str
        Path to the ContractionNet model weights (.pt file).
    threshold : float or None, optional
        Binary threshold for contraction state prediction. None (the default) uses the
        operating point the model was tuned for, read from the checkpoint -- 0.5 for
        ContractionNetV2, 0.3 for the older model, which is not interchangeable.
    contr_time_min : float, optional
        Minimal contraction duration in s; shorter contractions are removed. Default is 0.2.
    merge_time_max : float, optional
        Maximal gap in s between two contractions; closer ones are merged. Default is 0.05.
    buffer_frames : int, optional
        Frames from either end within which a contraction cycle counts as
        **incomplete**: its onset (or offset) lies outside the recorded window.
        Such cycles are *kept* in the mask — so they are plotted, excluded from the
        quiet/equilibrium baseline and contribute their onset to the beating rate —
        but are flagged via ``domain_contr_complete`` so that
        :func:`analyze_contraction_parameters` can NaN the metrics they cannot
        support. Default is 3.
    min_valid_frames : float, optional
        Minimum fraction of valid (non-NaN) frames required to analyze a group. Default is 0.5.
    group_label : str, optional
        Label used to name rows in log messages. Default is "Domain".
    id_offset : int, optional
        Offset added to the row index when naming rows in log messages. Default is 0.

    Returns
    -------
    dict
        Per-group contraction detection results (``max_n_contr`` is the max cycle
        count across groups):

        - 'domain_contr' : np.ndarray ``(n_domains, n_frames)``, binary contraction state
        - 'domain_n_contr' : np.ndarray ``(n_domains,)``, number of contraction cycles
          detected per group, **including** incomplete ones at the recording edges
        - 'domain_n_contr_complete' : np.ndarray ``(n_domains,)``, number of complete cycles
        - 'domain_contr_complete' : np.ndarray ``(n_domains, max_n_contr)``, 1.0 for a
          complete cycle, 0.0 for an incomplete one, NaN padding. Its per-group mean is
          the fraction of complete cycles.
        - 'domain_labels_contr' : np.ndarray ``(n_domains, n_frames)``, contraction cycle labels
        - 'domain_beating_rate' : np.ndarray ``(n_domains,)``, beating rate (Hz)
        - 'domain_beating_rate_variability' : np.ndarray ``(n_domains,)``, std of inter-beat interval (s)
    """
    if threshold is None:
        threshold = recommended_threshold(model_path)

    n_domains, n_frames = domain_slen_timeseries.shape
    
    # Initialize output arrays
    domain_contr = np.zeros((n_domains, n_frames), dtype=bool)
    domain_n_contr = np.zeros(n_domains, dtype=np.int32)
    domain_labels_contr = np.zeros((n_domains, n_frames), dtype=np.int32)
    domain_n_contr_complete = np.zeros(n_domains, dtype=np.int32)
    domain_beating_rate = np.full(n_domains, np.nan)
    domain_beating_rate_var = np.full(n_domains, np.nan)
    complete_per_domain: List[Optional[np.ndarray]] = [None] * n_domains

    # Morphological structuring elements
    structure_closing = np.ones(max(1, int(merge_time_max / frametime)))
    structure_opening = np.ones(max(1, int(contr_time_min / frametime)))
    
    # Process each domain
    for domain_idx in range(n_domains):
        slen_timeseries = domain_slen_timeseries[domain_idx]
        
        # Check if domain has enough valid data
        valid_fraction = np.sum(~np.isnan(slen_timeseries)) / n_frames
        if valid_fraction < min_valid_frames:
            logger.debug(f"{group_label} {domain_idx + id_offset} has insufficient valid data ({valid_fraction:.1%}), skipping.")
            continue
        
        # Interpolate NaN values for prediction
        slen_interp = _interpolate_nans(slen_timeseries)
        
        if np.all(np.isnan(slen_interp)):
            continue
        
        # Predict contractions using ContractionNet
        try:
            contr_pred = predict_contractions(slen_interp, model_path)
            contr = contr_pred[0] > threshold
        except Exception as e:
            logger.warning(f"ContractionNet prediction failed for {group_label.lower()} {domain_idx + id_offset}: {e}")
            continue
        
        # Apply morphological operations to clean up predictions
        contr = binary_opening(binary_closing(contr, structure=structure_closing), structure=structure_opening)
        
        # Store binary contraction state. Cycles at the recording edges are KEPT
        # (they are real contractions: they belong in the plot, they must not be
        # counted as quiescent when estimating the equilibrium length, and their
        # onset is a valid beat) and merely flagged as incomplete below.
        domain_contr[domain_idx] = contr

        # Label contraction cycles and flag the incomplete ones
        labels, n_contr = label(contr)
        domain_labels_contr[domain_idx] = labels
        domain_n_contr[domain_idx] = n_contr
        trunc_start, trunc_end = cycle_truncation_flags(labels, n_contr, buffer_frames)
        complete = ~(trunc_start | trunc_end)
        complete_per_domain[domain_idx] = complete
        domain_n_contr_complete[domain_idx] = int(complete.sum())

        # Calculate beating rate
        if n_contr > 1:
            start_frames = np.where(np.diff(contr.astype('float32')) > 0.5)[0]
            if len(start_frames) > 1:
                inter_beat_intervals = np.diff(start_frames) * frametime
                domain_beating_rate[domain_idx] = 1 / np.mean(inter_beat_intervals)
                domain_beating_rate_var[domain_idx] = np.std(inter_beat_intervals)
        else:
            logger.warning(f"{group_label} {domain_idx + id_offset}: Only {n_contr} contraction cycle(s) detected. "
                          f"Cannot compute beating rate (requires >= 2 cycles).")
    
    # Per-cycle completeness, NaN-padded to the widest group (like every other
    # per-cycle array). Float, so its per-group nanmean is the complete fraction.
    max_n_contr = int(domain_n_contr.max()) if domain_n_contr.size and domain_n_contr.max() > 0 else 1
    domain_contr_complete = np.full((n_domains, max_n_contr), np.nan)
    for domain_idx, complete in enumerate(complete_per_domain):
        if complete is not None and complete.size:
            domain_contr_complete[domain_idx, :complete.size] = complete.astype(float)

    n_incomplete = int(domain_n_contr.sum() - domain_n_contr_complete.sum())
    if n_incomplete:
        logger.info(
            f"{n_incomplete}/{int(domain_n_contr.sum())} contraction cycles are incomplete "
            f"(within {buffer_frames} frames of the recording start/end); they are kept in the "
            f"contraction mask but their duration-dependent metrics are NaN.")

    return {
        'domain_contr': domain_contr,
        'domain_n_contr': domain_n_contr,
        'domain_n_contr_complete': domain_n_contr_complete,
        'domain_contr_complete': domain_contr_complete,
        'domain_labels_contr': domain_labels_contr,
        'domain_beating_rate': domain_beating_rate,
        'domain_beating_rate_variability': domain_beating_rate_var,
    }


def analyze_contraction_parameters(
    domain_slen_timeseries: np.ndarray,
    domain_labels_contr: np.ndarray,
    domain_n_contr: np.ndarray,
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
    domain_slen_timeseries : np.ndarray
        Per-group mean sarcomere length time-series, shape ``(n_domains, n_frames)`` (µm).
    domain_labels_contr : np.ndarray
        Per-group contraction cycle labels, shape ``(n_domains, n_frames)``.
    domain_n_contr : np.ndarray
        Number of contractions per group, shape ``(n_domains,)``.
    frametime : float
        Time between frames in s.
    filter_params : tuple of int, optional
        Savitzky-Golay filter parameters ``(window_length, polyorder)`` for velocity
        smoothing. Default is (13, 5).
    buffer_frames : int, optional
        Frames from either end within which a cycle counts as incomplete (must match
        the value passed to :func:`detect_contractions`). Metrics an incomplete cycle
        cannot support are set to NaN rather than computed from a truncated window:
        ``time_contr`` whenever either edge is missing; ``time_to_peak`` when the
        onset is missing; ``time_to_relax`` when the offset is missing; and an
        amplitude/velocity extremum whenever it falls *on* the truncated boundary,
        because the true extremum then lies outside the recording. Default is 3.

    Returns
    -------
    dict
        Per-group contraction parameters (``max_n_contr`` is the max cycle count
        across groups). Entries for cycles that cannot support a given metric are NaN:

        - 'domain_equ' : np.ndarray ``(n_domains,)``, equilibrium/resting sarcomere length (µm)
        - 'domain_contr_max' : np.ndarray ``(n_domains, max_n_contr)``, max contraction per cycle (µm)
        - 'domain_elong_max' : np.ndarray ``(n_domains, max_n_contr)``, max elongation per cycle (µm)
        - 'domain_vel_contr_max' : np.ndarray ``(n_domains, max_n_contr)``, max shortening velocity (µm/s)
        - 'domain_vel_elong_max' : np.ndarray ``(n_domains, max_n_contr)``, max elongation velocity (µm/s)
        - 'domain_time_to_peak' : np.ndarray ``(n_domains, max_n_contr)``, time to maximal contraction (s)
        - 'domain_time_to_relax' : np.ndarray ``(n_domains, max_n_contr)``, time from peak to relaxation (s)
        - 'domain_time_contr' : np.ndarray ``(n_domains, max_n_contr)``, contraction duration (s)
    """
    n_domains = domain_slen_timeseries.shape[0]
    max_n_contr = int(np.max(domain_n_contr)) if np.max(domain_n_contr) > 0 else 1
    
    # Initialize output arrays
    domain_equ = np.full(n_domains, np.nan)
    domain_contr_max = np.full((n_domains, max_n_contr), np.nan)
    domain_elong_max = np.full((n_domains, max_n_contr), np.nan)
    domain_vel_contr_max = np.full((n_domains, max_n_contr), np.nan)
    domain_vel_elong_max = np.full((n_domains, max_n_contr), np.nan)
    domain_time_to_peak = np.full((n_domains, max_n_contr), np.nan)
    domain_time_to_relax = np.full((n_domains, max_n_contr), np.nan)
    domain_time_contr = np.full((n_domains, max_n_contr), np.nan)
    
    window_length, polyorder = filter_params
    
    for domain_idx in range(n_domains):
        slen = domain_slen_timeseries[domain_idx]
        labels = domain_labels_contr[domain_idx]
        n_contr = domain_n_contr[domain_idx]
        
        if n_contr == 0 or np.all(np.isnan(slen)):
            continue
        
        # Calculate equilibrium length (median of non-NaN values)
        valid_slen = slen[~np.isnan(slen)]
        if len(valid_slen) > 0:
            domain_equ[domain_idx] = np.median(valid_slen)
        
        # Calculate velocity using Savitzky-Golay filter
        slen_interp = _interpolate_nans(slen)
        if len(slen_interp) >= window_length:
            vel = savgol_filter(slen_interp, window_length, polyorder, deriv=1, delta=frametime)
        else:
            vel = np.gradient(slen_interp, frametime)
        
        # Calculate delta (change from equilibrium)
        delta_slen = slen_interp - domain_equ[domain_idx]
        
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
                domain_time_contr[domain_idx, contr_idx] = np.sum(cycle_mask) * frametime

            # Max contraction (most negative delta) and elongation (most positive delta)
            if len(delta_cycle) > 0 and not np.all(np.isnan(delta_cycle)):
                n_cyc = len(delta_cycle)
                peak_idx = int(np.nanargmin(delta_cycle))
                elong_idx = int(np.nanargmax(delta_cycle))
                peak_observed = _extremum_observed(peak_idx, n_cyc)
                if peak_observed:
                    domain_contr_max[domain_idx, contr_idx] = np.nanmin(delta_cycle)
                if _extremum_observed(elong_idx, n_cyc):
                    domain_elong_max[domain_idx, contr_idx] = np.nanmax(delta_cycle)

                # Max velocities — tested against their own arg-extremum, which need
                # not coincide with the length extremum.
                if not np.all(np.isnan(vel_cycle)):
                    if _extremum_observed(int(np.nanargmin(vel_cycle)), n_cyc):
                        domain_vel_contr_max[domain_idx, contr_idx] = np.nanmin(vel_cycle)
                    if _extremum_observed(int(np.nanargmax(vel_cycle)), n_cyc):
                        domain_vel_elong_max[domain_idx, contr_idx] = np.nanmax(vel_cycle)

                # Timing relative to the peak: onset->peak needs the onset,
                # peak->offset needs the offset, and both need an observed peak.
                if peak_observed:
                    if not cut_start:
                        domain_time_to_peak[domain_idx, contr_idx] = peak_idx * frametime
                    if not cut_end:
                        domain_time_to_relax[domain_idx, contr_idx] = (n_cyc - peak_idx) * frametime
    
    return {
        'domain_equ': domain_equ,
        'domain_contr_max': domain_contr_max,
        'domain_elong_max': domain_elong_max,
        'domain_vel_contr_max': domain_vel_contr_max,
        'domain_vel_elong_max': domain_vel_elong_max,
        'domain_time_to_peak': domain_time_to_peak,
        'domain_time_to_relax': domain_time_to_relax,
        'domain_time_contr': domain_time_contr,
    }


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaN values in a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        1D array potentially containing NaNs.

    Returns
    -------
    np.ndarray
        Array with interior NaNs replaced by linear interpolation.
    """
    arr = arr.copy()
    nans = np.isnan(arr)
    
    if np.all(nans):
        return arr
    
    if np.any(nans):
        indices = np.arange(len(arr))
        arr[nans] = np.interp(indices[nans], indices[~nans], arr[~nans])
    
    return arr
