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
m-band, myofibril, LOI, domain) via :func:`grouped_motion.run_cycle_engine`, as
well as the deprecated mask-based :meth:`SarcAsM.analyze_domain_motion`. The
one genuinely domain-specific helper here is :func:`compute_domain_timeseries`.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label
from scipy.signal import savgol_filter
from skimage.segmentation import clear_border

from contraction_net.prediction import predict_contractions
from sarcasm.analysis.domain_clustering import (
    analyze_domains,
    assign_vectors_to_domains,
)

logger = logging.getLogger(__name__)


def compute_domain_timeseries(
    pos_vectors_all: List[np.ndarray],
    sarcomere_length_vectors_all: List[np.ndarray],
    domain_mask: np.ndarray,
    pixelsize: float,
    n_domains: int,
) -> Dict[str, np.ndarray]:
    """
    Compute per-domain sarcomere length statistics over time.

    For each frame, assigns sarcomere vectors to domains by position and computes
    summary statistics (mean, median, std, quartiles) of sarcomere lengths per domain.

    Parameters
    ----------
    pos_vectors_all : list of np.ndarray
        Per-frame position vectors, each of shape ``(n_vectors, 2)`` in µm.
    sarcomere_length_vectors_all : list of np.ndarray
        Per-frame sarcomere length vectors, each of shape ``(n_vectors,)`` in µm.
    domain_mask : np.ndarray
        Integer-labeled domain mask from the reference frame. Domain IDs are 1, 2, 3,
        ...; background is 0.
    pixelsize : float
        Pixel size in µm.
    n_domains : int
        Number of domains in the mask (excluding background).

    Returns
    -------
    dict
        Per-domain time-series of shape ``(n_domains, n_frames)``:

        - 'domain_slen_timeseries' : mean sarcomere length (µm)
        - 'domain_slen_median_timeseries' : median sarcomere length (µm)
        - 'domain_slen_std_timeseries' : std of sarcomere length (µm)
        - 'domain_slen_q25_timeseries' : 25th percentile (µm)
        - 'domain_slen_q75_timeseries' : 75th percentile (µm)
        - 'domain_n_vectors_timeseries' : number of vectors
    """
    n_frames = len(pos_vectors_all)
    
    # Initialize output arrays
    domain_slen_mean = np.full((n_domains, n_frames), np.nan)
    domain_slen_median = np.full((n_domains, n_frames), np.nan)
    domain_slen_std = np.full((n_domains, n_frames), np.nan)
    domain_slen_q25 = np.full((n_domains, n_frames), np.nan)
    domain_slen_q75 = np.full((n_domains, n_frames), np.nan)
    domain_n_vectors = np.zeros((n_domains, n_frames), dtype=np.int32)
    
    # Process each frame
    for frame_idx, (pos_vectors, sarcomere_lengths) in enumerate(
        zip(pos_vectors_all, sarcomere_length_vectors_all)
    ):
        if pos_vectors is None or len(pos_vectors) == 0:
            continue
            
        # Assign vectors to domains
        domain_ids = assign_vectors_to_domains(pos_vectors, domain_mask, pixelsize)
        
        # Compute statistics for each domain
        for domain_id in range(1, n_domains + 1):
            mask = domain_ids == domain_id
            n_vec = np.sum(mask)
            domain_n_vectors[domain_id - 1, frame_idx] = n_vec
            
            if n_vec > 0:
                lengths = sarcomere_lengths[mask]
                domain_slen_mean[domain_id - 1, frame_idx] = np.nanmean(lengths)
                domain_slen_median[domain_id - 1, frame_idx] = np.nanmedian(lengths)
                domain_slen_std[domain_id - 1, frame_idx] = np.nanstd(lengths)
                domain_slen_q25[domain_id - 1, frame_idx] = np.nanpercentile(lengths, 25)
                domain_slen_q75[domain_id - 1, frame_idx] = np.nanpercentile(lengths, 75)
    
    return {
        'domain_slen_timeseries': domain_slen_mean,
        'domain_slen_median_timeseries': domain_slen_median,
        'domain_slen_std_timeseries': domain_slen_std,
        'domain_slen_q25_timeseries': domain_slen_q25,
        'domain_slen_q75_timeseries': domain_slen_q75,
        'domain_n_vectors_timeseries': domain_n_vectors,
    }


def detect_contractions(
    domain_slen_timeseries: np.ndarray,
    frametime: float,
    model_path: str,
    threshold: float = 0.3,
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
    threshold : float, optional
        Binary threshold for contraction state prediction. Default is 0.3.
    contr_time_min : float, optional
        Minimal contraction duration in s; shorter contractions are removed. Default is 0.2.
    merge_time_max : float, optional
        Maximal gap in s between two contractions; closer ones are merged. Default is 0.05.
    buffer_frames : int, optional
        Remove contraction cycles within this many frames of the start/end. Default is 3.
    min_valid_frames : float, optional
        Minimum fraction of valid (non-NaN) frames required to analyze a group. Default is 0.5.
    group_label : str, optional
        Label used to name rows in log messages. Default is "Domain".
    id_offset : int, optional
        Offset added to the row index when naming rows in log messages. Default is 0.

    Returns
    -------
    dict
        Per-group contraction detection results:

        - 'domain_contr' : np.ndarray ``(n_domains, n_frames)``, binary contraction state
        - 'domain_n_contr' : np.ndarray ``(n_domains,)``, number of contractions per group
        - 'domain_labels_contr' : np.ndarray ``(n_domains, n_frames)``, contraction cycle labels
        - 'domain_beating_rate' : np.ndarray ``(n_domains,)``, beating rate (Hz)
        - 'domain_beating_rate_variability' : np.ndarray ``(n_domains,)``, std of inter-beat interval (s)
    """
    n_domains, n_frames = domain_slen_timeseries.shape
    
    # Initialize output arrays
    domain_contr = np.zeros((n_domains, n_frames), dtype=bool)
    domain_n_contr = np.zeros(n_domains, dtype=np.int32)
    domain_labels_contr = np.zeros((n_domains, n_frames), dtype=np.int32)
    domain_beating_rate = np.full(n_domains, np.nan)
    domain_beating_rate_var = np.full(n_domains, np.nan)
    
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
        
        # Remove incomplete contractions at beginning/end
        contr = clear_border(contr, buffer_size=buffer_frames)
        
        # Store binary contraction state
        domain_contr[domain_idx] = contr
        
        # Label contraction cycles
        labels, n_contr = label(contr)
        domain_labels_contr[domain_idx] = labels
        domain_n_contr[domain_idx] = n_contr
        
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
    
    return {
        'domain_contr': domain_contr,
        'domain_n_contr': domain_n_contr,
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

    Returns
    -------
    dict
        Per-group contraction parameters (``max_n_contr`` is the max cycle count
        across groups):

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
        
        # Analyze each contraction cycle
        for contr_idx in range(n_contr):
            cycle_mask = labels == (contr_idx + 1)
            if not np.any(cycle_mask):
                continue
            
            delta_cycle = delta_slen[cycle_mask]
            vel_cycle = vel[cycle_mask]
            
            # Contraction duration
            domain_time_contr[domain_idx, contr_idx] = np.sum(cycle_mask) * frametime
            
            # Max contraction (most negative delta) and elongation (most positive delta)
            if len(delta_cycle) > 0:
                domain_contr_max[domain_idx, contr_idx] = np.nanmin(delta_cycle)
                domain_elong_max[domain_idx, contr_idx] = np.nanmax(delta_cycle)
                
                # Max velocities
                domain_vel_contr_max[domain_idx, contr_idx] = np.nanmin(vel_cycle)
                domain_vel_elong_max[domain_idx, contr_idx] = np.nanmax(vel_cycle)
                
                # Time to peak (time from start to minimum)
                if not np.all(np.isnan(delta_cycle)):
                    peak_idx = np.nanargmin(delta_cycle)
                    domain_time_to_peak[domain_idx, contr_idx] = peak_idx * frametime
                    domain_time_to_relax[domain_idx, contr_idx] = (len(delta_cycle) - peak_idx) * frametime
    
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
