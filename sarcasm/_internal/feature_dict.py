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

"""Feature metadata registries mapping result keys to descriptions, data types, producing functions, and display names."""

import numpy as np
from scipy import sparse

# structural features
structure_feature_dict = {
    'cell_mask_area': {
        'description': 'Area occupied by cells in image. NOT the area of individual cells. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_cell_mask',
        'name': 'Cell area [µm²]'
    },
    'cell_mask_area_ratio': {
        'description': 'Area ratio of total image occupied by cells. np.ndarray with value for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_cell_mask',
        'name': 'Cell area ratio'
    },
    'cell_mask_intensity': {
        'description': 'Average intensity at cell mask. np.ndarray with value for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_cell_mask',
        'name': 'Cell mask intensity'
    },
    'domain_area': {
        'description': 'Areas of individual sarcomere domains in µm^2. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain area [µm²]'
    },
    'domain_area_mean': {
        'description': 'Mean domain area in µm^2. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Mean domain area [µm²]'
    },
    'domain_area_std': {
        'description': 'Standard deviation of domain area in µm^2. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'STD domain area [µm²]'
    },
    'domain_mask': {
        'description': 'Masks of sarcomere domains, pixel values reflects domain indices, 0 is background. '
                       'Stored as list of sparse arrays. For conversion to np.ndarray, use mask.toarray().',
        'data type': list[sparse.coo_matrix],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Sarcomere domain mask'
    },
    'domain_oop': {
        'description': 'Sarcomere orientational order parameter (OOP) of individual sarcomere domains. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain OOP'
    },
    'domain_oop_mean': {
        'description': 'Mean sarcomere orientational order parameter (OOP) of all sarcomere domains in image. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Mean domain OOP'
    },
    'domain_oop_std': {
        'description': 'Standard deviation of sarcomere orientational order parameter (OOP) of all sarcomere domains in image. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Standard deviation of domain out-of-plane'
    },
    'domain_orientation': {
        'description': 'Sarcomere orientation in radians of individual sarcomere domains. ',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain orientation [rad]'
    },
    'domain_slen': {
        'description': 'Mean sarcomere length within each sarcomere domain. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain sarcomere length [µm]'
    },
    'domain_slen_timeseries': {
        'description': 'Time-series of mean sarcomere length within each sarcomere domain. '
                       'np.ndarray with shape (n_domains, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain sarcomere length time-series [µm]'
    },
    'domain_slen_median_timeseries': {
        'description': 'Time-series of median sarcomere length within each sarcomere domain. '
                       'np.ndarray with shape (n_domains, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain sarcomere length median time-series [µm]'
    },
    'domain_slen_std_timeseries': {
        'description': 'Time-series of standard deviation of sarcomere length within each domain. '
                       'np.ndarray with shape (n_domains, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain sarcomere length STD time-series [µm]'
    },
    'domain_slen_q25_timeseries': {
        'description': 'Time-series of 25th percentile of sarcomere length within each domain. '
                       'np.ndarray with shape (n_domains, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain sarcomere length Q25 time-series [µm]'
    },
    'domain_slen_q75_timeseries': {
        'description': 'Time-series of 75th percentile of sarcomere length within each domain. '
                       'np.ndarray with shape (n_domains, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain sarcomere length Q75 time-series [µm]'
    },
    'domain_n_vectors_timeseries': {
        'description': 'Time-series of number of sarcomere vectors within each domain. '
                       'np.ndarray with shape (n_domains, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain vector count time-series'
    },
    'domain_contr': {
        'description': 'Binary contraction state for each domain over time. '
                       'np.ndarray with shape (n_domains, n_frames). True = contracting, False = quiescent.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain contraction state'
    },
    'domain_n_contr': {
        'description': 'Number of contraction cycles detected for each domain, including cycles that are '
                       'incomplete at the start/end of the recording. np.ndarray with shape (n_domains,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain number of contractions'
    },
    'domain_n_contr_complete': {
        'description': 'Number of complete contraction cycles for each domain, i.e. cycles whose onset and '
                       'offset both fall inside the recording. np.ndarray with shape (n_domains,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain number of complete contractions'
    },
    'domain_contr_complete': {
        'description': 'Per-cycle completeness flag: 1.0 = complete, 0.0 = incomplete (truncated by the start '
                       'or end of the recording), NaN = padding. np.ndarray with shape (n_domains, max_n_contr). '
                       'Incomplete cycles are kept in the contraction mask but their duration-dependent '
                       'metrics are NaN.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain contraction completeness'
    },
    'domain_labels_contr': {
        'description': 'Contraction cycle labels for each domain over time. '
                       'np.ndarray with shape (n_domains, n_frames). Values 1, 2, 3, ... label each contraction cycle.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain contraction labels'
    },
    'domain_beating_rate': {
        'description': 'Beating rate in Hz for each domain. np.ndarray with shape (n_domains,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain beating rate [Hz]'
    },
    'domain_beating_rate_variability': {
        'description': 'Standard deviation of inter-beat interval for each domain. np.ndarray with shape (n_domains,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain beating rate variability [s]'
    },
    'domain_equ': {
        'description': 'Equilibrium (resting) sarcomere length for each domain. np.ndarray with shape (n_domains,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain equilibrium sarcomere length [µm]'
    },
    'domain_contr_max': {
        'description': 'Maximum contraction (most negative sarcomere length change from equilibrium) '
                       'for each domain and contraction cycle. np.ndarray with shape (n_domains, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain max contraction [µm]'
    },
    'domain_elong_max': {
        'description': 'Maximum elongation (most positive sarcomere length change from equilibrium) '
                       'for each domain and contraction cycle. np.ndarray with shape (n_domains, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain max elongation [µm]'
    },
    'domain_vel_contr_max': {
        'description': 'Maximum shortening velocity for each domain and contraction cycle. '
                       'np.ndarray with shape (n_domains, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain max shortening velocity [µm/s]'
    },
    'domain_vel_elong_max': {
        'description': 'Maximum elongation velocity for each domain and contraction cycle. '
                       'np.ndarray with shape (n_domains, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain max elongation velocity [µm/s]'
    },
    'domain_time_to_peak': {
        'description': 'Time from contraction start to maximum contraction for each domain and cycle. '
                       'np.ndarray with shape (n_domains, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain time to peak [s]'
    },
    'domain_time_to_relax': {
        'description': 'Time from maximum contraction to relaxation for each domain and cycle. '
                       'np.ndarray with shape (n_domains, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain time to relaxation [s]'
    },
    'domain_time_contr': {
        'description': 'Duration of each contraction cycle for each domain. '
                       'np.ndarray with shape (n_domains, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Domain contraction duration [s]'
    },
    'domains': {
        'description': 'Set of sarcomere vectors of each sarcomere domain. List with list of np.arrays for each frame, '
                       'storing the indices of sarcomere vectors for each domain.',
        'data type': list[list[np.ndarray]],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Sarcomere domains'
    },
    'midline_id_vectors': {
        'description': 'Midline identifier of each sarcomere vector. '
                       'Value reflects midline label, with unique label for each sarcomere midline. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Midline ID sarcomere vectors'
    },
    'midline_length_vectors': {
        'description': 'Length of repsective sarcomere midline of each sarcomere vector. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Midline length vectors [µm]'
    },
    'myof_length': {
        'description': 'Length of myofibril lines. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril lengths [µm]'
    },
    'myof_length_max': {
        'description': 'Maximum length of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Max. myofibril length [µm]'
    },
    'myof_length_mean': {
        'description': 'Mean length of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Mean myofibril length [µm]'
    },
    'myof_length_std': {
        'description': 'Standard deviation of length of myofibril lines in each frame. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'STD myofibril length [µm]'
    },
    'myof_lines': {
        'description': 'Sarcomere vector IDs of myofibril lines. List with list of np.arrays for each frame.',
        'data type': list[list[np.ndarray]],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril lines'
    },
    'myof_bending': {
        'description': 'Bending (mean squared curvature) of myofibril lines. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril bending'
    },
    'myof_bending_mean': {
        'description': 'Mean of bending (mean squared curvature) of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Mean myofibril bending'
    },
    'myof_bending_std': {
        'description': 'Standard deviation of bending (mean squared curvature) of myofibril lines in each frame. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'STD myofibril bending'
    },
    'myof_straightness': {
        'description': 'Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'List with np.ndarray for each frame.',
        'data type': list[list[np.ndarray]],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril straightness'
    },
    'myof_straightness_mean': {
        'description': 'Mean of Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Mean myofibril straightness'
    },
    'myof_straightness_std': {
        'description': 'Standard deviation of Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'STD myofibril straightness'
    },
    'n_domains': {
        'description': 'Number of sarcomere domains in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': '# Sarcomere domains'
    },
    'n_mbands': {
        'description': 'Number of estimated m-bands in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': '# M-bands'
    },
    'n_vectors': {
        'description': 'Number of sarcomere vectors in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': '# Sarcomere vectors'
    },
    'n_zbands': {
        'description': 'Number of Z-bands in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': '# Z-bands'
    },
    'pos_vectors': {
        'description': 'Position of sarcomere vectors in each frame in pixels. '
                       'List of np.ndarray for each frame',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere vector positions [px]'
    },
    'sarcomere_area': {
        'description': 'Area occupied by sarcomeres. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere area [µm²]'
    },
    'sarcomere_area_ratio': {
        'description': 'Ratio of cell mask area occupied by sarcomeres. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere area ratio'
    },
    'sarcomere_length_mean': {
        'description': 'Mean sarcomere length of sarcomere vectors in each frame. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Mean sarcomere length [µm]'
    },
    'sarcomere_length_vectors': {
        'description': 'Sarcomere length of sarcomere vectors in each frame. List of np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere length vectors [µm]'
    },
    'sarcomere_length_std': {
        'description': 'Standard deviation of sarcomere length of sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'STD sarcomere length [µm]'
    },
    'sarcomere_oop': {
        'description': 'Sarcomere orientational order parameter (OOP) of all sarcomere vectors in frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere OOP'
    },
    'sarcomere_orientation_mean': {
        'description': 'Mean sarcomere orientation of all sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Mean sarcomere orientation [rad]'
    },
    'sarcomere_orientation_vectors': {
        'description': 'Sarcomere orientation of sarcomere vectors. List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere orientation vectors [rad]'
    },
    'sarcomere_orientation_std': {
        'description': 'Standard deviation of sarcomere orientation of all sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'STD sarcomere orientation [rad]'
    },
    'z_avg_intensity': {
        'description': 'Average intensity of Z-bands, i.e. average pixel values of all Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band intensity'
    },
    'z_ends': {
        'description': 'Position of Z-band ends in pixels.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band ends [px]'
    },
    'z_intensity': {
        'description': 'Intensity of individual Z-band objects. List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band intensity'
    },
    'z_intensity_mean': {
        'description': 'Mean intensity of Z-band objects. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band intensity'
    },
    'z_intensity_std': {
        'description': 'Standard devialtion of intensity of Z-band objects. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band intensity'
    },
    'z_labels': {
        'description': 'Z-band labels. Image with pixel values reflecting object labels. '
                       'Stored as a sparse matrix, use labels.to_numpy() to convert to np.ndarray.',
        'data type': sparse.csr_matrix,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band labels'
    },
    'z_lat_alignment': {
        'description': 'Lateral alignment A of pairs of adjacent Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral alignment'
    },
    'z_lat_alignment_groups': {
        'description': 'Mean alignment of pairs of adjacent Z-bands in lateral groups. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band alignment lat. groups'
    },
    'z_lat_alignment_groups_mean': {
        'description': 'Frame-level average of mean alignment of pairs of Z-bands in lateral groups. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean alignment in lateral Z-band groups'
    },
    'z_lat_alignment_groups_std': {
        'description': 'Frame-level standard deviation of mean alignment in lateral groups. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD alignment in lateral Z-band groups'
    },
    'z_lat_alignment_mean': {
        'description': 'Mean lateral alignment of adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band lateral alignment'
    },
    'z_lat_alignment_std': {
        'description': 'Standard deviation of lateral alignment of adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band lateral alignment'
    },
    'z_lat_dist': {
        'description': 'Distance of pairs of laterally adjacent Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral distances [µm]'
    },
    'z_lat_dist_mean': {
        'description': 'Mean lateral distance of pairs of laterally adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band lateral distance'
    },
    'z_lat_dist_std': {
        'description': 'Standard deviation of lateral distance of pairs of laterally adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band lateral distance'
    },
    'z_lat_groups': {
        'description': 'Groups of laterally aligned Z-band objects. '
                       'List with lists of Z-band indices for each frame.',
        'data type': list[list[list[int]]],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral groups'
    },
    'z_lat_length_groups': {
        'description': 'Lengths of groups of laterally aligned Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Lengths lat. Z-band groups [µm]'
    },
    'z_lat_length_groups_mean': {
        'description': 'Mean length of groups of laterally aligned Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean length lat. Z-band groups [µm]'
    },
    'z_lat_length_groups_std': {
        'description': 'Standard deviation of lengths of groups of laterally aligned Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD length lat. Z-band groups [µm]'
    },
    'z_lat_links': {
        'description': 'Links between laterally aligned Z-band ends. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral links'
    },
    'z_lat_neighbors': {
        'description': 'Number of lateral neighbors of each Z-band object (0, 1 or 2). '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral neighbors [#]'
    },
    'z_lat_neighbors_mean': {
        'description': 'Mean number of lateral neighbors of each Z-band object for each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band lateral neighbors [#]'
    },
    'z_lat_neighbors_std': {
        'description': 'Standard deviation of number of lateral neighbors of each Z-band object for each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band lateral neighbors [#]'
    },
    'z_lat_size_groups': {
        'description': 'Size of groups of laterally aligned Z-band objects. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Size of laterally aligned Z-band groups [#]'
    },
    'z_lat_size_groups_mean': {
        'description': 'Mean size of groups of laterally aligned Z-band objects. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean size groups lat. aligned Z-band [#]'
    },
    'z_lat_size_groups_std': {
        'description': 'Standard deviation of size of groups of laterally aligned Z-band objects. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD size lat. aligned Z-band [#]'
    },
    'z_length': {
        'description': 'Length of Z-band objects. '
                       'List with np.ndarray of each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band length [µm]'
    },
    'z_length_max': {
        'description': 'here description',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Max Z length [µm]'
    },
    'z_length_mean': {
        'description': 'Mean Z-band length in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band length [µm]'
    },
    'z_length_std': {
        'description': 'Standard deviation of Z-band lengths in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band length [µm]'
    },
    'z_oop': {
        'description': 'Z-band orientation order parameter. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band OOP'
    },
    'z_orientation': {
        'description': 'Orientation of individual Z-band objects. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band orientation [rad]'
    },
    'z_mask_area': {
        'description': 'Total area occupied by Z-bands in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band mask area'
    },
    'z_mask_area_ratio': {
        'description': 'Ratio of area occupied by Z-bands to total cell area.'
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band mask area ratio'
    },
    'z_mask_intensity': {
        'description': 'Average intensity of Z-band mask.'
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band mask intensity'
    },
    'z_straightness': {
        'description': 'Straightness of Z-band objects, measured by ratio of end-to-end length to contour length. '
                       'List with np.ndarray of each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band straightness'
    },
    'z_straightness_mean': {
        'description': 'Mean Z-band straightness for each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band straightness'
    },
    'z_straightness_std': {
        'description': 'Standard deviation of Z-band straightness for each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band straightness'
    },
    # --- 2D sarcomere-vector tracking (SarcAsM.track_sarcomere_vectors) ---
    'n_tracks': {
        'description': 'Number of sarcomere query-point tracks kept after the min_track_length filter.',
        'data type': int,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Number of tracks'
    },
    'track_ids': {
        'description': 'Internal slot id of each kept track. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track ids'
    },
    'track_start_frame': {
        'description': 'Frame at which each track first appears. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track start frame'
    },
    'track_lengths': {
        'description': 'Number of frames each track actually snapped to a detection. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track length [frames]'
    },
    'track_drift_um': {
        'description': 'Drift of each track relative to the coherent motion of its local '
                       'neighbourhood (µm); ~one sarcomere length indicates a changed identity. '
                       'NaN if too short to score. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track drift [µm]'
    },
    'tracks_positions_um': {
        'description': 'Per-track sarcomere-centre positions (y, x) in µm. np.ndarray, shape (n_tracks, T, 2); '
                       'NaN before start / after close.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track positions [µm]'
    },
    'tracks_positions_px': {
        'description': 'Per-track sarcomere-centre positions (y, x) in pixels. np.ndarray, shape (n_tracks, T, 2).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track positions [px]'
    },
    'tracks_slen': {
        'description': 'Per-track sarcomere length over time in µm. np.ndarray, shape (n_tracks, T); '
                       'NaN on gap (non-snapped) frames. The core per-sarcomere length-vs-time signal.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track sarcomere length [µm]'
    },
    'tracks_orientations': {
        'description': 'Per-track sarcomere orientation over time in radians. np.ndarray, shape (n_tracks, T); '
                       'NaN on gap frames.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track orientation [rad]'
    },
    'tracks_snapped': {
        'description': 'Boolean mask, True where a real detection was snapped (vs flow-predicted gap). '
                       'np.ndarray, shape (n_tracks, T).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track snapped mask'
    },
    'tracks_detection_id': {
        'description': 'Index of the snapped detection into pos_vectors_px of that frame (-1 on gap frames). '
                       'np.ndarray, shape (n_tracks, T). Joins a track back to the per-frame vector analysis.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track detection id'
    },
    'tracks_midline_id': {
        'description': 'M-band (midline) id of the snapped detection per frame (-1 on gap frames). '
                       'np.ndarray, shape (n_tracks, T). Basis for M-band-level grouping.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track M-band id'
    },
    'n_merges': {
        'description': 'Number of fragmented trajectory pairs stitched by the post-loop merge step.',
        'data type': int,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Number of track merges'
    },
    'motionfield_source': {
        'description': "Which producer last wrote the motion-field keys: 'tracker' "
                       '(track_sarcomere_vectors) or \'standalone\' (compute_motion_field).',
        'data type': str,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Motion-field source'
    },
    'displacement_along_sarcomere': {
        'description': 'Per-frame optical-flow displacement projected onto the sarcomere axis, in µm. '
                       'List of np.ndarray (one per frame). Also stored namespaced as motionfield_<source>_*.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Displacement along sarcomere [µm]'
    },
    'displacement_perpendicular': {
        'description': 'Per-frame optical-flow displacement perpendicular to the sarcomere axis, in µm. '
                       'List of np.ndarray (one per frame).',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Displacement perpendicular [µm]'
    },
    'velocity_magnitude': {
        'description': 'Per-frame optical-flow speed at each detection, in µm/s. List of np.ndarray (one per frame).',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Velocity magnitude [µm/s]'
    }
}

motion_feature_dict = {
    # Track-based grouped motion (one value per group). Suffix keys are resolved
    # to <kind>_<suffix> per grouping kind by SarcAsM.analyze_track_motion.
    'kind': {
        'description': 'Grouping used for track-based motion analysis (pool, mband, myofibril, domain, loi).',
        'data type': str, 'function': 'SarcAsM.group_tracks', 'name': 'Grouping kind'},
    'group_id': {
        'description': 'Group index within the chosen grouping (0 .. n_groups-1).',
        'data type': int, 'function': 'SarcAsM.group_tracks', 'name': 'Group id'},
    'group_member_count': {
        'description': 'Number of tracks assigned to the group.',
        'data type': int, 'function': 'SarcAsM.group_tracks', 'name': 'Members'},
    'beating_rate': {
        'description': 'Beating rate of the group (Hz); 1 / mean inter-beat interval.',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Beating rate [Hz]'},
    'beating_rate_variability': {
        'description': 'Standard deviation of inter-beat intervals of the group (s).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Beating rate variability [s]'},
    'n_contr': {
        'description': 'Number of detected contraction cycles in the group, including cycles that are '
                       'incomplete at the start/end of the recording.',
        'data type': int, 'function': 'SarcAsM.analyze_track_motion', 'name': 'N contractions'},
    'n_contr_complete': {
        'description': 'Number of complete contraction cycles in the group (onset and offset both inside '
                       'the recording). Only these back the timing features below.',
        'data type': int, 'function': 'SarcAsM.analyze_track_motion', 'name': 'N complete contractions'},
    'contr_complete': {
        'description': 'Fraction of the group\'s contraction cycles that are complete (per-cycle flag, '
                       'averaged over cycles on export).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Complete cycle fraction'},
    'equ': {
        'description': 'Equilibrium (resting) sarcomere length of the group (µm).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Equ. SL [µm]'},
    'contr_max': {
        'description': 'Maximal contraction (shortening) per cycle, mean over cycles (µm).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Contr. dSL- [µm]'},
    'elong_max': {
        'description': 'Maximal elongation per cycle, mean over cycles (µm).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Elong. dSL+ [µm]'},
    'vel_contr_max': {
        'description': 'Maximal shortening velocity per cycle, mean over cycles (µm/s).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Contr. velocity [µm/s]'},
    'vel_elong_max': {
        'description': 'Maximal elongation velocity per cycle, mean over cycles (µm/s).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Elong. velocity [µm/s]'},
    'time_to_peak': {
        'description': 'Time from contraction onset to maximum shortening, mean over cycles (s). NaN for a '
                       'cycle whose onset falls outside the recording, so only complete-at-the-start cycles '
                       'contribute.',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Time to peak [s]'},
    'time_to_relax': {
        'description': 'Time from peak shortening to end of contraction, mean over cycles (s). NaN for a '
                       'cycle whose offset falls outside the recording, so only complete-at-the-end cycles '
                       'contribute.',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Time to relax [s]'},
    'time_contr': {
        'description': 'Total contraction duration per cycle, mean over cycles (s). Only complete cycles '
                       'contribute — a cycle truncated by the start or end of the recording has no '
                       'measurable duration and is NaN (see n_contr_complete).',
        'data type': float, 'function': 'SarcAsM.analyze_track_motion', 'name': 'Contraction time [s]'},
    'slen_timeseries': {
        'description': 'Aggregated per-group sarcomere length over time, shape (n_groups, T).',
        'data type': np.ndarray, 'function': 'SarcAsM.analyze_track_motion', 'name': 'SL(t) [µm]'},
}
