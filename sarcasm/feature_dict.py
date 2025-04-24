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


import numpy as np
from scipy import sparse

# structural features
structure_feature_dict = {
    'cell_mask_area': {
        'description': 'Area occupied by cells in image. NOT the area of individual cells. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_cell_mask',
        'name': 'Cell area [µm²]'
    },
    'cell_mask_area_ratio': {
        'description': 'Area ratio of total image occupied by cells. np.ndarray with value for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_cell_mask',
        'name': 'Cell area ratio'
    },
    'cell_mask_intensity': {
        'description': 'Average intensity at cell mask. np.ndarray with value for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_cell_mask',
        'name': 'Cell mask intensity'
    },
    'domain_area': {
        'description': 'Areas of individual sarcomere domains in µm^2. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Domain area [µm²]'
    },
    'domain_area_mean': {
        'description': 'Mean domain area in µm^2. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Mean domain area [µm²]'
    },
    'domain_area_std': {
        'description': 'Standard deviation of domain area in µm^2. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'STD domain area [µm²]'
    },
    'domain_mask': {
        'description': 'Masks of sarcomere domains, pixel values reflects domain indices, 0 is background. '
                       'Stored as list of sparse arrays. For conversion to np.ndarray, use mask.toarray().',
        'data type': list[sparse.coo_matrix],
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Sarcomere domain mask'
    },
    'domain_oop': {
        'description': 'Sarcomere orientational order parameter (OOP) of individual sarcomere domains. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Domain OOP'
    },
    'domain_oop_mean': {
        'description': 'Mean sarcomere orientational order parameter (OOP) of all sarcomere domains in image. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Mean domain OOP'
    },
    'domain_oop_std': {
        'description': 'Standard deviation of sarcomere orientational order parameter (OOP) of all sarcomere domains in image. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Standard deviation of domain out-of-plane'
    },
    'domain_orientation': {
        'description': 'Sarcomere orientation in radians of individual sarcomere domains. ',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Domain orientation [rad]'
    },
    'domain_slen': {
        'description': 'Mean sarcomere length within each sarcomere domain. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Domain sarcomere length [µm]'
    },
    'domains': {
        'description': 'Set of sarcomere vectors of each sarcomere domain. List with list of np.arrays for each frame, '
                       'storing the indices of sarcomere vectors for each domain.',
        'data type': list[list[np.ndarray]],
        'function': 'Structure.analyze_sarcomere_domains',
        'name': 'Sarcomere domains'
    },
    'max_score_vectors': {
        'description': 'Wavelet score of each sarcomere vector. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Max score sarcomere vectors'
    },
    'midline_id_vectors': {
        'description': 'Midline identifier of each sarcomere vector. '
                       'Value reflects midline label, with unique label for each sarcomere midline. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Midline ID sarcomere vectors'
    },
    'midline_length_vectors': {
        'description': 'Length of repsective sarcomere midline of each sarcomere vector. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Midline length vectors [µm]'
    },
    'myof_length': {
        'description': 'Length of myofibril lines. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_myofibrils',
        'name': 'Myofibril lengths [µm]'
    },
    'myof_length_max': {
        'description': 'Maximum length of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_myofibrils',
        'name': 'Max. myofibril length [µm]'
    },
    'myof_length_mean': {
        'description': 'Mean length of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_myofibrils',
        'name': 'Mean myofibril length [µm]'
    },
    'myof_length_std': {
        'description': 'Standard deviation of length of myofibril lines in each frame. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_myofibrils',
        'name': 'STD myofibril length [µm]'
    },
    'myof_lines': {
        'description': 'Sarcomere vector IDs of myofibril lines. List with list of np.arrays for each frame.',
        'data type': list[list[np.ndarray]],
        'function': 'Structure.analyze_myofibrils',
        'name': 'Myofibril lines'
    },
    'myof_bending': {
        'description': 'Bending (mean squared curvature) of myofibril lines. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_myofibrils',
        'name': 'Myofibril bending'
    },
    'myof_bending_mean': {
        'description': 'Mean of bending (mean squared curvature) of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_myofibrils',
        'name': 'Mean myofibril bending'
    },
    'myof_bending_std': {
        'description': 'Standard deviation of bending (mean squared curvature) of myofibril lines in each frame. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_myofibrils',
        'name': 'STD myofibril bending'
    },
    'myof_straightness': {
        'description': 'Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'List with np.ndarray for each frame.',
        'data type': list[list[np.ndarray]],
        'function': 'Structure.analyze_myofibrils',
        'name': 'Myofibril straightness'
    },
    'myof_straightness_mean': {
        'description': 'Mean of Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_myofibrils',
        'name': 'Mean myofibril straightness'
    },
    'myof_straightness_std': {
        'description': 'Standard deviation of Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_myofibrils',
        'name': 'STD myofibril straightness'
    },
    'n_domains': {
        'description': 'Number of sarcomere domains in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_domains',
        'name': '# sarcomere domains'
    },
    'pos_vectors': {
        'description': 'Position of sarcomere vectors in each frame in pixels. '
                       'List of np.ndarray for each frame',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere vector positions [px]'
    },
    'sarcomere_area': {
        'description': 'Area occupied by sarcomeres. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere area [µm²]'
    },
    'sarcomere_area_ratio': {
        'description': 'Ratio of image area occupied by sarcomeres. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere area ratio'
    },
    'sarcomere_length_mean': {
        'description': 'Mean sarcomere length of sarcomere vectors in each frame. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Mean sarcomere length [µm]'
    },
    'sarcomere_length_vectors': {
        'description': 'Sarcomere length of sarcomere vectors in each frame. List of np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere length vectors [µm]'
    },
    'sarcomere_length_std': {
        'description': 'Standard deviation of sarcomere length of sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'STD sarcomere length [µm]'
    },
    'sarcomere_oop': {
        'description': 'Sarcomere orientational order parameter (OOP) of all sarcomere vectors in frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere OOP'
    },
    'sarcomere_orientation_mean': {
        'description': 'Mean sarcomere orientation of all sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Mean sarcomere orientation [rad]'
    },
    'sarcomere_orientation_vectors': {
        'description': 'Sarcomere orientation of sarcomere vectors. List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere orientation vectors [rad]'
    },
    'sarcomere_orientation_std': {
        'description': 'Standard deviation of sarcomere orientation of all sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'STD sarcomere orientation [rad]'
    },
    'wavelet_bank': {
        'description': 'Wavelet bank. Only stored if save_all=True.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Wavelet bank'
    },
    'wavelet_max_score': {
        'description': 'Spatial map of maximum wavelet score. Only stored if save_all=True.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Wavelet max score'
    },
    'wavelet_sarcomere_length': {
        'description': 'Spatial map of sarcomere length. Only stored if save_all=True.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere length [µm]'
    },
    'wavelet_sarcomere_orientation': {
        'description': 'Spatial map of sarcomere orientation. Only stored if save_all=True.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_sarcomere_vectors',
        'name': 'Sarcomere orientation [rad]'
    },
    'z_avg_intensity': {
        'description': 'Average intensity of Z-bands, i.e. average pixel values of all Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band intensity'
    },
    'z_ends': {
        'description': 'Position of Z-band ends in pixels.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band ends [px]'
    },
    'z_intensity': {
        'description': 'Intensity of individual Z-band objects. List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band intensity'
    },
    'z_intensity_mean': {
        'description': 'Mean intensity of Z-band objects. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean Z-band intensity'
    },
    'z_intensity_std': {
        'description': 'Standard devialtion of intensity of Z-band objects. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD Z-band intensity'
    },
    'z_labels': {
        'description': 'Z-band labels. Image with pixel values reflecting object labels. '
                       'Stored as a sparse matrix, use labels.to_numpy() to convert to np.ndarray.',
        'data type': sparse.csr_matrix,
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band labels'
    },
    'z_lat_alignment': {
        'description': 'Lateral alignment A of pairs of adjacent Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band lateral alignment'
    },
    'z_lat_alignment_groups': {
        'description': 'Mean alignment of pairs of adjacent Z-bands in lateral groups. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band alignment lat. groups'
    },
    'z_lat_alignment_groups_mean': {
        'description': 'Frame-level average of mean alignment of pairs of Z-bands in lateral groups. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean alignment in lateral Z-band groups'
    },
    'z_lat_alignment_groups_std': {
        'description': 'Frame-level standard deviation of mean alignment in lateral groups. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD alignment in lateral Z-band groups'
    },
    'z_lat_alignment_mean': {
        'description': 'Mean lateral alignment of adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean Z-band lateral alignment'
    },
    'z_lat_alignment_std': {
        'description': 'Standard deviation of lateral alignment of adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD Z-band lateral alignment'
    },
    'z_lat_dist': {
        'description': 'Distance of pairs of laterally adjacent Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band lateral distances [µm]'
    },
    'z_lat_dist_mean': {
        'description': 'Mean lateral distance of pairs of laterally adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean Z-band lateral distance'
    },
    'z_lat_dist_std': {
        'description': 'Standard deviation of lateral distance of pairs of laterally adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD Z-band lateral distance'
    },
    'z_lat_groups': {
        'description': 'Groups of laterally aligned Z-band objects. '
                       'List with lists of Z-band indices for each frame.',
        'data type': list[list[list[int]]],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band lateral groups'
    },
    'z_lat_length_groups': {
        'description': 'Lengths of groups of laterally aligned Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Lengths lat. Z-band groups [µm]'
    },
    'z_lat_length_groups_mean': {
        'description': 'Mean length of groups of laterally aligned Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean length lat. Z-band groups [µm]'
    },
    'z_lat_length_groups_std': {
        'description': 'Standard deviation of lengths of groups of laterally aligned Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD length lat. Z-band groups [µm]'
    },
    'z_lat_links': {
        'description': 'Links between laterally aligned Z-band ends. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band lateral links'
    },
    'z_lat_neighbors': {
        'description': 'Number of lateral neighbors of each Z-band object (0, 1 or 2). '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band lateral neighbors [#]'
    },
    'z_lat_neighbors_mean': {
        'description': 'Mean number of lateral neighbors of each Z-band object for each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean Z-band lateral neighbors [#]'
    },
    'z_lat_neighbors_std': {
        'description': 'Standard deviation of number of lateral neighbors of each Z-band object for each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD Z-band lateral neighbors [#]'
    },
    'z_lat_size_groups': {
        'description': 'Size of groups of laterally aligned Z-band objects. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Size of laterally aligned Z-band groups [#]'
    },
    'z_lat_size_groups_mean': {
        'description': 'Mean size of groups of laterally aligned Z-band objects. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean size groups lat. aligned Z-band [#]'
    },
    'z_lat_size_groups_std': {
        'description': 'Standard deviation of size of groups of laterally aligned Z-band objects. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD size lat. aligned Z-band [#]'
    },
    'z_length': {
        'description': 'Length of Z-band objects. '
                       'List with np.ndarray of each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band length [µm]'
    },
    'z_length_max': {
        'description': 'here description',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Max Z length [µm]'
    },
    'z_length_mean': {
        'description': 'Mean Z-band length in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean Z-band length [µm]'
    },
    'z_length_std': {
        'description': 'Standard deviation of Z-band lengths in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD Z-band length [µm]'
    },
    'z_oop': {
        'description': 'Z-band orientation order parameter. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band OOP'
    },
    'z_orientation': {
        'description': 'Orientation of individual Z-band objects. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band orientation [rad]'
    },
    'z_mask_area': {
        'description': 'Total area occupied by Z-bands in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band mask area'
    },
    'z_mask_area_ratio': {
        'description': 'Ratio of area occupied by Z-bands to total cell area.'
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band mask area ratio'
    },
    'z_mask_intensity': {
        'description': 'Average intensity of Z-band mask.'
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band mask intensity'
    },
    'z_straightness': {
        'description': 'Straightness of Z-band objects, measured by ratio of end-to-end length to contour length. '
                       'List with np.ndarray of each frame.',
        'data type': list[np.ndarray],
        'function': 'Structure.analyze_z_bands',
        'name': 'Z-band straightness'
    },
    'z_straightness_mean': {
        'description': 'Mean Z-band straightness for each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'Mean Z-band straightness'
    },
    'z_straightness_std': {
        'description': 'Standard deviation of Z-band straightness for each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'Structure.analyze_z_bands',
        'name': 'STD Z-band straightness'
    }
}

motion_feature_dict = {
    'contr_max': {
        'description': 'Maximal contraction/shortening of each individual sarcomeres in each contraction cycle. '
                       'Array with shape (n_sarcomeres, n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Contr. $\Delta SL_-$ [µm]'
    },
    'contr_max_avg': {
        'description': 'Maximal contraction/shortening of sarcomere average in LOI. '
                       'Array with shape (c_contractions)',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Avg. Contr. $\overline{\Delta SL}_-$ [µm]'
    },
    'elong_max': {
        'description': 'Maximal elongation of each individual sarcomeres in each contraction cycle. '
                       'Array with shape (n_sarcomeres, n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Elong. $\Delta SL_+$ [µm]'
    },
    'elong_max_avg': {
        'description': 'Maximal elongation of sarcomere average in LOI. '
                       'Array with shape (c_contractions)',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Avg. Elong. $\overline{\Delta SL}_+$ [µm]'
    },
    'vel_contr_max': {
        'description': 'Maximal shortening velocity each individual sarcomeres in each contraction cycle. '
                       'Array with shape (n_sarcomeres, n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Vel. Contr. $V_-$ [µm/s]'
    },
    'vel_contr_max_avg': {
        'description': 'Maximal shortening velocity of sarcomere average in LOI. '
                       'Array with shape (n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Avg. Vel. Contr. $\overline{V}_-$ [µm/s]'
    },
    'vel_elong_max': {
        'description': 'Maximal elongation velocity of each individual sarcomeres in each contraction cycle. '
                       'Array with shape (n_sarcomeres, n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Vel. Elong. $V_+$ [µm/s]'
    },
    'vel_elong_max_avg': {
        'description': 'Maximal elongation velocity of sarcomere average in LOI. '
                       'Array with shape (n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Avg. Vel. Elong. $\overline{V}_+$ [µm/s]'
    },
    'equ': {
        'description': 'Resting length of each individual sarcomere. '
                       'Array with shape (n_sarcomeres).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Rest. Length $RL$ [µm]'
    },
    'beating_rate': {
        'description': 'Beating rate of LOI.',
        'data type': float,
        'function': 'Motion.detect_analyze_contractions',
        'name': 'Beating Rate $BR$ [Hz]'
    },
    'beating_rate_variability': {
        'description': 'Beating rate variability. Standard deviation of time between contraction starts.',
        'data type': float,
        'function': 'Motion.detect_analyze_contractions',
        'name': 'BR Variability $BRV$ [s]'
    },
    'time_contr': {
        'description': 'Duration of each individual contraction cycle.'
                       'Array with shape (n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Time Contr. $T_C$ [s]'
    },
    'time_contr_avg': {
        'description': 'Average duration of contraction cycles.',
        'data type': float,
        'function': 'Motion.analyze_trajectories',
        'name': 'Avg. Time Contr. $\overline{T}_C$ [s]'
    },
    'time_quiet': {
        'description': 'Duration of each quiescent period between contraction cycles. '
                       'Array with shape (n_contractions-1).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Time quiet. $T_Q$ [s]'
    },
    'time_quiet_avg': {
        'description': 'Average duration of quiescent periods between contraction cycles. ',
        'data type': float,
        'function': 'Motion.analyze_trajectories',
        'name': 'Time quiet. $T_Q$ [s]'
    },
    'time_to_peak': {
        'description': 'Time to maximal contraction of each individual sarcomere for each contraction cycle. '
                       'Array with shape (n_sarcomeres, n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Time to Peak $T_P$ [s]'
    },
    'time_to_peak_avg': {
        'description': 'Time to maximal contraction of sarcomere average in LOI. '
                       'Array with shape (n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Avg. Time to Peak $\overline{T}_P$ [s]'
    },
    'time_to_relax': {
        'description': 'Time from maximal to end of contraction of each individual sarcomere for each contraction cycle. '
                       'Array with shape (n_sarcomeres, n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Time to Relax $T_R$ [s]'
    },
    'time_to_relax_avg': {
        'description': 'Time from maximal to end of contraction of sarcomere average in LOI. '
                       'Array with shape (n_contractions).',
        'data type': np.ndarray,
        'function': 'Motion.analyze_trajectories',
        'name': 'Avg. Time to Relax $\overline{T}_R$ [s]'
    },
    'popping_events': {
        'description': 'Sarcomere popping events, extensions of sarcomeres far beyond resting length (e.g. 0.25 µm). '
                       'Binary array with shape (n_sarcomeres, n_contractions) with 0 for no popping and 1 for popping.',
        'data type': np.ndarray,
        'function': 'Motion.analyze_popping',
        'name': 'Popping Events'
    },
    'popping_rate': {
        'description': 'Average popping rate in LOI.',
        'data type': float,
        'function': 'Motion.analyze_popping',
        'name': 'Popping Rate $P$'
    },
    'popping_rate_sarcomeres': {
        'description': 'Popping rate of each individual sarcomere.',
        'data type': np.ndarray,
        'function': 'Motion.analyze_popping',
        'name': 'Sarcomere Popping Rate $P_s$'
    },
    'popping_rate_contr': {
        'description': 'Popping rate at each contraction cycle.',
        'data type': np.ndarray,
        'function': 'Motion.analyze_popping',
        'name': 'Contraction Popping Rate $P_c$'
    },
    'ratio_delta_slen_mutual_serial': {
        'description': 'Ratio of mutual to serial correlation for sarcomere length changes.',
        'data type': float,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Ratio Mutual Serial $R_{\Delta SL}$'
    },
    'ratio_vel_mutual_serial': {
        'description': 'Ratio of mutual to serial correlation for sarcomere velocities.',
        'data type': float,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Ratio Mutual Serial $R_{V}$'
    },
    'corr_delta_slen_serial': {
        'description': 'Average serial correlation for sarcomere length changes.',
        'data type': float,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Serial Corr. $\Delta SL$'
    },
    'corr_delta_slen_mutual': {
        'description': 'Average mutual correlation for sarcomere length changes.',
        'data type': float,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Mutual Corr. $\Delta SL$'
    },
    'corr_vel_serial': {
        'description': 'Average serial correlation for sarcomere velocities.',
        'data type': float,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Serial Corr. $V$'
    },
    'corr_vel_mutual': {
        'description': 'Average mutual correlation for sarcomere velocities.',
        'data type': float,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Mutual Corr. $V$'
    },
    'corr_delta_slen': {
        'description': 'Correlation matrix for sarcomere length changes.',
        'data type': np.ndarray,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Corr. $\Delta SL$'
    },
    'corr_vel': {
        'description': 'Correlation matrix for sarcomere velocities.',
        'data type': np.ndarray,
        'function': 'Motion.analyze_sarcomere_correlations',
        'name': 'Corr. $V$'
    }
}