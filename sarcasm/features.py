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

"""Feature metadata registries mapping result keys to descriptions, data types, producing functions, and display names.

The registries are keyed differently: :data:`structure_feature_dict` uses the
full flat result key (``'structure.sarcomere.oop'``), while :data:`motion_feature_dict`
uses the bare feature suffix (``'beating_rate'``) because
:meth:`sarcasm.SarcAsM.analyze_track_motion` writes it once per grouping level
as ``'<kind>_<suffix>'``. Use :func:`describe_key` rather than indexing the
dicts directly — it knows both conventions.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse

# structural features
structure_feature_dict = {
    'structure.cell.mask_area': {
        'description': 'Area occupied by cells in image. NOT the area of individual cells. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_cell_mask',
        'name': 'Cell area [µm²]'
    },
    'structure.cell.mask_area_ratio': {
        'description': 'Area ratio of total image occupied by cells. np.ndarray with value for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_cell_mask',
        'name': 'Cell area ratio'
    },
    'structure.cell.mask_intensity': {
        'description': 'Average intensity at cell mask. np.ndarray with value for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_cell_mask',
        'name': 'Cell mask intensity'
    },
    'structure.domain.area': {
        'description': 'Areas of individual sarcomere domains in µm^2. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain area [µm²]'
    },
    'structure.domain.area_mean': {
        'description': 'Mean domain area in µm^2. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Mean domain area [µm²]'
    },
    'structure.domain.area_std': {
        'description': 'Standard deviation of domain area in µm^2. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'STD domain area [µm²]'
    },
    'structure.domain.mask': {
        'description': 'Masks of sarcomere domains, pixel values reflects domain indices, 0 is background. '
                       'Stored as list of sparse arrays. For conversion to np.ndarray, use mask.toarray().',
        'data type': list[sparse.coo_matrix],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Sarcomere domain mask'
    },
    'structure.domain.oop': {
        'description': 'Sarcomere orientational order parameter (OOP) of individual sarcomere domains. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain OOP'
    },
    'structure.domain.oop_mean': {
        'description': 'Mean sarcomere orientational order parameter (OOP) of all sarcomere domains in image. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Mean domain OOP'
    },
    'structure.domain.oop_std': {
        'description': 'Standard deviation of sarcomere orientational order parameter (OOP) of all sarcomere domains in image. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'STD domain OOP'
    },
    'structure.domain.orientation': {
        'description': 'Sarcomere orientation in radians of individual sarcomere domains. ',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain orientation [rad]'
    },
    'structure.domain.slen_mean': {
        'description': 'Mean sarcomere length across all sarcomere domains in the image, in µm. '
                       'np.ndarray with one value per frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Mean domain sarcomere length [µm]'
    },
    'structure.domain.slen_std': {
        'description': 'Standard deviation of sarcomere length across all sarcomere domains in the '
                       'image, in µm. np.ndarray with one value per frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'STD domain sarcomere length [µm]'
    },
    'structure.domain.slen': {
        'description': 'Mean sarcomere length within each sarcomere domain. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Domain sarcomere length [µm]'
    },
    'structure.domain.members': {
        'description': 'Set of sarcomere vectors of each sarcomere domain. List with list of np.arrays for each frame, '
                       'storing the indices of sarcomere vectors for each domain.',
        'data type': list[list[np.ndarray]],
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': 'Sarcomere domains'
    },
    'structure.sarcomere.midline_id': {
        'description': 'Midline identifier of each sarcomere vector. '
                       'Value reflects midline label, with unique label for each sarcomere midline. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Midline ID sarcomere vectors'
    },
    'structure.sarcomere.midline_length': {
        'description': 'Length of repsective sarcomere midline of each sarcomere vector. '
                       'List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Midline length vectors [µm]'
    },
    'structure.myofibril.length': {
        'description': 'Length of myofibril lines. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril lengths [µm]'
    },
    'structure.myofibril.length_max': {
        'description': 'Maximum length of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Max. myofibril length [µm]'
    },
    'structure.myofibril.length_mean': {
        'description': 'Mean length of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Mean myofibril length [µm]'
    },
    'structure.myofibril.length_std': {
        'description': 'Standard deviation of length of myofibril lines in each frame. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'STD myofibril length [µm]'
    },
    'structure.myofibril.lines': {
        'description': 'Sarcomere vector IDs of myofibril lines. List with list of np.arrays for each frame.',
        'data type': list[list[np.ndarray]],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril lines'
    },
    'structure.myofibril.bending': {
        'description': 'Bending (mean squared curvature) of myofibril lines. List with np.array for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril bending'
    },
    'structure.myofibril.bending_mean': {
        'description': 'Mean of bending (mean squared curvature) of myofibril lines in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Mean myofibril bending'
    },
    'structure.myofibril.bending_std': {
        'description': 'Standard deviation of bending (mean squared curvature) of myofibril lines in each frame. '
                       'np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'STD myofibril bending'
    },
    'structure.myofibril.straightness': {
        'description': 'Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'List with np.ndarray for each frame.',
        'data type': list[list[np.ndarray]],
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Myofibril straightness'
    },
    'structure.myofibril.straightness_mean': {
        'description': 'Mean of Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'Mean myofibril straightness'
    },
    'structure.myofibril.straightness_std': {
        'description': 'Standard deviation of Frechet straightness (max. perpendicular distance to direct end-to-end line) of myofibril lines in each frame. ' 
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_myofibrils',
        'name': 'STD myofibril straightness'
    },
    'structure.domain.n': {
        'description': 'Number of sarcomere domains in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_domains',
        'name': '# Sarcomere domains'
    },
    'structure.sarcomere.n_mbands': {
        'description': 'Number of estimated m-bands in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': '# M-bands'
    },
    'structure.sarcomere.n_vectors': {
        'description': 'Number of sarcomere vectors in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': '# Sarcomere vectors'
    },
    'structure.zbands.n': {
        'description': 'Number of Z-bands in each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': '# Z-bands'
    },
    'structure.sarcomere.pos': {
        'description': 'Position (y, x) of each sarcomere vector in µm. '
                       'List with np.ndarray of shape (2, n_vectors) for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere vector position [µm]'
    },
    'structure.sarcomere.area': {
        'description': 'Area occupied by sarcomeres. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere area [µm²]'
    },
    'structure.sarcomere.area_ratio': {
        'description': 'Ratio of cell mask area occupied by sarcomeres. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere area ratio'
    },
    'structure.sarcomere.slen_mean': {
        'description': 'Mean sarcomere length of sarcomere vectors in each frame. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Mean sarcomere length [µm]'
    },
    'structure.sarcomere.slen': {
        'description': 'Sarcomere length of sarcomere vectors in each frame. List of np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere length vectors [µm]'
    },
    'structure.sarcomere.slen_std': {
        'description': 'Standard deviation of sarcomere length of sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'STD sarcomere length [µm]'
    },
    'structure.sarcomere.oop': {
        'description': 'Sarcomere orientational order parameter (OOP) of all sarcomere vectors in frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere OOP'
    },
    'structure.sarcomere.orientation_mean': {
        'description': 'Mean sarcomere orientation of all sarcomere vectors in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Mean sarcomere orientation [rad]'
    },
    'structure.sarcomere.orientation': {
        'description': 'Sarcomere orientation of sarcomere vectors. List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere orientation vectors [rad]'
    },
    'structure.sarcomere.orientation_std': {
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
    'structure.zbands.ends': {
        'description': 'Position of Z-band ends in pixels.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band ends [px]'
    },
    'structure.zbands.intensity': {
        'description': 'Intensity of individual Z-band objects. List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band intensity'
    },
    'structure.zbands.intensity_mean': {
        'description': 'Mean intensity of Z-band objects. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band intensity'
    },
    'structure.zbands.intensity_std': {
        'description': 'Standard devialtion of intensity of Z-band objects. np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band intensity'
    },
    'structure.zbands.labels': {
        'description': 'Z-band labels. Image with pixel values reflecting object labels. '
                       'Stored as a sparse matrix, use labels.to_numpy() to convert to np.ndarray.',
        'data type': sparse.csr_matrix,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band labels'
    },
    'structure.zbands.lat_alignment': {
        'description': 'Lateral alignment A of pairs of adjacent Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral alignment'
    },
    'structure.zbands.lat_alignment_groups': {
        'description': 'Mean alignment of pairs of adjacent Z-bands in lateral groups. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band alignment lat. groups'
    },
    'structure.zbands.lat_alignment_groups_mean': {
        'description': 'Frame-level average of mean alignment of pairs of Z-bands in lateral groups. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean alignment in lateral Z-band groups'
    },
    'structure.zbands.lat_alignment_groups_std': {
        'description': 'Frame-level standard deviation of mean alignment in lateral groups. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD alignment in lateral Z-band groups'
    },
    'structure.zbands.lat_alignment_mean': {
        'description': 'Mean lateral alignment of adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band lateral alignment'
    },
    'structure.zbands.lat_alignment_std': {
        'description': 'Standard deviation of lateral alignment of adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band lateral alignment'
    },
    'structure.zbands.lat_dist': {
        'description': 'Distance of pairs of laterally adjacent Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral distances [µm]'
    },
    'structure.zbands.lat_dist_mean': {
        'description': 'Mean lateral distance of pairs of laterally adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band lateral distance'
    },
    'structure.zbands.lat_dist_std': {
        'description': 'Standard deviation of lateral distance of pairs of laterally adjacent Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band lateral distance'
    },
    'structure.zbands.lat_groups': {
        'description': 'Groups of laterally aligned Z-band objects. '
                       'List with lists of Z-band indices for each frame.',
        'data type': list[list[list[int]]],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral groups'
    },
    'structure.zbands.lat_length_groups': {
        'description': 'Lengths of groups of laterally aligned Z-bands. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Lengths lat. Z-band groups [µm]'
    },
    'structure.zbands.lat_length_groups_mean': {
        'description': 'Mean length of groups of laterally aligned Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean length lat. Z-band groups [µm]'
    },
    'structure.zbands.lat_length_groups_std': {
        'description': 'Standard deviation of lengths of groups of laterally aligned Z-bands. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD length lat. Z-band groups [µm]'
    },
    'structure.zbands.lat_links': {
        'description': 'Links between laterally aligned Z-band ends. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral links'
    },
    'structure.zbands.lat_neighbors': {
        'description': 'Number of lateral neighbors of each Z-band object (0, 1 or 2). '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band lateral neighbors [#]'
    },
    'structure.zbands.lat_neighbors_mean': {
        'description': 'Mean number of lateral neighbors of each Z-band object for each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band lateral neighbors [#]'
    },
    'structure.zbands.lat_neighbors_std': {
        'description': 'Standard deviation of number of lateral neighbors of each Z-band object for each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band lateral neighbors [#]'
    },
    'structure.zbands.lat_size_groups': {
        'description': 'Size of groups of laterally aligned Z-band objects. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Size of laterally aligned Z-band groups [#]'
    },
    'structure.zbands.lat_size_groups_mean': {
        'description': 'Mean size of groups of laterally aligned Z-band objects. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean size groups lat. aligned Z-band [#]'
    },
    'structure.zbands.lat_size_groups_std': {
        'description': 'Standard deviation of size of groups of laterally aligned Z-band objects. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD size lat. aligned Z-band [#]'
    },
    'structure.zbands.length': {
        'description': 'Length of Z-band objects. '
                       'List with np.ndarray of each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band length [µm]'
    },
    'structure.zbands.length_max': {
        'description': 'here description',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Max Z length [µm]'
    },
    'structure.zbands.length_mean': {
        'description': 'Mean Z-band length in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band length [µm]'
    },
    'structure.zbands.length_std': {
        'description': 'Standard deviation of Z-band lengths in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band length [µm]'
    },
    'structure.zbands.oop': {
        'description': 'Z-band orientation order parameter. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band OOP'
    },
    'structure.zbands.orientation': {
        'description': 'Orientation of individual Z-band objects. '
                       'List with np.ndarray for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band orientation [rad]'
    },
    'structure.zbands.mask_area': {
        'description': 'Total area occupied by Z-bands in each frame. '
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band mask area'
    },
    'structure.zbands.mask_area_ratio': {
        'description': 'Ratio of area occupied by Z-bands to total cell area.'
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band mask area ratio'
    },
    'structure.zbands.mask_intensity': {
        'description': 'Average intensity of Z-band mask.'
                       'np.ndarray with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band mask intensity'
    },
    'structure.zbands.straightness': {
        'description': 'Straightness of Z-band objects, measured by ratio of end-to-end length to contour length. '
                       'List with np.ndarray of each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Z-band straightness'
    },
    'structure.zbands.straightness_mean': {
        'description': 'Mean Z-band straightness for each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'Mean Z-band straightness'
    },
    'structure.zbands.straightness_std': {
        'description': 'Standard deviation of Z-band straightness for each frame. np.array with value for each frame.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_z_bands',
        'name': 'STD Z-band straightness'
    },
    # --- 2D sarcomere-vector tracking (SarcAsM.track_sarcomere_vectors) ---
    'structure.sarcomere.pos_px': {
        'description': 'Midline pixel coordinates (y, x) of each sarcomere vector, in pixels. '
                       'List with np.ndarray of shape (2, n_vectors) for each frame.',
        'data type': list[np.ndarray],
        'function': 'SarcAsM.analyze_sarcomere_vectors',
        'name': 'Sarcomere vector position [px]'
    },
    'motion.loi.data': {
        'description': 'Geometry of the detected lines of interest (LOIs): the line points, their '
                       'features, cluster assignment and the sampled LOI lines. Nested dict.',
        'data type': dict,
        'function': 'SarcAsM.detect_lois',
        'name': 'LOI geometry'
    },
    'motion.tracks.group_id': {
        'description': 'Group each track was assigned to by the current grouping, -1 for '
                       'unassigned. np.ndarray of shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.group_tracks',
        'name': 'Track group id'
    },
    'motion.tracks.group_order': {
        'description': 'Rank of the track along its group, for groupings that order their members '
                       'head-to-tail into a fibre (myofibril, loi). np.ndarray of shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.group_tracks',
        'name': 'Track order within group'
    },
    'motion.groups.kind': {
        'description': 'Grouping the tracks were grouped by: pool, mband, myofibril, domain, loi '
                       'or custom.',
        'data type': str,
        'function': 'SarcAsM.group_tracks',
        'name': 'Grouping'
    },
    'motion.groups.analyzed_kind': {
        'description': 'Grouping the motion analysis was last run on. Differs from '
                       'motion.groups.kind if the tracks were regrouped without re-running '
                       'analyze_track_motion.',
        'data type': str,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Analyzed grouping'
    },
    'motion.groups.n': {
        'description': 'Number of groups in the current grouping.',
        'data type': int,
        'function': 'SarcAsM.group_tracks',
        'name': '# Groups'
    },
    'motion.groups.member_counts': {
        'description': 'Number of tracks assigned to each group. np.ndarray of shape (n_groups,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.group_tracks',
        'name': 'Group member count'
    },
    'motion.groups.n_vectors_total': {
        'description': 'Total number of sarcomere vector observations across all tracked frames, '
                       'the denominator of the grouping coverage QC.',
        'data type': int,
        'function': 'SarcAsM.group_tracks',
        'name': '# Vectors (total)'
    },
    'motion.groups.n_vectors_in_long_tracks': {
        'description': 'Number of vector observations that fall in tracks long enough to be '
                       'grouped. Divided by motion.groups.n_vectors_total this is the fraction of '
                       'the detected sarcomeres the grouped motion analysis actually covers.',
        'data type': int,
        'function': 'SarcAsM.group_tracks',
        'name': '# Vectors (in long tracks)'
    },
    'motion.groups.track_ids': {
        'description': 'Track ids the current grouping was built from, kept so a grouped result '
                       'read after re-tracking can be detected as stale. np.ndarray (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.group_tracks',
        'name': 'Grouping track ids'
    },
    'motion.groups.hash': {
        'description': 'Fingerprint of the tracks and grouping recipe, used to hard-raise when a '
                       'grouped motion result is read after its grouping changed.',
        'data type': str,
        'function': 'SarcAsM.group_tracks',
        'name': 'Grouping fingerprint'
    },
    'motion.tracks.n': {
        'description': 'Number of sarcomere query-point tracks kept after the '
                       'min_track_duration_s filter.',
        'data type': int,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Number of tracks'
    },
    'motion.tracks.ids': {
        'description': 'Internal slot id of each kept track. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track ids'
    },
    'motion.tracks.start_frame': {
        'description': 'Frame at which each track first appears. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track start frame'
    },
    'motion.tracks.n_frames': {
        'description': 'Number of frames in which each track was matched to a real detection. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track length [frames]'
    },
    'motion.tracks.drift_um': {
        'description': 'Drift of each track relative to the coherent motion of its local '
                       'neighbourhood (µm); ~one sarcomere length indicates a changed identity. '
                       'NaN if too short to score. np.ndarray, shape (n_tracks,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track drift [µm]'
    },
    'motion.tracks.positions_um': {
        'description': 'Per-track sarcomere-centre positions (y, x) in µm. np.ndarray, shape (n_tracks, T, 2); '
                       'NaN before start / after close.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track positions [µm]'
    },
    'motion.tracks.positions_px': {
        'description': 'Per-track sarcomere-centre positions (y, x) in pixels. np.ndarray, shape (n_tracks, T, 2).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track positions [px]'
    },
    'motion.tracks.slen': {
        'description': 'Per-track sarcomere length over time in µm. np.ndarray, shape (n_tracks, T); '
                       'NaN on gap (unobserved) frames. The core per-sarcomere length-vs-time signal.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track sarcomere length [µm]'
    },
    'motion.tracks.orientations': {
        'description': 'Per-track sarcomere orientation over time in radians. np.ndarray, shape (n_tracks, T); '
                       'NaN on gap frames.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track orientation [rad]'
    },
    'motion.tracks.observed': {
        'description': 'Boolean mask, True where a real detection was matched (vs a predicted gap frame). '
                       'np.ndarray, shape (n_tracks, T).',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track observed mask'
    },
    'motion.tracks.detection_id': {
        'description': 'Index of the matched detection into pos_vectors_px of that frame (-1 on gap frames). '
                       'np.ndarray, shape (n_tracks, T). Joins a track back to the per-frame vector analysis.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track detection id'
    },
    'motion.tracks.midline_id': {
        'description': 'M-band (midline) id of the matched detection per frame (-1 on gap frames). '
                       'np.ndarray, shape (n_tracks, T). Basis for M-band-level grouping.',
        'data type': np.ndarray,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track M-band id'
    },
    'motion.tracks.fragmentation_ratio': {
        'description': 'Number of tracks divided by the median number of sarcomere vectors per '
                       'frame. 1.0 means one track per vector across the whole recording; larger '
                       'values mean the same vector was split into that many trajectories. The '
                       'headline tracking-continuity quality number.',
        'data type': float,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Track fragmentation ratio'
    },
    'motion.tracks.n_interpolated_gap_frames': {
        'description': 'Number of (track, frame) entries whose sarcomere length / orientation was '
                       'filled by interpolation across a short interior gap '
                       '(max_gap_interpolation_s). These frames stay False in tracks_observed, so no '
                       'coverage or real-observation metric counts them.',
        'data type': int,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Number of interpolated gap frames'
    },
    'motion.tracks.n_retired': {
        'description': 'Number of tracks closed because they went unmatched for longer than '
                       'retire_after_s (0 with the default, where tracks never retire).',
        'data type': int,
        'function': 'SarcAsM.track_sarcomere_vectors',
        'name': 'Number of retired tracks'
    },
}

motion_feature_dict = {
    # Track-based grouped motion, one value (or row) per group under motion.<kind>.<suffix>;
    # SarcAsM.analyze_track_motion writes the same members for every grouping kind.
    'beating_rate': {
        'description': 'Beating rate of the group (Hz); 1 / mean inter-beat interval. np.ndarray with shape '
                       '(n_groups,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group beating rate [Hz]'},
    'beating_rate_variability': {
        'description': 'Standard deviation of inter-beat intervals of the group (s). np.ndarray with shape '
                       '(n_groups,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group beating rate variability [s]'},
    'n_contr': {
        'description': 'Number of detected contraction cycles in the group, including cycles that are incomplete '
                       'at the start/end of the recording. np.ndarray with shape (n_groups,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group number of contractions'},
    'n_contr_complete': {
        'description': 'Number of complete contraction cycles in the group (onset and offset both inside the '
                       'recording). Only these back the timing features below. np.ndarray with shape '
                       '(n_groups,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group number of complete contractions'},
    'contr_complete': {
        'description': "Fraction of the group's contraction cycles that are complete (per-cycle flag, averaged "
                       'over cycles on export). np.ndarray with shape (n_groups, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group contraction completeness'},
    'equ': {
        'description': 'Equilibrium (resting) sarcomere length of the group (µm). np.ndarray with shape '
                       '(n_groups,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group equilibrium sarcomere length [µm]'},
    'contr_max': {
        'description': 'Maximal contraction (shortening) per cycle, mean over cycles (µm). np.ndarray with shape '
                       '(n_groups, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group max contraction [µm]'},
    'elong_max': {
        'description': 'Maximal elongation per cycle, mean over cycles (µm). np.ndarray with shape (n_groups, '
                       'max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group max elongation [µm]'},
    'vel_contr_max': {
        'description': 'Maximal shortening velocity per cycle, mean over cycles (µm/s). np.ndarray with shape '
                       '(n_groups, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group max shortening velocity [µm/s]'},
    'vel_elong_max': {
        'description': 'Maximal elongation velocity per cycle, mean over cycles (µm/s). np.ndarray with shape '
                       '(n_groups, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group max elongation velocity [µm/s]'},
    'time_to_peak': {
        'description': 'Time from contraction onset to maximum shortening, mean over cycles (s). NaN for a cycle '
                       'whose onset falls outside the recording, so only complete-at-the-start cycles '
                       'contribute. np.ndarray with shape (n_groups, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group time to peak [s]'},
    'time_to_relax': {
        'description': 'Time from peak shortening to end of contraction, mean over cycles (s). NaN for a cycle '
                       'whose offset falls outside the recording, so only complete-at-the-end cycles contribute. '
                       'np.ndarray with shape (n_groups, max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group time to relaxation [s]'},
    'time_contr': {
        'description': 'Total contraction duration per cycle, mean over cycles (s). Only complete cycles '
                       'contribute — a cycle truncated by the start or end of the recording has no measurable '
                       'duration and is NaN (see n_contr_complete). np.ndarray with shape (n_groups, '
                       'max_n_contr).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group contraction duration [s]'},
    'slen': {
        'description': 'Aggregated per-group sarcomere length over time, shape (n_groups, T).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group sarcomere length time-series [µm]'},
    'contr': {
        'description': 'Binary contraction state for each group over time. np.ndarray with shape (n_groups, '
                       'n_frames). True = contracting, False = quiescent.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group contraction state'},
    'labels_contr': {
        'description': 'Contraction cycle labels for each group over time. np.ndarray with shape (n_groups, '
                       'n_frames). Values 1, 2, 3, ... label each contraction cycle.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group contraction labels'},
    'n_members': {
        'description': 'Time-series of number of sarcomere vectors within each group. np.ndarray with shape '
                       '(n_groups, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group vector count time-series'},
    'slen_median': {
        'description': 'Time-series of median sarcomere length within each sarcomere group. np.ndarray with '
                       'shape (n_groups, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group sarcomere length median time-series [µm]'},
    'slen_q25': {
        'description': 'Time-series of 25th percentile of sarcomere length within each group. np.ndarray with '
                       'shape (n_groups, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group sarcomere length Q25 time-series [µm]'},
    'slen_q75': {
        'description': 'Time-series of 75th percentile of sarcomere length within each group. np.ndarray with '
                       'shape (n_groups, n_frames).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Group sarcomere length Q75 time-series [µm]'},
    'slen_std': {
        'description': 'Standard deviation of sarcomere length over the members of each group, per frame '
                       '(the within-group spread of SL). np.ndarray with shape (n_groups, n_frames); '
                       'exported as its time mean.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Within-group SL std [µm]'},
    'equ_std': {
        'description': "Standard deviation of the members' resting sarcomere lengths (µm): how much the "
                       'sarcomeres of a group differ in resting length (static heterogeneity). np.ndarray '
                       'with shape (n_groups,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Resting SL std across members [µm]'},
    'corr_delta_slen_serial': {
        'description': 'Serial correlation r_s of dSL: mean Pearson correlation of the same sarcomere between '
                       'different contraction cycles (cycle-to-cycle consistency), per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Serial corr. dSL'},
    'corr_delta_slen_mutual': {
        'description': 'Mutual correlation r_m of dSL: mean Pearson correlation of different sarcomeres within '
                       'the same contraction cycle (synchrony), per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Mutual corr. dSL'},
    'ratio_delta_slen_mutual_serial': {
        'description': 'R = r_m / r_s of dSL, per group. R < 1: sarcomeres differ consistently (static '
                       'heterogeneity); R ~ 1: they differ randomly from beat to beat (stochastic). NaN when r_s '
                       '<= 0. Meaningful for groups of distinct sarcomeres (fibre chains, domains, pool).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'R dSL (mutual/serial)'},
    'corr_vel_serial': {
        'description': 'Serial correlation r_s of the sarcomere velocity, per group (see '
                       'corr_delta_slen_serial).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Serial corr. velocity'},
    'corr_vel_mutual': {
        'description': 'Mutual correlation r_m of the sarcomere velocity, per group (see '
                       'corr_delta_slen_mutual).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Mutual corr. velocity'},
    'ratio_vel_mutual_serial': {
        'description': 'R = r_m / r_s of the sarcomere velocity, per group (see ratio_delta_slen_mutual_serial).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'R velocity (mutual/serial)'},
    'corr_n_cycles': {
        'description': 'Number of contraction cycles entering the serial/mutual correlation, per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Correlation cycles'},
    'oscill_frequencies': {
        'description': 'Frequencies (Hz) of the wavelet oscillation spectrum, shared by all groups. np.ndarray '
                       'with shape (num_scales,).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Oscillation frequencies [Hz]'},
    'oscill_magnitudes_avg': {
        'description': 'Wavelet magnitude spectrum of the group-mean dSL over the contracting frames. np.ndarray '
                       'with shape (n_groups, num_scales).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Oscillation spectrum (mean dSL)'},
    'oscill_magnitudes_single': {
        'description': 'Wavelet magnitude spectrum of the individual sarcomeres, mean over members, over the '
                       'contracting frames. np.ndarray with shape (n_groups, num_scales).',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Oscillation spectrum (single sarcomeres)'},
    'oscill_peak_avg': {
        'description': 'Frequency (Hz) of the strongest component of the group-mean dSL spectrum, per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Peak frequency (mean dSL) [Hz]'},
    'oscill_amp_avg': {
        'description': 'Magnitude of the strongest component of the group-mean dSL spectrum, per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Peak magnitude (mean dSL)'},
    'oscill_peak_1_single': {
        'description': 'Beating-frequency peak (Hz) of the single-sarcomere spectrum, per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Single-sarcomere beat peak [Hz]'},
    'oscill_amp_1_single': {
        'description': 'Magnitude of the beating peak of the single-sarcomere spectrum, per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Single-sarcomere beat magnitude'},
    'oscill_peak_2_single': {
        'description': 'High-frequency peak (Hz) of the single-sarcomere spectrum above the beating band '
                       '(strongest local maximum), per group; NaN when the spectrum only decays there.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Single-sarcomere high-freq. peak [Hz]'},
    'oscill_amp_2_single': {
        'description': 'Magnitude of the high-frequency peak of the single-sarcomere spectrum, per group.',
        'data type': np.ndarray,
        'function': 'SarcAsM.analyze_track_motion',
        'name': 'Single-sarcomere high-freq. magnitude'},
}


#: Grouping levels that :meth:`sarcasm.SarcAsM.analyze_track_motion` can write.
#: Motion result keys are ``'<kind>_<suffix>'`` with the suffix documented in
#: :data:`motion_feature_dict`.
MOTION_KINDS = ('pool', 'mband', 'myofibril', 'domain', 'loi', 'custom')


def describe_key(key: str, registry: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up the registry entry for a result key.

    Resolves the three shapes a key can have: an exact path documented in
    :data:`structure_feature_dict`, a per-group motion feature
    ``motion.<kind>.<suffix>`` documented once by suffix in
    :data:`motion_feature_dict`, and ``params.<step>.<name>``.

    Parameters
    ----------
    key : str
        Result key as used in ``sarc.data``, e.g. ``'structure.sarcomere.oop'``
        or ``'motion.pool.beating_rate'``.
    registry : {'structure', 'motion'} or None, optional
        Restrict the lookup to one registry. ``'motion'`` additionally accepts a
        bare suffix (``'beating_rate'``), which is how the GUI's curated tier
        lists name them. Default is None, which resolves the key by its shape.

    Returns
    -------
    dict or None
        The registry entry with an added ``'registry'`` field
        (``'structure'``, ``'motion'`` or ``'params'``), plus ``'kind'`` and
        ``'suffix'`` for per-group motion features. None if undocumented.
    """
    if registry == 'motion' and key in motion_feature_dict:
        return {**motion_feature_dict[key], 'registry': 'motion',
                'kind': None, 'suffix': key}
    if registry in (None, 'structure') and key in structure_feature_dict:
        return {**structure_feature_dict[key], 'registry': 'structure'}
    if registry in (None, 'motion'):
        head, _, rest = key.partition('.')
        kind, _, suffix = rest.partition('.')
        if head == 'motion' and kind in MOTION_KINDS and suffix in motion_feature_dict:
            return {**motion_feature_dict[suffix], 'registry': 'motion',
                    'kind': kind, 'suffix': suffix}
    if registry is None and key.startswith('params.'):
        parts = key.split('.')
        step = parts[1] if len(parts) > 1 else ''
        return {'description': f'Analysis parameter of SarcAsM.{step}(), recorded when the '
                               f'step was run. See the method docstring for its meaning.',
                'data type': None, 'function': f'SarcAsM.{step}', 'name': parts[-1],
                'registry': 'params', 'step': step}
    return None


def pretty_name(key: str, registry: Optional[str] = None) -> str:
    """Human-readable display label for a result key, falling back to the key.

    Parameters
    ----------
    key : str
        Flat result key.
    registry : {'structure', 'motion'} or None, optional
        Passed through to :func:`describe_key`. Default is None.

    Returns
    -------
    str
        The registry display name, or ``key`` itself if undocumented.
    """
    entry = describe_key(key, registry)
    return entry.get('name', key) if entry else key
