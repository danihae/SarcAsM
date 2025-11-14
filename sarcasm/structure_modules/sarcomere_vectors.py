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

"""Sarcomere vector extraction and analysis module."""

from typing import Tuple, Union, List
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize
from joblib import Parallel, delayed

from sarcasm.utils import Utils


def get_sarcomere_vectors(
        zbands: np.ndarray,
        mbands: np.ndarray,
        orientation_field: np.ndarray,
        pixelsize: float,
        median_filter_radius: float = 0.25,
        slen_lims: Tuple[float, float] = (1, 3),
        interp_factor: int = 4,
        linewidth: float = 0.3,
        backend: str = 'loky',
) -> Tuple[Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List],
Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List]]:
    """
    Extract sarcomere orientation and length vectors.

    Parameters
    ----------
    zbands : np.ndarray
        2D array representing the semantic segmentation map of Z-bands.
    mbands : np.ndarray
        2D array representing the semantic segmentation map of mbands.
    orientation_field : np.ndarray
        2D array representing the orientation field.
    pixelsize : float
        Size of a pixel in micrometers.
    median_filter_radius : float, optional
        Radius of kernel to smooth orientation field before assessing orientation at M-points, in µm (default 0.25 µm).
    slen_lims : tuple of float, optional
        Sarcomere size limits in micrometers (default is (1, 3)).
    interp_factor : int, optional
        Interpolation factor for profiles to calculate sarcomere length. Defaults to 4.
    linewidth : float, optional
        Line width of profiles to calculate sarcomere length. Defaults to 0.3 µm.

    Returns
    -------
    pos_vectors : np.ndarray
        Array of position vectors for sarcomeres.
    sarcomere_orientation_vectors : np.ndarray
        Sarcomere orientation values at midline points.
    sarcomere_length_vectors : np.ndarray
        Sarcomere length values at midline points.
    sarcomere_mask : np.ndarray
        Mask indicating the presence of sarcomeres.
    """
    radius_pixels = max(int(round(median_filter_radius / pixelsize, 0)), 1)
    linewidth_pixels = max(int(round(linewidth / pixelsize, 0)), 1)

    # skeletonize mbands
    mbands_skel = skeletonize(mbands, method='lee')

    # calculate and preprocess orientation map
    orientation = Utils.get_orientation_angle_map(orientation_field, use_median_filter=True, radius=radius_pixels)

    # label mbands
    midline_labels, n_mbands = ndimage.label(mbands_skel,
                                               ndimage.generate_binary_structure(2, 2))

    # iterate mbands and create an additional list with labels and midline length (approx. by max. Feret diameter)
    props = measure.regionprops_table(midline_labels, properties=['label', 'coords', 'feret_diameter_max'])
    list_labels, coords_mbands, length_mbands = (props['label'], props['coords'],
                                                     props['feret_diameter_max'] * pixelsize)

    pos_vectors_px, pos_vectors, midline_id_vectors, midline_length_vectors = [], [], [], []
    if n_mbands > 0:
        for i, (label_i, coords_i, length_midline_i) in enumerate(
                zip(list_labels, coords_mbands, length_mbands)):
            pos_vectors_px.append(coords_i)
            midline_length_vectors.append(np.ones(coords_i.shape[0]) * length_midline_i)
            midline_id_vectors.append(np.ones(coords_i.shape[0]) * label_i)

        pos_vectors_px = np.concatenate(pos_vectors_px, axis=0)
        midline_id_vectors = np.concatenate(midline_id_vectors)
        midline_length_vectors = np.concatenate(midline_length_vectors)

        sarcomere_orientation_vectors = orientation[pos_vectors_px[:, 0], pos_vectors_px[:, 1]]

        ends1 = pos_vectors_px.T + (slen_lims[1] * 1.3) / 2 / pixelsize * np.array(
            (np.sin(sarcomere_orientation_vectors), np.cos(sarcomere_orientation_vectors))
        )
        ends2 = pos_vectors_px.T - (slen_lims[1] * 1.3) / 2 / pixelsize * np.array(
            (np.sin(sarcomere_orientation_vectors), np.cos(sarcomere_orientation_vectors))
        )

        # Calculate sarcomere lengths by measuring peak-to-peak distance of Z-bands in intensity profile
        profiles = Utils.fast_profile_lines(zbands, ends1, ends2, linewidth=linewidth_pixels)

        # Use parallel processing for faster execution
        results = Parallel(n_jobs=-1, backend=backend)(
            delayed(Utils.process_profile)(
                profile, pixelsize, slen_lims=slen_lims, interp_factor=interp_factor
            ) for profile in profiles
        )

        # Convert results to array
        sarcomere_length_vectors, center_offsets = np.array(results).T

        # get vector positions in µm and correct center of vectors
        pos_vectors = pos_vectors_px * pixelsize
        offset_vectors = np.stack((np.sin(sarcomere_orientation_vectors) * center_offsets,
                                  np.cos(sarcomere_orientation_vectors) * center_offsets), axis=-1)
        pos_vectors -= offset_vectors

        # remove NaNs
        nan_mask = np.isnan(sarcomere_length_vectors)
        pos_vectors_px = pos_vectors_px[~nan_mask]
        pos_vectors = pos_vectors[~nan_mask]
        midline_id_vectors = midline_id_vectors[~nan_mask]
        sarcomere_orientation_vectors = sarcomere_orientation_vectors[~nan_mask]
        sarcomere_length_vectors = sarcomere_length_vectors[~nan_mask]


    else:
        sarcomere_length_vectors, _z_band_thickness_vectors, sarcomere_orientation_vectors = [], [], []

    return (pos_vectors_px, pos_vectors, midline_id_vectors, midline_length_vectors, sarcomere_length_vectors,
            sarcomere_orientation_vectors, n_mbands)
