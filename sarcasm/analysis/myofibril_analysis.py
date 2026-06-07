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

"""Myofibril line detection and analysis module."""

import logging
from typing import Union
from collections import deque
import random
import numpy as np
from scipy.spatial import cKDTree

from sarcasm.utils import Utils

logger = logging.getLogger(__name__)


def grow_line(seed, points_t, sarcomere_length_vectors_t, sarcomere_orientation_vectors_t, tree,
              threshold_distance, pixelsize, persistence):
    """
    Grow a single line from a seed point.

    Parameters
    ----------
    seed : int
        Index of the seed point.
    points_t : np.ndarray
        Point coordinates.
    sarcomere_length_vectors_t : np.ndarray
        Sarcomere lengths at each point (µm).
    sarcomere_orientation_vectors_t : np.ndarray
        Sarcomere orientations at each point (radians).
    tree : scipy.spatial.cKDTree
        cKDTree for nearest-neighbor queries.
    threshold_distance : float
        Maximum neighbor search distance in µm.
    pixelsize : float
        Pixel size in µm.
    persistence : int
        Number of points averaged for length and orientation.

    Returns
    -------
    np.ndarray
        Indices of the points forming the line.
    """
    line_i = deque([seed])
    stop_right = stop_left = False

    sarcomere_orientation_vectors_t = sarcomere_orientation_vectors_t + np.pi / 2

    threshold_distance_pixels = threshold_distance / pixelsize

    def angle_diff(alpha, beta):
        """Return the signed difference between angles alpha and beta.
        The result is in the range [-pi, pi]."""
        return np.arctan2(np.sin(beta - alpha), np.cos(beta - alpha))

    def calculate_mean_orientation(orientations):
        """Return the circular mean of axial orientations (period pi), in radians."""
        # Convert orientations to complex numbers on the unit circle
        complex_orientations = np.exp(2j * np.array(orientations))
        # Calculate the mean of the complex numbers
        mean_complex = np.mean(complex_orientations)
        # Convert back to angle and halve it to get the original range
        return np.angle(mean_complex) / 2

    def adjust_orientation(current_orientation, previous_orientation):
        """Flip the current orientation by pi so it stays consistent with the previous one."""
        diff = angle_diff(current_orientation, previous_orientation)
        if diff > np.pi / 2:
            return current_orientation - np.pi
        elif diff < -np.pi / 2:
            return current_orientation + np.pi
        return current_orientation

    # Initialize orientations
    orientation_left = orientation_right = sarcomere_orientation_vectors_t[seed]

    points_t = points_t.T

    while not stop_left or not stop_right:
        line_i_list = list(line_i)

        if not stop_left:
            end_left = points_t[:, line_i_list[0]]
            length_left = np.mean(sarcomere_length_vectors_t[line_i_list[:persistence]]) / pixelsize
            new_orientation_left = calculate_mean_orientation(
                sarcomere_orientation_vectors_t[line_i_list[:persistence]]) if persistence > 1 else sarcomere_orientation_vectors_t[line_i_list[0]]
            orientation_left = adjust_orientation(new_orientation_left, orientation_left)

        if not stop_right:
            end_right = points_t[:, line_i_list[-1]]
            length_right = np.mean(sarcomere_length_vectors_t[line_i_list[-persistence:]]) / pixelsize
            new_orientation_right = calculate_mean_orientation(
                sarcomere_orientation_vectors_t[line_i_list[-persistence:]]) if persistence > 1 else sarcomere_orientation_vectors_t[line_i_list[-1]]
            orientation_right = adjust_orientation(new_orientation_right, orientation_right)

        # grow left
        if not stop_left:
            prior_left = (end_left[0] + np.cos(orientation_left) * length_left,
                          end_left[1] - np.sin(orientation_left) * length_left)
            distance_left, index_left = tree.query(prior_left, k=1)
            if distance_left < threshold_distance_pixels:
                line_i.appendleft(int(index_left))
            else:
                stop_left = True

        # grow right
        if not stop_right:
            prior_right = (end_right[0] - np.cos(orientation_right) * length_right,
                           end_right[1] + np.sin(orientation_right) * length_right)
            distance_right, index_right = tree.query(prior_right, k=1)
            if distance_right < threshold_distance_pixels:
                line_i.append(int(index_right))
            else:
                stop_right = True

    return np.asarray(line_i)


def line_growth(points_t: np.ndarray, sarcomere_length_vectors_t: np.ndarray,
                sarcomere_orientation_vectors_t: np.ndarray,
                midline_length_vectors_t: np.ndarray, pixelsize: float, ratio_seeds: float = 0.1,
                persistence: int = 4, threshold_distance: float = 0.3, n_min: int = 5,
                random_seed: Union[None, int] = None):
    """
    Grow myofibril lines along the sarcomere axis (perpendicular to z-bands) from random seeds.

    Parameters
    ----------
    points_t : np.ndarray
        Midline point positions.
    sarcomere_length_vectors_t : np.ndarray
        Sarcomere lengths at midline points (µm).
    sarcomere_orientation_vectors_t : np.ndarray
        Sarcomere orientation angles at midline points (radians).
    midline_length_vectors_t : np.ndarray
        Lengths of sarcomere m-bands at midline points (µm).
    pixelsize : float
        Pixel size in µm.
    ratio_seeds : float, optional
        Fraction of sarcomere vectors used as seeds for line growth. Default is 0.1.
    persistence : int, optional
        Number of points averaged for length and orientation. Default is 4.
    threshold_distance : float, optional
        Maximum neighbor search distance in µm during growth. Default is 0.3.
    n_min : int, optional
        Minimum number of points; shorter lines are discarded. Default is 5.
    random_seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    dict
        Line data with keys 'lines' (list of point-index arrays) and 'line_features'
        (dict of per-line feature arrays).
    """
    # select random origins for line growth
    points_t = np.asarray(points_t)
    if points_t.shape[0] == 2:
        points_t = points_t.T

    if len(points_t) == 0:
        logger.warning('No sarcomeres in image (len(points) = 0), could not grow lines.')
        return {'lines': [], 'line_features': {}}

    if random_seed:
        random.seed(random_seed)
    n_vectors = len(points_t)
    seed_idx = random.sample(range(n_vectors), max(1, int(ratio_seeds * n_vectors)))

    # Precompute nearest-neighbor tree (scipy cKDTree — faster single-query than sklearn BallTree)
    tree = cKDTree(points_t)

    # Prepare arguments for parallel processing
    args = [
        (seed, points_t, sarcomere_length_vectors_t, sarcomere_orientation_vectors_t, tree, threshold_distance,
         pixelsize, persistence) for seed in seed_idx]

    # grow lines
    lines = [grow_line(*arg) for arg in args]

    # remove short lines (< n_min)
    lines = [l for l in lines if len(l) >= n_min]

    # calculate features of lines
    n_vectors_lines = np.asarray([len(l) for l in lines])  # number of sarcomeres in line
    length_line_segments = [sarcomere_length_vectors_t[l] for l in lines]
    length_lines = [np.sum(lengths) for lengths in length_line_segments]

    # sarcomere lengths
    sarcomere_mean_length_lines = [np.mean(sarcomere_length_vectors_t[l]) for l in lines]
    sarcomere_std_length_lines = [np.std(sarcomere_length_vectors_t[l]) for l in lines]

    # midline lengths
    midline_mean_length_lines = [np.nanmean(midline_length_vectors_t[l]) for l in lines]
    midline_std_length_lines = [np.nanstd(midline_length_vectors_t[l]) for l in lines]
    midline_min_length_lines = [np.nanmin(midline_length_vectors_t[l]) for l in lines]

    # Straightness
    def frechet_straightness(points):
        """
        Compute a Fréchet-inspired straightness measure.

        Defined as ``1 - (max perpendicular deviation from chord / chord length)``.

        Parameters
        ----------
        points : np.ndarray
            Polyline vertices, shape ``(n_points, 2)``.

        Returns
        -------
        float
            Straightness measure (1 = perfectly straight).
        """

        if len(points) < 2:
            return 1.0  # Single point is trivially straight

        # Calculate chord vector between first and last points
        chord_vector = points[-1] - points[0]
        chord_length = np.linalg.norm(chord_vector)

        if chord_length < 1e-9:  # Handle degenerate chord
            return 0.0

        # Unit vector along chord direction
        unit_chord = chord_vector / chord_length

        # Vectors from first point to each polyline vertex
        displacement_vectors = points - points[0]

        # Scalar projections onto chord (dot product with unit vector)
        chord_projections = np.sum(displacement_vectors * unit_chord, axis=1)

        # Ideal points along chord line
        projected_points = points[0] + chord_projections[:, np.newaxis] * unit_chord

        # Perpendicular deviations from actual path
        deviation_vectors = points - projected_points
        perpendicular_deviations = np.linalg.norm(deviation_vectors, axis=1)

        max_deviation = np.max(perpendicular_deviations)

        return 1.0 - (max_deviation / chord_length)

    straightness_lines = [
        frechet_straightness(points_t[line])
        for line in lines
    ]

    # Bending: mean squared angular change
    tangential_vector_line_segments = [np.diff(points_t[l], axis=0) for l in lines]
    tangential_angle_line_segments = [np.asarray([np.arctan2(v[1], v[0]) for v in vectors]) for vectors in
                                      tangential_vector_line_segments]
    bending_lines = [
        np.mean(np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles))) ** 2) if len(angles) > 1 else 0.0
        for angles in tangential_angle_line_segments
    ]

    # create dictionary
    line_features = {'n_vectors_lines': n_vectors_lines, 'length_lines': length_lines,
                     'sarcomere_mean_length_lines': sarcomere_mean_length_lines,
                     'sarcomere_std_length_lines': sarcomere_std_length_lines,
                     'bending_lines': bending_lines,
                     'straightness_lines': straightness_lines,
                     'midline_mean_length_lines': midline_mean_length_lines,
                     'midline_std_length_lines': midline_std_length_lines,
                     'midline_min_length_lines': midline_min_length_lines}
    line_features = Utils.convert_lists_to_arrays_in_dict(line_features)
    line_data = {'lines': lines, 'line_features': line_features}
    return line_data


def create_myofibril_length_map(
        myof_lines: np.ndarray,
        myof_length: np.ndarray,
        pos_vectors: np.ndarray,
        sarcomere_orientation_vectors: np.ndarray,
        sarcomere_length_vectors: np.ndarray,
        size: tuple,
        pixelsize: float,
        median_filter_radius: float = 0.6,
) -> np.ndarray:
    """
    Generate a 2D spatial map of myofibril lengths.

    Rasterizes myofibril line segments, assigns each line's length to the pixels it
    covers, averages overlapping pixels, and optionally smooths with a median filter.

    Parameters
    ----------
    myof_lines : np.ndarray
        Per-line point indices for myofibril structures.
    myof_length : np.ndarray
        Length value for each myofibril line (µm).
    pos_vectors : np.ndarray
        Position vectors in µm.
    sarcomere_orientation_vectors : np.ndarray
        Orientation angles in radians.
    sarcomere_length_vectors : np.ndarray
        Sarcomere lengths in µm.
    size : tuple of int
        Output map dimensions ``(height, width)`` in pixels.
    pixelsize : float
        Pixel size in µm.
    median_filter_radius : float, optional
        Median filter radius in µm. Default is 0.6.

    Returns
    -------
    np.ndarray
        2D map of myofibril lengths (µm), with NaN in empty regions.
    """
    from skimage.draw import line
    
    # Convert median filter radius to pixels
    median_radius_px = int(round(median_filter_radius / pixelsize))

    # Initialize accumulation maps
    length_sum_map = np.zeros(size, dtype=np.float32)
    weight_map = np.zeros(size, dtype=np.float32)

    # Process each myofibril segment
    for line_idx, line_length in zip(myof_lines, myof_length):
        # Extract vector data for current line
        points = pos_vectors[line_idx]
        orientations = sarcomere_orientation_vectors[line_idx] + np.pi / 2
        lengths = sarcomere_length_vectors[line_idx]

        # Calculate direction vectors
        dir_x = np.cos(orientations)
        dir_y = -np.sin(orientations)
        directions = np.vstack([dir_x, dir_y])

        # Calculate endpoints in pixel coordinates
        end_offset = directions * lengths / 2
        end_points = np.stack([
            (points.T + end_offset) / pixelsize,
            (points.T - end_offset) / pixelsize
        ]).astype(np.int32)

        # Rasterize lines
        for (x0, y0), (x1, y1) in zip(end_points[0].T, end_points[1].T):
            rr, cc = line(x0, y0, x1, y1)
            # Apply boundary constraints
            valid = (rr >= 0) & (rr < size[0]) & (cc >= 0) & (cc < size[1])
            np.add.at(length_sum_map, (rr[valid], cc[valid]), line_length)
            np.add.at(weight_map, (rr[valid], cc[valid]), 1)

    # Calculate weighted average
    myof_map = np.divide(length_sum_map, weight_map,
                         out=np.full_like(length_sum_map, np.nan),
                         where=weight_map > 0)

    # Apply median filtering if required
    if median_radius_px > 0:
        window_size = 2 * median_radius_px + 1
        myof_map = Utils.nanmedian_filter_numba(myof_map, window_size)

    return myof_map
