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

"""Kymograph generation module."""

from multiprocessing import Pool
import numpy as np
from scipy import ndimage


def kymograph_movie(movie: np.ndarray, line: np.ndarray, linewidth: int = 10, order: int = 0):
    """
    Generate a kymograph using multiprocessing.

    Parameters
    --------
    movie : np.ndarray, shape (N, H, W)
        The movie.
    line : np.ndarray, shape (N, 2)
        The coordinates of the segmented line (N>1)
    linewidth : int, optional
        Width of the scan in pixels, perpendicular to the line
    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Return
    ---------
    return_value : ndarray
        Kymograph along segmented line

    Notes
    -------
    Adapted from scikit-image
    (https://scikit-image.org/docs/0.22.x/api/skimage.measure.html#skimage.measure.profile_line).
    """
    # prepare coordinates of segmented line
    perp_lines = curved_line_profile_coordinates(points=line, linewidth=linewidth)

    # Prepare arguments for each frame
    args = [(movie[frame], perp_lines, linewidth, order) for frame in range(movie.shape[0])]

    # Create a Pool and map process_frame to each frame
    with Pool() as pool:
        results = pool.map(process_frame, args)

    # Convert list of results to a numpy array
    kymograph = np.array(results)

    return kymograph


def process_frame(args):
    """Process a single frame for kymograph generation."""
    frame, perp_lines, linewidth, order = args
    pixels = ndimage.map_coordinates(frame, perp_lines, prefilter=order > 1,
                                     order=order, mode='reflect', cval=0.0)
    pixels = np.flip(pixels, axis=1)
    intensities = np.mean(pixels, axis=1)
    return intensities


def curved_line_profile_coordinates(points: np.ndarray, linewidth: int = 10):
    """
    Calculate the coordinates of a curved line profile composed of multiple segments with specified linewidth.

    Parameters
    ----------
    points : np.ndarray
        A list of points (y, x) defining the segments of the curved line.
    linewidth : int, optional
        The width of the line in pixels.

    Returns
    -------
    coords : ndarray
        The coordinates of the curved line profile. Shape is (2, N, linewidth),
        where N is the total number of points in the line.
    """
    all_perp_rows = []
    all_perp_cols = []

    for i in range(len(points) - 1):
        src, dst = np.asarray(points[i], dtype=float), np.asarray(points[i + 1], dtype=float)
        d_row, d_col = dst - src
        theta = np.arctan2(d_row, d_col)
        length = int(np.ceil(np.hypot(d_row, d_col) + 1))
        line_col = np.linspace(src[1], dst[1], length)
        line_row = np.linspace(src[0], dst[0], length)
        col_width, row_width = (linewidth - 1) * np.sin(-theta) / 2, (linewidth - 1) * np.cos(theta) / 2
        perp_rows = np.stack([np.linspace(row - row_width, row + row_width, linewidth) for row in line_row])
        perp_cols = np.stack([np.linspace(col - col_width, col + col_width, linewidth) for col in line_col])

        all_perp_rows.append(perp_rows)
        all_perp_cols.append(perp_cols)

    # Concatenate all segments
    final_perp_rows = np.concatenate(all_perp_rows, axis=0)
    final_perp_cols = np.concatenate(all_perp_cols, axis=0)

    return np.stack([final_perp_rows, final_perp_cols])
