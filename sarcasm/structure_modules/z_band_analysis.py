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

"""Z-band segmentation and analysis module."""

from typing import Tuple
import numpy as np
import networkx as nx
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform, pdist
from skimage import segmentation, morphology
from skimage.measure import label, regionprops_table
from skimage.morphology import skeletonize

from sarcasm.utils import Utils


def segment_z_bands(image: np.ndarray, threshold: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment z-bands from U-Net result (threshold, make binary, skeletonize, label regions).

    Parameters
    ----------
    image : np.ndarray
        Input image from U-Net.
    threshold : float, optional
        Threshold value for binarizing the image. Defaults to 0.15.

    Returns
    -------
    labels : np.ndarray
        Labeled regions in the thresholded image.
    labels_skel : np.ndarray
        Labeled regions in the skeletonized image.
    """
    mask = image > threshold
    mask_skel = morphology.skeletonize(mask, method='lee')
    labels = label(mask)
    labels_skel = mask_skel * labels
    return labels, labels_skel


def analyze_z_bands(zbands: np.ndarray, labels: np.ndarray, labels_skel: np.ndarray,
                    image_raw: np.ndarray, orientation_field: np.ndarray,
                    pixelsize: float, min_length: float = 1.0, threshold: float = 0.5,
                    median_filter_radius: float = 0.25,
                    a_min: float = 0.3, theta_phi_min: float = 0.2, d_max: float = 4.0,
                    d_min: float = 0.25) -> Tuple:
    """
    Analyzes segmented z-bands in a single frame, extracting metrics such as length, intensity, orientation,
    straightness, lateral distance, alignment, number of lateral neighbors per z-band, and characteristics of
    groups of lateral z-bands (length, alignment, size).

    Parameters
    ----------
    zbands : np.ndarray
        The segmented map of z-bands.
    labels : np.ndarray
        The labeled image of z-bands.
    labels_skel : np.ndarray
        The skeletonized labels of z-bands.
    image_raw : np.ndarray
        The raw image.
    orientation_field : np.ndarray
        Sarcomere orientation field.
    pixelsize : float
        The size of pixels in the image.
    min_length : float, optional
        The minimum length threshold for z-bands. Default is 1.0.
    threshold : float, optional
        The threshold value for intensity. Default is 0.1.
    median_filter_radius : float, optional
        Radius of kernel to smooth orientation field. Default is 0.2 µm.
    a_min : float, optional
        The minimum value for alignment. Default is 0.25. Links with smaller alignment are set to np.nan.
    theta_phi_min : float, optional
        The minimum dot product/cosine between the direction of a Z-band end and the direction of line from end to other Z-band end.
    d_max : float, optional
        The maximum distance between z-band ends. Default is 5.0 µm. Larger distances are set to np.nan.
    d_min : float, optional
        The minimum distance between z-band ends. Default is 0 µm. Smaller distances are set to np.nan.

    Returns
    -------
    tuple
        A comprehensive tuple containing arrays and values describing the analyzed properties of z-bands:
        - Lengths, intensities, straightness, ratio of intensities, average intensity, orientations,
          orientational order parameter, list of z-band labels, processed labels image, number of lateral neighbors,
          lateral distances, lateral alignments, links between z-band ends, coordinates of z-band ends,
          linked groups of z-bands, and their respective sizes, lengths, and alignments.
    """
    # analyze skeletonized labels to determine z-band backbone length
    props_skel = regionprops_table(labels_skel, properties=['label', ],
                                   extra_properties=(Utils.skeleton_length_igraph, ))
    labels_list = props_skel['label']

    # remove short z-bands
    length = props_skel['skeleton_length_igraph'] * pixelsize
    labels_list_ = labels_list.copy()
    labels_list[length < min_length] = 0
    labels_list = np.insert(labels_list, 0, 0)
    labels_list_ = np.insert(labels_list_, 0, 0)
    labels = Utils.map_array(labels, labels_list_, labels_list)
    labels, forward_map, inverse_map = segmentation.relabel_sequential(labels)
    labels_list = labels_list[labels_list != 0]

    # sarcomere orientation map
    smooth_radius_px = int(median_filter_radius / pixelsize)
    sarcomere_orientation = Utils.get_orientation_angle_map(orientation_field, use_median_filter=True,
                                                            radius=smooth_radius_px)

    # analyze z-band labels
    props = regionprops_table(labels, intensity_image=image_raw, properties=['label', 'area', 'convex_area',
                                                                             'mean_intensity', 'orientation',
                                                                             'image', 'bbox', 'centroid'])
    # z-band length
    length = length[length >= min_length]

    # straightness of z-bands (area/convex_hull)
    straightness = props['area'] / props['convex_area']

    # fluorescence intensity of each individual z-band, the total area, and the average intensity of z-band mask
    intensity = props['mean_intensity']
    z_mask = zbands > threshold
    z_mask_area = np.sum(z_mask.astype('uint8')) * pixelsize ** 2
    z_mask_intensity = np.mean(image_raw[z_mask])

    # z band orientational order parameter
    orientation = props['orientation']
    if len(orientation) > 0:
        oop = 1 / len(orientation) * np.abs(np.sum(np.exp(orientation * 2 * 1j)))
    else:
        oop = np.nan

    # local lateral z-band alignment and distance
    n_z = len(np.unique(labels)) - 1

    if n_z > 0:

        # get two ends of each z-band
        z_ends = np.zeros((n_z, 2, 2)) * np.nan  # (z-band idx, upper/lower end, x/y)
        z_orientation = np.zeros((n_z, 2)) * np.nan  # (z-band idx, upper/lower)
        pad_width = int(round(1 / pixelsize, 0))

        for i, zbands_i in enumerate(props['image']):
            zbands_i = np.pad(zbands_i, (pad_width, pad_width))

            # skeletonize
            skel_i = skeletonize(zbands_i, method='lee')

            # detect line ends
            def line_end_filter(d):
                return (d[4] == 1) and np.sum(d) == 2

            z_ends_i = ndimage.generic_filter(skel_i, line_end_filter, (3, 3))
            z_ends_i = np.asarray(np.where(z_ends_i == 1))
            z_ends_i[0] += props['bbox-0'][i] - pad_width
            z_ends_i[1] += props['bbox-1'][i] - pad_width
            centroid_i = (props['centroid-0'][i], props['centroid-1'][i])

            if len(z_ends_i.T) == 2:
                if z_ends_i[1, 0] > z_ends_i[1, 1]:
                    z_ends_i = z_ends_i[:, ::-1]
                # Get orientations from map and add π/2
                orientation_ends_i = np.asarray([sarcomere_orientation[z_ends_i[0][0], z_ends_i[1][0]] + np.pi / 2,
                                                 sarcomere_orientation[z_ends_i[0][1], z_ends_i[1][1]] + np.pi / 2])

                # Calculate local directions from endpoints to their own positions in skeleton
                _orient_1 = np.arctan2(z_ends_i[0, 0] - centroid_i[0], z_ends_i[1, 0] - centroid_i[1])
                _orient_2 = np.arctan2(z_ends_i[0, 1] - centroid_i[0], z_ends_i[1, 1] - centroid_i[1])

                # Better angle difference calculation (minimum angle in range [0, π])
                def angle_diff(a1, a2):
                    return np.abs((a1 - a2 + np.pi) % (2 * np.pi) - np.pi)

                # Apply π shift if angles differ by more than π/2
                if angle_diff(orientation_ends_i[0], _orient_1) > np.pi / 2:
                    orientation_ends_i[0] = orientation_ends_i[0] + np.pi
                if angle_diff(orientation_ends_i[1], _orient_2) > np.pi / 2:
                    orientation_ends_i[1] = orientation_ends_i[1] + np.pi

                orientation_ends_i = -orientation_ends_i + np.pi / 2

                # # Ensure angles stay in range [-π, π]
                orientation_ends_i[0] = (orientation_ends_i[0] + np.pi) % (2 * np.pi) - np.pi
                orientation_ends_i[1] = (orientation_ends_i[1] + np.pi) % (2 * np.pi) - np.pi

                z_orientation[i] = orientation_ends_i
                z_ends[i] = z_ends_i.T * pixelsize

        # lateral alignment index and distance of z-bands
        def lateral_alignment(pos_i, pos_j, theta_i, theta_j):
            phi_ij = np.arctan2((pos_j[1] - pos_i[1]), (pos_j[0] - pos_i[0])) % (2 * np.pi)
            phi_ji = (phi_ij + np.pi) % (2 * np.pi)

            a_ji = np.cos(theta_i - theta_j + np.pi) * np.cos(theta_i - phi_ij) * np.cos(theta_j - phi_ji)

            if np.cos(theta_i - theta_j + np.pi) > 0 and np.cos(theta_i - phi_ij) > theta_phi_min and np.cos(
                    theta_j - phi_ji) > theta_phi_min:
                return a_ji
            else:
                return np.nan

        # distance of z-band ends
        _z_ends = np.reshape(z_ends, (n_z * 2, 2), order='F')
        D = squareform(pdist(_z_ends, 'euclidean'))

        # Set NaNs for specified indices (ends of same objects) and the lower triangle
        indices = np.arange(0, n_z * 2, 2)
        mask = np.ones((n_z * 2, n_z * 2))
        mask[indices, indices] = 0
        mask[indices, indices + 1] = 0
        mask[indices + 1, indices] = 0
        mask[indices + 1, indices + 1] = 0
        mask[np.tril(mask) > 0] = np.nan

        # filter distance matrix
        D[(D > d_max) | (D < d_min) | (mask == 0)] = np.nan

        # indices of end-end-distances shorter than d_max
        _z_orientation = np.reshape(z_orientation, (n_z * 2), order='F')
        _idxs = np.asarray(np.where(~np.isnan(D)))

        # matrix with lateral alignments A
        A = np.zeros_like(D) * np.nan
        for (i, j) in _idxs.T:
            A_ij = lateral_alignment(_z_ends[i], _z_ends[j], _z_orientation[i], _z_orientation[j])
            A[i, j] = A_ij if A_ij >= a_min else np.nan
        D[np.isnan(A)] = np.nan

        # make matrices symmetric for undirected graph
        D = (D + D.T) / 2
        A = (A + A.T) / 2

        def compute_cost_matrix(D, A, penalty=1e6):
            """
            Compute the cost matrix for linking Z-band ends based on a cost 1 - A favoring optimal alignment.

            Parameters:
            ----------
            D : ndarray
                Distance matrix between Z-band ends.
            A : ndarray
                Alignment matrix between Z-band ends.
            w_dist : float
                Weight for distance in the cost function.
            w_align : float
                Weight for alignment in the cost function.
            penalty : float
                Penalty for invalid links (e.g., NaN or out-of-range values).

            Returns:
            -------
            C : ndarray
                Cost matrix for linking Z-band ends.
            """
            # Ensure alignment values are valid (replace NaNs with 0)
            A = np.nan_to_num(A, nan=0.0)

            # Compute cost matrix
            C = 1 - A

            # Set invalid links (e.g., NaNs in D) to a very high cost
            C[np.isnan(D)] = penalty

            return C

        def solve_linking(C):
            """
            Solve the optimal linking problem using the Hungarian algorithm.

            Parameters:
            ----------
            C : ndarray
                Cost matrix for linking Z-band ends.

            Returns:
            -------
            row_ind : ndarray
                Row indices of the optimal assignment.
            col_ind : ndarray
                Column indices of the optimal assignment.
            """
            # Use scipy's linear_sum_assignment to solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(C)

            return row_ind, col_ind

        # Step 1: Compute cost matrix
        C = compute_cost_matrix(D, A)

        # Step 2: Solve optimal linking using Hungarian algorithm
        row_ind, col_ind = solve_linking(C)

        # Step 3: Create adjacency matrix for valid links
        links = np.zeros_like(D)
        for i, j in zip(row_ind, col_ind):
            links[i, j] = 1 if D[i, j] <= d_max and A[i, j] >= a_min else 0

        # reshape arrays
        links = links.reshape((n_z, 2, n_z, 2), order='F')
        lat_dist = D.reshape((n_z, 2, n_z, 2), order='F')
        lat_alignment = A.reshape((n_z, 2, n_z, 2), order='F')

        # number of lateral neighbors
        links_z = np.sum(links, axis=(1, 3))
        lat_neighbors = np.count_nonzero(links_z, axis=1)

        # convert links, lat_dist and lat_alignment to lists
        links = np.where(links == 1)
        lat_dist = lat_dist[links]
        lat_alignment = lat_alignment[links]
        links = np.asarray(links)

        # analyze laterally linked groups
        def analyze_linked_groups(connectivity_matrix, distance_matrix, alignment_matrix):
            G = nx.Graph()

            for n in range(n_z):
                G.add_node(n)

            # Efficiently add edges based on connectivity and criteria
            for n, (idx_i, end_i, idx_j, end_j) in enumerate(connectivity_matrix.T):
                G.add_edge(idx_i, idx_j, alignment=alignment_matrix[n], distance=distance_matrix[n])

            # Find connected components in the graph with best matches
            _linked_groups = list(nx.connected_components(G))

            _size_groups = np.asarray([len(group) for group in _linked_groups])
            # Calculate length of each group
            _length_groups = []
            _alignment_groups = []
            for group in _linked_groups:
                sum_distance = 0
                sum_alignment = 0
                for node in group:
                    edges = G.edges(node, data=True)
                    for _, _, data in edges:
                        if G.has_edge(_, node):  # Check if edge is within the current group
                            sum_distance += data['distance']
                            sum_alignment += data['alignment']
                sum_distance /= 2  # Each edge is counted twice (undirected graph), so divide by 2
                _length_groups.append(sum_distance + np.sum(length[list(group)]))
                _alignment_groups.append(sum_alignment / len(group))
            _linked_groups = [list(s) for s in _linked_groups]
            return (_linked_groups, np.asarray(_size_groups), np.asarray(_length_groups),
                    np.asarray(_alignment_groups))

        linked_groups, size_groups, length_groups, alignment_groups = analyze_linked_groups(links, lat_dist,
                                                                                            lat_alignment)
    else:
        (lat_neighbors, lat_dist, lat_alignment, links, z_ends,
         linked_groups, size_groups, length_groups, alignment_groups) = [], [], [], [], [], [], [], [], []

    return (length, intensity, straightness, z_mask_intensity, z_mask_area, orientation, oop, labels_list, labels,
            lat_neighbors, lat_dist, lat_alignment, links, z_ends, linked_groups, size_groups, length_groups,
            alignment_groups)
