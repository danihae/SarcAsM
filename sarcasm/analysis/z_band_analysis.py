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

from typing import Dict, Optional, Tuple
import numpy as np
import networkx as nx
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform, pdist
from skimage import segmentation, morphology
from skimage.measure import label, regionprops_table
from skimage.morphology import skeletonize
from skimage.draw import line as draw_line

from sarcasm.utils import Utils

# length (µm) of skeleton behind a Z-band end that sets the direction of that end
END_DIRECTION_LENGTH = 0.5
# width (µm) of the ring of non-Z-band pixels around a Z-band that its contrast is measured against
CONTRAST_RING_WIDTH = 1.0
# smallest ring area (µm²) a Z-band contrast is computed from
CONTRAST_MIN_RING_AREA = 0.1
# smallest fraction of a frame outside the cell mask whose median is taken as the background
MIN_BACKGROUND_FRACTION = 0.005


def estimate_background(image: np.ndarray, cell_mask: Optional[np.ndarray] = None) -> float:
    """
    Background intensity of a frame (camera offset plus medium): the median of the pixels outside the cell mask.

    Parameters
    ----------
    image : np.ndarray
        Raw image.
    cell_mask : np.ndarray or None, optional
        Binary cell mask. Default is None.

    Returns
    -------
    float
        Median intensity outside the cell mask; NaN without a cell mask or when cells cover all but
        ``MIN_BACKGROUND_FRACTION`` of the frame.
    """
    if cell_mask is None:
        return np.nan
    outside = np.asarray(image)[~np.asarray(cell_mask, dtype=bool)]
    if outside.size == 0 or outside.size < MIN_BACKGROUND_FRACTION * np.size(image):
        return np.nan
    return float(np.median(outside))


def z_band_contrast(labels: np.ndarray, z_mask: np.ndarray, image: np.ndarray, pixelsize: float,
                    background: float, ring_width: float = CONTRAST_RING_WIDTH,
                    cell_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratiometric contrast of each Z-band against the non-Z-band pixels around it.

    The contrast of a Z-band is (mean intensity of the Z-band - background) / (mean intensity of its ring -
    background). The ring of a Z-band is the non-Z-band pixels within ``ring_width`` µm whose nearest Z-band
    pixel belongs to it, so neighbouring Z-bands split the space between them; pixels nearest to a Z-band pixel
    without a label (too short), and pixels outside the cell mask, belong to no ring.

    Parameters
    ----------
    labels : np.ndarray
        Labelled Z-bands, 1..n.
    z_mask : np.ndarray
        Binary mask of all Z-band pixels, labelled or not.
    image : np.ndarray
        Raw image.
    pixelsize : float
        Pixel size in µm.
    background : float
        Background intensity of the frame.
    ring_width : float, optional
        Width of the ring in µm. Default is ``CONTRAST_RING_WIDTH``.
    cell_mask : np.ndarray or None, optional
        Binary cell mask; ring pixels outside it are left out. None keeps all. Default is None.

    Returns
    -------
    contrast : np.ndarray
        Contrast of each Z-band, shape (n,). NaN where the ring is smaller than ``CONTRAST_MIN_RING_AREA`` or
        not brighter than the background.
    surround : np.ndarray
        Mean intensity of the ring of each Z-band minus the background, shape (n,): the denominator of the
        contrast. NaN where the ring is smaller than ``CONTRAST_MIN_RING_AREA``.
    """
    n = int(labels.max()) if labels.size else 0
    if n == 0:
        return np.zeros(0), np.zeros(0)
    in_z = np.asarray(z_mask, dtype=bool) | (labels > 0)
    dist, (rr, cc) = ndimage.distance_transform_edt(~in_z, return_indices=True)
    ring = labels[rr, cc]                                       # label of the nearest Z-band pixel
    ring[in_z | (dist > ring_width / pixelsize)] = 0
    if cell_mask is not None:
        ring[~np.asarray(cell_mask, dtype=bool)] = 0
    img = np.asarray(image, dtype=np.float64).ravel()
    lab, ring = labels.ravel(), ring.ravel()
    n_z = np.bincount(lab, minlength=n + 1)[1:]
    n_ring = np.bincount(ring, minlength=n + 1)[1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_z = np.bincount(lab, weights=img, minlength=n + 1)[1:] / n_z
        surround = np.bincount(ring, weights=img, minlength=n + 1)[1:] / n_ring - background
        contrast = (mean_z - background) / surround
    small = n_ring * pixelsize ** 2 < CONTRAST_MIN_RING_AREA
    surround[small] = np.nan
    contrast[small | ~(surround > 0)] = np.nan
    return contrast, surround


def segment_z_bands(image: np.ndarray, threshold: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment z-bands from U-Net result (threshold, make binary, skeletonize, label regions).

    Parameters
    ----------
    image : np.ndarray
        Input image from U-Net.
    threshold : float, optional
        Threshold for binarizing the image. Default is 0.15.

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
                    d_min: float = 0.25, max_axial_offset: float = 0.9,
                    background: Optional[float] = None, cell_mask: Optional[np.ndarray] = None) -> Dict:
    """
    Analyzes segmented z-bands in a single frame, extracting metrics such as length, intensity, ratiometric
    contrast, orientation, lateral distance, number of lateral neighbors per z-band, and the length of groups of
    laterally connected z-bands.

    Parameters
    ----------
    zbands : np.ndarray
        Segmented z-band map.
    labels : np.ndarray
        Labeled image of z-bands.
    labels_skel : np.ndarray
        Skeletonized labels of z-bands.
    image_raw : np.ndarray
        Raw image.
    orientation_field : np.ndarray
        Sarcomere orientation field.
    pixelsize : float
        Pixel size in µm.
    min_length : float, optional
        Minimum z-band length in µm; shorter z-bands are removed. Default is 1.0.
    threshold : float, optional
        Intensity threshold for the z-band mask. Default is 0.5.
    median_filter_radius : float, optional
        Radius in µm of the kernel smoothing the orientation field. Default is 0.25.
    a_min : float, optional
        Minimum linking score of two z-band ends (product of the cosines between the two end directions and
        between each end direction and the connecting line, 0-1); pairs with a smaller score are not linked.
        Default is 0.3.
    theta_phi_min : float, optional
        Minimum cosine between a z-band end's direction and the line from that end to
        the other z-band's end. Default is 0.2.
    d_max : float, optional
        Maximum distance between z-band ends in µm; larger distances are set to NaN. Default is 4.0.
    d_min : float, optional
        Minimum distance between z-band ends in µm; smaller distances are set to NaN. Default is 0.25.
    max_axial_offset : float or None, optional
        Maximum offset in µm of a lateral connection along the sarcomere axis (perpendicular to the Z-bands).
        A connection continues a Z-band across a gap; one offset by about half a sarcomere or more reaches the
        next Z-band of the neighbouring myofibril instead. None disables the limit. Default is 0.9.
    background : float or None, optional
        Background intensity of the raw image (camera offset plus medium), subtracted before the contrast
        ratio is taken; see :func:`estimate_background`. None takes the 1st percentile of the image.
        Default is None.
    cell_mask : np.ndarray or None, optional
        Binary cell mask; the rings of the contrast are restricted to it. Default is None.

    Returns
    -------
    dict
        ``length`` (µm), ``intensity`` (mean raw intensity), ``contrast`` and ``surround`` (see
        :func:`z_band_contrast`) of each z-band; ``mask_intensity``, ``mask_area`` (µm²) of the z-band mask; ``orientation`` of each z-band
        and their orientational order parameter ``oop``; ``labels_list`` and the relabelled ``labels`` image;
        ``lat_neighbors`` (number of lateral neighbors of each z-band), ``lat_dist`` (µm) of the lateral links,
        ``links`` between z-band ends and the ``ends`` (µm) of each z-band; ``lat_groups`` (laterally linked
        groups of z-bands) with their lengths ``lat_length_groups`` (µm); and the ``background`` used.
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
    props = regionprops_table(labels, intensity_image=image_raw, properties=['label', 'mean_intensity',
                                                                             'orientation', 'image', 'bbox',
                                                                             'centroid'])
    # z-band length
    length = length[length >= min_length]

    # fluorescence intensity of each individual z-band, the total area, and the average intensity of z-band mask
    intensity = props['mean_intensity']
    z_mask = zbands > threshold
    z_mask_area = np.sum(z_mask.astype('uint8')) * pixelsize ** 2
    z_mask_intensity = np.mean(image_raw[z_mask])

    # ratiometric contrast of each z-band against the non-z-band pixels around it, background-corrected
    if background is None or not np.isfinite(background):
        background = float(np.percentile(image_raw, 1))
    contrast, surround = z_band_contrast(labels, z_mask, image_raw, pixelsize, background, cell_mask=cell_mask)

    # z band orientational order parameter
    orientation = props['orientation']
    if len(orientation) > 0:
        oop = 1 / len(orientation) * np.abs(np.sum(np.exp(orientation * 2 * 1j)))
    else:
        oop = np.nan

    # lateral connections of z-band ends and their distances
    n_z = len(np.unique(labels)) - 1

    if n_z > 0:

        # get two ends of each z-band
        z_ends = np.zeros((n_z, 2, 2)) * np.nan  # (z-band idx, upper/lower end, x/y)
        z_orientation = np.zeros((n_z, 2)) * np.nan  # (z-band idx, upper/lower)
        pad_width = int(round(1 / pixelsize, 0))

        # 3x3 neighbor kernel (excluding center) used for vectorized endpoint detection below
        _nbr_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)

        for i, zbands_i in enumerate(props['image']):
            zbands_i = np.pad(zbands_i, (pad_width, pad_width))

            # skeletonize
            skel_i = skeletonize(zbands_i, method='lee')

            # detect line ends: skeleton pixels with exactly 1 foreground neighbor (degree 1).
            skel_bool = skel_i.astype(bool)
            nbr_count = ndimage.convolve(skel_bool.astype(np.int8), _nbr_kernel, mode='constant')
            z_ends_i = np.asarray(np.where(skel_bool & (nbr_count == 1)))
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

                # direction of each end from the z-band's own skeleton (outwards over its last END_DIRECTION_LENGTH),
                # since the orientation field is least defined at z-band ends; the field direction is kept where
                # less than one pixel of skeleton lies behind the end
                skel_px = np.asarray(np.where(skel_bool), dtype=float)
                skel_px[0] += props['bbox-0'][i] - pad_width
                skel_px[1] += props['bbox-1'][i] - pad_width
                for k in (0, 1):
                    end_k = z_ends_i[:, k].astype(float)
                    near = skel_px[:, np.hypot(*(skel_px - end_k[:, None])) <= END_DIRECTION_LENGTH / pixelsize]
                    v = end_k - near.mean(axis=1)
                    if np.hypot(*v) >= 1.0:
                        orientation_ends_i[k] = np.arctan2(v[1], v[0])

                z_orientation[i] = orientation_ends_i
                z_ends[i] = z_ends_i.T * pixelsize

        # linking score of two z-band ends (used only to choose the lateral connections, not reported)
        def linking_score(pos_i, pos_j, theta_i, theta_j):
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

        # Set NaNs for the two ends of the same z-band (row z + n_z * end in the Fortran-order reshape above) and
        # the lower triangle
        z_of_end = np.tile(np.arange(n_z), 2)
        mask = np.where(z_of_end[:, None] == z_of_end[None, :], 0.0, 1.0)
        mask[np.tril(mask) > 0] = np.nan

        # filter distance matrix
        D[(D > d_max) | (D < d_min) | (mask == 0)] = np.nan

        # indices of end-end-distances shorter than d_max
        _z_orientation = np.reshape(z_orientation, (n_z * 2), order='F')
        _idxs = np.asarray(np.where(~np.isnan(D)))

        # matrix with linking scores A
        A = np.zeros_like(D) * np.nan
        for (i, j) in _idxs.T:
            A_ij = linking_score(_z_ends[i], _z_ends[j], _z_orientation[i], _z_orientation[j])
            A[i, j] = A_ij if A_ij >= a_min else np.nan
        D[np.isnan(A)] = np.nan

        # a lateral connection continues a z-band across a gap: reject candidates offset along the sarcomere axis by
        # more than max_axial_offset (they reach the next z-band of the neighbouring myofibril) and candidates whose
        # straight segment runs over a third z-band
        for (i, j) in np.asarray(np.where(~np.isnan(A))).T:
            d = _z_ends[j] - _z_ends[i]
            ref = np.array([np.cos(_z_orientation[i]) - np.cos(_z_orientation[j]),
                            np.sin(_z_orientation[i]) - np.sin(_z_orientation[j])])   # z-band line, end i outwards
            norm = np.hypot(*ref)
            if max_axial_offset is not None and (not norm > 0 or
                                                 abs(d[0] * ref[1] - d[1] * ref[0]) / norm > max_axial_offset):
                A[i, j] = np.nan
                continue
            rr, cc = draw_line(*np.round(_z_ends[i] / pixelsize).astype(int), *np.round(_z_ends[j] / pixelsize).astype(int))
            inside = (rr >= 0) & (rr < labels.shape[0]) & (cc >= 0) & (cc < labels.shape[1])
            crossed = np.unique(labels[rr[inside], cc[inside]])
            if np.any((crossed != 0) & (crossed != i % n_z + 1) & (crossed != j % n_z + 1)):
                A[i, j] = np.nan
        D[np.isnan(A)] = np.nan

        # make matrices symmetric for undirected graph
        D = (D + D.T) / 2
        A = (A + A.T) / 2

        def compute_cost_matrix(D, A, penalty=1e6):
            """
            Compute the cost matrix (1 - A) for linking Z-band ends, favoring the highest linking scores.

            Parameters
            ----------
            D : np.ndarray
                Distance matrix between Z-band ends.
            A : np.ndarray
                Linking-score matrix between Z-band ends.
            penalty : float, optional
                Cost assigned to invalid links (NaN in ``D``). Default is 1e6.

            Returns
            -------
            np.ndarray
                Cost matrix for linking Z-band ends.
            """
            # Ensure scores are valid (replace NaNs with 0)
            A = np.nan_to_num(A, nan=0.0)

            # Compute cost matrix
            C = 1 - A

            # Set invalid links (e.g., NaNs in D) to a very high cost
            C[np.isnan(D)] = penalty

            return C

        def solve_linking(C):
            """
            Solve the optimal linking problem using the Hungarian algorithm.

            Parameters
            ----------
            C : np.ndarray
                Cost matrix for linking Z-band ends.

            Returns
            -------
            row_ind : np.ndarray
                Row indices of the optimal assignment.
            col_ind : np.ndarray
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

        # number of lateral neighbors
        links_z = np.sum(links, axis=(1, 3))
        lat_neighbors = np.count_nonzero(links_z, axis=1)

        # convert links and lat_dist to lists
        links = np.where(links == 1)
        lat_dist = lat_dist[links]
        links = np.asarray(links)

        # analyze laterally linked groups
        def analyze_linked_groups(connectivity_matrix, distance_matrix):
            G = nx.Graph()

            for n in range(n_z):
                G.add_node(n)

            # Efficiently add edges based on connectivity and criteria
            for n, (idx_i, end_i, idx_j, end_j) in enumerate(connectivity_matrix.T):
                G.add_edge(idx_i, idx_j, distance=distance_matrix[n])

            # Find connected components in the graph with best matches
            _linked_groups = list(nx.connected_components(G))

            # Calculate length of each group
            _length_groups = []
            for group in _linked_groups:
                sum_distance = 0
                for node in group:
                    edges = G.edges(node, data=True)
                    for _, _, data in edges:
                        if G.has_edge(_, node):  # Check if edge is within the current group
                            sum_distance += data['distance']
                sum_distance /= 2  # Each edge is counted twice (undirected graph), so divide by 2
                _length_groups.append(sum_distance + np.sum(length[list(group)]))
            _linked_groups = [list(s) for s in _linked_groups]
            return _linked_groups, np.asarray(_length_groups)

        linked_groups, length_groups = analyze_linked_groups(links, lat_dist)
    else:
        lat_neighbors, lat_dist, links, z_ends, linked_groups, length_groups = [], [], [], [], [], []

    return {'length': length, 'intensity': intensity, 'contrast': contrast, 'surround': surround,
            'mask_intensity': z_mask_intensity,
            'mask_area': z_mask_area, 'orientation': orientation, 'oop': oop, 'labels_list': labels_list,
            'labels': labels, 'lat_neighbors': lat_neighbors, 'lat_dist': lat_dist,
            'links': links, 'ends': z_ends, 'lat_groups': linked_groups, 'lat_length_groups': length_groups,
            'background': background}
