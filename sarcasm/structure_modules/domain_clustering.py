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

"""Domain clustering and analysis module."""

from typing import Tuple, List
import random
import numpy as np
import igraph as ig
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.morphology import binary_dilation, disk

from sarcasm.utils import Utils


def cluster_sarcomeres(pos_vectors: np.ndarray,
                       sarcomere_length_vectors: np.ndarray,
                       sarcomere_orientation_vectors: np.ndarray,
                       pixelsize: float,
                       size: Tuple[int, int],
                       d_max: float = 3,
                       cosine_min: float = 0.65,
                       leiden_resolution: float = 0.06,
                       random_seed: int = 42,
                       area_min: float = 20,
                       dilation_radius: float = 0.3) -> Tuple[int, List, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function clusters sarcomeres into domains based on their spatial and orientational properties
    using the Leiden method for community detection in igraph. It considers sarcomere lengths, orientations,
    and positions along mbands to form networks of connected sarcomeres. Domains are then identified
    as communities within these networks, with additional criteria for minimum domain area
    and connectivity thresholds. Finally, this function quantifies the mean and std of sarcomere lengths,
    and the orientational order parameter and mean orientation of each domain.

    Parameters
    ----------
    pos_vectors : np.ndarray
        Array of sarcomere midline point positions in µm.
    sarcomere_length_vectors : np.ndarray
        List of midline point sarcomere lengths
    sarcomere_orientation_vectors : np.ndarray
        List of midline point sarcomere orientations, in radians
    pixelsize : float
        Pixel size in µm
    size : tuple(int, int)
        Shape of the image in pixels
    d_max : float
        Max. distance threshold for creating a network edge between vector ends
    cosine_min : float
        Minimal absolute cosine between vector angles for creating a network edge between vector ends
    leiden_resolution : float
        Resolution parameter for the Leiden algorithm
    random_seed : int
        Random seed for reproducibility
    area_min : float
        Minimal area (in µm²) for a domain to be kept
    dilation_radius : float
        Dilation radius for refining domain area masks (in µm)

    Returns
    -------
    n_domains : int
        Number of domains
    domains : list
        List of domain sets with point indices
    area_domains : list
        List with domain areas
    sarcomere_length_mean_domains : list
        Mean sarcomere length within each domain
    sarcomere_length_std_domains : list
        Standard deviation of sarcomere length within each domain
    sarcomere_oop_domains : list
        Orientational order parameter of sarcomeres in each domain
    sarcomere_orientation_domains : list
        Main orientation of domains
    mask_domains : ndarray
        Masks of domains with value representing domain label
    """

    if len(pos_vectors) < 10:
        return 0, [], np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    n_vectors = sarcomere_length_vectors.shape[0]

    # Calculate orientation vectors using trigonometry
    orientation_vectors = np.column_stack([np.sin(sarcomere_orientation_vectors),
                                           np.cos(sarcomere_orientation_vectors)])

    # Calculate end points of the vectors
    ends_0 = pos_vectors + orientation_vectors * sarcomere_length_vectors[:, None] / 2
    ends_1 = pos_vectors - orientation_vectors * sarcomere_length_vectors[:, None] / 2

    # Interleave ends_0 and ends_1
    ends = np.empty((2 * n_vectors, 2), dtype=np.float64)
    ends[0::2] = ends_0
    ends[1::2] = ends_1

    # Interleave orientation vectors
    orientation_ends = np.empty((2 * n_vectors, 2), dtype=np.float64)
    orientation_ends[0::2] = orientation_vectors
    orientation_ends[1::2] = -orientation_vectors

    # Use cKDTree to find pairs within the distance threshold
    tree = cKDTree(ends)
    pairs = tree.query_pairs(d_max, output_type='ndarray')

    # Compute cosine similarity for all pairs at once
    dot_products = np.sum(orientation_ends[pairs[:, 0]] * orientation_ends[pairs[:, 1]], axis=1)
    norms = np.linalg.norm(orientation_ends[pairs[:, 0]], axis=1) * np.linalg.norm(
        orientation_ends[pairs[:, 1]], axis=1)
    cosine_similarities = np.abs(dot_products / norms)

    # Filter pairs based on cosine similarity
    valid_pairs = cosine_similarities > cosine_min
    filtered_pairs = pairs[valid_pairs]

    # Calculate distances for valid pairs
    distances = (np.sqrt(np.sum((ends[filtered_pairs[:, 0]] - ends[filtered_pairs[:, 1]]) ** 2, axis=1))
                 / cosine_similarities[valid_pairs])

    # Create edges list
    edges = filtered_pairs.tolist()

    # Add zero-cost connections between the two ends of each vector
    zero_cost_edges = [(2 * i, 2 * i + 1) for i in range(n_vectors)]
    edges.extend(zero_cost_edges)

    # Create weights list
    weights = distances.tolist()
    weights.extend([0] * n_vectors)

    # Create the graph
    graph = ig.Graph(2 * n_vectors)
    graph.add_edges(edges)
    graph.es['weight'] = weights

    # Create a mapping to contract pairs of vertices
    mapping = [i // 2 for i in range(graph.vcount())]

    # Contract the vertices
    graph.contract_vertices(mapping)

    # Set random seed
    random.seed(random_seed)

    # Run Leiden
    # CommunityLeiden returns a VertexClustering, from which we can get memberships
    clusters = graph.community_leiden(
        weights="weight",
        resolution_parameter=leiden_resolution,
        n_iterations=-1,
        objective_function="modularity"
    )

    # Build domain sets
    membership = clusters.membership
    domains_dict = {}
    for idx, c_id in enumerate(membership):
        domains_dict.setdefault(c_id, []).append(idx)
    domains = list(domains_dict.values())

    # Shuffle domains for random ordering
    random.shuffle(domains)

    (mask_domains, area_domains, sarcomere_length_mean_domains,
     sarcomere_length_std_domains, sarcomere_oop_domains,
     sarcomere_orientation_domains) = analyze_domains(domains, pos_vectors,
                                                      sarcomere_orientation_vectors,
                                                      sarcomere_length_vectors, size=size,
                                                      pixelsize=pixelsize,
                                                      dilation_radius=dilation_radius, area_min=area_min)

    area_domains = np.asarray(area_domains)
    sarcomere_length_mean_domains = np.asarray(sarcomere_length_mean_domains)
    sarcomere_length_std_domains = np.asarray(sarcomere_length_std_domains)
    sarcomere_oop_domains = np.asarray(sarcomere_oop_domains)
    sarcomere_orientation_domains = np.asarray(sarcomere_orientation_domains)
    n_domains = len(area_domains)

    return (n_domains,
            domains,
            area_domains,
            sarcomere_length_mean_domains,
            sarcomere_length_std_domains,
            sarcomere_oop_domains,
            sarcomere_orientation_domains,
            mask_domains)


def sarcomere_mask(points: np.ndarray,
                   sarcomere_orientation_vectors: np.ndarray,
                   sarcomere_length_vectors: np.ndarray,
                   shape: Tuple[int, int],
                   pixelsize: float,
                   dilation_radius: float = 0.3) -> np.ndarray:
    """
    Calculates a binary mask of areas with sarcomeres.

    Parameters
    ----------
    points : ndarray
        Positions of sarcomere vectors in µm. (n_vectors, 2)
    sarcomere_orientation_vectors : ndarray
        Orientations of sarcomere vectors.
    sarcomere_length_vectors : ndarray
        Lengths of sarcomere vectors in µm.
    shape : tuple
        Shape of the image, in pixels.
    pixelsize : float
        Pixel size in µm.
    dilation_radius : float, optional
        Dilation radius to close small holes in mask, in µm (default is 0.3).

    Returns
    -------
    mask : ndarray
        Binary mask of sarcomeres.
    """
    # Calculate orientation vectors using trigonometry
    sarcomere_orientation_vectors += np.pi / 2

    orientation_vectors = np.asarray([np.cos(sarcomere_orientation_vectors),
                                      -np.sin(sarcomere_orientation_vectors)])
    # Calculate the ends of the vectors based on their orientation and length
    ends_0 = points.T + orientation_vectors * sarcomere_length_vectors / 2  # End point 1 of each vector
    ends_1 = points.T - orientation_vectors * sarcomere_length_vectors / 2  # End point 2 of each vector
    ends_0, ends_1 = ends_0 / pixelsize, ends_1 / pixelsize
    mask = np.zeros(shape, dtype='bool')
    for e0, e1 in zip(ends_0.T.astype('int'), ends_1.T.astype('int')):
        rr, cc = line(*e0, *e1)
        try:
            mask[rr, cc] = True
        except:
            pass
    dilation_radius_pixels = int(round(dilation_radius / pixelsize, 0))
    mask = binary_dilation(mask, disk(dilation_radius_pixels))
    return mask


def analyze_domains(domains: List, pos_vectors: np.ndarray,
                    sarcomere_orientation_vectors: np.ndarray,
                    sarcomere_length_vectors: np.ndarray,
                    size: Tuple[int, int],
                    pixelsize: float,
                    dilation_radius: float,
                    area_min: float):
    """
    Creates a domain mask, where each domain has a distinct label, and analyzes the individual domains.

    Parameters
    __________
    domains : list
        List with domain labels for each vector. Each domain is labeled with a unique integer.
    pos_vectors : ndarray
        Position vectors in micrometers.
    sarcomere_orientation_vectors : ndarray
        Orientation angles in radians.
    sarcomere_length_vectors : ndarray
        Sarcomere lengths in micrometers.
    size : tuple of int
        Output map dimensions (height, width) in pixels.
    pixelsize : float
        Physical size of one pixel in micrometers.
    dilation_radius : float, optional
        Dilation radius for refining domain masks, in µm.
    area_min : float, optional
        Minimal area of a domain in µm^2, smaller domains are discarded.
    """
    # calculate domain properties and remove small domains
    (area_domains, sarcomere_orientation_domains, sarcomere_oop_domains, sarcomere_length_mean_domains,
     sarcomere_length_std_domains) = [], [], [], [], []

    mask_domains = np.zeros(size, dtype='uint8')

    j = 1
    for i, domain_i in enumerate(domains):
        pos_vectors_i = pos_vectors[domain_i]
        orientations_i = sarcomere_orientation_vectors[domain_i]
        lengths_i = sarcomere_length_vectors[domain_i]
        if pos_vectors_i.shape[0] > 10:
            # bounding box
            min_i = (
                max(int((pos_vectors_i[:, 0].min() - 3) // pixelsize), 0),
                max(int((pos_vectors_i[:, 1].min() - 3) // pixelsize), 0))
            max_i = (min(int((pos_vectors_i[:, 0].max() + 3) // pixelsize), size[0]),
                     min(int((pos_vectors_i[:, 1].max() + 3) // pixelsize), size[1]))
            size_i = (max_i[0] - min_i[0], max_i[1] - min_i[1])

            _pos_vectors_i = pos_vectors_i.copy()
            _pos_vectors_i[:, 0] -= min_i[0] * pixelsize
            _pos_vectors_i[:, 1] -= min_i[1] * pixelsize

            mask_i = sarcomere_mask(_pos_vectors_i, orientations_i, lengths_i, size_i,
                                   pixelsize=pixelsize,
                                   dilation_radius=dilation_radius)
            area_i = np.sum(mask_i) * pixelsize ** 2
            if area_i >= area_min:
                ind_i = np.where(mask_i)
                ind_i = (ind_i[0] + min_i[0], ind_i[1] + min_i[1])
                mask_domains[ind_i] = j
                area_i = np.sum(mask_i) * pixelsize ** 2
                area_domains.append(area_i)
                sarcomere_length_mean_domains.append(np.mean(lengths_i))
                sarcomere_length_std_domains.append(np.std(lengths_i))
                oop, angle = Utils.analyze_orientations(orientations_i)
                sarcomere_oop_domains.append(oop)
                sarcomere_orientation_domains.append(angle)
                j += 1

    return (mask_domains, area_domains, sarcomere_length_mean_domains, sarcomere_length_std_domains,
            sarcomere_oop_domains, sarcomere_orientation_domains)
