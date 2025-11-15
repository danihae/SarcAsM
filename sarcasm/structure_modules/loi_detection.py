"""
Lines of Interest (LOI) Detection Module

This module provides functions for detecting, filtering, clustering, and analyzing
lines of interest (LOIs) in sarcomere structures. LOIs are linear or curved paths
along myofibrils used for tracking sarcomere motion in high-speed microscopy movies.

Functions
---------
filter_lois : Filter LOIs based on geometric and morphological criteria
hausdorff_distance_lois : Compute Hausdorff distances between LOIs
cluster_lois : Perform agglomerative clustering of LOIs
fit_straight_line_to_clusters : Fit linear lines to clustered LOI points
select_longest_in_cluster : Select the longest LOI from each cluster
select_random_from_cluster : Select a random LOI from each cluster
select_random_lois : Select random LOIs without clustering
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import curve_fit
from sklearn.cluster import AgglomerativeClustering
import random


def filter_lois(
        lois: List[np.ndarray],
        loi_features: Dict[str, List],
        lois_vectors: List[np.ndarray],
        number_lims: Tuple[int, int] = (10, 100),
        length_lims: Tuple[float, float] = (0, 200),
        sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
        sarcomere_std_length_lims: Tuple[float, float] = (0, 1),
        midline_mean_length_lims: Tuple[float, float] = (0, 50),
        midline_std_length_lims: Tuple[float, float] = (0, 50),
        midline_min_length_lims: Tuple[float, float] = (0, 50)
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, List]]:
    """
    Filter Lines of Interest (LOIs) based on various geometric and morphological criteria.

    Parameters
    ----------
    lois : list of np.ndarray
        List of LOI indices into sarcomere vectors
    loi_features : dict
        Dictionary containing LOI features (n_vectors, length, sarcomere stats, etc.)
    lois_vectors : list of np.ndarray
        List of actual position vectors for each LOI
    number_lims : tuple of int, optional
        Limits of sarcomere numbers in LOI (min, max). Defaults to (10, 100).
    length_lims : tuple of float, optional
        Limits for LOI lengths (in µm) (min, max). Defaults to (0, 200).
    sarcomere_mean_length_lims : tuple of float, optional
        Limits for mean length of sarcomeres in LOI (min, max). Defaults to (1, 3).
    sarcomere_std_length_lims : tuple of float, optional
        Limits for standard deviation of sarcomere lengths in LOI (min, max). Defaults to (0, 1).
    midline_mean_length_lims : tuple of float, optional
        Limits for mean length of the midline in LOI (min, max). Defaults to (0, 50).
    midline_std_length_lims : tuple of float, optional
        Limits for standard deviation of the midline length in LOI (min, max). Defaults to (0, 50).
    midline_min_length_lims : tuple of float, optional
        Limits for minimum length of the midline in LOI (min, max). Defaults to (0, 50).

    Returns
    -------
    filtered_lois : list of np.ndarray
        Filtered LOI indices
    filtered_lois_vectors : list of np.ndarray
        Filtered position vectors
    filtered_features : dict
        Filtered features dictionary
    """
    # Convert feature lists to numpy arrays for boolean operations
    n_vectors = np.array(loi_features['n_vectors_lines'])
    length = np.array(loi_features['length_lines'])
    sarc_mean = np.array(loi_features['sarcomere_mean_length_lines'])
    sarc_std = np.array(loi_features['sarcomere_std_length_lines'])
    mid_mean = np.array(loi_features['midline_mean_length_lines'])
    mid_std = np.array(loi_features['midline_std_length_lines'])
    mid_min = np.array(loi_features['midline_min_length_lines'])

    # Apply filters based on the provided limits
    is_good = (
            (n_vectors >= number_lims[0]) & (n_vectors < number_lims[1]) &
            (length >= length_lims[0]) & (length < length_lims[1]) &
            (sarc_mean >= sarcomere_mean_length_lims[0]) & (sarc_mean < sarcomere_mean_length_lims[1]) &
            (sarc_std >= sarcomere_std_length_lims[0]) & (sarc_std < sarcomere_std_length_lims[1]) &
            (mid_mean >= midline_mean_length_lims[0]) & (mid_mean < midline_mean_length_lims[1]) &
            (mid_std >= midline_std_length_lims[0]) & (mid_std < midline_std_length_lims[1]) &
            (mid_min >= midline_min_length_lims[0]) & (mid_min < midline_min_length_lims[1])
    )

    # Filter the lines and vectors
    filtered_lois = [loi for i, loi in enumerate(lois) if is_good[i]]
    filtered_lois_vectors = [pos_vectors for i, pos_vectors in enumerate(lois_vectors) if is_good[i]]

    # Filter the features dataframe and convert back to dict
    df_features = pd.DataFrame(loi_features)
    filtered_df_features = df_features[is_good].reset_index(drop=True)
    filtered_features = filtered_df_features.to_dict(orient='list')

    return filtered_lois, filtered_lois_vectors, filtered_features


def hausdorff_distance_lois(lines_vectors: List[np.ndarray], symmetry_mode: str = 'max') -> np.ndarray:
    """
    Compute Hausdorff distances between all LOIs.

    The Hausdorff distance measures how far two sets of points are from each other.
    It's used to quantify similarity between LOI trajectories.

    Parameters
    ----------
    lines_vectors : list of np.ndarray
        List of position vectors for each LOI
    symmetry_mode : {'min', 'max'}, optional
        Whether to use min or max of H(loi_i, loi_j) and H(loi_j, loi_i). Defaults to 'max'.

    Returns
    -------
    hausdorff_dist_matrix : np.ndarray
        Symmetric matrix of pairwise Hausdorff distances
    """
    n_lois = len(lines_vectors)
    hausdorff_dist_matrix = np.zeros((n_lois, n_lois))

    for i, loi_i in enumerate(lines_vectors):
        for j, loi_j in enumerate(lines_vectors):
            if symmetry_mode == 'min':
                hausdorff_dist_matrix[i, j] = min(
                    directed_hausdorff(loi_i, loi_j)[0],
                    directed_hausdorff(loi_j, loi_i)[0]
                )
            elif symmetry_mode == 'max':
                hausdorff_dist_matrix[i, j] = max(
                    directed_hausdorff(loi_i, loi_j)[0],
                    directed_hausdorff(loi_j, loi_i)[0]
                )
            else:
                raise ValueError(f"symmetry_mode must be 'min' or 'max', got '{symmetry_mode}'")

    return hausdorff_dist_matrix


def cluster_lois(
        hausdorff_dist_matrix: np.ndarray,
        distance_threshold: float = 40,
        linkage: str = 'single'
) -> Tuple[np.ndarray, int]:
    """
    Perform agglomerative clustering of LOIs using Hausdorff distance matrix.

    Parameters
    ----------
    hausdorff_dist_matrix : np.ndarray
        Precomputed pairwise distance matrix
    distance_threshold : float, optional
        The linkage distance threshold above which clusters will not be merged. Defaults to 40.
    linkage : {'complete', 'average', 'single'}, optional
        Which linkage criterion to use:
        - 'single' uses the minimum of distances between all observations of the two sets
        - 'average' uses the average of the distances of each observation of the two sets
        - 'complete' uses the maximum distances between all observations of the two sets
        Defaults to 'single'.

    Returns
    -------
    cluster_labels : np.ndarray
        Cluster label for each LOI
    n_clusters : int
        Number of unique clusters
    """
    n_lois = hausdorff_dist_matrix.shape[0]

    if n_lois == 0:
        return np.array([]), 0
    elif n_lois == 1:
        return np.array([0]), 1
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage=linkage
        ).fit(hausdorff_dist_matrix)

        cluster_labels = clustering.labels_
        n_clusters = len(np.unique(cluster_labels))

        return cluster_labels, n_clusters


def fit_straight_line_to_clusters(
        lines_vectors: List[np.ndarray],
        cluster_labels: np.ndarray,
        n_clusters: int,
        pixelsize: float,
        add_length: float = 1.0,
        n_lois: int = None
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Fit linear lines to clustered LOI points.

    For each cluster, fits a linear regression to all points and creates a line
    that spans the extent of the cluster with optional extension.

    Parameters
    ----------
    lines_vectors : list of np.ndarray
        List of position vectors for each LOI
    cluster_labels : np.ndarray
        Cluster label for each LOI
    n_clusters : int
        Number of clusters
    pixelsize : float
        Pixel size in micrometers
    add_length : float, optional
        Length to extend line at each end (in micrometers). Defaults to 1.0.
    n_lois : int, optional
        If specified, only the n longest LOIs are returned. If None, all are returned.

    Returns
    -------
    loi_lines : list of np.ndarray
        List of fitted line coordinates [(y0, x0), (y1, x1)]
    len_loi_lines : list of float
        Length of each fitted line in pixels
    """

    def linear(x, a, b):
        return a * x + b

    add_length_px = add_length / pixelsize
    loi_lines = []
    len_loi_lines = []

    for label_i in range(n_clusters):
        # Collect all points from this cluster
        points_cluster_i = []
        for k in np.where(cluster_labels == label_i)[0]:
            points_cluster_i.append(lines_vectors[k])
        points_cluster_i = np.concatenate(points_cluster_i).T

        # Fit linear regression
        p_i, _ = curve_fit(linear, points_cluster_i[1], points_cluster_i[0])

        # Create line spanning cluster extent plus extension
        x_min = np.min(points_cluster_i[1]) - add_length_px / np.sqrt(1 + p_i[0] ** 2)
        x_max = np.max(points_cluster_i[1]) + add_length_px / np.sqrt(1 + p_i[0] ** 2)
        x_range_i = np.linspace(x_min, x_max, num=2)
        y_i = linear(x_range_i, p_i[0], p_i[1])

        # Calculate line length
        len_i = np.sqrt(np.diff(x_range_i) ** 2 + np.diff(y_i) ** 2)[0]

        # Round coordinates
        x_range_i, y_i = np.round(x_range_i, 1), np.round(y_i, 1)
        loi_lines.append(np.asarray((y_i, x_range_i)).T)
        len_loi_lines.append(len_i)

    len_loi_lines = np.asarray(len_loi_lines).flatten()
    loi_lines = np.asarray(loi_lines, dtype=object)

    # Sort lines by length (longest first)
    length_idxs = len_loi_lines.argsort()[::-1]
    loi_lines = loi_lines[length_idxs]
    len_loi_lines = len_loi_lines[length_idxs]

    # Select top n if specified
    if n_lois is not None:
        loi_lines = loi_lines[:n_lois]
        len_loi_lines = len_loi_lines[:n_lois]

    return list(loi_lines), list(len_loi_lines)


def select_longest_in_cluster(
        lines: List[np.ndarray],
        pos_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        n_lois: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Select the longest LOI from each cluster.

    Parameters
    ----------
    lines : list of np.ndarray
        List of LOI indices
    pos_vectors : np.ndarray
        Position vectors array
    cluster_labels : np.ndarray
        Cluster label for each LOI
    n_clusters : int
        Number of clusters
    n_lois : int
        Maximum number of LOIs to return

    Returns
    -------
    loi_lines : list of np.ndarray
        Selected LOI position vectors
    len_loi_lines : list of int
        Length (number of points) of each LOI
    """
    longest_lines = []

    for label_i in range(n_clusters):
        # Get all lines in this cluster
        lines_cluster_i = [line_j for j, line_j in enumerate(lines) if cluster_labels[j] == label_i]
        points_lines_cluster_i = [pos_vectors[line_j] for j, line_j in enumerate(lines) if
                                  cluster_labels[j] == label_i]
        length_lines_cluster_i = [len(line_j) for line_j in lines_cluster_i]

        # Select longest
        longest_line = points_lines_cluster_i[np.argmax(length_lines_cluster_i)]
        longest_lines.append(longest_line)

    # Sort by length and select top n
    sorted_by_length = sorted(longest_lines, key=lambda x: len(x), reverse=True)
    if len(longest_lines) < n_lois:
        print(f'Only {len(longest_lines)}<{n_lois} clusters identified.')

    loi_lines = sorted_by_length[:n_lois]
    len_loi_lines = [len(line_i) for line_i in loi_lines]

    return loi_lines, len_loi_lines


def select_random_from_cluster(
        lines: List[np.ndarray],
        pos_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        n_lois: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Select a random LOI from each cluster.

    Parameters
    ----------
    lines : list of np.ndarray
        List of LOI indices
    pos_vectors : np.ndarray
        Position vectors array
    cluster_labels : np.ndarray
        Cluster label for each LOI
    n_clusters : int
        Number of clusters
    n_lois : int
        Number of LOIs to randomly select from available clusters

    Returns
    -------
    loi_lines : list of np.ndarray
        Selected LOI position vectors
    len_loi_lines : list of int
        Length (number of points) of each LOI
    """
    random_lines = []

    for label_i in range(n_clusters):
        # Get all lines in this cluster
        points_lines_cluster_i = [pos_vectors[line_j] for j, line_j in enumerate(lines) if
                                  cluster_labels[j] == label_i]
        # Select one randomly
        random_line = random.choice(points_lines_cluster_i)
        random_lines.append(random_line)

    # Randomly select n_lois from the available clusters
    loi_lines = random.sample(random_lines, min(n_lois, len(random_lines)))
    len_loi_lines = [len(line_i) for line_i in loi_lines]

    return loi_lines, len_loi_lines


def select_random_lois(
        lines: List[np.ndarray],
        pos_vectors: np.ndarray,
        n_lois: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Select random LOIs without clustering.

    Parameters
    ----------
    lines : list of np.ndarray
        List of LOI indices
    pos_vectors : np.ndarray
        Position vectors array
    n_lois : int
        Number of LOIs to randomly select

    Returns
    -------
    loi_lines : list of np.ndarray
        Selected LOI position vectors
    len_loi_lines : list of int
        Length (number of points) of each LOI
    """
    selected_lines = random.sample(lines, min(n_lois, len(lines)))
    loi_lines = [pos_vectors[line_i] for line_i in selected_lines]
    len_loi_lines = [len(line_i) for line_i in loi_lines]

    return loi_lines, len_loi_lines
