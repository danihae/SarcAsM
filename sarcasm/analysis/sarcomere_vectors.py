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
from scipy.spatial import cKDTree
from skimage import measure
from skimage.morphology import disk, skeletonize

from sarcasm.utils import Utils


def _axial_double_angle_encoding(orientation_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(cos 2θ, sin 2θ)`` stacks from a ``(T, 2, H, W)`` vector field.

    The raw field stores ``(x, y)`` components; the axial-correct smoothing
    variable is the double-angle vector, which is unique per line axis.
    """
    x = orientation_field[:, 0].astype(np.float32)
    y = orientation_field[:, 1].astype(np.float32)
    angles = np.arctan2(y, x)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    angles = np.where(angles > np.pi, angles - np.pi, angles)
    return np.cos(2 * angles), np.sin(2 * angles)


def _axial_double_angle_decode(
    cos2_sm: np.ndarray,
    sin2_sm: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Inverse of :func:`_axial_double_angle_encoding`. Re-emits ``(x, y)``
    components as ``(cos θ, sin θ)`` so downstream consumers can keep using
    ``arctan2`` as-is."""
    doubled = np.arctan2(sin2_sm, cos2_sm)
    smoothed = (doubled / 2.0 + np.pi) % np.pi
    out = np.empty((cos2_sm.shape[0], 2, *cos2_sm.shape[1:]), dtype=dtype)
    out[:, 0] = np.cos(smoothed).astype(dtype)
    out[:, 1] = np.sin(smoothed).astype(dtype)
    return out


def smooth_orientation_field_temporal(
    orientation_field: np.ndarray,
    sigma: float,
    mode: str = 'nearest',
    backend: str = 'scipy',
    device: str = 'auto',
) -> np.ndarray:
    """Temporally smooth a stack of orientation vector fields, axially correct.

    The U-Net predicts sarcomere orientation frame-by-frame, producing small
    per-frame jitter that propagates into vector positions, lengths, and
    downstream tracking. Averaging the raw ``(x, y)`` channels naively would
    destroy the signal because axial orientations can be encoded by either
    ``(cos θ, sin θ)`` or its negative — both describe the same axis.

    Implementation uses the double-angle trick:

    1. Compute axial angles per frame from ``arctan2(field[:, 1], field[:, 0])``
       and wrap to ``[0, π)``.
    2. Encode as ``(cos 2θ, sin 2θ)`` — a unique vector per axis.
    3. Gaussian-smooth both channels along the time axis (scipy or torch).
    4. Recover a smoothed axial angle via ``arctan2(smoothed_sin2, smoothed_cos2) / 2``.
    5. Re-emit the field as ``(cos θ, sin θ)`` so downstream consumers can keep
       calling ``arctan2`` as before.

    Parameters
    ----------
    orientation_field : np.ndarray
        ``(T, 2, H, W)`` stack of orientation vector fields. Channel 0 is the
        x-component, channel 1 is the y-component.
    sigma : float
        Gaussian sigma in frames. ``sigma ≈ 1`` corresponds to an effective
        span of ~5 frames; ``0`` returns the input unchanged.
    mode : str, optional
        Boundary condition: ``'nearest'`` extends the end values; ``'reflect'``
        and ``'mirror'`` are also accepted (mapped to the backend's
        equivalent). Default is 'nearest'.
    backend : {'scipy', 'torch'}, optional
        ``'scipy'`` (fastest on CPU for typical stacks) uses
        ``scipy.ndimage.gaussian_filter1d``; ``'torch'`` uses a depthwise 1D
        convolution (useful for very large stacks on CUDA). Default is 'scipy'.
    device : str, optional
        Used when ``backend='torch'``: ``'auto'`` picks cuda / mps / cpu; can
        also be ``'cpu'``, ``'cuda'``, ``'mps'``. Default is 'auto'.

    Returns
    -------
    np.ndarray
        Smoothed orientation field, same shape and dtype as input.
    """
    if sigma <= 0:
        return orientation_field
    if orientation_field.ndim != 4 or orientation_field.shape[1] != 2:
        raise ValueError(
            f"orientation_field must have shape (T, 2, H, W); got {orientation_field.shape}"
        )
    if orientation_field.shape[0] < 2:
        return orientation_field

    cos2, sin2 = _axial_double_angle_encoding(orientation_field)

    if backend == 'scipy':
        cos2_sm = ndimage.gaussian_filter1d(cos2, sigma=sigma, axis=0, mode=mode)
        sin2_sm = ndimage.gaussian_filter1d(sin2, sigma=sigma, axis=0, mode=mode)
    elif backend == 'torch':
        import torch
        import torch.nn.functional as F
        if device == 'auto':
            if torch.cuda.is_available():
                dev = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                dev = torch.device('mps')
            else:
                dev = torch.device('cpu')
        else:
            dev = torch.device(device)
        T, _, H, W = orientation_field.shape
        # Pack (cos2, sin2) into (H*W, 2, T) so a single depthwise conv1d
        # smooths both double-angle components at once.
        stacked = np.stack([cos2, sin2], axis=0).transpose(2, 3, 0, 1)
        stacked = stacked.reshape(H * W, 2, T).astype(np.float32, copy=False)
        t = torch.from_numpy(stacked).to(dev)
        half = max(1, int(np.ceil(3.0 * sigma)))
        ks = torch.arange(-half, half + 1, dtype=torch.float32, device=dev)
        kernel = torch.exp(-0.5 * (ks / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1).expand(2, 1, -1).contiguous()
        pad_mode = 'replicate' if mode == 'nearest' else mode
        t_pad = F.pad(t, (half, half), mode=pad_mode)
        t_sm = F.conv1d(t_pad, kernel, groups=2).cpu().numpy()
        out_np = t_sm.reshape(H, W, 2, T).transpose(2, 3, 0, 1)
        cos2_sm, sin2_sm = out_np[0], out_np[1]
    else:
        raise ValueError(f"backend must be 'scipy' or 'torch'; got {backend!r}")

    return _axial_double_angle_decode(cos2_sm, sin2_sm, orientation_field.dtype)


def orientation_at_points(
        orientation_field: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        radius: int,
) -> np.ndarray:
    """Axial-median sarcomere orientation, sampled only at the given pixels.

    Equivalent to indexing :func:`Utils.get_orientation_angle_map` (disk median,
    double-angle correct) at ``(rows, cols)``, but gathers the footprint at those
    points instead of filtering the whole frame. Orientation is only needed at the
    M-band midline pixels, a small fraction of the image.

    Out-of-bounds neighbours are clipped to the edge rather than zero-padded,
    because zero is not a valid unit vector and would bias the median within
    ``radius`` px of the border.

    Parameters
    ----------
    orientation_field : np.ndarray
        ``(2, H, W)`` orientation vector field; channel 0 is x, channel 1 is y.
    rows, cols : np.ndarray
        Integer pixel coordinates at which to evaluate the orientation.
    radius : int
        Radius of the disk footprint, in pixels.

    Returns
    -------
    np.ndarray
        Axial orientation angle in radians at each point, in ``[0, π)``.
    """
    if orientation_field.ndim != 3 or orientation_field.shape[0] != 2:
        raise ValueError(
            f"orientation_field must have shape (2, H, W); got {orientation_field.shape}"
        )
    rows = np.asarray(rows)
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)

    height, width = orientation_field.shape[1], orientation_field.shape[2]

    # Same encoding as Utils.get_orientation_angle_map: wrap to the axial range,
    # then double so opposite-sign vectors describing one axis coincide.
    angles = np.arctan2(orientation_field[1], orientation_field[0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    angles = np.where(angles > np.pi, angles - np.pi, angles)
    doubled = 2 * angles
    comp_x, comp_y = np.cos(doubled), np.sin(doubled)

    d_row, d_col = np.nonzero(disk(radius, strict_radius=False))
    d_row = d_row - radius
    d_col = d_col - radius

    idx_row = np.clip(rows[:, None] + d_row[None, :], 0, height - 1)
    idx_col = np.clip(np.asarray(cols)[:, None] + d_col[None, :], 0, width - 1)

    x_med = np.median(comp_x[idx_row, idx_col], axis=1)
    y_med = np.median(comp_y[idx_row, idx_col], axis=1)

    filtered = np.arctan2(y_med, x_med)
    filtered = (filtered + 2 * np.pi) % (2 * np.pi)
    return filtered / 2


def get_sarcomere_vectors(
        zbands: np.ndarray,
        mbands: np.ndarray,
        orientation_field: np.ndarray,
        pixelsize: float,
        median_filter_radius: float = 0.25,
        slen_lims: Tuple[float, float] = (1, 3),
        interp_factor: int = 4,
        linewidth: float = 0.3,
        interpolation_method: str = 'linear',
        peak_prominence: float = 0.4,
) -> Tuple[Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List],
Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List]]:
    """Extract sarcomere orientation and length vectors.

    Parameters
    ----------
    zbands : np.ndarray
        2D semantic segmentation map of Z-bands.
    mbands : np.ndarray
        2D semantic segmentation map of M-bands.
    orientation_field : np.ndarray
        Sarcomere orientation field.
    pixelsize : float
        Pixel size in µm.
    median_filter_radius : float, optional
        Radius to smooth the orientation field before sampling orientation at
        M-points, in µm. Default is 0.25.
    slen_lims : tuple of float, optional
        Sarcomere length limits in µm. Default is (1, 3).
    interp_factor : int, optional
        Profile interpolation factor for length estimation. Default is 4.
    linewidth : float, optional
        Profile line width in µm. Default is 0.3.
    interpolation_method : str, optional
        Profile interpolation: 'linear' (fast) or 'akima' (smooth). Default is
        'linear'.
    peak_prominence : float, optional
        ``scipy.signal.find_peaks`` prominence threshold used inside
        :func:`Utils.process_profiles_batch`, applied to the per-profile
        min-max normalised profile. Default is 0.4.

    Returns
    -------
    pos_vectors_px : np.ndarray
        Midline-point positions in pixels (row, col).
    pos_vectors : np.ndarray
        Center-corrected midline-point positions in µm.
    midline_id_vectors : np.ndarray
        M-band (midline) label for each vector.
    midline_length_vectors : np.ndarray
        Midline length (max Feret diameter) per vector in µm.
    sarcomere_length_vectors : np.ndarray
        Sarcomere length at each midline point in µm.
    sarcomere_orientation_vectors : np.ndarray
        Sarcomere orientation at each midline point in radians.
    n_mbands : int
        Number of detected M-bands.
    """
    radius_pixels = max(int(round(median_filter_radius / pixelsize, 0)), 1)
    linewidth_pixels = max(int(round(linewidth / pixelsize, 0)), 1)

    # skeletonize mbands
    mbands_skel = skeletonize(mbands, method='lee')

    # label mbands
    midline_labels, n_mbands = ndimage.label(mbands_skel,
                                             ndimage.generate_binary_structure(2, 2))

    # iterate mbands and create an additional list with labels and midline length (approx. by max. Feret diameter)
    props = measure.regionprops_table(midline_labels, properties=['label', 'coords', 'feret_diameter_max'])
    list_labels, coords_mbands, length_mbands = (props['label'], props['coords'],
                                                     props['feret_diameter_max'] * pixelsize)

    # a frame without M-bands yields empty arrays of the same dimensionality as a frame with them, so the
    # per-frame lists stay homogeneous (the results store writes them as ragged arrays)
    pos_vectors_px, pos_vectors = np.empty((0, 2), dtype=np.int64), np.empty((0, 2))
    midline_id_vectors, midline_length_vectors = np.empty(0), np.empty(0)
    if n_mbands > 0:
        # Pre-calculate total number of points for efficient pre-allocation
        total_points = sum(coords.shape[0] for coords in coords_mbands)
        
        # Pre-allocate and slice-assign rather than appending per label
        pos_vectors_px = np.empty((total_points, 2), dtype=coords_mbands[0].dtype)
        midline_id_vectors = np.empty(total_points, dtype=np.float64)
        midline_length_vectors = np.empty(total_points, dtype=np.float64)
        
        # Fill arrays with vectorized operations
        idx = 0
        for label_i, coords_i, length_midline_i in zip(list_labels, coords_mbands, length_mbands):
            n_coords = coords_i.shape[0]
            pos_vectors_px[idx:idx + n_coords] = coords_i
            midline_id_vectors[idx:idx + n_coords] = label_i
            midline_length_vectors[idx:idx + n_coords] = length_midline_i
            idx += n_coords

        # Smooth and sample the orientation field at the midline points only.
        sarcomere_orientation_vectors = orientation_at_points(
            orientation_field, pos_vectors_px[:, 0], pos_vectors_px[:, 1], radius_pixels)

        # Pre-compute trigonometric values and scaling factor
        half_length_scale = (slen_lims[1] * 1.3) / 2 / pixelsize
        sin_vals = np.sin(sarcomere_orientation_vectors) * half_length_scale
        cos_vals = np.cos(sarcomere_orientation_vectors) * half_length_scale
        
        # Vectorized endpoint calculation
        direction_vectors = np.stack((sin_vals, cos_vals), axis=0)
        ends1 = pos_vectors_px.T + direction_vectors
        ends2 = pos_vectors_px.T - direction_vectors

        # Calculate sarcomere lengths by measuring peak-to-peak distance of Z-bands in intensity profile
        profiles = Utils.fast_profile_lines(zbands, ends1, ends2, linewidth=linewidth_pixels)

        sarcomere_length_vectors, center_offsets = Utils.process_profiles_batch(
            profiles, pixelsize, slen_lims=slen_lims, interp_factor=interp_factor,
            interpolation_method=interpolation_method, prominence=peak_prominence,
        )

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
        midline_length_vectors = midline_length_vectors[~nan_mask]
        sarcomere_orientation_vectors = sarcomere_orientation_vectors[~nan_mask]
        sarcomere_length_vectors = sarcomere_length_vectors[~nan_mask]


    else:
        sarcomere_length_vectors, sarcomere_orientation_vectors = np.empty(0), np.empty(0)

    return (pos_vectors_px, pos_vectors, midline_id_vectors, midline_length_vectors, sarcomere_length_vectors,
            sarcomere_orientation_vectors, n_mbands)


def neighbor_orientation_deviation(
        pos: np.ndarray,
        orientation: np.ndarray,
        radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deviation of each sarcomere vector from the mean orientation of its neighbours.

    Parameters
    ----------
    pos : np.ndarray
        Vector positions ``(n, 2)`` in µm.
    orientation : np.ndarray
        Vector orientations ``(n,)`` in radians. Orientations are axial: θ and
        θ + π denote the same sarcomere axis.
    radius : float
        Neighbourhood radius in µm.

    Returns
    -------
    deviation : np.ndarray
        Axial deviation in degrees, in ``[0, 90]``, of each vector from the
        circular mean orientation of the other vectors within ``radius``. NaN
        for vectors without a neighbour.
    n_neighbors : np.ndarray
        Number of vectors within ``radius`` of each vector.
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    theta = np.asarray(orientation, dtype=float).reshape(-1)
    n = len(theta)
    deviation = np.full(n, np.nan)
    n_neighbors = np.zeros(n, dtype=np.int64)
    valid = np.flatnonzero(np.isfinite(theta) & np.isfinite(pos).all(axis=1))
    if len(valid) < 2:
        return deviation, n_neighbors
    pairs = cKDTree(pos[valid]).query_pairs(radius, output_type='ndarray')
    if len(pairs) == 0:
        return deviation, n_neighbors
    i = valid[np.concatenate([pairs[:, 0], pairs[:, 1]])]
    j = valid[np.concatenate([pairs[:, 1], pairs[:, 0]])]
    # mean axis through the double-angle vector, which is the same point for θ and θ + π
    mean_vector = np.zeros(n, dtype=complex)
    np.add.at(mean_vector, i, np.exp(2j * theta[j]))
    np.add.at(n_neighbors, i, 1)
    has_neighbors = n_neighbors > 0
    mean_theta = np.angle(mean_vector[has_neighbors]) / 2
    deviation[has_neighbors] = np.degrees(
        np.abs((theta[has_neighbors] - mean_theta + np.pi / 2) % np.pi - np.pi / 2))
    return deviation, n_neighbors


def filter_vectors_by_neighbors(
        pos: np.ndarray,
        orientation: np.ndarray,
        neighbor_radius: float = 1.0,
        max_neighbor_deviation: float = 30.0,
) -> np.ndarray:
    """Mark sarcomere vectors that point across their neighbours.

    A vector whose orientation deviates by more than ``max_neighbor_deviation``
    from the mean orientation of the vectors within ``neighbor_radius`` is a
    false positive in nearly every case: an M-band fragment at the crossing of
    differently oriented myofibrils, or a stray detection. Vectors without any
    neighbour inside the radius are kept, so that short or thin but correct
    M-bands are not penalised for the number of points detected on them.

    Parameters
    ----------
    pos : np.ndarray
        Vector positions ``(n, 2)`` in µm.
    orientation : np.ndarray
        Vector orientations ``(n,)`` in radians.
    neighbor_radius : float, optional
        Neighbourhood radius in µm. Default is 1.0.
    max_neighbor_deviation : float, optional
        Largest tolerated deviation from the neighbours' mean orientation, in
        degrees. Default is 30.0.

    Returns
    -------
    keep : np.ndarray
        Boolean mask, True for the vectors to keep.
    """
    deviation, _ = neighbor_orientation_deviation(pos, orientation, neighbor_radius)
    return ~(deviation > max_neighbor_deviation)


# --- Z-band registry --------------------------------------------------------

# implementation constants of zband_registry (not parameters): one vector per cell of this size is scored (the
# vectors sample each M-band about one pixel apart), the sarcomere length of a vector is the median over the
# vectors within this radius, and a neighbour counts as "in the row" below this distance off the axis
REGISTRY_THIN_CELL = 0.3      # µm
REGISTRY_LENGTH_RADIUS = 1.0  # µm
REGISTRY_ROW_WIDTH = 0.5      # µm


def _zband_phase_score(pos, orientation, slen, I, J):
    """Z-disc phase score of the neighbour pairs (I, J): cos(2π x / L_i) of the offset x of each of j's two Z-discs
    from the nearest Z-disc line of i (i's Z-discs repeated every L_i along its axis), averaged over the two discs.
    Returns (a, b, dtheta, score): offset of j's centre along i's axis, across it, axial angle between the two."""
    t = np.stack([np.sin(orientation), np.cos(orientation)], axis=1)
    n = np.stack([np.cos(orientation), -np.sin(orientation)], axis=1)
    d = pos[J] - pos[I]
    a = np.einsum('ij,ij->i', d, t[I])
    b = np.abs(np.einsum('ij,ij->i', d, n[I]))
    dtheta = np.abs(np.angle(np.exp(2j * (orientation[J] - orientation[I])))) / 2   # axial: θ and θ + π agree
    L_i, L_j = slen[I], slen[J]
    # the discs of j lie at a ± L_j / 2 along i's axis; the lines of i at ± L_i / 2 + k L_i
    score = -0.5 * (np.cos(2 * np.pi * (a - L_j / 2) / L_i) + np.cos(2 * np.pi * (a + L_j / 2) / L_i))
    return a, b, dtheta, score


def zband_registry(
        pos: np.ndarray,
        orientation: np.ndarray,
        slen: np.ndarray,
        lateral_band: Tuple[float, float] = (1.5, 2.5),
        axial_range: int = 3,
        max_angle: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lateral and axial Z-band registry of every sarcomere vector.

    Each sarcomere vector (centre ``pos``, axis ``orientation``, length ``L``)
    defines a Z-disc lattice: its two Z-discs at ``± L / 2`` along the axis,
    repeated every ``L``. A neighbouring sarcomere's Z-disc at offset ``x``
    along that axis from the nearest lattice line scores ``cos(2π x / L)``,
    averaged over the neighbour's two discs: 1 when the Z-discs are level with
    the vector's own, 0 on average for random positions, −1 when they sit at
    the M-band position (half a sarcomere off).

    The **lateral** registry is the mean score over the neighbours one
    myofibril over: ``lateral_band[0]`` to ``lateral_band[1]`` µm across the
    axis and within ``L / 2`` along it, whatever their orientation (a tilted
    neighbour lowers the score through the phase of its projected discs). The
    **axial** registry is the mean score over the neighbours along the
    vector's own row (less than 0.5 µm off the axis) that are 1 to
    ``axial_range`` sarcomeres away and parallel within ``max_angle``. A
    vector without a neighbour of the required kind gets NaN, never 0.

    ``L`` is the median vector length within 1 µm, and one vector per
    0.3 µm cell is scored (the vectors sample each M-band about a pixel
    apart); the scored value is assigned to every vector of the cell.

    Parameters
    ----------
    pos : np.ndarray
        Vector positions ``(n, 2)`` in µm.
    orientation : np.ndarray
        Vector orientations ``(n,)`` in radians (axial, ``[0, π)``).
    slen : np.ndarray
        Sarcomere lengths ``(n,)`` in µm.
    lateral_band : tuple of float, optional
        Distance across the axis, in µm, within which a neighbour counts as
        one myofibril over. Default is (1.5, 2.5).
    axial_range : int, optional
        Farthest neighbour along the row, in sarcomeres. Default is 3.
    max_angle : float, optional
        Largest axial angle, in degrees, between a row neighbour and the
        vector for it to count. Default is 20.0.

    Returns
    -------
    lateral, axial : np.ndarray
        Registry per vector ``(n,)``, in ``[-1, 1]``, NaN without neighbours.
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    orientation = np.asarray(orientation, dtype=float).ravel()
    slen = np.asarray(slen, dtype=float).ravel()
    n_all = len(pos)
    lateral = np.full(n_all, np.nan)
    axial = np.full(n_all, np.nan)
    ok = np.isfinite(pos).all(axis=1) & np.isfinite(orientation) & np.isfinite(slen) & (slen > 0)
    if ok.sum() < 2:
        return lateral, axial
    idx_ok = np.flatnonzero(ok)
    p, o, L = pos[ok], orientation[ok], slen[ok]

    # one vector per thinning cell; the cell's value is given to all of its vectors afterwards
    cell = np.floor(p / REGISTRY_THIN_CELL).astype(np.int64)
    key = cell[:, 0] * (10 ** 7) + cell[:, 1]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    ps, os_, Ls = p[first], o[first], L[first]
    n = len(ps)
    tree = cKDTree(ps)

    # local sarcomere length: median over the thinned vectors within REGISTRY_LENGTH_RADIUS
    if n > 1:
        pairs = tree.query_pairs(REGISTRY_LENGTH_RADIUS, output_type='ndarray')
        I = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        J = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        order = np.argsort(I, kind='stable')
        I, J = I[order], J[order]
        counts = np.bincount(I, minlength=n)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        M = np.full((n, int(counts.max())), np.nan)
        M[I, np.arange(len(I)) - np.repeat(start, counts)] = Ls[J]
        Ls = np.nanmedian(M, axis=1)
    L_max = float(np.max(Ls))

    def _mean_by_source(I, mask, score):
        s = np.bincount(I[mask], weights=score[mask], minlength=n)
        c = np.bincount(I[mask], minlength=n)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    # lateral: neighbours in the band across the axis and within L / 2 along it
    radius = float(np.hypot(lateral_band[1], 0.5 * L_max))
    pairs = tree.query_pairs(radius, output_type='ndarray')
    if len(pairs):
        I = np.concatenate([pairs[:, 0], pairs[:, 1]])
        J = np.concatenate([pairs[:, 1], pairs[:, 0]])
        a, b, _, score = _zband_phase_score(ps, os_, Ls, I, J)
        beside = (np.abs(a) <= Ls[I] / 2) & (b >= lateral_band[0]) & (b < lateral_band[1])
        lat_thin = _mean_by_source(I, beside, score)
    else:
        lat_thin = np.full(n, np.nan)

    # axial: the parallel neighbours 1..axial_range sarcomeres along the row, found from the lattice positions
    t = np.stack([np.sin(os_), np.cos(os_)], axis=1)
    ks = np.arange(1, int(axial_range) + 1)
    query = np.concatenate([ps + sign * k * Ls[:, None] * t for k in ks for sign in (1.0, -1.0)])
    source = np.tile(np.arange(n), 2 * len(ks))
    k_of = np.repeat(ks, 2 * n)
    reach = float(np.hypot(0.5 * L_max, REGISTRY_ROW_WIDTH))
    hits = cKDTree(query).sparse_distance_matrix(tree, reach, output_type='coo_matrix')
    I, J, K = source[hits.row], hits.col, k_of[hits.row]
    keep = I != J
    I, J, K = I[keep], J[keep], K[keep]
    if len(I):
        a, b, dtheta, score = _zband_phase_score(ps, os_, Ls, I, J)
        in_row = ((b < REGISTRY_ROW_WIDTH) & (np.rint(np.abs(a) / Ls[I]) == K)
                  & (dtheta <= np.radians(max_angle)))
        ax_thin = _mean_by_source(I, in_row, score)
    else:
        ax_thin = np.full(n, np.nan)

    lateral[idx_ok] = lat_thin[inverse]
    axial[idx_ok] = ax_thin[inverse]
    return lateral, axial



# Below this ratio of the raw image's stripe period to the measured sarcomere length the pixel size is suspect.
MIN_STRIPE_RATIO = 0.75


def stripe_period(image: np.ndarray, pos_px: np.ndarray, orientation: np.ndarray, slen_px: float,
                  n_profiles: int = 500, seed: int = 0) -> float:
    """Repeat distance of the raw image's stripes along the sarcomere axis, in pixels.

    Samples the raw intensity along the sarcomere axis of up to ``n_profiles`` vectors (± 2.5 measured
    sarcomere lengths around each centre), averages the profiles' autocorrelations and returns the lag of
    the first peak. It reads the pixels only, so it does not depend on the pixel size, and it does not use
    the U-Net output beyond where to look and in which direction.

    Parameters
    ----------
    image : np.ndarray
        Raw image of one frame, shape ``(H, W)``.
    pos_px : np.ndarray
        Vector centres ``(N, 2)`` as (row, col) pixels.
    orientation : np.ndarray
        Sarcomere orientation of each vector (rad); the axis is ``(sin o, cos o)`` in (row, col).
    slen_px : float
        Median measured sarcomere length in pixels; sets the profile length.
    n_profiles : int, optional
        Number of vectors sampled. Default is 500.
    seed : int, optional
        Seed of the random vector sample. Default is 0.

    Returns
    -------
    float
        Stripe period in pixels, NaN if there is no periodic signal (no autocorrelation peak above 0.05
        at a lag of 3 px or more).
    """
    pos_px = np.asarray(pos_px, dtype=float).reshape(-1, 2)
    orientation = np.asarray(orientation, dtype=float).reshape(-1)
    ok = np.isfinite(orientation) & np.isfinite(pos_px).all(axis=1)
    pos_px, orientation = pos_px[ok], orientation[ok]
    if len(pos_px) == 0 or not np.isfinite(slen_px) or slen_px <= 0:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pos_px), min(n_profiles, len(pos_px)), replace=False)
    step = 0.25
    t = np.arange(-2.5 * slen_px, 2.5 * slen_px + step, step)
    rows = pos_px[idx, :1] + np.sin(orientation[idx])[:, None] * t
    cols = pos_px[idx, 1:] + np.cos(orientation[idx])[:, None] * t
    profiles = ndimage.map_coordinates(np.asarray(image, dtype=float), [rows.ravel(), cols.ravel()],
                                       order=1, mode='nearest').reshape(rows.shape)
    profiles -= profiles.mean(axis=1, keepdims=True)
    power = np.sum(profiles ** 2, axis=1)
    profiles = profiles[power > 0]
    if len(profiles) == 0:
        return np.nan
    n = profiles.shape[1]
    spectrum = np.fft.rfft(profiles, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :n]
    ac = np.mean(ac / ac[:, :1], axis=0)
    lags = np.arange(n) * step
    peaks = np.flatnonzero((ac[1:-1] > ac[:-2]) & (ac[1:-1] >= ac[2:]) & (ac[1:-1] > 0.05)) + 1
    peaks = peaks[lags[peaks] >= 3]             # a stripe period below 3 px cannot be resolved
    return float(lags[peaks[0]]) if peaks.size else np.nan


def pixelsize_plausibility(period_px: float, slen_um: float, pixelsize: float) -> Union[str, None]:
    """Warning text when the pixel size looks about 2x too small, else None.

    A Z-disc label repeats once per sarcomere, so the raw image's stripe period should equal the measured
    sarcomere length. If it is about half of it, the pixel size is too small: the detection pairs every other
    Z-disc, the lengths still look normal, and every trajectory jumps by half a sarcomere.

    Parameters
    ----------
    period_px : float
        Stripe period of the raw image in pixels (:func:`stripe_period`).
    slen_um : float
        Median measured sarcomere length (µm).
    pixelsize : float
        Pixel size used for the analysis (µm).

    Returns
    -------
    str or None
        Warning text, None if the ratio is fine or the period could not be measured.
    """
    if not (np.isfinite(period_px) and np.isfinite(slen_um) and slen_um > 0):
        return None
    ratio = period_px * pixelsize / slen_um
    if ratio >= MIN_STRIPE_RATIO:
        return None
    return (f'Are you sure that {pixelsize:.4g} µm/px is the right pixel size? The image stripes repeat every '
            f'{period_px * pixelsize:.2f} µm, {ratio:.2f}x the measured sarcomere length of {slen_um:.2f} µm. '
            f'This happens when the pixel size is too small (about {pixelsize / ratio:.3g} µm/px would match): '
            f'the lengths look normal, but the trajectories jump by half a sarcomere. Ignore this if the label '
            f'shows two stripes per sarcomere.')
