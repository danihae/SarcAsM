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
        span of ~5 frames. ``0`` returns the input unchanged.
    mode : str, optional
        Boundary condition. ``'nearest'`` (default) extends the end values;
        ``'reflect'`` and ``'mirror'`` are also accepted (mapped to the
        backend's equivalent).
    backend : str, optional
        ``'scipy'`` (default, fastest on CPU for typical stacks) uses
        ``scipy.ndimage.gaussian_filter1d``.  ``'torch'`` uses a depthwise 1D
        convolution (useful for very large stacks on CUDA).
    device : str, optional
        When ``backend='torch'``: ``'auto'`` picks cuda / mps / cpu; can also
        be ``'cpu'``, ``'cuda'``, ``'mps'``.

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
        peak_prominence: float = 0.5,
        peak_algorithm: str = 'default',
        precomputed_angle_map: Union[np.ndarray, None] = None,
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
    interpolation_method : str, optional
        Interpolation method: 'linear' (fast) or 'akima' (smooth). Defaults to 'linear'.
    peak_prominence : float, optional
        ``scipy.signal.find_peaks`` prominence threshold used inside
        :func:`Utils.process_profiles_batch`. Default 0.5 (matches LOI). Only
        used when ``peak_algorithm='default'``.
    peak_algorithm : {'default', 'loi'}, optional
        Which peak-detection routine to apply to each profile.

        * ``'default'`` — :func:`Utils.process_profiles_batch` (fast, batched,
          with ``peak_prominence`` + ``interp_factor`` + ``interpolation_method``
          configurable).
        * ``'loi'`` — route every profile through :func:`Utils.peakdetekt`, the
          exact peak-detection + Akima-upsampling + COM-refinement pipeline used
          by the LOI analysis. Parameter presets match LOI (``interp_factor=6``,
          ``prominence=0.5``, Akima). Slightly slower.

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
    if peak_algorithm not in ('default', 'loi'):
        raise ValueError(f"peak_algorithm must be 'default' or 'loi'; got {peak_algorithm!r}")
    radius_pixels = max(int(round(median_filter_radius / pixelsize, 0)), 1)
    linewidth_pixels = max(int(round(linewidth / pixelsize, 0)), 1)

    # skeletonize mbands
    mbands_skel = skeletonize(mbands, method='lee')

    # calculate and preprocess orientation map
    if precomputed_angle_map is not None:
        orientation = precomputed_angle_map
    else:
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
        # Pre-calculate total number of points for efficient pre-allocation
        total_points = sum(coords.shape[0] for coords in coords_mbands)
        
        # Pre-allocate arrays (much faster than appending and concatenating)
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

        sarcomere_orientation_vectors = orientation[pos_vectors_px[:, 0], pos_vectors_px[:, 1]]

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

        # Use batch processing for better performance (avoids parallel processing overhead)
        if peak_algorithm == 'loi':
            # Route through Utils.peakdetekt — the exact pipeline the LOI
            # analysis uses. Preset parameters match LOI.
            sarcomere_length_vectors, center_offsets = Utils.process_profiles_batch_loi(
                profiles, pixelsize, slen_lims=slen_lims,
            )
        else:
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
        sarcomere_orientation_vectors = sarcomere_orientation_vectors[~nan_mask]
        sarcomere_length_vectors = sarcomere_length_vectors[~nan_mask]


    else:
        sarcomere_length_vectors, _z_band_thickness_vectors, sarcomere_orientation_vectors = [], [], []

    return (pos_vectors_px, pos_vectors, midline_id_vectors, midline_length_vectors, sarcomere_length_vectors,
            sarcomere_orientation_vectors, n_mbands)
