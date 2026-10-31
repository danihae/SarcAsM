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

"""Miscellaneous helper utilities for SarcAsM (filtering, profiles, geometry, I/O)."""

import logging
import os
import warnings
from pathlib import Path
from typing import Tuple, List, Union
os.environ["KMP_WARNINGS"] = "False"
warnings.filterwarnings("ignore")
import numpy as np
import torch
import igraph as ig
from numba import njit, prange
from scipy.interpolate import Akima1DInterpolator

logger = logging.getLogger(__name__)
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from skimage.morphology import disk



def _batched_linear_interp(
    x: np.ndarray, y: np.ndarray, x_new: np.ndarray,
) -> np.ndarray:
    """Batched linear interpolation equivalent to per-row ``np.interp``.

    Parameters
    ----------
    x : np.ndarray
        Shared 1-D sample positions, shape ``(L,)``.
    y : np.ndarray
        Values to interpolate, shape ``(N, L)``.
    x_new : np.ndarray
        1-D positions to interpolate at.

    Returns
    -------
    np.ndarray
        Interpolated values, shape ``(N, len(x_new))``. Matches ``np.interp``
        including constant extrapolation at the boundaries.
    """
    L = x.shape[0]
    idx = np.clip(np.searchsorted(x, x_new, side='right') - 1, 0, L - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    w = (x_new - x0) / (x1 - x0)
    # Clamp for exact endpoint / out-of-range behaviour to match np.interp.
    w = np.clip(w, 0.0, 1.0)
    return y[:, idx] * (1.0 - w) + y[:, idx + 1] * w


@njit(cache=True, parallel=True)
def _akima_slopes(x, y):
    """Node slopes of scipy's ``Akima1DInterpolator(x, y, axis=1)`` (``method='akima'``), row by row.

    Same operations in the same order as scipy, so the slopes are bit-identical, including scipy's
    break threshold ``1e-9 * max(f1 + f2)``, which is taken over the whole batch, not per row.
    ``y`` is float64 ``(N, L)`` with ``L >= 3``."""
    n, L = y.shape
    t = np.empty((n, L))
    f1 = np.empty((n, L))
    f2 = np.empty((n, L))
    m = np.empty((n, L + 3))
    row_max = np.full(n, -np.inf)
    row_nan = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        for j in range(L - 1):
            m[i, j + 2] = (y[i, j + 1] - y[i, j]) / (x[j + 1] - x[j])
        m[i, 1] = 2. * m[i, 2] - m[i, 3]
        m[i, 0] = 2. * m[i, 1] - m[i, 2]
        m[i, L + 1] = 2. * m[i, L] - m[i, L - 1]
        m[i, L + 2] = 2. * m[i, L + 1] - m[i, L]
        for j in range(L):
            t[i, j] = .5 * (m[i, j + 3] + m[i, j])
            f1[i, j] = abs(m[i, j + 3] - m[i, j + 2])
            f2[i, j] = abs(m[i, j + 1] - m[i, j])
            f12 = f1[i, j] + f2[i, j]
            if np.isnan(f12):
                row_nan[i] = True
            elif f12 > row_max[i]:
                row_max[i] = f12
    if row_nan.any():          # np.max propagates NaN, and nothing compares greater than NaN
        return t
    threshold = 1.e-9 * row_max.max()
    for i in prange(n):
        for j in range(L):
            f12 = f1[i, j] + f2[i, j]
            if f12 > threshold:
                t[i, j] = m[i, j + 1] + ((f2[i, j] / f12) * (m[i, j + 2] - m[i, j + 1]))
    return t


@njit(cache=True, parallel=True)
def _cubic_hermite_eval(x, y, t, x_new, segment):
    """Evaluate the cubic Hermite spline through ``(x, y[i])`` with node slopes ``t[i]`` at ``x_new``, as
    scipy's ``CubicHermiteSpline`` / ``PPoly`` does (same coefficients, same power-sum evaluation);
    ``segment`` is the interval index of each ``x_new``."""
    n = y.shape[0]
    out = np.empty((n, x_new.shape[0]))
    for i in prange(n):
        for q in range(x_new.shape[0]):
            j = segment[q]
            dx = x[j + 1] - x[j]
            slope = (y[i, j + 1] - y[i, j]) / dx
            c_t = (t[i, j] + t[i, j + 1] - 2 * slope) / dx
            c0 = c_t / dx
            c1 = (slope - t[i, j]) / dx - c_t
            s = x_new[q] - x[j]
            res = 0.0
            z = 1.0
            res = res + y[i, j] * z
            z *= s
            res = res + t[i, j] * z
            z *= s
            res = res + c1 * z
            z *= s
            res = res + c0 * z
            out[i, q] = res
    return out


def _batched_akima(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """``Akima1DInterpolator(x, y, axis=1)(x_new)``, bit-identical, for ``x_new`` inside ``[x[0], x[-1]]``.

    Compiled and parallel instead of scipy's array passes: about 8x faster on a frame's ~4,500 profiles."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    x_new = np.ascontiguousarray(x_new, dtype=np.float64)
    if x.shape[0] < 3:                       # scipy interpolates two points linearly
        return Akima1DInterpolator(x, y, axis=1)(x_new)
    y = np.ascontiguousarray(y, dtype=np.float64)
    segment = np.clip(np.searchsorted(x, x_new, side='right') - 1, 0, x.shape[0] - 2)
    return _cubic_hermite_eval(x, y, _akima_slopes(x, y), x_new, segment)


@njit(cache=True)
def _pairwise_sum(values, n):
    """numpy's summation of ``values[:n]`` (``pairwise_sum`` for n <= 128): sequential below 8 values,
    otherwise 8 interleaved partial sums, combined pairwise, then the remainder. In the values' dtype."""
    if n < 8:                          # numpy adds to 0.0; starting at values[0] is the same for any value
        total = values[0]
        for k in range(1, n):
            total += values[k]
        return total
    r = values[:8].copy()
    k = 8
    while k < n - n % 8:
        for q in range(8):
            r[q] += values[k + q]
        k += 8
    total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    while k < n:
        total += values[k]
        k += 1
    return total


@njit(cache=True, parallel=True)
def _bilinear_line_profiles(image, centres, unit, perp, steps, offsets, divisor):
    """Bilinear profiles along many lines: profile ``i`` at step ``s`` is the mean over ``w`` of the image at
    ``centres[i] + steps[s] * unit[i] + offsets[w] * perp[i]`` (``offsets`` empty: one sample, no mean).

    Bit-identical to ``map_coordinates(image, ..., order=1, mode='constant', cval=0)`` followed by
    ``.mean(axis=-1)`` over the line width (``divisor`` = ``[n_offsets]`` in the image dtype; at most 128
    offsets): points outside ``[0, H-1] x [0, W-1]`` are 0, each sample sums ``value * w_row * w_col`` over
    the 2x2 support in row-major order, in float64, with scipy's weights ``1 - t`` and ``1 - (1 - t)``, and is
    cast to the image dtype; the mean is numpy's pairwise sum in the image dtype, divided by the count."""
    height, width = image.shape
    n, n_steps = centres.shape[0], steps.shape[0]
    n_off = max(offsets.shape[0], 1)
    out = np.empty((n, n_steps), dtype=image.dtype)
    for i in prange(n):
        out_w = np.empty(n_off, dtype=image.dtype)
        for s in range(n_steps):
            r_line = centres[i, 0] + steps[s] * unit[i, 0]
            c_line = centres[i, 1] + steps[s] * unit[i, 1]
            for w in range(n_off):
                if offsets.shape[0] > 0:
                    r = r_line + perp[i, 0] * offsets[w]
                    c = c_line + perp[i, 1] * offsets[w]
                else:
                    r, c = r_line, c_line
                if not (r >= 0.0 and r <= height - 1 and c >= 0.0 and c <= width - 1):   # also NaN
                    out_w[w] = 0.0
                    continue
                r0 = min(int(np.floor(r)), height - 2)
                c0 = min(int(np.floor(c)), width - 2)
                # scipy's order-1 weights: w0 = 1 - t, w1 = 1 - w0 (not t, which differs in the last bit
                # where t carries more precision than 1 - t, i.e. in the first pixel cell)
                wr0 = 1.0 - (r - r0)
                wc0 = 1.0 - (c - c0)
                tr = 1.0 - wr0
                tc = 1.0 - wc0
                acc = 0.0
                acc += image[r0, c0] * wr0 * wc0
                acc += image[r0, c0 + 1] * wr0 * tc
                acc += image[r0 + 1, c0] * tr * wc0
                acc += image[r0 + 1, c0 + 1] * tr * tc
                out_w[w] = acc
            out[i, s] = _pairwise_sum(out_w, n_off) / divisor[0] if offsets.shape[0] > 0 else out_w[0]
    return out


def _profile_lines(image, start_points, end_points, linewidth=3, mode='constant', cval=0.0):
    """:meth:`Utils.fast_profile_lines`, except that lines of one length come back as one ``(n_lines, L)``
    array instead of a list of its rows (lines of different lengths still as a list)."""
    start_points = np.asarray(start_points, dtype=float).T
    end_points = np.asarray(end_points, dtype=float).T
    n_lines = len(start_points)
    if n_lines == 0:
        return []
    vectors = end_points - start_points
    norms = np.sqrt(np.sum(vectors ** 2, axis=1))
    unit = vectors / np.maximum(norms, 1e-12)[:, None]
    centres = (start_points + end_points) / 2
    halves = np.floor(norms / 2).astype(int)            # samples on each side of the midpoint
    if linewidth > 1:
        perp = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
        offsets = np.linspace(-(linewidth - 1) / 2, (linewidth - 1) / 2, linewidth)

    # fast path for uniform length: fully vectorized coordinates
    if int(halves.min()) == int(halves.max()):
        K = int(halves[0])
        L = 2 * K + 1
        s = np.arange(-K, K + 1, dtype=float)
        image = np.asarray(image)
        if (mode == 'constant' and cval == 0.0 and image.ndim == 2 and min(image.shape) >= 2
                and image.dtype in (np.float32, np.float64) and linewidth <= 128):
            # compiled sampler, bit-identical to the map_coordinates path below
            return _bilinear_line_profiles(
                image, np.ascontiguousarray(centres), np.ascontiguousarray(unit),
                np.ascontiguousarray(perp) if linewidth > 1 else np.zeros((n_lines, 2)),
                s, offsets if linewidth > 1 else np.empty(0), np.array([linewidth], dtype=image.dtype))
        line_coords = centres[:, None, :] + s[None, :, None] * unit[:, None, :]      # (n_lines, L, 2)
        if linewidth > 1:
            full = line_coords[:, :, None, :] + perp[:, None, None, :] * offsets[None, None, :, None]
            samples = map_coordinates(image, full.reshape(-1, 2).T, order=1, mode=mode, cval=cval)
            return samples.reshape(n_lines, L, linewidth).mean(axis=-1)
        samples = map_coordinates(image, line_coords.reshape(-1, 2).T, order=1, mode=mode, cval=cval)
        return samples.reshape(n_lines, L)

    # Fallback: variable-length profiles.
    coords_list, lengths = [], []
    for i in range(n_lines):
        K = int(halves[i])
        s = np.arange(-K, K + 1, dtype=float)
        line = centres[i][None, :] + s[:, None] * unit[i][None, :]                   # (L_i, 2)
        if linewidth > 1:
            line = line[:, None, :] + perp[i][None, None, :] * offsets[None, :, None]   # (L_i, lw, 2)
        coords_list.append(line.reshape(-1, 2))
        lengths.append(2 * K + 1)
    samples = map_coordinates(image, np.concatenate(coords_list, axis=0).T, order=1, mode=mode, cval=cval)
    result = []
    start_idx = 0
    for i in range(n_lines):
        n = lengths[i] * (linewidth if linewidth > 1 else 1)
        profile = samples[start_idx:start_idx + n]
        if linewidth > 1:
            profile = profile.reshape(lengths[i], linewidth).mean(axis=1)
        result.append(profile)
        start_idx += n
    return result


@njit(cache=True, parallel=True)
def _slen_from_profiles(y_up, x_interp, flat_mask, thres, distance, prominence,
                        window_size, center, slen_min, slen_max):
    """Sarcomere length + center offset for a batch of interpolated profiles.

    Processes the whole batch in one compiled, parallel pass rather than one
    ``scipy.signal.find_peaks`` call per profile.

    The peak selection reproduces ``find_peaks(height=thres, distance=distance,
    prominence=prominence)`` exactly, including scipy's filter *order*
    (height, then distance, then prominence) — a peak dropped by the distance
    rule is never prominence-tested, and a low-prominence peak can still
    suppress a neighbour, so the order is load-bearing.

    Center-of-mass accumulation is done in float64 regardless of input dtype;
    the profiles themselves are typically float32.

    Returns
    -------
    sarcomere_lengths, center_offsets : np.ndarray
        Both ``(N,)`` float64, NaN where no valid straddling peak pair exists.
    """
    n, length = y_up.shape
    out_slen = np.full(n, np.nan)
    out_off = np.full(n, np.nan)

    for i in prange(n):
        if flat_mask[i]:
            continue
        y = y_up[i]

        # --- local maxima, plateau midpoints (scipy _local_maxima_1d) ---
        peaks = np.empty(length // 2 + 1, dtype=np.int64)
        n_pk = 0
        j = 1
        j_max = length - 1
        while j < j_max:
            if y[j - 1] < y[j]:
                j_ahead = j + 1
                while j_ahead < j_max and y[j_ahead] == y[j]:
                    j_ahead += 1
                if y[j_ahead] < y[j]:
                    peaks[n_pk] = (j + j_ahead - 1) // 2
                    n_pk += 1
                    j = j_ahead
            j += 1
        if n_pk < 2:
            continue

        # --- height ---
        m = 0
        for k in range(n_pk):
            if y[peaks[k]] >= thres:
                peaks[m] = peaks[k]
                m += 1
        n_pk = m
        if n_pk < 2:
            continue

        # --- distance: greedy, tallest first (scipy _select_by_peak_distance) ---
        keep = np.ones(n_pk, dtype=np.bool_)
        heights = np.empty(n_pk, dtype=np.float64)
        for k in range(n_pk):
            heights[k] = y[peaks[k]]
        order = np.argsort(heights)
        for oi in range(n_pk - 1, -1, -1):
            k = order[oi]
            if not keep[k]:
                continue
            t = k - 1
            while t >= 0 and peaks[k] - peaks[t] < distance:
                keep[t] = False
                t -= 1
            t = k + 1
            while t < n_pk and peaks[t] - peaks[k] < distance:
                keep[t] = False
                t += 1
        m = 0
        for k in range(n_pk):
            if keep[k]:
                peaks[m] = peaks[k]
                m += 1
        n_pk = m
        if n_pk < 2:
            continue

        # --- prominence (scipy _peak_prominences, wlen=None) ---
        m = 0
        for k in range(n_pk):
            p = peaks[k]
            h = y[p]
            left_min = h
            t = p
            while t >= 0 and y[t] <= h:
                if y[t] < left_min:
                    left_min = y[t]
                t -= 1
            right_min = h
            t = p
            while t < length and y[t] <= h:
                if y[t] < right_min:
                    right_min = y[t]
                t += 1
            base = left_min if left_min > right_min else right_min
            if h - base >= prominence:
                peaks[m] = p
                m += 1
        n_pk = m
        if n_pk < 2:
            continue

        # --- COM refinement, then the pair straddling the profile centre ---
        left_com = 0.0
        right_com = 0.0
        have_left = False
        have_right = False
        for k in range(n_pk):
            idx = peaks[k]
            start = idx - window_size
            if start < 0:
                start = 0
            end = idx + window_size + 1
            if end > length:
                end = length
            y_min = y[start]
            for t in range(start + 1, end):
                if y[t] < y_min:
                    y_min = y[t]
            y_sum = 0.0
            acc = 0.0
            for t in range(start, end):
                w = y[t] - y_min
                y_sum += w
                acc += x_interp[t] * w
            com = acc / y_sum if y_sum > 0 else x_interp[idx]
            # Mirrors peaks[left][-1] / peaks[right][0]: last below centre,
            # first at-or-above it, both in peak-index order.
            if com < center:
                left_com = com
                have_left = True
            elif not have_right:
                right_com = com
                have_right = True
        if not (have_left and have_right):
            continue

        slen = right_com - left_com
        if slen_min <= slen <= slen_max:
            out_slen[i] = slen
            out_off[i] = (left_com + right_com) * 0.5 - center

    return out_slen, out_off


class Utils:
    """Miscellaneous utility functions."""

    @staticmethod
    def get_device():
        """
        Determine the most suitable device (CUDA, MPS, or CPU) for PyTorch operations.

        Returns
        -------
        torch.device
            The selected device for PyTorch operations.
        """
        # Check for CUDA support
        if torch.cuda.is_available():
            device = torch.device('cuda')
        # Check for MPS support (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            logger.warning("No CUDA or MPS device found. Calculations will run on the CPU, which might be slower.")
        logger.info(f"Using device: {device}")

        return device


    @staticmethod
    def get_models_dir() -> Path:
        """Return the path to the 'sarcasm/models' directory.

        Returns
        -------
        Path
            Path to the models directory.

        Raises
        ------
        FileNotFoundError
            If the models directory does not exist.
        """
        base_path = Path(__file__).resolve().parent

        models_dir = base_path / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory missing: {models_dir}")
        return models_dir


    @staticmethod
    def check_and_round_max_patch_size(max_patch_size):
        """Round each element of a tuple up to the next multiple of 16.

        Parameters
        ----------
        max_patch_size : tuple of int or str
            Patch dimensions to check and round. The string ``'auto'`` is passed
            through, to be resolved from device memory at prediction time.

        Returns
        -------
        tuple of int or str
            Rounded patch dimensions, or ``'auto'`` unchanged.
        """
        if isinstance(max_patch_size, str):  # 'auto' is resolved from device memory later
            return max_patch_size
        rounded_patch_size = []
        for dim in max_patch_size:
            if dim % 16 != 0:
                rounded_dim = ((dim + 15) // 16) * 16
                logger.warning(f"Dimension {dim} is not divisible by 16, rounding up to {rounded_dim}.")
                rounded_patch_size.append(rounded_dim)
            else:
                rounded_patch_size.append(dim)
        return tuple(rounded_patch_size)


    @staticmethod
    def skeleton_length_igraph(regionmask: np.ndarray, intensity_image=None) -> float:
        """
        Return the arc-length of a non-branching skeleton in physical units.

        Parameters
        ----------
        regionmask : np.ndarray
            2-D boolean one-pixel-wide skeleton (True = foreground).
        intensity_image : np.ndarray or None, optional
            Unused; accepted for compatibility with regionprops callers.
            Default is None.

        Returns
        -------
        float
            Path length in pixels.
        """
        # coordinates of all skeleton pixels
        coords = np.column_stack(np.nonzero(regionmask))
        n = len(coords)
        _sum = regionmask.sum()
        if n == 0 or _sum == 0:
            return 0.0

        # build graph
        g = ig.Graph(n)
        coord_to_idx = {tuple(p): i for i, p in enumerate(coords)}
        edges, weights = [], []

        for idx, (r, c) in enumerate(coords):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nbr = (r + dr, c + dc)
                    j = coord_to_idx.get(nbr)
                    if j is not None and j > idx:  # avoid duplicates
                        edges.append((idx, j))
                        weights.append(np.hypot(dr, dc))  # 1 or √2

        g.add_edges(edges)
        g.es["weight"] = weights

        # end points = degree-1 vertices
        ends = [v.index for v in g.vs if g.degree(v) == 1]
        if len(ends) != 2:  # branched loop: sum all edges
            length_px = sum(weights)
        else:
            path_edges = g.get_shortest_paths(ends[0], ends[1],
                                              weights="weight",
                                              output="epath")[0]
            length_px = sum(g.es[e]["weight"] for e in path_edges)

        return length_px


    @staticmethod
    def process_profiles_batch(
            profiles: List[np.ndarray],
            pixelsize: float,
            slen_lims: tuple = (1, 3),
            thres: float = 0.25,
            min_dist: float = 1,
            width: float = 0.5,
            interp_factor: int = 4,
            interpolation_method: str = 'linear',
            prominence: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch process multiple profiles for better performance.

        This function processes multiple profiles at once, reducing function call overhead
        compared to processing them individually.

        Parameters
        ----------
        profiles : list of np.ndarray or np.ndarray
            1-D intensity profiles: a list, or the rows of a 2-D array.
        pixelsize : float
            Physical size per pixel, in µm.
        slen_lims : tuple of float, optional
            (min, max) valid peak separation range in µm. Default is (1, 3).
        thres : float, optional
            Peak detection height threshold (0-1). Default is 0.25.
        min_dist : float, optional
            Minimum peak separation in µm. Default is 1.
        width : float, optional
            Half-width of COM window in µm. Default is 0.5.
        interp_factor : int, optional
            Interpolation upsampling factor. Default is 4.
        interpolation_method : {'linear', 'akima'}, optional
            Interpolation method ('linear' is fast, 'akima' is smooth).
            Default is 'linear'.
        prominence : float, optional
            ``scipy.signal.find_peaks`` prominence threshold, applied to the
            per-profile min-max normalised profile; lower values accept weaker,
            noisier peaks. Default is 0.4.

            Below ~0.4 a tail of poorly-conditioned lengths appears: measured
            against the local spatial consensus on real Z-band profiles, the 1–2 %
            of vectors admitted by 0.3 but rejected by 0.5 sit ~4–8× further from
            it than the rest (mean ~70 nm vs ~12 nm), and ~2 % of them are grossly
            wrong (>300 nm). Above that the estimate is unchanged and only the
            detection count falls.

        Returns
        -------
        sarcomere_lengths : np.ndarray
            Array of sarcomere lengths for each profile.
        center_offsets : np.ndarray
            Array of center offsets for each profile.
        """
        n_profiles = len(profiles)
        sarcomere_lengths = np.empty(n_profiles, dtype=np.float64)
        center_offsets = np.empty(n_profiles, dtype=np.float64)

        if n_profiles == 0:
            return sarcomere_lengths, center_offsets

        # Pre-compute constants
        min_dist_pixel = int(np.round(min_dist / pixelsize, 0))
        width_pixels = int(np.round(width / pixelsize, 0))
        window_size = width_pixels * max(interp_factor, 1)

        # fast path for uniform profile length: one interpolator over the (N, L) batch
        is_matrix = isinstance(profiles, np.ndarray) and profiles.ndim == 2
        lengths0 = profiles.shape[1] if is_matrix else len(profiles[0])
        uniform_length = is_matrix or all(len(p) == lengths0 for p in profiles)

        if uniform_length and lengths0 >= 2:
            L = lengths0
            # Preserve the caller's dtype through normalization so the batched and
            # per-profile paths agree numerically.
            y_mat = profiles if is_matrix else np.stack(profiles, axis=0)
            if y_mat.ndim != 2:
                y_mat = y_mat.reshape(n_profiles, L)
            pmin = y_mat.min(axis=1)
            pmax = y_mat.max(axis=1)
            flat_mask = pmax <= pmin
            denom = np.where(flat_mask, y_mat.dtype.type(1.0), pmax - pmin)
            y_norm = (y_mat - pmin[:, None]) / denom[:, None]

            pos_array = np.arange(L) * pixelsize
            if interp_factor >= 1:
                L_up = L * interp_factor
                x_interp = np.linspace(pos_array[0], pos_array[-1], num=L_up)
                if interpolation_method == 'akima':
                    y_up = _batched_akima(pos_array, y_norm, x_interp)
                else:
                    # Batched linear interpolation — equivalent to per-row
                    # ``np.interp`` but without the Python loop.
                    y_up = _batched_linear_interp(pos_array, y_norm, x_interp)
                actual_interp_factor = interp_factor
            else:
                L_up = L
                y_up = y_norm
                x_interp = pos_array
                actual_interp_factor = 1

            peak_distance = max(1, min_dist_pixel * actual_interp_factor)
            center = (pos_array[-1] + pos_array[0]) * 0.5

            return _slen_from_profiles(
                np.ascontiguousarray(y_up), np.ascontiguousarray(x_interp, dtype=np.float64),
                np.ascontiguousarray(flat_mask), thres, peak_distance, prominence,
                window_size, center, slen_lims[0], slen_lims[1],
            )

        # Fallback: variable-length profiles are handled one at a time. The common
        # caller passes uniform endpoints, but external callers may supply ragged
        # lists.
        for i, profile in enumerate(profiles):
            pmin = profile.min()
            pmax = profile.max()
            if pmax == pmin:
                sarcomere_lengths[i] = np.nan
                center_offsets[i] = np.nan
                continue

            profile_norm = (profile - pmin) / (pmax - pmin)
            pos_array = np.arange(len(profile)) * pixelsize

            if interp_factor >= 1:
                x_interp = np.linspace(pos_array[0], pos_array[-1],
                                       num=len(profile) * interp_factor)
                if interpolation_method == 'akima':
                    interp_func = Akima1DInterpolator(pos_array, profile_norm)
                    y_interp = interp_func(x_interp)
                else:
                    y_interp = np.interp(x_interp, pos_array, profile_norm)
                actual_interp_factor = interp_factor
            else:
                y_interp = profile_norm
                x_interp = pos_array
                actual_interp_factor = 1

            peak_distance = max(1, min_dist_pixel * actual_interp_factor)
            peaks_idx, _ = find_peaks(y_interp,
                                     height=thres,
                                     distance=peak_distance,
                                     prominence=prominence)

            if len(peaks_idx) < 2:
                sarcomere_lengths[i] = np.nan
                center_offsets[i] = np.nan
                continue

            peaks = np.empty(len(peaks_idx), dtype=np.float64)
            for j, idx in enumerate(peaks_idx):
                start = max(0, idx - window_size)
                end = min(len(x_interp), idx + window_size + 1)
                x_window = x_interp[start:end]
                y_window = y_interp[start:end]
                y_window = y_window - y_window.min()
                y_sum = y_window.sum()
                if y_sum > 0:
                    peaks[j] = np.dot(x_window, y_window) / y_sum
                else:
                    peaks[j] = x_interp[idx]

            center = (pos_array[-1] + pos_array[0]) * 0.5
            left_mask = peaks < center
            right_mask = peaks >= center
            if not (left_mask.any() and right_mask.any()):
                sarcomere_lengths[i] = np.nan
                center_offsets[i] = np.nan
                continue

            left_peak = peaks[left_mask][-1]
            right_peak = peaks[right_mask][0]
            slen_profile = right_peak - left_peak
            center_offset = (left_peak + right_peak) * 0.5 - center

            if slen_lims[0] <= slen_profile <= slen_lims[1]:
                sarcomere_lengths[i] = slen_profile
                center_offsets[i] = center_offset
            else:
                sarcomere_lengths[i] = np.nan
                center_offsets[i] = np.nan

        return sarcomere_lengths, center_offsets


    @staticmethod
    def analyze_orientations(orientations: np.ndarray):
        """
        Calculate the orientational order parameter and mean vector of non-polar elements in 2D.
        Orientations are expected to be in the range [0, pi].
        See https://physics.stackexchange.com/questions/65358/2-d-orientational-order-parameter

        Parameters
        ----------
        orientations : np.ndarray
            Array of orientations in radians.

        Returns
        -------
        oop : float
            The orientational order parameter.
        angle : float
            The mean vector angle in radians.
        """
        oop = 1 / len(orientations) * np.abs(np.sum(np.exp(orientations * 2 * 1j)))
        angle = np.angle(np.sum(np.exp(orientations * 2 * 1j))) / 2
        return oop, angle


    @staticmethod
    def map_array(array: np.ndarray,
                  from_values: Union[List, np.ndarray],
                  to_values: Union[List, np.ndarray]) -> np.ndarray:
        """
        Map a numpy array from one set of values to a new set of values.

        Parameters
        ----------
        array : np.ndarray
            The input numpy array.
        from_values : list or np.ndarray
            Original values.
        to_values : list or np.ndarray
            Target values.

        Returns
        -------
        np.ndarray
            The array with values mapped from ``from_values`` to ``to_values``.
        """
        sort_idx = np.argsort(from_values)
        idx = np.searchsorted(from_values, array, sorter=sort_idx)
        out = to_values[sort_idx][idx]
        return out

    @staticmethod
    def shuffle_labels(labels: np.ndarray, seed=0):
        """
        Shuffle labels randomly.

        Parameters
        ----------
        labels : np.ndarray
            The labels to be shuffled.
        seed : int, optional
            Seed for the random number generator. Default is 0.

        Returns
        -------
        np.ndarray
            The input labels, randomly shuffled.
        """
        values = np.unique(labels)
        values_in = values.copy()
        # shuffle cell labels
        np.random.seed(seed)
        np.random.shuffle(values[1:])
        labels_shuffled = Utils.map_array(labels, values_in, values)
        return labels_shuffled

    @staticmethod
    def convert_lists_to_arrays_in_dict(d):
        """Convert every list value in a dict to a numpy array in place."""
        for key, value in d.items():
            if isinstance(value, list):
                d[key] = np.array(value)
        return d



    @staticmethod
    def get_orientation_angle_map(orientation_field: np.ndarray,
                                  use_median_filter: bool = True,
                                  radius: int = 3,
                                  min_magnitude: float = 0.0,
                                  progress_notifier=None) -> np.ndarray:
        """
        Convert a polar vector field into a map of angles for sarcomere orientations.

        The function supports both single-image and multi-image inputs. For single-image
        inputs, the expected shape is (2, H, W). For multi-image inputs, the expected
        shape is (N, 2, H, W), where N is the number of images.

        Parameters
        ----------
        orientation_field : np.ndarray
            Polar vector field(s). For a single image, shape ``(2, H, W)``.
            For multiple images, shape ``(N, 2, H, W)``.
        min_magnitude : float, optional
            Pixels whose predicted vector is shorter than this are returned as NaN
            instead of an arbitrary angle. Default 0.0 (previous behaviour).
        use_median_filter : bool, optional
            Whether to apply a median filter to the resulting angle map.
            Default is True.
        radius : int, optional
            Radius of the disk-shaped footprint for the median filter.
            Default is 3.
        progress_notifier : object or None, optional
            Optional notifier providing an ``iterator`` method to report
            per-frame progress. Default is None.

        Returns
        -------
        np.ndarray
            Array of angles in radians, mapped to the range [0, π]. Shape
            ``(H, W)`` for a single image, ``(N, H, W)`` for multiple images.
        """
        # Reshape input to (N, 2, H, W) if necessary
        if orientation_field.ndim == 3 and orientation_field.shape[0] == 2:
            orientation_field = orientation_field[np.newaxis, ...]
        elif not (orientation_field.ndim == 4 and orientation_field.shape[1] == 2):
            raise ValueError(
                "orientation_field must have shape (2, H, W) or (N, 2, H, W)."
            )

        # a near-zero orientation vector has an arbitrary angle: gate it out
        undefined = None
        if min_magnitude > 0:
            magnitude = np.sqrt((orientation_field ** 2).sum(axis=1))
            undefined = magnitude < min_magnitude

        # Compute angles
        angles = np.arctan2(orientation_field[:, 1], orientation_field[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        angles = np.where(angles > np.pi, angles - np.pi, angles)

        # Apply orientation-aware median filter if requested
        if use_median_filter:
            footprint = disk(radius, strict_radius=False)
            filtered = np.empty_like(angles)
            _frames = range(angles.shape[0])
            if progress_notifier is not None:
                _frames = progress_notifier.iterator(_frames, total=angles.shape[0])
            for i in _frames:
                # Double the angles to map [0, π] to [0, 2π]
                doubled_angles = 2 * angles[i]

                # Convert doubled angles to unit vectors
                x = np.cos(doubled_angles)
                y = np.sin(doubled_angles)

                # Undefined pixels must not contribute their arbitrary direction to a
                # neighbour's median, so blank them and use the NaN-aware filter.
                if undefined is not None:
                    x = np.where(undefined[i], np.nan, x)
                    y = np.where(undefined[i], np.nan, y)
                    filtered_doubled = np.arctan2(
                        Utils.nanmedian_filter_numba(y, window_size=2 * radius + 1),
                        Utils.nanmedian_filter_numba(x, window_size=2 * radius + 1))
                    filtered[i] = ((filtered_doubled + 2 * np.pi) % (2 * np.pi)) / 2
                    continue

                # Apply median filter to vector components
                x_filtered = Utils.median_filter_numba(x, footprint=footprint)
                y_filtered = Utils.median_filter_numba(y, footprint=footprint)

                # Convert back to angles
                filtered_doubled_angles = np.arctan2(y_filtered, x_filtered)

                # Ensure angles are in [0, 2π)
                filtered_doubled_angles = (filtered_doubled_angles + 2 * np.pi) % (2 * np.pi)

                # Convert back to [0, π] range by halving
                filtered[i] = filtered_doubled_angles / 2

            angles = filtered

        if undefined is not None:
            angles = np.where(undefined, np.nan, angles)

        return angles.squeeze()


    @staticmethod
    def fast_profile_lines(image, start_points, end_points, linewidth=3, mode='constant', cval=0.0):
        """
        Sample many profile lines at once, one sample per pixel along each line.

        The samples of a line are centred on its midpoint and spaced exactly one pixel
        apart along it (so a distance of *n* samples is *n* pixels, as
        :meth:`process_profiles_batch` assumes), and the image is interpolated
        bilinearly at every sample. Nearest-neighbour sampling, as used before 1.0,
        rounded every sample to a pixel and thereby lengthened the peak-to-peak
        sarcomere lengths by ~2 % with a spread that depended on the sub-pixel
        position of the line; samples spread over ``ceil(length + 1)`` points were
        also 1 % closer than a pixel.

        Parameters
        ----------
        image : np.ndarray
            Input image from which to sample the profile lines.
        start_points : array_like
            Starting coordinates of the lines, shape ``(2, N)`` (row, col).
        end_points : array_like
            Ending coordinates of the lines, shape ``(2, N)`` (row, col).
        linewidth : int, optional
            Width of the profile line, in pixels; the profile is the mean over
            ``linewidth`` parallel lines one pixel apart. Default is 3.
        mode : str, optional
            Boundary mode passed to ``map_coordinates``. Default is 'constant'.
        cval : float, optional
            Value used for points outside the image boundaries. Default is 0.0.

        Returns
        -------
        list of np.ndarray
            Sampled profile values for each line; line *i* has
            ``2 * floor(|end_i - start_i| / 2) + 1`` samples, the middle one at its
            midpoint.
        """
        profiles = _profile_lines(image, start_points, end_points, linewidth=linewidth, mode=mode, cval=cval)
        return list(profiles) if isinstance(profiles, np.ndarray) else profiles

    @staticmethod
    @njit(parallel=True)
    def median_filter_numba(data, footprint):
        """Numba-accelerated median filter over a 2-D array with a boolean footprint.

        Parameters
        ----------
        data : np.ndarray
            2-D input array.
        footprint : np.ndarray
            2-D boolean footprint defining the filter window.

        Returns
        -------
        np.ndarray
            Filtered array, same shape as ``data``.
        """
        H, W = data.shape
        fH, fW = footprint.shape
        pad_h, pad_w = fH // 2, fW // 2

        padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=data.dtype)
        padded[pad_h:pad_h + H, pad_w:pad_w + W] = data

        out = np.empty_like(data)

        for i in prange(H):
            for j in range(W):
                count = 0
                window_vals = []
                for m in range(fH):
                    for n in range(fW):
                        if footprint[m, n]:
                            val = padded[i + m, j + n]
                            window_vals.append(val)
                            count += 1
                sorted_vals = np.sort(np.array(window_vals))
                mid = count // 2
                if count % 2 == 1:
                    out[i, j] = sorted_vals[mid]
                else:
                    out[i, j] = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

        return out


    @staticmethod
    @njit(parallel=True)
    def nanmedian_filter_numba(data, window_size):
        """
        Apply a nanmedian filter to a 2-D array using a sliding window.

        The median of each window is computed ignoring NaN values.

        Parameters
        ----------
        data : np.ndarray
            2-D input array of floats with possible NaN values.
        window_size : int
            Size (assumed odd) of the square window.

        Returns
        -------
        np.ndarray
            Filtered array, same shape as ``data``.
        """
        H, W = data.shape
        pad = window_size // 2
        out = np.empty((H, W), dtype=data.dtype)

        # Create a padded array filled with NaNs.
        padded = np.empty((H + 2 * pad, W + 2 * pad), dtype=data.dtype)
        for i in range(H + 2 * pad):
            for j in range(W + 2 * pad):
                padded[i, j] = np.nan
        for i in range(H):
            for j in range(W):
                padded[i + pad, j + pad] = data[i, j]

        # Process each row in parallel.
        for i in prange(H):
            # Allocate a temporary array to hold one window's values. (private to each row)
            temp = np.empty(window_size * window_size, dtype=data.dtype)
            for j in range(W):
                count = 0
                # Extract values from the window, ignoring NaNs.
                for m in range(window_size):
                    for n in range(window_size):
                        val = padded[i + m, j + n]
                        # Use Numba-friendly check for NaN.
                        if not (val != val):
                            temp[count] = val
                            count += 1

                if count == 0:
                    out[i, j] = np.nan
                else:
                    sorted_vals = np.sort(temp[:count])
                    # Compute median from sorted values.
                    if count & 1:  # odd number of valid elements
                        out[i, j] = sorted_vals[count // 2]
                    else:
                        mid = count // 2
                        out[i, j] = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

        return out