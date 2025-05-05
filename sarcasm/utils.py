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


import datetime
import glob
import os
import platform
import subprocess
import warnings
from typing import Tuple, Any, List, Union
os.environ["KMP_WARNINGS"] = "False"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*omp_set_nested.*")

import numpy as np
import tifffile
import torch
from numba import njit, prange
from numpy import ndarray, dtype
from scipy.interpolate import griddata, Akima1DInterpolator
from scipy.ndimage import label, map_coordinates
from scipy.signal import correlate, savgol_filter, butter, filtfilt, find_peaks
from scipy.stats import stats
from skimage.draw import line
from skimage.morphology import disk


class Utils:
    """ Miscellaneous utility functions """

    @staticmethod
    def get_device(print_device=False, no_cuda_warning=False):
        """
        Determines the most suitable device (CUDA, MPS, or CPU) for PyTorch operations.

        Parameters:
        - print_device (bool): If True, prints the device being used.
        - no_cuda_warning (bool): If True, prints a warning if neither CUDA nor MPS is available.

        Returns:
        - torch.device: The selected device for PyTorch operations.
        """
        # Check for CUDA support
        if torch.cuda.is_available():
            device = torch.device('cuda')
        # Check for MPS support (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            if no_cuda_warning:
                print("Warning: No CUDA or MPS device found. Calculations will run on the CPU, which might be slower.")

        if print_device:
            print(f"Using device: {device}")

        return device

    @staticmethod
    def today_date():
        """
        Get today's date in the format 'YYYYMMDD'.

        Returns
        -------
        str
            Today's date in 'YYYYMMDD' format.
        """
        t = datetime.datetime.today()
        return t.strftime('%Y%m%d')

    @staticmethod
    def get_tif_files_in_folder(folder: str) -> List[str]:
        """
        Find all .tif files in a specified folder.

        Parameters
        ----------
        folder : str
            Path to the folder.

        Returns
        -------
        list
            List of file paths to the .tif files.
        """
        files = glob.glob(folder + '*.tif')
        print(f'{len(files)} files founds')
        return files

    @staticmethod
    def get_lois_of_file(filename_cell: str) -> List[Tuple[str, str]]:
        """
        Get the lines of interests (LOIs) of a tif-file.

        Parameters
        ----------
        filename_cell : str
            Path to the file of the cell.

        Returns
        -------
        list
            List of tuples, each containing the cell file path and LOI filename.
        """
        cell_dir = filename_cell[:-4] + '/'
        assert os.path.isdir(cell_dir), "File not yet analyzed."
        list_lois = glob.glob(cell_dir + '*.json')
        return [(filename_cell, os.path.basename(loi)) for loi in list_lois]

    @staticmethod
    def open_folder(path: str):
        """
        Open a folder in the file explorer.

        Parameters
        ----------
        path : str
            Path to the folder.
        """
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    @staticmethod
    def two_sample_t_test(data: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pair-wise two sample t-test of multiple conditions.

        Parameters
        ----------
        data : array-like
            Input data for the t-test.
        alpha : float, optional
            Significance level. Default is 0.05.

        Returns
        -------
        tuple
            p-values and significance levels for each pair of conditions.
        """
        p_values = np.zeros((len(data), len(data))) * np.nan
        significance = np.zeros((len(data), len(data))) * np.nan
        for i, d_i in enumerate(data):
            for j, d_j in enumerate(data):
                if i < j:
                    t_value, p_value = stats.ttest_ind(d_i, d_j)
                    p_values[i, j] = p_value

                    if p_value < alpha:
                        significance[i, j] = 1
                    else:
                        significance[i, j] = 0

        return p_values, significance

    @staticmethod
    def nan_sav_golay(data: np.ndarray, window_length: int, polyorder: int, axis: int = 0) -> np.ndarray:
        """
        Apply a Savitzky-Golay filter to data with NaN values along the specified axis.

        Parameters
        ----------
        data : array-like
            Input data.
        window_length : int
            Length of the filter window, must be odd and greater than polyorder.
        polyorder : int
            Order of the polynomial used for the filtering.
        axis : int, optional
            The axis along which to apply the filter. The default is 0 (first axis).

        Returns
        -------
        array-like
            Filtered data with NaN values preserved.
        """

        # Ensure window_length is odd and > polyorder
        if window_length % 2 == 0:
            window_length += 1

        # Placeholder for filtered data
        filtered_data = np.full(data.shape, np.nan)

        # Function to apply filter on 1D array
        def filter_1d(segment):
            not_nan_indices = np.where(~np.isnan(segment))[0]
            split_indices = np.split(not_nan_indices, np.where(np.diff(not_nan_indices) != 1)[0] + 1)
            for indices in split_indices:
                if len(indices) >= window_length:
                    segment[indices] = savgol_filter(segment[indices], window_length, polyorder)
            return segment

        # Apply filter along the specified axis
        if axis == -1 or axis == data.ndim - 1:
            for i in range(data.shape[axis]):
                filtered_data[..., i] = filter_1d(data[..., i])
        else:
            for i in range(data.shape[axis]):
                filtered_data[i] = filter_1d(data[i])

        return filtered_data

    @staticmethod
    def nan_low_pass(x: np.ndarray, N: int = 6, crit_freq: float = 0.25, min_len: int = 31) -> np.ndarray:
        """
        Apply a Butterworth low-pass filter to data with NaN values.

        Parameters
        ----------
        x : np.ndarray
            Input data.
        N : int, optional
            Filter order. The higher the order, the steeper the spectral cutoff.
            Default is 6.
        crit_freq : float, optional
            Maximum passed frequency. Default is 0.25.
        min_len : int, optional
            Minimum length of data required to apply the filter. Default is 31.

        Returns
        -------
        np.ndarray
            Filtered data with NaN values preserved.
        """
        x_filt = np.zeros(x.shape) * np.nan
        idx_no_nan = np.where(~np.isnan(x))[0]
        if len(idx_no_nan) >= min_len:
            b, a = butter(N, crit_freq)
            x_filt[idx_no_nan] = filtfilt(b, a, x[idx_no_nan])
        return x_filt

    @staticmethod
    def most_freq_val(array: np.ndarray, bins: int = 20) -> ndarray[Any, dtype[Any]]:
        """
        Calculate the most frequent value in an array.

        Parameters
        ----------
        array : np.ndarray
            Input array.
        bins : int, optional
            Number of bins for the histogram calculation. Default is 20.

        Returns
        -------
        float
            Most frequent value in the array.
        """
        a, b = np.histogram(array, bins=bins, range=(np.nanmin(array), np.nanmax(array)))
        val = b[np.argmax(a)]
        return val

    @staticmethod
    def weighted_avg_and_std(x: np.ndarray, weights: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the weighted average and standard deviation.

        Parameters
        ----------
        x : array-like
            Values.
        weights : array-like
            Weights.
        axis : int, optional
            Axis along which to compute the average and standard deviation. Default is 0.

        Returns
        -------
        tuple
            Weighted average and weighted standard deviation.
        """
        average = np.nansum(x * weights, axis=axis) / ((~np.isnan(x)) * weights).sum(axis=axis)
        variance = np.nansum((x - average) ** 2 * weights, axis=axis) / ((~np.isnan(x)) * weights).sum(axis=axis)
        return average, np.sqrt(variance)

    @staticmethod
    def weighted_quantile(data: np.ndarray, weights: np.ndarray, quantile: float) -> Union[
        float, ndarray[Any, dtype[Any]]]:
        """
        Compute the weighted quantile of a 1D numpy array.

        Parameters
        ----------
        data : np.ndarray
            Input array (one dimension array).
        weights : np.ndarray
            Array with the weights of the same size of data.
        quantile : float
            Desired quantile.

        Returns
        -------
        result : np.ndarray
            Weighted quantile of data.
        """
        # Flatten the arrays and remove NaNs
        data = data.flatten()
        weights = weights.flatten()
        mask = ~np.isnan(data)
        data = data[mask]
        weights = weights[mask]

        # Sort the data
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Compute the cumulative sum of weights
        Sn = np.cumsum(sorted_weights)

        # Compute the threshold for the desired quantile
        threshold = quantile / 100 * np.sum(sorted_weights)

        # Check if any cumulative sum of weights exceeds the threshold
        over_threshold = Sn >= threshold
        if not np.any(over_threshold):
            return np.nan

        # Return the data value where the cumulative sum of weights first exceeds the threshold
        return sorted_data[over_threshold][0]

    @staticmethod
    def column_weighted_quantiles(data: np.ndarray, weights: np.ndarray, quantiles: list) -> np.ndarray:
        """
        Compute the weighted quantile for each column of a 2D numpy array.

        Parameters
        ----------
        data : np.ndarray
            Input array (two dimension array).
        weights : np.ndarray
            Array with the weights of the same size of data.
        quantiles : list of float
            List with desired quantiles.

        Returns
        -------
        result : np.array
            2D array with weighted quantiles of each data column.
        """
        results = np.zeros((len(quantiles), data.shape[1]))
        for i in range(data.shape[1]):
            for j, q in enumerate(quantiles):
                results[j, i] = Utils.weighted_quantile(data[:, i], weights[:, i], q)
        return results

    @staticmethod
    def custom_diff(x: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute derivative of `x` using central differences.

        This function computes the derivative of the input time-series `x` using
        central differences. At the edges of `x`, forward and backward differences
        are used. The time-series `x` can be either 1D or 2D.

        Parameters
        ----------
        x : ndarray
            The input time-series, must be 1D or 2D.
        dt : float
            The time interval between pos_vectors in `x`.

        Returns
        -------
        v : ndarray
            The derivative of `x`, has the same shape as `x`.

        """

        v = np.zeros_like(x)

        if len(x.shape) == 1:
            v[0] = (x[1] - x[0]) / dt
            v[-1] = (x[-1] - x[-2]) / dt
            v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        elif len(x.shape) == 2:
            v[:, 0] = (x[:, 1] - x[:, 0]) / dt
            v[:, -1] = (x[:, -1] - x[:, -2]) / dt
            v[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2 * dt)

        return v

    @staticmethod
    def process_profile(
            profile: np.ndarray,
            pixelsize: float,
            slen_lims: tuple = (1, 3),
            thres: float = 0.25,
            min_dist: float = 1,
            width: float = 0.5,
            interp_factor: int = 4
    ) -> Tuple[float, float]:
        """
        Find peak distance in a 1D intensity profile using interpolation and center of mass (COM).

        This function detects peaks in a normalized 1D intensity profile, optionally interpolates
        the profile using Akima interpolation, and refines the peak positions using the center of mass
        within a local window.

        Parameters
        ----------
        profile : np.ndarray
            1D intensity profile.
        pixelsize : float
            Physical size per pixel.
        slen_lims : tuple of float, optional
            (min, max) valid peak separation range, by default (1, 3).
        thres : float, optional
            Peak detection height threshold (0-1), by default 0.25.
        min_dist : float, optional
            Minimum peak separation in µm, by default 1.
        width : float, optional
            Half-width of COM window in µm, by default 0.5.
        interp_factor : int, optional
            Interpolation upsampling factor, by default 4. If ≤ 1, no interpolation is performed.

        Returns
        -------
        slen_profile : float
            Peak separation distance in micrometer, or np.nan if invalid.
        center_offsets : float
            Offset of the profile center in micrometer, or np.nan if invalid.

        Notes
        -----
        - For `interp_factor` ≤ 1, no interpolation is performed and the original profile is used.
        - The function uses Akima1DInterpolator for smooth interpolation when requested.
        - Center of mass calculation is performed in a window around each detected peak for sub-pixel accuracy.
        - If less than two peaks are detected, or the separation is outside `slen_lims`, returns (np.nan, np.nan).
        """
        # convert parameter to pixels
        min_dist_pixel = int(np.round(min_dist / pixelsize, 0))
        width_pixels = int(np.round(width / pixelsize, 0))

        # Normalize profile to [0,1] range
        profile = (profile - profile.min()) / (profile.max() - profile.min())

        # Create position array
        pos_array = np.arange(len(profile)) * pixelsize

        if interp_factor >= 1:
            # Create interpolation function with padded data
            interp_func = Akima1DInterpolator(pos_array[np.isfinite(profile)],
                                              profile[np.isfinite(profile)], method='akima')
            x_interp = np.linspace(pos_array[0], pos_array[-1],
                                   num=len(profile) * interp_factor)
            y_interp = interp_func(x_interp)
        else:
            y_interp = profile
            x_interp = pos_array
            interp_factor = 1

        # Find peaks with prominence to avoid noise
        peaks_idx, properties = find_peaks(y_interp,
                                           height=thres,
                                           distance=min_dist_pixel * interp_factor,
                                           prominence=0.2)

        if len(peaks_idx) < 2:
            return np.nan, np.nan

        # Calculate refined peak positions using center of mass
        peaks = []
        for idx in peaks_idx:
            start = max(0, idx - width_pixels * interp_factor)
            end = min(len(x_interp), idx + width_pixels * interp_factor + 1)
            x_window = x_interp[start:end]
            y_window = y_interp[start:end]
            # Subtract baseline to improve COM calculation
            y_window = y_window - y_window.min()
            peak_pos = np.sum(x_window * y_window) / np.sum(y_window)
            peaks.append(peak_pos)

        peaks = np.array(peaks)
        center = (pos_array[-1] + pos_array[0]) / 2

        # Split peaks into left and right of center
        left_peaks = peaks[peaks < center]
        right_peaks = peaks[peaks > center]

        if len(left_peaks) == 0 or len(right_peaks) == 0:
            return np.nan, np.nan

        # Take rightmost peak from left side and leftmost peak from right side
        left_peak = left_peaks[-1]  # rightmost peak from left side
        right_peak = right_peaks[0]  # leftmost peak from right side
        slen_profile = np.abs(right_peak - left_peak)
        center_offsets = (left_peak + right_peak) / 2 - center  # position of center for correction of pos_vectors

        if slen_lims[0] <= slen_profile <= slen_lims[1]:
            return slen_profile, center_offsets

        return np.nan, np.nan

    @staticmethod
    def peakdetekt(x_pos, y, thres=0.2, thres_abs=False, min_dist=10, width=6, interp_factor=6):
        """
        A customized peak detection algorithm using scipy with Akima interpolation.

        Parameters
        ----------
        x_pos : ndarray
            An array containing the positions in µm.
        y : ndarray
            The intensity profile.
        thres : float, optional
            Threshold for the peak detection. Default is 0.3.
        thres_abs : bool, optional
            Whether the peak detection threshold is absolute. Default is True.
        min_dist : int, optional
            Minimum distance between detected peaks, in pixels. Default is 10.
        width : int, optional
            Width of the region of interest around the detected peaks for the
            method of moments computation. Default is 6.
        interp_factor : int, optional
            Factor by which to increase the resolution through interpolation. Default is 10.

        Returns
        -------
        refined_peaks : ndarray
            An array containing the refined peak positions in µm.
        """
        # Apply Akima interpolation to refine the intensity profile
        akima_interpolator = Akima1DInterpolator(x_pos, y)
        x_interp = np.linspace(x_pos[0], x_pos[-1], len(x_pos) * interp_factor)
        y_interp = akima_interpolator(x_interp)

        # Approximate peak position using scipy's find_peaks
        height = thres if thres_abs else thres * np.max(y_interp)
        peaks_idx, _ = find_peaks(y_interp, height=height, distance=min_dist * interp_factor, prominence=0.5)

        # Refine peak positions using the center of mass method
        refined_peaks = []
        for idx in peaks_idx:
            start = max(0, idx - width * interp_factor)
            end = min(len(y_interp), idx + width * interp_factor + 1)
            roi_x = x_interp[start:end]
            roi_y = y_interp[start:end]
            com = np.sum(roi_x * roi_y) / np.sum(roi_y)
            refined_peaks.append(com)

        # plt.figure(figsize=(12, 4), dpi=200)
        # plt.plot(x_interp, y_interp)
        # for peak in peaks_idx:
        #     plt.axvline(x_interp[peak], color='r')
        # for peak in refined_peaks:
        #     plt.axvline(peak, color='k', lw=2)
        # plt.show()

        return np.array(refined_peaks)

    @staticmethod
    def peak_by_first_moment(x: np.ndarray, y: np.ndarray):
        """
        Calculate the peak of y using the first moment method.

        Parameters
        ----------
        x : numpy.ndarray
            The x-values of the data.
        y : numpy.ndarray
            The y-values of the data.

        Returns
        -------
        peak : float
            The calculated peak value.
        """
        return np.sum(x * y) / np.sum(y)

    @staticmethod
    def analyze_orientations(orientations: np.ndarray):
        """
        Calculate the orientational order parameter and mean vector of non-polar elements in 2D.
        Orientations are expected to be in the range [0, pi].
        See https://physics.stackexchange.com/questions/65358/2-d-orientational-order-parameter

        Parameters
        ----------
        orientations : numpy.ndarray
            Array of orientations. In radians.

        Returns
        -------
        oop : float
            The calculated orientational order parameter.
        angle : float
            The calculated mean vector angle.
        """
        oop = 1 / len(orientations) * np.abs(np.sum(np.exp(orientations * 2 * 1j)))
        angle = np.angle(np.sum(np.exp(orientations * 2 * 1j))) / 2
        return oop, angle

    @staticmethod
    def correct_phase_confocal(tif_file: str, shift_max=30):
        """
        Correct phase shift for images of Leica confocal resonant scanner in bidirectional mode while conserving metadata.

        Parameters
        ----------
        tif_file : str
            Path to the input .tif file.
        shift_max : int, optional
            Maximum allowed shift, by default 30.
        """

        # read data
        data = tifffile.imread(tif_file)
        data_0 = data[0].astype('float32')

        # split data in -> and <-
        row_even = data_0[::2, :].reshape(-1)
        row_uneven = data_0[1::2, :].reshape(-1)
        if row_even.shape != row_uneven.shape:
            row_even = data_0[2::2, :].reshape(-1)
            row_uneven = data_0[1::2, :].reshape(-1)

        # correlate lines of both directions and calculate phase shift
        corr = correlate(row_even, row_uneven, mode='same')
        corr_window = corr[int(corr.shape[0] / 2 - shift_max): int(corr.shape[0] / 2 + shift_max)]
        x = np.arange(corr.shape[0]) - corr.shape[0] / 2
        x_window = np.arange(corr_window.shape[0]) - corr_window.shape[0] / 2
        shift = int(x_window[np.argmax(corr_window)])
        print(f'Phase shift = {shift} pixel')

        # correct data
        data_correct = np.copy(data)
        data_correct[:, ::2, :] = np.roll(data[:, ::2, :], shift=-shift, axis=2)

        # get metadata from old file
        tif = tifffile.TiffFile(tif_file)
        ij_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags

        resolution = [tags['XResolution'].value, tags['YResolution'].value]
        metadata = {'unit': 'um', 'finterval': ij_metadata['finterval'], 'axes': 'TYX', 'info': ij_metadata['Info']}

        # save tif file under previous name
        tifffile.imwrite(tif_file, data_correct, imagej=True, metadata=metadata, resolution=resolution)

    @staticmethod
    def map_array(array: np.ndarray,
                  from_values: Union[List, np.ndarray],
                  to_values: Union[List, np.ndarray]) -> np.ndarray:
        """
        Map a numpy array from one set of values to a new set of values.

        Parameters
        ----------
        array : numpy.ndarray
            The input 2D numpy array.
        from_values : list
            List of original values.
        to_values : list
            List of target values.

        Returns
        -------
        out : numpy.ndarray
            The array with values mapped from 'from_values' to 'to_values'.
        """
        sort_idx = np.argsort(from_values)
        idx = np.searchsorted(from_values, array, sorter=sort_idx)
        out = to_values[sort_idx][idx]
        return out

    @staticmethod
    def shuffle_labels(labels: np.ndarray, seed=0):
        """
        Shuffle labels randomly

        Parameters
        ----------
        labels : numpy.ndarray
            The labels to be shuffled.
        seed : int, optional
            The seed for the random number generator, by default 0.

        Returns
        -------
        labels_shuffled : numpy.ndarray
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
        for key, value in d.items():
            if isinstance(value, list):
                d[key] = np.array(value)
        return d

    @staticmethod
    def find_closest(array, x):
        # Calculate the absolute differences
        differences = np.abs(array - x)

        # Find the index of the minimum difference
        index = np.argmin(differences)

        # Get the value at the found index
        closest_value = array[index]

        return index, closest_value

    @staticmethod
    def max_orientation_change(angles):
        # Ensure angles are in the range [-π/2, π/2]
        angles = np.mod(angles + np.pi / 2, np.pi) - np.pi / 2

        # Calculate angle differences
        angle_diffs = np.diff(angles)

        # Adjust for non-polar nature (180-degree symmetry)
        angle_diffs = np.minimum(np.abs(angle_diffs), np.pi - np.abs(angle_diffs))

        # Find and return the maximum angle change
        max_change = np.max(angle_diffs)

        return max_change

    @staticmethod
    def get_orientation_angle_map(orientation_field: np.ndarray,
                                  use_median_filter: bool = True,
                                  radius: int = 3) -> np.ndarray:
        """
        Convert a polar vector field into a map of angles for sarcomere orientations.

        The function supports both single-image and multi-image inputs. For single-image
        inputs, the expected shape is (2, H, W). For multi-image inputs, the expected
        shape is (N, 2, H, W), where N is the number of images.

        Parameters
        ----------
        orientation_field : numpy.ndarray
            Polar vector field(s). For a single image, a 3D array of shape (2, H, W).
            For multiple images, a 4D array of shape (N, 2, H, W).
        use_median_filter : bool, optional
            Whether to apply a median filter to the resulting angle map. Default is True.
        radius : int, optional
            Radius of the disk-shaped footprint for the median filter. Default is 3.

        Returns
        -------
        numpy.ndarray
            A 2D or 3D array of angles in radians, mapped to the range [0, π].
            If the input is a single image of shape (2, H, W), the output shape is (H, W).
            If the input contains multiple images of shape (N, 2, H, W), the output
            shape is (N, H, W).
        """
        # Reshape input to (N, 2, H, W) if necessary
        if orientation_field.ndim == 3 and orientation_field.shape[0] == 2:
            orientation_field = orientation_field[np.newaxis, ...]
        elif not (orientation_field.ndim == 4 and orientation_field.shape[1] == 2):
            raise ValueError(
                "orientation_field must have shape (2, H, W) or (N, 2, H, W)."
            )

        # Compute angles
        angles = np.arctan2(orientation_field[:, 1], orientation_field[:, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        angles = np.where(angles > np.pi, angles - np.pi, angles)

        # Apply orientation-aware median filter if requested
        if use_median_filter:
            footprint = disk(radius, strict_radius=False)
            filtered = np.empty_like(angles)
            for i in range(angles.shape[0]):
                # Double the angles to map [0, π] to [0, 2π]
                doubled_angles = 2 * angles[i]

                # Convert doubled angles to unit vectors
                x = np.cos(doubled_angles)
                y = np.sin(doubled_angles)

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

        return angles.squeeze()

    @staticmethod
    def create_distance_map(sarc_obj):
        """
        Creates distance map for sarcomeres from a Structure object. The distance map is 0 at Z-bands and 1 at M-bands.

        Parameters
        ----------
        sarc_obj : Structure
            An object of the Structure class.

        Returns
        -------
        distance : numpy.ndarray
            A 2D array with normalized distances (0 to 1) along sarcomeres.
        """

        # Validate sarc_obj data
        structure = sarc_obj.data
        pixelsize = sarc_obj.metadata.get('pixelsize', None)

        if not all(key in structure for key in
                   ['pos_vectors', 'sarcomere_orientation_vectors', 'sarcomere_length_vectors']):
            raise Warning("Missing required data in sarc_obj.data.")

        if pixelsize is None:
            raise Warning("Missing 'pixelsize' in sarc_obj.metadata.")

        # Extract data from sarc_obj
        pos_vectors = structure['pos_vectors'][0]
        orientation_vectors = np.asarray([
            -np.sin(structure['sarcomere_orientation_vectors'][0]),
            np.cos(structure['sarcomere_orientation_vectors'][0])
        ])
        sarcomere_length_vectors = structure['sarcomere_length_vectors'][0] / pixelsize

        # Calculate endpoints of each vector based on orientation and length
        ends_0 = pos_vectors + orientation_vectors * sarcomere_length_vectors / 2  # End point 1
        ends_1 = pos_vectors - orientation_vectors * sarcomere_length_vectors / 2  # End point 2

        # Initialize output arrays
        distance = np.full(sarc_obj.metadata['size'], np.nan, dtype='float32')

        def create_distance_array(l):
            """Creates a normalized distance array for a line segment."""
            if l < 2:
                raise ValueError("Length must be at least 2.")
            midpoint = (l + 1) // 2
            return np.concatenate((np.linspace(0, 1, midpoint), np.linspace(1, 0, l - midpoint)))

        # Populate distance and length arrays for each sarcomere
        for e0, e1, in zip(ends_0.T.astype('int'), ends_1.T.astype('int')):
            rr, cc = line(*e0, *e1)  # Get pixel coordinates for the line

            dist = create_distance_array(len(rr))  # Create normalized distance values

            # Assign values to output arrays
            try:
                distance[rr, cc] = dist
            except:
                pass

        return distance

    @staticmethod
    def interpolate_distance_map(image, N=50, method='linear'):
        """
        Interpolates NaN regions in a 2D image, filling only those regions whose size
        is less than or equal to a specified threshold.

        Parameters
        ----------
        image : numpy.ndarray
            A 2D array representing the input image. NaN values represent gaps to be filled.
        N : int
            The maximum size (in pixels) of connected NaN regions to interpolate. Regions larger
            than this threshold will remain unaltered.
        method : str, optional
            The interpolation method to use. Options are 'linear', 'nearest', and 'cubic'.
            Default is 'linear'.

        Returns
        -------
        numpy.ndarray
            A 2D array with the same shape as the input `image`, where small NaN regions
            (size <= N) have been interpolated. Larger NaN regions are left unchanged.
        """

        # Get indices and mask valid points
        x, y = np.indices(image.shape)
        valid_points = ~np.isnan(image)
        valid_coords = np.array((x[valid_points], y[valid_points])).T
        valid_values = image[valid_points]

        # Label connected NaN regions
        nan_mask = np.isnan(image)
        labeled_nan_regions, num_features = label(nan_mask)

        # Combine masks for all small regions
        combined_small_nan_mask = np.zeros_like(image, dtype=bool)

        for region_label in range(1, num_features + 1):
            region_mask = labeled_nan_regions == region_label
            region_size = np.sum(region_mask)

            if region_size <= N:
                combined_small_nan_mask |= region_mask

        # Interpolate all small NaN regions at once
        if np.any(combined_small_nan_mask):
            invalid_coords = np.array((x[combined_small_nan_mask], y[combined_small_nan_mask])).T
            interpolated_values = griddata(valid_coords, valid_values, invalid_coords, method=method)
            image[combined_small_nan_mask] = interpolated_values

        return image

    @staticmethod
    def fast_profile_lines(image, start_points, end_points, linewidth=3, mode='constant', cval=0.0):
        """
        Vectorized version of profile_line from scikit-image that processes multiple lines simultaneously.

        Parameters
        ----------
        image : ndarray
            The input image from which to sample the profile lines.
        start_points : array_like
            An array of shape (N, 2) containing the starting coordinates of the lines.
        end_points : array_like
            An array of shape (N, 2) containing the ending coordinates of the lines.
        linewidth : int, optional
            The width of the profile line, in pixels. Default is 1.
        mode : str, optional
            The mode parameter for map_coordinates. Default is 'constant'.
        cval : float, optional
            The value used for points outside the boundaries of the input image. Default is 0.0.

        Returns
        -------
        result : list of ndarray
            A list containing the sampled profile values for each line.
        """
        # Convert to array and swap row/col order to match image coordinates
        start_points = np.asarray(start_points).T
        end_points = np.asarray(end_points).T

        # Calculate pixel coordinates along each line
        vectors = end_points - start_points
        lengths = np.ceil(np.sqrt(np.sum(vectors ** 2, axis=1)) + 1).astype(int)

        # Create coordinates matrix for each line
        coords_list = []

        for i in range(len(start_points)):
            t = np.linspace(0, 1, lengths[i])[:, np.newaxis]
            line_coords = start_points[i] + t * vectors[i]

            if linewidth > 1:
                # Calculate perpendicular vector
                perp = np.array([-vectors[i, 1], vectors[i, 0]])
                perp = perp / np.sqrt(np.sum(perp ** 2))

                # Create parallel lines
                offsets = np.linspace(-(linewidth - 1) / 2, (linewidth - 1) / 2, linewidth)
                line_coords = (line_coords[:, np.newaxis, :] +
                               perp[np.newaxis, np.newaxis, :] * offsets[:, np.newaxis])

                # Reshape to separate rows and columns
                rows = line_coords[..., 0].reshape(-1)
                cols = line_coords[..., 1].reshape(-1)
                line_coords = np.stack([rows, cols])
            else:
                line_coords = np.stack([line_coords[:, 0], line_coords[:, 1]])

            coords_list.append(line_coords)

        # Sample all points in one call to map_coordinates
        all_coords = np.hstack(coords_list)
        profiles = map_coordinates(image, all_coords, order=0, mode=mode, cval=cval)

        # Split and average profiles
        result = []
        start_idx = 0
        for i in range(len(start_points)):
            if linewidth > 1:
                profile = profiles[start_idx:start_idx + lengths[i] * linewidth]
                profile = profile.reshape(lengths[i], linewidth).mean(axis=1)
            else:
                profile = profiles[start_idx:start_idx + lengths[i]]
            result.append(profile)
            start_idx += lengths[i] * linewidth if linewidth > 1 else lengths[i]

        return result

    @staticmethod
    @njit(parallel=True)
    def median_filter_numba(data, footprint):
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
        Applies a nanmedian filter to a 2D array using a sliding window.
        The function computes the median of each window ignoring NaN values.

        text
        Parameters:
          data : 2D numpy array of float
            Input array with possible NaN values.
          window_size : int
            The size (assumed odd) of the square window.

        Returns:
          out : 2D numpy array of the same shape as data containing the filtered result.
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