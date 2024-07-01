import datetime
import glob
import os
import platform
import subprocess
import warnings
from typing import Tuple, Any, List, Union

import numpy as np
import peakutils
import statsmodels.api as sm
import tifffile
import torch
from numpy import ndarray, dtype
from scipy.signal import correlate, savgol_filter, hilbert, butter, filtfilt
from scipy.stats import stats

warnings.filterwarnings("ignore")


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
    def get_lois_of_cell(filename_cell: str) -> List[Tuple[str, str]]:
        """
        Get the lines of interests (LOIs) of a specified cell.

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
    def hpfilter(data: np.ndarray, lamb=1600) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the Hodrick-Prescott filter to data.

        Parameters
        ----------
        data : array-like
            Input data.
        lamb : float, optional
            Smoothing parameter. Default is 1600.

        Returns
        -------
        tuple
            Filtered data and the cyclical component.
        """
        data_filt = np.zeros_like(data) * np.nan
        delta_data = np.zeros_like(data) * np.nan
        idx_no_nan = np.where(~np.isnan(data))
        data_no_nan = data[idx_no_nan]
        filt_res = sm.tsa.filters.hpfilter(data_no_nan, lamb)
        data_filt[idx_no_nan] = filt_res[1]
        delta_data[idx_no_nan] = filt_res[0]
        return data_filt, delta_data

    @staticmethod
    def nan_hilbert(data: np.ndarray, min_len: int = 20) -> np.ndarray:
        """
        Apply the Hilbert transform to data with NaN values.

        Parameters
        ----------
        data : array-like
            Input data.
        min_len : int, optional
            Minimum length of non-NaN values required to apply the Hilbert transform.
            Default is 20.

        Returns
        -------
        array-like
            Hilbert-transformed data with NaN values preserved.
        """
        dat_hilbert = np.zeros(data.shape, dtype=complex) * np.nan
        idx_no_nan = np.where(~np.isnan(data))[0]
        if len(idx_no_nan) >= min_len:
            dat_hilbert[idx_no_nan] = hilbert(data[idx_no_nan])
        return dat_hilbert

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
    def hilbert_transform(data: np.ndarray, dt: float):
        """
            Perform a Hilbert transform on the input data to get the analytic signal for amplitude,
            instantaneous phase and frequency, ignores NaN values.

            Parameters
            ----------
            data : numpy.ndarray
                The input data to be transformed.
            dt : float
                The time interval between frames.

            Returns
            -------
            amplitude_envelope : numpy.ndarray
                The amplitude envelope of the transformed data.
            instantaneous_phase : numpy.ndarray
                The instantaneous phase of the transformed data.
            instantaneous_frequency : numpy.ndarray
                The instantaneous frequency of the transformed data.
            """
        analytic_signal = Utils.nan_hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_phase[~np.isnan(instantaneous_phase)] = np.unwrap(
            instantaneous_phase[~np.isnan(instantaneous_phase)])
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) / dt)
        return amplitude_envelope, instantaneous_phase, instantaneous_frequency

    @staticmethod
    def peakdetekt(x_pos, y, thres=0.3, min_dist=10, width=6):
        """
        A customized peak detection algorithm.

        This function uses peakutils to detect the peaks in the intensity profile `y`
        against `x_pos`. The detected peaks are then fine-tuned by calculating their
        mean position using the method of moments.

        Parameters
        ----------
        x_pos : ndarray
            An array containing the positions in µm.
        y : ndarray
            The intensity profile.
        thres : float, optional
            Threshold for the peak detection. Default is 0.3.
        min_dist : int, optional
            Minimum distance between detected peaks. Default is 10.
        width : int, optional
            Width of the region of interest around the detected peaks for the
            method of moments computation. Default is 6.

        Returns
        -------
        peaks : ndarray
            An array containing the detected peak positions in µm.

        """
        # approximate peak position
        peaks_idx = peakutils.indexes(y, thres=thres, min_dist=min_dist)
        # get mean of peak sum(x*y)/sum(y)
        peaks = peakutils.interpolate(x_pos, y, ind=peaks_idx, width=width, func=Utils.peak_by_first_moment)
        return peaks

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
    def map_array(array: np.ndarray, from_values: List, to_values: List) -> np.ndarray:
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
        # Convert angles to unit vectors
        vectors = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])

        # Calculate angles between adjacent vectors
        angle_changes = [np.arccos(np.clip(np.dot(vectors[i], vectors[i + 1]), -1.0, 1.0)) for i in
                         range(len(vectors) - 1)]

        # Find and return the maximum angle change
        max_change = np.max(angle_changes)

        return max_change
