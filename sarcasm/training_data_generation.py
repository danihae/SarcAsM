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


"""Generation of training data (Z-band, M-band, orientation, sarcomere masks) via wavelet analysis."""

import os.path
from typing import Union

import numpy as np
import skimage
import tifffile
import torch
import torch.nn.functional as F
from bio_image_unet.unet import Predict
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import label
from skimage.draw import line
from skimage.morphology import skeletonize

from sarcasm import PlotUtils, SarcAsM, Utils


class TrainingDataGenerator:
    """
    Generate training data targets from a microscopy image.

    Produces Z-band mask, M-band mask, sarcomere orientation field and
    sarcomere mask for a single TIFF file.

    Parameters
    ----------
    image_path : str
        Path to microscopy image TIFF file.
    output_dirs : dict
        Mapping of target name to output directory, e.g.
        ``{'zbands': '.../zbands/', 'mbands': '.../mbands/',
        'orientation': '.../orientation/', 'sarcomere_mask': '.../sarcomere_mask/'}``.
    pixelsize : float or None, optional
        Pixel size of image in µm. If None, read from the TIFF metadata.
        Default is None.
    """

    def __init__(self, image_path: str, output_dirs: dict, pixelsize: float = None) -> None:
        """
        Initialize TrainingDataGenerator for a single TIFF file.

        Parameters
        ----------
        image_path : str
            Path to microscopy image TIFF file.
        output_dirs : dict
            Mapping of target name to output directory, e.g.
            ``{'zbands': '.../zbands/', 'mbands': '.../mbands/',
            'orientation': '.../orientation/', 'sarcomere_mask': '.../sarcomere_mask/'}``.
        pixelsize : float or None, optional
            Pixel size of image in µm. If None, read from the TIFF metadata.
            Default is None.
        """
        self.image_path = image_path
        self.image = tifffile.imread(image_path)
        self.basename = os.path.basename(self.image_path)
        self.output_dirs = output_dirs
        self.shape = self.image.shape
        self.wavelet_dict = None

        if pixelsize is None:
            self.pixelsize = self.get_pixel_size(image_path)[0]
        else:
            self.pixelsize = pixelsize

    def __getattr__(self, attr):
        if attr in self.output_dirs:
            value = tifffile.imread(self.output_dirs[attr] + self.basename)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __dir__(self):
        """Include dynamic attributes in autocompletion"""
        return super().__dir__() + list(self.output_dirs.keys())

    def predict_zbands(self, model_path: str, network: str = 'Unet_v0', patch_size: tuple[int, int] = (1024, 1024)):
        """
        Predict sarcomere Z-bands using a pre-trained U-Net model.

        Optional; manually annotated Z-band masks in the 'zbands' directory may
        be used instead.

        Parameters
        ----------
        model_path : str
            Path of U-Net model for sarcomere Z-band detection.
        network : str, optional
            Model type, chosen from models in the bio-image-unet package.
            Default is 'Unet_v0'.
        patch_size : tuple of int, optional
            Patch size ``(height, width)`` for prediction; should be multiples
            of 16. Default is (1024, 1024).
        """
        Predict(self.image, result_name=self.output_dirs['zbands'] + self.basename, model_params=model_path,
                network=network, resize_dim=patch_size)

    def wavelet_analysis(self, kernel: str = 'half_gaussian', size: float = 3.0, minor: float = 0.33,
                         major: float = 1.0, len_lims: tuple[float, float] = (1.45, 2.7),
                         len_step: float = 0.05, orient_lims: tuple[float, float] = (-90, 90),
                         orient_step: float = 10, add_negative_center_kernel: bool = False,
                         patch_size: int = 1024, score_threshold: float = 0.25,
                         abs_threshold: bool = True, gating: bool = True, load_mbands: bool = False,
                         dtype: Union[torch.dtype, str] = 'auto', save_memory: bool = False,
                         device: torch.device = torch.device('cpu')):
        """
        AND-gated double wavelet analysis of sarcomere length and orientation.

        Parameters
        ----------
        kernel : {'gaussian', 'half_gaussian', 'binary'}, optional
            Filter kernel. 'gaussian' is a bivariate Gaussian; 'half_gaussian'
            is a univariate Gaussian in the minor axis direction and a step
            function in the major axis direction; 'binary' is a step function in
            both directions. Default is 'half_gaussian'.
        size : float, optional
            Size of wavelet filters in µm; must exceed the upper limit of
            ``len_lims``. Default is 3.0.
        minor : float, optional
            Minor axis width in µm (FWHM), should match Z-band thickness; for
            kernel 'gaussian' and 'half_gaussian'. Default is 0.33.
        major : float, optional
            Major axis width in µm, should match Z-band width (FWHM for
            'gaussian', full width for 'half_gaussian'). Default is 1.0.
        len_lims : tuple of float, optional
            Limits of lengths / wavelet distances in µm. Default is (1.45, 2.7).
        len_step : float, optional
            Step size of sarcomere lengths in µm. Default is 0.05.
        orient_lims : tuple of float, optional
            Limits of sarcomere orientation angles in degrees. Default is (-90, 90).
        orient_step : float, optional
            Step size of orientation angles in degrees. Default is 10.
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel between the two wavelets to avoid
            detecting two Z-bands two sarcomeres apart as a sarcomere; only for
            kernel 'gaussian'. Default is False.
        patch_size : int, optional
            Patch size in pixels for wavelet analysis; adapt to GPU memory.
            Default is 1024.
        score_threshold : float, optional
            Threshold score for clipping the length and orientation map. If
            ``abs_threshold`` is False, interpreted as a percentile for adaptive
            thresholding. Default is 0.25.
        abs_threshold : bool, optional
            If True, ``score_threshold`` is an absolute value; if False, a
            percentile for adaptive thresholding. Default is True.
        gating : bool, optional
            If True, AND-gated wavelet filtering is used; if False, both wavelet
            filters are applied jointly. Default is True.
        load_mbands : bool, optional
            If True, load a manually curated M-band mask. Default is False.
        dtype : torch.dtype or str, optional
            Torch data type (torch.float32 or torch.float16); 'auto' chooses
            float16 for cuda and mps, float32 for cpu. Default is 'auto'.
        save_memory : bool, optional
            Whether to save memory by moving intermediate results to CPU.
            Default is False.
        device : torch.device, optional
            Device for 2D convolutions. Default is ``torch.device('cpu')``.
        """
        assert size > 1.1 * len_lims[1], (f"The size of wavelet filter {size} is too small for the maximum sarcomere "
                                          f"length {len_lims[1]}")

        # select precision
        if device.type == 'cpu':
            dtype = torch.float32
        elif device.type == 'cuda' or device.type == 'mps':
            dtype = torch.float16

        zbands = tifffile.imread(self.output_dirs['zbands'] + self.basename)

        # create filter bank
        bank, len_range, orient_range = TrainingDataGenerator.create_wavelet_bank(pixelsize=self.pixelsize,
                                                                                  kernel=kernel,
                                                                                  size=size, minor=minor, major=major,
                                                                                  len_lims=len_lims,
                                                                                  len_step=len_step,
                                                                                  orient_lims=orient_lims,
                                                                                  orient_step=orient_step,
                                                                                  add_negative_center_kernel=add_negative_center_kernel)
        len_range_tensor = torch.from_numpy(len_range).to(device).to(dtype=dtype)
        orient_range_tensor = torch.from_numpy(np.radians(orient_range)).to(device).to(dtype=dtype)

        # convolve zbands with wavelet kernels
        result = TrainingDataGenerator.convolve_image_with_bank(zbands, bank, device=device, gating=gating,
                                                                dtype=dtype, save_memory=save_memory,
                                                                patch_size=patch_size)

        # argmax
        (wavelet_sarcomere_length, wavelet_sarcomere_orientation,
         wavelet_max_score) = TrainingDataGenerator.argmax_wavelets(result,
                                                                    len_range_tensor,
                                                                    orient_range_tensor)

        # evaluate wavelet results at sarcomere mbands
        if load_mbands:
            mbands = self.mbands
        else:
            mbands = None
        (pos_vectors_px, mband_id_vectors, mband_length_vectors, sarcomere_length_vectors,
         sarcomere_orientation_vectors, max_score_vectors, mbands, mbands_labels,
         score_threshold) = self.get_sarcomere_vectors_wavelet(
            wavelet_sarcomere_length, wavelet_sarcomere_orientation, wavelet_max_score,
            len_range=len_range, mbands=mbands,
            score_threshold=score_threshold,
            abs_threshold=abs_threshold)

        # empty memory
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.wavelet_dict = {
            'sarcomere_length': wavelet_sarcomere_length,
            'sarcomere_orientation': wavelet_sarcomere_orientation,
            'max_score': wavelet_max_score,
            'pos_vectors_px': pos_vectors_px,
            'sarcomere_length_vectors': sarcomere_length_vectors,
            'mband_length_vectors': mband_length_vectors,
            'mband_id_vectors': mband_id_vectors,
            'sarcomere_orientation_vectors': sarcomere_orientation_vectors,
            'max_score_vectors': max_score_vectors,
            'mbands': mbands,
            'mbands_labels': mbands_labels}

        # save mbands as tiff
        tifffile.imwrite(self.output_dirs['mbands'] + self.basename, mbands)

    def create_sarcomere_mask(self, dilation_radius):
        """
        Create binary sarcomere mask from sarcomere vectors.

        Parameters
        ----------
        dilation_radius : float
            Dilation radius to dilate sarcomere mask and close small gaps, in µm.
        """
        # get sarcomere vectors
        pos_vectors_px = self.wavelet_dict['pos_vectors_px']
        sarcomere_orientation_vectors = self.wavelet_dict['sarcomere_orientation_vectors']
        sarcomere_length_vectors = self.wavelet_dict['sarcomere_length_vectors']

        # calculate sarcomere mask
        if len(pos_vectors_px) > 0:
            sarcomere_mask = SarcAsM.sarcomere_mask(pos_vectors_px * self.pixelsize,
                                                      -sarcomere_orientation_vectors,
                                                      sarcomere_length_vectors,
                                                      shape=self.shape,
                                                      pixelsize=self.pixelsize,
                                                      dilation_radius=dilation_radius)
        else:
            sarcomere_mask = np.zeros(self.shape, dtype='bool')

        # save sarcomere mask as tiff
        tifffile.imwrite(self.output_dirs['sarcomere_mask'] + self.basename, sarcomere_mask)

    def create_orientation_map(self):
        """
        Create a 2D sarcomere orientation map from sarcomere vectors and save it.

        The map holds the local sarcomere orientation angle (direction of unit
        vectors pointing from M-bands to Z-bands); undefined regions are np.nan.
        The result is written to the 'orientation' output directory.
        """
        # Extract data from wavelet_dict
        pos_vectors = self.wavelet_dict['pos_vectors_px']
        sarcomere_orientation_vectors = self.wavelet_dict['sarcomere_orientation_vectors']
        orientation_vectors = np.asarray([np.sin(sarcomere_orientation_vectors),
                                          -np.cos(sarcomere_orientation_vectors)])
        sarcomere_length_vectors = self.wavelet_dict['sarcomere_length_vectors'] / self.pixelsize

        # Calculate endpoints of each vector based on orientation and length
        ends_0 = pos_vectors.T + orientation_vectors * sarcomere_length_vectors / 2  # End point 1
        ends_1 = pos_vectors.T - orientation_vectors * sarcomere_length_vectors / 2  # End point 2

        # Initialize output array
        orientation_map = np.full(self.image.shape, np.nan, dtype='float32')

        def orientation_angle_line(o, len_line):
            """
            Creates an array with:
            - First half: o + π
            - Second half: o
            """
            if len_line < 2:
                raise ValueError("Length must be at least 2.")

            midpoint = len_line // 2
            return np.concatenate([
                np.full(midpoint, o + np.pi),
                np.full(len_line - midpoint, o)
            ])

        # Populate orientation array for each sarcomere
        for e0, e1, o in zip(ends_0.T.astype('int'), ends_1.T.astype('int'), sarcomere_orientation_vectors):
            rr, cc = line(*e0, *e1)  # Get pixel coordinates for the line

            # Check for out-of-bounds coordinates
            if (np.any(rr < 0) or
                    np.any(cc < 0) or
                    np.any(rr >= self.image.shape[0]) or
                    np.any(cc >= self.image.shape[1])):
                continue

            # store -o: the wavelet draws along (sin o, -cos o) but get_sarcomere_vectors
            # reads an angle as (sin o, cos o); -o is the same line in the reader's convention
            orientation_map[rr, cc] = orientation_angle_line(-o, len(cc))

        tifffile.imwrite(self.output_dirs['orientation'] + self.basename, orientation_map)

    def smooth_orientation_map(self, window_size: int = 3):
        """
        Smooth the orientation angle map using a nanmedian filter and save it.

        To handle the angle discontinuity at 2*pi -> 0, the map is converted to
        a 2D orientation field, both components are smoothed, then converted back.

        Parameters
        ----------
        window_size : int, optional
            Size of smoothing kernel; must be an odd integer. Default is 3.
        """
        # load orientation map
        orientation_map = tifffile.imread(self.output_dirs['orientation'] + self.basename)

        # convert to orientation field
        orientation_field = np.stack((np.cos(orientation_map), np.sin(orientation_map)))

        # smooth both components with custom nanmedian filter
        orientation_field_smoothed = orientation_field.copy()
        for i, comp_i in enumerate(orientation_field):
            orientation_field_smoothed[i] = Utils.nanmedian_filter_numba(comp_i, window_size=window_size)

        # convert back to orientation map
        orientation_map_smoothed = np.arctan2(orientation_field_smoothed[1], orientation_field_smoothed[0])

        # save smoothed orientation map as tiff
        tifffile.imwrite(self.output_dirs['orientation'] + self.basename, orientation_map_smoothed)

    def plot_results(self, save_path=None, xlim=None, ylim=None):
        """
        Plot image, Z-bands, M-bands, sarcomere vectors, mask and orientation map.

        Parameters
        ----------
        save_path : str or None, optional
            If given, path to save the figure. Default is None.
        xlim : tuple of float or None, optional
            x-axis limits applied to all panels. Default is None.
        ylim : tuple of float or None, optional
            y-axis limits applied to all panels. Default is None.
        """
        mosaic = """
        ABC
        DEF
        """
        fig, axs = plt.subplot_mosaic(figsize=(PlotUtils.width_1p5cols, 4), mosaic=mosaic, constrained_layout=True,
                                      dpi=300)

        # image
        image_i = tifffile.imread(self.image_path)
        image_i = np.clip(image_i, a_min=np.percentile(image_i, 0.1),
                          a_max=np.percentile(image_i, 99.9))
        axs['A'].imshow(image_i, cmap='gray')
        axs['A'].set_title('Image')

        # zbands
        zbands_i = tifffile.imread(self.output_dirs['zbands'] + self.basename)
        axs['B'].imshow(zbands_i, cmap='gray')
        axs['B'].set_title('Z-bands')

        # mbands
        zbands_i = tifffile.imread(self.output_dirs['mbands'] + self.basename)
        axs['C'].imshow(zbands_i, cmap='gray')
        axs['C'].set_title('M-bands')

        # sarcomere vectors
        pos_vectors = self.wavelet_dict['pos_vectors_px']
        sarcomere_orientation_vectors = self.wavelet_dict['sarcomere_orientation_vectors']
        sarcomere_length_vectors = self.wavelet_dict['sarcomere_length_vectors'] / self.pixelsize
        orientation_vectors = np.asarray(
            [np.cos(sarcomere_orientation_vectors), np.sin(sarcomere_orientation_vectors)])

        # adjust sarcomere lengths to appear correct in quiver plot
        half_length = sarcomere_length_vectors * 0.5
        headaxislength = 4
        linewidths = 0.5
        color_arrows = 'k'
        color_points = 'darkgreen'
        s_points = 5

        axs['D'].imshow(self.zbands, cmap='Purples')

        axs['D'].quiver(pos_vectors[:, 1], pos_vectors[:, 0], -orientation_vectors[0] * half_length,
                  orientation_vectors[1] * half_length, width=linewidths, headaxislength=headaxislength, units='xy',
                  angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5, label='Sarcomere vectors')
        axs['D'].quiver(pos_vectors[:, 1], pos_vectors[:, 0], orientation_vectors[0] * half_length,
                  -orientation_vectors[1] * half_length, headaxislength=headaxislength, units='xy',
                  angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5, width=linewidths)

        axs['D'].scatter(pos_vectors[:, 1], pos_vectors[:, 0], marker='.', c=color_points, edgecolors='none',
                   s=s_points * 0.5,
                   label='Midline pos_vectors')
        axs['D'].set_title('Sarcomere vectors')

        # sarcomere mask
        sarcomere_mask_i = tifffile.imread(self.output_dirs['sarcomere_mask'] + self.basename)
        axs['E'].imshow(sarcomere_mask_i, cmap='gray')
        axs['E'].set_title('Sarcomere mask')

        # sarcomere orientation angle field
        orientation_i = tifffile.imread(self.output_dirs['orientation'] + self.basename)
        plot = axs['F'].imshow(orientation_i, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axs['F'].set_title('Orientation angle')
        colorbar = plt.colorbar(ax=axs['F'], mappable=plot, orientation='horizontal', shrink=0.7)
        colorbar.set_ticks([-np.pi, 0, np.pi])
        colorbar.set_ticklabels([r'-$\pi$', '0', r'$\pi$'])

        [PlotUtils.remove_ticks(axs[key]) for key in axs.keys()]
        if xlim:
            [axs[key].set_xlim(xlim) for key in axs.keys()]
        if ylim:
            [axs[key].set_ylim(ylim[::-1]) for key in axs.keys()]

        if save_path:
            fig.savefig(save_path, dpi=500)

        plt.show()

    @staticmethod
    def binary_kernel(d: float, sigma: float, width: float, orient: float, size: float,
                      pixelsize: float, mode: str = 'both') -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Return a binary kernel pair for AND-gated double wavelet analysis.

        Parameters
        ----------
        d : float
            Distance between the two wavelets.
        sigma : float
            Minor axis width of single wavelets.
        width : float
            Major axis width of single wavelets.
        orient : float
            Rotation orientation in degrees.
        size : float
            Size of kernel in µm.
        pixelsize : float
            Pixel size in µm.
        mode : {'both', 'separate'}, optional
            'separate' returns two separate kernels, 'both' returns a single
            combined kernel. Default is 'both'.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            The generated binary kernel, or kernel pair if ``mode='separate'``.
        """
        # meshgrid
        size_pixel = TrainingDataGenerator.round_up_to_odd(size / pixelsize)
        _range = np.linspace(-size / 2, size / 2, size_pixel, dtype='float32')
        x_mesh, y_mesh = np.meshgrid(_range, _range)
        # build kernel
        kernel0 = np.zeros_like(x_mesh)
        kernel0[np.abs((-x_mesh - d / 2)) < sigma / 2] = 1
        kernel0[np.abs(y_mesh) > width / 2] = 0
        kernel1 = np.zeros_like(x_mesh)
        kernel1[np.abs((x_mesh - d / 2)) < sigma / 2] = 1
        kernel1[np.abs(y_mesh) > width / 2] = 0

        # Normalize the kernels
        kernel0 /= np.sum(kernel0)
        kernel1 /= np.sum(kernel1)

        kernel0 = ndimage.rotate(kernel0, orient, reshape=False, order=3)
        kernel1 = ndimage.rotate(kernel1, orient, reshape=False, order=3)
        if mode == 'separate':
            return kernel0, kernel1
        elif mode == 'both':
            return kernel0 + kernel1
        else:
            raise ValueError(f'Kernel mode {mode} not defined.')

    @staticmethod
    def gaussian_kernel(dist: float, minor: float, major: float, orient: float, size: float,
                        pixelsize: float, mode: str = 'both',
                        add_negative_center_kernel: bool = False) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Return a Gaussian kernel pair for AND-gated double wavelet analysis.

        Parameters
        ----------
        dist : float
            Distance between the two wavelets.
        minor : float
            Minor axis width of single wavelets in µm.
        major : float
            Major axis width of single wavelets in µm.
        orient : float
            Rotation orientation in degrees.
        size : float
            Size of kernel in µm.
        pixelsize : float
            Pixel size in µm.
        mode : {'both', 'separate'}, optional
            'separate' returns two separate kernels, 'both' returns a single
            combined kernel. Default is 'both'.
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel between the two wavelets to avoid
            detecting two Z-bands two sarcomeres apart as a sarcomere.
            Default is False.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            The combined Gaussian kernel, or kernel pair if ``mode='separate'``.
        """
        # Transform FWHM to sigma
        minor_sigma = minor / 2.355
        major_sigma = major / 2.355

        # Calculate the size of the kernel in pixels and create meshgrid
        size_pixel = TrainingDataGenerator.round_up_to_odd(size / pixelsize)
        _range = np.linspace(-size / 2, size / 2, size_pixel, dtype='float32')
        x_mesh, y_mesh = np.meshgrid(_range, _range)

        # Create the first Gaussian kernel
        kernel0 = (1 / (2 * np.pi * minor_sigma * major_sigma) * np.exp(
            -((x_mesh - dist / 2) ** 2 / (2 * minor_sigma ** 2) + y_mesh ** 2 / (2 * major_sigma ** 2))))

        # Create the second Gaussian kernel
        kernel1 = (1 / (2 * np.pi * minor_sigma * major) * np.exp(
            -((x_mesh + dist / 2) ** 2 / (2 * minor_sigma ** 2) + y_mesh ** 2 / (2 * major_sigma ** 2))))

        # Create the middle Gaussian kernel
        kernelmid = (1 / (2 * np.pi * minor_sigma * major_sigma) * np.exp(
            -(x_mesh ** 2 / (2 * minor_sigma ** 2) + y_mesh ** 2 / (2 * major_sigma ** 2))))

        # Normalize the kernels
        kernel0 /= np.sum(kernel0)
        kernel1 /= np.sum(kernel1)
        kernelmid /= np.sum(kernelmid)
        kernelmid *= -1

        if add_negative_center_kernel:
            kernel0 += kernelmid
            kernel1 += kernelmid

        # Rotate the kernels
        kernel0 = ndimage.rotate(kernel0, orient, reshape=False, order=2)
        kernel1 = ndimage.rotate(kernel1, orient, reshape=False, order=2)

        # Return the kernels based on the mode
        if mode == 'separate':
            return kernel0, kernel1
        elif mode == 'both':
            return kernel0 + kernel1
        else:
            raise ValueError(f'Kernel model {mode} not defined!')

    @staticmethod
    def half_gaussian_kernel(dist: float, minor: float, major: float, orient: float, size: float,
                             pixelsize: float, mode: str = 'both',
                             add_negative_center_kernel: bool = False) -> Union[
        tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Return a kernel pair for AND-gated double wavelet analysis.

        Uses a univariate Gaussian profile in the longitudinal minor axis
        direction and a step function in the lateral major axis direction.

        Parameters
        ----------
        dist : float
            Distance between the two wavelets.
        minor : float
            Minor axis width (FWHM) of single wavelets in µm.
        major : float
            Major axis width of single wavelets in µm.
        orient : float
            Rotation orientation in degrees.
        size : float
            Size of kernel in µm.
        pixelsize : float
            Pixel size in µm.
        mode : {'both', 'separate'}, optional
            'separate' returns two separate kernels, 'both' returns a single
            combined kernel. Default is 'both'.
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel between the two wavelets to avoid
            detecting two Z-bands two sarcomeres apart as a sarcomere.
            Default is False.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            The combined kernel, or kernel pair if ``mode='separate'``.
        """
        # Transform FWHM to sigma
        minor_sigma = minor / 2.355
        major / 2.355

        # Calculate the size of the kernel in pixels and create meshgrid
        size_pixel = TrainingDataGenerator.round_up_to_odd(size / pixelsize)
        _range = np.linspace(-size / 2, size / 2, size_pixel, dtype='float32')
        x_mesh, y_mesh = np.meshgrid(_range, _range)

        # Create the first Gaussian kernel
        kernel0 = 1 / (np.sqrt(2 * np.pi) * minor_sigma) * np.exp(-((x_mesh - dist / 2) ** 2 / (2 * minor_sigma ** 2)))

        # Create the second Gaussian kernel
        kernel1 = 1 / (np.sqrt(2 * np.pi) * minor_sigma) * np.exp(-((x_mesh + dist / 2) ** 2 / (2 * minor_sigma ** 2)))

        # Create the middle Gaussian kernel
        kernelmid = 1 / (np.sqrt(2 * np.pi) * minor_sigma) * np.exp(-(x_mesh ** 2 / (2 * minor_sigma ** 2)))

        # set to 0 where wider than major axis
        kernel0[np.abs(y_mesh) > major / 2] = 0
        kernel1[np.abs(y_mesh) > major / 2] = 0
        kernelmid[np.abs(y_mesh) > major / 2] = 0

        # Normalize the kernels
        kernel0 /= np.sum(kernel0)
        kernel1 /= np.sum(kernel1)
        kernelmid /= np.sum(kernelmid)
        kernelmid *= -1

        if add_negative_center_kernel:
            kernel0 += kernelmid
            kernel1 += kernelmid

        # Rotate the kernels
        kernel0 = ndimage.rotate(kernel0, orient, reshape=False, order=2)
        kernel1 = ndimage.rotate(kernel1, orient, reshape=False, order=2)

        # Return the kernels based on the mode
        if mode == 'separate':
            return kernel0, kernel1
        elif mode == 'both':
            return kernel0 + kernel1
        else:
            raise ValueError(f'Kernel mode {mode} not defined!')

    @staticmethod
    def round_up_to_odd(f: float) -> int:
        """
        Rounds float up to the next odd integer.
    
        Parameters
        ----------
        f : float
            The input float number.
    
        Returns
        -------
        int
            The next odd integer.
        """
        return int(np.ceil(f) // 2 * 2 + 1)

    @staticmethod
    def create_wavelet_bank(pixelsize: float, kernel: str = 'half_gaussian', size: float = 3, minor: float = 0.15,
                            major: float = 0.5, len_lims: tuple[float, float] = (1.3, 2.5), len_step: float = 0.025,
                            orient_lims: tuple[float, float] = (-90, 90), orient_step: float = 5,
                            add_negative_center_kernel: bool = False) -> list[np.ndarray]:
        """
        Return a bank of double wavelets over lengths and orientations.

        Parameters
        ----------
        pixelsize : float
            Pixel size in µm.
        kernel : {'half_gaussian', 'gaussian', 'binary'}, optional
            Filter kernel. Default is 'half_gaussian'.
        size : float, optional
            Size of kernel in µm. Default is 3.
        minor : float, optional
            Minor axis width of single wavelets. Default is 0.15.
        major : float, optional
            Major axis width of single wavelets. Default is 0.5.
        len_lims : tuple of float, optional
            Limits of lengths / wavelet distances in µm. Default is (1.3, 2.5).
        len_step : float, optional
            Step size in µm. Default is 0.025.
        orient_lims : tuple of float, optional
            Limits of orientation angle in degrees. Default is (-90, 90).
        orient_step : float, optional
            Step size in degrees. Default is 5.
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel between the two wavelets to avoid
            detecting two Z-bands two sarcomeres apart as a sarcomere; only for
            kernel 'gaussian' or 'half_gaussian'. Default is False.

        Returns
        -------
        bank : np.ndarray
            Wavelet bank of shape ``(n_lengths, n_orientations, 2, size, size)``.
        len_range : np.ndarray
            Array of sarcomere lengths used in the bank.
        orient_range : np.ndarray
            Array of orientation angles (degrees) used in the bank.
        """

        len_range = np.arange(len_lims[0] - len_step, len_lims[1] + len_step, len_step, dtype='float32')
        orient_range = np.arange(orient_lims[0], orient_lims[1], orient_step, dtype='float32')
        size_pixel = TrainingDataGenerator.round_up_to_odd(size / pixelsize)

        bank = np.zeros((len_range.shape[0], orient_range.shape[0], 2, size_pixel, size_pixel))
        for i, d in enumerate(len_range):
            for j, orient in enumerate(orient_range):
                if kernel == 'gaussian':
                    bank[i, j] = TrainingDataGenerator.gaussian_kernel(d, minor, major, orient=orient, size=size,
                                                                       pixelsize=pixelsize, mode='separate',
                                                                       add_negative_center_kernel=add_negative_center_kernel)
                elif kernel == 'half_gaussian':
                    bank[i, j] = TrainingDataGenerator.half_gaussian_kernel(d, minor, major, orient=orient, size=size,
                                                                            pixelsize=pixelsize, mode='separate',
                                                                            add_negative_center_kernel=add_negative_center_kernel)
                elif kernel == 'binary':
                    bank[i, j] = TrainingDataGenerator.binary_kernel(d, minor, major, orient, size, pixelsize,
                                                                     mode='separate')
                else:
                    raise ValueError("Unsupported kernel type. Choose from 'gaussian', 'binary', or 'half_gaussian'.")
        return bank, len_range, orient_range

    @staticmethod
    def convolve_image_with_bank(image: np.ndarray, bank: np.ndarray, device: torch.device, gating: bool = True,
                                 dtype: torch.dtype = torch.float16, save_memory: bool = False,
                                 patch_size: int = 512) -> torch.Tensor:
        """
        AND-gated double-wavelet convolution of an image using a filter bank.

        Processes the image in overlapping patches to manage GPU memory usage
        and avoid edge effects.

        Parameters
        ----------
        image : np.ndarray
            Input image to be convolved.
        bank : np.ndarray
            Filter bank containing the wavelet kernels.
        device : torch.device
            Device on which to perform the computation (e.g., 'cuda', 'mps' or 'cpu').
        gating : bool, optional
            Whether to use AND-gated double-wavelet convolution. Default is True.
        dtype : torch.dtype, optional
            Data type for the tensors. Default is torch.float16.
        save_memory : bool, optional
            Whether to save memory by moving intermediate results to CPU. Default is False.
        patch_size : int, optional
            Size of the patches to process the image in. Default is 512.
    
        Returns
        -------
        torch.Tensor
            The result of the convolution, reshaped to match the input image dimensions.
        """
        # Convert image to dtype and normalize
        image_torch = torch.from_numpy((image)).to(dtype=dtype).to(device).view(1, 1, image.shape[0], image.shape[1])
        kernel_size = bank.shape[3]
        margin = kernel_size // 2

        def process_patch(patch: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                if gating:
                    # Convert filters to float32
                    bank_0, bank_1 = bank[:, :, 0], bank[:, :, 1]
                    filters_torch_0 = torch.from_numpy(bank_0).to(dtype=dtype).to(device).view(
                        bank_0.shape[0] * bank_0.shape[1], 1, bank_0.shape[2], bank_0.shape[3])
                    filters_torch_1 = torch.from_numpy(bank_1).to(dtype=dtype).to(device).view(
                        bank_1.shape[0] * bank_1.shape[1], 1, bank_1.shape[2], bank_1.shape[3])

                    # Perform convolutions
                    if save_memory:
                        res0 = F.conv2d(patch, filters_torch_0, padding='same').to('cpu')
                        del filters_torch_0
                        res1 = F.conv2d(patch, filters_torch_1, padding='same').to('cpu')
                        del filters_torch_1
                    else:
                        res0 = F.conv2d(patch, filters_torch_0, padding='same')
                        del filters_torch_0
                        res1 = F.conv2d(patch, filters_torch_1, padding='same')
                        del filters_torch_1
                    del patch

                    # Multiply results as torch tensors
                    result = res0 * res1
                    del res0, res1
                else:
                    # Combine filters
                    combined_filters = bank[:, :, 0] + bank[:, :, 1]
                    filters_torch = torch.from_numpy(combined_filters).to(dtype=dtype).to(device).view(
                        combined_filters.shape[0] * combined_filters.shape[1], 1, combined_filters.shape[2],
                        combined_filters.shape[3])

                    # Perform convolution
                    if save_memory:
                        result = F.conv2d(patch, filters_torch, padding='same').to('cpu')
                    else:
                        result = F.conv2d(patch, filters_torch, padding='same')

            return result

        # Process image in patches with overlap
        if image.shape[0] <= patch_size and image.shape[1] <= patch_size:
            return process_patch(image_torch).view(bank.shape[0], bank.shape[1], image.shape[0], image.shape[1])

        output = torch.zeros(bank.shape[0], bank.shape[1], image.shape[0], image.shape[1], dtype=dtype, device=device)
        for i in range(0, image.shape[0], patch_size - 2 * margin):
            for j in range(0, image.shape[1], patch_size - 2 * margin):
                patch = image_torch[:, :, max(i - margin, 0):min(i + patch_size + margin, image.shape[0]),
                        max(j - margin, 0):min(j + patch_size + margin, image.shape[1])]
                patch_result = process_patch(patch).view(bank.shape[0], bank.shape[1], patch.shape[2], patch.shape[3])

                # Determine the region to place the patch result
                start_i = i
                end_i = min(i + patch_size, image.shape[0])
                start_j = j
                end_j = min(j + patch_size, image.shape[1])

                # Calculate the corresponding region in the patch result
                patch_start_i = 0 if i == 0 else margin
                patch_end_i = (end_i - start_i) + patch_start_i
                patch_start_j = 0 if j == 0 else margin
                patch_end_j = (end_j - start_j) + patch_start_j

                output[:, :, start_i:end_i, start_j:end_j] = patch_result[:, :, patch_start_i:patch_end_i,
                                                             patch_start_j:patch_end_j]

        return output.view(bank.shape[0], bank.shape[1], image.shape[0], image.shape[1])

    @staticmethod
    def argmax_wavelets(result: torch.Tensor, len_range: torch.Tensor, orient_range: torch.Tensor) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute argmax of wavelet convolution results to extract length, orientation and score maps.

        Parameters
        ----------
        result : torch.Tensor
            Wavelet convolution result of shape ``(num_lengths, num_orientations, height, width)``.
        len_range : torch.Tensor
            Lengths used in the wavelet bank, shape ``(num_lengths,)``.
        orient_range : torch.Tensor
            Orientation angles (radians) used in the wavelet bank, shape ``(num_orientations,)``.

        Returns
        -------
        length_np : np.ndarray
            Optimal length per pixel, shape ``(height, width)``.
        orient_np : np.ndarray
            Optimal orientation (radians) per pixel, shape ``(height, width)``.
        max_score_np : np.ndarray
            Maximum convolution score per pixel, shape ``(height, width)``.
        """
        # Keep the reshaping and max operation on the GPU
        result_reshaped = result.permute(2, 3, 0, 1).view(result.shape[2] * result.shape[3], -1)
        max_score, argmax = torch.max(result_reshaped, 1)
        max_score = max_score.view(result.shape[2], result.shape[3])

        # Calculate indices for lengths and orientations using PyTorch
        len_indices = argmax // result.shape[1]
        orient_indices = argmax % result.shape[1]
        length = len_range[len_indices].view(result.shape[2], result.shape[3])
        orient = orient_range[orient_indices].view(result.shape[2], result.shape[3])

        return length.cpu().numpy(), orient.cpu().numpy(), max_score.cpu().numpy()

    @staticmethod
    def get_sarcomere_vectors_wavelet(length: np.ndarray, orientation: np.ndarray, max_score: np.ndarray,
                                      len_range: np.ndarray, mbands: Union[np.ndarray, None] = None,
                                      score_threshold: float = 0.2, abs_threshold: bool = True) -> tuple:
        """
        Extract vector positions on sarcomere M-bands and compute length and orientation.

        Thresholds the score map, binarizes and skeletonizes M-band regions,
        labels connected components, extracts midline coordinates and reads the
        sarcomere length, orientation and score at each point.

        Parameters
        ----------
        length : np.ndarray
            Sarcomere length map from wavelet analysis.
        orientation : np.ndarray
            Sarcomere orientation angle map from wavelet analysis.
        max_score : np.ndarray
            Map of maximal wavelet scores.
        len_range : np.ndarray
            Lengths used in the wavelet bank.
        mbands : np.ndarray or None, optional
            If not None, a manually curated / corrected M-band mask to use.
            Default is None.
        score_threshold : float, optional
            Threshold for filtering detected sarcomeres; an absolute value if
            ``abs_threshold`` is True, otherwise a percentile for adaptive
            thresholding. Default is 0.2.
        abs_threshold : bool, optional
            If True, ``score_threshold`` is an absolute value; if False, a
            percentile for adaptive thresholding. Default is True.

        Returns
        -------
        tuple
            ``(pos_vectors_px, mband_id_vectors, mband_length_vectors,
            sarcomere_length_vectors, sarcomere_orientation_vectors,
            max_score_vectors, mbands, mbands_labels, score_threshold)``:
            M-band point coordinates (pixels), M-band labels, approximate M-band
            lengths (pixels), sarcomere lengths (µm), orientation angles, maximal
            wavelet scores, the binarized M-band mask, the labeled M-band mask and
            the final threshold value used.
        """
        # rough thresholding of sarcomere structures to better identify adaptive threshold
        # determine adaptive threshold from value distribution
        if not abs_threshold:
            score_threshold_val = max_score.max() * score_threshold
        else:
            score_threshold_val = score_threshold

        # binarize and skeletonize B-bands
        if mbands is None:
            mbands = max_score >= score_threshold_val
        mbands_skel = skeletonize(mbands, method='lee') > 0

        # label mbands
        mbands_labels, n_mbands = ndimage.label(mbands_skel, ndimage.generate_binary_structure(2, 2))

        # iterate mbands and create additional list with labels and mbands length (approximated by max. Feret diameter)
        props = skimage.measure.regionprops_table(mbands_labels, properties=['label', 'coords', 'feret_diameter_max'])
        list_labels, coords_mbands, length_mbands = props['label'], props['coords'], props['feret_diameter_max']

        pos_vectors_px, _pos_vectors, mband_id_vectors, mband_length_vectors = [], [], [], []
        if n_mbands > 0:
            for i, (label_i, coords_i, length_mband_i) in enumerate(
                    zip(list_labels, coords_mbands, length_mbands)):
                pos_vectors_px.append(coords_i)
                mband_length_vectors.append(np.ones(coords_i.shape[0]) * length_mband_i)
                mband_id_vectors.append(np.ones(coords_i.shape[0]) * label_i)

            pos_vectors_px = np.concatenate(pos_vectors_px, axis=0)
            mband_id_vectors = np.concatenate(mband_id_vectors)
            mband_length_vectors = np.concatenate(mband_length_vectors)

            # get sarcomere orientation and distance at vectors, additionally filter score
            sarcomere_length_vectors = length[pos_vectors_px[:, 0], pos_vectors_px[:, 1]]
            sarcomere_orientation_vectors = orientation[pos_vectors_px[:, 0], pos_vectors_px[:, 1]]
            max_score_vectors = max_score[pos_vectors_px[:, 0], pos_vectors_px[:, 1]]

            # remove vectors outside range of sarcomere lengths in wavelet bank
            ids_in = (sarcomere_length_vectors >= len_range[1]) & (sarcomere_length_vectors <= len_range[-2])
            pos_vectors_px = pos_vectors_px[ids_in]
            mband_length_vectors = mband_length_vectors[ids_in]
            mband_id_vectors = mband_id_vectors[ids_in]
            sarcomere_length_vectors = sarcomere_length_vectors[ids_in]
            sarcomere_orientation_vectors = sarcomere_orientation_vectors[ids_in]
            max_score_vectors = max_score_vectors[ids_in]
        else:
            sarcomere_length_vectors, sarcomere_orientation_vectors, max_score_vectors = [], [], []

        return (pos_vectors_px, mband_id_vectors, mband_length_vectors, sarcomere_length_vectors,
                sarcomere_orientation_vectors, max_score_vectors, mbands, mbands_labels, score_threshold)

    @staticmethod
    def interpolate_distance_map(image, N=50, method='linear'):
        """
        Interpolate small NaN regions in a 2D image.

        Only connected NaN regions with size <= N are filled; larger regions are
        left unchanged.

        Parameters
        ----------
        image : np.ndarray
            2D input image; NaN values mark gaps to be filled.
        N : int, optional
            Maximum size (in pixels) of connected NaN regions to interpolate.
            Default is 50.
        method : {'linear', 'nearest', 'cubic'}, optional
            Interpolation method. Default is 'linear'.

        Returns
        -------
        np.ndarray
            Image with the same shape, with small NaN regions interpolated.
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
    def get_pixel_size(file_path):
        """
        Retrieve pixel size (x, y) in µm from a TIFF file.

        Prioritizes ImageJ metadata, then falls back to TIFF resolution tags.

        Parameters
        ----------
        file_path : str
            Path to the TIFF file.

        Returns
        -------
        tuple of float
            Pixel size ``(x, y)`` in µm.

        Raises
        ------
        ValueError
            If the pixel size cannot be determined.
        """
        with tifffile.TiffFile(file_path) as tif:
            # Handle ImageJ metadata case
            if ij_meta := tif.imagej_metadata:
                unit_conversion = {
                    'm': 1e6, 'cm': 1e4, 'mm': 1e3,
                    'um': 1, 'µm': 1, 'nm': 1e-3
                }.get(ij_meta.get('unit', 'um').lower(), 1)

                x_res = tif.pages[0].tags.get('XResolution', (1, 1)).value
                y_res = tif.pages[0].tags.get('YResolution', (1, 1)).value

                if x_res == (1, 1) or y_res == (1, 1):
                    raise ValueError("Could not determine pixel size from ImageJ metadata")

                return (
                    (x_res[1] / x_res[0]) * unit_conversion,
                    (y_res[1] / y_res[0]) * unit_conversion,
                )

            # Handle standard TIFF case
            page = tif.pages[0]
            x_px = page.tags.get('XResolution', (1, 1)).value
            y_px = page.tags.get('YResolution', (1, 1)).value

            if x_px == (1, 1) or y_px == (1, 1):
                raise ValueError("Could not determine pixel size from TIFF tags")

            unit = page.tags.get('ResolutionUnit', 1).value  # 1=None, 2=inch, 3=cm

            conversion = 1
            if unit == 2:  # Convert inches to µm
                conversion = 25400
            elif unit == 3:  # Convert cm to µm
                conversion = 10000

            return (
                (x_px[1] / x_px[0]) * conversion,
                (y_px[1] / y_px[0]) * conversion,
            )

