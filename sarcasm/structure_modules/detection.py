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

"""Neural network-based detection module for sarcomeres and Z-bands."""

import os
from typing import Tuple, Union, List
import numpy as np
import torch
import tifffile
from bio_image_unet import multi_output_unet3d as unet3d
from bio_image_unet.multi_output_unet.multi_output_nested_unet import MultiOutputNestedUNet_3Levels
from bio_image_unet.multi_output_unet.predict import Predict as Predict_UNet
from bio_image_unet.progress import ProgressNotifier

from sarcasm.utils import Utils


def detect_sarcomeres_unet(images: np.ndarray, model_path: str, base_dir: str, model_dir: str,
                          pixelsize: float, max_patch_size: Tuple[int, int] = (1024, 1024),
                          normalization_mode: str = 'all', clip_thres: Tuple[float, float] = (0., 99.98),
                          rescale_factor: float = 1.0, device: Union[torch.device, str] = 'auto',
                          progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
    """
    Predict sarcomeres (Z-bands, mbands, distance, orientation) with U-Net.

    Parameters
    ----------
    images : np.ndarray
        Images to process.
    model_path : str, optional
        Path of trained network weights for U-Net. Default is None.
    base_dir : str
        Base directory for output files.
    model_dir : str
        Directory containing model files.
    pixelsize : float
        Pixel size in micrometers.
    max_patch_size : tuple of int, optional
        Maximal patch dimensions for convolutional neural network (n_x, n_y).
        Default is (1024, 1024).
    normalization_mode : str, optional
        Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
        'all': based on histogram of full stack, 'first': based on histogram of first image in stack).
        Default is 'all'.
    clip_thres : tuple of float, optional
        Clip threshold (lower / upper) for intensity normalization. Default is (0., 99.8).
    rescale_factor : float, optional
        Factor by which to rescale the input images in the XY dimensions before prediction.
        For example, 0.5 reduces the XY resolution by half.
        The images and all subsequent outputs will be rescaled back to their original resolution after prediction.
        Default is 1.0 (no rescaling).
    device : torch.device or str
        Device on which PyTorch kernels are executed.
    progress_notifier : ProgressNotifier, optional
        Progress notifier for inclusion in GUI. Default is ProgressNotifier.progress_notifier_tqdm().

    Returns
    -------
    None
    """
    max_patch_size = Utils.check_and_round_max_patch_size(max_patch_size)
    
    if images.ndim < 2:
        raise ValueError("Images must be at least 2D (Y,X) to have XY dimensions for rescaling.")
    original_xy_shape = images.shape[-2:]

    if rescale_factor != 1.0:
        from skimage.transform import rescale

        current_ndim = images.ndim
        if current_ndim == 2:  # Input is (Y, X)
            # Scale factors for Y, X
            scale_vector = (rescale_factor, rescale_factor)
        elif current_ndim == 3:  # Input is (Z, Y, X) or (T, Y, X)
            # Scale factors for Z, Y, X (or T, Y, X) - only scale last two
            scale_vector = (1.0, rescale_factor, rescale_factor)
        else:
            raise ValueError(f"Unsupported image dimensionality for rescaling: {current_ndim}D. Expected 2D or 3D.")

        print(f"Rescaling image from {images.shape} by factor {round(rescale_factor, 4)} on XY axes...")
        images = rescale(
            images,
            scale_vector,
            order=0,
            mode='reflect',
            preserve_range=True,
            channel_axis=None
        ).astype(images.dtype)
        print(f"Rescaled image shape: {images.shape}")

    print('\nPredicting sarcomeres ...')
    if model_path is None or model_path == 'generalist':
        model_path = os.path.join(model_dir, 'model_sarcomeres_generalist.pt')
        if pixelsize is not None and pixelsize < 0.1:
            print(
                f"\nWARNING FOR GENERALIST MODEL: Pixel size ({round(pixelsize, 3)} µm) is smaller than the optimal range "
                f"(0.1-0.35 µm). For using it pixelsize might be too small. Consider increasing rescale_factor for optimal results.")
        elif pixelsize is not None and pixelsize > 0.35:
            print(
                f"\nWARNING FOR GENERALIST MODEL: Pixel size ({round(pixelsize, 3)} µm) is larger than the optimal range "
                f"(0.1-0.35 µm). For using it pixelsize might be too large. Consider decreasing rescale_factor for optimal results.")
        print(f"Using default model: {model_path}. ")
    _ = Predict_UNet(images, model_params=model_path, result_path=base_dir,
                     max_patch_size=max_patch_size, normalization_mode=normalization_mode,
                     network=MultiOutputNestedUNet_3Levels,
                     clip_threshold=clip_thres, device=device,
                     progress_notifier=progress_notifier)
    del _
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rescale_factor != 1.0:
        # Get output file paths
        output_files = [
            os.path.join(base_dir, 'zbands.tif'),
            os.path.join(base_dir, 'mbands.tif'),
            os.path.join(base_dir, 'orientation.tif'),
            os.path.join(base_dir, 'cell_mask.tif'),
            os.path.join(base_dir, 'sarcomere_mask.tif')
        ]

        output_dir = os.path.dirname(output_files[0])  # Save in same directory

        Utils.scale_back(
            paths=output_files,
            original_xy_shape=original_xy_shape,
            output_dir=output_dir,
            mask_data=False
        )


def detect_z_bands_fast_movie_unet(images: np.ndarray, model_path: str, base_dir: str, model_dir: str,
                                  max_patch_size: Tuple[int, int, int] = (32, 256, 256),
                                  normalization_mode: str = 'all',
                                  clip_thres: Tuple[float, float] = (0., 99.8),
                                  device: Union[torch.device, str] = 'auto',
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
    """
    Predict sarcomere z-bands with 3D U-Net for high-speed movies for improved temporal consistency.

    Parameters
    ----------
    images : np.ndarray
        Images to process.
    model_path : str, optional
        Path of trained network weights for 3D U-Net. Default is None.
    base_dir : str
        Base directory for output.
    model_dir : str
        Directory containing model files.
    max_patch_size : tuple of int, optional
        Maximal patch dimensions for convolutional neural network (n_frames, n_x, n_y).
        Dimensions need to be divisible by 16. Default is (32, 256, 256).
    normalization_mode : str, optional
        Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
        'all': based on histogram of full stack, 'first': based on histogram of first image in stack).
        Default is 'all'.
    clip_thres : tuple of float, optional
        Clip threshold (lower / upper) for intensity normalization. Default is (0., 99.8).
    device : torch.device or str
        Device for PyTorch.
    progress_notifier : ProgressNotifier, optional
        Progress notifier for inclusion in GUI. Default is ProgressNotifier.progress_notifier_tqdm().

    Returns
    -------
    None
    """
    print('\nPredicting sarcomere z-bands ...')

    if model_path is None:
        model_path = os.path.join(model_dir, 'model_z_bands_unet3d.pt')
    max_patch_size = Utils.check_and_round_max_patch_size(max_patch_size)
    if len(max_patch_size) != 3:
        raise ValueError('patch size for prediction has to be be (frames, x, y)')
    _ = unet3d.Predict(images, model_params=model_path, result_path=base_dir,
                       max_patch_size=max_patch_size, normalization_mode=normalization_mode,
                       device=device, clip_threshold=clip_thres, progress_notifier=progress_notifier)
    del _
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze_cell_mask_from_file(file_cell_mask: str, images: np.ndarray, pixelsize: float,
                                frames: Union[str, int, List[int], np.ndarray] = 'all',
                                threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyzes the area occupied by cells in the given image(s) and calculates the average cell intensity and
    cell area ratio.

    Parameters
    ----------
    file_cell_mask : str
        Path to the cell mask file.
    images : np.ndarray
        Raw images.
    pixelsize : float
        Pixel size in micrometers.
    frames: {'all', int, list, np.ndarray}, optional
        Frames for z-band analysis ('all' for all frames, int for a single frame, list or ndarray for
        selected frames). Defaults to 'all'.
    threshold : float, optional
        Threshold value for binarizing the cell mask image. Pixels with intensity
        above threshold are considered cell. Defaults to 0.1.
        
    Returns
    -------
    cell_area : np.ndarray
        Cell area in square micrometers.
    cell_area_ratio : np.ndarray
        Ratio of cell area to total image area.
    cell_mask_intensity : np.ndarray
        Average intensity in the cell mask.
    """
    if not os.path.exists(file_cell_mask):
        raise FileNotFoundError("Cell mask not found. Please run detect_sarcomeres first.")
        
    if isinstance(frames, str) and frames == 'all':
        cell_mask = tifffile.imread(file_cell_mask)
    else:
        cell_mask = tifffile.imread(file_cell_mask, key=frames)

    if len(cell_mask.shape) == 2:
        cell_mask = np.expand_dims(cell_mask, 0)
    if len(images.shape) == 2:
        images = np.expand_dims(images, 0)

    n_imgs = len(images)

    # create empty arrays
    cell_area, cell_area_ratio = np.full(n_imgs, fill_value=np.nan), np.full(n_imgs, fill_value=np.nan)
    cell_mask_intensity = np.full(n_imgs, fill_value=np.nan)

    for i, (img_i, cell_mask_i) in enumerate(zip(images, cell_mask)):
        # binarize mask
        mask_i = cell_mask_i > threshold

        # average cell intensity
        cell_mask_intensity[i] = np.mean(img_i[mask_i])

        # total cell area and ratio to total image area
        cell_area[i] = np.sum(mask_i) * pixelsize ** 2
        cell_area_ratio[i] = cell_area[i] / (img_i.shape[0] * img_i.shape[1] * pixelsize ** 2)

    return cell_area, cell_area_ratio, cell_mask_intensity
