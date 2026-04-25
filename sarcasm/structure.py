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


import glob
import logging
import os
import shutil
from typing import Optional, Tuple, Union, List, Literal, Any

import numpy as np
import tifffile
import torch
from bio_image_unet.progress import ProgressNotifier
from scipy import stats, sparse

from sarcasm.core import SarcAsM
from sarcasm.ioutils import IOUtils
from sarcasm.utils import Utils

logger = logging.getLogger(__name__)

# Import structure modules
from sarcasm.structure_modules import (
    z_band_analysis,
    sarcomere_vectors,
    myofibril_analysis,
    domain_clustering,
    domain_motion,
    kymograph,
    detection,
    loi_detection,
    sarcomere_tracking,
)

class Structure(SarcAsM):
    """
    Class for analyzing sarcomere morphology.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to the image tif file.
    restart : bool, optional
        If ``True`` the previous analysis folder is deleted and a fresh run is
        started (default: ``False``).
    pixelsize : float or None, optional
        Physical pixel size in µm.  If ``None`` the value is taken from file
        metadata; otherwise the supplied number overrides all metadata.
    frametime : float or None, optional
        Time between frames in s.  If ``None`` it is taken from metadata; an
        explicit number overrides it.
    channel : int | None, optional
        Index of the fluorescence channel that shows the sarcomeres.  If the
        image has only one channel this argument is ignored.
    axes : str | None, optional
        Explicit dimension order (e.g. ``'TXYC'``).  ``None`` lets the base
        class auto-detect the order.
    auto_save : bool, optional
        Write analysis results to disk automatically (default ``True``).
    use_gui : bool, optional
        Activate GUI mode (default ``False``).
    device : torch.device | Literal['auto'], optional
        Device on which PyTorch kernels are executed.  ``'auto'`` selects CUDA
        or MPS when available (default ``'auto'``).
    **info : Any
        Additional key-value pairs that are stored in the metadata file.

    Attributes
    ----------
    data : dict
        Dictionary that contains numeric results of the morphology analysis
        (populated after running the respective detection routines).
    """

    def __init__(self,
                 file_path: Union[str, os.PathLike],
                 restart: bool = False,
                 pixelsize: Union[float, None] = None,
                 frametime: Union[float, None] = None,
                 channel: Union[int, None] = None,
                 axes: Union[str, None] = None,
                 auto_save: bool = True,
                 use_gui: bool = False,
                 device: Union[torch.device, Literal['auto']] = 'auto',
                 **info: Any) -> None:
        """
        Instantiate a Structure object and initialize the common SarcAsM base.
        """
        super().__init__(
            file_path=file_path,
            restart=restart,
            pixelsize=pixelsize,
            frametime=frametime,
            channel=channel,
            axes=axes,
            auto_save=auto_save,
            use_gui=use_gui,
            device=device,
            **info
        )

        # Initialize structure data dictionary
        if os.path.exists(self.__get_structure_data_file()):
            self._load_structure_data()
        else:
            self.data = {}

    def __get_structure_data_file(self, is_temp_file: bool = False) -> str:
        """
        Returns the path to the structure data file.

        Parameters
        ----------
        is_temp_file : bool, optional
            If True, returns the path to a temporary file. This temporary file is used to prevent
            creating corrupted data files due to aborted operations (e.g., exceptions or user intervention).
            The temporary file can be committed to a final file by renaming it. Default is False.

        Returns
        -------
        str
            The path to the structure data file, either temporary or final.
        """
        file_name = "structure.temp.json" if is_temp_file else "structure.json"
        return os.path.join(self.data_dir, file_name)

    def commit(self) -> None:
        """
        Commit data by renaming the temporary file to the final data file.
        """
        temp_file_path = self.__get_structure_data_file(is_temp_file=True)
        final_file_path = self.__get_structure_data_file()

        if os.path.exists(temp_file_path):
            if os.path.exists(final_file_path):
                os.remove(final_file_path)
            os.rename(temp_file_path, final_file_path)

    def store_structure_data(self, override: bool = True) -> None:
        """
        Store structure data in a JSON file.

        Parameters
        ----------
        override : bool, optional
            If True, override the file.
        """
        if override or not os.path.exists(self.__get_structure_data_file(is_temp_file=False)):
            IOUtils.json_serialize(self.data, self.__get_structure_data_file(is_temp_file=True))
            self.commit()

    def _load_structure_data(self) -> None:
        """
        Load structure data from the final data file; fall back to the temporary file if needed.
        Raises
        ------
        Exception
            If no valid structure data could be loaded.
        """
        try:
            if os.path.exists(self.__get_structure_data_file(is_temp_file=False)):
                self.data = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=False))
            elif os.path.exists(self.__get_structure_data_file(is_temp_file=True)):
                self.data = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=True))
        except Exception as e:
            logger.warning(f"Failed to load persistent structure data file: {e}. Attempting to load temporary file...")
            if os.path.exists(self.__get_structure_data_file(is_temp_file=True)):
                try:
                    self.data = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=True))
                    logger.debug("Successfully loaded structure data from temporary file")
                except Exception as temp_e:
                    logger.error(f"Failed to load temporary structure data file: {temp_e}")

        # ensure compatibility with data from early version
        keys_old = {'points': 'pos_vectors', 'sarcomere_length_points': 'sarcomere_length_vectors',
                    'midline_length_points': 'midline_length_vectors', 'midline_id_points': 'midline_id_vectors',
                    'sarcomere_orientation_points': 'sarcomere_orientation_vectors',
                    'max_score_points': 'max_score_vectors'}
        for key, val in keys_old.items():
            if key in self.data:
                self.data[val] = self.data[key]
        keys = [key for key in self.data.keys() if 'timepoints' in key]
        for key in keys:
            new_key = key.replace('timepoints', 'frames')
            self.data[new_key] = self.data[key]
            if isinstance(self.data[new_key], str) and self.data[new_key] == 'all':
                n_stack = self.metadata.n_stack if self.metadata.n_stack is not None else 0
                self.data[new_key] = list(range(n_stack))

        if self.data is None:
            raise Exception('Loading of structure failed')

    def get_list_lois(self):
        """Returns list of LOIs"""
        return Utils.get_lois_of_file(self.file_path)

    def detect_sarcomeres(self, frames: Union[str, int, List[int], np.ndarray] = 'all',
                          model_path: str = None, max_patch_size: Tuple[int, int] = (1024, 1024),
                          normalization_mode: str = 'all', clip_thres: Tuple[float, float] = (0., 99.98),
                          rescale_factor: float = 1.0,
                          progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """
        Predict sarcomeres (Z-bands, mbands, distance, orientation) with U-Net.

        Parameters
        ----------
        frames : Union[str, int, List[int], np.ndarray]
            Frames for sarcomere detection ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). Defaults to 'all'.
        model_path : str, optional
            Path of trained network weights for U-Net. Default is None.
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
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in GUI. Default is ProgressNotifier.progress_notifier_tqdm().

        Returns
        -------
        None
        """
        max_patch_size = Utils.check_and_round_max_patch_size(max_patch_size)
        if isinstance(frames, str) and frames == 'all':
            images = self.read_imgs()
            list_frames = list(range(len(images)))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            images = self.read_imgs(frames=frames)
            if np.issubdtype(type(frames), np.integer):
                list_frames = [frames]
            else:
                list_frames = list(frames)
        else:
            raise ValueError('frames argument not valid')

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

            logger.info(f"Rescaling image from {images.shape} by factor {round(rescale_factor, 4)} on XY axes...")
            images = rescale(
                images,
                scale_vector,
                order=0,
                mode='reflect',
                preserve_range=True,
                channel_axis=None
            ).astype(images.dtype)
            logger.info(f"Rescaled image shape: {images.shape}")

        # Check pixelsize is not None
        if self.metadata.pixelsize is None:
            raise ValueError("Pixel size is not available. Please provide pixelsize during initialization.")
        
        # Delegate to detection module
        detection.detect_sarcomeres_unet(
            images=images,
            model_path=model_path,
            base_dir=self.base_dir,
            model_dir=str(self.model_dir),
            pixelsize=self.metadata.pixelsize,
            max_patch_size=max_patch_size,
            normalization_mode=normalization_mode,
            clip_thres=clip_thres,
            rescale_factor=rescale_factor,
            device=self.device,
            progress_notifier=progress_notifier
        )

        _dict = {
            'params.detect_sarcomeres.frames': list_frames,
            'params.detect_sarcomeres.model': model_path,
            'params.detect_sarcomeres.normalization_mode': normalization_mode,
            'params.detect_sarcomeres.clip_threshold': clip_thres,
            'params.detect_sarcomeres.rescale_factor': rescale_factor,
        }
        self.data.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def _remap_mask_key(self, list_frames: List[int], detected_frames: Any) -> Union[int, List[int]]:
        """Translate movie-frame indices to page indices inside a sparsely-saved mask TIFF.

        Masks are stored only for frames passed to detect_sarcomeres, in detection order.
        When `detected_frames` covers every frame this is an identity mapping, so we just
        return the original indices; otherwise we look up each requested frame's position.
        """
        if detected_frames == 'all' or detected_frames is None:
            return list_frames[0] if len(list_frames) == 1 else list_frames
        if isinstance(detected_frames, (int, np.integer)):
            detected_list = [int(detected_frames)]
        else:
            detected_list = [int(f) for f in detected_frames]
        if detected_list == list(range(self.metadata.n_stack)):
            return list_frames[0] if len(list_frames) == 1 else list_frames
        try:
            keys = [detected_list.index(f) for f in list_frames]
        except ValueError:
            raise ValueError(
                f"Requested frame(s) {list_frames} not present in detected frames "
                f"{detected_list}. Run detect_sarcomeres on these frames first.")
        return keys[0] if len(keys) == 1 else keys

    def load_mask_full_stack(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load a mask TIFF and expand it to full stack length in memory for display.

        Masks are saved sparsely (only for detected frames) to save disk space. For napari
        display alongside the raw movie, this returns an (n_stack, ...) array with computed
        frames placed at their original frame indices and zeros elsewhere. Returns None if
        the file does not exist.
        """
        if not os.path.exists(file_path):
            return None
        arr = tifffile.imread(file_path)
        n_stack = self.metadata.n_stack
        if n_stack <= 1:
            return arr
        detected = self.data.get('params.detect_sarcomeres.frames', 'all')
        if detected == 'all' or detected is None:
            return arr
        if isinstance(detected, (int, np.integer)):
            detected = [int(detected)]
        else:
            detected = [int(f) for f in detected]
        if len(detected) == n_stack:
            return arr
        if arr.ndim >= 3 and arr.shape[0] == len(detected):
            per_frame_shape = arr.shape[1:]
            frames_iter = lambda i: arr[i]
        elif len(detected) == 1:
            per_frame_shape = arr.shape
            frames_iter = lambda i: arr
        else:
            return arr  # unexpected shape, pass through
        full = np.zeros((n_stack,) + per_frame_shape, dtype=arr.dtype)
        for i, f in enumerate(detected):
            if 0 <= f < n_stack:
                full[f] = frames_iter(i)
        return full

    def detect_z_bands_fast_movie(self, model_path: Optional[str] = None,
                                  max_patch_size: Tuple[int, int, int] = (32, 256, 256),
                                  normalization_mode: str = 'all',
                                  clip_thres: Tuple[float, float] = (0., 99.8),
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Predict sarcomere z-bands with 3D U-Net for high-speed movies for improved temporal consistency.

        Parameters
        ----------
        model_path : str, optional
            Path of trained network weights for 3D U-Net. Default is None.
        max_patch_size : tuple of int, optional
            Maximal patch dimensions for convolutional neural network (n_frames, n_x, n_y).
            Dimensions need to be divisible by 16. Default is (32, 256, 256).
        normalization_mode : str, optional
            Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
            'all': based on histogram of full stack, 'first': based on histogram of first image in stack).
            Default is 'all'.
        clip_thres : tuple of float, optional
            Clip threshold (lower / upper) for intensity normalization. Default is (0., 99.8).
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in GUI. Default is ProgressNotifier.progress_notifier_tqdm().

        Returns
        -------
        None
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'model_z_bands_unet3d.pt')
        
        # Delegate to detection module
        detection.detect_z_bands_fast_movie_unet(
            images=self.read_imgs(),
            model_path=model_path,
            base_dir=self.base_dir,
            model_dir=str(self.model_dir),
            max_patch_size=max_patch_size,
            normalization_mode=normalization_mode,
            clip_thres=clip_thres,
            device=self.device,
            progress_notifier=progress_notifier
        )
        _dict = {'params.detect_z_bands_fast_movie.model': model_path,
                 'params.detect_z_bands_fast_movie.max_patch_size': max_patch_size,
                 'params.detect_z_bands_fast_movie.normalization_mode': normalization_mode,
                 'params.predict_z_bands_fast_movie.clip_threshold': clip_thres}
        self.data.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def analyze_cell_mask(self, frames: Union[str, int, List[int], np.ndarray] = 'all', threshold: float = 0.1) -> None:
        """
        Analyzes the area occupied by cells in the given image(s) and calculates the average cell intensity and
        cell area ratio.

        Parameters
        ----------
        threshold : float, optional
            Threshold value for binarizing the cell mask image. Pixels with intensity
            above threshold are considered cell. Defaults to 0.1.
        frames: {'all', int, list, np.ndarray}, optional
            Frames for z-band analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). Defaults to 'all'.
        """
        if not os.path.exists(self.file_cell_mask):
            raise FileNotFoundError("Cell mask not found. Please run detect_sarcomeres first.")
        _detected_frames = self.data.get('params.detect_sarcomeres.frames', 'all')
        if (isinstance(frames, str) and frames == 'all') or (self.metadata.n_stack == 1 and frames == 0):
            cell_mask = tifffile.imread(self.file_cell_mask)
            images = self.read_imgs()
            list_frames = list(range(len(images)))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            if np.issubdtype(type(frames), np.integer):
                list_frames = [int(frames)]
            else:
                list_frames = [int(f) for f in frames]
            mask_key = self._remap_mask_key(list_frames, _detected_frames)
            cell_mask = tifffile.imread(self.file_cell_mask, key=mask_key)
            images = self.read_imgs(frames=frames)
        else:
            raise ValueError('frames argument not valid')

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
            if self.metadata.pixelsize is not None:
                cell_area[i] = np.sum(mask_i) * self.metadata.pixelsize ** 2
                cell_area_ratio[i] = cell_area[i] / (img_i.shape[0] * img_i.shape[1] * self.metadata.pixelsize ** 2)

        _dict = {'cell_mask_area': cell_area, 'cell_mask_area_ratio': cell_area_ratio,
                 'cell_mask_intensity': cell_mask_intensity,
                 'params.analyze_cell_mask.frames': list_frames,
                 'params.analyze_cell_mask.threshold': threshold}
        self.data.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def analyze_z_bands(self, frames: Union[str, int, List[int], np.ndarray] = 'all', threshold: float = 0.5,
                        min_length: float = 0.2, median_filter_radius: float = 0.2, theta_phi_min: float = 0.4, 
                        a_min: float = 0.3, d_max: float = 3.0, d_min: float = 0.0,
                        progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Segment and analyze sarcomere z-bands.

        Parameters
        ----------
        frames: {'all', int, list, np.ndarray}, optional
            Frames for z-band analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). Defaults to 'all'.
        threshold : float, optional
            Threshold for binarizing z-bands prior to labeling (0 - 1). Defaults to 0.1.
        min_length : float, optional
            Minimal length of z-bands; smaller z-bands are removed (in µm). Defaults to 0.5.
        median_filter_radius : float, optional
            Radius of kernel to smooth sarcomere orientation field. Default is 0.2 µm.
        theta_phi_min : float, optional
            Minimal cosine of the angle between the pointed z-band vector and the connecting vector between ends of z-bands.
            Smaller values are not recognized as connections (for lateral alignment and distance analysis). Defaults to 0.4.
        a_min: float, optional
            Minimal lateral alignment between z-band ends to create a lateral connection. Defaults to 0.3.
        d_max : float, optional
            Maximal distance between z-band ends (in µm). Z-band end pairs with larger distances are not connected
            (for lateral alignment and distance analysis). Defaults to 3.0.
        d_min : float, optional
            Minimal distance between z-band ends (in µm). Z-band end pairs with smaller distances are not connected.
            Defaults to 0.0.
        progress_notifier: ProgressNotifier
            Wraps progress notification, default is progress notification done with tqdm
        """
        if not os.path.exists(self.file_zbands):
            raise FileNotFoundError("Z-band mask not found. Please run detect_sarcomeres first.")
        _detected_frames = self.data.get('params.detect_sarcomeres.frames', 'all')
        if (isinstance(frames, str) and frames == 'all') or (self.metadata.n_stack == 1 and frames == 0):
            zbands = tifffile.imread(self.file_zbands)
            orientation_field = tifffile.imread(self.file_orientation)
            images = self.read_imgs()
            list_frames = list(range(len(images)))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            if np.issubdtype(type(frames), np.integer):
                list_frames = [int(frames)]
            else:
                list_frames = [int(f) for f in frames]
            mask_key = self._remap_mask_key(list_frames, _detected_frames)
            zbands = tifffile.imread(self.file_zbands, key=mask_key)
            # orientation.tif stores 2 pages (channels) per detected frame.
            _ori_keys = ([int(mask_key)] if isinstance(mask_key, (int, np.integer))
                         else [int(k) for k in mask_key])
            _ori_pages = [p for k in _ori_keys for p in (k * 2, k * 2 + 1)]
            orientation_field = tifffile.imread(self.file_orientation, key=_ori_pages)
            orientation_field = orientation_field.reshape(-1, 2, *orientation_field.shape[-2:])
            images = self.read_imgs(frames=frames)
        else:
            raise ValueError('frames argument not valid')

        if len(zbands.shape) == 2:
            zbands = np.expand_dims(zbands, 0)
        if len(images.shape) == 2:
            images = np.expand_dims(images, 0)
        if len(orientation_field.shape) == 3:
            orientation_field = np.expand_dims(orientation_field, 0)
        n_imgs = len(zbands)

        # create empty lists
        def none_lists():
            return [None] * self.metadata.n_stack
        z_length, z_intensity, z_straightness, z_orientation = (none_lists() for _ in range(4))
        z_lat_neighbors, z_lat_alignment, z_lat_dist = (none_lists() for _ in range(3))
        z_lat_size_groups, z_lat_length_groups, z_lat_alignment_groups = (none_lists() for _ in range(3))
        z_labels, z_ends, z_lat_links, z_lat_groups = (none_lists() for _ in range(4))

        # create empty arrays
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        z_length_mean, z_length_std, z_length_max, z_length_sum, z_oop = (nan_arrays() for _ in range(5))
        n_zbands, z_intensity_mean, z_intensity_std = (nan_arrays() for _ in range(3))
        z_mask_area, z_mask_intensity, z_mask_area_ratio = (nan_arrays() for _ in range(3))
        z_straightness_mean, z_straightness_std = (nan_arrays() for _ in range(2))
        z_lat_neighbors_mean, z_lat_neighbors_std = (nan_arrays() for _ in range(2))
        z_lat_alignment_mean, z_lat_alignment_std = (nan_arrays() for _ in range(2))
        z_lat_dist_mean, z_lat_dist_std = (nan_arrays() for _ in range(2))
        z_lat_size_groups_mean, z_lat_size_groups_std = (nan_arrays() for _ in range(2))
        z_lat_length_groups_mean, z_lat_length_groups_std = (nan_arrays(), nan_arrays())
        z_lat_alignment_groups_mean, z_lat_alignment_groups_std = (nan_arrays() for _ in range(2))

        # iterate images
        logger.info('Starting Z-band analysis...')
        for i, (frame_i, zbands_i, image_i, orientation_field_i) in enumerate(
                progress_notifier.iterator(zip(list_frames, zbands, images, orientation_field), total=n_imgs)):

            # Delegate to z_band_analysis module
            labels_i, labels_skel_i = z_band_analysis.segment_z_bands(zbands_i, threshold=threshold)

            # analyze z-band features
            z_band_features = z_band_analysis.analyze_z_bands(
                zbands_i, labels_i, labels_skel_i, image_i, orientation_field_i,
                pixelsize=self.metadata.pixelsize, threshold=threshold,
                min_length=min_length, median_filter_radius=median_filter_radius,
                a_min=a_min, theta_phi_min=theta_phi_min,
                d_max=d_max, d_min=d_min
            )

            (
                z_length_i, z_intensity_i, z_straightness_i, z_mask_intensity_i, z_mask_area_i, orientation_i,
                z_oop_i,
                labels_list_i, labels_i, z_lat_neighbors_i, z_lat_dist_i, z_lat_alignment_i, z_lat_links_i, z_ends_i,
                z_lat_groups_i, z_lat_size_groups_i, z_lat_length_groups_i, z_lat_alignment_groups_i,
            ) = z_band_features

            # fill lists and arrays
            z_length[frame_i] = z_length_i
            z_intensity[frame_i] = z_intensity_i
            z_straightness[frame_i] = z_straightness_i
            z_lat_alignment[frame_i] = z_lat_alignment_i
            z_lat_neighbors[frame_i] = z_lat_neighbors_i
            z_orientation[frame_i] = orientation_i
            z_lat_dist[frame_i] = z_lat_dist_i
            z_lat_size_groups[frame_i] = z_lat_size_groups_i
            z_lat_length_groups[frame_i] = z_lat_length_groups_i
            z_lat_alignment_groups[frame_i] = z_lat_alignment_groups_i
            z_mask_area[frame_i], z_mask_intensity[frame_i], z_oop[
                frame_i] = z_mask_area_i, z_mask_intensity_i, z_oop_i
            if 'cell_mask_area' in self.data:
                z_mask_area_ratio[frame_i] = z_mask_area_i / self.data['cell_mask_area'][frame_i]
            else:
                z_mask_area_ratio[frame_i] = z_mask_area_i / (self.metadata.size[0] * self.metadata.size[1])

            z_labels[frame_i] = sparse.coo_matrix(labels_i)
            z_lat_links[frame_i] = z_lat_links_i
            z_ends[frame_i] = z_ends_i
            z_lat_groups[frame_i] = z_lat_groups_i

            # calculate mean and std of z-band features
            if len(z_length_i) > 0:
                z_length_mean[frame_i], z_length_std[frame_i], z_length_max[frame_i], z_length_sum[frame_i] = np.mean(
                    z_length_i), np.std(
                    z_length_i), np.max(z_length_i), np.sum(z_length_i)
            n_zbands[frame_i] = len(z_length_i)
            z_intensity_mean[frame_i], z_intensity_std[frame_i] = np.mean(z_intensity_i), np.std(z_intensity_i)
            z_straightness_mean[frame_i], z_straightness_std[frame_i] = np.mean(z_straightness_i), np.std(
                z_straightness_i)
            z_lat_neighbors_mean[frame_i], z_lat_neighbors_std[frame_i] = np.mean(z_lat_neighbors_i), np.std(
                z_lat_neighbors_i)
            z_lat_alignment_mean[frame_i], z_lat_alignment_std[frame_i] = np.nanmean(z_lat_alignment_i), np.nanstd(
                z_lat_alignment_i)
            z_lat_dist_mean[frame_i], z_lat_dist_std[frame_i] = np.nanmean(z_lat_dist_i), np.nanstd(z_lat_dist_i)
            z_lat_size_groups_mean[frame_i], z_lat_size_groups_std[frame_i] = np.nanmean(
                z_lat_size_groups_i), np.nanstd(
                z_lat_size_groups_i)
            z_lat_length_groups_mean[frame_i], z_lat_length_groups_std[frame_i] = np.nanmean(
                z_lat_length_groups_i), np.nanstd(
                z_lat_length_groups_i)
            z_lat_alignment_groups_mean[frame_i], z_lat_alignment_groups_std[frame_i] = np.nanmean(
                z_lat_alignment_groups_i), np.nanstd(z_lat_alignment_groups_i)

        # create and save dictionary for cell structure
        z_band_data = {'n_zbands': n_zbands, 'z_length': z_length, 'z_length_mean': z_length_mean, 'z_length_std': z_length_std,
                       'z_length_max': z_length_max, 'z_intensity': z_intensity, 'z_intensity_mean': z_intensity_mean,
                       'z_intensity_std': z_intensity_std, 'z_orientation': z_orientation, 'z_oop': z_oop,
                       'z_straightness': z_straightness, 'z_mask_intensity': z_mask_intensity, 'z_labels': z_labels,
                       'z_straightness_mean': z_straightness_mean, 'z_straightness_std': z_straightness_std,
                       'z_mask_area': z_mask_area, 'z_mask_area_ratio': z_mask_area_ratio, 'z_lat_neighbors': z_lat_neighbors,
                       'z_lat_neighbors_mean': z_lat_neighbors_mean, 'z_lat_neighbors_std': z_lat_neighbors_std,
                       'z_lat_alignment': z_lat_alignment, 'z_lat_alignment_mean': z_lat_alignment_mean,
                       'z_lat_alignment_std': z_lat_neighbors_std, 'z_lat_dist': z_lat_dist, 'z_ends': z_ends,
                       'z_lat_dist_mean': z_lat_dist_mean, 'z_lat_dist_std': z_lat_dist_std, 'z_lat_links': z_lat_links,
                       'z_lat_groups': z_lat_groups, 'z_lat_size_groups': z_lat_size_groups,
                       'z_lat_size_groups_mean': z_lat_size_groups_mean, 'z_lat_size_groups_std': z_lat_size_groups_std,
                       'z_lat_length_groups': z_lat_length_groups, 'z_lat_alignment_groups': z_lat_alignment_groups,
                       'z_lat_length_groups_mean': z_lat_length_groups_mean,
                       'z_lat_length_groups_std': z_lat_length_groups_std,
                       'z_lat_alignment_groups_mean': z_lat_alignment_groups_mean,
                       'z_lat_alignment_groups_std': z_lat_alignment_groups_std,
                       'params.analyze_z_bands.frames': list_frames, 'params.analyze_z_bands.threshold': threshold,
                       'params.analyze_z_bands.min_length': min_length, 'params.analyze_z_bands.median_filter_radius': median_filter_radius,
                       'params.analyze_z_bands.theta_phi_min': theta_phi_min, 'params.analyze_z_bands.d_max': d_max,
                       'params.analyze_z_bands.d_min': d_min}
        self.data.update(z_band_data)
        if self.auto_save:
            self.store_structure_data()

    def analyze_sarcomere_vectors(self, frames: Union[str, int, List[int], np.ndarray] = 'all', threshold_mbands: float = 0.25,
                                  median_filter_radius: float = 0.25, linewidth: float = 0.2, interp_factor: int = 4,
                                  slen_lims: Tuple[float, float] = (1, 3), threshold_sarcomere_mask=0.1,
                                  interpolation_method: str = 'akima',
                                  smooth_orientation_sigma: float = 0.0,
                                  peak_prominence: float = 0.5,
                                  peak_algorithm: str = 'default',
                                  use_fast_movie_zbands: bool = True,
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Extract sarcomere orientation and length vectors.

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}, optional
            frames for sarcomere vector analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). Defaults to 'all'.
        threshold_mbands : float, optional
            Threshold to binarize sarcomere M-bands. Lower values might result in more false-positive sarcomere vectors. Defaults to 0.2.
        median_filter_radius : float, optional
            Radius of kernel to smooth orientation field before assessing orientation at M-points, in µm (default 0.25 µm).
        linewidth : float, optional
            Line width of profile lines to analyze sarcomere lengths, in µm (default is 0.3 µm).
            LOI analysis, tuned for maximum accuracy, uses 0.65 µm — increasing ``linewidth``
            toward that averages over more transverse pixels and smooths per-frame slen.
        interp_factor: int, optional
            Akima/linear upsampling factor applied to each profile before peak detection.
            Default **4** (was 0). LOI analysis uses 6; raising from 0 → 4 was the single
            largest driver of the accuracy gap (sub-pixel peak localisation).
        slen_lims : tuple of float, optional
            Sarcomere size limits in µm (default is (1, 3) µm).
        threshold_sarcomere_mask : float
            Threshold to binarize sarcomere masks. Defaults to 0.1.
        interpolation_method : str, optional
            Interpolation method for profile analysis: 'linear' (fast) or 'akima' (smooth). Defaults to 'akima'.
        smooth_orientation_sigma : float, optional
            Temporal Gaussian sigma (in frames) for smoothing the orientation field across
            the time axis *before* per-frame vector extraction. The U-Net emits orientation
            frame-by-frame with small jitter that propagates into vector positions and
            downstream tracking; temporal smoothing (axially correct via the double-angle
            trick) reduces this jitter. ``0`` disables smoothing (default). ``sigma ≈ 1``
            corresponds to a ~5-frame effective span. Only meaningful for multi-frame
            stacks.
        peak_prominence : float, optional
            ``scipy.signal.find_peaks`` prominence threshold for Z-band peak detection
            inside each profile. Default 0.5 (matches LOI). Lower values accept weaker,
            noisier peaks. Only used when ``peak_algorithm='default'``.
        peak_algorithm : {'default', 'loi'}, optional
            Peak detection routine.

            * ``'default'`` — :func:`sarcasm.utils.Utils.process_profiles_batch` (fast,
              batched, with ``interp_factor`` + ``peak_prominence`` configurable).
            * ``'loi'`` — route every profile through
              :func:`sarcasm.utils.Utils.peakdetekt`, the exact peak-detection +
              6× Akima + COM-refinement pipeline used by the LOI analysis. Parameter
              presets match LOI; ``interp_factor`` + ``peak_prominence`` are ignored.
              Slightly slower but maximum per-frame slen accuracy.
        use_fast_movie_zbands : bool, optional
            If True (default) and ``zbands_fast_movie.tif`` exists on disk (produced by
            :meth:`detect_z_bands_fast_movie`), read Z-bands from that 3D U-Net output
            instead of the per-frame ``zbands.tif``. The 3D model uses a temporal window
            and produces Z-band masks that are ~3× less noisy frame-to-frame at "active"
            pixels, which directly propagates into smoother per-frame slen. Set to
            False if the 3D model is unreliable on this movie (e.g. rejected by manual
            QC) — the method will then fall back to the per-frame 2D masks. The active
            choice is logged and stored in ``params.analyze_sarcomere_vectors.zbands_source``.
        progress_notifier: ProgressNotifier
            Wraps progress notification, default is progress notification done with tqdm

        Returns
        -------
        sarcomere_orientation_points : np.ndarray
            Sarcomere orientation values at midline points.
        sarcomere_length_points : np.ndarray
            Sarcomere length values at midline points.
        """
        if not os.path.exists(self.file_zbands):
            raise FileNotFoundError("Z-band mask not found. Please run detect_sarcomeres first.")

        # Decide which Z-band stack to use (fast-movie 3D U-Net vs per-frame 2D).
        if use_fast_movie_zbands and os.path.exists(self.file_zbands_fast_movie):
            zbands_file = self.file_zbands_fast_movie
            zbands_source = 'fast_movie_3d'
            logger.info(
                'analyze_sarcomere_vectors: using 3D U-Net Z-bands (zbands_fast_movie.tif) — '
                'temporally consistent, recommended for smoother per-frame slen.'
            )
        else:
            zbands_file = self.file_zbands
            if use_fast_movie_zbands:
                zbands_source = 'per_frame_2d'
                logger.info(
                    'analyze_sarcomere_vectors: using per-frame 2D U-Net Z-bands (zbands.tif). '
                    'The 3D fast-movie output is not available; run detect_z_bands_fast_movie() '
                    'to use it.'
                )
            else:
                zbands_source = 'per_frame_2d_forced'
                logger.info(
                    'analyze_sarcomere_vectors: using per-frame 2D U-Net Z-bands '
                    '(zbands.tif) — use_fast_movie_zbands=False.'
                )

        _detected_frames = self.data['params.detect_sarcomeres.frames']
        if ((isinstance(frames, str) and frames == 'all') or (self.metadata.n_stack == 1 and frames == 0)
                or (_detected_frames != 'all' and len(_detected_frames) == 1)):
            list_frames = list(range(self.metadata.n_stack))
            z_bands = tifffile.imread(zbands_file)
            mbands = tifffile.imread(self.file_mbands)
            orientation_field = tifffile.imread(self.file_orientation)
            sarcomere_mask = tifffile.imread(self.file_sarcomere_mask)
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or isinstance(frames, np.ndarray):
            z_bands = tifffile.imread(zbands_file, key=frames)
            mbands = tifffile.imread(self.file_mbands, key=frames)
            orientation_field = tifffile.imread(self.file_orientation)[frames]
            sarcomere_mask = tifffile.imread(self.file_sarcomere_mask, key=frames)
            if np.issubdtype(type(frames), np.integer):
                list_frames = [frames]
            else:
                list_frames = [int(f) for f in frames]
        else:
            raise ValueError('frames argument not valid')
        if len(z_bands.shape) == 2:
            z_bands = np.expand_dims(z_bands, axis=0)
        if len(mbands.shape) == 2:
            mbands = np.expand_dims(mbands, axis=0)
        if len(sarcomere_mask.shape) == 2:
            sarcomere_mask = np.expand_dims(sarcomere_mask, axis=0)
        if len(orientation_field.shape) == 3:
            orientation_field = np.expand_dims(orientation_field, axis=0)

        # Optional temporal smoothing of the orientation field.
        if smooth_orientation_sigma > 0 and orientation_field.shape[0] > 1:
            logger.info(
                f'Temporally smoothing orientation field with sigma={smooth_orientation_sigma:.2f} frames...'
            )
            orientation_field = sarcomere_vectors.smooth_orientation_field_temporal(
                orientation_field, sigma=smooth_orientation_sigma,
            )

        # binarize M-bands
        mbands = mbands > threshold_mbands

        n_frames = len(z_bands)
        pixelsize = self.metadata.pixelsize

        # Check pixelsize is not None
        if pixelsize is None:
            raise ValueError("Pixel size is not available. Please provide pixelsize during initialization.")

        # Pre-compute the orientation angle map for the entire stack once.
        # Inside ``get_sarcomere_vectors`` this call is the second-largest
        # per-frame cost (median filter on a disk footprint); batching it here
        # lets the filter run over the full (N, 2, H, W) tensor and avoids
        # redundant work when ``precomputed_angle_map`` is passed below.
        radius_pixels = max(int(round(median_filter_radius / pixelsize, 0)), 1)
        angle_maps = Utils.get_orientation_angle_map(
            orientation_field, use_median_filter=True, radius=radius_pixels,
        )
        # ``get_orientation_angle_map`` squeezes a single-frame stack down to
        # (H, W); re-expand so ``angle_maps[i]`` is always valid.
        if angle_maps.ndim == 2:
            angle_maps = angle_maps[np.newaxis, ...]

        # create empty arrays
        def none_lists():
            return [None] * self.metadata.n_stack
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        (pos_vectors, pos_vectors_px, sarcomere_length_vectors,
         sarcomere_orientation_vectors) = (none_lists() for _ in range(4))
        midline_id_vectors, midline_length_vectors = (none_lists() for _ in range(2))
        sarcomere_masks = np.zeros((self.metadata.n_stack, *self.metadata.size), dtype=bool)
        (sarcomere_length_mean, sarcomere_length_std) = (nan_arrays() for _ in range(2))
        sarcomere_orientation_mean, sarcomere_orientation_std = nan_arrays(), nan_arrays()
        n_vectors, n_mbands, oop, sarcomere_area, sarcomere_area_ratio, score_thresholds = (nan_arrays() for _ in range(6))

        # iterate images
        logger.info('Starting sarcomere length and orientation analysis...')
        for i, (frame_i, zbands_i, mbands_i, orientation_field_i, sarcomere_mask_i) in enumerate(
                progress_notifier.iterator(zip(list_frames, z_bands, mbands, orientation_field, sarcomere_mask),
                                           total=n_frames)):

            # Delegate to sarcomere_vectors module
            (
                pos_vectors_px_i, pos_vectors_i, midline_id_vectors_i, midline_length_vectors_i,
                sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                n_mbands_i) = sarcomere_vectors.get_sarcomere_vectors(zbands_i, mbands_i,
                                                         orientation_field_i,
                                                         pixelsize=pixelsize,
                                                         median_filter_radius=median_filter_radius,
                                                         slen_lims=slen_lims,
                                                         interp_factor=interp_factor,
                                                         linewidth=linewidth,
                                                         interpolation_method=interpolation_method,
                                                         peak_prominence=peak_prominence,
                                                         peak_algorithm=peak_algorithm,
                                                         precomputed_angle_map=angle_maps[i])

            # write in list
            n_vectors[frame_i] = len(sarcomere_length_vectors_i)
            n_mbands[frame_i] = n_mbands_i
            pos_vectors_px[frame_i] = pos_vectors_px_i
            pos_vectors[frame_i] = pos_vectors_i
            sarcomere_length_vectors[frame_i] = sarcomere_length_vectors_i
            sarcomere_orientation_vectors[frame_i] = sarcomere_orientation_vectors_i
            midline_id_vectors[frame_i] = midline_id_vectors_i
            midline_length_vectors[frame_i] = midline_length_vectors_i

            # calculate mean and std of sarcomere length and orientation
            sarcomere_length_mean[frame_i], sarcomere_length_std[frame_i], = np.nanmean(
                sarcomere_length_vectors_i), np.nanstd(sarcomere_length_vectors_i)
            if np.count_nonzero(~np.isnan(sarcomere_orientation_vectors_i)) > 1:
                sarcomere_orientation_mean[frame_i], sarcomere_orientation_std[frame_i] = stats.circmean(
                    sarcomere_orientation_vectors_i[~np.isnan(sarcomere_orientation_vectors_i)]), stats.circstd(
                    sarcomere_orientation_vectors_i[~np.isnan(sarcomere_orientation_vectors_i)])

            # orientation order parameter
            if len(sarcomere_orientation_vectors_i) > 0:
                oop[frame_i], _ = Utils.analyze_orientations(
                    sarcomere_orientation_vectors_i[~np.isnan(sarcomere_orientation_vectors_i)])

            # calculate sarcomere mask area
            sarcomere_masks[frame_i] = sarcomere_mask_i > threshold_sarcomere_mask
            sarcomere_area[frame_i] = np.sum(sarcomere_mask_i) * self.metadata.pixelsize ** 2
            if 'cell_mask_area' in self.data:
                sarcomere_area_ratio[frame_i] = sarcomere_area[frame_i] / self.data['cell_mask_area'][i]

        vectors_dict = {'params.analyze_sarcomere_vectors.frames': list_frames,
                        'params.analyze_sarcomere_vectors.threshold_sarcomere_mask': threshold_sarcomere_mask,
                        'params.analyze_sarcomere_vectors.median_filter_radius': median_filter_radius,
                        'params.analyze_sarcomere_vectors.slen_lims': slen_lims,
                        'params.analyze_sarcomere_vectors.interp_factor': interp_factor,
                        'params.analyze_sarcomere_vectors.linewidth': linewidth,
                        'params.analyze_sarcomere_vectors.smooth_orientation_sigma': smooth_orientation_sigma,
                        'params.analyze_sarcomere_vectors.peak_prominence': peak_prominence,
                        'params.analyze_sarcomere_vectors.peak_algorithm': peak_algorithm,
                        'params.analyze_sarcomere_vectors.use_fast_movie_zbands': use_fast_movie_zbands,
                        'params.analyze_sarcomere_vectors.zbands_source': zbands_source,
                        'n_vectors': n_vectors, 'n_mbands': n_mbands, 'pos_vectors_px': pos_vectors_px,
                        'pos_vectors': pos_vectors, 'sarcomere_length_vectors': sarcomere_length_vectors,
                        'sarcomere_orientation_vectors': sarcomere_orientation_vectors,
                        'sarcomere_area': sarcomere_area, 'sarcomere_area_ratio': sarcomere_area_ratio,
                        'midline_length_vectors': midline_length_vectors, 'midline_id_vectors': midline_id_vectors,
                        'sarcomere_length_mean': sarcomere_length_mean,
                        'sarcomere_length_std': sarcomere_length_std,
                        'sarcomere_orientation_mean': sarcomere_orientation_mean,
                        'sarcomere_orientation_std': sarcomere_orientation_std,
                        'sarcomere_oop': oop}
        self.data.update(vectors_dict)
        if self.auto_save:
            self.store_structure_data()

    def analyze_myofibrils(self, frames: Optional[Union[str, int, List[int], np.ndarray]] = None,
                           ratio_seeds: float = 0.1, persistence: int = 3, threshold_distance: float = 0.5,
                           n_min: int = 4, median_filter_radius: float = 0.5,
                           progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Estimate myofibril lines by line growth algorithm and analyze length and curvature.

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}, optional
            frames for myofibril analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). If None, frames from sarcomere vector analysis are used. Defaults to None.
        ratio_seeds : float, optional
            Ratio of sarcomere vector used as seeds for line growth. Defaults to 0.1.
        persistence : int, optional
            Persistence of line (average vector length and orientation for prior estimation), needs to be > 0.
            Defaults to 3.
        threshold_distance : float, optional
            Maximal distance for nearest neighbor estimation (in micrometers). Defaults to 0.3.
        n_min : int, optional
            Minimal number of sarcomere line segments per line. Shorter lines are removed. Defaults to 5.
        median_filter_radius : float, optional
            Filter radius for smoothing myofibril length map (in micrometers). Defaults to 0.5.
        progress_notifier: ProgressNotifier
            Wraps progress notification, default is progress notification done with tqdm
        """
        if 'pos_vectors_px' not in self.data:
            raise ValueError('Sarcomere length and orientation not yet analyzed. Run analyze_sarcomere_vectors first.')
        if frames is not None:
            if (isinstance(frames, str) and frames == 'all') or (self.metadata.n_stack == 1 and frames == 0):
                frames = list(range(self.metadata.n_stack))
            if np.issubdtype(type(frames), np.integer):
                frames = [frames]
            if not set(frames).issubset(self.data['params.analyze_sarcomere_vectors.frames']):
                raise ValueError(f'Run analyze_sarcomere_vectors first for frames {frames}.')
        elif frames is None:
            if 'params.analyze_sarcomere_vectors.frames' in self.data.keys():
                frames = self.data['params.analyze_sarcomere_vectors.frames']
            else:
                raise ValueError("To use frames from sarcomere vector analysis, run 'analyze_sarcomere vectors' first!")

        if frames == 'all':
            n_imgs = self.metadata.n_stack
            list_frames = list(range(n_imgs))
        elif isinstance(frames, int):
            list_frames = [frames]
        elif isinstance(frames, list) or type(frames) is np.ndarray:
            list_frames = list(frames)
        else:
            raise ValueError('Selection of frames not valid!')

        pos_vectors_px = [self.data['pos_vectors_px'][frame] for frame in list_frames]
        pos_vectors = [self.data['pos_vectors'][frame] for frame in list_frames]
        sarcomere_length_vectors = [self.data['sarcomere_length_vectors'][frame] for frame in list_frames]
        sarcomere_orientation_vectors = [self.data['sarcomere_orientation_vectors'][frame] for frame in list_frames]
        midline_length_vectors = [self.data['midline_length_vectors'][frame] for frame in list_frames]

        # create empty arrays
        def none_lists():
            return [None] * self.metadata.n_stack
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        length_mean, length_std, length_max = (nan_arrays() for _ in range(3))
        straightness_mean, straightness_std = (nan_arrays() for _ in range(2))
        bending_mean, bending_std = (nan_arrays() for _ in range(2))
        myof_lines, lengths, straightness, frechet_straightness, bending = (none_lists() for _ in range(5))

        # iterate frames
        logger.info('Starting myofibril line analysis...')
        for i, (
                frame_i, pos_vectors_px_i, pos_vectors_i, sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                midline_length_vectors_i) in enumerate(
            progress_notifier.iterator(
                zip(list_frames, pos_vectors_px, pos_vectors, sarcomere_length_vectors, sarcomere_orientation_vectors,
                    midline_length_vectors),
                total=len(pos_vectors_px))):
            if pos_vectors_px_i is None:
                continue
            if len(np.asarray(pos_vectors_px_i).T) > 0:
                # Delegate to myofibril_analysis module
                line_data_i = myofibril_analysis.line_growth(pos_vectors_px_i, sarcomere_length_vectors_i,
                                               sarcomere_orientation_vectors_i,
                                               midline_length_vectors_t=midline_length_vectors_i,
                                               pixelsize=self.metadata.pixelsize, ratio_seeds=ratio_seeds,
                                               persistence=persistence, threshold_distance=threshold_distance,
                                               n_min=n_min)
                lines_i = line_data_i['lines']

                if len(lines_i) > 0:
                    # line lengths and mean squared curvature (msc)
                    lengths_i = line_data_i['line_features']['length_lines']
                    straightness_i = line_data_i['line_features']['straightness_lines']
                    bending_i = line_data_i['line_features']['bending_lines']

                    if len(lengths_i) > 0:
                        # Delegate to myofibril_analysis module for map creation
                        myof_map_i = myofibril_analysis.create_myofibril_length_map(myof_lines=lines_i, myof_length=lengths_i,
                                                                      pos_vectors=pos_vectors_i,
                                                                      sarcomere_orientation_vectors=sarcomere_orientation_vectors_i,
                                                                      sarcomere_length_vectors=sarcomere_length_vectors_i,
                                                                      size=self.metadata.size,
                                                                      pixelsize=self.metadata.pixelsize,
                                                                      median_filter_radius=median_filter_radius)

                        myof_map_flat_i = myof_map_i.flatten()
                        myof_map_flat_i = myof_map_flat_i[~np.isnan(myof_map_flat_i)]
                        weights_i = 1.0 / myof_map_flat_i
                        weighted_mean_length_i = np.average(myof_map_flat_i, weights=weights_i)
                        weighted_std_length_i = np.sqrt(np.average((myof_map_flat_i - weighted_mean_length_i) ** 2,
                                                                   weights=weights_i))
                        length_mean[frame_i], length_std[frame_i], length_max[frame_i] = (weighted_mean_length_i,
                                                                                          weighted_std_length_i,
                                                                                          np.nanmax(myof_map_flat_i))
                        straightness_mean[frame_i], straightness_std[frame_i] = (np.mean(straightness_i),
                                                                                 np.std(straightness_i))
                        bending_mean[frame_i], bending_std[frame_i] = (np.mean(bending_i),
                                                                                     np.std(bending_i))
                    myof_lines[frame_i] = lines_i
                    lengths[frame_i] = lengths_i
                    straightness[frame_i] = straightness_i
                    bending[frame_i] = bending_i

        # update structure dictionary
        myofibril_data = {'myof_length_mean': length_mean,
                          'myof_length_std': length_std, 'myof_lines': myof_lines,
                          'myof_length_max': length_max, 'myof_length': lengths,
                          'myof_straightness': straightness, 'myof_straightness_mean': straightness_mean,
                          'myof_straightness_std': straightness_std,
                          'myof_bending': bending,
                          'myof_bending_mean': bending_mean,
                          'myof_bending_std': bending_std,
                          'params.analyze_myofibrils.persistence': persistence,
                          'params.analyze_myofibrils.threshold_distance': threshold_distance,
                          'params.analyze_myofibrils.frames': list_frames,
                          'params.analyze_myofibrils.n_min': n_min,
                          'params.analyze_myofibrils.ratio_seeds': ratio_seeds,
                          'params.analyze_myofibrils.median_filter_radius': median_filter_radius
                          }

        self.data.update(myofibril_data)
        if self.auto_save:
            self.store_structure_data()

    def analyze_sarcomere_domains(self, frames: Optional[Union[str, int, List[int], np.ndarray]] = None,
                                  d_max: float = 3, cosine_min: float = 0.65, leiden_resolution: float = 0.06,
                                  random_seed: int = 42, area_min: float = 20.0, dilation_radius: float = 0.3,
                                  store_mask: bool = False,
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Cluster sarcomeres into domains based on their spatial and orientational properties using the Leiden algorithm
        for community detection.

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}, optional
            frames for domain analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). If None, frames from sarcomere vector analysis are used. Defaults to None.
        d_max : float
            Max. distance threshold for creating a network edge between vector ends
        cosine_min : float
            Minimal absolute cosine between vector angles for creating a network edge between vector ends
        leiden_resolution : float, optional
            Control parameter for domain size. If resolution is small, the algorithm favors larger domains.
            Greater resolution favors smaller domains. Defaults to 0.05.
        random_seed : int, optional
            Random seed for Leiden algorithm, to ensure reproducibility. Defaults to 2.
        area_min : float, optional
            Minimal area of domains/clusters (in µm^2). Defaults to 50.0.
        dilation_radius : float, optional
            Dilation radius for refining domain area masks, in µm. Defaults to 0.3.
        store_mask : bool, optional
            If True, store the domain mask (integer-labeled image with domain IDs) in self.data.
            Can be memory-intensive for large time-series. Defaults to False.
        progress_notifier: ProgressNotifier
            Wraps progress notification, default is progress notification done with tqdm
        """
        if 'pos_vectors' not in self.data:
            raise ValueError('Sarcomere length and orientation not yet analyzed. Run analyze_sarcomere_vectors first.')
        if frames is not None:
            if (isinstance(frames, str) and frames == 'all') or (self.metadata.n_stack == 1 and frames == 0):
                frames = list(range(self.metadata.n_stack))
            if np.issubdtype(type(frames), np.integer):
                frames = [frames]
            if not set(frames).issubset(self.data['params.analyze_sarcomere_vectors.frames']):
                raise ValueError(f'Run analyze_sarcomere_vectors first for frames {frames}.')
        elif frames is None:
            if 'params.analyze_sarcomere_vectors.frames' in self.data.keys():
                frames = self.data['params.analyze_sarcomere_vectors.frames']
            else:
                raise ValueError("To use frames from sarcomere vector analysis, run 'analyze_sarcomere_vectors' first!")

        if frames == 'all':
            n_imgs = self.metadata.n_stack
            list_frames = list(range(n_imgs))
        elif isinstance(frames, int):
            n_imgs = 1
            list_frames = [frames]
        elif isinstance(frames, list) or type(frames) is np.ndarray:
            n_imgs = len(frames)
            list_frames = list(frames)
        else:
            raise ValueError('Selection of frames not valid!')

        pos_vectors = [np.asarray(self.data['pos_vectors'][t]) for t in list_frames]
        sarcomere_length_vectors = [np.asarray(self.data['sarcomere_length_vectors'][t]) for t in list_frames]
        sarcomere_orientation_vectors = [np.asarray(self.data['sarcomere_orientation_vectors'][t]) for t in list_frames]
        midline_id_vectors = [np.asarray(self.data['midline_id_vectors'][t]) for t in list_frames]

        # create empty arrays
        def none_lists():
            return [None] * self.metadata.n_stack
        def nan_arrays():
            return np.full(self.metadata.n_stack, np.nan)
        n_domains, domain_area_mean, domain_area_std = (nan_arrays() for _ in range(3))
        domain_slen_mean, domain_slen_std = (nan_arrays() for _ in range(2))
        domain_oop_mean, domain_oop_std = (nan_arrays() for _ in range(2))

        (domains, domain_area, domain_slen, domain_slen_std,
         domain_oop, domain_orientation) = (none_lists() for _ in range(6))
        
        # optionally store domain masks
        if store_mask:
            domain_mask = none_lists()

        # iterate frames
        logger.info('Starting sarcomere domain analysis...')
        for i, (frame_i, pos_vectors_i, sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                midline_id_vectors_i) in enumerate(
            progress_notifier.iterator(
                zip(list_frames, pos_vectors, sarcomere_length_vectors, sarcomere_orientation_vectors,
                    midline_id_vectors),
                total=len(pos_vectors))):
            if pos_vectors_i.ndim == 0:
                continue
            # Delegate to domain_clustering module
            cluster_data_t = domain_clustering.cluster_sarcomeres(pos_vectors_i, sarcomere_length_vectors_i,
                                                     sarcomere_orientation_vectors_i,
                                                     pixelsize=self.metadata.pixelsize,
                                                     size=self.metadata.size,
                                                     d_max=d_max, cosine_min=cosine_min,
                                                     leiden_resolution=leiden_resolution, random_seed=random_seed,
                                                     area_min=area_min, dilation_radius=dilation_radius)
            (n_domains[frame_i], domains[frame_i], domain_area[frame_i], domain_slen[frame_i], domain_slen_std[frame_i],
             domain_oop[frame_i], domain_orientation[frame_i], domain_mask_i) = cluster_data_t

            # optionally store domain mask as sparse matrix
            if store_mask:
                domain_mask[frame_i] = sparse.coo_matrix(domain_mask_i)

            # calculate mean and std of domains
            domain_area_mean[frame_i], domain_area_std[frame_i] = np.mean(domain_area[frame_i]), np.std(
                domain_area[frame_i])
            domain_slen_mean[frame_i], domain_slen_std[frame_i] = (
                np.mean(domain_slen[frame_i]), np.std(domain_slen[frame_i]))
            domain_oop_mean[frame_i], domain_oop_std[frame_i] = (
                np.mean(domain_oop[frame_i]), np.std(domain_oop[frame_i]))

        # update structure dictionary
        domain_data = {'n_domains': n_domains, 'domains': domains,
                       'domain_area': domain_area, 'domain_area_mean': domain_area_mean,
                       'domain_area_std': domain_area_std,
                       'domain_slen': domain_slen, 'domain_slen_mean': domain_slen_mean,
                       'domain_slen_std': domain_slen_std,
                       'domain_oop': domain_oop, 'domain_oop_mean': domain_oop_mean,
                       'domain_oop_std': domain_oop_std,
                       'domain_orientation': domain_orientation,
                       'params.analyze_sarcomere_domains.frames': list_frames,
                       'params.analyze_sarcomere_domains.d_max': d_max,
                       'params.analyze_sarcomere_domains.cosine_min': cosine_min,
                       'params.analyze_sarcomere_domains.leiden_resolution': leiden_resolution,
                       'params.analyze_sarcomere_domains.area_min': area_min,
                       'params.analyze_sarcomere_domains.dilation_radius': dilation_radius,
                       'params.analyze_sarcomere_domains.store_mask': store_mask}
        
        # add domain mask if stored
        if store_mask:
            domain_data['domain_mask'] = domain_mask

        self.data.update(domain_data)
        if self.auto_save:
            self.store_structure_data()

    def analyze_domain_motion(
        self,
        reference_frame: int = 0,
        model: Optional[str] = None,
        threshold: float = 0.3,
        contr_time_min: float = 0.2,
        merge_time_max: float = 0.05,
        buffer_frames: int = 3,
        min_valid_frames: float = 0.5,
        filter_params: Tuple[int, int] = (13, 5),
        progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm(),
    ) -> None:
        """
        Analyze sarcomere contraction dynamics within sarcomere domains over time.
        
        Uses domain masks from a reference frame to track mean sarcomere length within
        each domain across all frames. Then detects contraction cycles using ContractionNet
        and computes per-domain contraction parameters.
        
        Prerequisites: Run `analyze_sarcomere_vectors` and `analyze_sarcomere_domains` first.
        
        Parameters
        ----------
        reference_frame : int, optional
            Frame index to use as reference for domain masks. Must be a frame where domains
            were analyzed. Defaults to 0.
        model : str, optional
            Path to ContractionNet model weights (.pt file). If None, uses default model.
        threshold : float, optional
            Binary threshold for contraction state prediction. Default 0.3.
        contr_time_min : float, optional
            Minimal time of contraction in seconds. Shorter contractions are removed. Default 0.2.
        merge_time_max : float, optional
            Maximal time between two contractions. Closer contractions are merged. Default 0.05.
        buffer_frames : int, optional
            Remove contraction cycles within this many frames of start/end of time-series. Default 3.
        min_valid_frames : float, optional
            Minimum fraction of valid (non-NaN) frames required for a domain to be analyzed. Default 0.5.
        filter_params : Tuple[int, int], optional
            Savitzky-Golay filter parameters (window_length, polyorder) for velocity calculation.
            Default (13, 5).
        progress_notifier : ProgressNotifier, optional
            Progress notification wrapper. Default uses tqdm.
        
        Raises
        ------
        ValueError
            If prerequisites are not met or if reference_frame is invalid.
        """
        # Validate prerequisites
        if 'pos_vectors' not in self.data:
            raise ValueError('Sarcomere vectors not analyzed. Run analyze_sarcomere_vectors first.')
        if 'domains' not in self.data:
            raise ValueError('Sarcomere domains not analyzed. Run analyze_sarcomere_domains first.')
        if self.metadata.frametime is None:
            raise ValueError('Frame time not defined in metadata. Required for motion analysis.')
        
        # Get frames where domains were analyzed
        domain_frames = self.data.get('params.analyze_sarcomere_domains.frames', [])
        if reference_frame not in domain_frames:
            raise ValueError(
                f'Reference frame {reference_frame} was not analyzed for domains. '
                f'Available frames: {domain_frames}'
            )
        
        # Get or regenerate domain mask for reference frame
        if 'domain_mask' in self.data and self.data['domain_mask'][reference_frame] is not None:
            domain_mask_ref = self.data['domain_mask'][reference_frame]
            # Convert sparse matrix to dense if needed
            if hasattr(domain_mask_ref, 'toarray'):
                domain_mask_ref = domain_mask_ref.toarray()
        else:
            # Regenerate domain mask from stored domain data
            logger.info(f'Regenerating domain mask for reference frame {reference_frame}...')
            domains_ref = self.data['domains'][reference_frame]
            pos_vectors_ref = np.asarray(self.data['pos_vectors'][reference_frame])
            sarcomere_orientation_ref = np.asarray(self.data['sarcomere_orientation_vectors'][reference_frame])
            sarcomere_length_ref = np.asarray(self.data['sarcomere_length_vectors'][reference_frame])
            
            dilation_radius = self.data.get('params.analyze_sarcomere_domains.dilation_radius', 0.3)
            area_min = self.data.get('params.analyze_sarcomere_domains.area_min', 20.0)
            
            domain_mask_ref, *_ = domain_clustering.analyze_domains(
                domains_ref, pos_vectors_ref, sarcomere_orientation_ref, sarcomere_length_ref,
                size=self.metadata.size, pixelsize=self.metadata.pixelsize,
                dilation_radius=dilation_radius, area_min=area_min
            )
        
        # Get number of domains
        n_domains = int(self.data['n_domains'][reference_frame])
        if n_domains == 0:
            logger.warning('No domains found in reference frame. Cannot analyze domain motion.')
            return
        
        logger.info(f'Analyzing domain motion for {n_domains} domains...')
        
        # Get all frames where vectors were analyzed
        vector_frames = self.data.get('params.analyze_sarcomere_vectors.frames', list(range(self.metadata.n_stack)))
        
        # Collect sarcomere vectors for all frames
        pos_vectors_all = [
            np.asarray(self.data['pos_vectors'][t]) if self.data['pos_vectors'][t] is not None else np.array([])
            for t in vector_frames
        ]
        sarcomere_length_vectors_all = [
            np.asarray(self.data['sarcomere_length_vectors'][t]) if self.data['sarcomere_length_vectors'][t] is not None else np.array([])
            for t in vector_frames
        ]
        
        # Compute per-domain time-series
        logger.info('Computing per-domain sarcomere length time-series...')
        timeseries_results = domain_motion.compute_domain_timeseries(
            pos_vectors_all=pos_vectors_all,
            sarcomere_length_vectors_all=sarcomere_length_vectors_all,
            domain_mask=domain_mask_ref,
            pixelsize=self.metadata.pixelsize,
            n_domains=n_domains,
        )
        
        # Select model for ContractionNet
        if model is None or model == 'default':
            model = os.path.join(self.model_dir, 'model_ContractionNet.pt')
        
        # Detect contractions using ContractionNet
        logger.info('Detecting contraction cycles using ContractionNet...')
        contraction_results = domain_motion.detect_domain_contractions(
            domain_slen_timeseries=timeseries_results['domain_slen_timeseries'],
            frametime=self.metadata.frametime,
            model_path=model,
            threshold=threshold,
            contr_time_min=contr_time_min,
            merge_time_max=merge_time_max,
            buffer_frames=buffer_frames,
            min_valid_frames=min_valid_frames,
        )
        
        # Analyze contraction parameters
        logger.info('Analyzing contraction parameters...')
        param_results = domain_motion.analyze_domain_contraction_parameters(
            domain_slen_timeseries=timeseries_results['domain_slen_timeseries'],
            domain_labels_contr=contraction_results['domain_labels_contr'],
            domain_n_contr=contraction_results['domain_n_contr'],
            frametime=self.metadata.frametime,
            filter_params=filter_params,
        )
        
        # Store results in data dictionary
        domain_motion_data = {
            # Time-series data
            'domain_slen_timeseries': timeseries_results['domain_slen_timeseries'],
            'domain_slen_median_timeseries': timeseries_results['domain_slen_median_timeseries'],
            'domain_slen_std_timeseries': timeseries_results['domain_slen_std_timeseries'],
            'domain_slen_q25_timeseries': timeseries_results['domain_slen_q25_timeseries'],
            'domain_slen_q75_timeseries': timeseries_results['domain_slen_q75_timeseries'],
            'domain_n_vectors_timeseries': timeseries_results['domain_n_vectors_timeseries'],
            # Contraction detection
            'domain_contr': contraction_results['domain_contr'],
            'domain_n_contr': contraction_results['domain_n_contr'],
            'domain_labels_contr': contraction_results['domain_labels_contr'],
            'domain_beating_rate': contraction_results['domain_beating_rate'],
            'domain_beating_rate_variability': contraction_results['domain_beating_rate_variability'],
            # Contraction parameters
            'domain_equ': param_results['domain_equ'],
            'domain_contr_max': param_results['domain_contr_max'],
            'domain_elong_max': param_results['domain_elong_max'],
            'domain_vel_contr_max': param_results['domain_vel_contr_max'],
            'domain_vel_elong_max': param_results['domain_vel_elong_max'],
            'domain_time_to_peak': param_results['domain_time_to_peak'],
            'domain_time_to_relax': param_results['domain_time_to_relax'],
            'domain_time_contr': param_results['domain_time_contr'],
            # Parameters
            'params.analyze_domain_motion.reference_frame': reference_frame,
            'params.analyze_domain_motion.model': model,
            'params.analyze_domain_motion.threshold': threshold,
            'params.analyze_domain_motion.contr_time_min': contr_time_min,
            'params.analyze_domain_motion.merge_time_max': merge_time_max,
            'params.analyze_domain_motion.buffer_frames': buffer_frames,
            'params.analyze_domain_motion.min_valid_frames': min_valid_frames,
            'params.analyze_domain_motion.filter_params': filter_params,
        }
        
        self.data.update(domain_motion_data)
        logger.info(f'Domain motion analysis complete. Analyzed {n_domains} domains.')

        if self.auto_save:
            self.store_structure_data()

    def track_sarcomere_vectors(
        self,
        frames: Union[str, int, List[int], np.ndarray] = 'all',
        threshold_mbands: float = 0.25,
        threshold_zbands: float = 0.5,
        dt_clip: float = 20.0,
        max_disp_along_px: float = 15.0,
        max_disp_perp_px: float = 2.0,
        ori_tol_deg: float = 45.0,
        memory: int = 5,
        min_track_length: int = 5,
        max_gap_interpolation: int = 5,
        merge_tracks: bool = True,
        merge_max_disp_along_px: float = 25.0,
        merge_max_disp_perp_px: float = 4.0,
        merge_ori_tol_deg: float = 45.0,
        merge_slen_tol_um: float = 0.30,
        slen_lims: Tuple[float, float] = (1.0, 3.0),
        compute_motion_field: bool = True,
        store_flow_fields: bool = False,
    ) -> None:
        """2D full-field sarcomere-vector tracking + motion field.

        Complements :meth:`Motion.track_z_bands` (LOI / 1D). Each sarcomere
        detection in the first analyzed frame seeds a **query point**; every
        subsequent frame the query point is flow-advected (Lagrangian
        prediction) and then *snapped* to the nearest sarcomere detection
        consistent with its prediction under anisotropic (along-/perpendicular-
        to-sarcomere) and orientation gates. No M-band identity is tracked;
        anti-convergence is guaranteed by snapping to discrete detections.

        Prerequisites: :meth:`analyze_sarcomere_vectors` must have been run.

        Parameters
        ----------
        frames
            Frame selection (``'all'`` or a list/int).
        threshold_mbands, threshold_zbands
            Thresholds used to binarize the probability masks before DT
            computation.
        dt_clip
            Distance-transform clipping distance (pixels).
        max_disp_along_px, max_disp_perp_px
            Anisotropic tolerance for the snap gate, in pixels. Motion along
            the sarcomere axis (contraction direction) can be large; motion
            perpendicular should be small.
        ori_tol_deg
            Orientation tolerance for the snap gate, in degrees. Orientations
            are compared modulo π.
        memory
            Frames a query point may go without snapping before it is closed.
        min_track_length
            Minimum number of *actual snaps* required to keep a track.
        max_gap_interpolation
            Maximum absence span (in frames) that may be bridged by the
            post-loop merge step (and linearly interpolated in the gap).
        merge_tracks
            If True (default), run a final pass that stitches respawned
            trajectory fragments back to their parent track when the flow-
            predicted tail of one matches the head of another. Reduces
            fragmentation caused by transient U-Net misses without per-
            dataset tuning.
        merge_max_disp_along_px, merge_max_disp_perp_px
            Anisotropic position-residual tolerances for the merge step,
            in pixels. Both scale linearly with the bridged gap (random-walk
            position uncertainty). Defaults are intentionally looser than
            the per-frame snap gates: a multi-frame bridge has a larger
            uncertainty budget than a 1-frame snap.
        merge_ori_tol_deg
            Orientation tolerance for merging, in degrees (axial / mod π).
        merge_slen_tol_um
            Sarcomere-length continuity tolerance for merging, in micrometers.
            Two fragments are only stitched if their seam-frame slens differ
            by at most this much. Strongest single guard against same-
            myofibril neighbour swaps.
        slen_lims
            Physiologically valid sarcomere-length range (μm). A merge is
            rejected if either seam slen is finite and falls outside the
            range. Same semantics as ``slen_lims`` in the LOI motion
            analysis (see :class:`~sarcasm.motion.Motion`).
        compute_motion_field
            Sample flow at every detection position and decompose into
            along-sarcomere / perpendicular components (independent of tracking
            success).
        store_flow_fields
            If True, also return int16-quantized dense flow fields. Large
            memory cost; off by default.
        """
        if 'pos_vectors_px' not in self.data:
            raise ValueError('Sarcomere vectors not analyzed. Run analyze_sarcomere_vectors first.')

        # Frame selection matches the pattern in analyze_sarcomere_vectors.
        _detected_frames = self.data.get('params.detect_sarcomeres.frames', 'all')
        if ((isinstance(frames, str) and frames == 'all')
                or (self.metadata.n_stack == 1 and frames == 0)
                or (_detected_frames != 'all' and len(_detected_frames) == 1)):
            list_frames = list(range(self.metadata.n_stack))
            z_bands = tifffile.imread(self.file_zbands)
            mbands = tifffile.imread(self.file_mbands)
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, (list, np.ndarray)):
            z_bands = tifffile.imread(self.file_zbands, key=frames)
            mbands = tifffile.imread(self.file_mbands, key=frames)
            if np.issubdtype(type(frames), np.integer):
                list_frames = [int(frames)]
            else:
                list_frames = [int(f) for f in frames]
        else:
            raise ValueError('frames argument not valid')
        if z_bands.ndim == 2:
            z_bands = np.expand_dims(z_bands, axis=0)
        if mbands.ndim == 2:
            mbands = np.expand_dims(mbands, axis=0)

        if len(list_frames) < 2:
            raise ValueError('Need at least 2 frames for tracking.')

        # Collect the per-frame vector data that analyze_sarcomere_vectors stored.
        pos_px_all = [
            np.asarray(self.data['pos_vectors_px'][t], dtype=np.int32)
            if self.data['pos_vectors_px'][t] is not None
            else np.zeros((0, 2), np.int32)
            for t in list_frames
        ]
        mid_all = [
            np.asarray(self.data['midline_id_vectors'][t], dtype=np.int64)
            if self.data['midline_id_vectors'][t] is not None
            else np.zeros(0, np.int64)
            for t in list_frames
        ]
        slen_all = [
            np.asarray(self.data['sarcomere_length_vectors'][t], dtype=np.float32)
            if self.data['sarcomere_length_vectors'][t] is not None
            else np.zeros(0, np.float32)
            for t in list_frames
        ]
        ori_all = [
            np.asarray(self.data['sarcomere_orientation_vectors'][t], dtype=np.float32)
            if self.data['sarcomere_orientation_vectors'][t] is not None
            else np.zeros(0, np.float32)
            for t in list_frames
        ]

        logger.info(f'Tracking {len(list_frames)} frames...')
        out = sarcomere_tracking.track_sarcomere_vectors(
            z_bands, mbands,
            pos_px_all, mid_all, slen_all, ori_all,
            pixelsize=self.metadata.pixelsize,
            frametime=self.metadata.frametime,
            threshold_mbands=threshold_mbands,
            threshold_zbands=threshold_zbands,
            dt_clip=dt_clip,
            max_disp_along_px=max_disp_along_px,
            max_disp_perp_px=max_disp_perp_px,
            ori_tol_deg=ori_tol_deg,
            memory=memory,
            min_track_length=min_track_length,
            max_gap_interpolation=max_gap_interpolation,
            merge_tracks=merge_tracks,
            merge_max_disp_along_px=merge_max_disp_along_px,
            merge_max_disp_perp_px=merge_max_disp_perp_px,
            merge_ori_tol_deg=merge_ori_tol_deg,
            merge_slen_tol_um=merge_slen_tol_um,
            slen_lims=slen_lims,
            compute_motion_field=compute_motion_field,
            store_flow_fields=store_flow_fields,
        )

        tracking_data = {
            'n_tracks': out['n_tracks'],
            'track_ids': out['track_ids'],
            'track_start_frame': out['track_start_frame'],
            'track_lengths': out['track_lengths'],
            'tracks_positions_um': out['tracks_positions_um'],
            'tracks_positions_px': out['tracks_positions_px'],
            'tracks_slen': out['tracks_slen'],
            'tracks_orientations': out['tracks_orientations'],
            'tracks_snapped': out['tracks_snapped'],
            'n_merges': out['n_merges'],
            'params.track_sarcomere_vectors.frames': list_frames,
            'params.track_sarcomere_vectors.threshold_mbands': threshold_mbands,
            'params.track_sarcomere_vectors.threshold_zbands': threshold_zbands,
            'params.track_sarcomere_vectors.dt_clip': dt_clip,
            'params.track_sarcomere_vectors.max_disp_along_px': max_disp_along_px,
            'params.track_sarcomere_vectors.max_disp_perp_px': max_disp_perp_px,
            'params.track_sarcomere_vectors.ori_tol_deg': ori_tol_deg,
            'params.track_sarcomere_vectors.memory': memory,
            'params.track_sarcomere_vectors.min_track_length': min_track_length,
            'params.track_sarcomere_vectors.max_gap_interpolation': max_gap_interpolation,
            'params.track_sarcomere_vectors.merge_tracks': merge_tracks,
            'params.track_sarcomere_vectors.merge_max_disp_along_px': merge_max_disp_along_px,
            'params.track_sarcomere_vectors.merge_max_disp_perp_px': merge_max_disp_perp_px,
            'params.track_sarcomere_vectors.merge_ori_tol_deg': merge_ori_tol_deg,
            'params.track_sarcomere_vectors.merge_slen_tol_um': merge_slen_tol_um,
            'params.track_sarcomere_vectors.slen_lims': list(slen_lims),
            'params.track_sarcomere_vectors.compute_motion_field': compute_motion_field,
        }
        if compute_motion_field:
            tracking_data.update({
                'flow_at_vectors': out['flow_at_vectors'],
                'displacement_magnitude': out['displacement_magnitude'],
                'displacement_along_sarcomere': out['displacement_along_sarcomere'],
                'displacement_perpendicular': out['displacement_perpendicular'],
                'velocity_magnitude': out['velocity_magnitude'],
            })
        if store_flow_fields:
            tracking_data.update({
                'flow_fields_int16': out['flow_fields_int16'],
                'flow_fields_scales': out['flow_fields_scales'],
            })

        self.data.update(tracking_data)
        logger.info(f'Tracked {out["n_tracks"]} sarcomere query points over {len(list_frames)} frames.')
        if self.auto_save:
            self.store_structure_data()

    def compute_motion_field(
        self,
        frames: Union[str, int, List[int], np.ndarray] = 'all',
        threshold: float = 0.5,
        dt_clip: float = 20.0,
    ) -> None:
        """Compute optical flow + per-vector motion field without tracking.

        Useful for quick motion assessment, strain maps, or as input to
        downstream contraction detection. Prerequisites:
        :meth:`analyze_sarcomere_vectors`.
        """
        if 'pos_vectors_px' not in self.data:
            raise ValueError('Sarcomere vectors not analyzed. Run analyze_sarcomere_vectors first.')

        _detected_frames = self.data.get('params.detect_sarcomeres.frames', 'all')
        if ((isinstance(frames, str) and frames == 'all')
                or (self.metadata.n_stack == 1 and frames == 0)
                or (_detected_frames != 'all' and len(_detected_frames) == 1)):
            list_frames = list(range(self.metadata.n_stack))
            z_bands = tifffile.imread(self.file_zbands)
            mbands = tifffile.imread(self.file_mbands)
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, (list, np.ndarray)):
            z_bands = tifffile.imread(self.file_zbands, key=frames)
            mbands = tifffile.imread(self.file_mbands, key=frames)
            if np.issubdtype(type(frames), np.integer):
                list_frames = [int(frames)]
            else:
                list_frames = [int(f) for f in frames]
        else:
            raise ValueError('frames argument not valid')
        if z_bands.ndim == 2:
            z_bands = np.expand_dims(z_bands, axis=0)
        if mbands.ndim == 2:
            mbands = np.expand_dims(mbands, axis=0)
        if len(list_frames) < 2:
            raise ValueError('Need at least 2 frames to compute motion field.')

        pos_px_all = [
            np.asarray(self.data['pos_vectors_px'][t], dtype=np.float32)
            if self.data['pos_vectors_px'][t] is not None
            else np.zeros((0, 2), np.float32)
            for t in list_frames
        ]
        ori_all = [
            np.asarray(self.data['sarcomere_orientation_vectors'][t], dtype=np.float32)
            if self.data['sarcomere_orientation_vectors'][t] is not None
            else np.zeros(0, np.float32)
            for t in list_frames
        ]

        out = sarcomere_tracking.compute_motion_field(
            z_bands, mbands, pos_px_all, ori_all,
            pixelsize=self.metadata.pixelsize,
            frametime=self.metadata.frametime,
            threshold=threshold, dt_clip=dt_clip,
        )

        self.data.update({
            'flow_at_vectors': out['flow_at_vectors'],
            'displacement_magnitude': out['displacement_magnitude'],
            'displacement_along_sarcomere': out['displacement_along_sarcomere'],
            'displacement_perpendicular': out['displacement_perpendicular'],
            'velocity_magnitude': out['velocity_magnitude'],
            'params.compute_motion_field.frames': list_frames,
            'params.compute_motion_field.threshold': threshold,
            'params.compute_motion_field.dt_clip': dt_clip,
        })
        if self.auto_save:
            self.store_structure_data()

    def _grow_lois(self, frame: int = 0, ratio_seeds: float = 0.1, persistence: int = 2,
                   threshold_distance: float = 0.3, random_seed: Union[None, int] = None) -> None:
        """
        Find LOIs (lines of interest) using a line growth algorithm. The parameters **lims can be used to filter LOIs.

        Parameters
        ----------
        frame : int, optional
            Frame to select frame. Selects i-th frame of frames specified in sarcomere vector analysis. Defaults to 0.
        ratio_seeds : float, optional
            Ratio of sarcomere vectors to take as seeds for line growth. Default 0.1.
        persistence : int, optional
            Persistence of line (average vector length and orientation for prior estimation). Defaults to 2.
        threshold_distance : float, optional
            Maximal distance for nearest neighbor estimation. Defaults to 0.5.
        random_seed : int, optional
            Random seed for reproducibility. Defaults to None.
        """
        # select midline point data at frame
        (pos_vectors, sarcomere_length_vectors,
         sarcomere_orientation_vectors, midline_length_vectors) = self.data['pos_vectors_px'][frame], \
            self.data['sarcomere_length_vectors'][frame], \
            self.data['sarcomere_orientation_vectors'][frame], \
            self.data['midline_length_vectors'][frame]
        # Delegate to myofibril_analysis module
        loi_data = myofibril_analysis.line_growth(points_t=pos_vectors, sarcomere_length_vectors_t=sarcomere_length_vectors,
                                    sarcomere_orientation_vectors_t=sarcomere_orientation_vectors,
                                    midline_length_vectors_t=midline_length_vectors,
                                    pixelsize=self.metadata.pixelsize,
                                    ratio_seeds=ratio_seeds, persistence=persistence,
                                    threshold_distance=threshold_distance, random_seed=random_seed)
        self.data['loi_data'] = loi_data
        lois_vectors = [self.data['pos_vectors_px'][frame][loi_i] for loi_i in self.data['loi_data']['lines']]
        self.data['loi_data']['lines_vectors'] = lois_vectors
        if self.auto_save:
            self.store_structure_data()

    def _filter_lois(self, number_lims: Tuple[int, int] = (10, 100), length_lims: Tuple[float, float] = (0, 200),
                     sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
                     sarcomere_std_length_lims: Tuple[float, float] = (0, 1),
                     midline_mean_length_lims: Tuple[float, float] = (0, 50),
                     midline_std_length_lims: Tuple[float, float] = (0, 50),
                     midline_min_length_lims: Tuple[float, float] = (0, 50),
                     ) -> None:
        """
        Filters Lines of Interest (LOIs) based on various geometric and morphological criteria.

        Parameters
        ----------
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
        """
        # Delegate to loi_detection module
        (filtered_lois, filtered_lois_vectors,
         filtered_features) = loi_detection.filter_lois(
            lois=self.data['loi_data']['lines'],
            loi_features=self.data['loi_data']['line_features'],
            lois_vectors=self.data['loi_data']['lines_vectors'],
            number_lims=number_lims,
            length_lims=length_lims,
            sarcomere_mean_length_lims=sarcomere_mean_length_lims,
            sarcomere_std_length_lims=sarcomere_std_length_lims,
            midline_mean_length_lims=midline_mean_length_lims,
            midline_std_length_lims=midline_std_length_lims,
            midline_min_length_lims=midline_min_length_lims
        )

        self.data['loi_data']['lines'] = filtered_lois
        self.data['loi_data']['lines_vectors'] = filtered_lois_vectors
        self.data['loi_data']['line_features'] = filtered_features

    def _hausdorff_distance_lois(self, symmetry_mode: str = 'max') -> None:
        """
        Compute Hausdorff distances between all good LOIs.

        Parameters
        ----------
        symmetry_mode : str, optional
            Choose 'min' or 'max', whether min/max(H(loi_i, loi_j), H(loi_j, loi_i)). Defaults to 'max'.
        """
        # Delegate to loi_detection module
        hausdorff_dist_matrix = loi_detection.hausdorff_distance_lois(
            lines_vectors=self.data['loi_data']['lines_vectors'],
            symmetry_mode=symmetry_mode
        )

        self.data['loi_data']['hausdorff_dist_matrix'] = hausdorff_dist_matrix
        if self.auto_save:
            self.store_structure_data()

    def _cluster_lois(self, distance_threshold_lois: float = 40, linkage: str = 'single') -> None:
        """
        Agglomerative clustering of good LOIs using predefined Hausdorff distance matrix using scikit-learn.

        Parameters
        ----------
        distance_threshold_lois : float, optional
            The linkage distance threshold above which clusters will not be merged. Defaults to 40.
        linkage : {'complete', 'average', 'single'}, optional
            Which linkage criterion to use. The linkage criterion determines which distance to use between sets of
            observations. The algorithm will merge the pairs of clusters that minimize this criterion.
            - 'average' uses the average of the distances of each observation of the two sets.
            - 'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.
            - 'single' uses the minimum of the distances between all observations of the two sets.
            Defaults to 'single'.
        """
        # Delegate to loi_detection module
        cluster_labels, n_clusters = loi_detection.cluster_lois(
            hausdorff_dist_matrix=self.data['loi_data']['hausdorff_dist_matrix'],
            distance_threshold=distance_threshold_lois,
            linkage=linkage
        )

        self.data['loi_data']['line_cluster'] = cluster_labels
        self.data['loi_data']['n_lines_clusters'] = n_clusters
        if self.auto_save:
            self.store_structure_data()

    def _fit_straight_line(self, add_length=1, n_lois=None):
        """Fit linear lines to cluster points

        Parameters
        ----------
        add_length : float
            Elongate line at end with add_length (in length unit)
        n_lois : int
            If int, only n longest LOIs are saved. If None, all are saved.
        """
        # Delegate to loi_detection module
        loi_lines, len_loi_lines = loi_detection.fit_straight_line_to_clusters(
            lines_vectors=self.data['loi_data']['lines_vectors'],
            cluster_labels=self.data['loi_data']['line_cluster'],
            n_clusters=self.data['loi_data']['n_lines_clusters'],
            pixelsize=self.metadata.pixelsize,
            add_length=add_length,
            n_lois=n_lois
        )

        self.data['loi_data']['loi_lines'] = np.asarray(loi_lines, dtype=object)
        self.data['loi_data']['len_loi_lines'] = np.asarray(len_loi_lines)
        if self.auto_save:
            self.store_structure_data()

    def _longest_in_cluster(self, n_lois, frame):
        # Delegate to loi_detection module
        loi_lines, len_loi_lines = loi_detection.select_longest_in_cluster(
            lines=self.data['loi_data']['lines'],
            pos_vectors=self.data['pos_vectors_px'][frame],
            cluster_labels=self.data['loi_data']['line_cluster'],
            n_clusters=self.data['loi_data']['n_lines_clusters'],
            n_lois=n_lois
        )

        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = len_loi_lines
        if self.auto_save:
            self.store_structure_data()

    def _random_from_cluster(self, n_lois, frame):
        # Delegate to loi_detection module
        loi_lines, len_loi_lines = loi_detection.select_random_from_cluster(
            lines=self.data['loi_data']['lines'],
            pos_vectors=self.data['pos_vectors_px'][frame],
            cluster_labels=self.data['loi_data']['line_cluster'],
            n_clusters=self.data['loi_data']['n_lines_clusters'],
            n_lois=n_lois
        )

        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = len_loi_lines
        if self.auto_save:
            self.store_structure_data()

    def _random_lois(self, n_lois, frame):
        # Delegate to loi_detection module
        loi_lines, len_loi_lines = loi_detection.select_random_lois(
            lines=self.data['loi_data']['lines'],
            pos_vectors=self.data['pos_vectors_px'][frame],
            n_lois=n_lois
        )

        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = len_loi_lines
        if self.auto_save:
            self.store_structure_data()

    def create_loi_data(self, line: np.ndarray, linewidth: float = 0.65, order: int = 0,
                        export_raw: bool = False) -> None:
        """
        Extract intensity kymograph along LOI and create LOI file from line.

        Parameters
        ----------
        line : np.ndarray
            Line start and end coordinates ((start_x, start_y), (end_x, end_y))
            or list of segments [(x0, y0), (x1, y1), (x2, y2), ...]
        linewidth : float, optional
            Width of the scan in µm, perpendicular to the line. Defaults to 0.65.
        order : int, optional
            The order of the spline interpolation, default is 0 if image.dtype is bool and 1 otherwise.
            The order has to be in the range 0-5. See `skimage.transform.warp` for details. Defaults to 0.
        export_raw : bool, optional
            If True, intensity kymograph along LOI from raw microscopy image is additionally stored. Defaults to False.
        """
        if self.metadata.pixelsize is None:
            raise ValueError("Pixel size is not available. Please provide pixelsize during initialization.")
        if os.path.exists(self.file_zbands_fast_movie):
            file_z_bands = self.file_zbands_fast_movie
        else:
            file_z_bands = self.file_zbands
        imgs_sarcomeres = tifffile.imread(file_z_bands)
        profiles = kymograph.kymograph_movie(imgs_sarcomeres, line, order=order,
                                        linewidth=int(linewidth / self.metadata.pixelsize))
        profiles = np.asarray(profiles)
        if export_raw:
            imgs_raw = self.image
            profiles_raw = kymograph.kymograph_movie(imgs_raw, line, order=order,
                                                linewidth=int(linewidth / self.metadata.pixelsize))
        else:
            profiles_raw = None

        # length of line
        def __calculate_segmented_line_length(line):
            # Ensure line is a proper numeric numpy array
            line = np.asarray(line, dtype=np.float64)
            diffs = np.diff(line, axis=0)
            lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            return np.sum(lengths)

        length = __calculate_segmented_line_length(line) * self.metadata.pixelsize

        loi_data = {'profiles': profiles, 'profiles_raw': profiles_raw,
                    'line': line, 'linewidth': linewidth, 'length': length}
        for key, value in loi_data.items():
            loi_data[key] = np.asarray(value)
        save_name = os.path.join(self.base_dir,
                                 f'{line[0][0]}_{line[0][1]}_{line[-1][0]}_{line[-1][1]}_{linewidth}_loi.json')
        IOUtils.json_serialize(loi_data, save_name)

    def detect_lois(self, frame: int = 0, 
                    n_lois: int = 4, 
                    ratio_seeds: float = 0.1, 
                    persistence: int = 4,
                    threshold_distance: float = 0.5,
                    mode: str = 'longest_in_cluster', 
                    random_seed: Optional[int] = None,
                    number_lims: Tuple[int, int] = (10, 50), 
                    length_lims: Tuple[float, float] = (0, 200),
                    sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
                    sarcomere_std_length_lims: Tuple[float, float] = (0, 1),
                    midline_mean_length_lims: Tuple[float, float] = (0, 50),
                    midline_std_length_lims: Tuple[float, float] = (0, 50),
                    midline_min_length_lims: Tuple[float, float] = (0, 50), 
                    distance_threshold_lois: float = 40,
                    linkage: str = 'single', 
                    linewidth: float = 0.65, 
                    order: int = 0, 
                    export_raw: bool = False
                    ) -> None:
        """
        Detects Regions of Interest (LOIs) for tracking sarcomere Z-band motion and creates kymographs.

        This method integrates several steps: growing LOIs based on seed vectors, filtering LOIs based on
        specified criteria, clustering LOIs, fitting lines to LOI clusters, and extracting intensity profiles
        to generate kymographs.

        Parameters
        ----------
        frame : int
            The index of the frame to select for analysis.
        n_lois : int
            Number of LOIs.
        ratio_seeds : float
            Ratio of sarcomere vectors to take as seed vectors for initiating LOI growth.
        persistence : int
            Persistence parameter influencing line growth direction and termination.
        threshold_distance : float
            Maximum distance for nearest neighbor estimation during line growth.
        mode : str
            Mode for selecting LOIs from identified clusters.
            - 'fit_straight_line' fits a straight line to all midline points in the cluster.
            - 'longest_in_cluster' selects the longest line of each cluster, also allowing curved LOIs.
            - 'random_from_cluster' selects a random line from each cluster, also allowing curved LOIs.
            - 'random_line' selects a set of random lines that fulfil the filtering criteria.
        random_seed : int, optional
            Random seed for selection of random starting vectors for line growth algorithm, for reproducible outcomes.
            If None, no random seed is set, and outcomes in every run will differ.
        number_lims : tuple of int
            Limits for the number of sarcomeres within an LOI (min, max).
        length_lims : tuple of float
            Length limits for LOIs (in µm) (min, max).
        sarcomere_mean_length_lims : tuple of float
            Limits for the mean length of sarcomeres within an LOI (min, max).
        sarcomere_std_length_lims : tuple of float
            Limits for the standard deviation of sarcomere lengths within an LOI (min, max).
        midline_mean_length_lims : tuple of float
            Limits for the mean length of the midline of vectors in LOI (min, max).
        midline_std_length_lims : tuple of float
            Limits for the standard deviation of the midline length of vectors in LOI (min, max).
        midline_min_length_lims : tuple of float
            Limits for the minimum length of the midline of vectors in LOI (min, max).
        distance_threshold_lois : float
            Distance threshold for clustering LOIs. Clusters will not be merged above this threshold.
        linkage : str
            Linkage criterion for clustering ('complete', 'average', 'single').
        linewidth : float
            Width of the scan line (in µm), perpendicular to the LOIs.
        order : int
            Order of spline interpolation for transforming LOIs (range 0-5).
        export_raw : bool
            If True, exports raw intensity kymographs along LOIs.

        Returns
        -------
        None
        """
        if 'pos_vectors' not in self.data:
            raise ValueError('Sarcomere length and orientation not yet analyzed. Run analyze_sarcomere_vectors first.')

        if self.metadata.n_stack == 1:
            raise ValueError('LOI detection not possible in single images. '
                             'Sarcomere motion tracking is only possible in high-speed movies; (t, x, y) stacks.')

        # Grow LOIs based on seed vectors and specified parameters
        self._grow_lois(frame=frame, ratio_seeds=ratio_seeds, random_seed=random_seed, persistence=persistence,
                        threshold_distance=threshold_distance)
        # Filter LOIs based on geometric and morphological criteria
        self._filter_lois(number_lims=number_lims, length_lims=length_lims,
                          sarcomere_mean_length_lims=sarcomere_mean_length_lims,
                          sarcomere_std_length_lims=sarcomere_std_length_lims,
                          midline_mean_length_lims=midline_mean_length_lims,
                          midline_std_length_lims=midline_std_length_lims,
                          midline_min_length_lims=midline_min_length_lims)
        if mode == 'fit_straight_line' or mode == 'longest_in_cluster' or mode == 'random_from_cluster':
            # Calculate Hausdorff distance between LOIs and perform clustering
            self._hausdorff_distance_lois()
            self._cluster_lois(distance_threshold_lois=distance_threshold_lois, linkage=linkage)
            # Fit lines to LOIs clusters and select LOIs for analysis
            if mode == 'fit_straight_line':
                self._fit_straight_line(add_length=2, n_lois=n_lois)
            elif mode == 'longest_in_cluster':
                self._longest_in_cluster(n_lois=n_lois, frame=frame)
            elif mode == 'random_from_cluster':
                self._random_from_cluster(n_lois=n_lois, frame=frame)
        elif mode == 'random_line':
            self._random_lois(n_lois=n_lois, frame=frame)
        else:
            raise ValueError(f'mode {mode} not valid.')

        # extract intensity kymographs profiles and save LOI files
        for line_i in self.data['loi_data']['loi_lines']:
            self.create_loi_data(line_i, linewidth=linewidth, order=order, export_raw=export_raw)

    def delete_lois(self):
        """
        Delete all LOIs, their associated data files, and their directories.
        """
        self.data.pop('loi_data', None)

        loi_files = glob.glob(os.path.join(self.base_dir, '*loi.json'))
        for loi_file in loi_files:
            try:
                # Remove the LOI file
                os.remove(loi_file)

                # Remove the associated data file
                data_file = os.path.join(self.data_dir,
                                         f"{os.path.splitext(os.path.basename(loi_file))[0]}_data.json")
                if os.path.exists(data_file):
                    os.remove(data_file)

                # Remove the directory and its contents
                directory = loi_file[:-len('_loi.json')] + '/'
                if os.path.exists(directory):
                    shutil.rmtree(directory)

            except Exception as e:
                logger.debug(f"Error deleting LOI directory: {e}. Continuing anyway.")

    def full_analysis_structure(self, frames='all'):
        """
        Analyze sarcomere structure with default parameters at specified frames

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}
            frames for analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames).
        """
        self.auto_save = False
        self.analyze_cell_mask()
        self.analyze_z_bands(frames=frames)
        self.analyze_sarcomere_vectors(frames=frames)
        self.analyze_myofibrils(frames=frames)
        self.analyze_sarcomere_domains(frames=frames)
        if not self.auto_save:
            self.store_structure_data()
            self.auto_save = True
