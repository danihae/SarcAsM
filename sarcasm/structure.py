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
import os
import random
import shutil
import warnings
from collections import deque
from multiprocessing import Pool
from typing import Optional, Tuple, Union, List, Literal, Any, Dict
os.environ["KMP_WARNINGS"] = "False"
warnings.filterwarnings("ignore", message=".*omp_set_nested.*")


import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import tifffile
import torch
from bio_image_unet import multi_output_unet3d as unet3d
from bio_image_unet.multi_output_unet.multi_output_nested_unet import MultiOutputNestedUNet_3Levels
from bio_image_unet.multi_output_unet.predict import Predict as Predict_UNet
from bio_image_unet.progress import ProgressNotifier
from joblib import Parallel, delayed
from scipy import ndimage, stats, sparse
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff, squareform, pdist
from skimage import segmentation, morphology, measure
from skimage.draw import line
from skimage.measure import label, regionprops_table
from skimage.morphology import skeletonize, disk, binary_dilation
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from sarcasm.core import SarcAsM
from sarcasm.ioutils import IOUtils
from sarcasm.utils import Utils


class Structure(SarcAsM):
    """
    Class to analyze sarcomere morphology.

    Attributes
    ----------
    data : dict
        A dictionary with structure data.
    """

    def __init__(self,
                 filepath: Union[str, os.PathLike],
                 restart: bool = False,
                 channel: Union[int, None, Literal['RGB']] = None,
                 auto_save: bool = True,
                 use_gui: bool = False,
                 device: Union[torch.device, Literal['auto']] = 'auto',
                 **info: Dict[str, Any]
                 ) -> None:
        """
        Initialize the structural analysis of a tiff file.

        Parameters
        ----------
        filepath : str | os.PathLike
            Path to the TIFF file for analysis.
        restart : bool, optional
            If True, deletes existing analysis and starts fresh (default: False).
        channel : int, None or Literal['RGB'], optional
            Specifies the channel with sarcomeres in multicolor stacks (default: None).
        auto_save : bool, optional
            Automatically saves analysis results when True (default: True).
        use_gui : bool, optional
            Indicates GUI mode operation (default: False).
        device : Union[torch.device, Literal['auto']], optional
            Device for PyTorch computations. 'auto' selects CUDA/MPS if available (default: 'auto').
        **info : Any
            Additional metadata as keyword arguments (e.g. cell_line='wt').
        """
        # init super SarcAsM object
        super().__init__(filepath=filepath, restart=restart, channel=channel, auto_save=auto_save, use_gui=use_gui,
                         device=device, **info)


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
        except:
            if os.path.exists(self.__get_structure_data_file(is_temp_file=True)):
                self.data = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=True))

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
                self.data[new_key] = list(range(self.metadata['frames']))

        if self.data is None:
            raise Exception('Loading of structure failed')

    def get_list_lois(self):
        """Returns list of LOIs"""
        return Utils.get_lois_of_file(self.filepath)

    def detect_sarcomeres(self, frames: Union[str, int, List[int], np.ndarray] = 'all',
                          model_path: str = None, max_patch_size: Tuple[int, int] = (1024, 1024),
                          normalization_mode: str = 'all', clip_thres: Tuple[float, float] = (0., 99.98),
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
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in GUI. Default is ProgressNotifier.progress_notifier_tqdm().

        Returns
        -------
        None
        """
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

        print('\nPredicting sarcomeres ...')
        if model_path is None or model_path == 'generalist':
            model_path = os.path.join(self.model_dir, 'model_sarcomeres_generalist.pt')
        _ = Predict_UNet(images, model_params=model_path, result_path=self.base_dir,
                         max_patch_size=max_patch_size, normalization_mode=normalization_mode,
                         network=MultiOutputNestedUNet_3Levels,
                         clip_threshold=clip_thres, device=self.device,
                         progress_notifier=progress_notifier)
        del _
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _dict = {'params.detect_sarcomeres.frames': list_frames, 'params.detect_sarcomeres.model': model_path,
                 'params.detect_sarcomeres.normalization_mode': normalization_mode,
                 'params.detect_sarcomeres.clip_threshold': clip_thres}
        self.data.update(_dict)
        if self.auto_save:
            self.store_structure_data()

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
        print('\nPredicting sarcomere z-bands ...')

        if model_path is None:
            model_path = os.path.join(self.model_dir, 'model_z_bands_unet3d.pt')
        if len(max_patch_size) != 3:
            raise ValueError('patch size for prediction has to be be (frames, x, y)')
        _ = unet3d.Predict(self.read_imgs(), model_params=model_path, result_path=self.base_dir,
                           max_patch_size=max_patch_size, normalization_mode=normalization_mode,
                           device=self.device, clip_threshold=clip_thres, progress_notifier=progress_notifier)
        del _
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        if (isinstance(frames, str) and frames == 'all') or (self.metadata['frames'] == 1 and frames == 0):
            cell_mask = tifffile.imread(self.file_cell_mask)
            images = self.read_imgs()
            list_frames = list(range(len(images)))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            cell_mask = tifffile.imread(self.file_cell_mask, key=frames)
            images = self.read_imgs(frames=frames)
            if np.issubdtype(type(frames), np.integer):
                list_frames = [frames]
            else:
                list_frames = list(frames)
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
            cell_area[i] = np.sum(mask_i) * self.metadata['pixelsize'] ** 2
            cell_area_ratio[i] = cell_area[i] / (img_i.shape[0] * img_i.shape[1] * self.metadata['pixelsize'] ** 2)

        _dict = {'cell_mask_area': cell_area, 'cell_mask_area_ratio': cell_area_ratio,
                 'cell_mask_intensity': cell_mask_intensity,
                 'params.analyze_cell_mask.frames': list_frames,
                 'params.analyze_cell_mask.threshold': threshold}
        self.data.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def analyze_z_bands(self, frames: Union[str, int, List[int], np.ndarray] = 'all', threshold: float = 0.5,
                        min_length: float = 0.2, median_filter_radius: float = 0.2, theta_phi_min: float = 0.4, 
                        a_min: float = 0.3, d_max: float = 4.0, d_min: float = 0.0,
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
            Smaller values are not recognized as connections (for lateral alignment and distance analysis). Defaults to 0.25.
        a_min: float, optional
            Minimal lateral alignment between z-band ends to create a lateral connection. Defaults to 0.3.
        d_max : float, optional
            Maximal distance between z-band ends (in µm). Z-band end pairs with larger distances are not connected
            (for lateral alignment and distance analysis). Defaults to 5.0.
        d_min : float, optional
            Minimal distance between z-band ends (in µm). Z-band end pairs with smaller distances are not connected.
            Defaults to 0.25.
        progress_notifier: ProgressNotifier
            Wraps progress notification, default is progress notification done with tqdm
        """
        if not os.path.exists(self.file_zbands):
            raise FileNotFoundError("Z-band mask not found. Please run detect_sarcomeres first.")
        if (isinstance(frames, str) and frames == 'all') or (self.metadata['frames'] == 1 and frames == 0):
            zbands = tifffile.imread(self.file_zbands)
            orientation_field = tifffile.imread(self.file_orientation)
            images = self.read_imgs()
            list_frames = list(range(len(images)))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            zbands = tifffile.imread(self.file_zbands, key=frames)
            orientation_field = tifffile.imread(self.file_orientation)[frames]
            images = self.read_imgs(frames=frames)
            if np.issubdtype(type(frames), np.integer):
                list_frames = [frames]
            else:
                list_frames = list(frames)
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
        none_lists = lambda: [None] * self.metadata['frames']
        z_length, z_intensity, z_straightness, z_orientation = (none_lists() for _ in range(4))
        z_lat_neighbors, z_lat_alignment, z_lat_dist = (none_lists() for _ in range(3))
        z_lat_size_groups, z_lat_length_groups, z_lat_alignment_groups = (none_lists() for _ in range(3))
        z_labels, z_ends, z_lat_links, z_lat_groups = (none_lists() for _ in range(4))

        # create empty arrays
        nan_arrays = lambda: np.full(self.metadata['frames'], np.nan)
        z_length_mean, z_length_std, z_length_max, z_length_sum, z_oop = (nan_arrays() for _ in range(5))
        z_intensity_mean, z_intensity_std = (nan_arrays() for _ in range(2))
        z_mask_area, z_mask_intensity, z_mask_area_ratio = (nan_arrays() for _ in range(3))
        z_straightness_mean, z_straightness_std = (nan_arrays() for _ in range(2))
        z_lat_neighbors_mean, z_lat_neighbors_std = (nan_arrays() for _ in range(2))
        z_lat_alignment_mean, z_lat_alignment_std = (nan_arrays() for _ in range(2))
        z_lat_dist_mean, z_lat_dist_std = (nan_arrays() for _ in range(2))
        z_lat_size_groups_mean, z_lat_size_groups_std = (nan_arrays() for _ in range(2))
        z_lat_length_groups_mean, z_lat_length_groups_std = (nan_arrays(), nan_arrays())
        z_lat_alignment_groups_mean, z_lat_alignment_groups_std = (nan_arrays() for _ in range(2))

        # iterate images
        print('\nStarting Z-band analysis...')
        for i, (frame_i, zbands_i, image_i, orientation_field_i) in enumerate(
                progress_notifier.iterator(zip(list_frames, zbands, images, orientation_field), total=n_imgs)):

            # segment z-bands
            labels_i, labels_skel_i = self.segment_z_bands(zbands_i)

            # analyze z-band features
            z_band_features = self._analyze_z_bands(zbands_i, labels_i, labels_skel_i, image_i, orientation_field_i,
                                                    pixelsize=self.metadata['pixelsize'], threshold=threshold,
                                                    min_length=min_length, median_filter_radius=median_filter_radius,
                                                    a_min=a_min, theta_phi_min=theta_phi_min,
                                                    d_max=d_max, d_min=d_min)

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
                z_mask_area_ratio[frame_i] = z_mask_area_i / (self.metadata['size'][0] * self.metadata['size'][1])

            z_labels[frame_i] = sparse.coo_matrix(labels_i)
            z_lat_links[frame_i] = z_lat_links_i
            z_ends[frame_i] = z_ends_i
            z_lat_groups[frame_i] = z_lat_groups_i

            # calculate mean and std of z-band features
            if len(z_length_i) > 0:
                z_length_mean[frame_i], z_length_std[frame_i], z_length_max[frame_i], z_length_sum[frame_i] = np.mean(
                    z_length_i), np.std(
                    z_length_i), np.max(z_length_i), np.sum(z_length_i)
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
        z_band_data = {'z_length': z_length, 'z_length_mean': z_length_mean, 'z_length_std': z_length_std,
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

    def analyze_sarcomere_vectors(self, frames: Union[str, int, List[int], np.ndarray] = 'all', threshold_mbands: float = 0.5,
                                  median_filter_radius: float = 0.25, linewidth: float = 0.2, interp_factor: int = 0,
                                  slen_lims: Tuple[float, float] = (1, 3), threshold_sarcomere_mask=0.1, backend='loky',
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
        interp_factor: int, optional
            Interpolation factor for profiles to calculate sarcomere length. Default to 4.
        slen_lims : tuple of float, optional
            Sarcomere size limits in µm (default is (1, 3) µm).
        threshold_sarcomere_mask : float
            Threshold to binarize sarcomere masks. Defaults to 0.1.
        backend : str, optional
            Backend for parallelization of profile processing. Defaults to 'loky'.
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

        _detected_frames = self.data['params.detect_sarcomeres.frames']
        if ((isinstance(frames, str) and frames == 'all') or (self.metadata['frames'] == 1 and frames == 0)
                or (_detected_frames != 'all' and len(_detected_frames) == 1)):
            list_frames = list(range(self.metadata['frames']))
            z_bands = tifffile.imread(self.file_zbands)
            mbands = tifffile.imread(self.file_mbands)
            orientation_field = tifffile.imread(self.file_orientation)
            sarcomere_mask = tifffile.imread(self.file_sarcomere_mask)
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or isinstance(frames, np.ndarray):
            z_bands = tifffile.imread(self.file_zbands, key=frames)
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

        # binarize M-bands
        mbands = mbands > threshold_mbands

        n_frames = len(z_bands)
        pixelsize = self.metadata['pixelsize']

        # create empty arrays
        none_lists = lambda: [None] * self.metadata['frames']
        nan_arrays = lambda: np.full(self.metadata['frames'], np.nan)
        (pos_vectors, pos_vectors_px, sarcomere_length_vectors,
         sarcomere_orientation_vectors) = (none_lists() for _ in range(4))
        midline_id_vectors, midline_length_vectors = (none_lists() for _ in range(2))
        sarcomere_masks = np.zeros((self.metadata['frames'], *self.metadata['size']), dtype=bool)
        (sarcomere_length_mean, sarcomere_length_std) = (nan_arrays() for _ in range(2))
        sarcomere_orientation_mean, sarcomere_orientation_std = nan_arrays(), nan_arrays()
        oop, sarcomere_area, sarcomere_area_ratio, score_thresholds = (nan_arrays() for _ in range(4))

        # iterate images
        print('\nStarting sarcomere length and orientation analysis...')
        for i, (frame_i, zbands_i, mbands_i, orientation_field_i, sarcomere_mask_i) in enumerate(
                progress_notifier.iterator(zip(list_frames, z_bands, mbands, orientation_field, sarcomere_mask),
                                           total=n_frames)):

            (
                pos_vectors_px_i, pos_vectors_i, midline_id_vectors_i, midline_length_vectors_i,
                sarcomere_length_vectors_i,
                sarcomere_orientation_vectors_i) = self.get_sarcomere_vectors(zbands_i, mbands_i,
                                                                              orientation_field_i,
                                                                              pixelsize=pixelsize,
                                                                              median_filter_radius=median_filter_radius,
                                                                              slen_lims=slen_lims,
                                                                              interp_factor=interp_factor,
                                                                              linewidth=linewidth,
                                                                              backend=backend)

            # write in list
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
            sarcomere_area[frame_i] = np.sum(sarcomere_mask_i) * self.metadata['pixelsize'] ** 2
            if 'cell_mask_area' in self.data:
                sarcomere_area_ratio[frame_i] = sarcomere_area[frame_i] / self.data['cell_mask_area'][i]

        vectors_dict = {'params.analyze_sarcomere_vectors.frames': list_frames,
                        'params.analyze_sarcomere_vectors.threshold_sarcomere_mask': threshold_sarcomere_mask,
                        'params.analyze_sarcomere_vectors.median_filter_radius': median_filter_radius,
                        'params.analyze_sarcomere_vectors.slen_lims': slen_lims,
                        'params.analyze_sarcomere_vectors.interp_factor': interp_factor,
                        'params.analyze_sarcomere_vectors.linewidth': linewidth, 'pos_vectors_px': pos_vectors_px,
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
            if (isinstance(frames, str) and frames == 'all') or (self.metadata['frames'] == 1 and frames == 0):
                frames = list(range(self.metadata['frames']))
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
            n_imgs = self.metadata['frames']
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
        none_lists = lambda: [None] * self.metadata['frames']
        nan_arrays = lambda: np.full(self.metadata['frames'], np.nan)
        length_mean, length_std, length_max = (nan_arrays() for _ in range(3))
        straightness_mean, straightness_std = (nan_arrays() for _ in range(2))
        bending_mean, bending_std = (nan_arrays() for _ in range(2))
        myof_lines, lengths, straightness, frechet_straightness, bending = (none_lists() for _ in range(5))

        # iterate frames
        print('\nStarting myofibril line analysis...')
        for i, (
                frame_i, pos_vectors_px_i, pos_vectors_i, sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                midline_length_vectors_i) in enumerate(
            progress_notifier.iterator(
                zip(list_frames, pos_vectors_px, pos_vectors, sarcomere_length_vectors, sarcomere_orientation_vectors,
                    midline_length_vectors),
                total=len(pos_vectors_px))):
            if len(np.asarray(pos_vectors_px_i).T) > 0:
                line_data_i = self.line_growth(pos_vectors_px_i, sarcomere_length_vectors_i,
                                               sarcomere_orientation_vectors_i,
                                               midline_length_vectors_t=midline_length_vectors_i,
                                               pixelsize=self.metadata['pixelsize'], ratio_seeds=ratio_seeds,
                                               persistence=persistence, threshold_distance=threshold_distance,
                                               n_min=n_min)
                lines_i = line_data_i['lines']

                if len(lines_i) > 0:
                    # line lengths and mean squared curvature (msc)
                    lengths_i = line_data_i['line_features']['length_lines']
                    straightness_i = line_data_i['line_features']['straightness_lines']
                    bending_i = line_data_i['line_features']['bending_lines']

                    if len(lengths_i) > 0:
                        # create myofibril length map
                        myof_map_i = self.create_myofibril_length_map(myof_lines=lines_i, myof_length=lengths_i,
                                                                      pos_vectors=pos_vectors_i,
                                                                      sarcomere_orientation_vectors=sarcomere_orientation_vectors_i,
                                                                      sarcomere_length_vectors=sarcomere_length_vectors_i,
                                                                      size=self.metadata['size'],
                                                                      pixelsize=self.metadata['pixelsize'],
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
        progress_notifier: ProgressNotifier
            Wraps progress notification, default is progress notification done with tqdm
        """
        if 'pos_vectors' not in self.data:
            raise ValueError('Sarcomere length and orientation not yet analyzed. Run analyze_sarcomere_vectors first.')
        if frames is not None:
            if (isinstance(frames, str) and frames == 'all') or (self.metadata['frames'] == 1 and frames == 0):
                frames = list(range(self.metadata['frames']))
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
            n_imgs = self.metadata['frames']
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
        none_lists = lambda: [None] * self.metadata['frames']
        nan_arrays = lambda: np.full(self.metadata['frames'], np.nan)
        n_domains, domain_area_mean, domain_area_std = (nan_arrays() for _ in range(3))
        domain_slen_mean, domain_slen_std = (nan_arrays() for _ in range(2))
        domain_oop_mean, domain_oop_std = (nan_arrays() for _ in range(2))

        (domains, domain_area, domain_slen, domain_slen_std,
         domain_oop, domain_orientation) = (none_lists() for _ in range(6))

        # iterate frames
        print('\nStarting sarcomere domain analysis...')
        for i, (frame_i, pos_vectors_i, sarcomere_length_vectors_i, sarcomere_orientation_vectors_i,
                midline_id_vectors_i) in enumerate(
            progress_notifier.iterator(
                zip(list_frames, pos_vectors, sarcomere_length_vectors, sarcomere_orientation_vectors,
                    midline_id_vectors),
                total=len(pos_vectors))):
            cluster_data_t = self.cluster_sarcomeres(pos_vectors_i, sarcomere_length_vectors_i,
                                                     sarcomere_orientation_vectors_i,
                                                     pixelsize=self.metadata['pixelsize'],
                                                     size=self.metadata['size'],
                                                     d_max=d_max, cosine_min=cosine_min,
                                                     leiden_resolution=leiden_resolution, random_seed=random_seed,
                                                     area_min=area_min, dilation_radius=dilation_radius)
            (n_domains[frame_i], domains[frame_i], domain_area[frame_i], domain_slen[frame_i], domain_slen_std[frame_i],
             domain_oop[frame_i], domain_orientation[frame_i], domain_mask_i) = cluster_data_t

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
                       'params.analyze_sarcomere_domains.dilation_radius': dilation_radius}

        self.data.update(domain_data)
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
        loi_data = self.line_growth(points_t=pos_vectors, sarcomere_length_vectors_t=sarcomere_length_vectors,
                                    sarcomere_orientation_vectors_t=sarcomere_orientation_vectors,
                                    midline_length_vectors_t=midline_length_vectors,
                                    pixelsize=self.metadata['pixelsize'],
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
        # Retrieve LOIs and their features from the structure dict
        lois = self.data['loi_data']['lines']
        loi_features = self.data['loi_data']['line_features']
        lois_vectors = self.data['loi_data']['lines_vectors']

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
        self.data['loi_data']['lines'] = [loi for i, loi in enumerate(lois) if is_good[i]]
        self.data['loi_data']['lines_vectors'] = [pos_vectors for i, pos_vectors in enumerate(lois_vectors) if
                                                  is_good[i]]

        # Filter the features dataframe and convert back to dict
        df_features = pd.DataFrame(loi_features)
        filtered_df_features = df_features[is_good].reset_index(drop=True)
        self.data['loi_data']['line_features'] = filtered_df_features.to_dict(orient='list')

    def _hausdorff_distance_lois(self, symmetry_mode: str = 'max') -> None:
        """
        Compute Hausdorff distances between all good LOIs.

        Parameters
        ----------
        symmetry_mode : str, optional
            Choose 'min' or 'max', whether min/max(H(loi_i, loi_j), H(loi_j, loi_i)). Defaults to 'max'.
        """
        # get points of LOI lines
        lines_vectors = self.data['loi_data']['lines_vectors']

        # hausdorff distance between LOIss
        hausdorff_dist_matrix = np.zeros((len(lines_vectors), len(lines_vectors)))
        for i, loi_i in enumerate(lines_vectors):
            for j, loi_j in enumerate(lines_vectors):
                if symmetry_mode == 'min':
                    hausdorff_dist_matrix[i, j] = min(directed_hausdorff(loi_i, loi_j)[0],
                                                      directed_hausdorff(loi_j, loi_i)[0])
                if symmetry_mode == 'max':
                    hausdorff_dist_matrix[i, j] = max(directed_hausdorff(loi_i, loi_j)[0],
                                                      directed_hausdorff(loi_j, loi_i)[0])

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
        if len(self.data['loi_data']['lines_vectors']) == 0:
            self.data['loi_data']['line_cluster'] = []
            self.data['loi_data']['n_lines_clusters'] = 0
        elif len(self.data['loi_data']['lines_vectors']) == 1:
            self.data['loi_data']['line_cluster'] = [[0]]
            self.data['loi_data']['n_lines_clusters'] = 1
        else:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold_lois,
                                                 metric='precomputed',
                                                 linkage=linkage).fit(
                self.data['loi_data']['hausdorff_dist_matrix'])
            self.data['loi_data']['line_cluster'] = clustering.labels_
            self.data['loi_data']['n_lines_clusters'] = len(np.unique(clustering.labels_))
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

        def linear(x, a, b):
            return a * x + b

        points_clusters = []
        loi_lines = []
        len_loi_lines = []
        add_length = add_length / self.metadata['pixelsize']
        for label_i in range(self.data['loi_data']['n_lines_clusters']):
            points_cluster_i = []
            for k in np.where(self.data['loi_data']['line_cluster'] == label_i)[0]:
                points_cluster_i.append(self.data['loi_data']['lines_vectors'][k])
            points_clusters.append(np.concatenate(points_cluster_i).T)
            p_i, pcov_i = curve_fit(linear, points_clusters[label_i][1], points_clusters[label_i][0])
            x_range_i = np.linspace(np.min(points_clusters[label_i][1]) - add_length / np.sqrt(1 + p_i[0] ** 2),
                                    np.max(points_clusters[label_i][1]) + add_length / np.sqrt(1 + p_i[0] ** 2), num=2)
            y_i = linear(x_range_i, p_i[0], p_i[1])
            len_i = np.sqrt(np.diff(x_range_i) ** 2 + np.diff(y_i) ** 2)
            x_range_i, y_i = np.round(x_range_i, 1), np.round(y_i, 1)
            loi_lines.append(np.asarray((y_i, x_range_i)).T)
            len_loi_lines.append(len_i)

        len_loi_lines = np.asarray(len_loi_lines).flatten()
        loi_lines = np.asarray(loi_lines)

        # sort lines by length
        length_idxs = len_loi_lines.argsort()
        loi_lines = loi_lines[length_idxs[::-1]][:n_lois]
        len_loi_lines = len_loi_lines[length_idxs[::-1]][:n_lois]

        self.data['loi_data']['loi_lines'] = np.asarray(loi_lines)
        self.data['loi_data']['len_loi_lines'] = np.asarray(len_loi_lines)
        if self.auto_save:
            self.store_structure_data()

    def _longest_in_cluster(self, n_lois, frame):
        lines = self.data['loi_data']['lines']
        pos_vectors = self.data['pos_vectors_px'][frame]
        lines_cluster = np.asarray(self.data['loi_data']['line_cluster'])
        longest_lines = []
        for label_i in range(self.data['loi_data']['n_lines_clusters']):
            lines_cluster_i = [line_j for j, line_j in enumerate(lines) if lines_cluster[j] == label_i]
            points_lines_cluster_i = [pos_vectors[line_j] for j, line_j in enumerate(lines) if
                                      lines_cluster[j] == label_i]
            length_lines_cluster_i = [len(line_j) for line_j in lines_cluster_i]
            longest_line = points_lines_cluster_i[np.argmax(length_lines_cluster_i)]
            longest_lines.append(longest_line)
        # get n longest lines
        sorted_by_length = sorted(longest_lines, key=lambda x: len(x[1]), reverse=True)
        if len(longest_lines) < n_lois:
            print(f'Only {len(longest_lines)}<{n_lois} clusters identified.')
        loi_lines = sorted_by_length[:n_lois]
        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = [len(line_i) for line_i in loi_lines]
        if self.auto_save:
            self.store_structure_data()

    def _random_from_cluster(self, n_lois, frame):
        lines = self.data['loi_data']['lines']
        pos_vectors = self.data['pos_vectors_px'][frame]
        lines_cluster = np.asarray(self.data['loi_data']['line_cluster'])
        random_lines = []
        for label_i in range(self.data['loi_data']['n_lines_clusters']):
            points_lines_cluster_i = [pos_vectors[line_j] for j, line_j in enumerate(lines) if
                                      lines_cluster[j] == label_i]
            random_line = random.choice(points_lines_cluster_i)
            random_lines.append(random_line)
        # select clusters randomly
        loi_lines = random.sample(random_lines, n_lois)
        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = [len(line_i) for line_i in loi_lines]
        if self.auto_save:
            self.store_structure_data()

    def _random_lois(self, n_lois, frame):
        lines = self.data['loi_data']['lines']
        pos_vectors = self.data['pos_vectors_px'][frame]
        loi_lines = random.sample(lines, n_lois)
        loi_lines = [pos_vectors[line_i] for line_i in loi_lines]
        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = [len(line_i) for line_i in loi_lines]
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
        if os.path.exists(self.file_zbands_fast_movie):
            file_z_bands = self.file_zbands_fast_movie
        else:
            file_z_bands = self.file_zbands
        imgs_sarcomeres = tifffile.imread(file_z_bands)
        profiles = self.kymograph_movie(imgs_sarcomeres, line, order=order,
                                        linewidth=int(linewidth / self.metadata['pixelsize']))
        profiles = np.asarray(profiles)
        if export_raw:
            imgs_raw = tifffile.imread(self.filepath)
            profiles_raw = self.kymograph_movie(imgs_raw, line, order=order,
                                                linewidth=int(linewidth / self.metadata['pixelsize']))
        else:
            profiles_raw = None

        # length of line
        def __calculate_segmented_line_length(line):
            diffs = np.diff(line, axis=0)
            lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            return np.sum(lengths)

        length = __calculate_segmented_line_length(line) * self.metadata['pixelsize']

        loi_data = {'profiles': profiles, 'profiles_raw': profiles_raw,
                    'line': line, 'linewidth': linewidth, 'length': length}
        for key, value in loi_data.items():
            loi_data[key] = np.asarray(value)
        save_name = os.path.join(self.base_dir,
                                 f'{line[0][0]}_{line[0][1]}_{line[-1][0]}_{line[-1][1]}_{linewidth}_loi.json')
        IOUtils.json_serialize(loi_data, save_name)

    def detect_lois(self, frame: int = 0, n_lois: int = 4, ratio_seeds: float = 0.1, persistence: int = 4,
                    threshold_distance: float = 0.5,
                    mode: str = 'longest_in_cluster', random_seed: Optional[int] = None,
                    number_lims: Tuple[int, int] = (10, 50), length_lims: Tuple[float, float] = (0, 200),
                    sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
                    sarcomere_std_length_lims: Tuple[float, float] = (0, 1),
                    midline_mean_length_lims: Tuple[float, float] = (0, 50),
                    midline_std_length_lims: Tuple[float, float] = (0, 50),
                    midline_min_length_lims: Tuple[float, float] = (0, 50), distance_threshold_lois: float = 40,
                    linkage: str = 'single', linewidth: float = 0.65, order: int = 0, export_raw: bool = False) -> None:
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

        if self.metadata['frames'] == 1:
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

            except Exception:
                pass  # Silently continue if an error occurs

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

    @staticmethod
    def segment_z_bands(image: np.ndarray, threshold: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment z-bands from U-Net result (threshold, make binary, skeletonize, label regions).

        Parameters
        ----------
        image : np.ndarray
            Input image from U-Net.
        threshold : float, optional
            Threshold value for binarizing the image. Defaults to 0.15.

        Returns
        -------
        labels : np.ndarray
            Labeled regions in the thresholded image.
        labels_skel : np.ndarray
            Labeled regions in the skeletonized image.
        """
        mask = image > threshold
        mask_skel = morphology.skeletonize(mask)
        labels = label(mask)
        labels_skel = mask_skel * labels
        return labels, labels_skel

    @staticmethod
    def _analyze_z_bands(zbands: np.ndarray, labels: np.ndarray, labels_skel: np.ndarray,
                         image_raw: np.ndarray, orientation_field: np.ndarray,
                         pixelsize: float, min_length: float = 1.0, threshold: float = 0.1,
                         median_filter_radius: float = 0.25,
                         a_min: float = 0.3, theta_phi_min: float = 0.2, d_max: float = 4.0,
                         d_min: float = 0.25) -> Tuple:
        """
        Analyzes segmented z-bands in a single frame, extracting metrics such as length, intensity, orientation,
        straightness, lateral distance, alignment, number of lateral neighbors per z-band, and characteristics of
        groups of lateral z-bands (length, alignment, size).

        Parameters
        ----------
        zbands : np.ndarray
            The segmented map of z-bands.
        labels : np.ndarray
            The labeled image of z-bands.
        labels_skel : np.ndarray
            The skeletonized labels of z-bands.
        image_raw : np.ndarray
            The raw image.
        orientation_field : np.ndarray
            Sarcomere orientation field.
        pixelsize : float
            The size of pixels in the image.
        min_length : float, optional
            The minimum length threshold for z-bands. Default is 1.0.
        threshold : float, optional
            The threshold value for intensity. Default is 0.1.
        median_filter_radius : float, optional
            Radius of kernel to smooth orientation field. Default is 0.2 µm.
        a_min : float, optional
            The minimum value for alignment. Default is 0.25. Links with smaller alignment are set to np.nan.
        theta_phi_min : float, optional
            The minimum dot product/cosine between the direction of a Z-band end and the direction of line from end to other Z-band end.
        d_max : float, optional
            The maximum distance between z-band ends. Default is 5.0 µm. Larger distances are set to np.nan.
        d_min : float, optional
            The minimum distance between z-band ends. Default is 0 µm. Smaller distances are set to np.nan.

        Returns
        -------
        tuple
            A comprehensive tuple containing arrays and values describing the analyzed properties of z-bands:
            - Lengths, intensities, straightness, ratio of intensities, average intensity, orientations,
              orientational order parameter, list of z-band labels, processed labels image, number of lateral neighbors,
              lateral distances, lateral alignments, links between z-band ends, coordinates of z-band ends,
              linked groups of z-bands, and their respective sizes, lengths, and alignments.
        """
        # analyze skeletonized labels to determine z-band backbone length
        props_skel = regionprops_table(labels_skel, properties=['label', 'perimeter'])
        labels_list = props_skel['label']

        # remove short z-bands
        length = props_skel['perimeter'] * pixelsize
        labels_list_ = labels_list.copy()
        labels_list[length < min_length] = 0
        labels_list = np.insert(labels_list, 0, 0)
        labels_list_ = np.insert(labels_list_, 0, 0)
        labels = Utils.map_array(labels, labels_list_, labels_list)
        labels, forward_map, inverse_map = segmentation.relabel_sequential(labels)
        labels_list = labels_list[labels_list != 0]

        # sarcomere orientation map
        smooth_radius_px = int(median_filter_radius / pixelsize)
        sarcomere_orientation = Utils.get_orientation_angle_map(orientation_field, use_median_filter=True,
                                                                radius=smooth_radius_px)

        # analyze z-band labels
        props = regionprops_table(labels, intensity_image=image_raw, properties=['label', 'area', 'convex_area',
                                                                                 'mean_intensity', 'orientation',
                                                                                 'image', 'bbox', 'centroid'])
        # z-band length
        length = length[length >= min_length]

        # straightness of z-bands (area/convex_hull)
        straightness = props['area'] / props['convex_area']

        # fluorescence intensity of each individual z-band, the total area, and the average intensity of z-band mask
        intensity = props['mean_intensity']
        z_mask = zbands > threshold
        z_mask_area = np.sum(z_mask.astype('uint8')) * pixelsize ** 2
        z_mask_intensity = np.mean(image_raw[z_mask])

        # z band orientational order parameter
        orientation = props['orientation']
        if len(orientation) > 0:
            oop = 1 / len(orientation) * np.abs(np.sum(np.exp(orientation * 2 * 1j)))
        else:
            oop = np.nan

        # local lateral z-band alignment and distance
        n_z = len(np.unique(labels)) - 1

        if n_z > 0:

            # get two ends of each z-band
            z_ends = np.zeros((n_z, 2, 2)) * np.nan  # (z-band idx, upper/lower end, x/y)
            z_orientation = np.zeros((n_z, 2)) * np.nan  # (z-band idx, upper/lower)
            pad_width = int(round(1 / pixelsize, 0))

            for i, zbands_i in enumerate(props['image']):
                zbands_i = np.pad(zbands_i, (pad_width, pad_width))

                # skeletonize
                skel_i = skeletonize(zbands_i, method='lee')

                # detect line ends
                def line_end_filter(d):
                    return (d[4] == 1) and np.sum(d) == 2

                z_ends_i = ndimage.generic_filter(skel_i, line_end_filter, (3, 3))
                z_ends_i = np.asarray(np.where(z_ends_i == 1))
                z_ends_i[0] += props['bbox-0'][i] - pad_width
                z_ends_i[1] += props['bbox-1'][i] - pad_width
                centroid_i = (props['centroid-0'][i], props['centroid-1'][i])

                if len(z_ends_i.T) == 2:
                    if z_ends_i[1, 0] > z_ends_i[1, 1]:
                        z_ends_i = z_ends_i[:, ::-1]
                    # Get orientations from map and add π/2
                    orientation_ends_i = np.asarray([sarcomere_orientation[z_ends_i[0][0], z_ends_i[1][0]] + np.pi / 2,
                                                     sarcomere_orientation[z_ends_i[0][1], z_ends_i[1][1]] + np.pi / 2])

                    # Calculate local directions from endpoints to their own positions in skeleton
                    _orient_1 = np.arctan2(z_ends_i[0, 0] - centroid_i[0], z_ends_i[1, 0] - centroid_i[1])
                    _orient_2 = np.arctan2(z_ends_i[0, 1] - centroid_i[0], z_ends_i[1, 1] - centroid_i[1])

                    # Better angle difference calculation (minimum angle in range [0, π])
                    def angle_diff(a1, a2):
                        return np.abs((a1 - a2 + np.pi) % (2 * np.pi) - np.pi)

                    # Apply π shift if angles differ by more than π/2
                    if angle_diff(orientation_ends_i[0], _orient_1) > np.pi / 2:
                        orientation_ends_i[0] = orientation_ends_i[0] + np.pi
                    if angle_diff(orientation_ends_i[1], _orient_2) > np.pi / 2:
                        orientation_ends_i[1] = orientation_ends_i[1] + np.pi

                    orientation_ends_i = -orientation_ends_i + np.pi / 2

                    # # Ensure angles stay in range [-π, π]
                    orientation_ends_i[0] = (orientation_ends_i[0] + np.pi) % (2 * np.pi) - np.pi
                    orientation_ends_i[1] = (orientation_ends_i[1] + np.pi) % (2 * np.pi) - np.pi

                    z_orientation[i] = orientation_ends_i
                    z_ends[i] = z_ends_i.T * pixelsize

            # lateral alignment index and distance of z-bands
            def lateral_alignment(pos_i, pos_j, theta_i, theta_j):
                phi_ij = np.arctan2((pos_j[1] - pos_i[1]), (pos_j[0] - pos_i[0])) % (2 * np.pi)
                phi_ji = (phi_ij + np.pi) % (2 * np.pi)

                a_ji = np.cos(theta_i - theta_j + np.pi) * np.cos(theta_i - phi_ij) * np.cos(theta_j - phi_ji)

                if np.cos(theta_i - theta_j + np.pi) > 0 and np.cos(theta_i - phi_ij) > theta_phi_min and np.cos(
                        theta_j - phi_ji) > theta_phi_min:
                    return a_ji
                else:
                    return np.nan

            # distance of z-band ends
            _z_ends = np.reshape(z_ends, (n_z * 2, 2), order='F')
            D = squareform(pdist(_z_ends, 'euclidean'))

            # Set NaNs for specified indices (ends of same objects) and the lower triangle
            indices = np.arange(0, n_z * 2, 2)
            mask = np.ones((n_z * 2, n_z * 2))
            mask[indices, indices] = 0
            mask[indices, indices + 1] = 0
            mask[indices + 1, indices] = 0
            mask[indices + 1, indices + 1] = 0
            mask[np.tril(mask) > 0] = np.nan

            # filter distance matrix
            D[(D > d_max) | (D < d_min) | (mask == 0)] = np.nan

            # indices of end-end-distances shorter than d_max
            _z_orientation = np.reshape(z_orientation, (n_z * 2), order='F')
            _idxs = np.asarray(np.where(~np.isnan(D)))

            # matrix with lateral alignments A
            A = np.zeros_like(D) * np.nan
            for (i, j) in _idxs.T:
                A_ij = lateral_alignment(_z_ends[i], _z_ends[j], _z_orientation[i], _z_orientation[j])
                A[i, j] = A_ij if A_ij >= a_min else np.nan
            D[np.isnan(A)] = np.nan

            # make matrices symmetric for undirected graph
            D = (D + D.T) / 2
            A = (A + A.T) / 2

            def compute_cost_matrix(D, A, penalty=1e6):
                """
                Compute the cost matrix for linking Z-band ends based on a cost 1 - A favoring optimal alignment.

                Parameters:
                ----------
                D : ndarray
                    Distance matrix between Z-band ends.
                A : ndarray
                    Alignment matrix between Z-band ends.
                w_dist : float
                    Weight for distance in the cost function.
                w_align : float
                    Weight for alignment in the cost function.
                penalty : float
                    Penalty for invalid links (e.g., NaN or out-of-range values).

                Returns:
                -------
                C : ndarray
                    Cost matrix for linking Z-band ends.
                """
                # Ensure alignment values are valid (replace NaNs with 0)
                A = np.nan_to_num(A, nan=0.0)

                # Compute cost matrix
                C = 1 - A

                # Set invalid links (e.g., NaNs in D) to a very high cost
                C[np.isnan(D)] = penalty

                return C

            def solve_linking(C):
                """
                Solve the optimal linking problem using the Hungarian algorithm.

                Parameters:
                ----------
                C : ndarray
                    Cost matrix for linking Z-band ends.

                Returns:
                -------
                row_ind : ndarray
                    Row indices of the optimal assignment.
                col_ind : ndarray
                    Column indices of the optimal assignment.
                """
                # Use scipy's linear_sum_assignment to solve the assignment problem
                row_ind, col_ind = linear_sum_assignment(C)

                return row_ind, col_ind

            # Step 1: Compute cost matrix
            C = compute_cost_matrix(D, A)

            # Step 2: Solve optimal linking using Hungarian algorithm
            row_ind, col_ind = solve_linking(C)

            # Step 3: Create adjacency matrix for valid links
            links = np.zeros_like(D)
            for i, j in zip(row_ind, col_ind):
                links[i, j] = 1 if D[i, j] <= d_max and A[i, j] >= a_min else 0

            # reshape arrays
            links = links.reshape((n_z, 2, n_z, 2), order='F')
            lat_dist = D.reshape((n_z, 2, n_z, 2), order='F')
            lat_alignment = A.reshape((n_z, 2, n_z, 2), order='F')

            # number of lateral neighbors
            links_z = np.sum(links, axis=(1, 3))
            lat_neighbors = np.count_nonzero(links_z, axis=1)

            # convert links, lat_dist and lat_alignment to lists
            links = np.where(links == 1)
            lat_dist = lat_dist[links]
            lat_alignment = lat_alignment[links]
            links = np.asarray(links)

            # analyze laterally linked groups
            def analyze_linked_groups(connectivity_matrix, distance_matrix, alignment_matrix):
                G = nx.Graph()

                for n in range(n_z):
                    G.add_node(n)

                # Efficiently add edges based on connectivity and criteria
                for n, (idx_i, end_i, idx_j, end_j) in enumerate(connectivity_matrix.T):
                    G.add_edge(idx_i, idx_j, alignment=alignment_matrix[n], distance=distance_matrix[n])

                # Find connected components in the graph with best matches
                _linked_groups = list(nx.connected_components(G))

                _size_groups = np.asarray([len(group) for group in _linked_groups])
                # Calculate length of each group
                _length_groups = []
                _alignment_groups = []
                for group in _linked_groups:
                    sum_distance = 0
                    sum_alignment = 0
                    for node in group:
                        edges = G.edges(node, data=True)
                        for _, _, data in edges:
                            if G.has_edge(_, node):  # Check if edge is within the current group
                                sum_distance += data['distance']
                                sum_alignment += data['alignment']
                    sum_distance /= 2  # Each edge is counted twice (undirected graph), so divide by 2
                    _length_groups.append(sum_distance + np.sum(length[list(group)]))
                    _alignment_groups.append(sum_alignment / len(group))
                _linked_groups = [list(s) for s in _linked_groups]
                return (_linked_groups, np.asarray(_size_groups), np.asarray(_length_groups),
                        np.asarray(_alignment_groups))

            linked_groups, size_groups, length_groups, alignment_groups = analyze_linked_groups(links, lat_dist,
                                                                                                lat_alignment)
        else:
            (lat_neighbors, lat_dist, lat_alignment, links, z_ends,
             linked_groups, size_groups, length_groups, alignment_groups) = [], [], [], [], [], [], [], [], []

        return (length, intensity, straightness, z_mask_intensity, z_mask_area, orientation, oop, labels_list, labels,
                lat_neighbors, lat_dist, lat_alignment, links, z_ends, linked_groups, size_groups, length_groups,
                alignment_groups)

    @staticmethod
    def get_sarcomere_vectors(
            zbands: np.ndarray,
            mbands: np.ndarray,
            orientation_field: np.ndarray,
            pixelsize: float,
            median_filter_radius: float = 0.25,
            slen_lims: Tuple[float, float] = (1, 3),
            interp_factor: int = 4,
            linewidth: float = 0.3,
            backend: str = 'loky',
    ) -> Tuple[Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List],
    Union[np.ndarray, List], Union[np.ndarray, List], Union[np.ndarray, List]]:
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
        radius_pixels = max(int(round(median_filter_radius / pixelsize, 0)), 1)
        linewidth_pixels = max(int(round(linewidth / pixelsize, 0)), 1)

        # skeletonize mbands
        mbands_skel = skeletonize(mbands, method='lee')

        # calculate and preprocess orientation map
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
            for i, (label_i, coords_i, length_midline_i) in enumerate(
                    zip(list_labels, coords_mbands, length_mbands)):
                pos_vectors_px.append(coords_i)
                midline_length_vectors.append(np.ones(coords_i.shape[0]) * length_midline_i)
                midline_id_vectors.append(np.ones(coords_i.shape[0]) * label_i)

            pos_vectors_px = np.concatenate(pos_vectors_px, axis=0)
            midline_id_vectors = np.concatenate(midline_id_vectors)
            midline_length_vectors = np.concatenate(midline_length_vectors)

            sarcomere_orientation_vectors = orientation[pos_vectors_px[:, 0], pos_vectors_px[:, 1]]

            ends1 = pos_vectors_px.T + (slen_lims[1] * 1.3) / 2 / pixelsize * np.array(
                (np.sin(sarcomere_orientation_vectors), np.cos(sarcomere_orientation_vectors))
            )
            ends2 = pos_vectors_px.T - (slen_lims[1] * 1.3) / 2 / pixelsize * np.array(
                (np.sin(sarcomere_orientation_vectors), np.cos(sarcomere_orientation_vectors))
            )

            # Calculate sarcomere lengths by measuring peak-to-peak distance of Z-bands in intensity profile
            profiles = Utils.fast_profile_lines(zbands, ends1, ends2, linewidth=linewidth_pixels)

            # Use parallel processing for faster execution
            results = Parallel(n_jobs=-1, backend=backend)(
                delayed(Utils.process_profile)(
                    profile, pixelsize, slen_lims=slen_lims, interp_factor=interp_factor
                ) for profile in profiles
            )

            # Convert results to array
            sarcomere_length_vectors, center_offsets = np.array(results).T

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
            sarcomere_length_vectors, z_band_thickness_vectors, sarcomere_orientation_vectors = [], [], []

        return (pos_vectors_px, pos_vectors, midline_id_vectors, midline_length_vectors, sarcomere_length_vectors,
                sarcomere_orientation_vectors)

    @staticmethod
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
                           dilation_radius: float = 0.3) -> Tuple[int, List, List, List, List, List, np.ndarray]:
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
            return 0, [], [], [], [], [], [], []

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
         sarcomere_orientation_domains) = Structure._analyze_domains(domains, pos_vectors,
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

    @staticmethod
    def _grow_line(seed, points_t, sarcomere_length_vectors_t, sarcomere_orientation_vectors_t, nbrs,
                   threshold_distance, pixelsize, persistence):
        line_i = deque([seed])
        stop_right = stop_left = False

        sarcomere_orientation_vectors_t = sarcomere_orientation_vectors_t + np.pi / 2

        threshold_distance_pixels = threshold_distance / pixelsize

        def angle_diff(alpha, beta):
            """Return the signed difference between angles alpha and beta.
            The result is in the range [-pi, pi]."""
            return np.arctan2(np.sin(beta - alpha), np.cos(beta - alpha))

        def calculate_mean_orientation(orientations):
            # Convert orientations to complex numbers on the unit circle
            complex_orientations = np.exp(2j * np.array(orientations))
            # Calculate the mean of the complex numbers
            mean_complex = np.mean(complex_orientations)
            # Convert back to angle and halve it to get the original range
            return np.angle(mean_complex) / 2

        def adjust_orientation(current_orientation, previous_orientation):
            diff = angle_diff(current_orientation, previous_orientation)
            if diff > np.pi / 2:
                return current_orientation - np.pi
            elif diff < -np.pi / 2:
                return current_orientation + np.pi
            return current_orientation

        # Initialize orientations
        orientation_left = orientation_right = sarcomere_orientation_vectors_t[seed]

        points_t = points_t.T

        while not stop_left or not stop_right:
            line_i_list = list(line_i)

            if not stop_left:
                end_left = points_t[:, line_i_list[0]]
                length_left = np.mean(sarcomere_length_vectors_t[line_i_list[:persistence]]) / pixelsize
                new_orientation_left = calculate_mean_orientation(
                    sarcomere_orientation_vectors_t[line_i_list[:persistence]]) if persistence > 1 else sarcomere_orientation_vectors_t[line_i_list[0]]
                orientation_left = adjust_orientation(new_orientation_left, orientation_left)

            if not stop_right:
                end_right = points_t[:, line_i_list[-1]]
                length_right = np.mean(sarcomere_length_vectors_t[line_i_list[-persistence:]]) / pixelsize
                new_orientation_right = calculate_mean_orientation(
                    sarcomere_orientation_vectors_t[line_i_list[-persistence:]]) if persistence > 1 else sarcomere_orientation_vectors_t[line_i_list[-1]]
                orientation_right = adjust_orientation(new_orientation_right, orientation_right)

            # grow left
            if not stop_left:
                prior_left = [end_left[0] + np.cos(orientation_left) * length_left,
                              end_left[1] - np.sin(orientation_left) * length_left]
                distance_left, index_left = nbrs.kneighbors([prior_left], return_distance=True)
                if distance_left[0][0] < threshold_distance_pixels:
                    line_i.appendleft(index_left[0][0].astype('int'))
                else:
                    stop_left = True

            # grow right
            if not stop_right:
                prior_right = [end_right[0] - np.cos(orientation_right) * length_right,
                               end_right[1] + np.sin(orientation_right) * length_right]
                distance_right, index_right = nbrs.kneighbors([prior_right], return_distance=True)
                if distance_right[0][0] < threshold_distance_pixels:
                    line_i.append(index_right[0][0].astype('int'))
                else:
                    stop_right = True

        return np.asarray(line_i)

    @staticmethod
    def line_growth(points_t: np.ndarray, sarcomere_length_vectors_t: np.ndarray,
                    sarcomere_orientation_vectors_t: np.ndarray,
                    midline_length_vectors_t: np.ndarray, pixelsize: float, ratio_seeds: float = 0.1,
                    persistence: int = 4, threshold_distance: float = 0.3, n_min: int = 5,
                    random_seed: Union[None, int] = None):
        """
        Line growth algorithm to determine myofibril lines perpendicular to sarcomere z-bands

        Parameters
        ----------
        points_t : np.ndarray
            List of midline point positions
        sarcomere_length_vectors_t : list
            Sarcomere length at midline points
        sarcomere_orientation_vectors_t : list
            Sarcomere orientation angle at midline points, in radians
        midline_length_vectors_t : list
            Length of sarcomere mbands of midline points
        pixelsize : float
            Pixel size in µm
        ratio_seeds : float
            Ratio of sarcomere vectors to be takes as seeds for line growth
        persistence : int
            Number of points to consider for averaging length and orientation.
        random_seed : int, optional
            Random seed for reproducibility. Defaults to None.

        Returns
        -------
        line_data : dict
            Dictionary with LOI data keys = (lines, line_features)
        """
        # select random origins for line growth
        points_t = np.asarray(points_t)
        if points_t.shape[0] == 2:
            points_t = points_t.T

        if len(points_t) == 0:
            print('No sarcomeres in image (len(points) = 0), could not grow lines.')
            return {'lines': [], 'line_features': {}}

        if random_seed:
            random.seed(random_seed)
        n_vectors = len(points_t)
        seed_idx = random.sample(range(n_vectors), max(1, int(ratio_seeds * n_vectors)))

        # Precompute Nearest Neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_t)

        # Prepare arguments for parallel processing
        args = [
            (seed, points_t, sarcomere_length_vectors_t, sarcomere_orientation_vectors_t, nbrs, threshold_distance,
             pixelsize, persistence) for seed in seed_idx]

        # grow lines
        lines = [Structure._grow_line(*arg) for arg in args]

        # remove short lines (< n_min)
        lines = [l for l in lines if len(l) >= n_min]

        # calculate features of lines
        n_vectors_lines = np.asarray([len(l) for l in lines])  # number of sarcomeres in line
        length_line_segments = [sarcomere_length_vectors_t[l] for l in lines]
        length_lines = [np.sum(lengths) for lengths in length_line_segments]

        # sarcomere lengths
        sarcomere_mean_length_lines = [np.mean(sarcomere_length_vectors_t[l]) for l in lines]
        sarcomere_std_length_lines = [np.std(sarcomere_length_vectors_t[l]) for l in lines]

        # midline lengths
        midline_mean_length_lines = [np.nanmean(midline_length_vectors_t[l]) for l in lines]
        midline_std_length_lines = [np.nanstd(midline_length_vectors_t[l]) for l in lines]
        midline_min_length_lines = [np.nanmin(midline_length_vectors_t[l]) for l in lines]

        # Straightness
        def frechet_straightness(points):
            """
            Compute a Fréchet-inspired straightness measure:
            1 - (max perpendicular deviation from chord / chord length)

            Parameters
            ----------
            points : np.ndarray
                Array of shape (n_points, 2) representing polyline vertices

            Returns
            -------
            float
                Straightness measure (1 = perfectly straight)
            """

            if len(points) < 2:
                return 1.0  # Single point is trivially straight

            # Calculate chord vector between first and last points
            chord_vector = points[-1] - points[0]
            chord_length = np.linalg.norm(chord_vector)

            if chord_length < 1e-9:  # Handle degenerate chord
                return 0.0

            # Unit vector along chord direction
            unit_chord = chord_vector / chord_length

            # Vectors from first point to each polyline vertex
            displacement_vectors = points - points[0]

            # Scalar projections onto chord (dot product with unit vector)
            chord_projections = np.sum(displacement_vectors * unit_chord, axis=1)

            # Ideal points along chord line
            projected_points = points[0] + chord_projections[:, np.newaxis] * unit_chord

            # Perpendicular deviations from actual path
            deviation_vectors = points - projected_points
            perpendicular_deviations = np.linalg.norm(deviation_vectors, axis=1)

            max_deviation = np.max(perpendicular_deviations)

            return 1.0 - (max_deviation / chord_length)

        straightness_lines = [
            frechet_straightness(points_t[line])
            for line in lines
        ]

        # Bending: mean squared angular change
        tangential_vector_line_segments = [np.diff(points_t[l], axis=0) for l in lines]
        tangential_angle_line_segments = [np.asarray([np.arctan2(v[1], v[0]) for v in vectors]) for vectors in
                                          tangential_vector_line_segments]
        bending_lines = [
            np.mean(np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles))) ** 2) if len(angles) > 1 else 0.0
            for angles in tangential_angle_line_segments
        ]

        # create dictionary
        line_features = {'n_vectors_lines': n_vectors_lines, 'length_lines': length_lines,
                         'sarcomere_mean_length_lines': sarcomere_mean_length_lines,
                         'sarcomere_std_length_lines': sarcomere_std_length_lines,
                         'bending_lines': bending_lines,
                         'straightness_lines': straightness_lines,
                         'midline_mean_length_lines': midline_mean_length_lines,
                         'midline_std_length_lines': midline_std_length_lines,
                         'midline_min_length_lines': midline_min_length_lines}
        line_features = Utils.convert_lists_to_arrays_in_dict(line_features)
        line_data = {'lines': lines, 'line_features': line_features}
        return line_data

    @staticmethod
    def kymograph_movie(movie: np.ndarray, line: np.ndarray, linewidth: int = 10, order: int = 0):
        """
        Generate a kymograph using multiprocessing.

        Parameters
        --------
        movie : np.ndarray, shape (N, H, W)
            The movie.
        line : np.ndarray, shape (N, 2)
            The coordinates of the segmented line (N>1)
        linewidth : int, optional
            Width of the scan in pixels, perpendicular to the line
        order : int in {0, 1, 2, 3, 4, 5}, optional
            The order of the spline interpolation, default is 0 if
            image.dtype is bool and 1 otherwise. The order has to be in
            the range 0-5. See `skimage.transform.warp` for detail.

        Return
        ---------
        return_value : ndarray
            Kymograph along segmented line

        Notes
        -------
        Adapted from scikit-image
        (https://scikit-image.org/docs/0.22.x/api/skimage.measure.html#skimage.measure.profile_line).
        """
        # prepare coordinates of segmented line
        perp_lines = Structure.__curved_line_profile_coordinates(points=line, linewidth=linewidth)

        # Prepare arguments for each frame
        args = [(movie[frame], perp_lines, linewidth, order) for frame in range(movie.shape[0])]

        # Create a Pool and map process_frame to each frame
        with Pool() as pool:
            results = pool.map(Structure.process_frame, args)

        # Convert list of results to a numpy array
        kymograph = np.array(results)

        return kymograph

    @staticmethod
    def process_frame(args):
        frame, perp_lines, linewidth, order = args
        pixels = ndimage.map_coordinates(frame, perp_lines, prefilter=order > 1,
                                         order=order, mode='reflect', cval=0.0)
        pixels = np.flip(pixels, axis=1)
        intensities = np.mean(pixels, axis=1)
        return intensities

    @staticmethod
    def __curved_line_profile_coordinates(points: np.ndarray, linewidth: int = 10):
        """
        Calculate the coordinates of a curved line profile composed of multiple segments with specified linewidth.

        Parameters
        ----------
        points : np.ndarray
            A list of points (y, x) defining the segments of the curved line.
        linewidth : int, optional
            The width of the line in pixels.

        Returns
        -------
        coords : ndarray
            The coordinates of the curved line profile. Shape is (2, N, linewidth),
            where N is the total number of points in the line.
        """
        all_perp_rows = []
        all_perp_cols = []

        for i in range(len(points) - 1):
            src, dst = np.asarray(points[i], dtype=float), np.asarray(points[i + 1], dtype=float)
            d_row, d_col = dst - src
            theta = np.arctan2(d_row, d_col)
            length = int(np.ceil(np.hypot(d_row, d_col) + 1))
            line_col = np.linspace(src[1], dst[1], length)
            line_row = np.linspace(src[0], dst[0], length)
            col_width, row_width = (linewidth - 1) * np.sin(-theta) / 2, (linewidth - 1) * np.cos(theta) / 2
            perp_rows = np.stack([np.linspace(row - row_width, row + row_width, linewidth) for row in line_row])
            perp_cols = np.stack([np.linspace(col - col_width, col + col_width, linewidth) for col in line_col])

            all_perp_rows.append(perp_rows)
            all_perp_cols.append(perp_cols)

        # Concatenate all segments
        final_perp_rows = np.concatenate(all_perp_rows, axis=0)
        final_perp_cols = np.concatenate(all_perp_cols, axis=0)

        return np.stack([final_perp_rows, final_perp_cols])

    @staticmethod
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

    @staticmethod
    def _analyze_domains(domains: List, pos_vectors: np.ndarray,
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

                mask_i = Structure.sarcomere_mask(_pos_vectors_i, orientations_i, lengths_i, size_i,
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

    @staticmethod
    def create_myofibril_length_map(
            myof_lines: np.ndarray,
            myof_length: np.ndarray,
            pos_vectors: np.ndarray,
            sarcomere_orientation_vectors: np.ndarray,
            sarcomere_length_vectors: np.ndarray,
            size: tuple,
            pixelsize: float,
            median_filter_radius: float = 0.6,
    ) -> np.ndarray:
        """
        The `create_myofibril_length_map` function generates a **2D spatial map** of myofibril lengths represented
        as pixel values. It achieves this by rasterizing myofibril line segments, assigning their corresponding lengths
        to the pixels they occupy, and averaging these values at overlapping pixels. The resulting map is optionally
        smoothed using a median filter to reduce noise and provide a more coherent spatial representation.

        Parameters
        ----------
        myof_lines : ndarray
            Line indices for myofibril structures.
        myof_length : ndarray
            Length values for each myofibril line.
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
        median_filter_radius : float, optional
            Filter radius in micrometers, by default 0.6.

        Returns
        -------
        ndarray
            2D array of calculated myofibril lengths with NaN for empty regions.
        """
        # Convert median filter radius to pixels
        median_radius_px = int(round(median_filter_radius / pixelsize))

        # Initialize accumulation maps
        length_sum_map = np.zeros(size, dtype=np.float32)
        weight_map = np.zeros(size, dtype=np.float32)

        # Process each myofibril segment
        for line_idx, line_length in zip(myof_lines, myof_length):
            # Extract vector data for current line
            points = pos_vectors[line_idx]
            orientations = sarcomere_orientation_vectors[line_idx] + np.pi / 2
            lengths = sarcomere_length_vectors[line_idx]

            # Calculate direction vectors
            dir_x = np.cos(orientations)
            dir_y = -np.sin(orientations)
            directions = np.vstack([dir_x, dir_y])

            # Calculate endpoints in pixel coordinates
            end_offset = directions * lengths / 2
            end_points = np.stack([
                (points.T + end_offset) / pixelsize,
                (points.T - end_offset) / pixelsize
            ]).astype(np.int32)

            # Rasterize lines
            for (x0, y0), (x1, y1) in zip(end_points[0].T, end_points[1].T):
                rr, cc = line(x0, y0, x1, y1)
                # Apply boundary constraints
                valid = (rr >= 0) & (rr < size[0]) & (cc >= 0) & (cc < size[1])
                np.add.at(length_sum_map, (rr[valid], cc[valid]), line_length)
                np.add.at(weight_map, (rr[valid], cc[valid]), 1)

        # Calculate weighted average
        myof_map = np.divide(length_sum_map, weight_map,
                             out=np.full_like(length_sum_map, np.nan),
                             where=weight_map > 0)

        # Apply median filtering if required
        if median_radius_px > 0:
            window_size = 2 * median_radius_px + 1
            myof_map = Utils.nanmedian_filter_numba(myof_map, window_size)

        return myof_map



if __name__ == "__main__":
    print('Testing Structure class')

    test_file = '../test_data/antibody_staining_2D_hiPSC_CM/stained hiPSC d60 a-actinin 488 63x 5.tif'

    sarc = Structure(test_file, pixelsize=0.114)

    # detect sarcomeres
    sarc.detect_sarcomeres()

    # analyze Z-bands
    sarc.analyze_z_bands()

    # analyze sarcomere vectors
    sarc.analyze_sarcomere_vectors()

    # analyze myofibrils
    sarc.analyze_myofibrils()

    # analyze sarcomere domains
    sarc.analyze_sarcomere_domains()