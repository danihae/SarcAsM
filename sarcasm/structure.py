import glob
import os
import random
from collections import deque
from multiprocessing import Pool
from typing import Optional, Tuple, Union, List

import networkx as nx
import numpy as np
import pandas as pd
import skimage.measure
import tifffile
import torch
import torch.nn.functional as F
from biu import unet
from biu import unet3d as unet3d
from biu.progress import ProgressNotifier
from networkx.algorithms import community
from scipy import ndimage, stats, sparse
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff, squareform, pdist
from skimage import segmentation, morphology
from skimage.draw import disk as draw_disk, line
from skimage.measure import label, regionprops_table, regionprops, profile_line
from skimage.morphology import skeletonize, disk, binary_dilation
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm as tqdm

from .ioutils import IOUtils
from .utils import Utils


class Structure:
    """
    Class to analyze sarcomere morphology.

    Attributes
    ----------
    sarc_obj : SarcAsM
        An object containing file metadata and related methods.
    data : dict
        A dictionary to store structure data.
    """

    def __init__(self, sarc_obj) -> None:
        """
        Initialize the Structure class.

        Parameters
        ----------
        sarc_obj : SarcAsM
            A SarcAsM object of the base class.
        """
        self.sarc_obj = sarc_obj

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
        if is_temp_file:
            return os.path.join(self.sarc_obj.data_folder, "structure.temp.json")
        else:
            return os.path.join(self.sarc_obj.data_folder, "structure.json")

    def commit(self) -> None:
        """
        Commit data by either renaming the temporary file to the normal data file name or just writing it again
        and removing the temporary file.
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
            If True, overrides the existing file. If False, only stores the data if the file does not exist.
            Default is True.
        """
        if override or not os.path.exists(self.__get_structure_data_file(is_temp_file=False)):
            IOUtils.json_serialize(self.data, self.__get_structure_data_file(is_temp_file=True))
            self.commit()

    def _load_structure_data(self) -> None:
        """
        Load structure data. If the normal data file exists, load it. If it fails and a temporary file exists,
        load the temporary file.

        Raises
        ------
        Exception
            If loading of structure data fails.
        """
        try:
            if os.path.exists(self.__get_structure_data_file(is_temp_file=False)):
                self.data = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=False))
            elif os.path.exists(self.__get_structure_data_file(is_temp_file=True)):
                self.data = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=True))
        except:
            if os.path.exists(self.__get_structure_data_file(is_temp_file=True)):
                self.data = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=True))

        if self.data is None:
            raise Exception('Loading of structure failed')

    def read_imgs(self, frame: Union[str, int, List[int]] = None):
        """Load tif file, and optionally select channel"""
        if frame is None or frame == 'all':
            data = tifffile.imread(self.sarc_obj.filename)
        else:
            data = tifffile.imread(self.sarc_obj.filename, key=frame)

        if self.sarc_obj.channel is not None:
            if self.sarc_obj.channel == 'RGB':
                # Convert RGB to grayscale
                if data.ndim == 3 and data.shape[2] == 3:  # Single RGB image
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
                elif data.ndim == 4 and data.shape[3] == 3:  # Stack of RGB images
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
            elif isinstance(self.sarc_obj.channel, int):
                if data.ndim == 3:
                    data = data[:, :, self.sarc_obj.channel]
                elif data.ndim == 4:
                    data = data[:, :, :, self.sarc_obj.channel]
            else:
                raise Exception('Parameter "channel" must be either int or "RGB"')

        return data

    def predict_z_bands(self, time_consistent: bool = False, model_path: Optional[str] = None,
                        size: Tuple[int, int] = (1024, 1024), normalization_mode: str = 'all',
                        clip_thres: Tuple[float, float] = (0., 99.8),
                        progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Predict sarcomere z-bands with U-Net (single images or long time-lapses) or 3D U-Net (high framerate movies).

        Parameters
        ----------
        time_consistent : bool, optional
            If True, the temporally more consistent Siam-U-Net is used. Default is False.
        model_path : str, optional
            Path of trained network weights for U-Net or Siam-U-Net. Default is None.
        size : tuple of int, optional
            Resize dimensions for convolutional neural network. Dimensions need to be divisible by 16. Default is (1024, 1024).
        normalization_mode : str, optional
            Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
            'all': based on histogram of full stack, 'first': based on histogram of first image in stack). Default is 'all'.
        clip_thres : tuple of float, optional
            Clip threshold (lower / upper) for intensity normalization. Default is (0., 99.8).
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in GUI. Default is ProgressNotifier.progress_notifier_tqdm().

        Returns
        -------
        None
        """
        print('Predicting sarcomere z-bands ...')
        if time_consistent:
            if model_path is None:
                model_path = os.path.join(self.sarc_obj.model_dir, 'unet3d_z_bands.pth')
            assert len(size) == 3, 'patch size for prediction has to be be (frames, x, y)'
            _ = unet3d.Predict(self.read_imgs(), self.sarc_obj.file_sarcomeres, model_params=model_path,
                               resize_dim=size, normalization_mode=normalization_mode, device=self.sarc_obj.device,
                               clip_threshold=clip_thres, normalize_result=True, progress_notifier=progress_notifier)
            del _
        else:
            if model_path is None or model_path == 'generalist':
                model_path = os.path.join(self.sarc_obj.model_dir, 'unet_z_bands_generalist_v0.pth')
            _ = unet.Predict(self.read_imgs(), self.sarc_obj.file_sarcomeres, model_params=model_path,
                             resize_dim=size, normalization_mode=normalization_mode, network='Unet_v0',
                             clip_threshold=clip_thres, normalize_result=True, device=self.sarc_obj.device,
                             progress_notifier=progress_notifier)
            del _
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _dict = {'params.predict_z_bands_model': model_path,
                 'params.predict_z_bands_time_consistent': time_consistent,
                 'params.predict_z_bands_normalization_mode': normalization_mode,
                 'params.predict_z_bands_clip_threshold': clip_thres}
        self.data.update(_dict)
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def predict_cell_area(self, model_path: Optional[str] = None, size: Tuple[int, int] = (1024, 1024),
                          normalization_mode: str = 'all', clip_thres: Tuple[float, float] = (0.05, 99.95),
                          progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Predict binary mask of cells vs. background with U-Net.

        Parameters
        ----------
        model_path : str, optional
            Path of trained network weights for U-Net. Default is None, chooses default U-Net model.
        size : tuple of int, optional
            Resize dimensions for convolutional neural network. Dimensions need to be divisible by 16. Default is (1024, 1024).
        normalization_mode : str, optional
            Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
            'all': based on histogram of full stack, 'first': based on histogram of first image in stack). Default is 'all'.
        clip_thres : tuple of float, optional
            Clip threshold (lower / upper) for intensity normalization. Default is (0., 99.8).
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in GUI. Default is ProgressNotifier.progress_notifier_tqdm().
        """
        print('Predicting binary mask of cells ...')

        if model_path is None or model_path == 'generalist':
            model_path = self.sarc_obj.model_dir + 'unet_cell_mask_generalist.pth'
        _ = unet.Predict(self.read_imgs(), self.sarc_obj.file_cell_mask, model_params=model_path,
                         resize_dim=size, normalization_mode=normalization_mode, network='AttentionUnet',
                         device=self.sarc_obj.device,
                         clip_threshold=clip_thres, normalize_result=True, progress_notifier=progress_notifier)
        del _
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _dict = {'params.predict_cell_area_model': model_path,
                 'params.predict_cell_area_normalization_mode': normalization_mode,
                 'params.predict_cell_area_clip_threshold': clip_thres}
        self.data.update(_dict)
        self.store_structure_data()

    def analyze_cell_area(self, threshold: float = 0.1) -> None:
        """
        Analyzes the area of cells in the given image(s) and calculates the cell area ratio.

        Parameters
        ----------
        threshold : float, optional
            Threshold value for binarizing the cell mask image. Pixels with intensity
            above threshold * 255 are considered cell. Defaults to 0.1.
        """
        assert self.sarc_obj.file_cell_mask is not None, "Cell mask not found. Please run predict_cell_area first."

        imgs = tifffile.imread(self.sarc_obj.file_cell_mask)

        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs, 0)

        n_imgs = len(imgs)

        # create empty array
        cell_area, cell_area_ratio = np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan

        for i, img in enumerate(tqdm(imgs)):
            # binarize mask
            mask = np.zeros_like(img)
            mask[img > threshold * 255] = 1

            cell_area[i] = np.sum(mask) * self.sarc_obj.metadata['pixelsize'] ** 2
            cell_area_ratio[i] = cell_area[i] / (img.shape[0] * img.shape[1] * self.sarc_obj.metadata['pixelsize'] ** 2)
        _dict = {'cell_area': cell_area, 'cell_area_ratio': cell_area_ratio, 'params.cell_area_threshold': threshold}
        self.data.update(_dict)
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def analyze_z_bands(self, frames: Union[str, int, List[int], np.ndarray] = 'all', threshold: float = 0.1,
                        min_length: float = 0.5, end_radius: float = 0.75, theta_phi_min: float = 0.6,
                        d_max: float = 4.0, d_min: float = 0.25) -> None:
        """
        Segment and analyze sarcomere z-bands.

        Parameters
        ----------
        frames: {'all', int, list, np.ndarray}, optional
            frames for z-band analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). Defaults to 'all'.
        threshold : float, optional
            Threshold for binarizing z-bands prior to labeling (0 - 1). Defaults to 0.1.
        min_length : float, optional
            Minimal length of z-bands; smaller z-bands are removed (in µm). Defaults to 0.5.
        end_radius : float, optional
            Radius around z-band ends to quantify orientation of ends (in µm). Defaults to 0.75.
        theta_phi_min : float, optional
            Minimal cosine of the angle between the pointed z-band vector and the connecting vector between ends of z-bands.
            Smaller values are not recognized as connections (for lateral alignment and distance analysis). Defaults to 0.25.
        d_max : float, optional
            Maximal distance between z-band ends (in µm). Z-band end pairs with larger distances are not connected
            (for lateral alignment and distance analysis). Defaults to 5.0.
        d_min : float, optional
            Minimal distance between z-band ends (in µm). Z-band end pairs with smaller distances are not connected.
            Defaults to 0.25.
        """
        assert self.sarc_obj.file_sarcomeres is not None, ("Z-band mask not found. Please run predict_z_bands first.")
        if frames == 'all':
            imgs = tifffile.imread(self.sarc_obj.file_sarcomeres)
            imgs_raw = self.read_imgs()
            list_frames = list(range(len(imgs_raw)))
        elif isinstance(frames, int) or isinstance(frames, list) or type(frames) is np.ndarray:
            imgs = tifffile.imread(self.sarc_obj.file_sarcomeres, key=frames)
            imgs_raw = self.read_imgs(frame=frames)
            if isinstance(frames, int):
                list_frames = [frames]
            else:
                list_frames = list(frames)
        else:
            raise ValueError('frames argument not valid')
        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs, 0)
            imgs_raw = np.expand_dims(imgs_raw, 0)
        n_imgs = len(imgs)

        # create empty lists
        none_lists = lambda: [None] * self.sarc_obj.metadata['frames']
        z_length, z_intensity, z_straightness, z_ratio_intensity, z_orientation = (none_lists() for _ in range(5))
        z_lat_neighbors, z_lat_alignment, z_lat_dist = (none_lists() for _ in range(3))
        z_lat_size_groups, z_lat_length_groups, z_lat_alignment_groups = (none_lists() for _ in range(3))
        z_labels, z_ends, z_lat_links, z_lat_groups = (none_lists() for _ in range(4))

        # create empty arrays
        nan_arrays = lambda: np.full(self.sarc_obj.metadata['frames'], np.nan)
        z_length_mean, z_length_std, z_length_max, z_length_sum = (nan_arrays() for _ in range(4))
        z_intensity_mean, z_intensity_std = (nan_arrays() for _ in range(2))
        z_straightness_mean, z_straightness_std = (nan_arrays() for _ in range(2))
        z_ratio_intensity, z_oop, z_avg_intensity = (nan_arrays() for _ in range(3))
        z_lat_neighbors_mean, z_lat_neighbors_std = (nan_arrays() for _ in range(2))
        z_lat_alignment_mean, z_lat_alignment_std = (nan_arrays() for _ in range(2))
        z_lat_dist_mean, z_lat_dist_std = (nan_arrays() for _ in range(2))
        z_lat_size_groups_mean, z_lat_size_groups_std = (nan_arrays() for _ in range(2))
        z_lat_length_groups_mean, z_lat_length_groups_std = (nan_arrays(), nan_arrays())
        z_lat_alignment_groups_mean, z_lat_alignment_groups_std = (nan_arrays() for _ in range(2))

        # iterate images
        print('\nStarting Z-band analysis...')
        for i, (frame_i, img_i) in tqdm(enumerate(zip(list_frames, imgs)), total=n_imgs):
            # segment z-bands
            labels_i, labels_skel_i = self.segment_z_bands(img_i)

            # analyze z-band features
            z_band_features = self._analyze_z_bands(img_i, labels_i, labels_skel_i, imgs_raw[i],
                                                    pixelsize=self.sarc_obj.metadata['pixelsize'], threshold=threshold,
                                                    min_length=min_length, end_radius=end_radius,
                                                    theta_phi_min=theta_phi_min,
                                                    d_max=d_max, d_min=d_min)

            (
                z_length_i, z_intensity_i, z_straightness_i, z_ratio_intensity_i, z_avg_intensity_i, orientation_i,
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
            z_ratio_intensity[frame_i], z_avg_intensity[frame_i], z_oop[
                frame_i] = z_ratio_intensity_i, z_avg_intensity_i, z_oop_i

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
                       'z_straightness': z_straightness, 'z_avg_intensity': z_avg_intensity, 'z_labels': z_labels,
                       'z_straightness_mean': z_straightness_mean, 'z_straightness_std': z_straightness_std,
                       'z_ratio_intensity': z_ratio_intensity, 'z_lat_neighbors': z_lat_neighbors,
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
                       'params.z_frames': list_frames, 'params.z_threshold': threshold,
                       'params.z_min_length': min_length, 'params.z_end_radius': end_radius,
                       'params.z_theta_phi_min': theta_phi_min, 'params.z_d_max': d_max, 'params.z_d_min': d_min}
        self.data.update(z_band_data)
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def analyze_sarcomere_length_orient(self, frames: Union[str, int, List[int], np.ndarray] = 'all',
                                        kernel: str = 'half_gaussian', size: float = 3.0, minor: float = 0.33,
                                        major: float = 1.0, len_lims: Tuple[float, float] = (1.45, 2.7),
                                        len_step: float = 0.05, orient_lims: Tuple[float, float] = (-90, 90),
                                        orient_step: float = 10, add_negative_center_kernel: bool = False,
                                        patch_size: int = 1024, score_threshold: float = 0.25,
                                        abs_threshold: bool = True, gating: bool = True, dilation_radius: int = 3,
                                        dtype: Union[torch.dtype, str] = 'auto', save_memory: bool = False,
                                        save_all: bool = False) -> None:
        """
        AND-gated double wavelet analysis of sarcomere structure.

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}, optional
            frames for wavelet analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). Defaults to 'all'.
        kernel : str, optional
            Filter kernel ('gaussian' for double Gaussian kernel, 'half_gaussian' for Gaussian in minor axis direction
            and binary step function in major axis direction, 'binary' for binary step function in both directions).
            Defaults to 'half_gaussian'.
        size : float, optional
            Size of wavelet filters (in µm), needs to be larger than the upper limit of len_lims. Defaults to 3.0.
        minor : float, optional
            Minor axis width, quantified by full width at half-maximum (FWHM), should match the thickness of Z-bands,
            for kernel='gaussian' and kernel='half-gaussian'. Defaults to 0.33.
        major : float, optional
            Major axis width, should match the width of Z-bands.
            Full width at half-maximum (FWHM) for kernel='gaussian' and full width for kernel='half_gaussian'.
            Defaults to 1.0.
        len_lims : tuple(float, float), optional
            Limits of lengths / wavelet distances in µm, range of sarcomere lengths. Defaults to (1.3, 2.6).
        len_step : float, optional
            Step size of sarcomere lengths in µm. Defaults to 0.05.
        orient_lims : tuple(float, float), optional
            Limits of sarcomere orientation angles in degrees. Defaults to (-90, 90).
        orient_step : float, optional
            Step size of orientation angles in degrees. Defaults to 10.
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel in the middle of the two wavelets,
            to avoid detection of two Z-bands two sarcomeres apart as sarcomere, only for kernel='gaussian'. Defaults to False.
        patch_size : int, optional
            Patch size for wavelet analysis, default is 1024 pixels. Adapt to GPU storage. Defaults to 1024.
        score_threshold : float, optional
            Threshold score for clipping of length and orientation map (if abs_threshold=False, score_threshold is
            percentile (e.g., 90) for adaptive thresholding). Defaults to 0.25.
        abs_threshold : bool, optional
            If True, absolute threshold value is applied; if False, adaptive threshold based on percentile. Defaults to True.
        gating : bool, optional
            If True, AND-gated wavelet filtering is used. If False, both wavelet filters are applied jointly. Defaults to True.
        dilation_radius : int, optional
            Radius of dilation for sarcomere area calculation, in pixels. Defaults to 3.
        dtype : torch.dtype or str, optional
            Specify torch data type (torch.float32 or torch.float16),
            'auto' chooses float16 for cuda and mps, and float32 for cpu. Defaults to 'auto'.
        save_memory : bool, optional
            Whether to save GPU memory by performing only conv2d on GPU memory, which is slower but allows processing
            larger images or larger wavelet banks. Defaults to False.
        save_all : bool, optional
            If True, the wavelet filter results (wavelet_length_i, wavelet_orientation_i, wavelet_max_score) are stored.
            If False, only the points on the midlines are stored (recommended). Defaults to False.
        """
        assert size > 1.1 * len_lims[1], (f"The size of wavelet filter {size} is too small for the maximum sarcomere "
                                          f"length {len_lims[1]}")
        assert self.sarc_obj.file_sarcomeres is not None, "Z-band mask not found. Please run predict_z_bands first."

        if frames == 'all':
            list_frames = list(range(self.sarc_obj.metadata['frames']))
            if len(list_frames) == 1:
                imgs = tifffile.imread(self.sarc_obj.file_sarcomeres)
            elif len(list_frames) > 1:
                imgs = tifffile.imread(self.sarc_obj.file_sarcomeres, key=list_frames)
        elif isinstance(frames, int) or isinstance(frames, list) or isinstance(frames, np.ndarray):
            imgs = tifffile.imread(self.sarc_obj.file_sarcomeres, key=frames)
            if isinstance(frames, int):
                list_frames = [frames]
            else:
                list_frames = list(frames)
        else:
            raise ValueError('frames argument not valid')
        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs, 0)
        n_imgs = len(imgs)

        # choose dtype depending on device
        if dtype == 'auto':
            if self.sarc_obj.device == torch.device('cpu'):
                dtype = torch.float32
            else:
                dtype = torch.float16

        # create empty arrays
        none_lists = lambda: [None] * self.sarc_obj.metadata['frames']
        nan_arrays = lambda: np.full(self.sarc_obj.metadata['frames'], np.nan)
        (points, midline_length_points, midline_id_points, sarcomere_length_points,
         sarcomere_orientation_points, max_score_points) = (none_lists() for _ in range(6))
        sarcomere_masks = np.zeros((self.sarc_obj.metadata['frames'], *self.sarc_obj.metadata['size']), dtype=bool)
        (sarcomere_length_mean, sarcomere_length_std) = (nan_arrays() for _ in range())
        sarcomere_orientation_mean, sarcomere_orientation_std = nan_arrays(), nan_arrays()
        oop, sarcomere_area, sarcomere_area_ratio, score_thresholds = (nan_arrays() for _ in range(4))
        wavelet_sarcomere_length, wavelet_sarcomere_orientation, wavelet_max_score = (none_lists() for _ in range(3))

        # create filter bank
        bank, len_range, orient_range = self.create_wavelet_bank(pixelsize=self.sarc_obj.metadata['pixelsize'],
                                                                 kernel=kernel,
                                                                 size=size, minor=minor, major=major, len_lims=len_lims,
                                                                 len_step=len_step, orient_lims=orient_lims,
                                                                 orient_step=orient_step,
                                                                 add_negative_center_kernel=add_negative_center_kernel)
        len_range_tensor = torch.from_numpy(len_range).to(self.sarc_obj.device).to(dtype=dtype)
        orient_range_tensor = torch.from_numpy(np.radians(orient_range)).to(self.sarc_obj.device).to(dtype=dtype)
        # iterate images
        print('\nStarting sarcomere length and orientation analysis...')
        for i, (frame_i, img_i) in tqdm(enumerate(zip(list_frames, imgs)), total=n_imgs):
            result_i = self.convolve_image_with_bank(img_i, bank, device=self.sarc_obj.device, gating=gating,
                                                     dtype=dtype, save_memory=save_memory, patch_size=patch_size)
            (wavelet_sarcomere_length_i, wavelet_sarcomere_orientation_i,
             wavelet_max_score_i) = self.argmax_wavelets(result_i,
                                                         len_range_tensor,
                                                         orient_range_tensor)

            # evaluate wavelet results at sarcomere midlines
            (points_i, midline_id_points_i, midline_length_points_i, sarcomere_length_points_i,
             sarcomere_orientation_points_i, max_score_points_i, midline_i,
             score_threshold_i) = self.get_points_midline(
                wavelet_sarcomere_length_i, wavelet_sarcomere_orientation_i, wavelet_max_score_i, len_range,
                score_threshold=score_threshold,
                abs_threshold=abs_threshold)

            # empty memory
            del result_i, img_i
            if save_all:
                wavelet_sarcomere_length.append(wavelet_sarcomere_length_i)
                wavelet_sarcomere_orientation.append(wavelet_sarcomere_orientation_i)
                wavelet_max_score.append(wavelet_max_score_i)
            del wavelet_sarcomere_length_i, wavelet_sarcomere_orientation_i, wavelet_max_score_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # write in list
            points[frame_i] = points_i
            midline_length_points[frame_i] = midline_length_points_i * self.sarc_obj.metadata['pixelsize'] if any(
                midline_id_points_i) else []
            midline_id_points[frame_i] = midline_id_points_i
            sarcomere_length_points[frame_i] = sarcomere_length_points_i
            sarcomere_orientation_points[frame_i] = sarcomere_orientation_points_i
            max_score_points[frame_i] = max_score_points_i
            score_thresholds[frame_i] = score_threshold_i

            # calculate mean and std of sarcomere length and orientation
            sarcomere_length_mean[frame_i], sarcomere_length_std[frame_i],  = np.mean(
                sarcomere_length_points_i), np.std(sarcomere_length_points_i)
            sarcomere_orientation_mean[frame_i], sarcomere_orientation_std[frame_i] = np.mean(
                sarcomere_orientation_points_i), np.std(sarcomere_orientation_points_i)

            # orientation order parameter
            if len(sarcomere_orientation_points_i) > 0:
                oop[frame_i], _ = Utils.analyze_orientations(sarcomere_orientation_points_i)

            # calculate sarcomere area
            if len(points_i) > 0:
                mask_i = self.sarcomere_mask(points_i * self.sarc_obj.metadata['pixelsize'],
                                             sarcomere_orientation_points_i,
                                             sarcomere_length_points_i,
                                             size=self.sarc_obj.metadata['size'],
                                             pixelsize=self.sarc_obj.metadata['pixelsize'],
                                             dilation_radius=dilation_radius)
            else:
                mask_i = np.zeros(self.sarc_obj.metadata['size'], dtype='bool')
            sarcomere_masks[frame_i] = mask_i
            sarcomere_area[frame_i] = np.sum(mask_i) * self.sarc_obj.metadata['pixelsize'] ** 2
            if 'cell_area' in self.data.keys():
                sarcomere_area_ratio[frame_i] = sarcomere_area[frame_i] / self.data['cell_area'][i]
            else:
                area = self.sarc_obj.metadata['size'][0] * self.sarc_obj.metadata['size'][1] * self.sarc_obj.metadata[
                    'pixelsize'] ** 2
                sarcomere_area_ratio[frame_i] = sarcomere_area[i] / area

        tifffile.imwrite(self.sarc_obj.file_sarcomere_mask, np.asarray(sarcomere_masks).astype('bool'))

        wavelet_dict = {'params.wavelet_size': size, 'params.wavelet_minor': minor, 'params.wavelet_major': major,
                        'params.wavelet_len_lims': len_lims, 'params.wavelet_len_step': len_step,
                        'params.orient_lims': orient_lims, 'params.orient_step': orient_step,
                        'params.kernel': kernel,
                        'params.wavelet_frames': list_frames, 'params.len_range': len_range[1:-2],
                        'params.orient_range': orient_range, 'wavelet_sarcomere_length': wavelet_sarcomere_length,
                        'wavelet_sarcomere_orientation': wavelet_sarcomere_orientation,
                        'wavelet_max_score': wavelet_max_score,
                        'points': points, 'sarcomere_length_points': sarcomere_length_points,
                        'midline_length_points': midline_length_points, 'midline_id_points': midline_id_points,
                        'wavelet_bank': bank if save_all else None,
                        'sarcomere_orientation_points': sarcomere_orientation_points,
                        'max_score_points': max_score_points,
                        'sarcomere_area': sarcomere_area, 'sarcomere_area_ratio': sarcomere_area_ratio,
                        'sarcomere_length_mean': sarcomere_length_mean,
                        'sarcomere_length_std': sarcomere_length_std,
                        'sarcomere_orientation_mean': sarcomere_orientation_mean,
                        'sarcomere_orientation_std': sarcomere_orientation_std,
                        'sarcomere_oop': oop,
                        'params.score_threshold': score_thresholds, 'params.abs_threshold': abs_threshold,
                        'params.sarcomere_area_closing_radius': dilation_radius}
        self.data.update(wavelet_dict)
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def optimize_wavelet_minor_axis(self, frame: int = 0, n_sample: int = 50):
        """
        Find the optimal wavelet minor axis, in full width at half maximum (FWHM) units, that maximizes the number of
        sarcomeres identified for a given sample by determining width of Z-bands by fitting Gaussian to sample
        of sarcomere vectors. Before running this function, it is necessary to run analyze_sarcomere_length_orient with
        a prior set of parameters.

        Parameters
        ----------
        frame : int, optional
            The specific frame to analyze. Default is 0.
        n_sample : int, optional
            Number of random samples to use for optimization. Default is 50.

        Returns
        -------
        float
            The median sigma value from the Gaussian fits to a sample of sarcomere vectors.
        """
        assert 'points' in self.data.keys(), ('Sarcomere length and orientation not yet analyzed. '
                                              'Run analyze_sarcomere_length_orient first.')
        assert frame in self.data['params.z_frames'], f'Z-bands of frame {frame} not yet analyzed.'
        assert frame in self.data['params.wavelet_frames'], f'Sarcomere vectors of frame {frame} not yet analyzed.'

        z_bands_t = tifffile.imread(self.sarc_obj.file_sarcomeres, key=frame)
        points_t = self.data['points'][frame]
        sarcomere_orientation_points_t = self.data['sarcomere_orientation_points'][frame]
        sarcomere_length_points_t = self.data['sarcomere_length_points'][frame] / self.sarc_obj.metadata[
            'pixelsize']

        # Calculate orientation vectors using trigonometry
        orientation_vectors_t = np.asarray([-np.sin(sarcomere_orientation_points_t),
                                            np.cos(sarcomere_orientation_points_t)])

        # Calculate the ends of the vectors based on their orientation and length
        starts, ends = points_t, points_t + orientation_vectors_t * sarcomere_length_points_t

        # randomly select N lines
        idxs_random = np.random.randint(0, starts.shape[1], size=n_sample)
        starts, ends = starts[:, idxs_random], ends[:, idxs_random]

        def gaussian(x, A, mu, sigma):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        def fit_gaussian(x_data, y_data):
            # Initial guesses for fitting parameters
            A_guess = np.max(y_data)
            mu_guess = x_data[np.argmax(y_data)]
            sigma_guess = np.std(x_data) / 2

            # Perform the Gaussian fit
            params, _ = curve_fit(gaussian, x_data, y_data, p0=[A_guess, mu_guess, sigma_guess])

            return params

        # extract profiles perpendicular to Z-bands to determine Z-band width
        sigmas = []
        for start_i, end_i in zip(starts.T, ends.T):
            profile_i = profile_line(z_bands_t, start_i, end_i, linewidth=5)
            x_i = np.arange(profile_i.shape[0]) * self.sarc_obj.metadata['pixelsize']
            try:
                params = fit_gaussian(x_i, profile_i)
                sigmas.append(params[2])
            except:
                pass
        return np.median(sigmas) * 2.355  # convert sigma to FWHM (full width at half maximum)

    def analyze_myofibrils(self, frames: Optional[Union[str, int, List[int], np.ndarray]] = None,
                           n_seeds: int = 2000, persistence: int = 3, threshold_distance: float = 0.3,
                           n_min: int = 5) -> None:
        """
        Estimate myofibril lines by line growth algorithm and analyze length and curvature.

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}, optional
            frames for myofibril analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). If None, frames from wavelet analysis are used. Defaults to None.
        n_seeds : int, optional
            Number of random seeds for line growth. Defaults to 1000.
        persistence : int, optional
            Persistence of line (average points length and orientation for prior estimation), needs to be > 0. Defaults to 3.
        threshold_distance : float, optional
            Maximal distance for nearest neighbor estimation (in micrometers). Defaults to 0.3.
        n_min : int, optional
            Minimal number of sarcomere line segments per line. Shorter lines are removed. Defaults to 5.
        """
        assert 'points' in self.data.keys(), ('Sarcomere length and orientation not yet analyzed. '
                                              'Run analyze_sarcomere_length_orient first.')
        if frames is not None:
            if frames == 'all':
                frames = list(range(self.sarc_obj.metadata['frames']))
            if isinstance(frames, int):
                frames = [frames]
            assert set(frames).issubset(
                self.data['params.wavelet_frames']), f'Run analyze_sarcomere_length_orient first for frames {frames}.'
        elif frames is None:
            if 'params.wavelet_frames' in self.data.keys():
                frames = self.data['params.wavelet_frames']
            else:
                raise ValueError('To use frames from wavelet analysis, run wavelet analysis first!')

        if frames == 'all':
            n_imgs = self.sarc_obj.metadata['frames']
            list_frames = list(range(n_imgs))
        elif isinstance(frames, int):
            n_imgs = 1
            list_frames = [frames]
        elif isinstance(frames, list) or type(frames) is np.ndarray:
            n_imgs = len(frames)
            list_frames = list(frames)
        else:
            raise ValueError('Selection of frames not valid!')

        points = [self.data['points'][frame] for frame in list_frames]
        sarcomere_length_points = [self.data['sarcomere_length_points'][frame] for frame in list_frames]
        sarcomere_orientation_points = [self.data['sarcomere_orientation_points'][frame] for frame in list_frames]
        midline_length_points = [self.data['midline_length_points'][frame] for frame in list_frames]
        max_score_points = [self.data['max_score_points'][frame] for frame in list_frames]

        # create empty arrays
        none_lists = lambda: [None] * self.sarc_obj.metadata['frames']
        nan_arrays = lambda: np.full(self.sarc_obj.metadata['frames'], np.nan)
        length_mean, length_std, length_max = (nan_arrays() for _ in range(3))
        msc_mean, msc_std = (nan_arrays() for _ in range(2))
        myof_lines, lengths, msc = (none_lists() for _ in range(3))

        # iterate frames
        print('\nStarting myofibril line analysis...')
        for i, (frame_i, points_i, sarcomere_length_points_i, sarcomere_orientation_points_i, max_score_points_i,
                midline_length_points_i) in enumerate(
            tqdm(
                zip(list_frames, points, sarcomere_length_points, sarcomere_orientation_points, max_score_points,
                    midline_length_points),
                total=len(points))):
            if len(np.asarray(points_i).T) > 0:
                line_data_i = self.line_growth(points_i, sarcomere_length_points_i, sarcomere_orientation_points_i,
                                               max_score_points_i, midline_length_points_t=midline_length_points_i,
                                               pixelsize=self.sarc_obj.metadata['pixelsize'], n_seeds=n_seeds,
                                               persistence=persistence, threshold_distance=threshold_distance,
                                               n_min=n_min)
                lines_i = line_data_i['lines']

                # line lengths and mean squared curvature (msc)
                lengths_i = line_data_i['line_features']['length_lines']
                msc_i = line_data_i['line_features']['msc_lines']
                if len(lengths_i) > 0:
                    length_mean[frame_i], length_std[frame_i], length_max[frame_i] = np.mean(
                        lengths_i), np.std(lengths_i), np.max(lengths_i)
                    msc_mean[frame_i], msc_std[frame_i] = np.mean(msc_i), np.std(
                        msc_i)
                myof_lines[frame_i] = lines_i
                lengths[frame_i] = lengths_i
                msc[frame_i] = msc_i

        # update structure dictionary
        myofibril_data = {'myof_length_mean': length_mean,
                          'myof_length_std': length_std, 'myof_lines': myof_lines,
                          'myof_length_max': length_max, 'myof_length': lengths,
                          'myof_msc': msc, 'myof_msc_mean': msc_mean,
                          'myof_msc_std': msc_std, 'params.n_seeds': n_seeds, 'params.persistence': persistence,
                          'params.threshold_distance': threshold_distance, 'params.myof_frames': list_frames}
        self.data.update(myofibril_data)
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def analyze_sarcomere_domains(self, frames: Optional[Union[str, int, List[int], np.ndarray]] = None,
                                  dist_threshold_ends: float = 0.5, dist_threshold_midline_points: float = 0.5,
                                  louvain_resolution: float = 0.06, louvain_seed: int = 2, area_min: float = 20.0,
                                  dilation_radius: int = 3) -> None:
        """
        Cluster sarcomeres into domains based on their spatial and orientational properties using the Louvain method
        for community detection.

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}, optional
            frames for domain analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames). If None, frames from wavelet analysis are used. Defaults to None.
        dist_threshold_ends : float, optional
            Maximal distance threshold for connecting/creating network edge for adjacent sarcomere vector ends.
            Only the ends with the smallest distance are connected. Defaults to 0.5.
        dist_threshold_midline_points : float, optional
            Maximal distance threshold for connecting/creating network edge for midline points of the same midline.
            All points within this distance are connected. Defaults to 0.5.
        louvain_resolution : float, optional
            Control parameter for domain size. If resolution is small, the algorithm favors larger domains.
            Greater resolution favors smaller domains. Defaults to 0.05.
        louvain_seed : int, optional
            Random seed for Louvain algorithm, to ensure reproducibility. Defaults to 2.
        area_min : float, optional
            Minimal area of domains/clusters (in µm^2). Defaults to 50.0.
        dilation_radius : int, optional
            Dilation radius for refining domain area masks. Defaults to 3.
        """
        assert 'points' in self.data.keys(), ('Sarcomere length and orientation not yet analyzed. '
                                              'Run analyze_sarcomere_length_orient first.')
        if frames is not None:
            if frames == 'all':
                frames = list(range(self.sarc_obj.metadata['frames']))
            if isinstance(frames, int):
                frames = [frames]
            assert set(frames).issubset(
                self.data['params.wavelet_frames']), f'Run analyze_sarcomere_length_orient first for frames {frames}.'
        elif frames is None:
            if 'params.wavelet_frames' in self.data.keys():
                frames = self.data['params.wavelet_frames']
            else:
                raise ValueError('To use frames from wavelet analysis, run wavelet analysis first!')

        if frames == 'all':
            n_imgs = self.sarc_obj.metadata['frames']
            list_frames = list(range(n_imgs))
        elif isinstance(frames, int):
            n_imgs = 1
            list_frames = [frames]
        elif isinstance(frames, list) or type(frames) is np.ndarray:
            n_imgs = len(frames)
            list_frames = list(frames)
        else:
            raise ValueError('Selection of frames not valid!')

        points = [np.asarray(self.data['points'][t]) * self.sarc_obj.metadata['pixelsize'] for t in list_frames]
        sarcomere_length_points = [np.asarray(self.data['sarcomere_length_points'][t]) for t in list_frames]
        sarcomere_orientation_points = [np.asarray(self.data['sarcomere_orientation_points'][t]) for t in list_frames]
        max_score_points = [np.asarray(self.data['max_score_points'][t]) for t in list_frames]
        midline_id_points = [np.asarray(self.data['midline_id_points'][t]) for t in list_frames]

        # create empty arrays
        none_lists = lambda: [None] * self.sarc_obj.metadata['frames']
        nan_arrays = lambda: np.full(self.sarc_obj.metadata['frames'], np.nan)
        n_domains, domain_area_mean, domain_area_std = (nan_arrays() for _ in range(3))
        domain_slen_mean, domain_slen_std = (nan_arrays() for _ in range(2))
        domain_oop_mean, domain_oop_std = (nan_arrays() for _ in range(2))

        (domains, domain_area, domain_slen, domain_slen_std,
         domain_oop, domain_orientation, domain_mask) = (none_lists() for _ in range(7))

        # iterate frames
        print('\nStarting sarcomere domain analysis...')
        for i, (frame_i, points_i, sarcomere_length_points_i, sarcomere_orientation_points_i,
                max_score_points_t, midline_id_points_i) in enumerate(
            tqdm(
                zip(list_frames, points, sarcomere_length_points, sarcomere_orientation_points, max_score_points, midline_id_points),
                total=len(points))):
            cluster_data_t = self.cluster_sarcomeres(points_i, sarcomere_length_points_i,
                                                     sarcomere_orientation_points_i,
                                                     midline_id_points_i, pixelsize=self.sarc_obj.metadata['pixelsize'],
                                                     size=self.sarc_obj.metadata['size'],
                                                     dist_threshold_ends=dist_threshold_ends,
                                                     dist_threshold_midline_points=dist_threshold_midline_points,
                                                     louvain_resolution=louvain_resolution, louvain_seed=louvain_seed,
                                                     area_min=area_min, dilation_radius=dilation_radius)
            (n_domains[frame_i], domains[frame_i], domain_area[frame_i], domain_slen[frame_i], domain_slen_std[frame_i],
             domain_oop[frame_i], domain_orientation[frame_i], domain_mask_i) = cluster_data_t

            # write single domain / cluster in lists
            domain_mask[frame_i] = sparse.coo_matrix(domain_mask_i)

            # calculate mean and std of domains
            domain_area_mean[frame_i], domain_area_std[frame_i] = np.mean(domain_area[frame_i]), np.std(domain_area[frame_i])
            domain_slen_mean[frame_i], domain_slen_std[frame_i] = (np.mean(domain_slen[frame_i]), np.std(domain_slen[frame_i]))
            domain_oop_mean[frame_i], domain_oop_std[frame_i] = (np.mean(domain_oop[frame_i]), np.std(domain_oop[frame_i]))


        # update structure dictionary
        domain_data = {'n_domains': n_domains, 'domains': domains,
                       'domain_area': domain_area, 'domain_area_mean': domain_area_mean,
                       'domain_area_std': domain_area_std,
                       'domain_slen': domain_slen, 'domain_slen_mean': domain_slen_mean,
                       'domain_slen_std': domain_slen_std,
                       'domain_oop': domain_oop, 'domain_oop_mean': domain_oop_mean,
                       'domain_oop_std': domain_oop_std,
                       'domain_orientation': domain_orientation, 'domain_mask': domain_mask,
                       'params.domain_frames': list_frames,
                       'params.dist_threshold_ends': dist_threshold_ends,
                       'params.dist_threshold_midline_points': dist_threshold_midline_points,
                       'params.louvain_resolution': louvain_resolution,
                       'params.domain_area_min': area_min}

        self.data.update(domain_data)
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def _grow_lois(self, frame: int = 0, n_seeds: int = 500, score_threshold: Optional[float] = None,
                   persistence: int = 2, threshold_distance: float = 0.5, random_seed: Optional[int] = None) -> None:
        """
        Find LOIs (lines of interest) using a line growth algorithm. The parameters **lims can be used to filter LOIs.

        Parameters
        ----------
        frame : int, optional
            Frame to select frame. Selects i-th frame of frames specified in wavelet analysis. Defaults to 0.
        n_seeds : int, optional
            Number of random seeds for line growth. Defaults to 500.
        score_threshold : float, optional
            Score threshold for random seeds (needs to be <= score_threshold from get_points_midline). If None, automated
            score_threshold from wavelet analysis is used. Defaults to None.
        persistence : int, optional
            Persistence of line (average points length and orientation for prior estimation). Defaults to 2.
        threshold_distance : float, optional
            Maximal distance for nearest neighbor estimation. Defaults to 0.5.
        random_seed : int, optional
            Random seed for reproducibility. Defaults to None.
        """
        if score_threshold is None:
            if 'params.score_threshold' in self.data.keys():
                if len(self.data['params.score_threshold']) > 1:
                    score_threshold = self.data['params.score_threshold'][frame]
                else:
                    score_threshold = self.data['params.score_threshold']
            else:
                raise ValueError('To use score_threshold from wavelet analysis, run wavelet analysis first!')
        # select midline point data at frame
        (points, sarcomere_length_points,
         sarcomere_orientation_points, max_score_points, midline_length_points) = self.data['points'][frame], \
            self.data['sarcomere_length_points'][frame], \
            self.data['sarcomere_orientation_points'][frame], \
            self.data['max_score_points'][frame], \
            self.data['midline_length_points'][frame]
        loi_data = self.line_growth(points, sarcomere_length_points, sarcomere_orientation_points, max_score_points,
                                    midline_length_points, self.sarc_obj.metadata['pixelsize'], n_seeds=n_seeds,
                                    random_seed=random_seed, persistence=persistence,
                                    threshold_distance=threshold_distance)
        self.data['loi_data'] = loi_data
        lois_points = [self.data['points'][frame].T[loi_i] for loi_i in self.data['loi_data']['lines']]
        self.data['loi_data']['lines_points'] = lois_points
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def _filter_lois(self, number_lims: Tuple[int, int] = (10, 100), length_lims: Tuple[float, float] = (0, 200),
                     sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
                     sarcomere_std_length_lims: Tuple[float, float] = (0, 0.4),
                     msc_lims: Tuple[float, float] = (0, 1), midline_mean_length_lims: Tuple[float, float] = (2, 20),
                     midline_std_length_lims: Tuple[float, float] = (0, 5),
                     midline_min_length_lims: Tuple[float, float] = (2, 20),
                     max_orient_change: float = 30.0) -> None:
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
            Limits for standard deviation of sarcomere lengths in LOI (min, max). Defaults to (0, 0.4).
        msc_lims : tuple of float, optional
            Limits for LOI mean squared curvature (MSC) (min, max). Defaults to (0, 1).
        midline_mean_length_lims : tuple of float, optional
            Limits for mean length of the midline in LOI (min, max). Defaults to (2, 20).
        midline_std_length_lims : tuple of float, optional
            Limits for standard deviation of the midline length in LOI (min, max). Defaults to (0, 5).
        midline_min_length_lims : tuple of float, optional
            Limits for minimum length of the midline in LOI (min, max). Defaults to (2, 20).
        max_orient_change : float, optional
            Maximum orientation change allowed. Defaults to 30.0.
        """
        # Retrieve LOIs and their features from the structure dict
        lois, loi_features = self.data['loi_data']['lines'], self.data['loi_data']['line_features']
        lois_points = self.data['loi_data']['lines_points']

        # Apply filters based on the provided limits
        is_good = (
                (loi_features['n_points_lines'] >= number_lims[0]) & (loi_features['n_points_lines'] < number_lims[1]) &
                (loi_features['length_lines'] >= length_lims[0]) & (loi_features['length_lines'] < length_lims[1]) &
                (loi_features['sarcomere_mean_length_lines'] >= sarcomere_mean_length_lims[0]) &
                (loi_features['sarcomere_mean_length_lines'] < sarcomere_mean_length_lims[1]) &
                (loi_features['sarcomere_std_length_lines'] >= sarcomere_std_length_lims[0]) &
                (loi_features['sarcomere_std_length_lines'] < sarcomere_std_length_lims[1]) &
                (loi_features['msc_lines'] >= msc_lims[0]) & (loi_features['msc_lines'] < msc_lims[1]) &
                (loi_features['midline_mean_length_lines'] >= midline_mean_length_lims[0]) &
                (loi_features['midline_mean_length_lines'] < midline_mean_length_lims[1]) &
                (loi_features['midline_std_length_lines'] >= midline_std_length_lims[0]) &
                (loi_features['midline_std_length_lines'] < midline_std_length_lims[1]) &
                (loi_features['midline_min_length_lines'] >= midline_min_length_lims[0]) &
                (loi_features['midline_min_length_lines'] < midline_min_length_lims[1]) &
                (loi_features['max_orient_change_lines'] < np.radians(max_orient_change))
        )

        # remove bad lines
        self.data['loi_data']['lines'] = [loi for i, loi in enumerate(lois) if is_good[i]]
        self.data['loi_data']['lines_points'] = [points for i, points in enumerate(lois_points) if is_good[i]]
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
        lines_points = self.data['loi_data']['lines_points']

        # hausdorff distance between LOIss
        hausdorff_dist_matrix = np.zeros((len(lines_points), len(lines_points)))
        for i, loi_i in enumerate(lines_points):
            for j, loi_j in enumerate(lines_points):
                if symmetry_mode == 'min':
                    hausdorff_dist_matrix[i, j] = min(directed_hausdorff(loi_i, loi_j)[0],
                                                      directed_hausdorff(loi_j, loi_i)[0])
                if symmetry_mode == 'max':
                    hausdorff_dist_matrix[i, j] = max(directed_hausdorff(loi_i, loi_j)[0],
                                                      directed_hausdorff(loi_j, loi_i)[0])

        self.data['loi_data']['hausdorff_dist_matrix'] = hausdorff_dist_matrix
        if self.sarc_obj.auto_save:
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
        if len(self.data['loi_data']['lines_points']) == 0:
            self.data['loi_data']['line_cluster'] = []
            self.data['loi_data']['n_lines_clusters'] = 0
        elif len(self.data['loi_data']['lines_points']) == 1:
            self.data['loi_data']['line_cluster'] = [[0]]
            self.data['loi_data']['n_lines_clusters'] = 1
        else:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold_lois,
                                                 metric='precomputed',
                                                 linkage=linkage).fit(
                self.data['loi_data']['hausdorff_dist_matrix'])
            self.data['loi_data']['line_cluster'] = clustering.labels_
            self.data['loi_data']['n_lines_clusters'] = len(np.unique(clustering.labels_))
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def _fit_straight_line(self, add_length=2, n_lois=None):
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
        add_length = add_length / self.sarc_obj.metadata['pixelsize']
        for label_i in range(self.data['loi_data']['n_lines_clusters']):
            points_cluster_i = []
            for k in np.where(self.data['loi_data']['line_cluster'] == label_i)[0]:
                points_cluster_i.append(self.data['loi_data']['lines_points'][k])
            points_clusters.append(np.concatenate(points_cluster_i).T)
            p_i, pcov_i = curve_fit(linear, points_clusters[label_i][1], points_clusters[label_i][0])
            x_range_i = np.linspace(np.min(points_clusters[label_i][1]) - add_length / np.sqrt(1 + p_i[0] ** 2),
                                    np.max(points_clusters[label_i][1]) + add_length / np.sqrt(1 + p_i[0] ** 2), num=2)
            y_i = linear(x_range_i, p_i[0], p_i[1])
            len_i = np.sqrt(np.diff(x_range_i) ** 2 + np.diff(y_i) ** 2)
            x_range_i, y_i = np.round(x_range_i, 1), np.round(y_i, 1)
            loi_lines.append(np.asarray((x_range_i, y_i)).T)
            len_loi_lines.append(len_i)

        len_loi_lines = np.asarray(len_loi_lines).flatten()
        loi_lines = np.asarray(loi_lines)

        # sort lines by length
        length_idxs = len_loi_lines.argsort()
        loi_lines = loi_lines[length_idxs[::-1]][:n_lois]
        len_loi_lines = len_loi_lines[length_idxs[::-1]][:n_lois]

        self.data['loi_data']['loi_lines'] = np.asarray(loi_lines)
        self.data['loi_data']['len_loi_lines'] = np.asarray(len_loi_lines)
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def _longest_in_cluster(self, n_lois):
        lines = self.data['loi_data']['lines']
        points = self.data['points'][0][::-1]
        lines_cluster = np.asarray(self.data['loi_data']['line_cluster'])
        longest_lines = []
        for label_i in range(self.data['loi_data']['n_lines_clusters']):
            lines_cluster_i = [line_j for j, line_j in enumerate(lines) if lines_cluster[j] == label_i]
            points_lines_cluster_i = [points[:, line_j] for j, line_j in enumerate(lines) if
                                      lines_cluster[j] == label_i]
            length_lines_cluster_i = [len(line_j) for line_j in lines_cluster_i]
            longest_line = points_lines_cluster_i[np.argmax(length_lines_cluster_i)]
            longest_lines.append(longest_line)
        # get n longest lines
        sorted_by_length = sorted(longest_lines, key=lambda x: len(x[1].T), reverse=True)
        if len(longest_lines) < n_lois:
            print(f'Only {len(longest_lines)}<{n_lois} clusters identified.')
        loi_lines = sorted_by_length[:n_lois]
        loi_lines = [line_i.T for line_i in loi_lines]
        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = [len(line_i.T) for line_i in loi_lines]
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def _random_from_cluster(self, n_lois):
        lines = self.data['loi_data']['lines']
        points = self.data['points'][0][::-1]
        lines_cluster = np.asarray(self.data['loi_data']['line_cluster'])
        random_lines = []
        for label_i in range(self.data['loi_data']['n_lines_clusters']):
            lines_cluster_i = [line_j for j, line_j in enumerate(lines) if lines_cluster[j] == label_i]
            points_lines_cluster_i = [points[:, line_j] for j, line_j in enumerate(lines) if
                                      lines_cluster[j] == label_i]
            random_line = random.choice(points_lines_cluster_i)
            random_lines.append(random_line)
        # select clusters randomly
        random_lines = random.sample(random_lines, n_lois)
        loi_lines = [line_i.T for line_i in random_lines]
        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = [len(line_i.T) for line_i in loi_lines]
        if self.sarc_obj.auto_save:
            self.store_structure_data()

    def _random_lois(self, n_lois):
        lines = self.data['loi_data']['lines']
        points = self.data['points'][0][::-1]
        loi_lines = random.sample(lines, n_lois)
        loi_lines = [points[:, line_i].T for line_i in loi_lines]
        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = [len(line_i.T) for line_i in loi_lines]
        if self.sarc_obj.auto_save:
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
        imgs_sarcomeres = tifffile.imread(self.sarc_obj.file_sarcomeres)
        profiles = self.kymograph_movie(imgs_sarcomeres, line, order=order,
                                        linewidth=int(linewidth / self.sarc_obj.metadata['pixelsize']))
        profiles = np.asarray(profiles)
        if export_raw:
            imgs_raw = tifffile.imread(self.sarc_obj.filename)
            profiles_raw = self.kymograph_movie(imgs_raw, line, order=order,
                                                linewidth=int(linewidth / self.sarc_obj.metadata['pixelsize']))
        else:
            profiles_raw = None

        # length of line
        def __calculate_segmented_line_length(line):
            diffs = np.diff(line, axis=0)
            lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            return np.sum(lengths)

        length = __calculate_segmented_line_length(line) * self.sarc_obj.metadata['pixelsize']
        loi_data = {'profiles': profiles, 'profiles_raw': profiles_raw,
                    'line': line, 'linewidth': linewidth, 'length': length}
        for key, value in loi_data.items():
            loi_data[key] = np.asarray(value)
        save_name = os.path.join(self.sarc_obj.folder,
                                 f'{line[0][0]}_{line[0][1]}_{line[-1][0]}_{line[-1][1]}_{linewidth}_loi.json')
        IOUtils.json_serialize(loi_data, save_name)

    def detect_lois(self, frame: int = 0, n_lois: int = 4, n_seeds: int = 200, persistence: int = 2,
                    threshold_distance: float = 0.3, score_threshold: Optional[float] = None,
                    mode: str = 'longest_in_cluster', random_seed: Optional[int] = None,
                    number_lims: Tuple[int, int] = (10, 50), length_lims: Tuple[float, float] = (0, 200),
                    sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
                    sarcomere_std_length_lims: Tuple[float, float] = (0, 1),
                    msc_lims: Tuple[float, float] = (0, 1), max_orient_change: float = 30,
                    midline_mean_length_lims: Tuple[float, float] = (0, 20),
                    midline_std_length_lims: Tuple[float, float] = (0, 5),
                    midline_min_length_lims: Tuple[float, float] = (0, 20), distance_threshold_lois: float = 40,
                    linkage: str = 'single', linewidth: float = 0.65, order: int = 0, export_raw: bool = False) -> None:
        """
        Detects Regions of Interest (LOIs) for tracking sarcomere Z-band motion and creates kymographs.

        This method integrates several steps: growing LOIs based on seed points, filtering LOIs based on
        specified criteria, clustering LOIs, fitting lines to LOI clusters, and extracting intensity profiles
        to generate kymographs.

        Parameters
        ----------
        frame : int
            The index of the frame to select for analysis.
        n_lois : int
            Number of LOIs.
        n_seeds : int
            Number of seed points for initiating LOI growth.
        persistence : int
            Persistence parameter influencing line growth direction and termination.
        threshold_distance : float
            Maximum distance for nearest neighbor estimation during line growth.
        score_threshold : float, optional
            Minimum score threshold for seed points. Uses automated threshold if None.
        mode : str
            Mode for selecting LOIs from identified clusters.
            - 'fit_straight_line' fits a straight line to all points in the cluster.
            - 'longest_in_cluster' selects the longest line of each cluster, also allowing curved LOIs.
            - 'random_from_cluster' selects a random line from each cluster, also allowing curved LOIs.
            - 'random_line' selects a set of random lines that fulfil the filtering criteria.
        random_seed : int, optional
            Random seed for selection of random starting points for line growth algorithm, for reproducible outcomes.
            If None, no random seed is set, and outcomes in every run will differ.
        number_lims : tuple of int
            Limits for the number of sarcomeres within an LOI (min, max).
        length_lims : tuple of float
            Length limits for LOIs (in µm) (min, max).
        sarcomere_mean_length_lims : tuple of float
            Limits for the mean length of sarcomeres within an LOI (min, max).
        sarcomere_std_length_lims : tuple of float
            Limits for the standard deviation of sarcomere lengths within an LOI (min, max).
        msc_lims : tuple of float
            Limits for the mean squared curvature (MSC) of LOIs (min, max).
        max_orient_change : float
            Maximal change of orientation between adjacent line segments, in degrees.
        midline_mean_length_lims : tuple of float
            Limits for the mean length of the midline of points in LOI (min, max).
        midline_std_length_lims : tuple of float
            Limits for the standard deviation of the midline length of points in LOI (min, max).
        midline_min_length_lims : tuple of float
            Limits for the minimum length of the midline of points in LOI (min, max).
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
        assert 'points' in self.data.keys(), ('Sarcomere length and orientation not yet analyzed. '
                                              'Run analyze_sarcomere_length_orient first.')
        assert frame in self.data['params.wavelet_frames'], f'Sarcomere vectors of frame {frame} not yet analyzed.'

        # Grow LOIs based on seed points and specified parameters
        self._grow_lois(frame=frame, n_seeds=n_seeds, persistence=persistence,
                        threshold_distance=threshold_distance, score_threshold=score_threshold,
                        random_seed=random_seed)
        # Filter LOIs based on geometric and morphological criteria
        self._filter_lois(number_lims=number_lims, length_lims=length_lims,
                          sarcomere_mean_length_lims=sarcomere_mean_length_lims,
                          sarcomere_std_length_lims=sarcomere_std_length_lims, msc_lims=msc_lims,
                          midline_mean_length_lims=midline_mean_length_lims,
                          midline_std_length_lims=midline_std_length_lims,
                          midline_min_length_lims=midline_min_length_lims,
                          max_orient_change=max_orient_change)
        if mode == 'fit_straight_line' or mode == 'longest_in_cluster' or mode == 'random_from_cluster':
            # Calculate Hausdorff distance between LOIs and perform clustering
            self._hausdorff_distance_lois()
            self._cluster_lois(distance_threshold_lois=distance_threshold_lois, linkage=linkage)
            # Fit lines to LOIs clusters and select LOIs for analysis
            if mode == 'fit_straight_line':
                self._fit_straight_line(add_length=2, n_lois=n_lois)
            elif mode == 'longest_in_cluster':
                self._longest_in_cluster(n_lois=n_lois)
            elif mode == 'random_from_cluster':
                self._random_from_cluster(n_lois=n_lois)
        elif mode == 'random_line':
            self._random_lois(n_lois=n_lois)
        else:
            raise ValueError(f'mode {mode} not valid.')

        # extract intensity kymographs profiles and save LOI files
        for line_i in self.data['loi_data']['loi_lines']:
            self.create_loi_data(line_i, linewidth=linewidth, order=order, export_raw=export_raw)

    def delete_lois(self):
        """
        Delete all LOIs
        """
        _ = self.data.pop('loi_data', 'No existing LOIs found.')
        loi_files = glob.glob(self.sarc_obj.folder + '/*loi.json')
        for loi_file in loi_files:
            os.remove(loi_file)

    def full_analysis_structure(self, frames='all', save_all=False):
        """
        Analyze sarcomere structure with default parameters at specified time points

        Parameters
        ----------
        frames : {'all', int, list, np.ndarray}
            frames for analysis ('all' for all frames, int for a single frame, list or ndarray for
            selected frames).
        save_all : bool
            If True, all intermediary data is saved. Can take up large storage, and is only recommended for visualizing
            data.
        """
        self.analyze_z_bands(frames=frames)
        self.analyze_sarcomere_length_orient(frames=frames, save_all=save_all)
        self.analyze_myofibrils(frames=frames)
        self.analyze_sarcomere_domains(frames=frames)
        if not self.sarc_obj.auto_save:
            self.store_structure_data()

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
        image_thres = image.copy().astype('uint16')
        image_thres[image >= 255 * threshold] = 255
        image_thres[image < 255 * threshold] = 0
        image_skel = morphology.skeletonize(image_thres / 255)
        image_skel_plot = image_skel.copy().astype('float32')
        image_skel_plot[image_skel_plot == 0] = np.nan
        labels = label(image_thres)
        labels_skel = image_skel * labels
        return labels, labels_skel

    @staticmethod
    def _analyze_z_bands(image_unet: np.ndarray, labels: np.ndarray, labels_skel: np.ndarray, image_raw: np.ndarray,
                         pixelsize: float, min_length: float = 1.0, threshold: float = 0.1, end_radius: float = 0.75,
                         theta_phi_min: float = 0.25, d_max: float = 5.0, d_min: float = 0.25) -> Tuple:
        """
        Analyzes segmented z-bands in a single frame, extracting metrics such as length, intensity, orientation,
        straightness, lateral distance, alignment, number of lateral neighbors per z-band, and characteristics of
        groups of lateral z-bands (length, alignment, size).

        Parameters
        ----------
        image_unet : np.ndarray
            The segmented image of z-bands.
        labels : np.ndarray
            The labeled image of z-bands.
        labels_skel : np.ndarray
            The skeletonized labels of z-bands.
        image_raw : np.ndarray
            The raw image.
        pixelsize : float
            The size of pixels in the image.
        min_length : float, optional
            The minimum length threshold for z-bands. Default is 1.0.
        threshold : float, optional
            The threshold value for intensity. Default is 0.1.
        end_radius : float, optional
            The radius of z-band ends. Default is 0.75.
        theta_phi_min : float, optional
            The minimum value for theta and phi. Default is 0.25.
        d_max : float, optional
            The maximum distance between z-band ends. Default is 5.0 µm. Larger distances are set to np.nan.
        d_min : float, optional
            The minimum distance between z-band ends. Default is 0.25 µm. Smaller distances are set to np.nan.

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

        # analyze z-band labels
        props = regionprops_table(labels, intensity_image=image_raw, properties=['label', 'area', 'convex_area',
                                                                                 'mean_intensity', 'orientation',
                                                                                 'image', 'bbox'])
        # z-band length
        length = length[length >= min_length]

        # straightness of z-bands (area/convex_hull)
        straightness = props['area'] / props['convex_area']

        # fluorescence intensity
        intensity = props['mean_intensity']

        # ratio sum(sarcomere intensity) to sum(background intensity)
        ratio_intensity, avg_intensity = Structure.intensity_sarcomeres(image_unet, image_raw,
                                                                        pixelsize=pixelsize,
                                                                        threshold=threshold)

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
            end_radius_px = int(round(end_radius / pixelsize, 0))

            for i, img_i in enumerate(props['image']):
                img_i = np.pad(props['image'][i], (end_radius_px, end_radius_px))

                # skeletonize
                skel_i = skeletonize(img_i)

                # detect line ends
                def line_end_filter(d):
                    return (d[4] == 1) and np.sum(d) == 2

                z_ends_i = ndimage.generic_filter(skel_i, line_end_filter, (3, 3))
                z_ends_i = np.asarray(np.where(z_ends_i == 1))
                if len(z_ends_i.T) == 2:
                    if z_ends_i[1, 0] > z_ends_i[1, 1]:
                        z_ends_i = z_ends_i[:, ::-1]
                    _z_ends_i = z_ends_i.copy()
                    z_ends_i[0] += props['bbox-0'][i] - end_radius_px
                    z_ends_i[1] += props['bbox-1'][i] - end_radius_px
                    z_ends[i] = z_ends_i.T * pixelsize

                    # orientation (pointing direction of line ends)
                    # binary radial mask around ends to determine orientations of ends
                    mask_i_1 = np.zeros_like(skel_i, dtype='uint8')
                    mask_i_2 = np.zeros_like(skel_i, dtype='uint8')
                    # end 1
                    rr, cc = draw_disk((_z_ends_i[0, 0], _z_ends_i[1, 0]), end_radius_px)
                    mask_i_1[rr, cc] = 1
                    # end 2
                    rr, cc = draw_disk((_z_ends_i[0, 1], _z_ends_i[1, 1]), end_radius_px)
                    mask_i_2[rr, cc] = 2

                    # get orientation of ends
                    props_ends_i_1 = regionprops(mask_i_1 * skel_i)[0]
                    props_ends_i_2 = regionprops(mask_i_2 * skel_i)[0]
                    z_orientation_i = [props_ends_i_1.orientation, props_ends_i_2.orientation]
                    y_1, x_1 = props_ends_i_1.centroid
                    y_2, x_2 = props_ends_i_2.centroid
                    _orient_1 = np.arctan2(_z_ends_i[1, 0] - x_1, _z_ends_i[0, 0] - y_1)
                    _orient_2 = np.arctan2(_z_ends_i[1, 1] - x_2, _z_ends_i[0, 1] - y_2)
                    if np.abs(z_orientation_i[0] - _orient_1) > np.pi / 2:
                        z_orientation_i[0] += np.pi
                    if np.abs(z_orientation_i[1] - _orient_2) > np.pi / 2:
                        z_orientation_i[1] += np.pi
                    z_orientation[i] = z_orientation_i

            # lateral alignment index and distance of z-bands
            def lateral_alignment(pos_i, pos_j, theta_i, theta_j):
                phi_ij = np.arctan2((pos_j[1] - pos_i[1]), (pos_j[0] - pos_i[0]))
                phi_ji = phi_ij + np.pi

                if np.cos(theta_i - theta_j + np.pi) > 0 and np.cos(theta_i - phi_ij) > theta_phi_min and np.cos(
                        theta_j - phi_ji) > theta_phi_min:
                    return np.cos(theta_i - theta_j + np.pi) * np.cos(theta_i - phi_ij) * np.cos(theta_j - phi_ji)
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
                A[i, j] = lateral_alignment(_z_ends[i], _z_ends[j], _z_orientation[i], _z_orientation[j])
            D[np.isnan(A)] = np.nan

            # make matrices symmetric for undirected graph
            D = (D + D.T) / 2
            A = (A + A.T) / 2
            links = ~np.isnan(D)

            def prune_edges(L, D):
                N = L.shape[0]
                for i in range(N):
                    connected_nodes = np.where(L[i] == 1)[0]
                    if len(connected_nodes) > 0:
                        min_distance_node = connected_nodes[np.argmin(D[i, connected_nodes])]
                        L[i, :] = 0  # Remove all connections
                        L[:, i] = 0  # Symmetrically for undirected graph
                        L[i, min_distance_node] = 1  # Add back the connection with the smallest distance
                        L[min_distance_node, i] = 1  # Symmetrically for undirected graph
                return L

            links = prune_edges(links, D)
            A[links == 0] = np.nan
            D[links == 0] = np.nan

            # reshape arrays
            links = links.reshape((n_z, 2, n_z, 2), order='F')
            lat_dist = D.reshape((n_z, 2, n_z, 2), order='F')
            lat_alignment = A.reshape((n_z, 2, n_z, 2), order='F')

            # number of lateral neighbors
            lat_neighbors = np.count_nonzero(~np.isnan(lat_dist), axis=(1, 2, 3))

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

        return (length, intensity, straightness, ratio_intensity, avg_intensity, orientation, oop, labels_list, labels,
                lat_neighbors, lat_dist, lat_alignment, links, z_ends, linked_groups, size_groups, length_groups,
                alignment_groups)

    @staticmethod
    def intensity_sarcomeres(image_unet: np.ndarray, image_raw: np.ndarray, pixelsize: float, threshold: float = 0.1) -> \
            Tuple[float, float]:
        """
        Get ratio of sarcomere fluorescence to off-sarcomere fluorescence intensity.

        Parameters
        ----------
        image_unet : np.ndarray
            U-Net result.
        image_raw : np.ndarray
            Raw microscopy image.
        pixelsize : float
            Size of pixel in x,y in µm.
        threshold : float, optional
            Binary threshold for masks. Defaults to 0.1.

        Returns
        -------
        ratio_intensity : float
            Ratio of Z-band fluorescence to off-sarcomere fluorescence intensity.
        avg_intensity : float
            Average intensity of the Z-bands.
        """
        # Binarize the U-Net result
        mask = image_unet >= (threshold * 255)

        # Calculate sarcomere and off-sarcomere intensities
        sarcomere_intensity = np.sum(image_raw[mask])
        off_sarcomere_intensity = np.sum(image_raw[~mask])

        # Calculate the number of pixels in each region
        n_sarcomere_pixels = np.sum(mask)
        n_off_sarcomere_pixels = np.sum(~mask)

        # Calculate average intensities
        avg_sarcomere_intensity = sarcomere_intensity / n_sarcomere_pixels if n_sarcomere_pixels > 0 else 0
        avg_off_sarcomere_intensity = off_sarcomere_intensity / n_off_sarcomere_pixels if n_off_sarcomere_pixels > 0 else 0

        # Calculate the ratio of sarcomere to off-sarcomere intensity
        ratio_intensity = avg_sarcomere_intensity / avg_off_sarcomere_intensity if avg_off_sarcomere_intensity > 0 else 0

        return ratio_intensity, avg_sarcomere_intensity

    @staticmethod
    def binary_kernel(d: float, sigma: float, width: float, orient: float, size: Tuple[float, float],
                      pixelsize: float, mode: str = 'both') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns binary kernel pair for AND-gated double wavelet analysis.

        Parameters
        ----------
        d : float
            Distance between two wavelets.
        sigma : float
            Minor axis width of single wavelets.
        width : float
            Major axis width of single wavelets.
        orient : float
            Rotation orientation in degrees.
        size : Tuple[float, float]
            Size of kernel in µm.
        pixelsize : float
            Pixelsize in µm.
        mode : str, optional
            'separate' returns two separate kernels, 'both' returns a single kernel. Defaults to 'both'.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            The generated binary kernel(s).
        """
        # meshgrid
        size_pixel = Structure.round_up_to_odd(size / pixelsize)
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

    @staticmethod
    def gaussian_kernel(dist: float, minor: float, major: float, orient: float, size: float,
                        pixelsize: float, mode: str = 'both',
                        add_negative_center_kernel: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns gaussian kernel pair for AND-gated double wavelet analysis

        Parameters
        ----------
        dist : float
            Distance between two wavelets
        minor : float
            Minor axis width of single wavelets in µm
        major : float
            Major axis width of single wavelets in µm
        orient : float
            Rotation orientation in degree
        size : float
            Size of kernel in µm
        pixelsize : float
            Pixelsize in µm
        mode : str, optional
            'separate' returns two separate kernels, 'both' returns single kernel
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel in the middle of the two wavelets,
            to avoid detection of two Z-bands two sarcomeres apart as sarcomere

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Gaussian kernel pair
        """
        # Transform FWHM to sigma
        minor_sigma = minor / 2.355
        major_sigma = major / 2.355

        # Calculate the size of the kernel in pixels and create meshgrid
        size_pixel = Structure.round_up_to_odd(size / pixelsize)
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

    @staticmethod
    def half_gaussian_kernel(dist: float, minor: float, major: float, orient: float, size: float,
                             pixelsize: float, mode: str = 'both',
                             add_negative_center_kernel: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns half-gaussian kernel pair for AND-gated double wavelet analysis

        Parameters
        ----------
        dist : float
            Distance between two wavelets
        minor : float
            Minor axis width, in full width at half maximum (FWHM), of single wavelets in µm
        major : float
            Major axis width of single wavelets in µm.
        orient : float
            Rotation orientation in degree
        size : float
            Size of kernel in µm
        pixelsize : float
            Pixelsize in µm
        mode : str, optional
            'separate' returns two separate kernels, 'both' returns single kernel
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel in the middle of the two wavelets,
            to avoid detection of two Z-bands two sarcomeres apart as sarcomere

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Gaussian kernel pair
        """
        # Transform FWHM to sigma
        minor_sigma = minor / 2.355
        major_sigma = major / 2.355

        # Calculate the size of the kernel in pixels and create meshgrid
        size_pixel = Structure.round_up_to_odd(size / pixelsize)
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
                            major: float = 0.5, len_lims: Tuple[float, float] = (1.3, 2.5), len_step: float = 0.025,
                            orient_lims: Tuple[float, float] = (-90, 90), orient_step: float = 5,
                            add_negative_center_kernel: bool = False) -> List[np.ndarray]:
        """
        Returns bank of double wavelets.

        Parameters
        ----------
        pixelsize : float
            Pixel size in µm.
        kernel : str, optional
            Filter kernel ('gaussian' for double Gaussian kernel, 'binary' for binary double-line,
            'half_gaussian' for half Gaussian kernel). Defaults to 'half_gaussian'.
        size : float, optional
            Size of kernel in µm. Defaults to 3.
        minor : float, optional
            Minor axis width of single wavelets. Defaults to 0.15.
        major : float, optional
            Major axis width of single wavelets. Defaults to 0.5.
        len_lims : Tuple[float, float], optional
            Limits of lengths / wavelet distances in µm. Defaults to (1.3, 2.5).
        len_step : float, optional
            Step size in µm. Defaults to 0.025.
        orient_lims : Tuple[float, float], optional
            Limits of orientation angle in degrees. Defaults to (-90, 90).
        orient_step : float, optional
            Step size in degrees. Defaults to 5.
        add_negative_center_kernel : bool, optional
            Whether to add a negative kernel in the middle of the two wavelets,
            to avoid detection of two Z-bands two sarcomeres apart as sarcomere,
            only for kernel=='gaussian' or 'half_gaussian.
            Defaults to False.

        Returns
        -------
        List[np.ndarray]
            Bank of double wavelets.
        """

        len_range = np.arange(len_lims[0] - len_step, len_lims[1] + len_step, len_step, dtype='float32')
        orient_range = np.arange(orient_lims[0], orient_lims[1], orient_step, dtype='float32')
        size_pixel = Structure.round_up_to_odd(size / pixelsize)

        bank = np.zeros((len_range.shape[0], orient_range.shape[0], 2, size_pixel, size_pixel))
        for i, d in enumerate(len_range):
            for j, orient in enumerate(orient_range):
                if kernel == 'gaussian':
                    bank[i, j] = Structure.gaussian_kernel(d, minor, major, orient=orient, size=size,
                                                           pixelsize=pixelsize, mode='separate',
                                                           add_negative_center_kernel=add_negative_center_kernel)
                elif kernel == 'half_gaussian':
                    bank[i, j] = Structure.half_gaussian_kernel(d, minor, major, orient=orient, size=size,
                                                                pixelsize=pixelsize, mode='separate',
                                                                add_negative_center_kernel=add_negative_center_kernel)
                elif kernel == 'binary':
                    bank[i, j] = Structure.binary_kernel(d, minor, major, orient, size, pixelsize, mode='separate')
                else:
                    raise ValueError("Unsupported kernel type. Choose from 'gaussian', 'binary', or 'half_gaussian'.")
        return bank, len_range, orient_range

    @staticmethod
    def convolve_image_with_bank(image: np.ndarray, bank: np.ndarray, device: torch.device, gating: bool = True,
                                 dtype: torch.dtype = torch.float16, save_memory: bool = False,
                                 patch_size: int = 512) -> torch.Tensor:
        """
        AND-gated double-wavelet convolution of image using kernels from filter bank, with merged functionality.
        Processes the image in smaller overlapping patches to manage GPU memory usage and avoid edge effects.

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
        image_torch = torch.from_numpy((image / 255)).to(dtype=dtype).to(device).view(1, 1, image.shape[0],
                                                                                      image.shape[1])
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
    def argmax_wavelets(result: torch.Tensor, len_range: torch.Tensor, orient_range: torch.Tensor) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the argmax of wavelet convolution results to extract length, orientation, and maximum score map.

        This function processes the result of a wavelet convolution operation to determine the optimal
        length and orientation for each position in the input image. It leverages GPU acceleration for
        efficient computation and returns the results as NumPy arrays.

        Parameters
        ----------
        result : torch.Tensor
            The result tensor from a wavelet convolution operation, expected to be on a GPU device.
            Shape is expected to be (num_orientations, num_lengths, height, width).
        len_range : torch.Tensor
            A tensor containing the different lengths used in the wavelet bank. Shape: (num_lengths,).
        orient_range : torch.Tensor
            A tensor containing the different orientation angles used in the wavelet bank, in degrees.
            Shape: (num_orientations,).

        Returns
        -------
        length_np : np.ndarray
            A 2D array of the optimal length for each position in the input image. Shape: (height, width).
        orient_np : np.ndarray
            A 2D array of the optimal orientation (in radians) for each position in the input image.
            Shape: (height, width).
        max_score_np : np.ndarray
            A 2D array of the maximum convolution score for each position in the input image.
            Shape: (height, width).
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
    def get_points_midline(length: np.ndarray, orientation: np.ndarray, max_score: np.ndarray, len_range: torch.Tensor,
                           score_threshold: float = 90., abs_threshold: bool = False) -> Tuple:
        """
        Extracts points on sarcomere midlines and calculates sarcomere length and orientation at these points.

        This function performs the following steps:
        1. **Thresholding:** Applies a threshold to the length, orientation, and max_score arrays to refine sarcomere detection.
        2. **Binarization:** Creates a binary mask to isolate midline regions.
        3. **Skeletonization:** Thins the midline regions for easier analysis.
        4. **Labeling:** Assigns unique labels to each connected midline component.
        5. **Midline Point Extraction:** Identifies the coordinates of points along each midline.
        6. **Value Calculation:** Calculates sarcomere length, orientation, and maximal score at each midline point.

        Parameters
        ----------
        length : np.ndarray
            Sarcomere length map obtained from wavelet analysis.
        orientation : np.ndarray
            Sarcomere orientation angle map obtained from wavelet analysis.
        max_score : np.ndarray
            Map of maximal wavelet scores.
        len_range : torch.Tensor
            An array containing the different lengths used in the wavelet bank.
        score_threshold : float, optional
            Threshold for filtering detected sarcomeres. Can be either an absolute value (if abs_threshold=True) or
            a percentile value for adaptive thresholding (if abs_threshold=False). Default is 90.
        abs_threshold : bool, optional
            Flag to determine the thresholding method. If True, 'score_threshold' is used as an absolute value.
            If False, 'score_threshold' is interpreted as a percentile for adaptive thresholding. Default is False.

        Returns
        -------
        tuple
            * **points** (list): List of (x, y) coordinates for each midline point.
            * **midline_id_points** (list): List of corresponding midline labels for each point.
            * **midline_length_points** (list): List of approximate midline lengths associated with each point. In pixels.
            * **sarcomere_length_points** (list): List of sarcomere lengths at each midline point.
            * **sarcomere_orientation_points** (list): List of sarcomere orientation angles at each midline point.
            * **max_score_points** (list): List of maximal wavelet scores at each midline point.
            * **midline** (np.ndarray): The binarized midline mask.
            * **score_threshold** (float): The final threshold value used.
        """
        # rough thresholding of sarcomere structures to better identify adaptive threshold
        # determine adaptive threshold from value distribution
        if not abs_threshold:
            score_threshold_val = max_score.max() * score_threshold
        else:
            score_threshold_val = score_threshold

        # binarize midline
        midline = max_score >= score_threshold_val

        # skeletonize
        midline_skel = morphology.skeletonize(midline)

        # label midlines
        midline_labels, n_midlines = ndimage.label(midline_skel, ndimage.generate_binary_structure(2, 2))

        # iterate midlines and create additional list with labels and midline length (approximated by max. Feret diameter)
        props = skimage.measure.regionprops_table(midline_labels, properties=['label', 'coords', 'feret_diameter_max'])
        list_labels, coords_midlines, length_midlines = props['label'], props['coords'], props['feret_diameter_max']

        points, midline_id_points, midline_length_points = [], [], []
        if n_midlines > 0:
            for i, (label_i, coords_i, length_midline_i) in enumerate(
                    zip(list_labels, coords_midlines, length_midlines)):
                points.append(coords_i)
                midline_length_points.append(np.ones(coords_i.shape[0]) * length_midline_i)
                midline_id_points.append(np.ones(coords_i.shape[0]) * label_i)

            points = np.concatenate(points, axis=0).T
            midline_id_points = np.concatenate(midline_id_points)
            midline_length_points = np.concatenate(midline_length_points)

            # get sarcomere orientation and distance at points, additionally filter score
            sarcomere_length_points = length[points[0], points[1]]
            sarcomere_orientation_points = orientation[points[0], points[1]]
            max_score_points = max_score[points[0], points[1]]

            # remove points outside range of sarcomere lengths in wavelet bank
            ids_in = (sarcomere_length_points >= len_range[1]) & (sarcomere_length_points <= len_range[-2])
            points = points[:, ids_in]
            midline_length_points = midline_length_points[ids_in]
            midline_id_points = midline_id_points[ids_in]
            sarcomere_length_points = sarcomere_length_points[ids_in]
            sarcomere_orientation_points = sarcomere_orientation_points[ids_in]
            max_score_points = max_score_points[ids_in]
        else:
            sarcomere_length_points, sarcomere_orientation_points, max_score_points = [], [], []

        return (points, midline_id_points, midline_length_points, sarcomere_length_points,
                sarcomere_orientation_points, max_score_points, midline, score_threshold)

    @staticmethod
    def cluster_sarcomeres(points: np.ndarray,
                           sarcomere_length_points: np.ndarray,
                           sarcomere_orientation_points: np.ndarray,
                           midline_id_points: np.ndarray,
                           pixelsize: float,
                           size: Tuple[int, int],
                           dist_threshold_ends: float = 0.5,
                           dist_threshold_midline_points: float = 0.5,
                           louvain_resolution: float = 0.06,
                           louvain_seed: int = 2,
                           area_min: float = 50,
                           dilation_radius: int = 3) -> Tuple[int, List, List, List, List, List, np.ndarray]:
        """
        This function clusters sarcomeres into domains based on their spatial and orientational properties
        using the Louvain method for community detection. It considers sarcomere lengths, orientations,
        and positions along midlines to form networks of connected sarcomeres. Domains are then identified
        as communities within these networks, with additional criteria for minimum domain area
        and connectivity thresholds. Finally, this function quantifies the mean and std of sarcomere lengths,
        and the orientational order parameter and mean orientation of each domain.

        Parameters
        ----------
        points : np.ndarray
            List of sarcomere midline point positions
        sarcomere_length_points : np.ndarray
            List of midline point sarcomere lengths
        sarcomere_orientation_points : np.ndarray
            List of midline point sarcomere orientations, in radians
        midline_id_points : np.ndarray
            List of midline point indices, points of the same midline have same index.
        pixelsize : float
            Pixel size in µm
        size : tuple(int, int)
            Shape of the image in pixels.
        dist_threshold_ends : float
            Max. distance threshold for connecting / creating network edge for adjacent sarcomere vector ends.
            Only the ends with the smallest distance are connected.
        dist_threshold_midline_points : float
            Max. distance threshold for connecting / creating network edge for midline points of the same midline.
            All points within this distance are connected.
        louvain_resolution : float
            Control parameter for domain size. If resolution is small, the algorithm favors larger domains.
            Greater resolution favors smaller domains.
        louvain_seed : int
            Random seed for Louvain algorithm, to ensure reproducibility.
        area_min : float
            Minimal area of domains / clusters (in µm^2).
        dilation_radius : int
            Dilation radius to refine calculation of domain areas (in pixels).

        Returns
        -------
        n_domains : int
            Number of domains.
        domains : list
            List of domain sets with point indices.
        area_domains : list
            List with domain areas.
        sarcomere_length_domains : list
            List with mean sarcomere length within each domain
        oop_domains : list
            Orientational order parameter of sarcomeres in each domains
        orientation_domains : list
            Main orientation of domains
        mask_domains : ndarray
            Masks of domains with value representing domain label
        """

        if len(points.T) > 10:
            # Calculate orientation vectors using trigonometry
            orientation_vectors = np.asarray([-np.sin(sarcomere_orientation_points),
                                              np.cos(sarcomere_orientation_points)])

            # Calculate the ends of the vectors based on their orientation and length
            ends_0 = points + orientation_vectors * sarcomere_length_points / 2  # End point 1 of each vector
            ends_1 = points - orientation_vectors * sarcomere_length_points / 2  # End point 2 of each vector
            ends = np.concatenate((ends_0[:, :, None],
                                   ends_1[:, :, None]), axis=2).reshape(2, -1)  # Combine and reshape for KDTree
            midline_id_ends = np.repeat(midline_id_points, 2)
            orientation_ends = np.repeat(orientation_vectors, 2)

            # Create a KDTree for efficient nearest neighbor search
            tree_ends = cKDTree(ends.T)

            nearest_neighbors = []
            for i, point in enumerate(ends.T):
                # Start with k=2 since the first returned neighbor is the point itself
                distances, indices = tree_ends.query(point, k=10)  # Query more neighbors to ensure finding a valid one
                valid_neighbors = [(dist, idx) for dist, idx in zip(distances[1:], indices[1:]) if
                                   midline_id_ends[idx] != midline_id_ends[i] if dist < dist_threshold_ends if
                                   np.cos(orientation_ends[idx] - orientation_ends[i]) > np.cos(np.pi / 4)]

                if valid_neighbors:
                    # Sort valid neighbors by distance and select the closest one
                    closest_valid_neighbor = sorted(valid_neighbors, key=lambda x: x[0])[0]
                    nearest_neighbors.append((i, closest_valid_neighbor[1]))

            # Map pairs of ends back to their original vector indices
            pairs_points = set([(x // 2, y // 2) for x, y in nearest_neighbors])

            # connect adjacent midline points
            tree_points = cKDTree(points.T)

            # Find indices of neighbors for all points
            indices_list = tree_points.query_ball_tree(tree_points, dist_threshold_midline_points)

            # build graph
            G = nx.Graph()

            # Add nodes to the graph
            for i in range(len(points)):
                G.add_node(i)

            def cosine_similarity(i, j):
                orient_i, orient_j = sarcomere_orientation_points[i], sarcomere_orientation_points[j]
                return round(np.cos(orient_i - orient_j) ** 2,
                             4)  # round to avoid issues with floating point comparisons

            # Add edges to the graph based on pairs of ends

            for i, j in pairs_points:
                G.add_edge(i, j, weight=cosine_similarity(i, j))

            # Connect adjacent midline points
            for i, indices in enumerate(indices_list):
                for j in indices:
                    if i != j:
                        G.add_edge(i, j, weight=1)

            # Detect communities using the Louvain method
            communities_generator = community.louvain_communities(G, seed=louvain_seed, resolution=louvain_resolution,
                                                                  weight='weight')
            domains = list(communities_generator)
            n_domains = len(domains)

            # shuffle domains
            random.seed(123)
            random.shuffle(domains)

            # calculate domain areas and remove small domains
            _area_domains = np.zeros(n_domains) * np.nan
            _indices_to_remove = []
            for i, domain_i in enumerate(domains):
                points_i = points[:, list(domain_i)]
                orientations_i = sarcomere_orientation_points[list(domain_i)]
                lengths_i = sarcomere_length_points[list(domain_i)]
                if points_i.shape[1] > 10:
                    # bounding box
                    min_i = (
                        max(int((points_i[0].min() - 3) // pixelsize), 0),
                        max(int((points_i[1].min() - 3) // pixelsize), 0))
                    max_i = (min(int((points_i[0].max() + 3) // pixelsize), size[0]),
                             min(int((points_i[1].max() + 3) // pixelsize), size[1]))
                    size_i = (max_i[0] - min_i[0], max_i[1] - min_i[1])
                    _points_i = points_i.copy()
                    _points_i[0] -= min_i[0] * pixelsize
                    _points_i[1] -= min_i[1] * pixelsize
                    mask_i = Structure.sarcomere_mask(_points_i, orientations_i, lengths_i, size_i, pixelsize=pixelsize,
                                                      dilation_radius=dilation_radius)
                    area_i = np.sum(mask_i) * pixelsize ** 2
                    _area_domains[i] = area_i
                    if area_i <= area_min:
                        _indices_to_remove.append(i)
                else:
                    _indices_to_remove.append(i)
            _indices_to_remove.sort(reverse=True)
            domains = [list(domain) for i, domain in enumerate(domains) if i not in _indices_to_remove]
            n_domains = len(domains)

            # quantify domains
            (sarcomere_orientation_domains, sarcomere_oop_domains, sarcomere_length_mean_domains,
             sarcomere_length_std_domains) = np.zeros(n_domains), np.zeros(n_domains), np.zeros(n_domains), np.zeros(
                n_domains)
            area_domains = np.zeros(n_domains)
            mask_domains = np.zeros(size, dtype='uint16')
            for i, domain_i in enumerate(domains):
                points_i = points[:, list(domain_i)]
                lengths_i = sarcomere_length_points[list(domain_i)]
                orientations_i = sarcomere_orientation_points[list(domain_i)]
                # bounding box
                min_i = (
                    max(int((points_i[0].min() - 3) // pixelsize), 0),
                    max(int((points_i[1].min() - 3) // pixelsize), 0))
                max_i = (min(int((points_i[0].max() + 3) // pixelsize), size[0]),
                         min(int((points_i[1].max() + 3) // pixelsize), size[1]))
                size_i = (max_i[0] - min_i[0], max_i[1] - min_i[1])
                _points_i = points_i.copy()
                _points_i[0] -= min_i[0] * pixelsize
                _points_i[1] -= min_i[1] * pixelsize
                mask_i = Structure.sarcomere_mask(_points_i, orientations_i, lengths_i, size_i, pixelsize=pixelsize,
                                                  dilation_radius=dilation_radius)
                ind_i = np.where(mask_i)
                ind_i = (ind_i[0] + min_i[0], ind_i[1] + min_i[1])
                mask_domains[ind_i] = i
                area_i = np.sum(mask_i) * pixelsize ** 2
                area_domains[i] = area_i
                sarcomere_length_mean_domains[i] = np.mean(lengths_i)
                sarcomere_length_std_domains[i] = np.std(lengths_i)
                oop, angle = Utils.analyze_orientations(orientations_i)
                sarcomere_oop_domains[i] = oop
                sarcomere_orientation_domains[i] = angle
            return (n_domains, domains, area_domains, sarcomere_length_mean_domains, sarcomere_length_std_domains,
                    sarcomere_oop_domains, sarcomere_orientation_domains, mask_domains)
        else:
            return 0, [], [], [], [], [], [], []

    @staticmethod
    def _grow_line(seed, points_t, sarcomere_length_points_t, sarcomere_orientation_points_t, nbrs, threshold_distance,
                   pixelsize, persistence):

        line_i = deque([seed])
        stop_right = stop_left = False

        # threshold_distance from micrometer to pixels
        threshold_distance_pixels = threshold_distance / pixelsize

        end_left = end_right = points_t[:, seed]
        length_left = length_right = sarcomere_length_points_t[seed] / pixelsize
        orientation_left = orientation_right = sarcomere_orientation_points_t[seed]

        while not stop_left or not stop_right:
            n_i = len(line_i)
            if n_i > 1:
                line_i_list = list(line_i)  # Convert deque to list for slicing
                if not stop_left:
                    end_left = points_t[:, line_i_list[0]]
                    length_left = np.mean(sarcomere_length_points_t[line_i_list[:persistence]]) / pixelsize
                    orientation_left = stats.circmean(sarcomere_orientation_points_t[line_i_list[:persistence]])
                if not stop_right:
                    end_right = points_t[:, line_i_list[-1]]
                    length_right = np.mean(sarcomere_length_points_t[line_i_list[-persistence:]]) / pixelsize
                    orientation_right = stats.circmean(sarcomere_orientation_points_t[line_i_list[-persistence:]])

            # grow left
            if not stop_left:
                prior_left = [end_left[0] + np.sin(orientation_left) * length_left,
                              end_left[1] - np.cos(orientation_left) * length_left]
                # nearest neighbor left
                distance_left, index_left = nbrs.kneighbors([prior_left], return_distance=True)
                # extend list
                if distance_left[0][0] < threshold_distance_pixels:
                    line_i.appendleft(index_left[0][0].astype('int'))
                else:
                    stop_left = True

            # grow right
            if not stop_right:
                prior_right = [end_right[0] - np.sin(orientation_right) * length_right,
                               end_right[1] + np.cos(orientation_right) * length_right]
                # nearest neighbor right
                distance_right, index_right = nbrs.kneighbors([prior_right], return_distance=True)
                # extend list
                if distance_right[0][0] < threshold_distance_pixels:
                    line_i.append(index_right[0][0].astype('int'))
                else:
                    stop_right = True

        return np.asarray(line_i)

    @staticmethod
    def line_growth(points_t: np.ndarray, sarcomere_length_points_t: np.ndarray,
                    sarcomere_orientation_points_t: np.ndarray, max_score_points_t: np.ndarray,
                    midline_length_points_t: np.ndarray, pixelsize: float, n_seeds: int = 1000, random_seed=None,
                    persistence: int = 4, threshold_distance: float = 0.3, n_min: int = 5):
        """
        Line growth algorithm to determine myofibril lines perpendicular to sarcomere z-bands

        Parameters
        ----------
        points_t : np.ndarray
            List of midline point positions
        sarcomere_length_points_t : list
            Sarcomere length at midline points
        sarcomere_orientation_points_t : list
            Sarcomere orientation angle at midline points, in radians
        max_score_points_t : list
            Maximal score at midline points
        midline_length_points_t : list
            Length of sarcomere midlines of midline points
        pixelsize : float
            Pixel size in µm
        n_seeds : int
            Number of random seed for line growth
        random_seed : int
            Random seed for line growth starting points. If None, no random seed is set.
        persistence : int
            Number of points to consider for averaging length and orientation.

        Returns
        -------
        line_data : dict
            Dictionary with LOI data keys = (lines, line_features)
        """
        # select random origins for line growth
        points_t = np.asarray(points_t)
        assert len(points_t) > 0, 'No sarcomeres in image (len(points) = 0), could not grow lines.'
        random.seed(random_seed)
        n_points = len(points_t.T)
        seed_idx = random.sample(range(n_points), min(n_seeds, n_points))

        # Precompute Nearest Neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_t.T)

        # Prepare arguments for parallel processing
        args = [
            (seed, points_t, sarcomere_length_points_t, sarcomere_orientation_points_t, nbrs, threshold_distance,
             pixelsize,
             persistence) for seed in seed_idx]

        # grow lines
        lines = [Structure._grow_line(*arg) for arg in args]

        # remove short lines (< n_min)
        lines = [l for l in lines if len(l) >= n_min]
        # calculate features of lines
        n_points_lines = np.asarray([len(l) for l in lines])  # number of sarcomere in line
        length_line_segments = [sarcomere_length_points_t[l] for l in lines]
        length_lines = [np.sum(lengths) for lengths in length_line_segments]
        # sarcomere lengths
        sarcomere_mean_length_lines = [np.mean(sarcomere_length_points_t[l]) for l in lines]
        sarcomere_std_length_lines = [np.std(sarcomere_length_points_t[l]) for l in lines]
        # midline lengths
        midline_mean_length_lines = [np.nanmean(midline_length_points_t[l]) for l in lines]
        midline_std_length_lines = [np.nanstd(midline_length_points_t[l]) for l in lines]
        midline_min_length_lines = [np.nanmin(midline_length_points_t[l]) for l in lines]
        # wavelet scores
        mean_score_lines = [np.mean(max_score_points_t[l]) for l in lines]
        std_score_lines = [np.std(max_score_points_t[l]) for l in lines]
        # mean squared curvature
        tangential_vector_line_segments = [np.diff(points_t.T[l], axis=0) for l in lines]
        tangential_angle_line_segments = [np.asarray([np.arctan2(v[1], v[0]) for v in vectors]) for vectors in
                                          tangential_vector_line_segments]
        curvature_line_segments = [np.diff(angle_i) / length_line_segments[i][1:-1] for i, angle_i in
                                   enumerate(tangential_angle_line_segments)]
        msc_lines = [np.sum(curvature_line_segments[i] ** 2) / length_lines[i] for i in range(len(lines))]
        # mean and std orientation, and maximal change of orientation
        mean_orient_lines = [stats.circmean(sarcomere_orientation_points_t[l]) for l in lines]
        std_orient_lines = [stats.circstd(sarcomere_orientation_points_t[l]) for l in lines]
        max_orient_change_lines = [Utils.max_orientation_change(sarcomere_orientation_points_t[l]) for l in lines]
        # create dictionary
        line_features = {'n_points_lines': n_points_lines, 'length_lines': length_lines,
                         'sarcomere_mean_length_lines': sarcomere_mean_length_lines,
                         'sarcomere_std_length_lines': sarcomere_std_length_lines,
                         'mean_score_lines': mean_score_lines, 'std_score_lines': std_score_lines,
                         'msc_lines': msc_lines, 'mean_orient_lines': mean_orient_lines,
                         'std_orient_lines': std_orient_lines, 'midline_mean_length_lines': midline_mean_length_lines,
                         'max_orient_change_lines': max_orient_change_lines,
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
        line = line[:, ::-1]
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
            The coordinates of the curved line profile. Shape is (2, N, linewidth), where N is the total number of points in the line.
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
                       sarcomere_orientation_points: np.ndarray,
                       sarcomere_length_points: np.ndarray,
                       size: Tuple[int, int],
                       pixelsize: float,
                       dilation_radius: int = 3) -> np.ndarray:
        """
        Calculates a binary mask of areas with sarcomeres.

        Parameters
        ----------
        points : ndarray
            Positions of sarcomere vectors in µm.
        sarcomere_orientation_points : ndarray
            Orientations of sarcomere vectors.
        sarcomere_length_points : ndarray
            Lengths of sarcomere vectors in µm.
        size : tuple
            Size of the image, in pixels.
        pixelsize : float
            Pixel size in µm.
        dilation_radius : int, optional
            Dilation radius to close small holes (default is 3).

        Returns
        -------
        mask : ndarray
            Binary mask of sarcomeres.
        """
        # Calculate orientation vectors using trigonometry
        orientation_vectors = np.asarray([-np.sin(sarcomere_orientation_points),
                                          np.cos(sarcomere_orientation_points)])
        # Calculate the ends of the vectors based on their orientation and length
        ends_0 = points + orientation_vectors * sarcomere_length_points / 2  # End point 1 of each vector
        ends_1 = points - orientation_vectors * sarcomere_length_points / 2  # End point 2 of each vector
        ends_0, ends_1 = ends_0 / pixelsize, ends_1 / pixelsize
        mask = np.zeros(size, dtype='bool')
        for e0, e1 in zip(ends_0.T.astype('int'), ends_1.T.astype('int')):
            rr, cc = line(*e0, *e1)
            try:
                mask[rr, cc] = True
            except:
                pass
        mask = binary_dilation(mask, disk(dilation_radius))
        return mask
