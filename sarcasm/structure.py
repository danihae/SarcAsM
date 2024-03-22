import os
import random
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import skimage.measure
import tifffile
import torch
import torch.nn.functional as F
from biu import siam_unet as siam
from biu import unet
from biu.progress import ProgressNotifier
from joblib import Parallel, delayed
from networkx.algorithms import community
from scipy import ndimage, stats
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import directed_hausdorff, squareform, pdist
from skimage import segmentation, morphology
from skimage.draw import disk as draw_disk, line
from skimage.measure import label, regionprops_table, regionprops, profile_line
from skimage.morphology import skeletonize, binary_closing, disk, binary_dilation
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm as tqdm

from .ioutils import *
from .utils import map_array, analyze_orientations, model_dir, convert_lists_to_arrays_in_dict, max_orientation_change

# select device
if torch.has_cuda:
    device = torch.device('cuda:0')
elif hasattr(torch, 'has_mps'):  # only for apple m1/m2/...
    if torch.has_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
if device.type == 'cpu':
    print("Warning: No CUDA or MPS device found. Calculations will run on the CPU, "
          "which might be slower.")


class Structure:
    """Class to analyze sarcomere morphology"""

    def __init__(self):
        # structure data dictionary
        if os.path.exists(self.__get_structure_data_file()):
            self._load_structure_data()
        else:
            self.structure = {}

    def __get_structure_data_file(self, is_temp_file=False):
        """
        is_temp_file is used for normally only creating a temporary file
        and upon calling commit method the file is stored normally (the diff between temp file and "final" one is just the name)
        this should prevent creating corrupted data files due to aborted operations(exception or user intervention)
        """
        if is_temp_file:
            return self.data_folder + "structure.temp.json"
        else:
            return self.data_folder + "structure.json"

    def commit(self):
        """
        commit data (either rename temp-file to normal data file name or just write it again+remove temp-file)
        """
        if os.path.exists(self.__get_structure_data_file(is_temp_file=True)):
            if os.path.exists(self.__get_structure_data_file()):
                os.remove(self.__get_structure_data_file())
            os.rename(self.__get_structure_data_file(is_temp_file=True), self.__get_structure_data_file())

    def store_structure_data(self, override=True):
        """ Store structure data in json file """
        # only store if path doesn't exist or override is true
        if override or (not os.path.exists(self.__get_structure_data_file(is_temp_file=False))):
            IOUtils.json_serialize(self.structure, self.__get_structure_data_file(is_temp_file=True))
            self.commit()

    def _load_structure_data(self):
        """
        load structure data if exist load "normal" data file, if it fails and a temp file exists, load temp file
        """
        if os.path.exists(self.__get_structure_data_file(is_temp_file=False)):
            # persistent file exists, try using it
            try:
                self.structure = IOUtils.json_deserialize(self.__get_structure_data_file(is_temp_file=False))
            except:
                if os.path.exists(self.__get_structure_data_file()):
                    self.structure = IOUtils.json_deserialize(self.__get_structure_data_file())
        else:
            # no persistent file exists, look if a temp-file exists
            if os.path.exists(self.__get_structure_data_file()):
                self.structure = IOUtils.json_deserialize(self.__get_structure_data_file())
        if self.structure is None:
            raise Exception('loading of structure failed')

    def predict_z_bands(self, siam_unet=False, model_path=None, size=(1024, 1024),
                        normalization_mode='all', clip_thres=(0., 99.8),
                        progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """Predict sarcomere z-bands with U-Net or Siam-U-Net

        Parameters
        ----------
        siam_unet : bool
            If True, the temporally more consistent Siam-U-Net is used.
        model_path : str
            Path of trained network weights for U-Net or Siam-U-Net
        size : tuple[int, int]
            Resize dimensions for convolutional neural network. Dims need to be dividable by 16.
        normalization_mode : str
            Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
            'all': based on histogram of full stack, 'first': based on histogram of first image in stack)
        clip_thres : tuple[float, float]
            Clip threshold (lower / upper) for intensity normalization.
        progress_notifier
            Progress notifier for inclusion in GUI
        """
        print('Predicting sarcomere z-bands ...')
        if siam_unet:
            if model_path is None or model_path == 'generalist':
                model_path = model_dir + 'siam_unet_sarcomeres.pth'
            _ = siam.Predict(self.file_images, self.folder + 'sarcomeres.tif', model_params=model_path,
                             resize_dim=size, normalization_mode=normalization_mode,
                             clip_threshold=clip_thres, normalize_result=True, progress_notifier=progress_notifier)
        else:
            if model_path is None or model_path == 'generalist':
                model_path = model_dir + 'unet_sarcomeres_generalist.pth'
            _ = unet.Predict(self.file_images, self.folder + 'sarcomeres.tif', model_params=model_path,
                             resize_dim=size, normalization_mode=normalization_mode,
                             clip_threshold=clip_thres, normalize_result=True,
                             progress_notifier=progress_notifier)
        self.file_sarcomeres = self.folder + 'sarcomeres.tif'
        _dict = {'params.predict_z_bands_model': model_path,
                 'params.predict_z_bands_siam_unet': siam_unet,
                 'params.predict_z_bands_normalization_mode': normalization_mode,
                 'params.predict_z_bands_clip_threshold': clip_thres}
        self.structure.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def predict_cell_area(self, siam_unet=False, model_path=None, size=(1024, 1024), normalization_mode='all',
                          clip_thres=(0., 99.8),
                          progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """Predict binary mask of cells vs. background with U-Net or Siam-U-Net

        Parameters
        ----------
        siam_unet : bool
            If True, the temporally more consistent Siam-U-Net is used.
        model_path : str
            Path of trained network weights for U-Net or Siam-U-Net
        size : tuple[int, int]
            Resize dimensions for convolutional neural network. Dims need to be dividable by 16.
        normalization_mode : str
            Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
            'all': based on histogram of full stack, 'first': based on histogram of first image in stack)
        clip_thres : tuple[float, float]
            Clip threshold (lower / upper) for intensity normalization.
        progress_notifier
            Progress notifier for inclusion in GUI
        """
        print('Predicting binary mask of cells ...')
        if siam_unet:
            if model_path is None or model_path == 'generalist':
                model_path = model_dir + 'unet_cell_area_generalist.pth'
            _ = siam.Predict(self.file_images, self.folder + 'cell_mask.tif', model_params=model_path,
                             resize_dim=size, normalization_mode=normalization_mode, clip_threshold=clip_thres,
                             normalize_result=True)
        else:
            if model_path is None or model_path == 'generalist':
                model_path = model_dir + 'unet_cell_area_generalist.pth'
            _ = unet.Predict(self.file_images, self.folder + 'cell_mask.tif', model_params=model_path,
                             resize_dim=size, normalization_mode=normalization_mode,
                             clip_threshold=clip_thres, normalize_result=True, progress_notifier=progress_notifier)
        self.file_cell_mask = self.folder + 'cell_mask.tif'
        _dict = {'params.predict_cell_area_model': model_path,
                 'params.predict_cell_area_siam_unet': siam_unet,
                 'params.predict_cell_area_normalization_mode': normalization_mode,
                 'params.predict_cell_area_clip_threshold': clip_thres}
        self.structure.update(_dict)
        self.store_structure_data()

    def analyze_cell_area(self, timepoints='all', threshold=0.1):
        """
        Analyzes the area of cells in the given image(s) and calculates the cell area ratio.

        Parameters
        ----------
        timepoints : {'all', int, list, np.ndarray}, optional
            Specifies the timepoints to analyze. If 'all', analyzes all timepoints.
            If an integer, analyzes the specified timepoint. If a list or numpy array,
            analyzes the specified timepoints. Defaults to 'all'.
        threshold : float, optional
            Threshold value for binarizing the cell mask image. Pixels with intensity
            above threshold * 255 are considered cell. Defaults to 0.1.
        """
        assert self.file_cell_mask is not None, "Cell mask not found. Please run predict_cell_area first."
        if timepoints == 'all':
            imgs = tifffile.imread(self.file_cell_mask)
        elif isinstance(timepoints, int) or isinstance(timepoints, list) or type(timepoints) is np.ndarray:
            imgs = tifffile.imread(self.file_cell_mask, key=timepoints)
        else:
            ValueError('timepoints argument not valid')
        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs, 0)
        n_imgs = len(imgs)

        # create empty array
        cell_area, cell_area_ratio = np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan

        for i, img in enumerate(tqdm(imgs)):
            # binarize mask
            mask = np.zeros_like(img)
            mask[img > threshold * 255] = 1

            cell_area[i] = np.sum(mask) * self.metadata['pixelsize'] ** 2
            cell_area_ratio[i] = cell_area[i] / (img.shape[0] * img.shape[1] * self.metadata['pixelsize'] ** 2)
        _dict = {'cell_area': cell_area, 'cell_area_ratio': cell_area_ratio, 'params.cell_area_threshold': threshold}
        self.structure.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def analyze_z_bands(self, timepoints='all', threshold=0.1, min_length=1., end_radius=0.75, theta_phi_min=0.25,
                        d_max=5., d_min=0.25, save_all=False):
        """Segment and analyze sarcomere z-bands

        Parameters
        ----------
        timepoints : int/list/str
            Timepoints for z-band analysis ('all': all frames, int for single frame, list or ndarray for
            selected frames)
        threshold : float
            Threshold for binarizing z-bands prior to labeling (0 - 1)
        min_length : float
            Minimal length of z-bands, smaller z-bands are removed. (in um)
        end_radius : float
            Radius around z-band ends to quantify orientation of ends (in um)
        theta_phi_min : float
            Minimal cosine of angle between pointed z-band vector and connecting vector between ends of z-bands.
            Smaller values are not recognized as connections. (->for lateral alignment and distance analysis)
        d_max : float
            Maximal distance between z-band ends (in µm). Z-bands end pairs with larger distance are not connected.
            (->for lateral alignment and distance analysis)
        d_min : float
            Minimal distance between z-band ends (in µm). Z-band end pairs with smaller distances are not connected.
        save_all : bool
            If True, z_labels and lateral distance and alignment matrices are stored (large storage), if False not.
        """
        assert self.file_sarcomeres is not None, ("Z-band mask not found. Please run predict_z_bands first.")
        if timepoints == 'all':
            imgs = tifffile.imread(self.file_sarcomeres)
            imgs_raw = tifffile.imread(self.file_images)
        elif isinstance(timepoints, int) or isinstance(timepoints, list) or type(timepoints) is np.ndarray:
            imgs = tifffile.imread(self.folder + 'sarcomeres.tif', key=timepoints)
            imgs_raw = tifffile.imread(self.file_sarcomeres, key=timepoints)
        else:
            ValueError('timepoints argument not valid')
        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs, 0)
            imgs_raw = np.expand_dims(imgs_raw, 0)
        n_imgs = len(imgs)

        # create empty arrays
        z_length, z_intensity, z_straightness, z_ratio_intensity, z_orientation = ([] for _ in range(5))
        z_lat_neighbors, z_lat_alignment, z_lat_dist = ([] for _ in range(3))
        z_lat_size_groups, z_lat_length_groups, z_lat_alignment_groups = ([] for _ in range(3))

        nan_arrays = lambda: np.full(n_imgs, np.nan)
        z_length_mean, z_length_std, z_length_max, z_length_sum = (nan_arrays() for _ in range(4))
        z_intensity_mean, z_intensity_std = (nan_arrays() for _ in range(2))
        z_straightness_mean, z_straightness_std = (nan_arrays() for _ in range(2))
        z_ratio_intensity, z_oop, avg_intensity = (nan_arrays() for _ in range(3))
        z_lat_neighbors_mean, z_lat_neighbors_std = (nan_arrays() for _ in range(2))
        z_lat_alignment_mean, z_lat_alignment_std = (nan_arrays() for _ in range(2))
        z_lat_dist_mean, z_lat_dist_std = (nan_arrays() for _ in range(2))
        z_lat_size_groups_mean, z_lat_size_groups_std = (nan_arrays() for _ in range(2))
        z_lat_length_groups_mean, z_lat_length_groups_std = (nan_arrays(), nan_arrays())
        z_lat_alignment_groups_mean, z_lat_alignment_groups_std = (nan_arrays() for _ in range(2))

        if save_all:
            z_labels, z_ends, z_links, z_lat_groups = ([] for _ in range(4))
        else:
            z_labels = z_ends = z_links = z_lat_groups = None

        # iterate images
        print('Start z-band analysis!')
        for i, img_i in enumerate(tqdm(imgs)):
            # segment z-bands
            labels_i, labels_skel_i = segment_z_bands(img_i)

            # analyze z-band features
            z_band_features = analyze_z_bands(img_i, labels_i, labels_skel_i, imgs_raw[i],
                                              pixelsize=self.metadata['pixelsize'], threshold=threshold,
                                              min_length=min_length, end_radius=end_radius, theta_phi_min=theta_phi_min,
                                              d_max=d_max, d_min=d_min)

            (z_length_i, z_intensity_i, z_straightness_i, z_ratio_intensity_i, avg_intensity_i, orientation_i, z_oop_i,
             labels_list_i, labels_i, z_lat_neighbors_i, z_lat_dist_i, z_lat_alignment_i, z_links_i, z_ends_i,
             z_lat_groups_i, z_lat_size_groups_i, z_lat_length_groups_i, z_lat_alignment_groups_i,
             ) = z_band_features

            # write in arrays
            z_length.append(z_length_i)
            z_intensity.append(z_intensity_i)
            z_straightness.append(z_straightness_i)
            z_lat_alignment.append(z_lat_alignment_i)
            z_lat_neighbors.append(z_lat_neighbors_i)
            z_orientation.append(orientation_i)
            z_lat_dist.append(z_lat_dist_i)
            z_lat_size_groups.append(z_lat_size_groups_i)
            z_lat_length_groups.append(z_lat_length_groups_i)
            z_lat_alignment_groups.append(z_lat_alignment_groups_i)
            z_ratio_intensity[i], avg_intensity[i], z_oop[i] = z_ratio_intensity_i, avg_intensity_i, z_oop_i

            if save_all:
                z_labels.append(labels_i)
                z_links.append(z_links_i)
                z_ends.append(z_ends_i)
                z_lat_groups.append(z_lat_groups_i)

            # calculate mean and std of z-band features
            if len(z_length_i) > 0:
                z_length_mean[i], z_length_std[i], z_length_max[i], z_length_sum[i] = np.mean(z_length_i), np.std(
                    z_length_i), np.max(z_length_i), np.sum(z_length_i)
            z_intensity_mean[i], z_intensity_std[i] = np.mean(z_intensity_i), np.std(z_intensity_i)
            z_straightness_mean[i], z_straightness_std[i] = np.mean(z_straightness_i), np.std(z_straightness_i)
            z_lat_neighbors_mean[i], z_lat_neighbors_std[i] = np.mean(z_lat_neighbors_i), np.std(z_lat_neighbors_i)
            z_lat_alignment_mean[i], z_lat_alignment_std[i] = np.nanmean(z_lat_alignment_i), np.nanstd(
                z_lat_alignment_i)
            z_lat_dist_mean[i], z_lat_dist_std[i] = np.nanmean(z_lat_dist_i), np.nanstd(z_lat_dist_i)
            z_lat_size_groups_mean[i], z_lat_size_groups_std[i] = np.nanmean(z_lat_size_groups_i), np.nanstd(
                z_lat_size_groups_i)
            z_lat_length_groups_mean[i], z_lat_length_groups_std[i] = np.nanmean(z_lat_length_groups_i), np.nanstd(
                z_lat_length_groups_i)
            z_lat_alignment_groups_mean[i], z_lat_alignment_groups_std[i] = np.nanmean(
                z_lat_alignment_groups_i), np.nanstd(z_lat_alignment_groups_i)

        # create and save dictionary for cell structure
        z_band_data = {'z_length': z_length, 'z_length_mean': z_length_mean, 'z_length_std': z_length_std,
                       'z_length_max': z_length_max, 'z_intensity': z_intensity, 'z_intensity_mean': z_intensity_mean,
                       'z_intensity_std': z_intensity_std, 'z_orientation': z_orientation, 'z_oop': z_oop,
                       'z_straightness': z_straightness, 'avg_intensity': avg_intensity, 'z_labels': z_labels,
                       'z_straightness_mean': z_straightness_mean, 'z_straightness_std': z_straightness_std,
                       'z_ratio_intensity': z_ratio_intensity, 'z_lat_neighbors': z_lat_neighbors,
                       'z_lat_neighbors_mean': z_lat_neighbors_mean, 'z_lat_neighbors_std': z_lat_neighbors_std,
                       'z_lat_alignment': z_lat_alignment, 'z_lat_alignment_mean': z_lat_alignment_mean,
                       'z_lat_alignment_std': z_lat_neighbors_std, 'z_lat_dist': z_lat_dist, 'z_ends': z_ends,
                       'z_lat_dist_mean': z_lat_dist_mean, 'z_lat_dist_std': z_lat_dist_std, 'z_links': z_links,
                       'z_lat_groups': z_lat_groups, 'z_lat_size_groups': z_lat_size_groups,
                       'z_lat_size_groups_mean': z_lat_size_groups_mean, 'z_lat_size_groups_std': z_lat_size_groups_std,
                       'z_lat_length_groups': z_lat_length_groups, 'z_lat_alignment_groups': z_lat_alignment_groups,
                       'z_lat_length_groups_mean': z_lat_length_groups_mean,
                       'z_lat_length_groups_std': z_lat_length_groups_std,
                       'z_lat_alignment_groups_mean': z_lat_alignment_groups_mean,
                       'z_lat_alignment_groups_std': z_lat_alignment_groups_std,
                       'params.z_timepoints': timepoints, 'params.z_threshold': threshold,
                       'params.z_min_length': min_length, 'params.z_end_radius': end_radius,
                       'params.z_theta_phi_min': theta_phi_min, 'params.z_d_max': d_max, 'params.z_d_min': d_min}
        self.structure.update(z_band_data)
        if self.auto_save:
            self.store_structure_data()

    def analyze_sarcomere_length_orient(self, timepoints='all', kernel='gaussian', size=3, sigma=0.15, width=0.5,
                                        len_lims=(1.5, 2.3), len_step=0.05, orient_lims=(-90, 90), orient_step=15,
                                        score_threshold=90, abs_threshold=False, gating=True, dilation_radius=3,
                                        save_all=False):
        """AND-gated double wavelet analysis of sarcomere structure

        Parameters
        ----------
        timepoints : int/list/str
            Timepoints for wavelet analysis ('all': all frames, int for single frame, list or ndarray for
            selected frames)
        kernel : str
            Filter kernel ('gaussian' for double Gaussian kernel, 'binary' for binary double-line)
        size : float
            Size of wavelet filters (in µm), needs to be larger than upper limit len_lims.
        sigma : float
            Minor axis width of single wavelets
        width : float
            Major axis width of single wavelets
        len_lims : tuple(float, float)
            Limits of lengths / wavelet distances in µm, range of sarcomere lengths
        len_step : float
            Step size of sarcomere lengths in µm
        orient_lims : tuple(float, float)
            Limits of sarcomere orientation angles in degree
        orient_step : float
            Step size of orientation angles in degree
        score_threshold : float
            Threshold score for clipping of length and orientation map (if abs_threshold=False, score_threshold is
            percentile (e.g., 90) for adaptive thresholding)
        abs_threshold : bool
            If True, absolute threshold value is applied, else if False, adaptive threshold based on percentile
        gating : bool
            If True, AND-gated wavelet filtering is used. If False, both wavelets filters are applied jointly.
        dilation_radius : int
            Radius of dilation for sarcomere area calculation, in pixels.
        save_all : bool
            If True, the wavelet filter results (wavelet_length_i, wavelet_orientation_i, wavelet_max_score) are stored.
            If False, only the points on the midlines are stored (recommended).
        """
        assert self.file_sarcomeres is not None, ("Z-band mask not found. Please run predict_z_bands first.")
        if timepoints == 'all':
            imgs = tifffile.imread(self.file_sarcomeres)
        elif isinstance(timepoints, int) or isinstance(timepoints, list) or type(timepoints) is np.ndarray:
            imgs = tifffile.imread(self.file_sarcomeres, key=timepoints)
            if isinstance(timepoints, int):
                timepoints = [timepoints]
        else:
            ValueError('timepoints argument not valid')
        if len(imgs.shape) == 2:
            imgs = np.expand_dims(imgs, 0)
        n_imgs = len(imgs)

        print('Start wavelet analysis')
        # create empty arrays
        (points, midline_length_points, midline_id_points, sarcomere_length_points,
         sarcomere_orientation_points, max_score_points) = [], [], [], [], [], []
        sarcomere_length_mean, sarcomere_length_std, sarcomere_length_median = np.zeros(n_imgs) * np.nan, np.zeros(
            n_imgs) * np.nan, np.zeros(
            n_imgs) * np.nan
        sarcomere_orientation_mean, sarcomere_orientation_std = np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan
        oop, mean_angle, sarcomere_area, sarcomere_area_ratio = np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan, \
                                                                np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan
        score_thresholds = np.zeros(n_imgs) * np.nan
        if save_all:
            wavelet_sarcomere_length, wavelet_sarcomere_orientation, wavelet_max_score = [], [], []
        else:
            wavelet_sarcomere_length, wavelet_sarcomere_orientation, wavelet_max_score = None, None, None

        # create filter bank
        bank, len_range, orient_range = create_wavelet_bank(pixelsize=self.metadata['pixelsize'], kernel=kernel,
                                                            size=size, sigma=sigma, width=width, len_lims=len_lims,
                                                            len_step=len_step, orient_lims=orient_lims,
                                                            orient_step=orient_step)

        # iterate images
        for i, img_i in enumerate(tqdm(imgs)):
            result_i = convolve_image_with_bank(img_i, bank, gating=gating)
            (wavelet_sarcomere_length_i, wavelet_sarcomere_orientation_i,
             wavelet_max_score_i) = argmax_wavelets(result_i,
                                                    len_range,
                                                    orient_range)
            if save_all:
                wavelet_sarcomere_length.append(wavelet_sarcomere_length_i)
                wavelet_sarcomere_orientation.append(wavelet_sarcomere_orientation_i)
                wavelet_max_score.append(wavelet_max_score_i)

            # evaluate wavelet results at sarcomere midlines
            (points_i, midline_id_points_i, midline_length_points_i, sarcomere_length_points_i,
             sarcomere_orientation_points_i, max_score_points_i, midline_i, score_threshold_i) = get_points_midline(
                wavelet_sarcomere_length_i, wavelet_sarcomere_orientation_i, wavelet_max_score_i,
                score_threshold=score_threshold,
                abs_threshold=abs_threshold)

            # write in list
            points.append(points_i)
            midline_length_points.append(midline_length_points_i * self.metadata['pixelsize'])
            midline_id_points.append(midline_id_points_i)
            sarcomere_length_points.append(sarcomere_length_points_i)
            sarcomere_orientation_points.append(sarcomere_orientation_points_i)
            max_score_points.append(max_score_points_i)
            score_thresholds[i] = score_threshold_i

            # calculate mean and std of sarcomere length and orientation
            sarcomere_length_mean[i], sarcomere_length_std[i], sarcomere_length_median[i] = np.mean(
                sarcomere_length_points_i), np.std(
                sarcomere_length_points_i), np.median(sarcomere_length_points_i)
            sarcomere_orientation_mean[i], sarcomere_orientation_std[i] = np.mean(
                sarcomere_orientation_points_i), np.std(sarcomere_orientation_points_i)
            # orientational order parameter
            if len(sarcomere_orientation_points_i) > 0:
                oop[i], mean_angle[i] = analyze_orientations(sarcomere_orientation_points_i)

            # calculate sarcomere area
            mask_i = sarcomere_mask(points_i, sarcomere_orientation_points_i, sarcomere_length_points_i,
                                    size=self.metadata['size'], pixelsize=self.metadata['pixelsize'],
                                    dilation_radius=dilation_radius)
            sarcomere_area[i] = np.sum(mask_i) * self.metadata['pixelsize'] ** 2
            if 'cell_area' in self.structure.keys():
                sarcomere_area_ratio[i] = sarcomere_area[i] / self.structure['cell_area'][i]
            else:
                area = self.metadata['size'][0] * self.metadata['size'][1] * self.metadata['pixelsize'] ** 2
                sarcomere_area_ratio[i] = sarcomere_area[i] / area

        wavelet_dict = {'params.wavelet_size': size, 'params.wavelet_sigma': sigma, 'params.wavelet_width': width,
                        'params.wavelet_len_lims': len_lims, 'params.wavelet_len_step': len_step,
                        'params.orient_lims': orient_lims, 'params.orient_step': orient_step, 'params.kernel': kernel,
                        'params.wavelet_timepoints': timepoints, 'wavelet_sarcomere_length': wavelet_sarcomere_length,
                        'wavelet_sarcomere_orientation': wavelet_sarcomere_orientation,
                        'wavelet_max_score': wavelet_max_score,
                        'points': points, 'sarcomere_length_points': sarcomere_length_points,
                        'midline_length_points': midline_length_points, 'midline_id_points': midline_id_points,
                        'sarcomere_length': sarcomere_length_points,
                        'sarcomere_orientation_points': sarcomere_orientation_points,
                        'sarcomere_orientation': sarcomere_orientation_points, 'max_score_points': max_score_points,
                        'sarcomere_area': sarcomere_area, 'sarcomere_area_ratio': sarcomere_area_ratio,
                        'sarcomere_length_mean': sarcomere_length_mean, 'sarcomere_length_std': sarcomere_length_std,
                        'sarcomere_length_median': sarcomere_length_median,
                        'sarcomere_orientation_mean': sarcomere_orientation_mean,
                        'sarcomere_orientation_std': sarcomere_orientation_std,
                        'sarcomere_oop': oop, 'sarcomere_mean_angle': mean_angle,
                        'params.score_threshold': score_thresholds, 'params.abs_threshold': abs_threshold,
                        'params.sarcomere_area_closing_radius': dilation_radius}
        self.structure.update(wavelet_dict)
        if self.auto_save:
            self.store_structure_data()

    def analyze_myofibrils(self, timepoints=None, n_seeds=500, score_threshold=None, persistence=3,
                           threshold_distance=0.3, n_min=5):
        """Estimate myofibril lines by line growth algorithm and analyze length and curvature

        timepoints : int/list/str
            Timepoints for myofibril analysis ('all': all frames, int for single frame, list or ndarray for
            selected frames), If None, timepoints from wavelet analysis are used.
        n_seeds : int
            Number of random seeds for line growth
        score_threshold : float
            Score threshold for random seeds (needs to be <=score_threshold from get_points_midline). If None,
            score_threshold from previous wavelet midline analysis is used.
        persistence : int
            Persistence of line (average points length and orientation for prior estimation)
        threshold_distance : float
            Maximal distance for nearest neighbor estimation (in micrometer)
        n_min : int
            Minimal number of sarcomere line segments per line. Shorter lines are removed.
        """
        assert 'points' in self.structure.keys(), ('Sarcomere length and orientation not yet analyzed. '
                                                   'Run analyze_sarcomere_length_orient first.')
        if score_threshold is None:
            if 'params.score_threshold' in self.structure.keys():
                score_threshold = self.structure['params.score_threshold']
            else:
                raise ValueError('To use score_threshold from wavelet analysis, run wavelet analysis first!')
        if timepoints is None:
            if 'params.wavelet_timepoints' in self.structure.keys():
                timepoints = self.structure['params.wavelet_timepoints']
            else:
                raise ValueError('To use timepoints from wavelet analysis, run wavelet analysis first!')
        if timepoints == 'all':
            n_imgs = self.metadata['frames']
            timepoints = np.arange(0, n_imgs)
        elif isinstance(timepoints, int):
            n_imgs = 1
            timepoints = [timepoints]
        elif isinstance(timepoints, list) or type(timepoints) is np.ndarray:
            n_imgs = len(timepoints)
        else:
            raise ValueError('Selection of timepoints not valid!')

        points = [self.structure['points'][t] for t in timepoints]
        sarcomere_length_points = [self.structure['sarcomere_length_points'][t] for t in timepoints]
        sarcomere_orientation_points = [self.structure['sarcomere_orientation_points'][t] for t in timepoints]
        midline_length_points = [self.structure['midline_length_points'][t] for t in timepoints]
        max_score_points = [self.structure['max_score_points'][t] for t in timepoints]

        # create empty arrays
        length_mean, length_median, length_std, length_max = np.zeros(n_imgs) * np.nan, np.zeros(
            n_imgs) * np.nan, np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan
        msc_mean, msc_median, msc_std = np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan
        myof_lines, lengths, msc = [], [], []

        # iterate timepoints
        print('Start myofibril line analysis!')
        for i, (points_i, sarcomere_length_points_i, sarcomere_orientation_points_i, max_score_points_i,
                midline_length_points_i) in enumerate(
            tqdm(
                zip(points, sarcomere_length_points, sarcomere_orientation_points, max_score_points,
                    midline_length_points),
                total=len(points))):
            line_data_i = line_growth(points_i, sarcomere_length_points_i, sarcomere_orientation_points_i,
                                      max_score_points_i, midline_length_points_t=midline_length_points_i,
                                      pixelsize=self.metadata['pixelsize'], n_seeds=n_seeds,
                                      persistence=persistence, threshold_distance=threshold_distance, n_min=n_min)
            lines_i = line_data_i['lines']
            myof_lines.append(lines_i)
            # line lengths and mean squared curvature (msc)
            lengths_i = line_data_i['line_features']['length_lines']
            msc_i = line_data_i['line_features']['msc_lines']
            if len(lengths_i) > 0:
                length_mean[i], length_median[i], length_std[i], length_max[i] = np.mean(lengths_i), np.median(
                    lengths_i), np.std(lengths_i), np.max(lengths_i)
                msc_mean[i], msc_median[i], msc_std[i] = np.mean(msc_i), np.median(msc_i), np.std(msc_i)
            lengths.append(lengths_i)
            msc.append(msc_i)

        # update structure dictionary
        myofibril_data = {'myof_length_mean': length_mean, 'myof_length_median': length_median,
                          'myof_length_std': length_std, 'myof_lines': myof_lines,
                          'myof_length_max': length_max, 'myof_length': lengths,
                          'myof_msc': msc, 'myof_msc_mean': msc_mean, 'myof_msc_median': msc_median,
                          'myof_msc_std': msc_std, 'params.n_seeds': n_seeds, 'params.persistence': persistence,
                          'params.threshold_distance': threshold_distance, 'params.myof_timepoints': timepoints}
        self.structure.update(myofibril_data)
        if self.auto_save:
            self.store_structure_data()

    def analyze_sarcomere_domains(self, timepoints=None, dist_threshold_ends=0.5, dist_threshold_midline_points=0.5,
                                  louvain_resolution=0.05, louvain_seed=2, area_min=50):
        """
            This function clusters sarcomeres into domains based on their spatial and orientational properties
            using the Louvain method for community detection.

            Parameters
            ----------
            timepoints : int/list/sr
                Timepoints for domain analysis ('all': all frames, int for single frame, list or ndarray for
                selected frames), If None, timepoints from wavelet analysis are used.
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
                Minimal area of domains / clusters (in µm^2). Area is calculated by convex hull.
            """
        assert 'points' in self.structure.keys(), ('Sarcomere length and orientation not yet analyzed. '
                                                   'Run analyze_sarcomere_length_orient first.')
        if timepoints is None:
            if 'params.wavelet_timepoints' in self.structure.keys():
                timepoints = self.structure['params.wavelet_timepoints']
            else:
                raise ValueError('To use timepoints from wavelet analysis, run wavelet analysis first!')
        if timepoints == 'all':
            n_imgs = self.metadata['frames']
            timepoints = np.arange(0, n_imgs)
        elif isinstance(timepoints, int):
            n_imgs = 1
            timepoints = [timepoints]
        elif isinstance(timepoints, list) or type(timepoints) is np.ndarray:
            n_imgs = len(timepoints)
        else:
            raise ValueError('Selection of timepoints not valid!')

        points = [self.structure['points'][t] * self.metadata['pixelsize'] for t in timepoints]
        sarcomere_length_points = [self.structure['sarcomere_length_points'][t] for t in timepoints]
        sarcomere_orientation_points = [self.structure['sarcomere_orientation_points'][t] for t in timepoints]
        max_score_points = [self.structure['max_score_points'][t] for t in timepoints]
        midline_id_points = [self.structure['midline_id_points'][t] for t in timepoints]

        # create empty arrays
        n_domains, domain_area_mean, domain_area_median, domain_area_std = np.zeros(
            n_imgs) * np.nan, np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan, np.zeros(n_imgs) * np.nan
        domain_slen_mean, domain_slen_median, domain_slen_std = np.zeros(n_imgs) * np.nan, np.zeros(
            n_imgs) * np.nan, np.zeros(n_imgs) * np.nan
        domain_oop_mean, domain_oop_median, domain_oop_std = np.zeros(n_imgs) * np.nan, np.zeros(
            n_imgs) * np.nan, np.zeros(n_imgs) * np.nan

        domains, domain_area, domain_slen, domain_slen_std, domain_oop, domain_orientation = [], [], [], [], [], []

        # iterate timepoints
        print('Start sarcomere domain analysis!')
        for t, (points_t, sarcomere_length_points_t, sarcomere_orientation_points_t,
                max_score_points_t, midline_id_points_i) in enumerate(
            tqdm(
                zip(points, sarcomere_length_points, sarcomere_orientation_points, max_score_points, midline_id_points),
                total=len(points))):
            cluster_data_t = cluster_sarcomeres(points_t, sarcomere_length_points_t, sarcomere_orientation_points_t,
                                                midline_id_points_i, dist_threshold_ends, dist_threshold_midline_points,
                                                louvain_resolution, louvain_seed=louvain_seed, area_min=area_min)
            (n_domains[t], domains_t, area_domains_t, sarcomere_length_mean_domains_t, sarcomere_length_std_domains_t,
             sarcomere_oop_domains_t, sarcomere_orientation_domains_t) = cluster_data_t

            # write single domain / cluster in lists
            domains.append(domains_t)
            domain_area.append(area_domains_t)
            domain_slen.append(sarcomere_length_mean_domains_t)
            domain_slen_std.append(sarcomere_length_std_domains_t)
            domain_oop.append(sarcomere_oop_domains_t)
            domain_orientation.append(sarcomere_orientation_domains_t)

            # calculate mean, median and std of domains
            domain_area_mean[t], domain_area_median[t], domain_area_std[t] = np.mean(area_domains_t), np.median(
                area_domains_t), np.std(area_domains_t)
            domain_slen_mean[t], domain_slen_median[t], domain_slen_std[t] = (np.mean(sarcomere_length_mean_domains_t),
                                                                              np.median(
                                                                                  sarcomere_length_mean_domains_t),
                                                                              np.std(sarcomere_length_mean_domains_t))
            domain_oop_mean[t], domain_oop_median[t], domain_oop_std[t] = (np.mean(sarcomere_oop_domains_t),
                                                                           np.median(sarcomere_oop_domains_t),
                                                                           np.std(sarcomere_oop_domains_t))

        # update structure dictionary
        domain_data = {'n_domains': n_domains, 'domains': domains,
                       'domain_area': domain_area, 'domain_area_mean': domain_area_mean,
                       'domain_area_median': domain_area_median, 'domain_area_std': domain_area_std,
                       'domain_slen': domain_slen, 'domain_slen_mean': domain_slen_mean,
                       'domain_slen_median': domain_slen_median, 'domain_slen_std': domain_slen_std,
                       'domain_oop': domain_oop, 'domain_oop_mean': domain_oop_mean,
                       'domain_oop_median': domain_oop_mean, 'domain_oop_std': domain_oop_std,
                       'domain_orientation': domain_orientation,
                       'params.domain_timepoints': timepoints,
                       'params.dist_threshold_ends': dist_threshold_ends,
                       'params.dist_threshold_midline_points': dist_threshold_midline_points,
                       'params.louvain_resolution': louvain_resolution,
                       'params.domain_area_min': area_min}

        self.structure.update(domain_data)
        if self.auto_save:
            self.store_structure_data()

    def _grow_roi_lines(self, timepoint=0, n_seeds=500, score_threshold=None, persistence=2, threshold_distance=0.5,
                        random_seed=None):
        """Find ROI lines using line growth algorithm. The parameters **lims can be used to filter ROI lines.

        Parameters
        ----------
        timepoint : int
            Timepoint to select frame. Selects i-th timepoint of timepoints specified in wavelet analysis.
        n_seeds : int
            Number of random seeds for line growth
        score_threshold : float
            Score threshold for random seeds (needs to be <=score_threshold from get_points_midline). If None, automated
            score_threshold from wavelet analysis is used.
        persistence : int
            Persistence of line (average points length and orientation for prior estimation)
        threshold_distance : float
            Maximal distance for nearest neighbor estimation
        number_lims : tuple(int, int)
            Limits of sarcomere numbers in ROI (n_min, n_max)
        length_lims : tuple(float, float)
            Limits for ROI lengths (in µm)
        sarcomere_mean_length_lims : tuple(float, float)
            Limits for mean length of sarcomeres in ROI
        sarcomere_std_length_lims : tuple(float, float)
            Limits for standard deviation of sarcomeres in ROI
        msc_lims : tuple(float, float)
            Limits for ROI line mean squared curvature (MSC)
        random_seed : int, optional

        """
        if score_threshold is None:
            if 'params.score_threshold' in self.structure.keys():
                if len(self.structure['params.score_threshold']) > 1:
                    score_threshold = self.structure['params.score_threshold'][timepoint]
                else:
                    score_threshold = self.structure['params.score_threshold']
            else:
                raise ValueError('To use score_threshold from wavelet analysis, run wavelet analysis first!')
        # select midline point data at timepoint
        (points, sarcomere_length_points,
         sarcomere_orientation_points, max_score_points, midline_length_points) = self.structure['points'][timepoint], \
            self.structure['sarcomere_length_points'][timepoint], \
            self.structure['sarcomere_orientation_points'][timepoint], \
            self.structure['max_score_points'][timepoint], \
            self.structure['midline_length_points'][timepoint]
        roi_data = line_growth(points, sarcomere_length_points, sarcomere_orientation_points, max_score_points,
                               midline_length_points, self.metadata['pixelsize'], n_seeds=n_seeds,
                               random_seed=random_seed, persistence=persistence, threshold_distance=threshold_distance)
        self.structure['roi_data'] = roi_data
        rois_points = [self.structure['points'][timepoint].T[roi_i] for roi_i in self.structure['roi_data']['lines']]
        self.structure['roi_data']['lines_points'] = rois_points
        if self.auto_save:
            self.store_structure_data()

    def _filter_roi_lines(self, number_lims=(10, 100), length_lims=(0, 200), sarcomere_mean_length_lims=(1, 3),
                          sarcomere_std_length_lims=(0, 0.4), msc_lims=(0, 1), midline_mean_length_lims=(2, 20),
                          midline_std_length_lims=(0, 5), midline_min_length_lims=(2, 20), max_orient_change=30):
        """
        Filters Regions of Interest (ROI) lines based on various geometric and morphological criteria.

        Parameters
        ----------
        number_lims : tuple of int
            Limits of sarcomere numbers in ROI (min, max).
        length_lims : tuple of float
            Limits for ROI lengths (in µm) (min, max).
        sarcomere_mean_length_lims : tuple of float
            Limits for mean length of sarcomeres in ROI (min, max).
        sarcomere_std_length_lims : tuple of float
            Limits for standard deviation of sarcomere lengths in ROI (min, max).
        msc_lims : tuple of float
            Limits for ROI line mean squared curvature (MSC) (min, max).
        midline_mean_length_lims : tuple of float
            Limits for mean length of the midline in ROI (min, max).
        midline_std_length_lims : tuple of float
            Limits for standard deviation of the midline length in ROI (min, max).
        midline_min_length_lims : tuple of float
            Limits for minimum length of the midline in ROI (min, max).

        Returns
        -------
        None
            Updates the 'is_good' field in the ROI data dict to reflect whether each ROI meets the specified criteria.
        """
        # Retrieve ROI lines and their features from the structure dict
        rois, roi_features = self.structure['roi_data']['lines'], self.structure['roi_data']['line_features']
        rois_points = self.structure['roi_data']['lines_points']

        # Apply filters based on the provided limits
        is_good = (
                (roi_features['n_points_lines'] >= number_lims[0]) & (roi_features['n_points_lines'] < number_lims[1]) &
                (roi_features['length_lines'] >= length_lims[0]) & (roi_features['length_lines'] < length_lims[1]) &
                (roi_features['sarcomere_mean_length_lines'] >= sarcomere_mean_length_lims[0]) &
                (roi_features['sarcomere_mean_length_lines'] < sarcomere_mean_length_lims[1]) &
                (roi_features['sarcomere_std_length_lines'] >= sarcomere_std_length_lims[0]) &
                (roi_features['sarcomere_std_length_lines'] < sarcomere_std_length_lims[1]) &
                (roi_features['msc_lines'] >= msc_lims[0]) & (roi_features['msc_lines'] < msc_lims[1]) &
                (roi_features['midline_mean_length_lines'] >= midline_mean_length_lims[0]) &
                (roi_features['midline_mean_length_lines'] < midline_mean_length_lims[1]) &
                (roi_features['midline_std_length_lines'] >= midline_std_length_lims[0]) &
                (roi_features['midline_std_length_lines'] < midline_std_length_lims[1]) &
                (roi_features['midline_min_length_lines'] >= midline_min_length_lims[0]) &
                (roi_features['midline_min_length_lines'] < midline_min_length_lims[1]) &
                (roi_features['max_orient_change_lines'] < np.radians(max_orient_change))
        )

        # remove bad lines
        self.structure['roi_data']['lines'] = [roi for i, roi in enumerate(rois) if is_good[i]]
        self.structure['roi_data']['lines_points'] = [points for i, points in enumerate(rois_points) if is_good[i]]
        df_features = pd.DataFrame(roi_features)
        filtered_df_features = df_features[is_good].reset_index(drop=True)
        self.structure['roi_data']['line_features'] = filtered_df_features.to_dict(orient='list')

    def _hausdorff_distance_rois(self, symmetry_mode='max'):
        """Compute Hausdorff distances between all good ROIs

        Parameters
        ----------
        timepoint : int
            Timepoint to select frame. Selects i-th timepoint of timepoints specified in wavelet analysis.
        symmetry_mode : str
            Choose 'min' or 'max', whether min/max(H(roi_i, roi_j), H(roi_j, roi_i))
        """
        # get points of roi lines
        lines_points = self.structure['roi_data']['lines_points']

        # hausdorff distance between ROIs
        hausdorff_dist_matrix = np.zeros((len(lines_points), len(lines_points)))
        for i, roi_i in enumerate(lines_points):
            for j, roi_j in enumerate(lines_points):
                if symmetry_mode == 'min':
                    hausdorff_dist_matrix[i, j] = min(directed_hausdorff(roi_i, roi_j)[0],
                                                      directed_hausdorff(roi_j, roi_i)[0])
                if symmetry_mode == 'max':
                    hausdorff_dist_matrix[i, j] = max(directed_hausdorff(roi_i, roi_j)[0],
                                                      directed_hausdorff(roi_j, roi_i)[0])

        self.structure['roi_data']['hausdorff_dist_matrix'] = hausdorff_dist_matrix
        if self.auto_save:
            self.store_structure_data()

    def _cluster_rois(self, distance_threshold_rois=40, linkage='single'):
        """Agglomerative clustering of good ROIs using predefined Hausdorff distance matrix using scikit-learn

        Parameters
        ----------
        distance_threshold_rois : float, default=40
            The linkage distance threshold above which, clusters will not be merged.
        linkage : {‘complete’, ‘average’, ‘single’}, default='single'
            Which linkage criterion to use. The linkage criterion determines which distance to use between sets of
            observation. The algorithm will merge the pairs of cluster that minimize this criterion.
            - ‘average’ uses the average of the distances of each observation of the two sets.
            - ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
            - ‘single’ uses the minimum of the distances between all observations of the two sets.
        plot : bool
            If True, graph with clustered good ROIs is shown.
        """
        if len(self.structure['roi_data']['lines_points']) == 0:
            self.structure['roi_data']['line_cluster'] = []
            self.structure['roi_data']['n_lines_clusters'] = 0
        elif len(self.structure['roi_data']['lines_points']) == 1:
            self.structure['roi_data']['line_cluster'] = [[0]]
            self.structure['roi_data']['n_lines_clusters'] = 1
        else:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold_rois,
                                                 affinity='precomputed',
                                                 linkage=linkage).fit(
                self.structure['roi_data']['hausdorff_dist_matrix'])
            self.structure['roi_data']['line_cluster'] = clustering.labels_
            self.structure['roi_data']['n_lines_clusters'] = len(np.unique(clustering.labels_))
        if self.auto_save:
            self.store_structure_data()

    def _fit_straight_line(self, add_length=2, n_longest=None):
        """Fit linear lines to cluster points

        Parameters
        ----------
        add_length : float
            Elongate line at end with add_length (in length unit)
        n_longest : int
            If int, only n longest ROIs are saved. If None, all are saved.
        """

        def linear(x, a, b):
            return a * x + b

        points_clusters = []
        roi_lines = []
        len_roi_lines = []
        add_length = add_length / self.metadata['pixelsize']
        for label_i in range(self.structure['roi_data']['n_lines_clusters']):
            points_cluster_i = []
            for k in np.where(self.structure['roi_data']['line_cluster'] == label_i)[0]:
                points_cluster_i.append(self.structure['roi_data']['lines_points'][k])
            points_clusters.append(np.concatenate(points_cluster_i).T)
            p_i, pcov_i = curve_fit(linear, points_clusters[label_i][1], points_clusters[label_i][0])
            x_range_i = np.linspace(np.min(points_clusters[label_i][1]) - add_length / np.sqrt(1 + p_i[0] ** 2),
                                    np.max(points_clusters[label_i][1]) + add_length / np.sqrt(1 + p_i[0] ** 2), num=2)
            y_i = linear(x_range_i, p_i[0], p_i[1])
            len_i = np.sqrt(np.diff(x_range_i) ** 2 + np.diff(y_i) ** 2)
            roi_lines.append(np.asarray((x_range_i, y_i)).T)
            len_roi_lines.append(len_i)

        len_roi_lines = np.asarray(len_roi_lines).flatten()
        roi_lines = np.asarray(roi_lines)

        # sort lines by length
        length_idxs = len_roi_lines.argsort()
        roi_lines = roi_lines[length_idxs[::-1]][:n_longest]
        len_roi_lines = len_roi_lines[length_idxs[::-1]][:n_longest]

        self.structure['roi_data']['roi_lines'] = np.asarray(roi_lines)
        self.structure['roi_data']['len_roi_lines'] = np.asarray(len_roi_lines)
        if self.auto_save:
            self.store_structure_data()

    def _longest_in_cluster(self, n_longest):
        lines = self.structure['roi_data']['lines']
        points = self.structure['points'][0][::-1]
        lines_cluster = np.asarray(self.structure['roi_data']['line_cluster'])
        longest_lines = []
        for label_i in range(self.structure['roi_data']['n_lines_clusters']):
            lines_cluster_i = [line for i, line in enumerate(lines) if lines_cluster[i] == label_i]
            points_lines_cluster_i = [points[:, line] for i, line in enumerate(lines) if
                                      lines_cluster[i] == label_i]
            length_lines_cluster_i = [len(line) for line in lines_cluster_i]
            longest_line = points_lines_cluster_i[np.argmax(length_lines_cluster_i)]
            longest_lines.append(longest_line)
        # get n longest lines
        len_longest_lines = [line.shape for line in longest_lines]
        sorted_by_length = sorted(longest_lines, key=lambda x: len(x[1].T), reverse=True)
        if len(longest_lines) < n_longest:
            print(f'Only {len(longest_lines)}<{n_longest} clusters identified.')
        roi_lines = sorted_by_length[:n_longest]
        roi_lines = [line.T for line in roi_lines]
        self.structure['roi_data']['roi_lines'] = roi_lines
        self.structure['roi_data']['len_roi_lines'] = [len(line.T) for line in roi_lines]
        if self.auto_save:
            self.store_structure_data()

    def create_roi_data(self, line, linewidth=0.65, order=0, export_raw=False):
        """
        Extract intensity kymograph along ROI and create ROI file from line.

        Parameters
        ----------
        line : ndarray
            Line start and end coordinate ((start_x, start_y), (end_x, end_y))
        linewidth : float
            Width of the scan in µm, perpendicular to the line
        order : int in {0, 1, 2, 3, 4, 5}, optional
            The order of the spline interpolation, default is 0 if
            image.dtype is bool and 1 otherwise. The order has to be in
            the range 0-5. See `skimage.transform.warp` for detail.
        export_raw : bool
            If True, intensity kymograph along ROI from raw microscopy image are additionally stored
        """
        imgs_sarcomeres = tifffile.imread(self.folder + 'sarcomeres.tif')
        profiles = kymograph_movie(imgs_sarcomeres, line, order=order,
                                   linewidth=int(linewidth / self.metadata['pixelsize']))
        profiles = np.asarray(profiles)
        if export_raw:
            imgs_raw = tifffile.imread(self.filename)
            profiles_raw = kymograph_movie(imgs_raw, line, order=order,
                                           linewidth=int(linewidth / self.metadata['pixelsize']))
        else:
            profiles_raw = None
        if os.path.exists(self.folder + 'imgs_calcium.tif'):
            imgs_calcium = tifffile.imread(self.folder + 'imgs_calcium.tif')
            profiles_calcium = kymograph_movie(imgs_calcium, line, order=order,
                                               linewidth=int(linewidth / self.metadata['pixelsize']))
        else:
            profiles_calcium = None

        # length of line
        def __calculate_segmented_line_length(points):
            diffs = np.diff(points, axis=0)
            lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            return np.sum(lengths)

        length = __calculate_segmented_line_length(line) * self.metadata['pixelsize']
        roi_data = {'profiles': profiles, 'profiles_raw': profiles_raw, 'profiles_calcium': profiles_calcium,
                    'line': line, 'linewidth': linewidth, 'length': length}
        for key, value in roi_data.items():
            if value is not None:
                roi_data[key] = np.asarray(value)
        save_name = self.folder + f'{line[0][0]}_{line[0][1]}_{line[-1][0]}_{line[-1][1]}_{linewidth}_roi.json'
        IOUtils.json_serialize(roi_data, save_name)

    def detect_rois(self, timepoint=0, n_seeds=1000, persistence=2, threshold_distance=0.3, score_threshold=None,
                    mode='longest_in_cluster', random_seed=None, number_lims=(10, 50), length_lims=(0, 200),
                    sarcomere_mean_length_lims=(1, 3), sarcomere_std_length_lims=(0, 1), msc_lims=(0, 1),
                    max_orient_change=30, midline_mean_length_lims=(2, 20), midline_std_length_lims=(0, 5),
                    midline_min_length_lims=(2, 20), distance_threshold_rois=40, linkage='single', n_longest=4,
                    linewidth=0.65, order=0, export_raw=False):
        """
        Detects Regions of Interest (ROIs) for tracking sarcomere Z-band motion and creates kymographs.

        This method integrates several steps: growing ROI lines based on seed points, filtering ROIs based on
        specified criteria, clustering ROIs, fitting lines to ROI clusters, and extracting intensity profiles
        to generate kymographs.

        Parameters
        ----------
        timepoint : int
            The index of the timepoint to select for analysis.
        n_seeds : int
            Number of seed points for initiating ROI line growth.
        persistence : int
            Persistence parameter influencing line growth direction and termination.
        threshold_distance : float
            Maximum distance for nearest neighbor estimation during line growth.
        score_threshold : float, optional
            Minimum score threshold for seed points. Uses automated threshold if None.
        mode : str
            Mode for selecting ROI lines from identified clusters.
            - 'fit_straight_line' fits a straight line to all points in the cluster.
            - 'longest_in_cluster' selects the longest line of each cluster, also allowing curved ROI lines.
        random_seed : int, optional
            Random seed for selection of random starting points for line growth algorithm, for reproducible outcomes.
            If None, no random seed is set, and outcomes in every run will differ.
        number_lims : tuple of int
            Limits for the number of sarcomeres within an ROI (min, max).
        length_lims : tuple of float
            Length limits for ROIs (in µm) (min, max).
        sarcomere_mean_length_lims : tuple of float
            Limits for the mean length of sarcomeres within an ROI (min, max).
        sarcomere_std_length_lims : tuple of float
            Limits for the standard deviation of sarcomere lengths within an ROI (min, max).
        msc_lims : tuple of float
            Limits for the mean squared curvature (MSC) of ROI lines (min, max).
        max_orient_change : float
            Maximal change of orientation between adjacent line segments, in degrees.
        midline_mean_length_lims : tuple of float
            Limits for the mean length of the midline of points in ROI (min, max).
        midline_std_length_lims : tuple of float
            Limits for the standard deviation of the midline length of points in ROI (min, max).
        midline_min_length_lims : tuple of float
            Limits for the minimum length of the midline of points in ROI (min, max).
        distance_threshold_rois : float
            Distance threshold for clustering ROIs. Clusters will not be merged above this threshold.
        linkage : str
            Linkage criterion for clustering ('complete', 'average', 'single').
        n_longest : int
            Number of longest ROIs to save. Saves all if None.
        linewidth : float
            Width of the scan line (in µm), perpendicular to the ROI line.
        order : int
            Order of spline interpolation for transforming ROI lines (range 0-5).
        export_raw : bool
            If True, exports raw intensity kymographs along ROI lines.

        Returns
        -------
        None
        """
        assert 'points' in self.structure.keys(), ('Sarcomere length and orientation not yet analyzed. '
                                                   'Run analyze_sarcomere_length_orient first.')
        # Grow ROI lines based on seed points and specified parameters
        self._grow_roi_lines(timepoint=timepoint, n_seeds=n_seeds, persistence=persistence,
                             threshold_distance=threshold_distance, score_threshold=score_threshold,
                             random_seed=random_seed)
        # Filter ROIs based on geometric and morphological criteria
        self._filter_roi_lines(number_lims=number_lims, length_lims=length_lims,
                               sarcomere_mean_length_lims=sarcomere_mean_length_lims,
                               sarcomere_std_length_lims=sarcomere_std_length_lims, msc_lims=msc_lims,
                               midline_mean_length_lims=midline_mean_length_lims,
                               midline_std_length_lims=midline_std_length_lims,
                               midline_min_length_lims=midline_min_length_lims,
                               max_orient_change=max_orient_change)
        # Calculate Hausdorff distance between ROI lines and perform clustering
        self._hausdorff_distance_rois()
        self._cluster_rois(distance_threshold_rois=distance_threshold_rois, linkage=linkage)
        # Fit lines to ROI clusters and select ROIs for analysis
        if mode == 'fit_straight_line':
            self._fit_straight_line(add_length=2, n_longest=n_longest)
        elif mode == 'longest_in_cluster':
            self._longest_in_cluster(n_longest=n_longest)

        # extract intensity kymographs profiles and save ROI files
        for line in self.structure['roi_data']['roi_lines']:
            self.create_roi_data(line, linewidth=linewidth, order=order, export_raw=export_raw)

    def full_analysis_structure(self, timepoints='all', save_all=False):
        """Analyze cell structure with default parameters

        Parameters
        ----------
        timepoints : int/list/str
            Timepoints for wavelet analysis ('all': all frames, int for single frame, list or ndarray for
            selected frames)
        save_all : bool
            If True, all intermediary data is saved. Can take up large storage, and is only recommended for visualizing
            data.
        """
        self.analyze_z_bands(timepoints=timepoints, save_all=save_all)
        self.analyze_sarcomere_length_orient(timepoints=timepoints, save_all=save_all)
        self.analyze_myofibrils(timepoints=timepoints)
        self.analyze_sarcomere_domains(timepoints=timepoints)
        if not self.auto_save:
            self.store_structure_data()


def segment_z_bands(image, threshold=0.15):
    """Segment z-bands from U-Net result (threshold, make binary, skeletonize, label regions)"""
    image_thres = image.copy().astype('uint16')
    image_thres[image >= 255 * threshold] = 255
    image_thres[image < 255 * threshold] = 0
    image_skel = morphology.skeletonize(image_thres / 255)
    image_skel_plot = image_skel.copy().astype('float32')
    image_skel_plot[image_skel_plot == 0] = np.nan
    labels = label(image_thres)
    labels_skel = image_skel * labels
    return labels, labels_skel


def analyze_z_bands(image_unet, labels, labels_skel, image_raw, pixelsize, min_length=1.0, threshold=0.1,
                    end_radius=0.75, theta_phi_min=0.25, d_max=5.0, d_min=0.25):
    """
    Analyzes segmented z-bands in a single frame, extracting metrics such as length, intensity, orientation,
    straightness, lateral distance, alignment, number of lateral neighbors per z-band, and characteristics of
    groups of lateral z-bands (length, alignment, size).

    Parameters
    ----------
    image_unet : ndarray
        The segmented image of z-bands.
    labels : ndarray
        The labeled image of z-bands.
    labels_skel : ndarray
        The skeletonized labels of z-bands.
    image_raw : ndarray
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
    Returns:
    --------
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
    labels = map_array(labels, labels_list_, labels_list)
    labels, forward_map, inverse_map = segmentation.relabel_sequential(labels)
    labels_list = labels_list[labels_list != 0]

    # analyze z-band labels
    props = regionprops_table(labels, intensity_image=image_raw, properties=['label', 'area', 'convex_area',
                                                                             'mean_intensity', 'orientation',
                                                                             'image', 'bbox'])
    # z-band length
    length = length[length > min_length]

    # straightness of z-bands (area/convex_hull)
    straightness = props['area'] / props['convex_area']

    # fluorescence intensity
    intensity = props['mean_intensity']

    # ratio sum(sarcomere intensity) to sum(background intensity)
    ratio_intensity, avg_intensity = intensity_sarcomeres(image_unet, image_raw, pixelsize=pixelsize,
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
        end_radius_px = int(end_radius / pixelsize)

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
                props_ends_i_1 = regionprops(mask_i_1 * img_i)[0]
                props_ends_i_2 = regionprops(mask_i_2 * img_i)[0]
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
        links = np.ones_like(D)
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

            for i in range(n_z):
                G.add_node(i)

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


def intensity_sarcomeres(image_unet, image_raw, pixelsize, threshold=0.1, plot=False):
    """Get ratio of sarcomere fluorescence to off-sarcomere fluorescence intensity

    Parameters
    ----------
    image_unet : ndarray
        U-Net result
    image_raw : ndarray
        Raw microscopy image
    pixelsize : float
        Size of pixel in x,y in um
    threshold : float
        Binary threshold for masks
    plot : bool
        If True, masked raw images are plotted
    """

    # binary mask of sarcomere by thresholding
    mask = image_unet.copy()
    mask[image_unet > 255 * threshold] = 1
    mask[image_unet <= 255 * threshold] = 0

    # apply mask to raw image
    image_sarcomeres = image_raw.copy()
    image_sarcomeres[mask == 0] = 0
    image_minus_sarcomeres = image_raw.copy()
    image_minus_sarcomeres[mask == 1] = 0

    # calculate ratio
    ratio_fluorescence_sarc = np.sum(image_sarcomeres) / np.sum(image_minus_sarcomeres)

    # calculate sum fluorescence image
    fluorescence = np.mean(image_raw) / (pixelsize ** 2)

    if plot:
        fig, ax = plt.subplots(figsize=(20, 10), nrows=2)
        ax[0].imshow(255 - image_sarcomeres, cmap='Greys')
        ax[1].imshow(255 - image_minus_sarcomeres, cmap='Greys')
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].set_title('Sarcomere fluorescence')
        ax[1].set_title('Off-sarcomere fluorescence')
        plt.show()

    return ratio_fluorescence_sarc, fluorescence


def binary_kernel(d, sigma, width, orient, size, pixelsize, mode='both'):
    """Returns binary kernel pair for AND-gated double wavelet analysis

    Parameters
    ----------
    d : float
        Distance between two wavelets
    sigma : float
        Minor axis width of single wavelets
    width : float
        Major axis with of single wavelets
    orient : float
        Rotation orientation in degree
    size : tuple(float, float)
        Size of kernel in µm
    pixelsize : float
        Pixelsize in µm
    mode : string
        'separate' returns two separate kernels, 'both' returns single kernel
    """
    # meshgrid
    size_pixel = round_up_to_odd(size / pixelsize)
    _range = np.linspace(-size / 2, size / 2, size_pixel, dtype='float32')
    x_mesh, y_mesh = np.meshgrid(_range, _range)
    # build kernel
    kernel0 = np.zeros_like(x_mesh)
    kernel0[np.abs((-x_mesh - d / 2)) < sigma / 2] = 1
    kernel0[np.abs(y_mesh) > width / 2] = 0
    kernel1 = np.zeros_like(x_mesh)
    kernel1[np.abs((x_mesh - d / 2)) < sigma / 2] = 1
    kernel1[np.abs(y_mesh) > width / 2] = 0
    kernel0 = ndimage.rotate(kernel0, orient, reshape=False, order=3)
    kernel1 = ndimage.rotate(kernel1, orient, reshape=False, order=3)
    if mode == 'separate':
        return kernel0, kernel1
    elif mode == 'both':
        return kernel0 + kernel1


def gaussian_kernel(dist, sigma, width, orient, size, pixelsize, mode='both'):
    """Returns gaussian kernel pair for AND-gated double wavelet analysis

    Parameters
    ----------
    dist : float
        Distance between two wavelets
    sigma : float
        Minor axis width of single wavelets in µm
    width : float
        Major axis with of single wavelets in µm
    orient : float
        Rotation orientation in degree
    size : tuple(float, float)
        Size of kernel in µm
    pixelsize : float
        Pixelsize in µm
    mode : string
        'separate' returns two separate kernels, 'both' returns single kernel
    """
    # meshgrid
    size_pixel = round_up_to_odd(size / pixelsize)
    _range = np.linspace(-size / 2, size / 2, size_pixel, dtype='float32')
    x_mesh, y_mesh = np.meshgrid(_range, _range)
    # build kernels
    kernel0 = (1 / (2 * np.pi * sigma * width) * np.exp(
        -((x_mesh - dist / 2) ** 2 / (2 * sigma ** 2) + y_mesh ** 2 / (2 * width ** 2))))
    kernel1 = (1 / (2 * np.pi * sigma * width) * np.exp(
        -((x_mesh + dist / 2) ** 2 / (2 * sigma ** 2) + y_mesh ** 2 / (2 * width ** 2))))
    # rotate kernels
    kernel0 = ndimage.rotate(kernel0, orient, reshape=False, order=1)
    kernel1 = ndimage.rotate(kernel1, orient, reshape=False, order=1)
    if mode == 'separate':
        return kernel0, kernel1
    elif mode == 'both':
        return kernel0 + kernel1


def round_up_to_odd(f):
    """Rounds float up to next odd integer"""
    return int(np.ceil(f) // 2 * 2 + 1)


def create_wavelet_bank(pixelsize, kernel='binary', size=3, sigma=0.15, width=0.5, len_lims=(1.3, 2.5),
                        len_step=0.025, orient_lims=(-90, 90), orient_step=5):
    """Returns bank of double wavelets

    Parameters
    ----------
    pixelsize : float
        Pixel size in µm
    kernel : str
        Filter kernel ('gaussian' for double Gaussian kernel, 'binary' for binary double-line)
    size : float
        Size of kernel in µm
    sigma : float
        Minor axis width of single wavelets
    width : float
        Major axis with of single wavelets
    len_lims : tuple(float, float)
        Limits of lengths / wavelet distances in µm
    len_step : float
        Step size in µm
    orient_lims : tuple(float, float)
        Limits of orientation angle in degree
    orient_step : float
        Step size in degree
    """

    len_range = np.arange(len_lims[0], len_lims[1], len_step, dtype='float32')
    orient_range = np.arange(orient_lims[0], orient_lims[1], orient_step, dtype='float32')
    size_pixel = round_up_to_odd(size / pixelsize)

    bank = np.zeros((len_range.shape[0], orient_range.shape[0], 2, size_pixel, size_pixel))
    for i, d in enumerate(len_range):
        for j, orient in enumerate(orient_range):
            if kernel == 'gaussian':
                bank[i, j] = gaussian_kernel(d, sigma, width, orient, size, pixelsize, mode='separate')
            elif kernel == 'binary':
                bank[i, j] = binary_kernel(d, sigma, width, orient, size, pixelsize, mode='separate')
            else:
                raise ValueError(f'kernel {kernel} not valid!')
    return bank, len_range, orient_range


def custom_convolve(image, filters):
    """2D convolution of image with custom kernel/weights

    Parameters
    ----------
    image : ndarray
        2D image
    filters : ndarray
        2D convolution kernels
    """
    image_torch = torch.from_numpy(image.astype('float32')).to(device).view(1, 1, image.shape[0], image.shape[1])
    filters_torch = torch.from_numpy(filters.astype('float32')).to(device).view(filters.shape[0] * filters.shape[1], 1,
                                                                                filters.shape[2], filters.shape[3])
    result = F.conv2d(image_torch, filters_torch, padding='same').cpu().numpy()
    return result.reshape(filters.shape[0], filters.shape[1], image.shape[0], image.shape[1])


def convolve_image_with_bank(image, bank, gating=True):
    """AND-gated double-wavelet convolution of image using kernels from filter bank"""
    if gating:
        res0 = custom_convolve(image / 255, filters=bank[:, :, 0])
        res1 = custom_convolve(image / 255, filters=bank[:, :, 1])
        return res0 * res1
    else:
        res = custom_convolve(image / 255, filters=bank[:, :, 0] + bank[:, :, 1])
        return res


def argmax_wavelets(result, len_range, orient_range):
    """Argmax of wavelet convolution result to get length, orientation and max max_score map

    Parameters
    ----------
    result : ndarray
        Result from convolve_image_with_bank function
    len_range : ndarray
        List of lengths used for bank
    orient_range : ndarray
        List of orientation angles used for bank
    """
    result_ = torch.from_numpy(np.transpose(result, [2, 3, 0, 1]))
    result_ = result_.view((result.shape[2] * result.shape[3], -1))
    max_score, argmax = torch.max(result_, 1)
    max_score = max_score.view(result.shape[2], result.shape[3]).cpu().numpy()
    argmax = argmax.cpu().numpy()
    indices = np.unravel_index(argmax, (result.shape[0], result.shape[1]))
    length = len_range[indices[0]].reshape(result.shape[2], result.shape[3])
    orient = orient_range[indices[1]].reshape(result.shape[2], result.shape[3])

    return length, np.radians(orient), max_score


def get_points_midline(length, orientation, max_score, score_threshold=90., abs_threshold=False):
    """
    Extracts points on sarcomere midlines and calculates sarcomere length and orientation at these points.

    This function performs the following steps:
    1. **Thresholding:** Applies a threshold to the length, orientation, and max_score arrays to refine sarcomere detection.
    2. **Binarization:** Creates a binary mask to isolate midline regions.
    3. **Skeletonization:** Thins the midline regions for easier analysis.
    4. **Labeling:** Assigns unique labels to each connected midline component.
    5. **Midline Point Extraction:** Identifies the coordinates of points along each midline.
    6. **Value Calculation:** Calculates sarcomere length, orientation, and maximal score at each midline point.

    **Parameters**
    ----------
    length : ndarray
        Sarcomere length map obtained from wavelet analysis.
    orientation : ndarray
        Sarcomere orientation angle map obtained from wavelet analysis.
    max_score : ndarray
        Map of maximal wavelet scores.
    score_threshold : float, optional
        Threshold for filtering detected sarcomeres. Can be either an absolute value (if abs_threshold=True) or
        a percentile value for adaptive thresholding (if abs_threshold=False). Default is 90.
    abs_threshold : bool, optional
        Flag to determine the thresholding method. If True, 'score_threshold' is used as an absolute value.
        If False, 'score_threshold' is interpreted as a percentile for adaptive thresholding. Default is False.

    **Returns**
    -------
    tuple
        * **points** (list): List of (x, y) coordinates for each midline point.
        * **midline_id_points** (list): List of corresponding midline labels for each point.
        * **midline_length_points** (list): List of approximate midline lengths associated with each point. In pixels.
        * **sarcomere_length_points** (list): List of sarcomere lengths at each midline point.
        * **sarcomere_orientation_points** (list): List of sarcomere orientation angles at each midline point.
        * **max_score_points** (list): List of maximal wavelet scores at each midline point.
        * **midline** (ndarray): The binarized midline mask.
        * **score_threshold** (float): The final threshold value used.
    """
    # threshold sarcomere length and orientation array
    length_thres = length.copy().astype('float')
    orientation_thres = orientation.copy().astype('float')
    max_score_ = max_score.copy()
    # rough thresholding of sarcomere structures to better identify adaptive threshold
    max_score_[max_score <= np.percentile(max_score, 1)] = np.nan
    # determine adaptive threshold from value distribution
    if not abs_threshold:
        score_threshold = np.nanpercentile(max_score_, score_threshold)
    length_thres[max_score < score_threshold] = np.nan
    orientation_thres[max_score < score_threshold] = np.nan

    # binarize midline
    midline = np.zeros_like(max_score)
    midline[max_score < score_threshold] = 0
    midline[max_score >= score_threshold] = 1

    # skeletonize
    midline_skel = morphology.skeletonize(midline)

    # label midlines
    midline_labels = ndimage.label(midline_skel, ndimage.generate_binary_structure(2, 2))[0]

    # iterate midlines and create additional list with labels and midline length (approximated by max. Feret diameter)
    props = skimage.measure.regionprops_table(midline_labels, properties=['label', 'coords', 'feret_diameter_max'])
    list_labels, coords_midlines, length_midlines = props['label'], props['coords'], props['feret_diameter_max']

    points, midline_id_points, midline_length_points = [], [], []
    for i, (label_i, coords_i, length_midline_i) in enumerate(zip(list_labels, coords_midlines, length_midlines)):
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

    return (points, midline_id_points, midline_length_points, sarcomere_length_points, sarcomere_orientation_points,
            max_score_points, midline, score_threshold)


def cluster_sarcomeres_old(points_t, sarcomere_length_points_t, sarcomere_orientation_points_t, max_score_points_t,
                           pixelsize, score_threshold=0, reduce=3, weight_length=1, distance_threshold=3, area_min=200):
    """
    Find sarcomere domains with similar orientation using agglomerative clustering

    Parameters
    ----------
    points_t : ndarray
        List of sarcomere midline point positions
    sarcomere_length_points_t : ndarray
        List of midline point sarcomere lengths
    sarcomere_orientation_points_t : ndarray
        List of midline point sarcomere orientations, in radians
    max_score_points_t : ndarray
        List of midline point max score values
    pixelsize : float
        Pixel size in µm
    score_threshold : float
        Threshold (percentile of max_score distribution) for filtering points based on max_score
    reduce : int
        Reduce number of samples by subsampling with factor reduce
    weight_length : float
        Weight of differences in sarcomere length between points for distance function (0 = no contribution)
    distance_threshold : float
        Distance threshold for agglomerative clustering.
    area_min : float
        Minimal area of domains / clusters (in µm^2). Area is calculated by convex hull.

    Returns
    -------
    n_clusters : int
        Number of clusters / domains
    points_clusters : ndarray
        Positions of points contained in cluster
    labels_clusters : ndarray
        Labels of points contained in cluster
    area_clusters : list
        Area of clusters
    length_clusters :  list
        Mean sarcomere length inside each cluster
    oop_clusters : list
        Orientational order parameter of points in each cluster
    orientation_clusters : list
        Main orientation of clusters
    """

    # filter points for max score
    if score_threshold != 0 and len(max_score_points_t) > 10:
        _score_threshold = np.percentile(max_score_points_t, score_threshold)
        points_t = points_t[:, max_score_points_t > _score_threshold]
        sarcomere_length_points_t = sarcomere_length_points_t[max_score_points_t > _score_threshold]
        sarcomere_orientation_points_t = sarcomere_orientation_points_t[max_score_points_t > _score_threshold]

    # reduce number of points by factor "reduce"
    if isinstance(reduce, int) and reduce >= 1:
        points_t = points_t[:, ::reduce] * pixelsize  # convert to real space
        sarcomere_length_points_t = sarcomere_length_points_t[::reduce]
        sarcomere_orientation_points_t = sarcomere_orientation_points_t[::reduce]
        orientation_vectors_t = np.asarray(
            [np.cos(sarcomere_orientation_points_t), np.sin(sarcomere_orientation_points_t)])
    else:
        raise ValueError(f'reduce needs to be an integer >= 1')

    if len(points_t.T) > 10:
        # compute distance matrix
        # 1. compute euclidean distance
        dist_eucl = squareform(pdist(points_t.T, 'euclidean'))
        # 2. compute cosine distance
        dist_cosine = np.abs(1 - squareform(pdist(orientation_vectors_t.T, 'cosine'))) ** 4
        # 3. distance length (1 if equal)
        dist_length = 1 + weight_length * np.abs(
            squareform(pdist(np.expand_dims(sarcomere_length_points_t, axis=1), 'euclidean')))

        # 3. compute custom distance
        dist = dist_eucl / dist_cosine * dist_length
        # remove inf values
        dist[np.isinf(dist)] = 1000

        # accumulative clustering
        if dist.shape[0] > 20:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                                 affinity='precomputed',
                                                 linkage='single').fit(dist)
            labels_clusters = clustering.labels_
            n_clusters = clustering.n_clusters_
        else:
            labels_clusters = np.zeros(dist.shape[0])
            n_clusters = 0

        # convex hull to estimate cluster area
        _area_clusters = np.zeros(n_clusters)
        for i, label_i in enumerate(np.unique(labels_clusters)):
            points_i = points_t.T[labels_clusters == label_i]
            if len(points_i) > 20:
                hull_i = ConvexHull(points_i)
                _area_clusters[i] = hull_i.volume

        # remove small clusters
        points_clusters = points_t.T[np.isin(labels_clusters, np.argwhere(_area_clusters >= area_min).T[0])]
        _length_clusters = sarcomere_length_points_t.T[
            np.isin(labels_clusters, np.argwhere(_area_clusters >= area_min).T[0])]
        _orientation_clusters = sarcomere_orientation_points_t.T[
            np.isin(labels_clusters, np.argwhere(_area_clusters >= area_min).T[0])]
        labels_clusters = labels_clusters[np.isin(labels_clusters, np.argwhere(_area_clusters >= area_min).T[0])]
        n_clusters = len(np.unique(labels_clusters))

        # quantify clusters
        orientation_clusters, oop_clusters, length_clusters = np.zeros(n_clusters), np.zeros(n_clusters), np.zeros(
            n_clusters)
        area_clusters = np.zeros(n_clusters)
        for i, label_i in enumerate(np.unique(labels_clusters)):
            points_i = points_clusters[labels_clusters == label_i]
            lengths_i = _length_clusters[labels_clusters == label_i]
            orientations_i = _orientation_clusters[labels_clusters == label_i]
            if len(points_i) > 10:
                hull_i = ConvexHull(points_i)
                area_clusters[i] = hull_i.volume
            length_clusters[i] = np.mean(lengths_i)
            oop, angle = analyze_orientations(orientations_i)
            oop_clusters[i] = oop
            orientation_clusters[i] = angle
        return (n_clusters, points_clusters, area_clusters, length_clusters, oop_clusters,
                orientation_clusters)
    else:
        return 0, [], [], [], [], []


def cluster_sarcomeres(points, sarcomere_length_points, sarcomere_orientation_points, midline_id_points,
                       dist_threshold_ends=0.5, dist_threshold_midline_points=0.5, louvain_resolution=0.05,
                       louvain_seed=2, area_min=50):
    """
    This function clusters sarcomeres into domains based on their spatial and orientational properties
    using the Louvain method for community detection. It considers sarcomere lengths, orientations,
    and positions along midlines to form networks of connected sarcomeres. Domains are then identified
    as communities within these networks, with additional criteria for minimum domain area
    and connectivity thresholds. Finally, this function quantifies the mean and std of sarcomere lengths,
    and the orientational order parameter and mean orientation of each domain.

    Parameters
    ----------
    points : ndarray
        List of sarcomere midline point positions
    sarcomere_length_points : ndarray
        List of midline point sarcomere lengths
    sarcomere_orientation_points : ndarray
        List of midline point sarcomere orientations, in radians
    midline_id_points : ndarray
        List of midline point indices, points of the same midline have same index.
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
        Minimal area of domains / clusters (in µm^2). Area is calculated by convex hull.

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
            return np.cos(orient_i - orient_j) ** 2

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
        random.shuffle(domains)

        # convex hull to estimate domain areas and remove small domains
        _area_domains = np.zeros(n_domains) * np.nan
        _indices_to_remove = []
        for i, domain_i in enumerate(domains):
            points_i = points[:, list(domain_i)].T
            if len(points_i) > 10:
                hull_i = ConvexHull(points_i)
                area_i = hull_i.volume
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
        for i, domain_i in enumerate(domains):
            points_i = points[:, list(domain_i)].T
            lengths_i = sarcomere_length_points[list(domain_i)]
            orientations_i = sarcomere_orientation_points[list(domain_i)]
            hull_i = ConvexHull(points_i)
            area_domains[i] = hull_i.volume
            sarcomere_length_mean_domains[i] = np.mean(lengths_i)
            sarcomere_length_std_domains[i] = np.std(lengths_i)
            oop, angle = analyze_orientations(orientations_i)
            sarcomere_oop_domains[i] = oop
            sarcomere_orientation_domains[i] = angle
        return (n_domains, domains, area_domains, sarcomere_length_mean_domains, sarcomere_length_std_domains,
                sarcomere_oop_domains, sarcomere_orientation_domains)
    else:
        return 0, [], [], [], [], [], []


# function to grow lines along orientation of vector field
def _grow_line(seed, points_t, sarcomere_length_points_t, sarcomere_orientation_points_t, nbrs, threshold_distance,
               pixelsize, persistence):
    line_i = [seed]
    stop_right = stop_left = False

    # threshold_distance from micrometer to pixels
    threshold_distance_pixels = threshold_distance / pixelsize

    while not stop_left or not stop_right:
        n_i = len(line_i)
        if n_i == 1:
            end_left = end_right = points_t[:, seed]
            length_left = length_right = sarcomere_length_points_t[seed] / pixelsize
            orientation_left = orientation_right = sarcomere_orientation_points_t[seed]
        elif n_i > 1:
            if not stop_left:
                end_left = points_t[:, line_i[0]]
                if persistence > 1:
                    length_left = np.mean(sarcomere_length_points_t[line_i[:persistence]]) / pixelsize
                    orientation_left = stats.circmean(
                        sarcomere_orientation_points_t[line_i[:persistence]])
                else:
                    length_left = sarcomere_length_points_t[line_i[0]] / pixelsize
                    orientation_left = sarcomere_orientation_points_t[line_i[0]]
            if not stop_right:
                end_right = points_t[:, line_i[-1]]
                if persistence > 1:
                    length_right = np.mean(sarcomere_length_points_t[line_i[-persistence:]]) / pixelsize
                    orientation_right = stats.circmean(
                        sarcomere_orientation_points_t[line_i[-persistence:]])
                else:
                    length_right = sarcomere_length_points_t[line_i[-1]] / pixelsize
                    orientation_right = sarcomere_orientation_points_t[line_i[-1]]

        # grow left
        if not stop_left:
            prior_left = [end_left[0] + np.sin(orientation_left) * length_left,
                          end_left[1] - np.cos(orientation_left) * length_left]
            # nearest neighbor left
            distance_left, index_left = nbrs.kneighbors([prior_left])
            # extend list
            if distance_left < threshold_distance_pixels:
                line_i.insert(0, index_left[0][0].astype('int'))
            else:
                stop_left = True
        # grow right
        if not stop_right:
            prior_right = [end_right[0] - np.sin(orientation_right) * length_right,
                           end_right[1] + np.cos(orientation_right) * length_right]
            # nearest neighbor left
            distance_right, index_right = nbrs.kneighbors([prior_right])
            # extend list
            if distance_right < threshold_distance_pixels:
                line_i.append(index_right[0][0].astype('int'))
            else:
                stop_right = True

    return np.asarray(line_i)


def line_growth(points_t, sarcomere_length_points_t, sarcomere_orientation_points_t, max_score_points_t,
                midline_length_points_t, pixelsize, n_seeds=5000, random_seed=None, persistence=4,
                threshold_distance=0.3, n_min=5):
    """
    Line growth algorithm to determine myofibril lines perpendicular to sarcomere z-bands

    Parameters
    ----------
    points_t : list
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
        Persistence of line (average points length and orientation for prior estimation)
    threshold_distance : float
        Maximal distance for nearest neighbor estimation (in micrometer)
    n_min : int
        Minimal number of sarcomere line segments per line. Shorter lines are removed.

    Returns
    -------
    line_data : dict
        Dictionary with ROI data keys = (lines, line_features)
    """
    # select random origins for line growth
    random.seed(random_seed)
    seed_idx = random.sample(range(len(points_t.T)), n_seeds)

    # Precompute Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_t.T)

    # Prepare arguments for parallel processing
    args = [
        (seed, points_t, sarcomere_length_points_t, sarcomere_orientation_points_t, nbrs, threshold_distance, pixelsize,
         persistence) for seed in seed_idx]

    # Use multiprocessing Pool to grow lines in parallel
    with Pool() as pool:
        lines = pool.starmap(_grow_line, args)

    # Remove short lines (< n_min)
    lines = [line for line in lines if len(line) >= n_min]

    # remove short lines (< n_min)
    lines = [l for l in lines if len(l) >= n_min]
    # calculate features of lines
    n_points_lines = np.asarray([len(line) for line in lines])  # number of sarcomere in line
    length_line_segments = [sarcomere_length_points_t[line] for line in lines]
    length_lines = [np.sum(lengths) for lengths in length_line_segments]
    # sarcomere lengths
    sarcomere_mean_length_lines = [np.mean(sarcomere_length_points_t[line]) for line in lines]
    sarcomere_std_length_lines = [np.std(sarcomere_length_points_t[line]) for line in lines]
    # midline lengths
    midline_mean_length_lines = [np.nanmean(midline_length_points_t[line]) for line in lines]
    midline_std_length_lines = [np.nanstd(midline_length_points_t[line]) for line in lines]
    midline_min_length_lines = [np.nanmin(midline_length_points_t[line]) for line in lines]
    # wavelet scores
    mean_score_lines = [np.mean(max_score_points_t[line]) for line in lines]
    std_score_lines = [np.std(max_score_points_t[line]) for line in lines]
    # mean squared curvature
    tangential_vector_line_segments = [np.diff(points_t.T[line], axis=0) for line in lines]
    tangential_angle_line_segments = [np.asarray([np.arctan2(v[1], v[0]) for v in vectors]) for vectors in
                                      tangential_vector_line_segments]
    curvature_line_segments = [np.diff(angle_i) / length_line_segments[i][1:-1] for i, angle_i in
                               enumerate(tangential_angle_line_segments)]
    msc_lines = [np.sum(curvature_line_segments[i] ** 2) / length_lines[i] for i in range(len(lines))]
    # mean and std orientation, and maximal change of orientation
    mean_orient_lines = [stats.circmean(sarcomere_orientation_points_t[line]) for line in lines]
    std_orient_lines = [stats.circstd(sarcomere_orientation_points_t[line]) for line in lines]
    max_orient_change_lines = [max_orientation_change(sarcomere_orientation_points_t[line]) for line in lines]
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
    line_features = convert_lists_to_arrays_in_dict(line_features)
    line_data = {'lines': lines, 'line_features': line_features}
    return line_data


def kymograph_movie(movie, line, linewidth=10, order=0):
    """
    Generate a kymograph using multiprocessing.

    Parameters
    --------
    movie : array_like
        The movie.
    line : array_like, shape (N, 2)
        The coordinates of the segmented line (N>1)
    linewidth : int, optional
        Width of the scan in pixels, perpendicular to the line
    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns:
    return_value : array
        Kymograph along segmented line

    Notes
    -------
    Adapted from scikit-image
    (https://scikit-image.org/docs/0.22.x/api/skimage.measure.html#skimage.measure.profile_line).
    """
    line = line[:, ::-1]
    # prepare coordinates of segmented line
    perp_lines = _curved_line_profile_coordinates(points=line, linewidth=linewidth)

    # Prepare arguments for each frame
    args = [(movie[frame], perp_lines, linewidth, order) for frame in range(movie.shape[0])]

    # Create a Pool and map process_frame to each frame
    with Pool() as pool:
        results = pool.map(_process_frame, args)

    # Convert list of results to a numpy array
    kymograph = np.array(results)

    return kymograph


def _process_frame(args):
    frame, perp_lines, linewidth, order = args
    pixels = ndimage.map_coordinates(frame, perp_lines, prefilter=order > 1,
                                     order=order, mode='reflect', cval=0.0)
    pixels = np.flip(pixels, axis=1)
    intensities = np.mean(pixels, axis=1)
    return intensities


def _curved_line_profile_coordinates(points, linewidth=10):
    """
    Calculate the coordinates of a curved line profile composed of multiple segments with specified linewidth.

    Parameters
    ----------
    points : array_like
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


def sarcomere_mask(points, sarcomere_orientation_points, sarcomere_length_points, size, pixelsize,
                   dilation_radius=3):
    """
    Calculates a binary mask of areas with sarcomeres.

    Parameters
    ----------
    points : ndarray
        Position of sarcomere vectors
    sarcomere_orientation_points : ndarray
        Orientations of sarcomere vectors
    sarcomere_length_points : ndarray
        Lengths of sarcomere vectors
    size : tuple
        Size of image, in pixels
    pixelsize : float
        Pixelsize in µm
    dilation_radius : int
        Dilation radius to close small holes

    Returns
    -------
    mask : ndarray
        Binary mask of sarcomeres
    """
    # Calculate orientation vectors using trigonometry
    orientation_vectors = np.asarray([-np.sin(sarcomere_orientation_points),
                                      np.cos(sarcomere_orientation_points)])
    # Calculate the ends of the vectors based on their orientation and length
    points = points * pixelsize
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
