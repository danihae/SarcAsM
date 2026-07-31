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

"""Main :class:`SarcAsM` object: sarcomere morphology, full-field 2D tracking, and grouped motion analysis."""

import glob
import hashlib
import logging
import os
import shutil
import warnings
from typing import Optional, Tuple, Union, List, Literal, Any

import numpy as np
import pandas as pd
import torch
from bio_image_unet.progress import ProgressNotifier
from scipy import stats, sparse

from sarcasm.core import SarcAsMBase
from sarcasm.io.results_store import ResultsDict, Results, export_to_json
from sarcasm.utils import Utils

logger = logging.getLogger(__name__)

# Import structure modules
from sarcasm.analysis import (
    z_band_analysis,
    sarcomere_vectors,
    myofibril_analysis,
    domain_clustering,
    contraction_analysis,
    detection,
    loi_detection,
    sarcomere_tracking,
    grouped_motion,
)

class SarcAsM(SarcAsMBase):
    """
    Analyze sarcomere morphology, full-field 2D tracking, and grouped motion.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to the image TIFF file.
    restart : bool, optional
        If True, delete the previous analysis folder and start fresh.
        Default is False.
    pixelsize : float or None, optional
        Physical pixel size in µm. If None, taken from file metadata; an
        explicit value overrides the metadata. Default is None.
    frametime : float or None, optional
        Time between frames in s. If None, taken from metadata; an explicit
        value overrides it. Default is None.
    channel : int or None, optional
        Index of the fluorescence channel that shows the sarcomeres. Ignored
        for single-channel images. Default is None.
    axes : str or None, optional
        Explicit dimension order (e.g. ``'TXYC'``). None lets the base class
        auto-detect the order. Default is None.
    auto_save : bool, optional
        Write analysis results to disk automatically. Default is True.
    use_gui : bool, optional
        Activate GUI mode. Default is False.
    device : torch.device or {'auto'}, optional
        Device on which PyTorch kernels are executed. ``'auto'`` selects CUDA
        or MPS when available. Default is ``'auto'``.
    **info
        Additional key-value pairs stored in the metadata file.

    Attributes
    ----------
    data : ResultsDict
        Lazy, Zarr-backed store of morphology, tracking, and motion results
        (populated after running the respective routines).
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
        Instantiate a SarcAsM object and initialize the common SarcAsMBase.
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

        # Initialize structure data store (lazy Zarr-backed; migrates legacy JSON)
        self._load_structure_data()

    def __get_store_path(self) -> str:
        """Path to the analysis subgroup of the OME-Zarr store (``<name>.ome.zarr/sarcasm``)."""
        return str(self.store.results_path)

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
        """Persist any staged result writes to the Zarr store."""
        self.store_structure_data()

    def store_structure_data(self, override: bool = True) -> None:
        """Persist structure/track results to the Zarr store (incremental).

        Only members changed since the last call are rewritten. Image metadata
        is mirrored into the store so it is a single, self-contained artifact.

        Parameters
        ----------
        override : bool, optional
            If False and a store already exists, do nothing. Default is True.
        """
        if not override and os.path.exists(self.__get_store_path()):
            return
        if not isinstance(self.data, ResultsDict):
            # defensive: a caller replaced self.data with a plain dict
            self.data = ResultsDict(self.__get_store_path(), initial=self.data)
        self.data.flush()
        self.save_metadata()  # mirror metadata into the OME-Zarr store root

    @property
    def results(self) -> Results:
        """Grouped, lazy, read-only view of the results store.

        Examples
        --------
        ``self.results.tracks.slen[i]`` reads a single track's length series;
        ``self.results.structure.sarcomere.oop`` and
        ``self.results.params.track_sarcomere_vectors.frames`` read analysis
        outputs and parameters. Staged writes are flushed first.
        """
        if not isinstance(self.data, ResultsDict):
            self.data = ResultsDict(self.__get_store_path(), initial=self.data)
        return self.data.view()

    def export_json(self, path: Optional[str] = None, *, keys=None,
                    include_arrays: bool = True) -> str:
        """Export results to a legacy-format JSON file (reloadable by old code).

        Parameters
        ----------
        path : str or None, optional
            Output path. Default is None, which uses the historical
            ``structure.json`` location.
        keys : list of str or None, optional
            Subset of result keys to export. Default is None (all keys).
        include_arrays : bool, optional
            If False, skip large arrays for a readable dump. Default is True.

        Returns
        -------
        str
            Path to the written JSON file.
        """
        if path is None:
            path = self.__get_structure_data_file(is_temp_file=False)
        return str(export_to_json(self.data, path, keys=keys, include_arrays=include_arrays))

    def _load_structure_data(self) -> None:
        """Bind the lazy results store (the ``sarcasm/`` group of the OME-Zarr store)."""
        self.data = ResultsDict(self.__get_store_path())

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

        # persist a migration / compatibility upgrade to the store
        if self.auto_save and getattr(self.data, "_dirty", None):
            self.store_structure_data()

    def detect_sarcomeres(self, frames: Union[str, int, List[int], np.ndarray] = 'all',
                          model_path: str = None, max_patch_size: Union[Tuple[int, int], str] = 'auto',
                          normalization_mode: str = 'all', clip_thres: Tuple[float, float] = (0., 99.98),
                          rescale_factor: float = 1.0, batch_size: Union[int, str] = 'auto',
                          memory_budget_gb: float = 2.0, prune_level: int = None,
                          progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """
        Predict sarcomeres (Z-bands, mbands, distance, orientation) with U-Net.

        Parameters
        ----------
        frames : {'all', int, list of int, np.ndarray}, optional
            Frames for sarcomere detection ('all', a single frame index, or
            selected frames). Default is 'all'.
        model_path : str or None, optional
            Path of trained U-Net weights. None uses the default model.
            Default is None.
        max_patch_size : tuple of int or 'auto', optional
            Maximal patch dimensions ``(n_x, n_y)`` for the CNN. 'auto' derives
            them from free device memory and the model, which avoids splitting an
            image that would fit in one patch. Default is 'auto'.
        normalization_mode : str, optional
            Intensity normalization mode for 3D stacks ('single': each image
            individually, 'all': histogram of full stack, 'first': histogram of
            first image). Default is 'all'.
        clip_thres : tuple of float, optional
            Clip threshold (lower, upper percentiles) for intensity
            normalization. Default is (0., 99.98).
        rescale_factor : float, optional
            Factor to rescale input images in XY before prediction (e.g. 0.5
            halves the XY resolution); outputs are rescaled back afterwards.
            Intended to bring the pixel size into the model's trained range, not
            as a speed knob: resampling and the upscaling of the masks afterwards
            can introduce artefacts. Default is 1.0 (no rescaling).
        batch_size : int or 'auto', optional
            Patches per forward pass. 'auto' sizes from free GPU memory on CUDA
            and uses 1 elsewhere, where batching does not help. Default is 'auto'.
        memory_budget_gb : float, optional
            Rough ceiling on the working set while predicting. Long movies are
            predicted in blocks so the five output heads do not have to fit in
            memory at once. Default is 2.0.
        prune_level : int, optional
            Stop the U-Net at this nesting depth (1, 2 or 3) and read the matching
            deep-supervision head. Level 2 roughly halves the compute but changes
            the masks, so validate it on your data first
            (``_bench/validate_fast_modes.py``). None uses the full model, the
            default.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        max_patch_size = Utils.check_and_round_max_patch_size(max_patch_size)
        if isinstance(frames, str) and frames == 'all':
            # Hand the detector a lazy handle so the raw stack is read block by
            # block instead of being materialised in full.
            if not self.store.has_image():
                self.read_imgs()  # first open: ingest the source TIFF into the store
            images = self.store.image_handle() if self.store.has_image() else self.read_imgs()
            list_frames = list(range(images.shape[0] if len(images.shape) > 2 else 1))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            images = self.read_imgs(frames=frames)
            if np.issubdtype(type(frames), np.integer):
                list_frames = [frames]
            else:
                list_frames = list(frames)
        else:
            raise ValueError('frames argument not valid')

        # Rescaling is done inside detect_sarcomeres_unet, which also scales the
        # predicted masks back to the input resolution. Doing it here as well would
        # apply the factor twice and return masks at the wrong size.

        # Check pixelsize is not None
        if self.metadata.pixelsize is None:
            raise ValueError("Pixel size is not available. Please provide pixelsize during initialization.")
        
        # Delegate to the detection module, which writes each block of predicted
        # masks straight into the OME-Zarr store as it is produced.
        info = {}
        detection.detect_sarcomeres_unet(
            info=info,
            images=images,
            model_path=model_path,
            model_dir=str(self.model_dir),
            pixelsize=self.metadata.pixelsize,
            max_patch_size=max_patch_size,
            normalization_mode=normalization_mode,
            clip_thres=clip_thres,
            rescale_factor=rescale_factor,
            device=self.device,
            batch_size=batch_size,
            memory_budget_gb=memory_budget_gb,
            prune_level=prune_level,
            make_sink=self.store.create_mask,
            progress_notifier=progress_notifier
        )

        _dict = {
            'params.detect_sarcomeres.frames': list_frames,
            'params.detect_sarcomeres.model': model_path,
            'params.detect_sarcomeres.normalization_mode': normalization_mode,
            'params.detect_sarcomeres.clip_threshold': clip_thres,
            'params.detect_sarcomeres.rescale_factor': rescale_factor,
            'params.detect_sarcomeres.prune_level': prune_level,
            'params.detect_sarcomeres.max_patch_size': max_patch_size,
            # what 'auto' actually resolved to, so the run can be reproduced
            'params.detect_sarcomeres.patch_size_used': info.get('patch_size'),
        }
        self.data.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def _remap_mask_key(self, list_frames: List[int], detected_frames: Any) -> Union[int, List[int]]:
        """Translate movie-frame indices to page indices inside the sparsely-saved mask store.

        Masks are stored only for frames passed to detect_sarcomeres, in detection order.
        When ``detected_frames`` covers every frame this is an identity mapping, so the
        original indices are returned; otherwise each requested frame's position is looked up.
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

    def load_mask_full_stack(self, name: str) -> Optional[np.ndarray]:
        """
        Load a mask from the store by name and expand it to full stack length for display.

        Masks are stored sparsely (only for detected frames). For napari display alongside
        the raw movie, this returns an (n_stack, ...) array with computed frames placed at
        their original frame indices and zeros elsewhere. ``name`` is a store mask name
        (e.g. ``'zbands'``, ``'mbands'``, ``'cell_mask'``, ``'zbands_fast_movie'``). Returns
        None if the mask is not present.
        """
        if not self._mask_exists(name):
            return None
        arr = self._read_mask(name)
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
                                  max_patch_size: Union[Tuple[int, int, int], str] = 'auto',
                                  normalization_mode: str = 'all',
                                  clip_thres: Tuple[float, float] = (0., 99.8),
                                  batch_size: Union[int, str] = 'auto',
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Predict sarcomere z-bands with 3D U-Net for high-speed movies for improved temporal consistency.

        Parameters
        ----------
        model_path : str or None, optional
            Path of trained 3D U-Net weights. None uses the default model.
            Default is None.
        max_patch_size : tuple of int or 'auto', optional
            Maximal patch dimensions ``(n_frames, n_x, n_y)`` for the CNN; each
            must be divisible by 16. 'auto' derives them from free device memory
            and the model. Default is 'auto'.
        normalization_mode : str, optional
            Intensity normalization mode for 3D stacks ('single': each image
            individually, 'all': histogram of full stack, 'first': histogram of
            first image). Default is 'all'.
        clip_thres : tuple of float, optional
            Clip threshold (lower, upper percentiles) for intensity
            normalization. Default is (0., 99.8).
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'model_z_bands_unet3d.pt')
        
        # Delegate to detection module
        info = {}
        masks = detection.detect_z_bands_fast_movie_unet(
            info=info,
            images=self.read_imgs(),
            model_path=model_path,
            model_dir=str(self.model_dir),
            max_patch_size=max_patch_size,
            normalization_mode=normalization_mode,
            clip_thres=clip_thres,
            device=self.device,
            batch_size=batch_size,
            progress_notifier=progress_notifier
        )
        for name, arr in masks.items():
            self.store.write_mask(name, np.asarray(arr))
        _dict = {'params.detect_z_bands_fast_movie.model': model_path,
                 'params.detect_z_bands_fast_movie.max_patch_size': max_patch_size,
                 'params.detect_z_bands_fast_movie.patch_size_used': info.get('patch_size'),
                 'params.detect_z_bands_fast_movie.normalization_mode': normalization_mode,
                 'params.detect_z_bands_fast_movie.clip_threshold': clip_thres}
        self.data.update(_dict)
        if self.auto_save:
            self.store_structure_data()

    def analyze_cell_mask(self, frames: Union[str, int, List[int], np.ndarray] = 'all', threshold: float = 0.1) -> None:
        """
        Analyze the area occupied by cells and compute average cell intensity and
        cell area ratio.

        Parameters
        ----------
        frames : {'all', int, list of int, np.ndarray}, optional
            Frames to analyze ('all', a single frame index, or selected frames).
            Default is 'all'.
        threshold : float, optional
            Threshold for binarizing the cell mask; pixels above it are cell.
            Default is 0.1.
        """
        if not self._mask_exists('cell_mask'):
            raise FileNotFoundError("Cell mask not found. Please run detect_sarcomeres first.")
        _detected_frames = self.data.get('params.detect_sarcomeres.frames', 'all')
        if (isinstance(frames, str) and frames == 'all') or (self.metadata.n_stack == 1 and frames == 0):
            cell_mask = self._read_mask('cell_mask')
            images = self.read_imgs()
            list_frames = list(range(len(images)))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            if np.issubdtype(type(frames), np.integer):
                list_frames = [int(frames)]
            else:
                list_frames = [int(f) for f in frames]
            mask_key = self._remap_mask_key(list_frames, _detected_frames)
            cell_mask = self._read_mask('cell_mask', frames=mask_key)
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
        frames : {'all', int, list of int, np.ndarray}, optional
            Frames to analyze ('all', a single frame index, or selected frames).
            Default is 'all'.
        threshold : float, optional
            Threshold for binarizing z-bands prior to labeling (0-1).
            Default is 0.5.
        min_length : float, optional
            Minimal z-band length in µm; shorter z-bands are removed.
            Default is 0.2.
        median_filter_radius : float, optional
            Radius of the kernel smoothing the orientation field, in µm.
            Default is 0.2.
        theta_phi_min : float, optional
            Minimal cosine of the angle between the pointed z-band vector and the
            vector connecting z-band ends; smaller values are not recognized as
            connections (for lateral alignment and distance analysis).
            Default is 0.4.
        a_min : float, optional
            Minimal lateral alignment between z-band ends to create a lateral
            connection. Default is 0.3.
        d_max : float, optional
            Maximal distance between z-band ends in µm; pairs farther apart are
            not connected. Default is 3.0.
        d_min : float, optional
            Minimal distance between z-band ends in µm; pairs closer than this are
            not connected. Default is 0.0.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        if not self._mask_exists('zbands'):
            raise FileNotFoundError("Z-band mask not found. Please run detect_sarcomeres first.")
        _detected_frames = self.data.get('params.detect_sarcomeres.frames', 'all')
        if ((isinstance(frames, str) and frames == 'all') or (self.metadata.n_stack == 1 and frames == 0)
                or (_detected_frames != 'all' and len(_detected_frames) == 1)):
            zbands = self._read_mask('zbands')
            orientation_field = self._read_mask('orientation')
            images = self.read_imgs()
            list_frames = list(range(len(images)))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or type(frames) is np.ndarray:
            if np.issubdtype(type(frames), np.integer):
                list_frames = [int(frames)]
            else:
                list_frames = [int(f) for f in frames]
            mask_key = self._remap_mask_key(list_frames, _detected_frames)
            zbands = self._read_mask('zbands', frames=mask_key)
            # orientation is stored (frames, 2, H, W) — index by frame like zbands.
            orientation_field = self._read_mask('orientation', frames=mask_key)
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
                                  peak_prominence: float = 0.3,
                                  peak_algorithm: str = 'default',
                                  use_fast_movie_zbands: bool = True,
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Extract sarcomere orientation and length vectors.

        Parameters
        ----------
        frames : {'all', int, list of int, np.ndarray}, optional
            Frames to analyze ('all', a single frame index, or selected frames).
            Default is 'all'.
        threshold_mbands : float, optional
            Threshold to binarize sarcomere M-bands. Lower values may yield more
            false-positive sarcomere vectors. Default is 0.25.
        median_filter_radius : float, optional
            Radius of the kernel smoothing the orientation field before assessing
            orientation at M-points, in µm. Default is 0.25.
        linewidth : float, optional
            Line width of profile lines for analyzing sarcomere lengths, in µm.
            Default is 0.2. LOI analysis, tuned for maximum accuracy, uses 0.65 µm —
            increasing ``linewidth`` toward that averages over more transverse
            pixels and smooths per-frame slen.
        interp_factor : int, optional
            Akima/linear upsampling factor applied to each profile before peak
            detection. Default is 4. LOI analysis uses 6; sub-pixel peak
            localisation drives per-frame slen accuracy.
        slen_lims : tuple of float, optional
            Sarcomere length limits in µm. Default is (1, 3).
        threshold_sarcomere_mask : float, optional
            Threshold to binarize sarcomere masks. Default is 0.1.
        interpolation_method : str, optional
            Interpolation method for profile analysis: 'linear' (fast) or 'akima'
            (smooth). Default is 'akima'.
        smooth_orientation_sigma : float, optional
            Temporal Gaussian sigma (in frames) for smoothing the orientation
            field along the time axis before per-frame vector extraction,
            reducing frame-to-frame jitter (axially correct via the double-angle
            trick). 0 disables smoothing; ``sigma ≈ 1`` is a ~5-frame effective
            span. Only meaningful for multi-frame stacks. Default is 0.0.
        peak_prominence : float, optional
            ``scipy.signal.find_peaks`` prominence threshold for Z-band peak
            detection inside each profile; lower values accept weaker, noisier
            peaks. Only used when ``peak_algorithm='default'``. Default is 0.3
            (lowered from 0.5): validated on real 20 kPa data to recover ~+5.8%
            real detections — the extra peaks are real (slen median 1.668→1.666,
            IQR and edge-junk fraction unchanged), concentrated at peak
            contraction / fast motion where Z-band peaks weaken. NB: the ``'loi'``
            path ignores this and uses its own fixed prominence.
        peak_algorithm : {'default', 'loi'}, optional
            Peak detection routine. ``'default'`` uses the fast batched
            :func:`sarcasm.utils.Utils.process_profiles_batch` (``interp_factor``
            and ``peak_prominence`` configurable); ``'loi'`` routes every profile
            through :func:`sarcasm.utils.Utils.peakdetekt` (the LOI peak +
            6× Akima + COM-refinement pipeline; ``interp_factor`` and
            ``peak_prominence`` ignored), slower but most accurate. Default is
            'default'.
        use_fast_movie_zbands : bool, optional
            If True and a ``zbands_fast_movie`` mask exists (produced by
            :meth:`detect_z_bands_fast_movie`), use that 3D U-Net output instead
            of the per-frame ``zbands`` mask; it is less noisy frame-to-frame,
            yielding smoother per-frame slen. Falls back to the 2D mask when
            unavailable; the choice is stored in
            ``params.analyze_sarcomere_vectors.zbands_source``. Default is True.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        if not self._mask_exists('zbands'):
            raise FileNotFoundError("Z-band mask not found. Please run detect_sarcomeres first.")

        # Decide which Z-band stack to use (fast-movie 3D U-Net vs per-frame 2D).
        if use_fast_movie_zbands and self._mask_exists('zbands_fast_movie'):
            zbands_name = 'zbands_fast_movie'
            zbands_source = 'fast_movie_3d'
        else:
            zbands_name = 'zbands'
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
            z_bands = self._read_mask(zbands_name)
            mbands = self._read_mask('mbands')
            orientation_field = self._read_mask('orientation')
            sarcomere_mask = self._read_mask('sarcomere_mask')
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, list) or isinstance(frames, np.ndarray):
            z_bands = self._read_mask(zbands_name, frames=frames)
            mbands = self._read_mask('mbands', frames=frames)
            orientation_field = self._read_mask('orientation', frames=frames)
            sarcomere_mask = self._read_mask('sarcomere_mask', frames=frames)
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
                f'Temporally smoothing orientation field with sigma={smooth_orientation_sigma:.3f} frames...'
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
        logger.info(f'Smoothing orientation field (median filter, radius={radius_pixels} px) '
                    f'across {n_frames} frame(s)…')
        angle_maps = Utils.get_orientation_angle_map(
            orientation_field, use_median_filter=True, radius=radius_pixels,
            progress_notifier=progress_notifier,
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
        frames : {'all', int, list of int, np.ndarray} or None, optional
            Frames to analyze ('all', a single frame index, or selected frames).
            If None, frames from sarcomere vector analysis are used.
            Default is None.
        ratio_seeds : float, optional
            Ratio of sarcomere vectors used as seeds for line growth.
            Default is 0.1.
        persistence : int, optional
            Persistence of line (averaged vector length and orientation for prior
            estimation); must be > 0. Default is 3.
        threshold_distance : float, optional
            Maximal distance for nearest-neighbor estimation, in µm.
            Default is 0.5.
        n_min : int, optional
            Minimal number of sarcomere segments per line; shorter lines are
            removed. Default is 4.
        median_filter_radius : float, optional
            Filter radius for smoothing the myofibril length map, in µm.
            Default is 0.5.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
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
        frames : {'all', int, list of int, np.ndarray} or None, optional
            Frames to analyze ('all', a single frame index, or selected frames).
            If None, frames from sarcomere vector analysis are used.
            Default is None.
        d_max : float, optional
            Max. distance threshold (µm) for creating a network edge between
            vector ends. Default is 3.
        cosine_min : float, optional
            Minimal absolute cosine between vector angles for creating a network
            edge between vector ends. Default is 0.65.
        leiden_resolution : float, optional
            Control parameter for domain size; smaller values favor larger
            domains, larger values favor smaller domains. Default is 0.06.
        random_seed : int, optional
            Random seed for the Leiden algorithm (reproducibility).
            Default is 42.
        area_min : float, optional
            Minimal area of domains/clusters in µm². Default is 20.0.
        dilation_radius : float, optional
            Dilation radius for refining domain area masks, in µm. Default is 0.3.
        store_mask : bool, optional
            If True, store the integer-labeled domain mask in ``self.data``
            (memory-intensive for large time-series). Default is False.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
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

    def track_sarcomere_vectors(
        self,
        frames: Union[str, int, List[int], np.ndarray] = 'all',
        max_disp_along_um: float = 1.0,
        max_disp_perp_um: float = 0.2,
        ori_tol_deg: float = 45.0,
        retire_after_s: Optional[float] = None,
        min_track_duration_s: float = 0.08,
        max_gap_interpolation: int = 3,
    ) -> None:
        """2D full-field sarcomere-vector tracking.

        Complements :meth:`Motion.track_z_bands` (LOI / 1D). Each sarcomere
        vector in the first analyzed frame seeds a query point. Every frame, the
        candidate (query point, detection) pairs that pass the anisotropic
        (along-/perpendicular-to-sarcomere) and orientation gates are matched by an
        exact minimum-cost assignment, solved per connected component of the
        candidate graph — which, because the vectors densely sample each M-band
        midline, means each midline row is aligned jointly. A query point with no
        consistent detection records an honest gap frame (``tracks_snapped`` False,
        length NaN) and keeps its identity, so a detection dropout of any length no
        longer ends a trajectory. No M-band identity is tracked; anti-convergence
        is guaranteed because each detection is matched at most once.

        Prerequisites: :meth:`analyze_sarcomere_vectors` must have been run.

        Parameters
        ----------
        frames : {'all', int, list of int, np.ndarray}, optional
            Frames to track ('all', a single frame index, or selected frames).
            Default is 'all'.
        max_disp_along_um : float, optional
            Snap-gate tolerance for motion along the sarcomere axis, in µm — the
            maximum a track may move along its axis per frame. At the default
            1.0 µm tracks cannot jump more than ~1 µm regardless of pixel size.
            Default is 1.0.
        max_disp_perp_um : float, optional
            Snap-gate tolerance for motion perpendicular to the sarcomere axis,
            in µm; far tighter than the along gate (a perpendicular jump is a
            swap onto a neighbouring myofibril). Default is 0.2.
        ori_tol_deg : float, optional
            Orientation tolerance for the snap gate, in degrees (compared
            modulo π). Default is 45.0.
        retire_after_s : float or None, optional
            Time a track may go unmatched before it is closed, in seconds.
            ``None`` (default) means tracks never retire: an unmatched track is
            carried along by the coherent motion of its neighbourhood, so its
            identity stays valid through a dropout of any length and retiring it
            would only fragment the trajectory. Set a value (e.g. 5.0) for very
            long recordings, where sarcomeres genuinely appear and disappear, to
            bound the track count.
        min_track_duration_s : float, optional
            Minimum accumulated real observation time required to keep a track, in
            seconds. Falls back to 5 real snaps when frametime is unknown.
            Default is 0.08.
        max_gap_interpolation : int, optional
            Longest run of consecutive gap frames whose sarcomere length and
            orientation are filled by interpolating between the real snaps on
            either side, so brief detection flicker does not punch holes in the
            per-track traces. Interior gaps only, and ``tracks_snapped`` stays
            False on filled frames, so coverage and every real-observation metric
            are unaffected. Set to 0 to leave all gap frames NaN. Default is 3.
        """
        if 'pos_vectors_px' not in self.data:
            raise ValueError('Sarcomere vectors not analyzed. Run analyze_sarcomere_vectors first.')

        # Frame selection matches the pattern in analyze_sarcomere_vectors.
        _detected_frames = self.data.get('params.detect_sarcomeres.frames', 'all')
        if ((isinstance(frames, str) and frames == 'all')
                or (self.metadata.n_stack == 1 and frames == 0)
                or (_detected_frames != 'all' and len(_detected_frames) == 1)):
            list_frames = list(range(self.metadata.n_stack))
        elif np.issubdtype(type(frames), np.integer) or isinstance(frames, (list, np.ndarray)):
            if np.issubdtype(type(frames), np.integer):
                list_frames = [int(frames)]
            else:
                list_frames = [int(f) for f in frames]
        else:
            raise ValueError('frames argument not valid')

        if len(list_frames) < 2:
            raise ValueError('Need at least 2 frames for tracking.')

        # Tracking is temporal: real-time gaps between non-contiguous frames would
        # be treated as single-frame steps (the seconds-valued horizons assume
        # Δt = frametime).
        if not np.all(np.diff(list_frames) == 1):
            raise ValueError(
                f'track_sarcomere_vectors requires contiguous frames; got {list_frames}. '
                'The tracker assumes a single-frame step between consecutive entries.')

        # The per-frame vectors must ACTUALLY be present for every tracked frame.
        # (params.analyze_sarcomere_vectors.frames can be stale — e.g. left claiming
        # all frames after an interrupted run — so check the data itself, not params.)
        pv = self.data['pos_vectors_px']
        missing = [t for t in list_frames if t >= len(pv) or pv[t] is None]
        if missing:
            preview = missing[:8]
            raise ValueError(
                f'Sarcomere vectors are missing for {len(missing)} of {len(list_frames)} '
                f'requested frames (e.g. {preview}). analyze_sarcomere_vectors most likely '
                'did not finish (it stores results only after the last frame). Re-run '
                'analyze_sarcomere_vectors() to completion, or pass frames=<a fully analyzed '
                'contiguous range>.')

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
            pos_px_all, mid_all, slen_all, ori_all,
            pixelsize=self.metadata.pixelsize,
            frametime=self.metadata.frametime,
            max_disp_along_um=max_disp_along_um,
            max_disp_perp_um=max_disp_perp_um,
            ori_tol_deg=ori_tol_deg,
            retire_after_s=retire_after_s,
            min_track_duration_s=min_track_duration_s,
            max_gap_interpolation=max_gap_interpolation,
        )

        tracking_data = {
            'n_tracks': out['n_tracks'],
            'track_ids': out['track_ids'],
            'track_start_frame': out['track_start_frame'],
            'track_lengths': out['track_lengths'],
            'track_drift_um': out['track_drift_um'],
            'tracks_positions_um': out['tracks_positions_um'],
            'tracks_positions_px': out['tracks_positions_px'],
            'tracks_slen': out['tracks_slen'],
            'tracks_orientations': out['tracks_orientations'],
            'tracks_snapped': out['tracks_snapped'],
            'tracks_detection_id': out['tracks_detection_id'],
            'tracks_midline_id': out['tracks_midline_id'],
            'fragmentation_ratio': out['fragmentation_ratio'],
            'n_tracks_retired': out['n_tracks_retired'],
            'n_interpolated_gap_frames': out['n_interpolated_gap_frames'],
            'params.track_sarcomere_vectors.frames': list_frames,
            'params.track_sarcomere_vectors.max_disp_along_um': max_disp_along_um,
            'params.track_sarcomere_vectors.max_disp_perp_um': max_disp_perp_um,
            'params.track_sarcomere_vectors.ori_tol_deg': ori_tol_deg,
            'params.track_sarcomere_vectors.retire_after_s': retire_after_s,
            'params.track_sarcomere_vectors.min_track_duration_s': min_track_duration_s,
            'params.track_sarcomere_vectors.max_gap_interpolation': max_gap_interpolation,
        }
        self.data.update(tracking_data)
        # Re-tracking changes track identities, so any prior grouping no longer
        # matches the new tracks. Drop the stale grouping keys (grouped-motion
        # getters are additionally guarded by _assert_track_motion_fresh via the
        # grouping_hash / track_ids_snapshot) so the tracks dataframe and napari
        # overlays never mix an old grouping with the new tracks. Re-run
        # group_tracks (+ analyze_track_motion) to regroup.
        for _stale in ('track_group_id', 'track_group_order', 'group_kind',
                       'n_groups', 'group_member_counts', 'group_n_vectors_total',
                       'group_n_vectors_in_long_tracks', 'track_ids_snapshot',
                       'grouping_hash'):
            if _stale in self.data:
                del self.data[_stale]
        logger.info(f'Tracked {out["n_tracks"]} sarcomere query points over {len(list_frames)} frames.')
        if self.auto_save:
            self.store_structure_data()

    def get_tracks(self, min_coverage: float = 0.0) -> pd.DataFrame:
        """Tidy per-track summary of the 2D tracker output (one row per track).

        Discoverable accessor over the dense ``tracks_*`` arrays written by
        :meth:`track_sarcomere_vectors`, so downstream code does not have to
        reach into raw ``self.data`` and reconstruct shapes/semantics.

        Parameters
        ----------
        min_coverage : float, optional
            Only return tracks whose snap coverage (snapped frames / n_frames)
            is at least this value. Default 0.0 (all kept tracks).

        Returns
        -------
        pandas.DataFrame
            Columns: ``track_id``, ``start_frame``, ``length`` (number of snapped
            frames), ``n_frames``, ``coverage``, ``mean_slen``, ``std_slen``,
            ``drift_um`` (how far the track wandered from the coherent motion of
            its neighbours — roughly one sarcomere length means it probably
            changed identity; NaN if too short to score),
            ``start_y_um``, ``start_x_um``, ``ref_midline_id`` (the M-band id at
            the start frame), and — once :meth:`group_tracks` has been run —
            ``group_id`` / ``order_in_group`` (``order_in_group`` is 0 for the
            unordered levels pool/m-band/domain/custom; it carries the within-fibre
            rank only for the myofibril level).
        """
        if 'tracks_slen' not in self.data:
            raise ValueError('No tracks found. Run track_sarcomere_vectors first.')

        n_tracks = int(self.data.get('n_tracks', 0))
        columns = ['track_id', 'start_frame', 'length', 'n_frames', 'coverage',
                   'mean_slen', 'std_slen', 'drift_um', 'start_y_um', 'start_x_um',
                   'ref_midline_id']
        if n_tracks == 0:
            return pd.DataFrame(columns=columns)

        slen = np.asarray(self.data['tracks_slen'], dtype=float).reshape(n_tracks, -1)
        T = slen.shape[1]
        snapped = np.asarray(self.data['tracks_snapped']).astype(bool).reshape(n_tracks, T)
        pos = np.asarray(self.data['tracks_positions_um'], dtype=float).reshape(n_tracks, T, 2)
        det_mid = np.asarray(self.data.get('tracks_midline_id', np.full((n_tracks, T), -1))).reshape(n_tracks, T)

        track_ids = np.asarray(self.data.get('track_ids', np.arange(n_tracks)))
        start_frame = np.asarray(self.data.get('track_start_frame', np.zeros(n_tracks, int))).astype(int)
        length = np.asarray(self.data.get('track_lengths', snapped.sum(axis=1))).astype(int)
        coverage = length / float(T) if T else np.zeros(n_tracks)

        rows = np.arange(n_tracks)
        start_pos = pos[rows, start_frame]  # (n_tracks, 2) in (y, x)
        ref_mid = det_mid[rows, start_frame]
        with np.errstate(invalid='ignore'):
            mean_slen = np.nanmean(slen, axis=1)
            std_slen = np.nanstd(slen, axis=1)

        df = pd.DataFrame({
            'track_id': track_ids,
            'start_frame': start_frame,
            'length': length,
            'n_frames': T,
            'coverage': coverage,
            'mean_slen': mean_slen,
            'std_slen': std_slen,
            'drift_um': np.asarray(self.data.get(
                'track_drift_um', np.full(n_tracks, np.nan)), dtype=float).reshape(-1),
            'start_y_um': start_pos[:, 0],
            'start_x_um': start_pos[:, 1],
            'ref_midline_id': ref_mid,
        })
        # Grouping columns appear once group_tracks has been run.
        if 'track_group_id' in self.data:
            df['group_id'] = np.asarray(self.data['track_group_id']).reshape(-1)[:n_tracks]
        if 'track_group_order' in self.data:
            df['order_in_group'] = np.asarray(self.data['track_group_order']).reshape(-1)[:n_tracks]

        if min_coverage > 0.0:
            df = df[df['coverage'] >= min_coverage].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Post-tracking grouping + motion analysis (group_tracks -> analyze_track_motion)
    # ------------------------------------------------------------------
    # All post-tracking "methods" (pool / m-band / myofibril / domain / custom)
    # are groupings over the same per-track tracks_slen(t). group_tracks writes a
    # cheap, inspectable label artifact; analyze_track_motion aggregates per group
    # and runs the shared contraction engine. The two are decoupled so a grouping
    # can be QC'd / re-tried without re-tracking; a fingerprint hard-raises if a
    # grouped result is read after its grouping changed.

    _GROUPING_LEVELS = ('pool', 'mband', 'myofibril', 'loi', 'domain', 'custom')
    # Groupings that order their members head-to-tail into a 1D fibre thread.
    # They represent geometry, not just a set, so they keep partially-tracked
    # sarcomeres (a dropped member is a hole in the fibre, not removed noise).
    _CHAIN_LEVELS = ('myofibril', 'loi')
    _CHAIN_MIN_COVERAGE = 0.1

    def _grouping_hash(self, by: str, reference_frame: int, min_coverage: float,
                       min_group_size: int = 1,
                       max_drift_slen: Optional[float] = None,
                       labels: Optional[np.ndarray] = None) -> str:
        """Fingerprint of (current track identity + grouping recipe)."""
        h = hashlib.sha1()
        track_ids = np.ascontiguousarray(np.asarray(self.data.get('track_ids', []))).astype(np.int64)
        h.update(track_ids.tobytes())
        h.update(repr((by, int(reference_frame), float(min_coverage), int(min_group_size),
                       None if max_drift_slen is None else float(max_drift_slen))).encode())
        if labels is not None:
            h.update(np.ascontiguousarray(np.asarray(labels)).astype(np.int64).tobytes())
        return h.hexdigest()

    @staticmethod
    def _partition_fibre_chains(fibers: List[List[int]], gid: np.ndarray,
                                order: np.ndarray,
                                group_ids: Optional[List[int]] = None) -> None:
        """Greedy longest-first partition of ordered track chains into groups.

        Each track joins only one fibre; ``gid``/``order`` are written in place
        (group id and rank along the fibre). Shared by 'myofibril' and any other
        line-chain grouping that resolves to ordered track index lists.

        Parameters
        ----------
        fibers : list of list of int
            Ordered track-index chains, head to tail along each fibre.
        gid, order : np.ndarray
            Per-track group id / rank-along-fibre, written in place.
        group_ids : list of int or None, optional
            Group id to assign to each chain, parallel to ``fibers``. ``None``
            (default) numbers the surviving fibres sequentially. Pass explicit ids
            to preserve a fixed label space (e.g. LOI line *i* -> group *i*), which
            keeps the groups aligned with the lines they came from even when a
            chain is dropped for having too few unclaimed tracks.
        """
        idx = sorted(range(len(fibers)), key=lambda i: len(fibers[i]), reverse=True)
        claimed: set = set()
        fib_id = 0
        for i in idx:
            chain = fibers[i]
            unclaimed = [tr for tr in chain if tr not in claimed]
            if len(unclaimed) < 2:
                continue
            this_id = fib_id if group_ids is None else int(group_ids[i])
            for rank, tr in enumerate(unclaimed):
                gid[tr] = this_id
                order[tr] = rank
                claimed.add(tr)
            fib_id += 1

    @staticmethod
    def _assign_points_to_polylines(points: np.ndarray, lines: List[np.ndarray],
                                    max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
        """Assign each point to the nearest polyline within ``max_dist``.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 2)`` array of ``(row, col)`` positions.
        lines : list of np.ndarray
            List of ``(Li, 2)`` polyline vertex arrays (same coordinate space).
        max_dist : float
            Maximum point-to-polyline distance to accept an assignment.

        Returns
        -------
        line_id : np.ndarray
            ``(N,)`` int array; index of the nearest polyline, or -1 if none is
            within ``max_dist``.
        arclen : np.ndarray
            ``(N,)`` float array; arc length of the closest point along that
            polyline (NaN if unassigned), used to order points along the line.
        dist : np.ndarray
            ``(N,)`` float array; perpendicular distance to the assigned polyline
            (inf if unassigned), used to pick the on-line track when several
            candidates share an arc position.
        """
        n = len(points)
        line_id = np.full(n, -1, dtype=np.int64)
        arclen = np.full(n, np.nan, dtype=float)
        best = np.full(n, np.inf, dtype=float)
        pts = np.asarray(points, dtype=float)
        for li, L in enumerate(lines):
            L = np.asarray(L, dtype=float)
            if L.shape[0] < 2:
                continue
            a = L[:-1]                       # (S, 2) segment starts
            seg = L[1:] - a                  # (S, 2) segment vectors
            seg_len2 = (seg * seg).sum(1)    # (S,)
            seg_len = np.sqrt(seg_len2)
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # (S+1,)
            safe = np.where(seg_len2 > 0, seg_len2, 1.0)
            for i in range(n):
                p = pts[i]
                t = ((p - a) * seg).sum(1) / safe
                t = np.clip(t, 0.0, 1.0)
                proj = a + t[:, None] * seg
                d = np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])
                j = int(np.argmin(d))
                if d[j] < best[i] and d[j] <= max_dist:
                    best[i] = d[j]
                    line_id[i] = li
                    arclen[i] = cum[j] + t[j] * seg_len[j]
        return line_id, arclen, best

    def group_tracks(
        self,
        by: str = 'pool',
        *,
        reference_frame: int = 0,
        min_coverage: Optional[float] = None,
        min_group_size: int = 1,
        max_drift_slen: Optional[float] = 1.0,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Assign each track to a group (cheap, inspectable, freely re-callable).

        Writes only label arrays — runs no contraction analysis. Prerequisite:
        :meth:`track_sarcomere_vectors`.

        Parameters
        ----------
        by : {'pool', 'mband', 'myofibril', 'loi', 'domain', 'custom'}, optional
            Grouping level. ``'pool'`` = all eligible tracks in one group;
            ``'mband'`` = group tracks by the M-band (midline) they snapped to at
            ``reference_frame`` (laterally-registered sarcomeres); ``'myofibril'`` =
            order tracks into fibre chains from the ``reference_frame`` myofibril
            lines (requires :meth:`analyze_myofibrils`; sets ``track_group_order``
            = rank along the fibre, enabling the LOI-style per-fibre analysis via
            :meth:`get_track_motion`); ``'loi'`` = like ``'myofibril'`` but uses
            the **curated** lines from :meth:`detect_lois` instead of all myofibril
            lines (the same quality-filtered/clustered subset you would place manual
            LOIs on — far fewer, cleaner fibre groups; run
            ``detect_lois(frame=reference_frame)`` first);
            ``'domain'`` = assign each track to the
            Leiden domain its ``reference_frame`` position falls in (requires
            :meth:`analyze_sarcomere_domains`; preserves the domain mask's label
            space so the existing domain plots stay aligned); ``'custom'`` = use
            ``labels``. Default is 'pool'.
        reference_frame : int, optional
            Movie-frame whose geometry defines the grouping (must be a tracked
            frame). Used by ``'mband'`` / ``'myofibril'`` / ``'loi'`` / ``'domain'``.
            For ``'loi'`` this should match the ``frame`` passed to
            :meth:`detect_lois`. Default is 0.
        min_coverage : float or None, optional
            Tracks snapped in fewer than this fraction of frames are dropped
            (group id ``-1``). ``None`` (default) picks the floor from the
            grouping: 0.1 for the chain kinds ``'myofibril'`` / ``'loi'``, 0.5
            otherwise. A chain represents the fibre's geometry, so dropping a
            partially-tracked sarcomere punches a hole into it rather than
            removing noise; for a group *mean* the strict floor is right.
        min_group_size : int, optional
            Drop groups with fewer than this many member tracks — their members
            become unassigned (group id ``-1``), like a failed ``min_coverage``.
            Useful to discard the many 1–2 sarcomere fragments a fine grouping
            produces (e.g. ``min_group_size=6`` for ``by='myofibril'``), whose
            group-mean traces are too noisy to analyse. For ``'pool'``, ``'mband'``,
            ``'myofibril'`` and ``'custom'`` the surviving groups are renumbered
            contiguously; ``'domain'`` and ``'loi'`` keep their fixed label space
            (the emptied labels simply end up with a member count of 0) so the
            domain/LOI plots stay aligned. Default is 1 (no filtering).
        max_drift_slen : float or None, optional
            Drop tracks whose ``track_drift_um`` exceeds this many sarcomere
            lengths — they have wandered away from the coherent motion of their
            neighbours and most likely changed identity. Applied **only** to the
            chain groupings ``'myofibril'`` and ``'loi'``, where such a track
            breaks the head-to-tail order and can place the same physical
            sarcomere in one chain twice; a drifting track still contributes a
            valid length to a pooled/mband/domain group, so those are left
            untouched. ``None`` disables the filter. Default is 1.0.
        labels : np.ndarray or None, optional
            Required for ``by='custom'``: integer label per track, row-aligned to
            ``track_ids``. Negative labels drop the track. Default is None.

        Notes
        -----
        Stores ``track_group_id`` / ``track_group_order`` ``(n_tracks,)``,
        ``group_kind``, ``n_groups``, ``group_member_counts``,
        ``track_ids_snapshot`` and ``grouping_hash`` into ``self.data``. The
        ``grouping_hash`` changes whenever the tracks or the recipe change; a
        stale :meth:`analyze_track_motion` result is then detected on read.
        """
        if 'tracks_slen' not in self.data:
            raise ValueError('No tracks found. Run track_sarcomere_vectors first.')
        if by not in self._GROUPING_LEVELS:
            raise ValueError(f"Unknown grouping '{by}'. Valid: {self._GROUPING_LEVELS}.")

        # A chain must span the whole fibre, so it keeps partially-tracked
        # sarcomeres that a group *mean* would rather discard (see the parameter
        # docs). Explicit values are always honoured.
        if min_coverage is None:
            min_coverage = self._CHAIN_MIN_COVERAGE if by in self._CHAIN_LEVELS else 0.5

        n_tracks = int(self.data.get('n_tracks', 0))
        track_ids = np.asarray(self.data.get('track_ids', np.arange(n_tracks)))

        if n_tracks == 0:
            gid = np.zeros(0, np.int32)
            order = np.zeros(0, np.int32)
            n_groups = 0
            counts = np.zeros(0, np.int64)
            n_vectors_total = 0
            n_vectors_long = 0
        else:
            slen = np.asarray(self.data['tracks_slen'], dtype=float).reshape(n_tracks, -1)
            T = slen.shape[1]
            length = np.asarray(self.data.get('track_lengths',
                                np.asarray(self.data['tracks_snapped']).reshape(n_tracks, T).sum(axis=1)))
            coverage = length / float(T) if T else np.zeros(n_tracks)
            eligible = coverage >= min_coverage

            # Chain groupings order their members head-to-tail, so a track that has
            # drifted off its own sarcomere corrupts the order and can put the same
            # sarcomere in the chain twice. Drop those here; pooled/mband/domain
            # groups still get a valid length from such a track, so they keep it.
            if max_drift_slen is not None and by in self._CHAIN_LEVELS:
                drift = self.data.get('track_drift_um')
                if drift is not None:
                    drift = np.asarray(drift, dtype=float).reshape(-1)
                    med_slen = float(np.nanmedian(slen[eligible])) if eligible.any() else np.nan
                    if drift.shape[0] == n_tracks and np.isfinite(med_slen):
                        too_far = np.isfinite(drift) & (drift > max_drift_slen * med_slen)
                        n_drop = int((too_far & eligible).sum())
                        if n_drop:
                            logger.info(
                                f'max_drift_slen={max_drift_slen}: dropped {n_drop} track(s) '
                                f'drifting >{max_drift_slen * med_slen:.2f} µm from their '
                                f'neighbours before building the {by} chains.')
                        eligible &= ~too_far

            gid = np.full(n_tracks, -1, dtype=np.int32)
            order = np.zeros(n_tracks, dtype=np.int32)
            fixed_n_groups = None  # set by 'domain'/'loi' to preserve a fixed label space

            if by == 'pool':
                gid[eligible] = 0

            elif by == 'mband':
                rf_idx = self._tracked_frame_index(reference_frame)
                mid_ref = np.asarray(self.data['tracks_midline_id']).reshape(n_tracks, T)[:, rf_idx]
                valid = eligible & (mid_ref >= 0)
                uniq = np.unique(mid_ref[valid])
                remap = {int(m): i for i, m in enumerate(uniq)}
                for i in np.flatnonzero(valid):
                    gid[i] = remap[int(mid_ref[i])]

            elif by == 'domain':
                rf_idx = self._tracked_frame_index(reference_frame)
                mask, n_dom = self._domain_mask_for(reference_frame)
                pos_um_ref = np.asarray(self.data['tracks_positions_um'],
                                        dtype=float).reshape(n_tracks, T, 2)[:, rf_idx]
                use = eligible & np.isfinite(pos_um_ref).all(axis=1)
                idx_use = np.flatnonzero(use)
                if idx_use.size:
                    labels_use = domain_clustering.assign_vectors_to_domains(
                        pos_um_ref[idx_use], mask, self.metadata.pixelsize)
                    hit = labels_use > 0
                    gid[idx_use[hit]] = labels_use[hit] - 1  # mask label j -> group j-1
                # Preserve the full mask label space (0..n_dom-1) so the motion
                # rows stay aligned with plot_sarcomere_domains' labels even if
                # some domains end up with no assigned tracks.
                fixed_n_groups = int(n_dom)

            elif by == 'myofibril':
                rf_idx = self._tracked_frame_index(reference_frame)
                myof_lines = self.data.get('myof_lines')
                if myof_lines is None or myof_lines[reference_frame] is None:
                    raise ValueError(
                        f'Myofibril lines not available for reference_frame {reference_frame}. '
                        'Run analyze_myofibrils first.')
                lines = myof_lines[reference_frame]  # list of ordered vector-index arrays
                det_ref = np.asarray(self.data['tracks_detection_id']).reshape(n_tracks, T)[:, rf_idx]
                # reverse map: ref-frame detection (vector) index -> first eligible track
                det_to_track: Dict[int, int] = {}
                for tr in np.flatnonzero(eligible):
                    d = int(det_ref[tr])
                    if d >= 0:
                        det_to_track.setdefault(d, int(tr))
                # ordered track chains (skip vectors with no matching track)
                fibers = []
                for line in lines:
                    chain = [det_to_track[int(v)] for v in np.asarray(line).ravel()
                             if int(v) in det_to_track]
                    if len(chain) >= 2:
                        fibers.append(chain)
                self._partition_fibre_chains(fibers, gid, order)

            elif by == 'loi':
                # An LOI is a *1D thread* of consecutive sarcomeres along one fibre —
                # exactly what 'myofibril' builds, just restricted to the curated
                # detect_lois selection. get_track_motion() then cumulatively sums the
                # members' lengths into z_pos, which is only meaningful head-to-tail.
                #
                # So prefer the detection-index chain each LOI was selected from
                # (identical code path to 'myofibril'); only fall back to geometry for
                # lines that have no chain (a fitted straight line, or one drawn by
                # hand in the GUI), and then THIN the band to one track per sarcomere
                # step so the result is still a thread rather than a wide ribbon.
                rf_idx = self._tracked_frame_index(reference_frame)
                loi_data = self.data.get('loi_data')
                loi_lines = None if loi_data is None else loi_data.get('loi_lines')
                if loi_lines is None or len(loi_lines) == 0:
                    raise ValueError(
                        'LOI lines not available. Run detect_lois first, e.g. '
                        'detect_lois(frame=reference_frame).')
                index_lines = None if loi_data is None else loi_data.get('loi_index_lines')
                lines_px = [np.asarray(l, dtype=float).reshape(-1, 2) for l in loi_lines]

                med_slen_um = float(np.nanmedian(slen[eligible])) if eligible.any() else np.nan
                px = self.metadata.pixelsize
                med_slen_px = med_slen_um / px if (np.isfinite(med_slen_um) and px) else 30.0

                if index_lines is not None and len(index_lines) == len(lines_px):
                    # --- chain path: same construction as 'myofibril' -------------
                    det_ref = np.asarray(self.data['tracks_detection_id']).reshape(n_tracks, T)[:, rf_idx]
                    det_to_track: Dict[int, int] = {}
                    for tr in np.flatnonzero(eligible):
                        d = int(det_ref[tr])
                        if d >= 0:
                            det_to_track.setdefault(d, int(tr))
                    fibers, line_ids = [], []
                    for li, chain_idx in enumerate(index_lines):
                        chain = [det_to_track[int(v)] for v in np.asarray(chain_idx).ravel()
                                 if int(v) in det_to_track]
                        if len(chain) >= 2:
                            fibers.append(chain)
                            line_ids.append(li)
                    # explicit ids keep the LOI label space (line i -> group i)
                    self._partition_fibre_chains(fibers, gid, order, group_ids=line_ids)
                else:
                    # --- geometric fallback: hand-drawn / fitted lines ------------
                    pos_px_ref = np.asarray(self.data['tracks_positions_px'],
                                            dtype=float).reshape(n_tracks, T, 2)[:, rf_idx]
                    use = eligible & np.isfinite(pos_px_ref).all(axis=1)
                    idx_use = np.flatnonzero(use)
                    if idx_use.size:
                        # Capture candidates up to half a sarcomere off the line, then
                        # keep ONE per sarcomere-long bin along it (the one closest to
                        # the line). Without this thinning every laterally-adjacent
                        # sarcomere of the neighbouring myofibrils joins the group and
                        # the "chain" becomes a ribbon many sarcomeres wide.
                        line_id, arclen, dist = self._assign_points_to_polylines(
                            pos_px_ref[idx_use], lines_px, 0.5 * med_slen_px)
                        step = max(med_slen_px, 1e-6)
                        bins = np.floor(arclen / step)
                        for li in range(len(lines_px)):
                            sel = np.flatnonzero(line_id == li)
                            # one track per sarcomere-long bin: sort by (bin, distance)
                            # and take the first of each bin — the one on the line.
                            o = sel[np.lexsort((dist[sel], bins[sel]))]
                            keep = o[np.unique(bins[o], return_index=True)[1]]
                            if keep.size < 2:
                                continue
                            gid[idx_use[keep]] = li
                            order[idx_use[keep]] = np.arange(keep.size)
                # preserve the LOI line label space (line i -> group i)
                fixed_n_groups = len(lines_px)

            elif by == 'custom':
                if labels is None:
                    raise ValueError("by='custom' requires `labels` (one integer per track).")
                labels = np.asarray(labels).reshape(-1)
                if labels.shape[0] != n_tracks:
                    raise ValueError(
                        f'labels has length {labels.shape[0]}, expected n_tracks={n_tracks} '
                        '(row-aligned to track_ids).')
                valid = eligible & (labels >= 0)
                uniq = np.unique(labels[valid])
                remap = {int(m): i for i, m in enumerate(uniq)}
                for i in np.flatnonzero(valid):
                    gid[i] = remap[int(labels[i])]

            # Drop under-sized groups: their members become unassigned, exactly as
            # if they had failed min_coverage. Groups with a fixed label space
            # ('domain'/'loi') keep their numbering (the label just empties out);
            # the others are renumbered contiguously so downstream per-group rows
            # (and plot row counts) contain no empty groups.
            if min_group_size > 1 and (gid >= 0).any():
                raw_counts = np.bincount(gid[gid >= 0])
                too_small = np.flatnonzero(raw_counts < min_group_size)
                if too_small.size:
                    gid[np.isin(gid, too_small)] = -1
                    n_dropped = int(too_small.size)
                    if fixed_n_groups is None and (gid >= 0).any():
                        keep = np.unique(gid[gid >= 0])
                        remap = np.full(int(keep.max()) + 1, -1, dtype=np.int32)
                        remap[keep] = np.arange(keep.size, dtype=np.int32)
                        gid[gid >= 0] = remap[gid[gid >= 0]]
                    logger.info(
                        f'min_group_size={min_group_size}: dropped {n_dropped} group(s) '
                        f'with fewer than {min_group_size} tracks.')

            if fixed_n_groups is not None:
                n_groups = fixed_n_groups
            else:
                n_groups = int(gid.max()) + 1 if (gid >= 0).any() else 0
            counts = np.bincount(gid[gid >= 0], minlength=n_groups).astype(np.int64)

            # QC: how much of the actually-detected sarcomere signal made it into a
            # long-lived track. The tracks-assigned count above is a count of
            # *trajectories* and is dominated by many short fragments; this
            # observation-weighted number (sarcomere-vector observations belonging
            # to a track with coverage >= min_coverage, over all detections in the
            # tracked frames) reflects how much usable signal was captured. For
            # by='pool' the long-track set is exactly the assigned set.
            n_vectors_total = self._n_sarcomere_vectors_tracked()
            n_vectors_long = int(length[eligible].sum())

        grouping_hash = self._grouping_hash(
            by, reference_frame, min_coverage, min_group_size=min_group_size,
            max_drift_slen=max_drift_slen,
            labels=labels if by == 'custom' else None)

        self.data.update({
            'track_group_id': gid,
            'track_group_order': order,
            'group_kind': by,
            'n_groups': n_groups,
            'group_member_counts': counts,
            'group_n_vectors_total': int(n_vectors_total),
            'group_n_vectors_in_long_tracks': int(n_vectors_long),
            'track_ids_snapshot': np.asarray(track_ids).copy(),
            'grouping_hash': grouping_hash,
            'params.group_tracks.by': by,
            'params.group_tracks.reference_frame': reference_frame,
            'params.group_tracks.min_coverage': min_coverage,
            'params.group_tracks.min_group_size': min_group_size,
            'params.group_tracks.max_drift_slen': max_drift_slen,
        })
        logger.info(
            f"group_tracks(by='{by}'): {n_groups} groups, "
            f"{int((gid >= 0).sum())}/{n_tracks} tracks assigned.")
        if self.auto_save:
            self.store_structure_data()

    def _n_sarcomere_vectors_tracked(self) -> int:
        """Total number of detected sarcomere-vector observations over the tracked
        frames (``sum_t N_vectors(t)``) — the denominator for the long-track vector
        coverage QC. Uses the stored per-frame ``n_vectors`` count, restricted to the
        frames that were actually tracked; falls back to counting ``pos_vectors``."""
        nv = self.data.get('n_vectors')
        tracked = self.data.get('params.track_sarcomere_vectors.frames')
        if nv is not None:
            nv = np.asarray(nv).ravel()
            if tracked is not None:
                idx = [int(f) for f in tracked if 0 <= int(f) < nv.shape[0]]
                return int(nv[idx].sum())
            return int(nv.sum())
        pv = self.data.get('pos_vectors')
        if pv is not None:
            return int(sum(len(np.asarray(p)) for p in pv if p is not None))
        return 0

    def _tracked_frame_index(self, reference_frame: int) -> int:
        """Window index (into the tracks_* time axis) of an absolute movie frame."""
        tracked = self.data.get('params.track_sarcomere_vectors.frames')
        if tracked is None:
            return int(reference_frame)
        tracked = [int(f) for f in tracked]
        if reference_frame not in tracked:
            raise ValueError(
                f'reference_frame {reference_frame} was not tracked. '
                f'Tracked frames: {tracked[0]}..{tracked[-1]}.')
        return tracked.index(int(reference_frame))

    def _domain_mask_for(self, reference_frame: int) -> Tuple[np.ndarray, int]:
        """Integer-labelled domain mask + n_domains for a reference frame.

        Uses the stored ``domain_mask`` when available, else regenerates it from
        the stored ``domains`` exactly as :meth:`analyze_sarcomere_domains` /
        :func:`domain_clustering.analyze_domains` do, so the label space (1..n)
        matches what :meth:`Plots.plot_sarcomere_domains` draws.
        """
        if 'domains' not in self.data:
            raise ValueError("Sarcomere domains not analyzed. Run analyze_sarcomere_domains first.")
        domain_frames = self.data.get('params.analyze_sarcomere_domains.frames', [])
        if reference_frame not in list(domain_frames):
            raise ValueError(
                f'reference_frame {reference_frame} was not analyzed for domains. '
                f'Available domain frames: {list(domain_frames)}.')

        if 'domain_mask' in self.data and self.data['domain_mask'][reference_frame] is not None:
            mask = self.data['domain_mask'][reference_frame]
            if hasattr(mask, 'toarray'):
                mask = mask.toarray()
            mask = np.asarray(mask)
        else:
            domains_ref = self.data['domains'][reference_frame]
            pos_ref = np.asarray(self.data['pos_vectors'][reference_frame])
            ori_ref = np.asarray(self.data['sarcomere_orientation_vectors'][reference_frame])
            len_ref = np.asarray(self.data['sarcomere_length_vectors'][reference_frame])
            dilation_radius = self.data.get('params.analyze_sarcomere_domains.dilation_radius', 0.3)
            area_min = self.data.get('params.analyze_sarcomere_domains.area_min', 20.0)
            mask, *_ = domain_clustering.analyze_domains(
                domains_ref, pos_ref, ori_ref, len_ref,
                size=self.metadata.size, pixelsize=self.metadata.pixelsize,
                dilation_radius=dilation_radius, area_min=area_min)
            mask = np.asarray(mask)
        n_domains = int(self.data['n_domains'][reference_frame])
        return mask, n_domains

    def analyze_track_motion(
        self,
        *,
        by: Optional[str] = None,
        aggregate: Optional[str] = None,
        slen_lims: Tuple[float, float] = (1.0, 3.0),
        model: Optional[str] = None,
        threshold: Optional[float] = None,
        contr_time_min: float = 0.2,
        merge_time_max: float = 0.05,
        buffer_frames: int = 3,
        min_valid_frames: float = 0.5,
        filter_params: Tuple[int, int] = (13, 5),
        reference_frame: int = 0,
        min_coverage: Optional[float] = None,
        min_group_size: int = 1,
        max_drift_slen: Optional[float] = 1.0,
    ) -> None:
        """Per-group sarcomere-contraction analysis over the current track grouping.

        Aggregates the member tracks' ``slen(t)`` into one signal per group and
        runs the shared ContractionNet engine. Grouping-blind: it analyzes
        whatever :meth:`group_tracks` produced. Pass ``by=`` to run grouping
        inline (the convenient one-call front door for ``'pool'`` / ``'mband'``).

        Outputs are written under a ``<kind>_*`` prefix (e.g. ``pool_beating_rate``,
        ``mband_slen_timeseries``, ``mband_contr``), mirroring the ``domain_*``
        schema so the same plotting code serves every grouping. For ``kind='domain'``
        the prefix *is* ``domain``, so this writes the exact legacy ``domain_*`` keys
        and the existing domain plots / feature_dict / export keep working — it is the
        track-based domain-motion analysis (replacing the former static-mask method).

        Parameters
        ----------
        by : {'pool', 'mband', 'domain'} or None, optional
            If given, run ``group_tracks(by=..., reference_frame=..., min_coverage=...)``
            first. If None, use the existing grouping. (``'custom'`` must be set up
            via a separate :meth:`group_tracks` call with ``labels``.)
            Default is None.
        aggregate : {'nanmedian', 'nanmean'} or None, optional
            Reduction of member ``slen(t)`` into the per-group signal. None
            resolves to ``'nanmean'`` for ``domain`` (legacy parity) and
            ``'nanmedian'`` otherwise. Default is None.
        slen_lims : tuple of float, optional
            Member lengths outside this µm range are ignored in the aggregate.
            Default is (1.0, 3.0).
        model, threshold, contr_time_min, merge_time_max, buffer_frames, min_valid_frames, filter_params
            ContractionNet engine knobs (see :func:`grouped_motion.run_cycle_engine`).
        reference_frame, min_coverage, min_group_size, max_drift_slen
            Forwarded to :meth:`group_tracks` when ``by`` is given. Defaults are
            0, None (0.1 for chain kinds / 0.5 otherwise), 1 and 1.0.
            ``min_group_size`` discards groups with too few
            member tracks (e.g. ``min_group_size=6`` to skip 1–5 sarcomere
            myofibril fragments whose group mean is too noisy to analyse);
            ``max_drift_slen`` drops identity-drifting tracks from the chain
            groupings (myofibril/loi) only.
        """
        if by is not None:
            if by == 'custom':
                raise ValueError(
                    "by='custom' cannot be used as a front door; call "
                    "group_tracks(by='custom', labels=...) then analyze_track_motion().")
            self.group_tracks(by=by, reference_frame=reference_frame,
                              min_coverage=min_coverage, min_group_size=min_group_size,
                              max_drift_slen=max_drift_slen)
        elif min_group_size > 1:
            logger.warning(
                'min_group_size is only applied when by=... is given (it is a '
                'grouping parameter); the existing grouping is used unchanged. '
                'Call group_tracks(..., min_group_size=%d) to apply it.', min_group_size)

        if 'track_group_id' not in self.data:
            raise ValueError('No track grouping found. Run group_tracks(...) first '
                             '(or pass by=... to analyze_track_motion).')
        if self.metadata.frametime is None:
            raise ValueError('Frame time not defined in metadata. Required for motion analysis.')

        # Refuse to analyze a grouping that no longer matches the current tracks.
        snap = np.asarray(self.data.get('track_ids_snapshot', []))
        cur_ids = np.asarray(self.data.get('track_ids', []))
        if not np.array_equal(snap, cur_ids):
            raise ValueError('Tracks changed since group_tracks was run. Re-run '
                             'group_tracks(...) before analyze_track_motion().')

        kind = self.data['group_kind']
        n_groups = int(self.data['n_groups'])
        gid = np.asarray(self.data['track_group_id'])
        tracks_slen = np.asarray(self.data['tracks_slen'], dtype=float)

        # Domain mirrors the legacy mean-based domain_slen_timeseries; others default
        # to a robust median over member tracks.
        agg_method = aggregate if aggregate is not None else ('nanmean' if kind == 'domain' else 'nanmedian')

        agg = grouped_motion.aggregate_group_slen(
            tracks_slen, gid, n_groups, aggregate=agg_method, slen_lims=slen_lims)

        if model is None or model == 'default':
            model = os.path.join(self.model_dir, 'model_ContractionNet.pt')

        logger.info(f"analyze_track_motion: {n_groups} '{kind}' groups...")
        engine = grouped_motion.run_cycle_engine(
            agg['slen_timeseries'], frametime=self.metadata.frametime, model_path=model,
            threshold=threshold, contr_time_min=contr_time_min, merge_time_max=merge_time_max,
            buffer_frames=buffer_frames, min_valid_frames=min_valid_frames,
            filter_params=filter_params,
            group_label={'loi': 'LOI', 'mband': 'M-band'}.get(kind, str(kind).capitalize()),
            # domain group id = mask label - 1, so log 1-based to match the mask;
            # all other kinds log the 0-based group id (matches track_group_id / the API).
            id_offset=1 if kind == 'domain' else 0)

        # Write timeseries + engine outputs under the <kind>_* prefix.
        result: dict = {
            f'{kind}_slen_timeseries': agg['slen_timeseries'],
            f'{kind}_slen_median_timeseries': agg['slen_median_timeseries'],
            f'{kind}_slen_std_timeseries': agg['slen_std_timeseries'],
            f'{kind}_slen_q25_timeseries': agg['slen_q25_timeseries'],
            f'{kind}_slen_q75_timeseries': agg['slen_q75_timeseries'],
        }
        # Member count: domain keeps the legacy 'domain_n_vectors_timeseries' name.
        if kind == 'domain':
            result['domain_n_vectors_timeseries'] = agg['n_members_timeseries']
        else:
            result[f'{kind}_n_members_timeseries'] = agg['n_members_timeseries']
        # Engine keys are 'domain_<x>'; remap the leading token to <kind>.
        for k, v in engine.items():
            result[f'{kind}{k[len("domain"):]}'] = v

        result.update({
            'track_motion_kind': kind,
            'params.analyze_track_motion.group_kind': kind,
            'params.analyze_track_motion.grouping_hash': self.data['grouping_hash'],
            'params.analyze_track_motion.aggregate': agg_method,
            'params.analyze_track_motion.slen_lims': list(slen_lims),
            'params.analyze_track_motion.model': model,
            'params.analyze_track_motion.threshold': threshold,
            'params.analyze_track_motion.contr_time_min': contr_time_min,
            'params.analyze_track_motion.merge_time_max': merge_time_max,
            'params.analyze_track_motion.buffer_frames': buffer_frames,
            'params.analyze_track_motion.min_valid_frames': min_valid_frames,
            'params.analyze_track_motion.filter_params': filter_params,
            'params.analyze_track_motion.n_groups': n_groups,
        })
        self.data.update(result)
        logger.info(f"analyze_track_motion complete ('{kind}', {n_groups} groups).")
        if self.auto_save:
            self.store_structure_data()

    def _assert_track_motion_fresh(self) -> None:
        """Hard-raise if grouped track-motion results are missing or stale.

        Guards getters/plots so a grouping changed after analysis can never
        silently return the previous grouping's numbers.
        """
        used = self.data.get('params.analyze_track_motion.grouping_hash')
        if used is None:
            raise ValueError('No grouped track-motion analysis found. '
                             'Run analyze_track_motion() first.')
        cur = self.data.get('grouping_hash')
        if cur is None or cur != used:
            raise ValueError('Track grouping changed since analyze_track_motion() was run. '
                             'Re-run analyze_track_motion() before reading grouped results.')
        snap = np.asarray(self.data.get('track_ids_snapshot', []))
        cur_ids = np.asarray(self.data.get('track_ids', []))
        if not np.array_equal(snap, cur_ids):
            raise ValueError('Tracks changed since grouping. Re-run track_sarcomere_vectors '
                             '-> group_tracks -> analyze_track_motion.')

    def get_track_motion(self, group: int, *, analyze: bool = False,
                         persist_loi: bool = False) -> "Motion":
        """Return a :class:`~sarcasm.motion.Motion` view of one myofibril (fibre).

        The fibre's member tracks (ordered head-to-tail by ``track_group_order``)
        are turned into a synthesized LOI (``z_pos`` = cumulative arc-length of the
        member sarcomere lengths) and wrapped in a ``Motion`` object, so the full
        LOI analysis and **every existing LOI plot** (``plot_z_pos``,
        ``plot_delta_slen``, ``plot_phase_space``, ``plot_popping_events``, …) work
        unchanged on tracker-derived fibres. Requires a ``'myofibril'`` or
        ``'loi'`` grouping (:meth:`group_tracks` / :meth:`analyze_track_motion`
        with ``by='myofibril'`` or ``by='loi'``).

        Parameters
        ----------
        group : int
            Fibre group id (``0 .. n_groups-1``).
        analyze : bool, optional
            If True, run the standard LOI chain on the view
            (``detect_analyze_contractions`` -> ``get_trajectories`` ->
            ``analyze_trajectories``) so it is immediately plot-ready.
            Default is False.
        persist_loi : bool, optional
            If True, persist the synthesized LOI to ``{base}/track_myofibril_{group}/``.
            Default is False (purely in-memory, keeps the dataset directory clean).
        """
        from sarcasm.motion import Motion

        kind = self.data.get('group_kind')
        if kind not in ('myofibril', 'loi'):
            raise ValueError("get_track_motion requires a 'myofibril' or 'loi' grouping "
                             "(both order tracks along a fibre). "
                             "Run group_tracks(by='myofibril') or group_tracks(by='loi') first.")
        n_tracks = int(self.data['n_tracks'])
        gid = np.asarray(self.data['track_group_id']).reshape(-1)
        order = np.asarray(self.data['track_group_order']).reshape(-1)
        members = np.flatnonzero(gid == int(group))
        if members.size == 0:
            raise ValueError(f"No tracks in {kind} group {group} "
                             f'(n_groups={self.data.get("n_groups")}).')
        members = members[np.argsort(order[members])]  # head -> tail along the fibre

        slen = np.asarray(self.data['tracks_slen'], dtype=float).reshape(n_tracks, -1)[members]
        # Pass the members' measured positions so each Z-band boundary is placed from
        # its own sarcomere; accumulating them instead lets one missing member blank
        # every boundary below it.
        pos_um = np.asarray(self.data['tracks_positions_um'],
                            dtype=float).reshape(n_tracks, -1, 2)[members]
        # Anchor the chain geometry on the frame the grouping (and hence the
        # head-to-tail order) was built from.
        ref_frame = int(self.data.get('params.group_tracks.reference_frame', 0))
        try:
            ref_idx = self._tracked_frame_index(ref_frame)
        except (ValueError, KeyError):
            ref_idx = 0
        z_pos, slen_f, time = grouped_motion.synthesize_loi_chain(
            slen, self.metadata.frametime, member_pos=pos_um, ref_idx=ref_idx)
        loi_data = {
            'z_pos': z_pos, 'z_pos_raw': z_pos.copy(), 'slen': slen_f, 'time': time,
            'n_sarcomeres': int(members.size),
            'track_ids': np.asarray(self.data['track_ids']).reshape(-1)[members],
            'synthetic': True,
        }
        m = Motion.from_loi_data(self.file_path, f'track_{kind}_{int(group)}',
                                 loi_data, auto_save=persist_loi,
                                 frametime=self.metadata.frametime)
        if analyze:
            m.detect_analyze_contractions()
            m.get_trajectories()
            m.analyze_trajectories()
        return m

    def _grow_lois(self, frame: int = 0, ratio_seeds: float = 0.1, persistence: int = 2,
                   threshold_distance: float = 0.3, random_seed: Union[None, int] = None) -> None:
        """
        Find LOIs (lines of interest) using a line-growth algorithm.

        Parameters
        ----------
        frame : int, optional
            Index of the frame to analyze (i-th frame of the sarcomere-vector
            analysis frames). Default is 0.
        ratio_seeds : float, optional
            Ratio of sarcomere vectors used as seeds for line growth.
            Default is 0.1.
        persistence : int, optional
            Persistence of line (averaged vector length and orientation for prior
            estimation). Default is 2.
        threshold_distance : float, optional
            Maximal distance for nearest-neighbor estimation, in µm.
            Default is 0.3.
        random_seed : int or None, optional
            Random seed for reproducibility. Default is None.
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
        Filter Lines of Interest (LOIs) by geometric and morphological criteria.

        Parameters
        ----------
        number_lims : tuple of int, optional
            Limits (min, max) of sarcomere number in an LOI. Default is (10, 100).
        length_lims : tuple of float, optional
            Limits (min, max) for LOI length in µm. Default is (0, 200).
        sarcomere_mean_length_lims : tuple of float, optional
            Limits (min, max) for mean sarcomere length in an LOI.
            Default is (1, 3).
        sarcomere_std_length_lims : tuple of float, optional
            Limits (min, max) for the std of sarcomere lengths in an LOI.
            Default is (0, 1).
        midline_mean_length_lims : tuple of float, optional
            Limits (min, max) for mean midline length in an LOI.
            Default is (0, 50).
        midline_std_length_lims : tuple of float, optional
            Limits (min, max) for the std of midline length in an LOI.
            Default is (0, 50).
        midline_min_length_lims : tuple of float, optional
            Limits (min, max) for minimum midline length in an LOI.
            Default is (0, 50).
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
        symmetry_mode : {'min', 'max'}, optional
            Whether to take min or max of ``H(loi_i, loi_j)`` and
            ``H(loi_j, loi_i)``. Default is 'max'.
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
            Linkage distance threshold above which clusters are not merged.
            Default is 40.
        linkage : {'complete', 'average', 'single'}, optional
            Linkage criterion determining the inter-cluster distance to minimize
            when merging: 'average' uses the mean pairwise distance, 'complete'
            the maximum pairwise distance, 'single' the minimum pairwise distance.
            Default is 'single'.
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
        """Fit straight lines to cluster points.

        Parameters
        ----------
        add_length : float, optional
            Elongate each fitted line at its end by this amount, in µm.
            Default is 1.
        n_lois : int or None, optional
            If int, only the n longest LOIs are saved; if None, all are saved.
            Default is None.
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
        # A fitted straight line is synthetic geometry, not a chain of real
        # detections — drop any index chain from a previous selection so
        # group_tracks(by='loi') falls back to the geometric path instead of
        # pairing this line with a stale chain.
        self.data['loi_data']['loi_index_lines'] = None
        if self.auto_save:
            self.store_structure_data()

    def _longest_in_cluster(self, n_lois, frame):
        # Delegate to loi_detection module
        loi_lines, len_loi_lines, loi_index_lines = loi_detection.select_longest_in_cluster(
            lines=self.data['loi_data']['lines'],
            pos_vectors=self.data['pos_vectors_px'][frame],
            cluster_labels=self.data['loi_data']['line_cluster'],
            n_clusters=self.data['loi_data']['n_lines_clusters'],
            n_lois=n_lois
        )

        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = len_loi_lines
        # Keep the ordered detection-index chain of each selected LOI, so
        # group_tracks(by='loi') can rebuild it as a 1D head-to-tail thread of
        # sarcomeres instead of re-deriving membership from geometric proximity.
        self.data['loi_data']['loi_index_lines'] = np.asarray(loi_index_lines, dtype=object)
        if self.auto_save:
            self.store_structure_data()

    def _random_from_cluster(self, n_lois, frame):
        # Delegate to loi_detection module
        loi_lines, len_loi_lines, loi_index_lines = loi_detection.select_random_from_cluster(
            lines=self.data['loi_data']['lines'],
            pos_vectors=self.data['pos_vectors_px'][frame],
            cluster_labels=self.data['loi_data']['line_cluster'],
            n_clusters=self.data['loi_data']['n_lines_clusters'],
            n_lois=n_lois
        )

        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = len_loi_lines
        # Keep the ordered detection-index chain of each selected LOI, so
        # group_tracks(by='loi') can rebuild it as a 1D head-to-tail thread of
        # sarcomeres instead of re-deriving membership from geometric proximity.
        self.data['loi_data']['loi_index_lines'] = np.asarray(loi_index_lines, dtype=object)
        if self.auto_save:
            self.store_structure_data()

    def _random_lois(self, n_lois, frame):
        # Delegate to loi_detection module
        loi_lines, len_loi_lines, loi_index_lines = loi_detection.select_random_lois(
            lines=self.data['loi_data']['lines'],
            pos_vectors=self.data['pos_vectors_px'][frame],
            n_lois=n_lois
        )

        self.data['loi_data']['loi_lines'] = loi_lines
        self.data['loi_data']['len_loi_lines'] = len_loi_lines
        # Keep the ordered detection-index chain of each selected LOI, so
        # group_tracks(by='loi') can rebuild it as a 1D head-to-tail thread of
        # sarcomeres instead of re-deriving membership from geometric proximity.
        self.data['loi_data']['loi_index_lines'] = np.asarray(loi_index_lines, dtype=object)
        if self.auto_save:
            self.store_structure_data()

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
                    ) -> None:
        """
        Detect lines of interest (LOIs) — curated sarcomere/myofibril lines.

        Grows candidate lines from seed vectors, filters them by geometric and
        morphological criteria, clusters them, and selects one line per cluster
        (or fitted / random lines). The selected lines are stored as
        ``self.data['loi_data']['loi_lines']`` and consumed by :meth:`group_tracks`
        with ``by='loi'`` to group full-field tracks along the curated fibres.

        Parameters
        ----------
        frame : int, optional
            Index of the frame to analyze. Default is 0.
        n_lois : int, optional
            Number of LOIs to select. Default is 4.
        ratio_seeds : float, optional
            Ratio of sarcomere vectors used as seed vectors for LOI growth.
            Default is 0.1.
        persistence : int, optional
            Persistence parameter influencing line-growth direction and
            termination. Default is 4.
        threshold_distance : float, optional
            Maximum distance for nearest-neighbor estimation during line growth,
            in µm. Default is 0.5.
        mode : {'fit_straight_line', 'longest_in_cluster', 'random_from_cluster', 'random_line'}, optional
            Mode for selecting LOIs from identified clusters:
            'fit_straight_line' fits a straight line to all midline points in the
            cluster; 'longest_in_cluster' selects the longest (possibly curved)
            line per cluster; 'random_from_cluster' selects a random line per
            cluster; 'random_line' selects random lines passing the filters.
            Default is 'longest_in_cluster'.
        random_seed : int or None, optional
            Random seed for selecting random starting vectors (reproducibility).
            If None, results differ each run. Default is None.
        number_lims : tuple of int, optional
            Limits (min, max) for the number of sarcomeres within an LOI.
            Default is (10, 50).
        length_lims : tuple of float, optional
            Length limits (min, max) for LOIs in µm. Default is (0, 200).
        sarcomere_mean_length_lims : tuple of float, optional
            Limits (min, max) for mean sarcomere length within an LOI.
            Default is (1, 3).
        sarcomere_std_length_lims : tuple of float, optional
            Limits (min, max) for the std of sarcomere lengths within an LOI.
            Default is (0, 1).
        midline_mean_length_lims : tuple of float, optional
            Limits (min, max) for mean midline length within an LOI.
            Default is (0, 50).
        midline_std_length_lims : tuple of float, optional
            Limits (min, max) for the std of midline length within an LOI.
            Default is (0, 50).
        midline_min_length_lims : tuple of float, optional
            Limits (min, max) for minimum midline length within an LOI.
            Default is (0, 50).
        distance_threshold_lois : float, optional
            Distance threshold for clustering LOIs; clusters are not merged above
            it. Default is 40.
        linkage : {'complete', 'average', 'single'}, optional
            Linkage criterion for clustering. Default is 'single'.
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

        logger.info(
            f"detect_lois: selected {len(self.data['loi_data']['loi_lines'])} LOI line(s).")

    def full_analysis_structure(self, frames='all'):
        """
        Analyze sarcomere structure with default parameters at specified frames.

        Parameters
        ----------
        frames : {'all', int, list of int, np.ndarray}, optional
            Frames to analyze ('all', a single frame index, or selected frames).
            Default is 'all'.
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
