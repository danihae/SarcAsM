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

import json
import logging
import os.path
from typing import Union, List, Optional

import types
import numpy as np

import pandas as pd
from scipy import sparse
from tqdm import tqdm as tqdm

from sarcasm.meta_data_handler import ImageMetadata
from sarcasm.structure import Structure
from sarcasm.motion import Motion

logger = logging.getLogger(__name__)

# ImageMetadata dataclass fields — dropped from tabular (xlsx/csv) exports so
# the output contains only feature values. JSON exports retain metadata.
_METADATA_KEYS = frozenset(ImageMetadata.__dataclass_fields__.keys())


class MultiStructureAnalysis:
    """
    Class for multi-tif-file comparison of structure.

    Parameters
    ----------
    list_files : list
        List of tif files.
    folder : str
        Path to a folder to store data and results.
    experiment : str, optional
        Name of the experiment (default is None).
    load_data : bool, optional
        Whether to load the dataframe from previous analysis from the data folder (default is False).
    **conditions : dict
        Keyword arguments with regex functions to extract information from the filename.

    Attributes
    ----------
    folder : str
        Path to the folder with data and results.
    experiment : str
        Name of the experiment.
    files : list
        List of tif files.
    conditions : dict
        Keyword arguments with regex functions to extract information from the filename.
    data : pandas.DataFrame
        DataFrame to store the structure data.
    """

    def __init__(self, list_files: List, folder: str, experiment: str = None, load_data: bool = False, **conditions):
        self.folder = folder
        self.experiment = experiment
        self.files = list_files
        self.conditions = conditions
        self.data = pd.DataFrame

        if load_data:
            self.load_data()

    def get_data(self, structure_keys=None):
        """
        Iterate files and get structure data.

        Parameters
        ----------
        structure_keys : list, optional
            List of keys to extract structure data (default is None).

        Returns
        -------
        None
        """
        self.data = []
        for i, tif_file in enumerate(tqdm(self.files)):
            try:
                sarc_obj = Structure(file_path=tif_file)
                dict_i = Export.get_structure_dict(sarc_obj, structure_keys,
                                                   experiment=self.experiment,
                                                   **self.conditions)
                self.data.append(dict_i)
            except Exception as e:
                logger.error(f'{tif_file} failed!')
                logger.exception(f'Exception: {repr(e)}')

        self.data = pd.DataFrame.from_records(self.data)
        self.save_data()

    def save_data(self):
        """
        Save the DataFrame to the data folder.

        Returns
        -------
        None
        """
        self.data.to_pickle(self.folder + 'data_structure.pd')

    def load_data(self):
        """
        Load the DataFrame from the data folder.

        Returns
        -------
        None

        Raises
        ------
        FileExistsError
            If the data file does not exist in the specified folder.
        """
        if os.path.exists(self.folder + 'data_structure.pd'):
            self.data = pd.read_pickle(self.folder + 'data_structure.pd')
        else:
            raise FileExistsError('Data from previous analysis does not exist and cannot be loaded. '
                                  'Set load_data=False.')

    def export_data(self, file_path, format='.xlsx'):
        """
        Export the DataFrame to .xlsx or .csv format.

        Parameters
        ----------
        file_path : str
            Path to the output file.
        format : str, optional
            Format of the output file ('.xlsx' or '.csv') (default is '.xlsx').

        Returns
        -------
        None
        """
        _data = self.data.applymap(Export.flatten_single)
        if format == '.xlsx':
            _data.to_excel(file_path, index=False)
        elif format == '.csv':
            _data.to_csv(file_path, index=False)
        else:
            raise ValueError('Unsupported file format')


class MultiLOIAnalysis:
    """
    Class for multi-LOI comparison.

    Parameters
    ----------
    list_lois : list
        List of tuples containing tif file paths and LOI names.
    folder : str
        Path to a folder to store data and results.
    load_data : bool, optional
        Whether to load the dataframe from previous analysis from the folder (default is False).
    **conditions : dict
        Keyword arguments with regex functions to extract information from the filename.

    Attributes
    ----------
    folder : str
        Path to the folder with data and results.
    lois : list
        List of tuples containing tif file paths and LOI names.
    conditions : dict
        Keyword arguments with regex functions to extract information from the filename.
    data : pandas.DataFrame
        DataFrame to store the motion data.
    """

    def __init__(self, list_lois, folder, load_data=False, **conditions):
        self.folder = folder
        self.lois = list_lois
        self.conditions = conditions
        self.data = pd.DataFrame()

        if load_data:
            self.load_data()

    def get_data(self, loi_keys=None):
        """
        Iterate files and get motion data.

        Parameters
        ----------
        loi_keys : list, optional
            List of keys to extract motion data (default is None).

        Returns
        -------
        None
        """
        self.data = []
        for tif_file, loi_name in tqdm(self.lois):
            try:
                motion_obj = Motion(tif_file, loi_name)
                dict_i = Export.get_motion_dict(motion_obj, loi_keys, **self.conditions)
                self.data.append(dict_i)
            except Exception as e:
                logger.error(f'{tif_file}, {loi_name} failed!')
                logger.exception(f'Exception: {repr(e)}')

        self.data = pd.DataFrame.from_records(self.data)
        self.save_data()

    def save_data(self):
        """
        Save the DataFrame to the data folder as a pandas DataFrame.

        Returns
        -------
        None
        """
        self.data.to_pickle(self.folder + 'data_motion.pd')

    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the DataFrame from the data folder.

        Parameters
        ----------
        path : str, optional
            Path of pickle file with motion data. If None, fallback to data_motion.pd in self.folder.

        Returns
        -------
        None

        Raises
        ------
        FileExistsError
            If the data file does not exist in the specified folder.
        """
        if path is None:
            path = self.folder + 'data_motion.pd'
        if path is not None and os.path.exists(path):
            self.data = pd.read_pickle(path)
        else:
            raise FileExistsError('Data from previous analysis does not exist and cannot be loaded. '
                                  'Set load_data=False.')

    def export_data(self, file_path, format='.xlsx'):
        """
        Export the DataFrame to .xlsx or .csv format.

        Parameters
        ----------
        file_path : str
            Path to the output file.
        format : str, optional
            Format of the output file ('.xlsx' or '.csv') (default is '.xlsx').

        Returns
        -------
        None
        """
        _data = self.data.applymap(Export.flatten_single)
        if format == '.xlsx':
            _data.to_excel(file_path, index=False)
        elif format == '.csv':
            _data.to_csv(file_path, index=False)
        else:
            raise ValueError('Unsupported file format')


class Export:
    """
    A class used to export structure and motion data from SarcAsM and Motion objects.

    Attributes
    ----------
    structure_keys_default : list
        Default structure keys.
    motion_keys_default : list
        Default motion keys.
    """

    structure_keys_default = ['cell_mask_area', 'cell_mask_area_ratio', 'cell_mask_intensity',
                              'domain_area_mean', 'domain_area_std', 'domain_oop_mean',
                              'domain_oop_std', 'domain_slen_mean', 'n_domains',
                              'myof_length_max', 'myof_length_mean', 'myof_length_std',
                              'myof_bending_mean', 'myof_bending_std',
                              'myof_straightness_mean', 'myof_straightness_std',
                              'sarcomere_area', 'sarcomere_area_ratio', 'sarcomere_length_mean',
                              'sarcomere_length_std', 'sarcomere_oop', 'n_zbands', 'n_mbands', 'n_vectors',
                              'z_intensity_mean', 'z_intensity_std', 'z_lat_alignment_mean',
                              'z_lat_alignment_std', 'z_lat_dist_mean', 'z_lat_dist_std', 'z_lat_length_groups_mean',
                              'z_lat_neighbors_mean', 'z_lat_neighbors_std', 'z_length_max',
                              'z_length_mean', 'z_length_std', 'z_oop', 'z_mask_area', 'z_mask_area_ratio',
                              'z_mask_intensity', 'z_straightness_mean', 'z_straightness_std']

    motion_keys_default = ['beating_rate', 'beating_rate_variability', 'contr_max', 'contr_max_avg', 'elong_max',
                           'elong_max_avg', 'equ', 'time', 'vel_contr_max', 'vel_contr_max_avg', 'vel_elong_max',
                           'vel_elong_max_avg', 'n_sarcomeres', 'n_contr', 'ratio_nans',
                           'popping_rate_contr', 'popping_rate_sarcomeres', 'popping_rate',
                           'popping_events', 'popping_dist', 'popping_tau',
                           'popping_ks_dist_pvalue', 'popping_ks_dist_statistic', 'popping_p_dist', 'popping_p_tau',
                           'popping_ks_tau_pvalue', 'popping_ks_tau_statistic', 'time_to_peak', 'time_to_peak_avg',
                           'time_contr', 'time_quiet',
                           'corr_delta_slen', 'corr_vel',
                           'corr_delta_slen_serial', 'corr_delta_slen_mutual', 'corr_vel_serial', 'corr_vel_mutual',
                           'ratio_delta_slen_mutual_serial', 'ratio_vel_mutual_serial']

    @staticmethod
    def get_structure_dict(sarc_obj, structure_keys=None, **conditions):
        """
        Create a dictionary of structure and metadata features from a SarcAsM object.

        Parameters
        ----------
        sarc_obj : SarcAsM
            Object of SarcAsM class or Motion class.
        structure_keys : list, optional
            List of structure keys (default is None).
        conditions : kwargs
            Keyword arguments to add information to the dictionary (e.g., "cell_line"= "wt", "info_xyz"=42).

        Returns
        -------
        dict
            Dictionary containing selected metadata and structure features.
        """
        metadata_dict = sarc_obj.metadata.to_dict()
        if structure_keys is None:
            structure_keys = Export.structure_keys_default
        missing_structure_keys = [key for key in structure_keys if key not in sarc_obj.data]
        if missing_structure_keys:
            logger.warning(f'Missing structure keys: {missing_structure_keys}')
        dict_structure_select = {key: sarc_obj.data.get(key, np.nan) for key in structure_keys}
        dict_ = {**metadata_dict, **dict_structure_select}
        for condition, value in conditions.items():
            if isinstance(value, types.FunctionType):
                dict_[condition] = value(sarc_obj.file_path)
            else:
                dict_[condition] = value
        return dict_

    @staticmethod
    def export_structure_data(file_path, sarc_obj: Union[Structure, Motion], structure_keys=None,
                              fileformat='.xlsx', raw: bool = False):
        """
        Export structure data to a file.

        Summary mode (``raw=False``, default) writes one value per metric per
        frame: multi-frame analyses become a single table with one column per
        frame (``frame_0``, ``frame_1``, ...), single-frame analyses collapse
        to a single ``value`` column. Full mode (``raw=True``) preserves
        per-object distributions and requires ``fileformat='.json'``.
        See :meth:`Export.write_dict` for the full layout.

        Parameters
        ----------
        file_path : str
            Path to the output file.
        sarc_obj : SarcAsM
            Object of SarcAsM class.
        structure_keys : list, optional
            List of structure keys (default is None).
        fileformat : str, optional
            Format of the output file: ``'.xlsx'``, ``'.csv'``, or ``'.json'``
            (default is ``'.xlsx'``).
        raw : bool, optional
            If True, export raw per-object distributions (JSON only).
            Default False.
        """
        structure_dict = Export.get_structure_dict(sarc_obj, structure_keys=structure_keys)
        Export.write_dict(file_path, structure_dict, fileformat, raw=raw)

    @staticmethod
    def flatten_single(x):
        """Return the lone element if x is a 1-element list/ndarray; otherwise x."""
        if isinstance(x, (list, np.ndarray)) and len(x) == 1:
            return x[0]
        return x

    @staticmethod
    def get_motion_dict(motion_obj, loi_keys=None, concat=False, **conditions):
        """
        Create a dictionary of motion features and metadata from a Motion object.

        Parameters
        ----------
        motion_obj : Motion
            Object of Motion class for LOI analysis.
        loi_keys : list, optional
            List of LOI keys (default is None).
        concat : bool, optional
            If True, all 2D arrays will be concatenated to 1D arrays (default is False).
        conditions : kwargs
            Keyword arguments to add to the dictionary, can be any information, e.g., drug='ABC'.

        Returns
        -------
        dict
            Dictionary containing selected metadata and motion features.
        """
        metadata_dict = motion_obj.metadata.to_dict()
        if loi_keys is None:
            loi_keys = Export.motion_keys_default
        missing_loi_keys = [key for key in loi_keys if key not in motion_obj.loi_data]
        if missing_loi_keys:
            logger.warning(f'Missing loi keys: {missing_loi_keys}')
        dict_loi_select = {key: motion_obj.loi_data[key] if key in motion_obj.loi_data else np.nan for key in loi_keys}
        dict_ = {**metadata_dict, **dict_loi_select, 'loi_name': motion_obj.loi_name}
        for condition, value in conditions.items():
            if isinstance(value, types.FunctionType):
                dict_[condition] = value(motion_obj.file_path)
            else:
                dict_[condition] = value
        if concat:
            for key, value in dict_.items():
                if isinstance(value, np.ndarray):
                    if len(value.shape) == 2:
                        dict_[key] = np.concatenate(value)
        dict_['tif_name'] = motion_obj.file_path
        return dict_

    @staticmethod
    def to_json_friendly(d: dict) -> dict:
        """Recursively convert numpy / sparse types to JSON-serializable values."""
        def _cast(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating, float)):
                v = float(x)
                return v if np.isfinite(v) else None
            if isinstance(x, np.ndarray):
                return _cast(x.tolist())
            if sparse.issparse(x):
                return _cast(x.toarray().tolist())
            if isinstance(x, (list, tuple)):
                return [_cast(v) for v in x]
            if isinstance(x, dict):
                return {str(k): _cast(v) for k, v in x.items()}
            return x
        return {str(k): _cast(v) for k, v in d.items()}

    @staticmethod
    def _infer_n_frames(d: dict) -> Optional[int]:
        """Infer the number of frames / z-sections from a features dict."""
        for key in ('n_stack', 'n_frames'):
            v = d.get(key)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)
        lengths: List[int] = []
        for v in d.values():
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 0:
                lengths.append(v.size)
            elif isinstance(v, list) and v:
                lengths.append(len(v))
        if not lengths:
            return None
        from collections import Counter
        return Counter(lengths).most_common(1)[0][0]

    @staticmethod
    def _classify_for_framewise(d: dict, n_frames: Optional[int]):
        """Split dict into (scalars, per_frame, ragged, other) for tabular export.

        - scalars: plain scalar / str / None, or ndarrays with only one entry.
        - per_frame: 1D ndarray with length == n_frames (frame-indexed scalars).
        - ragged: lists of length n_frames whose elements are arrays or None
          (ragged per-object distributions per frame).
        - other: multi-dim arrays, mismatched-length arrays, etc.
        """
        scalars, per_frame, ragged, other = {}, {}, {}, {}
        for k, v in d.items():
            if v is None or isinstance(v, (int, float, str, bool, np.integer, np.floating)):
                scalars[k] = v
            elif isinstance(v, np.ndarray):
                if v.ndim == 1 and n_frames is not None and v.size == n_frames:
                    per_frame[k] = v
                elif v.size == 1:
                    scalars[k] = v.ravel()[0]
                else:
                    other[k] = v
            elif isinstance(v, list):
                if (n_frames is not None and len(v) == n_frames and
                        any(isinstance(x, np.ndarray) for x in v)):
                    ragged[k] = v
                else:
                    other[k] = v
            else:
                other[k] = v
        return scalars, per_frame, ragged, other

    @staticmethod
    def write_dict(file_path: str, d: dict, fileformat: str, raw: bool = False) -> None:
        """Write a features dict to disk.

        ``fileformat`` is one of ``'csv'``, ``'xlsx'``, ``'json'`` (leading dot optional).

        Two modes:

        * ``raw=False`` (default, *Summary*): one value per metric per frame.
          For xlsx/csv, rows are metric names and columns are
          ``frame_0, frame_1, ..., frame_{N-1}``; single-frame analyses
          collapse to a single ``value`` column. Scalar metadata values are
          broadcast across every frame column. Ragged per-object distributions
          are collapsed to a per-frame ``nanmean``. JSON writes the same
          content as scalars / per-frame lists (ragged collapsed).
        * ``raw=True`` (*Full*): full nested structure including per-object
          distributions. **JSON only** — per-object arrays can contain
          thousands of values per frame and do not fit a single table; xlsx
          and csv raise ``ValueError``.
        """
        fmt = fileformat.lower().lstrip('.')

        if raw:
            if fmt != 'json':
                raise ValueError(
                    f'Full (raw) export only supports JSON, got {fileformat!r}. '
                    f'Per-object distributions do not fit a single table.'
                )
            with open(file_path, 'w') as f:
                json.dump(Export.to_json_friendly(d), f, indent=2, allow_nan=False)
            return

        if fmt not in ('xlsx', 'csv', 'json'):
            raise ValueError(f'Unsupported file format: {fileformat}')

        n_frames = Export._infer_n_frames(d)
        if fmt in ('xlsx', 'csv'):
            d = {k: v for k, v in d.items() if k not in _METADATA_KEYS}
        scalars, per_frame, ragged, other = Export._classify_for_framewise(d, n_frames)

        # Collapse ragged per-object distributions to per-frame nanmean so
        # they join the single-table layout as regular per-frame rows.
        for k, v in ragged.items():
            vals = np.full(n_frames, np.nan)
            for i, arr in enumerate(v):
                if isinstance(arr, np.ndarray) and arr.size > 0 and arr.dtype.kind in 'biufc':
                    vals[i] = float(np.nanmean(arr.astype(float)))
            per_frame[k] = vals

        if fmt == 'json':
            summary = {**scalars, **per_frame}
            if other:
                logger.debug('write_dict(summary, json): dropping non-summarizable keys %s',
                             list(other))
            with open(file_path, 'w') as f:
                json.dump(Export.to_json_friendly(summary), f, indent=2, allow_nan=False)
            return

        if per_frame:
            cols = [f'frame_{i}' for i in range(n_frames)]
        else:
            cols = ['value']
        ncols = len(cols)

        def _stringify_other(v):
            if isinstance(v, np.ndarray):
                return f'ndarray shape={v.shape} dtype={v.dtype}'
            if isinstance(v, (list, tuple)):
                return str(list(v))
            return f'{type(v).__name__}'

        rows = {}
        for k in d:
            if k in scalars:
                rows[k] = [scalars[k]] * ncols
            elif k in per_frame:
                rows[k] = list(per_frame[k])
            elif k in other:
                rows[k] = [_stringify_other(other[k])] * ncols

        df = pd.DataFrame.from_dict(rows, orient='index', columns=cols)
        df.index.name = 'metric'

        if fmt == 'xlsx':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='data')
            return
        df.to_csv(file_path)

    @staticmethod
    def export_motion_data(mot_obj: Motion, file_path, motion_keys=None, fileformat='.xlsx',
                           raw: bool = False):
        """
        Export motion data to a file.

        Summary mode (``raw=False``, default) writes one value per metric per
        frame as a single table with one column per frame (``frame_0``,
        ``frame_1``, ...). Full mode (``raw=True``) preserves per-object
        distributions and requires ``fileformat='.json'``. See
        :meth:`Export.write_dict` for the full layout.

        Parameters
        ----------
        mot_obj : Motion
            Object of Motion class.
        file_path : str
            Path to the output file.
        motion_keys : list, optional
            List of motion keys (default is None).
        fileformat : str, optional
            Format of the output file: ``'.xlsx'``, ``'.csv'``, or ``'.json'``
            (default is ``'.xlsx'``).
        raw : bool, optional
            If True, export raw per-object distributions (JSON only).
            Default False.
        """
        motion_dict = Export.get_motion_dict(mot_obj, loi_keys=motion_keys)
        Export.write_dict(file_path, motion_dict, fileformat, raw=raw)
