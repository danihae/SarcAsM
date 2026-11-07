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

"""Export structure and motion features to tabular (.xlsx/.csv) and JSON files.

:class:`Export` writes features from a single analyzed object; :class:`BatchExport`
collects features from many recordings into one table.
"""

import json
import logging
import os.path
from typing import Dict, List, Optional, Tuple

import types
import numpy as np

import pandas as pd
from scipy import sparse
from tqdm import tqdm as tqdm

from sarcasm._internal.meta_data_handler import ImageMetadata
from sarcasm.core import SarcAsM
from sarcasm.features import motion_feature_dict

logger = logging.getLogger(__name__)

# ImageMetadata dataclass fields — dropped from tabular (xlsx/csv) exports so
# the output contains only feature values. JSON exports retain metadata.
_METADATA_KEYS = frozenset(ImageMetadata.__dataclass_fields__.keys())
# Metadata that is per-frame or nested: never fits one table cell (a per-frame
# vector would be stringified and silently truncated past 1000 elements).
_NON_TABULAR_METADATA_KEYS = frozenset({'time', 'timestamps', 'user_info'})


class BatchExport:
    """
    Collect already-analyzed features from many recordings into one table (.xlsx, .csv or .json).

    Parameters
    ----------
    list_files : list
        List of tif files.
    folder : str
        Path to a folder to store data and results.
    experiment : str or None, optional
        Name of the experiment. Default is None.
    load_data : bool, optional
        Whether to load the dataframe from a previous analysis in the data folder.
        Default is False.
    **conditions
        Keyword arguments with constants or regex functions to extract information
        from the filename.

    Attributes
    ----------
    folder : str
        Path to the folder with data and results.
    experiment : str
        Name of the experiment.
    files : list
        List of tif files.
    conditions : dict
        Keyword arguments with constants or regex functions to extract information
        from the filename.
    data : pandas.DataFrame
        DataFrame holding the collected feature data.
    """

    def __init__(self, list_files: List, folder: str, experiment: str = None, load_data: bool = False, **conditions):
        self.folder = folder
        self.experiment = experiment
        self.files = list_files
        self.conditions = conditions
        self.data = None

        if load_data:
            self.load_data()

    def _require_data(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError('No data collected yet: call get_data(), get_motion_data(), '
                             'load_data() or load_motion_data() first.')
        return self.data

    def get_data(self, structure_keys=None):
        """
        Iterate files and collect structure features into ``self.data``.

        Parameters
        ----------
        structure_keys : list or None, optional
            Structure keys to extract; uses
            :attr:`Export.structure_keys_default` when None. Default is None.
        """
        self.data = []
        for i, tif_file in enumerate(tqdm(self.files)):
            try:
                sarc_obj = SarcAsM(file_path=tif_file)
                dict_i = Export.get_structure_dict(sarc_obj, structure_keys,
                                                   experiment=self.experiment,
                                                   **self.conditions)
                self.data.append(dict_i)
            except Exception as e:
                logger.error(f'{tif_file} failed!')
                logger.exception(f'Exception: {repr(e)}')

        self.data = pd.DataFrame.from_records(self.data)
        self.save_data()

    def get_motion_data(self, motion_keys=None, kinds=None):
        """
        Iterate files and collect per-group track-based motion features.

        One row per group per file; files without track motion are skipped.
        Saves to ``<folder>data_motion.pkl``.

        Parameters
        ----------
        motion_keys : list or None, optional
            Feature suffixes to extract; uses
            :attr:`Export.motion_keys_default` when None. Default is None.
        kinds : str, list of str or None, optional
            Grouping kinds to export. ``None`` (default) exports every kind
            analysed for the file. Kinds missing from a file are skipped.
            Default is None.
        """
        if isinstance(kinds, str):
            kinds = [kinds]
        records = []
        for tif_file in tqdm(self.files):
            try:
                sarc_obj = SarcAsM(file_path=tif_file)
                available = sarc_obj.motion._analyzed_kinds()
                wanted = [k for k in (kinds or available) if k in available]
                if not wanted:
                    logger.warning(f'{tif_file}: no track motion analyzed, skipping')
                    continue
                for kind in wanted:
                    records.extend(Export.get_motion_dict_per_group(
                        sarc_obj, motion_keys=motion_keys, kind=kind,
                        experiment=self.experiment, **self.conditions))
            except Exception as e:
                logger.error(f'{tif_file} failed!')
                logger.exception(f'Exception: {repr(e)}')
        self.data = pd.DataFrame.from_records(records)
        self.data.to_pickle(os.path.join(self.folder, 'data_motion.pkl'))

    def save_data(self):
        """Save the DataFrame to ``<folder>/data_structure.pkl``."""
        self._require_data().to_pickle(os.path.join(self.folder, 'data_structure.pkl'))

    def load_data(self):
        """
        Load the DataFrame from ``<folder>/data_structure.pkl``.

        Falls back to a ``data_structure.pd`` file when the ``.pkl`` file is
        absent.

        Raises
        ------
        FileExistsError
            If the data file does not exist in the specified folder.
        """
        path = os.path.join(self.folder, 'data_structure.pkl')
        if not os.path.exists(path):
            # backward compatibility with the legacy '.pd' extension
            legacy_path = os.path.join(self.folder, 'data_structure.pd')
            if not os.path.exists(legacy_path):
                raise FileExistsError('Data from previous analysis does not exist and cannot be loaded. '
                                      'Set load_data=False.')
            path = legacy_path
        self.data = pd.read_pickle(path)

    def load_motion_data(self):
        """Load the per-group motion table written by :meth:`get_motion_data`."""
        path = os.path.join(self.folder, 'data_motion.pkl')
        if not os.path.exists(path):
            raise FileExistsError(f'{path} does not exist; run get_motion_data() first.')
        self.data = pd.read_pickle(path)

    def export_data(self, file_path, fileformat: Optional[str] = None):
        """
        Export the collected table, one row per file (structure) or group (motion).

        A per-frame feature of a movie becomes one number per cell: the value of the
        analyzed frame, or the mean over the analyzed frames when there are several.
        Every analyzed frame is then also written in long format (one row per file and
        frame): a ``per_frame`` sheet in the .xlsx, ``<name>_per_frame.csv`` next to the
        .csv, or a ``per_frame`` list in the .json. An .xlsx file also gets a ``metrics``
        sheet with each metric's name, unit and description (:meth:`Export.append_info_sheets`).

        Parameters
        ----------
        file_path : str
            Path to the output file.
        fileformat : {'.xlsx', '.csv', '.json'} or None, optional
            Format of the output file; None takes it from the file extension. Default is None.
        """
        fmt = Export._format_of(file_path, fileformat)
        main, per_frame = Export.split_per_frame(self._require_data())
        table = Export.tabular_frame(main)
        if fmt == 'json':
            out = {'data': json.loads(table.to_json(orient='records'))}
            if per_frame is not None:
                out['per_frame'] = json.loads(per_frame.to_json(orient='records'))
            with open(file_path, 'w') as f:
                json.dump(out, f, indent=2)
        elif fmt == 'xlsx':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                table.to_excel(writer, index=False)
                if per_frame is not None:
                    per_frame.to_excel(writer, sheet_name='per_frame', index=False)
            structure = [c for c in table.columns if str(c).startswith('structure.')]
            if structure:
                Export.append_info_sheets(str(file_path), None, structure, registry='structure')
            else:
                motion = [c for c in table.columns if c in motion_feature_dict]
                Export.append_info_sheets(str(file_path), None, motion, registry='motion')
        else:
            table.to_csv(file_path, index=False)
            if per_frame is not None:
                root, ext = os.path.splitext(str(file_path))
                per_frame.to_csv(f'{root}_per_frame{ext}', index=False)


class Export:
    """
    Export the structure and motion features of an analyzed recording.

    Attributes
    ----------
    structure_keys_default : list
        Default structure feature keys.
    motion_keys_default : list
        Default track-based motion feature suffixes.
    """

    structure_keys_default = ['structure.cell.mask_area', 'structure.cell.mask_area_ratio', 'structure.cell.mask_intensity',
                              'structure.domain.area_mean', 'structure.domain.area_std', 'structure.domain.oop_mean',
                              'structure.domain.oop_std', 'structure.domain.slen_mean', 'structure.domain.n',
                              'structure.myofibril.length_max', 'structure.myofibril.length_mean', 'structure.myofibril.length_std',
                              'structure.myofibril.bending_mean', 'structure.myofibril.bending_std',
                              'structure.sarcomere.area', 'structure.sarcomere.area_ratio', 'structure.sarcomere.slen_mean',
                              'structure.sarcomere.slen_std', 'structure.sarcomere.oop', 'structure.zbands.n', 'structure.sarcomere.n_mbands', 'structure.sarcomere.n_vectors',
                              'structure.sarcomere.n_vectors_removed',
                              'structure.sarcomere.registry_lateral_mean', 'structure.sarcomere.registry_axial_mean',
                              'structure.sarcomere.registry_mean',
                              'structure.domain.registry_lateral_mean', 'structure.domain.registry_axial_mean',
                              'structure.zbands.intensity_mean', 'structure.zbands.intensity_std', 'structure.zbands.contrast_median',
                              'structure.zbands.lat_dist_mean', 'structure.zbands.lat_dist_std', 'structure.zbands.lat_length_groups_mean',
                              'structure.zbands.lat_neighbors_mean', 'structure.zbands.lat_neighbors_std', 'structure.zbands.length_max',
                              'structure.zbands.length_mean', 'structure.zbands.length_std', 'structure.zbands.oop', 'structure.zbands.mask_area', 'structure.zbands.mask_area_ratio',
                              'structure.zbands.mask_intensity']

    # Track-based grouped-motion feature suffixes (resolved per grouping kind to
    # ``<kind>_<suffix>`` keys written by MotionAnalysis.analyze). One value
    # per group; per-cycle arrays are collapsed to a per-group nanmean.
    motion_keys_default = ['beating_rate', 'beating_rate_variability', 'equ', 'n_contr',
                           'n_contr_complete',
                           'contr_max', 'elong_max', 'vel_contr_max', 'vel_elong_max',
                           'time_to_peak', 'time_to_relax', 'time_contr',
                           'time_to_50', 'time_contr_10_90', 'time_relax_50', 'time_relax_90',
                           'slen_std', 'equ_std',
                           'corr_delta_slen_serial', 'corr_delta_slen_mutual', 'ratio_delta_slen_mutual_serial',
                           'corr_vel_serial', 'corr_vel_mutual', 'ratio_vel_mutual_serial',
                           'popping_rate', 'still_short_rate', 'starts_long_rate', 'lengthens_first_rate',
                           'oscill_peak_avg', 'oscill_peak_1_single', 'oscill_peak_2_single']

    @staticmethod
    def get_structure_dict(sarc_obj, structure_keys=None, **conditions):
        """
        Create a dictionary of structure and metadata features from a SarcAsM object.

        Parameters
        ----------
        sarc_obj : SarcAsM
            Analyzed recording.
        structure_keys : list or None, optional
            Structure keys; uses :attr:`Export.structure_keys_default` when None.
            Default is None.
        **conditions
            Extra columns: constants or filename-regex functions
            (e.g. ``cell_line='wt'``, ``info_xyz=42``).

        Returns
        -------
        dict
            Selected metadata and structure features.
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
    def export_structure_data(sarc_obj: SarcAsM, file_path: str, structure_keys=None,
                              fileformat: Optional[str] = None, raw: bool = False) -> None:
        """
        Export the structure features of one recording to a file.

        The GUI's structure export runs this too. Summary mode (``raw=False``, default)
        writes one value per metric per frame: multi-frame analyses become a single table
        with one column per frame (``frame_0``, ``frame_1``, ...), single-frame analyses
        collapse to a single ``value`` column. Full mode (``raw=True``) preserves per-object
        distributions and requires .json. See :meth:`Export.write_dict` for the full layout.
        An .xlsx file also gets a ``metadata`` and a ``metrics`` sheet
        (:meth:`append_info_sheets`).

        Parameters
        ----------
        sarc_obj : SarcAsM
            Analyzed recording.
        file_path : str
            Path to the output file.
        structure_keys : list or None, optional
            Structure keys; uses :attr:`Export.structure_keys_default` when None.
            Default is None.
        fileformat : {'.xlsx', '.csv', '.json'} or None, optional
            Format of the output file; None takes it from the file extension. Default is None.
        raw : bool, optional
            If True, export raw per-object distributions (JSON only).
            Default is False.
        """
        fmt = Export._format_of(file_path, fileformat)
        structure_keys = list(structure_keys) if structure_keys is not None else Export.structure_keys_default
        structure_dict = Export.get_structure_dict(sarc_obj, structure_keys=structure_keys)
        Export.write_dict(file_path, structure_dict, fmt, raw=raw)
        if fmt == 'xlsx':
            Export.append_info_sheets(file_path, sarc_obj, structure_keys, registry='structure')

    @staticmethod
    def _format_of(file_path: str, fileformat: Optional[str]) -> str:
        """``fileformat`` without its dot, or the extension of ``file_path`` when it is None."""
        fmt = fileformat if fileformat is not None else os.path.splitext(str(file_path))[1]
        fmt = fmt.lower().lstrip('.')
        if fmt not in ('xlsx', 'csv', 'json'):
            raise ValueError(f"Unsupported file format {fmt!r}: use .xlsx, .csv or .json.")
        return fmt

    @staticmethod
    def append_info_sheets(file_path: str, sarc_obj, keys: List[str], registry: str = 'structure') -> None:
        """Add a ``metadata`` and a ``metrics`` sheet to an exported .xlsx workbook.

        ``metadata`` lists the analyzed file and its calibration (one row per field;
        skipped when ``sarc_obj`` is None, e.g. for a batch table with one row per file);
        ``metrics`` lists each exported metric with its display name, unit and
        description. The existing sheets are left unchanged.
        """
        import re
        from sarcasm.features import describe_key
        meta_df = None
        if sarc_obj is not None:
            meta = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                    for k, v in sarc_obj.metadata.to_dict().items() if k not in _NON_TABULAR_METADATA_KEYS}
            meta_df = pd.DataFrame({'field': list(meta), 'value': list(meta.values())})
        rows = []
        for key in keys:
            entry = describe_key(key, registry) or {}
            name = entry.get('name', key)
            unit = re.search(r'\[([^\]]+)\]', name)
            rows.append({'metric': key, 'name': re.sub(r'\s*\[[^\]]+\]', '', name),
                         'unit': unit.group(1) if unit else '',
                         'description': entry.get('description', ''),
                         'computed by': entry.get('function', '')})
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            if meta_df is not None:
                meta_df.to_excel(writer, sheet_name='metadata', index=False)
            pd.DataFrame.from_records(rows).to_excel(writer, sheet_name='metrics', index=False)

    @staticmethod
    def split_per_frame(records) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Split per-frame feature arrays out of a one-row-per-file table.

        A cell holding a numeric array with one entry per frame (``n_stack`` > 1; frames
        not analyzed are NaN) is replaced by the analyzed frame's value, or by the mean
        over the analyzed frames when there are several. Returns that table and a long
        table (``file_name``, ``file_path``, ``frame`` and the per-frame features, one
        row per file and analyzed frame), or None when no cell held per-frame values.
        """
        df = (records if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records)).copy()
        if 'n_stack' not in df.columns:
            return df, None
        long_rows = []
        for i in df.index:
            n = df.at[i, 'n_stack']
            if not isinstance(n, (int, np.integer)) or n < 2:
                continue
            frames: Dict[int, dict] = {}
            for col in df.columns:
                if col in _METADATA_KEYS:
                    continue
                v = df.at[i, col]
                if not isinstance(v, (list, np.ndarray)) or len(v) != n:
                    continue
                arr = np.asarray(v, dtype=object)
                try:
                    arr = arr.astype(float)
                except (TypeError, ValueError):
                    continue                                   # ragged per-object lists stay as they are
                finite = np.flatnonzero(np.isfinite(arr))
                df.at[i, col] = float(arr[finite].mean()) if finite.size else np.nan
                for f in finite:
                    frames.setdefault(int(f), {})[col] = float(arr[f])
            for f in sorted(frames):
                long_rows.append({'file_name': df.at[i, 'file_name'] if 'file_name' in df.columns else None,
                                  'file_path': df.at[i, 'file_path'] if 'file_path' in df.columns else None,
                                  'frame': f, **frames[f]})
        return df, (pd.DataFrame.from_records(long_rows) if long_rows else None)

    @staticmethod
    def flatten_single(x):
        """Return the lone element if x is a 1-element list/ndarray; otherwise x."""
        if isinstance(x, (list, np.ndarray)) and len(x) == 1:
            return x[0]
        return x

    @staticmethod
    def tabular_frame(records) -> pd.DataFrame:
        """One-row-per-object table from records (or a DataFrame) for xlsx/csv output.

        Drops the per-frame / nested metadata columns (``time``, ``timestamps``,
        ``user_info``) that cannot live in a single cell and collapses 1-element
        sequences to scalars.
        """
        df = records if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records)
        df = df.drop(columns=[c for c in df.columns if c in _NON_TABULAR_METADATA_KEYS])
        return df.map(Export.flatten_single)

    @staticmethod
    def get_motion_dict_per_group(sarc_obj, motion_keys=None, kind=None, **conditions):
        """
        Build one record per group of track-based motion features from a SarcAsM object.

        Reads the ``motion.<kind>.<suffix>`` keys written by
        :meth:`MotionAnalysis.analyze` (kind = pool / mband / myofibril /
        domain / loi). Each record is one group; per-cycle arrays (e.g.
        ``contr_max``) are collapsed to a per-group ``nanmean``.

        Parameters
        ----------
        sarc_obj : SarcAsM
            Analyzed object holding the ``motion.<kind>.*`` keys.
        motion_keys : list or None, optional
            Feature suffixes; uses :attr:`Export.motion_keys_default` when None.
            Default is None.
        kind : str
            The grouping kind to export.
        **conditions
            Extra columns: constants or filename-regex functions.

        Returns
        -------
        list of dict
            One record per group (metadata + group_id + selected features).
        """
        data = sarc_obj.data
        if kind is None:
            raise ValueError("Name the grouping kind to export, e.g. kind='myofibril'.")
        if motion_keys is None:
            motion_keys = Export.motion_keys_default
        metadata_dict = sarc_obj.metadata.to_dict()
        if kind not in sarc_obj.motion._analyzed_kinds():
            raise ValueError(
                f"No '{kind}' track motion in this store. "
                f"Run sarc.motion.analyze('{kind}') first; "
                f"analysed kinds: {sarc_obj.motion._analyzed_kinds()}.")
        n_groups = int(data[f'motion.groups.{kind}.n'])
        member_counts = np.asarray(data.get(f'motion.groups.{kind}.member_counts',
                                            np.full(n_groups, np.nan)))

        cond = {}
        for condition, value in conditions.items():
            cond[condition] = value(sarc_obj.file_path) if isinstance(value, types.FunctionType) else value

        records = []
        for g in range(n_groups):
            row = {**metadata_dict, 'kind': kind, 'group_id': g,
                   'group_member_count': float(member_counts[g]) if g < member_counts.size else np.nan}
            for suffix in motion_keys:
                arr = data.get(f'motion.{kind}.{suffix}')
                row[suffix] = Export._group_feature_value(arr, g)
            row.update(cond)
            row['tif_name'] = sarc_obj.file_path
            records.append(row)
        return records

    @staticmethod
    def _group_feature_value(arr, g):
        """Per-group scalar for feature array ``arr``; per-cycle arrays -> nanmean."""
        if arr is None:
            return np.nan
        arr = np.asarray(arr)
        if arr.ndim == 0 or g >= arr.shape[0]:
            return np.nan
        vg = arr[g]
        if isinstance(vg, np.ndarray):
            vg = vg.astype(float)
            return float(np.nanmean(vg)) if vg.size and np.isfinite(vg).any() else np.nan
        return vg

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
    def write_records(file_path: str, records: list, fileformat: str) -> None:
        """Write per-group records as a tidy table (one row per group).

        ``fileformat`` is ``'csv'``, ``'xlsx'`` or ``'json'`` (leading dot optional).
        """
        fmt = fileformat.lower().lstrip('.')
        if fmt == 'json':
            with open(file_path, 'w') as f:
                json.dump([Export.to_json_friendly(r) for r in records], f, indent=2, allow_nan=False)
            return
        if fmt not in ('xlsx', 'csv'):
            raise ValueError(f'Unsupported file format: {fileformat}')
        df = Export.tabular_frame(records)
        if fmt == 'xlsx':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='data', index=False)
        else:
            df.to_csv(file_path, index=False)

    @staticmethod
    def export_motion_data(sarc_obj: SarcAsM, file_path: str, kind: str, motion_keys=None,
                           fileformat: Optional[str] = None) -> int:
        """
        Export the motion features of one grouping kind, one row per group.

        The GUI's motion export runs this too. Per-cycle arrays are averaged per group. An
        .xlsx file also gets a ``metadata`` and a ``metrics`` sheet (:meth:`append_info_sheets`).

        Parameters
        ----------
        sarc_obj : SarcAsM
            Analyzed recording (``sarc.motion.analyze(kind)`` has run).
        file_path : str
            Path to the output file.
        kind : str
            The grouping kind to export ('pool', 'mband', 'myofibril', ...).
        motion_keys : list or None, optional
            Feature suffixes; uses :attr:`Export.motion_keys_default` when None.
            Default is None.
        fileformat : {'.xlsx', '.csv', '.json'} or None, optional
            Format of the output file; None takes it from the file extension. Default is None.

        Returns
        -------
        int
            The number of groups written.
        """
        fmt = Export._format_of(file_path, fileformat)
        motion_keys = list(motion_keys) if motion_keys is not None else Export.motion_keys_default
        records = Export.get_motion_dict_per_group(sarc_obj, motion_keys=motion_keys, kind=kind)
        Export.write_records(file_path, records, fmt)
        if fmt == 'xlsx':
            Export.append_info_sheets(file_path, sarc_obj, motion_keys, registry='motion')
        return len(records)
