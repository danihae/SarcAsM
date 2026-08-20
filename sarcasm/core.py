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

"""Base class with file paths, metadata handling and lazy image/mask loading from the OME-Zarr store."""

import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, Literal, Dict, Any, List

import numpy as np
import tifffile
import torch

from sarcasm.exceptions import MetaDataError
from sarcasm._internal.meta_data_handler import ImageMetadata
from sarcasm.io.ome_store import (
    OmeZarrStore,
    detect_legacy_layout,
    legacy_layout_message,
    remove_tree,
    store_path_for,
)
from sarcasm.utils import Utils

logger = logging.getLogger(__name__)


class SarcAsMBase:
    """
    Base class for sarcomere structural and functional analysis.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to the TIFF file for analysis.
    restart : bool, optional
        If True, deletes existing analysis and starts fresh. Default is False.
    pixelsize : float or None, optional
        Physical pixel size in micrometres (µm). If None, extracted from file
        metadata; otherwise provided manually. Default is None.
    frametime : float or None, optional
        Time between frames in seconds. If None, extracted from file metadata;
        otherwise provided manually. Default is None.
    channel : int or None, optional
        Channel index holding the sarcomere signal in multi-channel stacks.
        Default is None.
    axes : str or None, optional
        Explicit order of image dimensions (e.g. ``'TXYC'`` or ``'YX'``).
        If None, auto-detected from OME-XML, ImageJ tags or shape heuristics.
        Default is None.
    auto_save : bool, optional
        Automatically save analysis results when True. Default is True.
    use_gui : bool, optional
        Enable GUI-mode behaviour. Default is False.
    device : torch.device or {'auto', 'mps', 'cuda', 'cpu'}, optional
        PyTorch computation device. ``'auto'`` selects CUDA/MPS if available.
        Default is 'auto'.
    log_level : str or int, optional
        Logging level for the sarcasm package, either a string ('DEBUG',
        'INFO', 'WARNING', 'ERROR', 'CRITICAL') or an integer (e.g.
        ``logging.DEBUG``). Default is 'INFO'.
    **info
        Additional user-supplied metadata key-value pairs
        (e.g. ``cell_line='wt'``).

    Attributes
    ----------
    file_path : str
        Absolute path to the input TIFF file.
    base_dir : str
        Base directory for all analysis artefacts of this TIFF.
    data_dir : str
        Sub-directory for intermediate data.
    analysis_dir : str
        Sub-directory for final analysis results.
    metadata : ImageMetadata
        Image metadata.
    device : torch.device
        PyTorch device on which computations are performed.
    zbands : np.ndarray
        Binary Z-band mask (loaded on demand from the OME-Zarr store).
    zbands_fast_movie : np.ndarray
        Binary Z-band mask for the high-temporal-resolution movie (loaded on demand).
    mbands : np.ndarray
        Binary M-band mask (loaded on demand).
    orientation : np.ndarray
        Sarcomere orientation map (loaded on demand).
    cell_mask : np.ndarray
        Binary cell mask (loaded on demand).
    sarcomere_mask : np.ndarray
        Binary sarcomere mask (loaded on demand).
    """

    def __init__(
            self,
            file_path: Union[str, os.PathLike],
            restart: bool = False,
            pixelsize: Union[float, None] = None,
            frametime: Union[float, None] = None,
            channel: Union[int, None] = None,
            axes: Union[str, None] = None,
            auto_save: bool = True,
            use_gui: bool = False,
            device: Union[torch.device, Literal['auto', 'mps', 'cuda', 'cpu']] = 'auto',
            log_level: Union[str, int] = 'INFO',
            **info: Dict[str, Any]
    ):
        # Guard the second positional argument before anything destructive runs.
        # Pre-1.0 the LOI workflow was entered as Motion(file_path, loi_name); that slot
        # now holds `restart`, and a non-empty LOI name is truthy -> the analysis store
        # would be deleted silently. Fail loudly instead of destroying data.
        if not isinstance(restart, (bool, int)):
            raise TypeError(
                f"{type(self).__name__}(...): 'restart' must be a bool, got "
                f"{type(restart).__name__} {restart!r}. The second positional argument is "
                f"'restart', and restart=True DELETES the existing analysis store. "
                f"The pre-1.0 form Motion(file_path, loi_name) was removed together with the "
                f"manual-LOI workflow — use SarcAsM.get_track_motion(group) to obtain a "
                f"Motion object for a tracked myofibril."
            )

        # Convert file_path to absolute path (as a string)
        self.file_path = os.path.abspath(str(file_path))
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        # Configuration
        self.auto_save = auto_save
        self.use_gui = use_gui
        self.restart = restart
        self.info = info
        
        # Configure logging for the sarcasm package
        self._setup_logging(log_level)

        # Directory structure: use the filename without extension as the base directory.
        # Strip compound suffixes so a '<name>.ome.zarr' input reuses the same '<name>/'
        # scratch dir as the original '<name>.tif' (not a spurious '<name>.ome/').
        _fname = os.path.basename(self.file_path)
        for _suf in ('.ome.zarr', '.zarr', '.ome.tif', '.ome.tiff', '.tif', '.tiff'):
            if _fname.lower().endswith(_suf):
                base_name = os.path.join(os.path.dirname(self.file_path), _fname[:-len(_suf)])
                break
        else:
            base_name = os.path.splitext(self.file_path)[0]
        self.base_dir = base_name + '/'  # This is a directory path as a string.
        self.data_dir = os.path.join(self.base_dir, "data/")
        self.analysis_dir = os.path.join(self.base_dir, "analysis/")

        # Handle restart: if restart is True and a legacy base_dir exists, remove it
        if restart and os.path.exists(self.base_dir):
            remove_tree(self.base_dir)

        # NB: base_dir/data_dir/analysis_dir are the pre-1.0 layout and are no
        # longer created here — merely constructing an object must not spawn an
        # empty '<name>/' tree. The few remaining consumers (legacy LOI data,
        # export_json, open_base_dir) create their target directory on demand.
        # In >=1.0 all artefacts live in the sibling '<name>.ome.zarr' store,
        # which IS created eagerly at construction (metadata only — the raw image
        # pixels are still ingested lazily on first read; see end of __init__).

        # --- single-store backing: everything lives in <name>.ome.zarr ---
        self.store_path = store_path_for(self.file_path)
        # Pre-1.0 (base_dir/structure.json) analyses are not read by >=1.0. Don't fail:
        # warn and start fresh in the new .ome.zarr store (the old analysis is left in place).
        legacy = detect_legacy_layout(self.file_path)
        if legacy is not None and not os.path.exists(self.store_path) and not self.restart:
            logger.warning(legacy_layout_message(legacy))
        if self.restart and os.path.exists(self.store_path):
            remove_tree(self.store_path)
        self.store = OmeZarrStore(self.store_path)

        # Initialize metadata
        self.metadata = ImageMetadata(
            file_name=os.path.basename(self.file_path),
            file_path=self.file_path,
            pixelsize = pixelsize,
            frametime = frametime,
            channel = channel,
            axes = axes,
        )

        # Load existing metadata from the store, else carry over the calibration from a
        # legacy analysis (results are not migrated, only pixelsize/frametime/axes/channel),
        # else harvest from the source image.
        self.meta_file = Path(self.data_dir) / "metadata.json"  # legacy metadata.json
        stored_meta = self.store.read_metadata() if self.store.exists else None
        if stored_meta and not self.restart:
            try:
                self.metadata = ImageMetadata.from_dict(stored_meta)
            except Exception as e:
                logger.error(f"Loading metadata from store failed: {e}")
                if not self.use_gui:
                    raise MetaDataError("Loading metadata from store failed.") from e
        elif legacy is not None and not self.restart and self.meta_file.exists():
            try:
                self.metadata = ImageMetadata.load_from_file(self.meta_file)
                logger.info("Carried over calibration metadata from the legacy analysis.")
            except Exception as e:
                logger.warning(f"Could not read legacy metadata ({e}); harvesting from the image.")
                if str(self.file_path).lower().endswith((".tif", ".tiff")):
                    self._extract_metadata_only(axes=axes)
        elif str(self.file_path).lower().endswith((".tif", ".tiff")):
            # Extract metadata without loading full image data (fast, even for large files on HDD).
            # Honour an explicit axes argument (e.g. 'TYX') so stacks aren't misread as channels.
            self._extract_metadata_only(axes=axes)

        # Create the sibling '<name>.ome.zarr' store eagerly so it exists (and is
        # inspectable) right after construction. Only the small metadata is written
        # now; the raw image pixels are still ingested lazily on the first
        # read_imgs()/analysis, keeping construction fast for large files.
        if not self.store.exists:
            try:
                self.store.write_metadata(self._metadata_jsonable())
            except Exception as e:
                logger.warning(f"Could not create the .ome.zarr store at construction: {e}")

        # Dictionary of models
        self.model_dir = Utils.get_models_dir()

        # Device configuration: auto-detect or validate provided device
        if device == "auto":
            self.device = Utils.get_device()
        else:
            if isinstance(device, str):
                try:
                    self.device = torch.device(device)
                except RuntimeError as e:
                    logger.error(f"Invalid device string: {device}")
                    raise ValueError(f"Invalid device string: {device}") from e
            elif isinstance(device, torch.device):
                self.device = device
            else:
                raise ValueError(
                    f"Invalid device type {type(device)}. "
                    "Expected torch.device instance or valid device string "
                    f"(e.g., 'cuda', 'cpu', 'mps')"
                )

    def _setup_logging(self, log_level: Union[str, int]) -> None:
        """
        Configure logging for the sarcasm package and all its submodules.

        Sets up a console handler for the 'sarcasm' logger. An existing GUI
        handler (e.g. ``QTextEditHandler``) is preserved.

        Parameters
        ----------
        log_level : str or int
            Logging level, either a string ('DEBUG', 'INFO', 'WARNING',
            'ERROR', 'CRITICAL') or an integer (e.g. ``logging.DEBUG``).

        Examples
        --------
        >>> sarc = SarcAsM(file_path, log_level='DEBUG')  # Verbose output
        >>> sarc = SarcAsM(file_path, log_level=logging.WARNING)  # Only warnings and errors
        """
        # Convert string to logging level if necessary
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure root logger for sarcasm package
        root_logger = logging.getLogger('sarcasm')
        root_logger.setLevel(log_level)
        
        # Remove only StreamHandlers to avoid duplicates, but preserve other handlers (e.g., GUI handlers)
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not hasattr(handler, 'signal_emitter'):
                root_logger.removeHandler(handler)
        
        # Only add console handler if not running in GUI mode (use_gui attribute)
        if not getattr(self, 'use_gui', False):
            # Create console handler with formatting
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            # Add handler to logger
            root_logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        root_logger.propagate = False

    # Mask attributes served lazily from the OME-Zarr store.
    _STORE_MASKS = (
        'zbands', 'zbands_fast_movie', 'mbands',
        'orientation', 'cell_mask', 'sarcomere_mask',
    )

    def __getattr__(self, name: str) -> Any:
        """Dynamic loading of the image / masks from the OME-Zarr store."""
        if name == 'image':
            return self.read_imgs()

        store = self.__dict__.get('store')
        if store is None:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

        if name in type(self)._STORE_MASKS:
            if not store.has_mask(name):
                raise FileNotFoundError(
                    f"Required analysis mask '{name}' not found in the store.\n"
                    f"Run 'detect_sarcomeres' to create it.")
            return store.read_mask(name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def _mask_exists(self, name: str) -> bool:
        """True if mask ``name`` is present in the store."""
        store = self.__dict__.get('store')
        return store is not None and store.has_mask(name)

    @staticmethod
    def _normalize_frames(frames):
        """Accept any sequence of frame indices, not just a list.

        ``frames=range(0, 400)`` is the natural way to write a contiguous window,
        and a tuple or an array is no less reasonable, while everything
        downstream wants a plain list of ints. ``'all'``, ``None`` and a single
        integer pass through untouched -- the callers tell those cases apart
        themselves.
        """
        if isinstance(frames, np.ndarray):
            return [int(f) for f in frames.ravel()]
        if isinstance(frames, (list, tuple, range)):
            return [int(f) for f in frames]
        return frames

    def _read_mask(self, name: str, frames=None) -> np.ndarray:
        """Read a mask from the store, optionally selecting ``frames`` (int, slice or
        any sequence of ints), mirroring ``tifffile.imread(..., key=frames)``."""
        if not self._mask_exists(name):
            raise FileNotFoundError(
                f"Required analysis mask '{name}' not found in the store.\n"
                f"Run 'detect_sarcomeres' to create it.")
        if frames is None:
            return self.store.read_mask(name)
        # When only a single frame was detected (or the movie is single-frame),
        # masks are stored without a leading frame axis, so a scalar frame index
        # refers to the whole stored mask, not a pixel row (mirrors read_imgs).
        data = getattr(self, 'data', None)
        detected = data.get('params.detect_sarcomeres.frames', 'all') if data is not None else 'all'
        single_frame_store = (
            self.metadata.n_stack in (None, 1)
            or (isinstance(detected, (list, tuple, np.ndarray)) and len(detected) == 1))
        if single_frame_store and isinstance(frames, (int, np.integer)):
            return self.store.read_mask(name)
        frames = self._normalize_frames(frames)
        # Slice inside the store so only the requested chunks (one frame each)
        # load, instead of materialising the whole stack and slicing in numpy.
        return self.store.read_mask(name, frames=frames)

    def __dir__(self) -> list[str]:
        """Augment autocomplete with dynamic attributes"""
        standard_attrs = list(super().__dir__())
        dynamic_attrs = [
            'zbands', 'zbands_fast_movie', 'mbands',
            'orientation', 'cell_mask', 'sarcomere_mask'
        ]
        return sorted(set(standard_attrs + dynamic_attrs))

    def __str__(self):
        """Returns a pretty, concise string representation of the SarcAsM object."""
        summary = [
            "╔══════════════════════════════════════════════════════",
            f"║ SarcAsM Analysis v{self.metadata.version}",
            "║ ─────────────────────────────────────────────────────",
            f"║ File path: {os.path.basename(self.file_path)}",
            f"║ Base directory: {os.path.dirname(self.base_dir)}",
            f"║ Device: {self.device}",
            f"║ Pixel size: {round(self.metadata.pixelsize, 5) if self.metadata.pixelsize is not None else 'N/A'} µm",
            f"║ Analysis timestamp: {self.metadata.timestamp_analysis}",
            "╚══════════════════════════════════════════════════════"
        ]

        return "\n".join(summary)

    def open_base_dir(self):
        """Open the folder holding this file's analysis in the file explorer.

        In >=1.0 the analysis lives in the sibling ``<name>.ome.zarr`` store, so
        this reveals the directory that contains the input image and its store
        (the legacy ``base_dir`` is no longer created).
        """
        Utils.open_folder(os.path.dirname(self.store_path))

    def _metadata_jsonable(self) -> dict:
        """Metadata as a JSON/attr-safe dict (numpy time array -> list)."""
        meta = self.metadata.to_dict()
        if isinstance(meta.get('time'), np.ndarray):
            meta['time'] = meta['time'].tolist()
        return meta

    def save_metadata(self):
        """Persist the current metadata into the OME-Zarr store (when it exists)."""
        store = self.__dict__.get('store')
        if store is None or not store.exists:
            return  # store is created on first image ingest / analysis write
        try:
            store.write_metadata(self._metadata_jsonable())
        except Exception as e:
            logger.debug(f"metadata save to store skipped: {e}")

    def _extract_metadata_only(self, axes: Union[str, None] = None) -> None:
        """
        Extract metadata from the TIFF file without loading the full image data.

        Optimized for large files (e.g. 15+ GB) on slow storage (HDD): only the
        file headers and metadata are read, not the pixel data.

        Parameters
        ----------
        axes : str or None, optional
            Dimension order override (e.g. 'TXYC'). Auto-detected if None.
            Default is None.
        """
        with tifffile.TiffFile(self.file_path) as tif:
            series = tif.series[0]

            # Determine axes order from metadata only (no data loading)
            if axes is None:
                axes = self._determine_axes(series, tif)
            else:
                axes = axes.upper()

            self._validate_axes(str(axes))

            # Extract metadata using file headers only
            self._harvest_metadata(series, tif, axes)

            # Determine shape for internal format (after channel selection)
            # We need to compute what the shape would be after processing
            shape_orig = series.shape
            processed_axes = axes

            # Account for channel selection (removes C axis)
            if 'C' in axes:
                c_axis = axes.index('C')
                n_chan = shape_orig[c_axis]
                # Shape after channel selection
                shape_after_channel = list(shape_orig)
                del shape_after_channel[c_axis]
                shape_after_channel = tuple(shape_after_channel)
                processed_axes = axes.replace('C', '')
            else:
                shape_after_channel = shape_orig

            # Compute final shape after permutation to internal format (Stack, Y, X) or (Y, X)
            stack_axis = 'T' if 'T' in processed_axes else ('Z' if 'Z' in processed_axes else None)
            target_axes = []
            if stack_axis:
                target_axes.append(stack_axis)
            if 'Y' in processed_axes:
                target_axes.append('Y')
            if 'X' in processed_axes:
                target_axes.append('X')

            # Build the final shape
            perm = [processed_axes.index(ax) for ax in target_axes]
            final_shape = tuple(shape_after_channel[i] for i in perm)

            # Squeeze singleton dimensions
            final_shape = tuple(d for d in final_shape if d > 1) or final_shape

            # Update metadata with computed shape
            self.metadata.shape = final_shape
            self.metadata.size = (final_shape[-2], final_shape[-1]) if len(final_shape) >= 2 else None

            # Save metadata
            if self.auto_save:
                self.save_metadata()

    def _internal_axes(self, ndim: int) -> str:
        """OME axes string for the internal (channel-selected, permuted) image."""
        if ndim <= 2:
            return "yx"
        stack = "t" if (self.metadata.axes and "T" in self.metadata.axes.upper()) else "z"
        return stack + "yx"

    def read_imgs(self, frames=None, axes=None):
        """
        Load image data from the OME-Zarr store (ingesting the source TIFF on first use).

        Parameters
        ----------
        frames : int, slice, sequence of int, or None, optional
            Frame selection for stacks; any sequence works, including
            ``range(0, 400)``. None loads all frames. Default is None.
        axes : str or None, optional
            Dimension order override (e.g. 'TXYC'). Auto-detected if None.
            Default is None.

        Returns
        -------
        np.ndarray
            Image data in internal format ``(Y, X)`` or ``(Stack, Y, X)``.
        """
        # Fast path: pixels already ingested into the store -> lazy zarr slice.
        if self.store.has_image():
            arr = self.store.image_handle()
            if (frames is None or (isinstance(frames, str) and frames == 'all')
                    or self.metadata.n_stack is None or self.metadata.n_stack <= 1):
                return arr[...]
            return arr[self._normalize_frames(frames)]

        # First open of a TIFF: read it, ingest into the store, then slice.
        data = self._read_source_tif(axes=axes)
        if (frames is None or (isinstance(frames, str) and frames == 'all')
                or self.metadata.n_stack is None or self.metadata.n_stack <= 1):
            return data
        return data[self._normalize_frames(frames)]

    def _read_source_tif(self, axes=None):
        """Read the full source TIFF into internal format and ingest it into the store."""
        with tifffile.TiffFile(self.file_path) as tif:
            series = tif.series[0]
            raw_data = series.asarray()

            # Determine axes: explicit arg > axes recorded at init (metadata) > auto-detect.
            if axes is None:
                axes = self.metadata.axes or self._determine_axes(series, tif)
            else:
                axes = axes.upper()

            self._validate_axes(str(axes))

            # Store original input axes in metadata before any processing
            original_axes = axes

            # Process data: select channel and update axes accordingly
            raw_data, processed_axes = self._select_channel(raw_data, axes)

            # Extract metadata using original axes order
            meta = self._harvest_metadata(series, tif, original_axes)
            self.__metadata_obj = meta  # cache for outsiders

            # Normalize to internal format (Stack, Y, X) or (Y, X)
            data = self._permute_to_internal(raw_data, processed_axes)
            data = data.squeeze()
            self.metadata.shape = data.shape  # shape after all processing
            self.metadata.size = (data.shape[-2], data.shape[-1]) if data.ndim >= 2 else None  # (height, width)

        # Ingest the pixels into the OME-Zarr store (copy-in) + persist metadata.
        self.store.ingest_image(
            data, axes=self._internal_axes(data.ndim),
            pixelsize=self.metadata.pixelsize, frametime=self.metadata.frametime,
            metadata=self._metadata_jsonable())
        return data

    @staticmethod
    def _determine_axes(series, tif: tifffile.TiffFile) -> str:
        """
        Return an upper-case axis string such as 'TCZYX', 'YXC', 'YX', …

        Parameters
        ----------
        series : tifffile.TiffPageSeries
            Image series whose axis order is inferred.
        tif : tifffile.TiffFile
            Open TIFF file, used to read OME/ImageJ metadata.

        Returns
        -------
        str
            Upper-case axis string.

        Raises
        ------
        ValueError
            If no reasonable guess is possible and the caller must supply
            the order manually.
        """
        # OME-TIFF
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                image_elem = root.find('.//{*}Image')
                if image_elem is None:
                    raise ValueError("OME Image element not found")
                detected = image_elem.attrib['DimensionOrder'].upper()
                logger.debug(f"Detected axes from OME: '{detected}'")
                
                # Validate that detected axes match actual data dimensions
                if len(detected) != len(series.shape):
                    logger.warning(f"OME axes '{detected}' has {len(detected)} chars but data has {len(series.shape)} dims")
                    # OME metadata is usually reliable, but verify
                    # Fall through to next strategy if mismatch
                    raise ValueError("OME axes length mismatch")
                
                return detected
            except Exception as e:
                logger.debug(f"OME detection failed: {e}")
                pass  # fall through to next strategy

        # ImageJ hyper-stack
        if tif.imagej_metadata:
            ij = tif.imagej_metadata
            order = ''
            if ij.get('frames', 1) > 1:
                order += 'T'
            if ij.get('slices', 1) > 1:
                order += 'Z'
            if ij.get('channels', 1) > 1:
                order += 'C'
            order += 'YX'
            
            # BUG FIX: ImageJ metadata might say channels=1 or slices=1, but the actual
            # data could still have singleton dimensions for these axes.
            # We need to verify the axes match the actual data shape.
            expected_ndim = len(order)
            actual_ndim = len(series.shape)
            
            if actual_ndim > expected_ndim:
                # Data has more dimensions than expected from metadata
                # This often means there's a singleton channel or Z dimension
                missing_dims = actual_ndim - expected_ndim
                logger.debug(f"ImageJ axes '{order}' has {expected_ndim} dims, but data has {actual_ndim} dims")
                logger.debug(f"Adding {missing_dims} missing dimension(s)")
                
                # Add missing dimensions in standard order: T, Z, C before YX
                if 'C' not in order and missing_dims > 0:
                    # Insert C before YX
                    order = order.replace('YX', 'CYX')
                    missing_dims -= 1
                    logger.debug("Added 'C' dimension")
                
                if 'Z' not in order and missing_dims > 0:
                    # Insert Z before YX (but after T if present)
                    if 'T' in order:
                        order = order.replace('YX', 'ZYX')
                    else:
                        order = 'Z' + order
                    missing_dims -= 1
                    logger.debug("Added 'Z' dimension")
                
                if missing_dims > 0:
                    # Still have extra dims - this is unusual
                    logger.warning(f"Still have {missing_dims} unaccounted dimensions!")
                    logger.debug("Falling through to next detection method")
                    # Don't return, fall through to tifffile's guess
                else:
                    logger.debug(f"Final ImageJ axes: '{order}'")
                    return order
            else:
                return order

        # tifffile's own guess
        if series.axes:
            axes = series.axes.upper().replace('S', 'C')  # S (samples) → C
            # tifffile labels a stack axis it cannot classify as 'I' (sequence)
            # or 'Q' (other) — e.g. a generic multi-page TIFF, or an OME/ImageJ
            # file whose metadata it could not fully resolve. Treat that extra
            # axis as time (a movie), consistent with the bare-stack heuristic
            # below; a genuine z-stack can still be forced via axes='ZYX'.
            axes = axes.replace('I', 'T').replace('Q', 'T')
            return axes

        # heuristics on raw shape
        shape = series.shape
        if len(shape) == 2:  # (Y, X)
            return 'YX'
        if len(shape) == 3 and shape[-1] <= 10:  # (Y, X, C)  small C
            return 'YXC'
        if len(shape) == 3 and shape[-1] > 10:
            return 'TXY'

        raise ValueError(
            f"Could not determine axis order for shape {shape}. "
            "Please specify it explicitly (e.g. axes='TXYC')."
        )

    def _select_channel(self,
                        data: np.ndarray,
                        axes: str) -> tuple[np.ndarray, str]:
        """
        Isolate the channel requested by ``self.metadata.channel`` and remove
        the channel axis from the array.

        Parameters
        ----------
        data : np.ndarray
            Array as read from disk (still in source order).
        axes : str
            Corresponding axis string (upper-case, e.g. ``'TYXC'``).

        Returns
        -------
        data_sel : np.ndarray
            Array with the channel axis removed.
        axes_sel : str
            Axis string without the ``'C'`` character.

        Raises
        ------
        ValueError
            If the requested channel index is out of range, or if
            ``self.metadata.channel`` is given but the image has no ``C`` axis.
        """
        # file actually contains a channel axis
        if 'C' in axes:
            c_axis = axes.index('C')
            n_chan = data.shape[c_axis]

            # choose channel index
            if n_chan == 1:
                chan_idx = 0  # trivial
            else:
                if self.metadata.channel is None:
                    logger.info(
                        f"Multi-channel image detected (n={n_chan}). "
                        f"Using channel 0 by default. "
                        f"Pass SarcAsM(..., channel=<int>) to override."
                    )
                    chan_idx = 0
                else:
                    chan_idx = int(self.metadata.channel)
                    if not (0 <= chan_idx < n_chan):
                        raise ValueError(
                            f"Channel {chan_idx} requested but only "
                            f"{n_chan} channel(s) available."
                        )

            # extract and drop the C-axis
            data = np.take(data, chan_idx, axis=c_axis)
            axes = axes.replace('C', '')  # update axis string
            self.metadata.channel = chan_idx

        # file has NO channel axis
        elif self.metadata.channel is not None:
            message = "Parameter 'channel' was supplied but the image contains no channel dimension."
            if not self.use_gui:
                raise ValueError(message)
            else:
                logger.warning(message)

        else:
            self.metadata.channel = None

        return data, axes

    def _harvest_metadata(self, series, tif, axes) -> ImageMetadata:
        """
        Collect metadata from the TIFF and update the instance metadata object.

        Parameters
        ----------
        series : tifffile.TiffPageSeries
            Image series providing the data shape.
        tif : tifffile.TiffFile
            Open TIFF file, used to read OME/ImageJ/resolution tags.
        axes : str
            Upper-case axis string for the source data.

        Returns
        -------
        ImageMetadata
            The updated instance metadata object.
        """

        # pixel size
        px = None
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                px_elem = root.find('.//{*}Pixels')
                if px_elem is not None:
                    px = px_elem.get('PhysicalSizeX')
                    px = float(px) if px else None
            except Exception as e:
                logger.debug(f"Failed to extract pixel size from OME metadata: {e}")
                pass

        if px is None and tif.imagej_metadata:
            ij = tif.imagej_metadata
            px = ij.get('pixel_width') or ij.get('PixelWidth')
            try:
                px = float(px) if px is not None else None
            except (TypeError, ValueError) as e:
                logger.debug(f"Failed to convert pixel size to float: {e}")
                pass

        if px is None:
            # fall back to TIFF XResolution / ResolutionUnit
            page = tif.pages[0]
            if 'XResolution' in page.tags and 'ResolutionUnit' in page.tags:
                try:
                    num, den = page.tags['XResolution'].value
                    unit = page.tags['ResolutionUnit'].value  # 2=inches, 3=cm
                    dpi = num / den
                    if dpi > 0:
                        # convert – inch: 25 400 µm ; centimetre: 10 000 µm
                        if unit == 2:
                            px = 25_400 / dpi
                        elif unit == 3:
                            px = 10_000 / dpi
                        else:
                            px = 1 / dpi
                except Exception as e:
                    logger.debug(f"Failed to extract pixel size from TIFF resolution tags: {e}")
                    pass

        # frame time & timestamps
        ft, ts = None, None
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                deltas = [float(p.get('DeltaT')) for p in
                          root.findall('.//{*}Plane') if p.get('DeltaT')]
                if deltas:
                    ts = deltas
                    ft = float(np.diff(deltas).mean()) if len(deltas) > 1 else deltas[0]
            except Exception as e:
                logger.debug(f"Failed to extract frame time from OME metadata: {e}")
                pass

        if ft is None and tif.imagej_metadata:
            ij = tif.imagej_metadata
            ft = ij.get('finterval') or ij.get('Frame interval')
            if ft is None and (fps := ij.get('fps')):
                try:
                    ft = 1 / float(fps)
                except (ValueError, ZeroDivisionError) as e:
                    logger.debug(f"Failed to compute frame time from fps: {e}")
                    pass

            if ts is None:
                ts = ij.get('timestamps')
                if isinstance(ts, str):
                    try:
                        ts = json.loads(ts)
                    except Exception as e:
                        logger.debug(f"Failed to parse timestamps from ImageJ metadata: {e}")
                        pass

        # Convert to proper types
        ft = float(ft) if ft else None

        # Apply overrides - user values take precedence when provided
        self.metadata.pixelsize = self.metadata.pixelsize if self.metadata.pixelsize is not None else (float(px) if px is not None else None)
        self.metadata.frametime = self.metadata.frametime if self.metadata.frametime is not None else ft

        # Calculate stack length
        stack_len = 1  # for single image
        if 'T' in axes:
            stack_len = series.shape[axes.index('T')]
        elif 'Z' in axes:
            stack_len = series.shape[axes.index('Z')]

        # Validation checks
        if self.metadata.pixelsize is None and not self.use_gui:
            raise MetaDataError(
                f"Pixel size could not be extracted from {self.file_path}. "
                f"Please enter manually (e.g., SarcAsM(file_path, pixelsize=0.1))."
            )

        if self.metadata.pixelsize and not (0.01 <= self.metadata.pixelsize <= 0.5):
            message = (f"Pixel size {self.metadata.pixelsize} µm is outside reasonable range (0.01-0.5 µm). "
                       f"Please check your input or file metadata.")
            if not self.use_gui:
                raise MetaDataError(message)
            else:
                logger.warning(message)

        if self.metadata.frametime is None and stack_len > 1:
            logger.warning('Frametime could not be extracted from tif file. '
                  'Please enter manually if needed (e.g., SarcAsM(file, frametime=0.1)).')

        # Update the existing metadata object with extracted values
        self.metadata.axes = axes
        self.metadata.shape_orig = tuple(series.shape)
        self.metadata.n_stack = int(stack_len)
        self.metadata.timestamps = ts
        self.metadata.channel = self.metadata.channel

        # Create time array if we have both frametime and a stack
        if self.metadata.frametime and self.metadata.n_stack > 1:
            self.metadata.time = np.arange(0, self.metadata.n_stack *
                                  self.metadata.frametime,
                                  self.metadata.frametime)
        else:
            self.metadata.time = None

        # Add user info
        self.metadata.add_user_info(**self.info)

        # Persist metadata into the store (no-op until the store exists)
        if self.auto_save:
            self.save_metadata()

        return self.metadata

    @staticmethod
    def _validate_axes(axes: str) -> None:
        """
        Raise if ``axes`` is not a unique subset of {X, Y, T, C, Z}.

        Parameters
        ----------
        axes : str
            Axis string to validate.
        """
        allowed = set("XYTCZ")
        illegal = set(axes) - allowed
        if illegal:
            raise ValueError(
                f"Invalid axis letter(s): {''.join(sorted(illegal))}. "
                f"Only {''.join(sorted(allowed))} are permitted."
            )
        if len(axes) != len(set(axes)):
            dup = ''.join(sorted({c for c in axes if axes.count(c) > 1}))
            raise ValueError(
                f"Duplicate axis letter(s): {dup}. "
                "Each axis may appear at most once."
            )


    @staticmethod
    def _permute_to_internal(data: np.ndarray, source_axes: str) -> np.ndarray:
        """
        Permute image data to the internal axis order.

        Parameters
        ----------
        data : np.ndarray
            The image data as stored on disk.
        source_axes : str
            Axis string returned by :meth:`_determine_axes`.

        Returns
        -------
        np.ndarray
            Array permuted to ``(Stack, Y, X)`` or ``(Y, X)``.
        """
        # Decide which dimension, if any, is treated as the stack
        stack_axis = 'T' if 'T' in source_axes else ('Z' if 'Z' in source_axes else None)

        target_axes: List[str] = []
        if stack_axis:
            target_axes.append(stack_axis)
        if 'Y' in source_axes:
            target_axes.append('Y')
        if 'X' in source_axes:
            target_axes.append('X')

        # Build the permutation list
        perm = [source_axes.index(ax) for ax in target_axes]
        
        # Validate permutation matches array dimensions
        if perm and len(perm) != data.ndim:
            raise ValueError(
                f"Permutation mismatch: data has {data.ndim} dimensions (shape={data.shape}), "
                f"but permutation list has {len(perm)} elements (perm={perm}).\n"
                f"Source axes: '{source_axes}', Target axes: {target_axes}\n"
                f"This typically occurs when the axes string doesn't match the actual data shape. "
                f"Please verify the image file format or specify axes explicitly."
            )
        
        if perm:
            data = data.transpose(perm)

        return data

    def remove_intermediate_masks(self, masks=None) -> None:
        """Delete derived masks from the store to free disk space.

        All removed masks are regenerable by re-running ``detect_sarcomeres``;
        the raw image and the analysis results are kept.

        Parameters
        ----------
        masks : sequence of str or None, optional
            Mask names to remove. None (default) removes all derived masks
            (:attr:`_STORE_MASKS`).
        """
        names = self._STORE_MASKS if masks is None else masks
        for name in names:
            self.store.delete_mask(name)
