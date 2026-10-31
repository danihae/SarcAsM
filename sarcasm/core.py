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

"""The :class:`SarcAsM` recording: file, calibration, the OME-Zarr store and sarcomere detection."""

import functools
import json
import logging
import math
import os
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile
import torch
from bio_image_unet.progress import ProgressNotifier

from sarcasm.exceptions import MetaDataError
from sarcasm._internal.gpu_watch import watch_metal_errors
from sarcasm.io.results_store import Results
from sarcasm.analysis import detection
from sarcasm._internal.frames import _as_int, canonical_frame, canonical_frames
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

if TYPE_CHECKING:
    from sarcasm.motion import MotionAnalysis
    from sarcasm.structure import StructureAnalysis


def _format_bytes(n: int) -> str:
    """Render a byte count in the largest unit that keeps it readable."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _is_ome_zarr(path: str) -> bool:
    """Does ``path`` look like an OME-Zarr (NGFF) image group?

    Accepts a ``*.zarr`` / ``*.ome.zarr`` suffix, or any directory already carrying
    NGFF v0.4 (``.zattrs``) or v0.5 (``zarr.json``) metadata.
    """
    p = Path(path)
    name = p.name.lower()
    if name.endswith('.ome.zarr') or name.endswith('.zarr'):
        return True
    return p.is_dir() and ((p / '.zattrs').exists() or (p / 'zarr.json').exists())


class SarcAsM:
    """
    One recording: image, calibration, analysis store and sarcomere detection.

    The analyses are reached through two namespaces: ``sarc.structure`` for sarcomere
    morphology (:class:`~sarcasm.structure.StructureAnalysis`) and ``sarc.motion`` for
    sarcomere motion from tracked sarcomere vectors (:class:`~sarcasm.motion.MotionAnalysis`).

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to the image TIFF file (or OME-Zarr).
    restart : bool, optional
        If True, delete the analysis store and start fresh. Default is False.
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
        Explicit dimension order (e.g. ``'TXYC'``). None auto-detects the order.
        Default is None.
    auto_save : bool, optional
        Write analysis results to disk automatically. Default is True.
    use_gui : bool, optional
        Activate GUI mode. Default is False.
    device : torch.device or {'auto', 'mps', 'cuda', 'cpu'}, optional
        Device on which PyTorch kernels are executed. ``'auto'`` selects CUDA
        or MPS when available. Default is ``'auto'``.
    log_level : str or int, optional
        Logging level of the sarcasm package ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' or an
        int such as ``logging.DEBUG``). Default is 'INFO'.
    **info
        Additional key-value pairs stored in the metadata file.

    Attributes
    ----------
    data : Results
        Lazy, Zarr-backed store of morphology, tracking and motion results
        (populated after running the respective routines). A key is its path, so
        ``sarc.data['motion.tracks.slen']`` and ``sarc.data.motion.tracks.slen``
        are the same value. ``structure.*`` is per-frame morphology, ``motion.*``
        is everything derived from the tracks, ``params.*`` records what each
        step ran with. ``print(sarc.data)`` gives an overview,
        ``sarc.data.find('slen')`` searches, ``sarc.data.describe(key)``
        explains one. Write with ``sarc.data['key'] = value`` (attributes are
        read-only) followed by :meth:`commit`.
    structure : StructureAnalysis
        Sarcomere morphology: Z-bands, sarcomere vectors, myofibrils, domains.
    motion : MotionAnalysis
        Sarcomere motion: tracking, grouping, contraction analysis.
    metadata : ImageMetadata
        Image metadata (calibration, shape, axes).
    zbands, zbands_fast_movie, mbands, orientation, cell_mask, sarcomere_mask : np.ndarray
        The detection masks, read from the store on access (see :meth:`read_mask`).
    """

    def __init__(
            self,
            file_path: Union[str, os.PathLike],
            *,
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
        # Convert file_path to absolute path (as a string)
        self.file_path = os.path.abspath(str(file_path))
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
        # A Git LFS pointer is a ~130-byte text stub in place of the image; without this
        # check tifffile fails later with an unrelated-looking "not a TIFF file" error.
        if os.path.isfile(self.file_path):
            with open(self.file_path, 'rb') as fh:
                if fh.read(64).startswith(b'version https://git-lfs'):
                    raise ValueError(
                        f"{self.file_path} is a Git LFS pointer, not an image. "
                        f"Fetch the real file with 'git lfs pull'.")

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

        # nothing is written to base_dir any more; everything lives in the sibling <name>.ome.zarr store
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
            self._check_explicit_calibration(pixelsize=pixelsize, frametime=frametime,
                                             channel=channel, axes=axes)
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
        elif _is_ome_zarr(self.file_path):
            self._extract_metadata_only_ngff(axes=axes)   # calibration from the multiscales metadata

        # create the store (metadata only) now; pixels are ingested on first read
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

        # bind the lazy results store (the sarcasm/ group of the OME-Zarr store)
        self._load_structure_data()

    def _check_explicit_calibration(self, **explicit) -> None:
        """Refuse constructor calibration that contradicts an existing store.

        The stored metadata wins on reopen because the stored pixels and results were
        computed with it, so a differing ``pixelsize=``/``frametime=``/``channel=``/``axes=``
        would otherwise be dropped without notice.
        """
        conflicts = []
        for name, value in explicit.items():
            if value is None:
                continue
            stored = getattr(self.metadata, name, None)
            if name == 'frametime' and stored is None:
                # nothing was computed with a frame time yet: fill it in
                self.metadata.frametime = float(value)
                if self.metadata.n_stack and self.metadata.n_stack > 1:
                    self.metadata.time = np.arange(self.metadata.n_stack) * self.metadata.frametime
                self.save_metadata()
                continue
            if isinstance(value, str) and isinstance(stored, str):
                same = value.upper() == stored.upper()
            elif isinstance(value, (int, float)) and isinstance(stored, (int, float)):
                same = math.isclose(float(value), float(stored), rel_tol=1e-6)
            else:
                same = stored is None and value is None
            if not same:
                conflicts.append(f"{name}={value!r} (store: {stored!r})")
        if not conflicts:
            return
        message = (f"{', '.join(conflicts)}: the existing analysis in {self.store_path} was made "
                   f"with different calibration. Pass restart=True to re-analyze with the new "
                   f"values (this deletes the stored results), or omit them to keep the stored ones.")
        if not self.use_gui:
            raise MetaDataError(message)
        logger.warning(message)

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

    def mask_exists(self, name: str) -> bool:
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

    def _canonical_frames(self, frames, name: str = 'frames'):
        """``'all'`` or the sorted unique frame numbers selected by ``frames``, validated against the movie.

        The single entry point of every analysis step: ``k``, ``np.int64(k)``, ``[k]``, ``(k,)``,
        ``range(k, k + 1)``, ``np.array([k])`` and ``np.array(k)`` all give ``[k]``; None gives ``'all'``.
        See :func:`sarcasm._internal.frames.canonical_frames`.
        """
        return canonical_frames(frames, self.metadata.n_stack, name=name)

    def _canonical_frame(self, frame, name: str = 'frame') -> int:
        """One validated frame number (int, numpy integer, 0-d array or one-element sequence)."""
        return canonical_frame(frame, self.metadata.n_stack, name=name)

    def _detected_frames(self) -> Optional[List[int]]:
        """Frame numbers of the pages of the detection masks, in page order; None when detection params are
        missing (masks then cover every frame)."""
        data = getattr(self, 'data', None)
        detected = data.get('params.detect_sarcomeres.frames', 'all') if data is not None else 'all'
        if detected is None or isinstance(detected, str):
            return None
        return [int(f) for f in np.atleast_1d(np.asarray(detected)).ravel()]

    # masks written by detect_sarcomeres: stored only for the detected frames, one page per frame
    _DETECTION_MASKS = ('cell_mask', 'zbands', 'mbands', 'orientation', 'sarcomere_mask')
    # number of axes of one frame of a mask; a stored array with one axis more holds a page axis
    _MASK_FRAME_NDIM = {'orientation': 3}
    # more channels than this without an explicit channel= is treated as a mislabelled T/Z axis
    _MAX_PLAUSIBLE_CHANNELS = 6

    def _warn_once(self, message: str) -> None:
        """Log a metadata warning once per object; the metadata is harvested again when
        the pixels are first read, which would otherwise repeat it."""
        seen = self.__dict__.setdefault('_warned', set())
        if message not in seen:
            seen.add(message)
            logger.warning(message)

    def _check_channel_count(self, n_chan: int, axes: str) -> None:
        """Fail on an implausible channel count when no channel was chosen explicitly."""
        if self.metadata.channel is not None or n_chan <= self._MAX_PLAUSIBLE_CHANNELS:
            return
        # Fluorescence stacks rarely carry more than a handful of channels; dozens
        # of "channels" almost always mean a time or Z axis mislabelled in the
        # file header, and analysing channel 0 would silently use one frame.
        message = (
            f"The image has {n_chan} channels according to its metadata (axes "
            f"'{axes}'), which usually means a time or Z axis was saved as channels. "
            f"Set the true axis order (e.g. TYX; Python: axes='TYX'), or choose the channel "
            f"if these really are channels.")
        if not self.use_gui:
            raise MetaDataError(message)
        self._warn_once(message)

    def read_mask(self, name: str, frames=None) -> np.ndarray:
        """Read a mask from the store, optionally selecting movie ``frames``, like ``read_imgs``.

        ``frames`` is a frame number (int, numpy integer, 0-d array) -> one frame, ``(H, W)`` (``(2, H, W)``
        for the orientation field); a sequence of frame numbers (any order, repeats allowed) or a slice -> one
        page per requested frame, ``(n, H, W)``. Frame numbers are mapped to the pages of a partially detected
        mask, whether the store holds the pages with a frame axis (``(n, H, W)``, also for a single detected
        frame) or a single frame without one (``(H, W)``, single images and stores written before the frame
        axis was kept). None returns the stored array unchanged.
        """
        if not self.mask_exists(name):
            raise FileNotFoundError(
                f"Required analysis mask '{name}' not found in the store.\n"
                f"Run 'detect_sarcomeres' to create it.")
        if frames is None:
            return self.store.read_mask(name)
        handle = self.store.mask_handle(name)
        paged = handle.ndim > self._MASK_FRAME_NDIM.get(name, 2)
        scalar = not isinstance(frames, (slice, list, tuple, range)) and np.ndim(frames) == 0
        if isinstance(frames, slice):
            requested = list(range(*frames.indices(int(self.metadata.n_stack or 1))))
        elif scalar:
            requested = [_as_int(frames)]
        else:
            requested = [_as_int(f) for f in np.asarray(list(frames) if isinstance(frames, range)
                                                            else frames, dtype=object).ravel()]
        if not requested:
            raise ValueError(f"No frames selected for mask '{name}'.")
        pages = self._mask_pages(name, requested, n_pages=handle.shape[0] if paged else 1)
        if not paged:                             # one stored frame, no frame axis
            single = np.asarray(handle[...])
            return single if scalar else np.stack([single] * len(pages))
        # Slice inside the store so only the requested chunks (one frame each)
        # load, instead of materialising the whole stack and slicing in numpy.
        return np.asarray(handle[pages[0]] if scalar else handle[pages])

    def _mask_pages(self, name: str, frames, n_pages: Optional[int] = None) -> List[int]:
        """Translate movie-frame numbers into page indices of a stored mask.

        ``detect_sarcomeres(frames=[32, 33])`` stores two pages; ``read_mask('zbands', frames=33)``
        must then read page 1. Masks covering every frame (all frames detected, the 3D fast-movie
        Z-bands, optical flow) have one page per frame.

        Raises
        ------
        ValueError
            For a frame that was not detected, naming the detected frames.
        """
        requested = [int(f) for f in frames]
        detected = self._detected_frames() if name in self._DETECTION_MASKS else None
        n = int(self.metadata.n_stack or 1)
        if detected is None or detected == list(range(n)):
            pages = requested
        else:
            missing = [f for f in requested if f not in detected]
            if missing:
                raise ValueError(f"Frame(s) {missing} of mask '{name}' were not detected (detected frames: "
                                 f"{detected}). Run detect_sarcomeres on these frames first.")
            pages = [detected.index(f) for f in requested]
        if n_pages is not None:
            bad = [f for f, p in zip(requested, pages) if not 0 <= p < n_pages]
            if bad:
                raise ValueError(f"Frame(s) {bad} out of range for mask '{name}', which holds {n_pages} "
                                 f"frame(s).")
        return pages

    def __dir__(self) -> list[str]:
        """Augment autocomplete with dynamic attributes"""
        standard_attrs = list(super().__dir__())
        dynamic_attrs = [
            'zbands', 'zbands_fast_movie', 'mbands',
            'orientation', 'cell_mask', 'sarcomere_mask'
        ]
        return sorted(set(standard_attrs + dynamic_attrs))

    def _file_summary_rows(self) -> list[tuple[str, str]]:
        """Label/value rows for :meth:`__str__`, cheap enough to print freely.

        Reads no pixels and materialises no result arrays — only in-memory
        metadata plus the store's own group listings.

        Returns
        -------
        list of (str, str)
            Rows rendered by :meth:`__str__`, in display order.
        """
        md = self.metadata
        rows = [("file", str(self.file_path))]

        store = self.__dict__.get('store')
        if store is None or not store.exists:
            rows.append(("store", f"{os.path.basename(self.store_path)}  ·  not created"))
        else:
            size = store.size_bytes()
            parts = [os.path.basename(self.store_path),
                     _format_bytes(size) if size is not None else "size not measured",
                     "image ingested" if store.has_image() else "image not yet ingested"]
            rows.append(("store", "  ·  ".join(parts)))

        if md.shape:
            shape = "×".join(str(d) for d in md.shape)
            extra = [f"axes {md.axes}" if md.axes else "",
                     f"{md.n_stack} frames" if md.n_stack else ""]
            rows.append(("image", "   ".join([shape] + [e for e in extra if e])))
        else:
            rows.append(("image", "—"))

        cal = [f"{md.pixelsize:.4g} µm/px" if md.pixelsize else "pixel size —"]
        if md.frametime:
            cal.append(f"{md.frametime:g} s/frame ({1 / md.frametime:.4g} fps)")
        cal.append(f"channel {md.channel}" if md.channel is not None else "channel —")
        rows.append(("calibration", "  ·  ".join(cal)))

        version = getattr(md, 'sarcasm_version', None)
        stamp = (md.timestamp_analysis or "")[:10]
        rows.append(("device", f"{self.device}  ·  SarcAsM {version or '?'}"
                               + (f", analysed {stamp}" if stamp else "")))

        masks = sorted(store.mask_names()) if (store is not None and store.exists) else []
        rows.append(("masks", ", ".join(masks) if masks else "—"))
        return rows

    def __str__(self) -> str:
        """A concise overview of the object: image, calibration, and progress."""
        rows = self._summary_rows()
        width = max(len(label) for label, _ in rows)
        indent = " " * (width + 4)
        head = f"{type(self).__name__} · {self.metadata.file_name or self.file_path}"
        body = []
        for label, value in rows:
            text = textwrap.fill(str(value), width=96, subsequent_indent=indent,
                                 break_long_words=False, break_on_hyphens=False)
            body.append(f"  {label:<{width}}  {text}")
        return "\n".join([head] + body)

    def __repr__(self) -> str:
        """One-line summary. Never raises — pytest calls it on assertion failure."""
        try:
            md = self.metadata
            shape = "×".join(str(d) for d in (md.shape or ())) or "?"
            bits = [f"<{type(self).__name__}", repr(md.file_name or self.file_path), shape]
            if md.pixelsize:
                bits.append(f"{md.pixelsize:.4g}µm/px")
            if md.frametime:
                bits.append(f"{md.frametime:g}s/frame")
            return " ".join(bits) + ">"
        except Exception:                       # partially constructed / patched objects
            return object.__repr__(self)

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

    # ------------------------------------------------------------------
    # OME-Zarr (NGFF) metadata
    # ------------------------------------------------------------------

    def _read_ngff_attrs(self):
        """Open the input as an NGFF image group and return what is needed to parse it.

        Returns ``(root, axes_str, level0_path, scale, units)``. ``scale`` already has
        any multiscales-level transform folded into the dataset-level one, and is in the
        axis order of ``axes_str``.
        """
        try:
            import zarr
        except ImportError as e:
            raise ImportError(
                "OME-Zarr input requires the 'zarr' package."
            ) from e

        root = zarr.open(self.file_path, mode='r')
        attrs = dict(root.attrs)
        # v0.5 nests the whole OME block under 'ome'; v0.4 puts it at the top level.
        ome = attrs.get('ome') if isinstance(attrs.get('ome'), dict) else attrs
        if 'multiscales' not in ome:
            if 'plate' in ome:
                raise MetaDataError(
                    f"{self.file_path} is an NGFF HCS plate root, not an image. Point "
                    "SarcAsM at a per-FOV image group inside the plate (e.g. .../A/1/0/)."
                )
            raise MetaDataError(
                f"No NGFF 'multiscales' metadata found at {self.file_path}. "
                "Is this a valid OME-Zarr image group?"
            )

        ms = ome['multiscales'][0]
        axes_meta = ms.get('axes') or []
        axes_str = ''.join(a.get('name', '?') for a in axes_meta).upper()
        units = [a.get('unit') for a in axes_meta]

        ds0 = ms['datasets'][0]
        level0_path = ds0['path']

        scale = None
        for t in ds0.get('coordinateTransformations', []):
            if t.get('type') == 'scale':
                scale = list(t['scale'])
                break
        if scale is None:
            scale = [1.0] * len(axes_meta)
        for t in ms.get('coordinateTransformations', []):
            if t.get('type') == 'scale':
                scale = [a * b for a, b in zip(scale, t['scale'])]

        return root, axes_str, level0_path, scale, units

    @staticmethod
    def _scale_to_pixelsize(axes_str, scale, units):
        """XY pixel size in micrometres from an NGFF scale + unit, or None."""
        per_axis = []
        for ax in ('X', 'Y'):
            if ax not in axes_str:
                continue
            i = axes_str.index(ax)
            s = float(scale[i])
            u = (units[i] or 'micrometer').lower()
            if u in ('micrometer', 'microns', 'micron', 'um', 'µm', 'μm'):
                pass
            elif u in ('nanometer', 'nm'):
                s *= 1e-3
            elif u in ('millimeter', 'mm'):
                s *= 1e3
            elif u in ('meter', 'm'):
                s *= 1e6
            else:
                logger.warning(
                    f"Unknown spatial unit '{u}' on NGFF axis '{ax}' — assuming micrometres.")
            per_axis.append(s)
        if not per_axis:
            return None
        if len(per_axis) == 2 and not np.isclose(per_axis[0], per_axis[1]):
            logger.warning(
                f"Anisotropic XY pixel size from NGFF (x={per_axis[0]:.6g}, "
                f"y={per_axis[1]:.6g} µm). Using the mean.")
        return float(np.mean(per_axis))

    @staticmethod
    def _scale_to_frametime(axes_str, scale, units):
        """Frame time in seconds from an NGFF scale + unit, or None when there is no T axis."""
        if 'T' not in axes_str:
            return None
        i = axes_str.index('T')
        s = float(scale[i])
        u = (units[i] or 'second').lower()
        if u in ('second', 's', 'sec'):
            pass
        elif u in ('millisecond', 'ms'):
            s *= 1e-3
        elif u in ('minute', 'min'):
            s *= 60.0
        elif u in ('hour', 'h'):
            s *= 3600.0
        else:
            logger.warning(f"Unknown time unit '{u}' on the NGFF T axis — assuming seconds.")
        return s

    def _extract_metadata_only_ngff(self, axes=None) -> None:
        """NGFF counterpart of :meth:`_extract_metadata_only`.

        Values passed to the constructor always win; this only fills what the user left
        unset, so an explicit ``pixelsize=`` still overrides a store's own calibration.
        """
        root, axes_str, level0_path, scale, units = self._read_ngff_attrs()
        if axes is not None:
            axes_str = str(axes).upper()
        self._validate_axes(axes_str)

        if self.metadata.pixelsize is None:
            self.metadata.pixelsize = self._scale_to_pixelsize(axes_str, scale, units)
        if self.metadata.frametime is None:
            self.metadata.frametime = self._scale_to_frametime(axes_str, scale, units)
        self.metadata.axes = axes_str

        shape_orig = tuple(root[level0_path].shape)
        if len(shape_orig) != len(axes_str):
            raise MetaDataError(
                f"NGFF axes '{axes_str}' do not match the array shape {shape_orig} "
                f"at {self.file_path}.")
        self.metadata.shape_orig = shape_orig

        processed_axes = axes_str
        if 'C' in axes_str:
            c_axis = axes_str.index('C')
            shape_after_channel = tuple(d for j, d in enumerate(shape_orig) if j != c_axis)
            processed_axes = axes_str.replace('C', '')
        else:
            shape_after_channel = shape_orig

        stack_axis = 'T' if 'T' in processed_axes else ('Z' if 'Z' in processed_axes else None)
        target_axes = ([stack_axis] if stack_axis else [])
        target_axes += [ax for ax in ('Y', 'X') if ax in processed_axes]
        perm = [processed_axes.index(ax) for ax in target_axes]
        final_shape = tuple(shape_after_channel[i] for i in perm)
        final_shape = tuple(d for d in final_shape if d > 1) or final_shape

        self.metadata.shape = final_shape
        self.metadata.size = (final_shape[-2], final_shape[-1]) if len(final_shape) >= 2 else None

        stack_len = 1
        if 'T' in axes_str:
            stack_len = int(shape_orig[axes_str.index('T')])
        elif 'Z' in axes_str:
            stack_len = int(shape_orig[axes_str.index('Z')])
        self.metadata.n_stack = stack_len

        if self.metadata.frametime and stack_len > 1:
            self.metadata.time = np.arange(0, stack_len * self.metadata.frametime,
                                           self.metadata.frametime)
        else:
            self.metadata.time = None

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
        frames = self._image_selection(frames)
        # Fast path: pixels already ingested into the store -> lazy zarr slice.
        if self.store.has_image():
            arr = self.store.image_handle()
            return arr[...] if frames is None else arr[frames]

        # First open of a TIFF: read it, ingest into the store, then slice.
        data = self._read_source_tif(axes=axes)
        return data if frames is None else data[frames]

    def _image_selection(self, frames):
        """Index for ``read_imgs``: None for the whole image/movie, an int for one frame (also from a numpy
        integer or 0-d array), a slice, or a list of ints in the caller's order. A single image has frame 0
        only; any other frame is refused instead of silently returning frame 0."""
        if frames is None or (isinstance(frames, str) and frames == 'all'):
            return None
        if self.metadata.n_stack is None or self.metadata.n_stack <= 1:
            if not isinstance(frames, slice):
                canonical_frames(frames, 1)               # raises for anything but frame 0
            return None
        if isinstance(frames, slice):
            return frames
        if not isinstance(frames, (list, tuple, range)) and np.ndim(frames) == 0:
            return _as_int(frames)
        return self._normalize_frames(frames)

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

            # Extract metadata using original axes order (updates self.metadata)
            self._harvest_metadata(series, tif, original_axes)

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
            
            # ImageJ metadata might say channels=1 or slices=1 while the data still
            # carries singleton dimensions for those axes, so verify the axis string
            # against the actual data shape.
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
            # an unclassified stack axis ('I' sequence / 'Q' other) is a movie;
            # a z-stack can still be forced via axes='ZYX'
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
                    self._check_channel_count(n_chan, axes)
                    if n_chan <= self._MAX_PLAUSIBLE_CHANNELS:
                        self._warn_once(
                            f"Multi-channel image detected (n={n_chan}). Using channel 0 by default; "
                            f"choose another channel if the sarcomeres are in a different one "
                            f"(Python: channel=<int>).")
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

        if px is None and tif.imagej_metadata:
            ij = tif.imagej_metadata
            px = ij.get('pixel_width') or ij.get('PixelWidth')
            try:
                px = float(px) if px is not None else None
            except (TypeError, ValueError) as e:
                logger.debug(f"Failed to convert pixel size to float: {e}")

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

        if ft is None and tif.imagej_metadata:
            ij = tif.imagej_metadata
            ft = ij.get('finterval') or ij.get('Frame interval')
            if ft is None and (fps := ij.get('fps')):
                try:
                    ft = 1 / float(fps)
                except (ValueError, ZeroDivisionError) as e:
                    logger.debug(f"Failed to compute frame time from fps: {e}")

            if ts is None:
                ts = ij.get('timestamps')
                if isinstance(ts, str):
                    try:
                        ts = json.loads(ts)
                    except Exception as e:
                        logger.debug(f"Failed to parse timestamps from ImageJ metadata: {e}")

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
        if 'C' in axes:
            self._check_channel_count(series.shape[axes.index('C')], axes)
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
                self._warn_once(message)

        if self.metadata.frametime is None and stack_len > 1:
            self._warn_once('Frame time not found in the file metadata. Enter it before motion '
                            'analysis (Python: frametime=<seconds per frame>).')

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

    def __get_store_path(self) -> str:
        """Path to the analysis subgroup of the OME-Zarr store (``<name>.ome.zarr/sarcasm``)."""
        return str(self.store.results_path)


    def _summary_rows(self) -> List[Tuple[str, str]]:
        """Extend the base summary with which steps ran and what they produced.

        Reads the manifest and a handful of scalar attrs only — no result array
        is materialised, so ``print(sarc)`` stays cheap on a 1 GB store.
        """
        rows = self._file_summary_rows()
        data = self.__dict__.get('data')
        if data is None:
            return rows

        def step_of(key):
            # params.<step>.<arg> on the recording, params.<namespace>.<step>.<arg> in a namespace
            parts = key.split('.')
            return '.'.join(parts[1:3]) if parts[1] in ('structure', 'motion') else parts[1]
        steps = {step_of(k) for k in data if k.startswith('params.') and k.count('.') >= 2}
        order = {name: i for i, name in enumerate(self._PIPELINE_STEPS)}
        ranked = sorted(steps, key=lambda s: (order.get(s, len(order)), s))
        rows.append(("steps", " → ".join(ranked) if ranked else
                     "— (no analysis yet; start with sarc.detect_sarcomeres())"))

        counts = [f"{len(data)} keys"]
        if 'structure.sarcomere.n_vectors' in data:
            n_vec = np.asarray(data['structure.sarcomere.n_vectors'], dtype=float)
            if n_vec.size and not np.all(np.isnan(n_vec)):
                counts.append(f"~{int(np.nanmedian(n_vec))} vectors/frame")
        if 'motion.tracks.n' in data:
            counts.append(f"{int(data['motion.tracks.n'])} tracks")
        grouped = [(kind, int(data[f'motion.groups.{kind}.n'])) for kind in self.motion._grouped_kinds()]
        if grouped:
            counts.append("groups: " + ", ".join(f"{n} {kind}" for kind, n in grouped))
        rows.append(("results", "  ·  ".join(counts)))
        return rows

    def commit(self) -> None:
        """Persist the staged result writes to the Zarr store (incremental).

        Only members changed since the last call are rewritten. Image metadata
        is mirrored into the store so it is a single, self-contained artifact.
        Every analysis step commits by itself unless ``auto_save=False``.
        """
        self.data.flush()
        self.save_metadata()  # mirror metadata into the OME-Zarr store root

    #: Params steps of stores written by SarcAsM 1.0.0b4 or earlier, before the structure/motion namespaces.
    _OLD_PARAMS_STEPS = ('analyze_cell_mask', 'analyze_z_bands', 'analyze_sarcomere_vectors', 'analyze_myofibrils',
                         'analyze_sarcomere_domains', 'detect_z_bands_fast_movie', 'track_sarcomere_vectors',
                         'group_tracks', 'analyze_track_motion')

    def _load_structure_data(self) -> None:
        """Bind the lazy results store (the ``sarcasm/`` group of the OME-Zarr store)."""
        self.data = Results(self.__get_store_path())
        old = sorted({k.split('.')[1] for k in self.data
                      if k.startswith('params.') and k.split('.')[1] in self._OLD_PARAMS_STEPS})
        if old:
            logger.warning(
                f"This analysis was made by SarcAsM 1.0.0b4 or earlier ({', '.join(old)}). Its results use key "
                f"names this version does not read, so these steps count as not run. Run them again "
                f"(sarc.structure.* and sarc.motion.*), or open with restart=True to start a fresh store.")

    def detect_sarcomeres(self, frames: Union[str, int, Sequence[int]] = 'all',
                          model_path: str = None, max_patch_size: Union[Tuple[int, int], str] = 'auto',
                          normalization_mode: str = 'all', clip_thres: Tuple[float, float] = (0., 99.98),
                          rescale_factor: float = 1.0, batch_size: Union[int, str] = 'auto',
                          memory_budget_gb: float = 2.0, prune_level: int = None,
                          progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """
        Predict sarcomeres (Z-bands, M-bands, sarcomere mask, orientation, cell mask) with the U-Net.

        The cell-mask features (``structure.cell.mask_area``, ``mask_area_ratio``,
        ``mask_intensity``) are computed for the detected frames right away
        (:meth:`~sarcasm.structure.StructureAnalysis.analyze_cell_mask`).

        Parameters
        ----------
        frames : {'all', int, sequence of int} or None, optional
            Frames for sarcomere detection: ``'all'`` (or None), a frame number, or
            any sequence of frame numbers (list, tuple, range, array; order and
            repeats do not matter). ``k``, ``[k]`` and ``np.array([k])`` are the same
            selection. Frames outside the movie raise a ValueError. Default is 'all'.
        model_path : str or None, optional
            None (or ``'generalist'``, ``'auto'``) uses the bundled ``'generalist'`` (v2) model at
            every pixel size. ``'legacy'`` selects the first bundled model, which reproduces
            high-magnification results of earlier versions (they picked it below 0.08 µm/px); any other
            string is a path to custom U-Net weights, e.g. the betas' v1 model (see sarcasm/models/README.md). The resolved model is stored as
            ``params.detect_sarcomeres.model_path``. Default is None.
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
            the masks, so validate it on your data first. None uses the full model,
            the default.
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        frames = self._canonical_frames(frames)
        max_patch_size = Utils.check_and_round_max_patch_size(max_patch_size)
        n_stack = self.metadata.n_stack or 1
        if frames == 'all' or frames == list(range(n_stack)):
            # Hand the detector a lazy handle so the raw stack is read block by
            # block instead of being materialised in full.
            if not self.store.has_image():
                self.read_imgs()  # first open: ingest the source TIFF into the store
            images = self.store.image_handle() if self.store.has_image() else self.read_imgs()
            list_frames = list(range(images.shape[0] if len(images.shape) > 2 else 1))
        else:
            # a subset of a movie: read as a (n, H, W) stack also for a single frame, so that the masks hold
            # one page per detected frame however the frame was spelled (k, [k], np.int64(k), ...)
            images = self.read_imgs(frames=frames)
            list_frames = frames

        # Rescaling is done inside detect_sarcomeres_unet, which also scales the
        # predicted masks back to the input resolution. Doing it here as well would
        # apply the factor twice and return masks at the wrong size.

        # Check pixelsize is not None
        if self.metadata.pixelsize is None:
            raise ValueError("Pixel size is not available. Please provide pixelsize during initialization.")
        
        # Delegate to the detection module, which writes each block of predicted
        # masks straight into the OME-Zarr store as it is produced.
        info = {}

        def predict():
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

        self._predict_on_gpu(predict, 'detect_sarcomeres')

        _dict = {
            'params.detect_sarcomeres.frames': list_frames,
            'params.detect_sarcomeres.model_path': info.get('model', model_path),
            'params.detect_sarcomeres.normalization_mode': normalization_mode,
            'params.detect_sarcomeres.clip_thres': clip_thres,
            'params.detect_sarcomeres.rescale_factor': rescale_factor,
            'params.detect_sarcomeres.prune_level': prune_level,
            'params.detect_sarcomeres.max_patch_size': max_patch_size,
            # what 'auto' actually resolved to, so the run can be reproduced
            'params.detect_sarcomeres.patch_size_used': info.get('patch_size'),
        }
        self.data.update(_dict)
        if self.auto_save:
            # the masks are already on disk; store the frames they belong to with them, so that a failure in
            # the steps below cannot leave new masks next to the frame list of an earlier detection
            self.commit()
        # the cell mask is one of the predicted outputs: its area / intensity
        # features belong to the detection step, not to a separate call
        self.structure.analyze_cell_mask(frames=list_frames)

    def _predict_on_gpu(self, predict, step: str) -> None:
        """Run ``predict()``; on Apple GPUs run it once more if macOS aborted GPU work during it
        (the masks would hold garbage, see :mod:`sarcasm._internal.gpu_watch`), and raise if
        that happens twice rather than store wrong masks."""
        for attempt in (1, 2):
            with watch_metal_errors(self.device) as watch:
                predict()
            if not watch.failed:
                return
            if attempt == 1:
                logger.warning(f'{step}: macOS aborted part of the GPU computation (the display needed the '
                               f'GPU), so the predicted masks may be wrong. Predicting again.')
        raise RuntimeError(f"{step}: macOS aborted the GPU computation twice, so the predicted masks cannot be "
                           f"trusted. Close GPU-heavy programs (or hide the movie in the viewer) and run it "
                           f"again, or use device='cpu'.")

    def load_mask_full_stack(self, name: str) -> Optional[np.ndarray]:
        """
        Load a mask from the store by name and expand it to full stack length for display.

        Masks are stored sparsely (only for detected frames). For napari display alongside
        the raw movie, this returns an (n_stack, ...) array with computed frames placed at
        their original frame indices and zeros elsewhere. ``name`` is a store mask name
        (e.g. ``'zbands'``, ``'mbands'``, ``'cell_mask'``, ``'zbands_fast_movie'``). Returns
        None if the mask is not present.
        """
        if not self.mask_exists(name):
            return None
        arr = self.read_mask(name)
        n_stack = self.metadata.n_stack or 1
        # only the detection masks are stored sparsely; the fast-movie Z-bands cover every frame
        detected = self._detected_frames() if name in self._DETECTION_MASKS else None
        if n_stack <= 1 or detected is None or detected == list(range(n_stack)):
            return arr
        pages = arr if arr.ndim > self._MASK_FRAME_NDIM.get(name, 2) else arr[None]
        if len(pages) != len(detected):
            return arr  # unexpected shape, pass through
        full = np.zeros((n_stack,) + pages.shape[1:], dtype=arr.dtype)
        for page, f in zip(pages, detected):
            if 0 <= f < n_stack:
                full[f] = page
        return full

    # Analysis steps in pipeline order, so ``print(sarc)`` lists what has run in
    # the order it ran rather than alphabetically. Steps not listed sort last.
    _PIPELINE_STEPS = ('detect_sarcomeres', 'motion.detect_z_bands_fast_movie',
                       'structure.analyze_cell_mask', 'structure.analyze_z_bands',
                       'structure.analyze_vectors', 'structure.analyze_myofibrils',
                       'structure.analyze_domains', 'motion.detect_lois',
                       'motion.track', 'motion.group_tracks',
                       'motion.analyze')

    @functools.cached_property
    def structure(self) -> "StructureAnalysis":
        """Sarcomere morphology of this recording (:class:`~sarcasm.structure.StructureAnalysis`)."""
        from sarcasm.structure import StructureAnalysis
        return StructureAnalysis(self)

    @functools.cached_property
    def motion(self) -> "MotionAnalysis":
        """Sarcomere motion of this recording (:class:`~sarcasm.motion.MotionAnalysis`)."""
        from sarcasm.motion import MotionAnalysis
        return MotionAnalysis(self)
