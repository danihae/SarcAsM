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
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, Literal, Dict, Any, List

import numpy as np
import tifffile
import torch

from sarcasm.exceptions import MetaDataError
from sarcasm.meta_data_handler import ImageMetadata
from sarcasm.utils import Utils


class SarcAsM:
    """
    Base class for sarcomere structural and functional analysis.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to the TIFF file for analysis.
    restart : bool, optional
        If True, deletes existing analysis and starts fresh (default: False).
    pixelsize : float or None, optional
        Physical pixel size in micrometres (µm). If None, the class tries to
        extract it from file metadata; otherwise it must be provided manually.
    frametime : float or None, optional
        Time between frames in seconds. If None, the class tries to extract it
        from file metadata; otherwise it must be provided manually.
    channel : int or None, optional
        Channel index that contains the sarcomere signal in multicolour stacks
        (default: None).
    axes : str or None, optional
        Explicit order of image dimensions (e.g. ``'TXYC'`` or ``'YX'``).
        If None, the order is auto-detected from OME-XML, ImageJ tags or shape
        heuristics; this is the recommended mode when the GUI offers a
        drop-down override.
    auto_save : bool, optional
        Automatically save analysis results when True (default: True).
    use_gui : bool, optional
        Enable GUI-mode behaviour (default: False).
    device : Union[torch.device, Literal['auto']], optional
        PyTorch computation device. ``'auto'`` selects CUDA/MPS if available
        (default: 'auto').
    **info : Any
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
        Image metadata
    device : torch.device
        PyTorch device on which computations are performed.

    Dynamic Attributes (loaded on demand)
    -------------------------------------
    zbands : ndarray
        Binary Z-band mask.
    zbands_fast_movie : ndarray
        Binary Z-band mask for the high-temporal-resolution movie.
    mbands : ndarray
        Binary M-band mask.
    orientation : ndarray
        Sarcomere orientation map.
    cell_mask : ndarray
        Binary cell mask.
    sarcomere_mask : ndarray
        Binary sarcomere mask.
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
            **info: Dict[str, Any]
    ):
        # Convert filename to absolute path (as a string)
        self.file_path = os.path.abspath(str(file_path))
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        # Configuration
        self.auto_save = auto_save
        self.use_gui = use_gui
        self.restart = restart
        self.info = info

        # Directory structure: use the filename without extension as the base directory
        base_name = os.path.splitext(self.file_path)[0]
        self.base_dir = base_name + '/'  # This is a directory path as a string.
        self.data_dir = os.path.join(self.base_dir, "data/")
        self.analysis_dir = os.path.join(self.base_dir, "analysis/")

        # Handle restart: if restart is True and base_dir exists, remove it
        if restart and os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)

        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        # File paths
        self.file_zbands = os.path.join(self.base_dir, "zbands.tif")
        self.file_zbands_fast_movie = os.path.join(self.base_dir, "zbands_fast_movie.tif")
        self.file_mbands = os.path.join(self.base_dir, "mbands.tif")
        self.file_orientation = os.path.join(self.base_dir, "orientation.tif")
        self.file_cell_mask = os.path.join(self.base_dir, "cell_mask.tif")
        self.file_sarcomere_mask = os.path.join(self.base_dir, "sarcomere_mask.tif")

        # Initialize metadata
        self.metadata = ImageMetadata(
            file_name=os.path.basename(self.file_path),
            file_path=self.file_path,
            pixelsize = pixelsize,
            frametime = frametime,
            channel = channel,
            axes = axes,
        )

        # Load existing or create new metadata
        self.meta_file = Path(self.data_dir) / "metadata.json"
        if self.meta_file.exists() and not self.restart:
            try:
                self.metadata = ImageMetadata.load_from_file(self.meta_file)
            except:
                if not self.use_gui:
                    MetaDataError(
                        "Loading metadata failed. This can happen when the metadata file was "
                        "created with an older version (<0.2.0). Restart the analysis by setting restart=True.")
                else:
                    pass
        else:
            # Will be populated by read_imgs, then saved
            _ = self.image
            pass

        # Dictionary of models
        self.model_dir = Utils.get_models_dir()

        # Device configuration: auto-detect or validate provided device
        if device == "auto":
            self.device = Utils.get_device(print_device=False)
        else:
            if isinstance(device, str):
                try:
                    self.device = torch.device(device)
                except RuntimeError as e:
                    raise ValueError(f"Invalid device string: {device}") from e
            elif isinstance(device, torch.device):
                self.device = device
            else:
                raise ValueError(
                    f"Invalid device type {type(device)}. "
                    "Expected torch.device instance or valid device string "
                    "(e.g., 'cuda', 'cpu', 'mps')"
                )

    def __getattr__(self, name: str) -> Any:
        """Dynamic loading of analysis result TIFFs"""
        attr_map = {
            'image': self.file_path,
            'zbands': self.file_zbands,
            'zbands_fast_movie': self.file_zbands_fast_movie,
            'mbands': self.file_mbands,
            'orientation': self.file_orientation,
            'cell_mask': self.file_cell_mask,
            'sarcomere_mask': self.file_sarcomere_mask
        }

        if name in attr_map:
            import tifffile
            file_path = attr_map[name]
            if name == 'image':
                return self.read_imgs()
            else:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Required analysis file missing: {os.path.basename(file_path)}\n"
                        f"Run the 'detect_sarcomeres' to create this file."
                    )
                return tifffile.imread(file_path)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __dir__(self) -> list[str]:
        """Augment autocomplete with dynamic attributes"""
        standard_attrs = super().__dir__()
        dynamic_attrs = [
            'zbands', 'zbands_fast_movie', 'mbands',
            'orientation', 'cell_mask', 'sarcomere_mask'
        ]
        return sorted(set(standard_attrs + dynamic_attrs))

    def __str__(self):
        """Returns a pretty, concise string representation of the SarcAsM object."""
        summary = [
            f"╔══════════════════════════════════════════════════════",
            f"║ SarcAsM Analysis v{self.metadata.version}",
            f"║ ─────────────────────────────────────────────────────",
            f"║ File path: {os.path.basename(self.file_path)}",
            f"║ Base directory: {os.path.dirname(self.base_dir)}",
            f"║ Device: {self.device}",
            f"║ Pixel size: {round(self.metadata.pixelsize, 5)} µm",
            f"║ Analysis timestamp: {self.metadata.timestamp_analysis}",
            f"╚══════════════════════════════════════════════════════"
        ]

        return "\n".join(summary)

    def open_base_dir(self):
        """
        Open the base directory of the tiff file in the file explorer.
        """
        Utils.open_folder(self.base_dir)

    def save_metadata(self):
        """
        Save the current metadata object to self.meta_file as JSON.
        """
        ImageMetadata.save_to_file(self.metadata, self.meta_file)

    def read_imgs(self, frames=None, axes=None):
        """
        Load and process TIFF data with metadata extraction.

        Parameters
        ----------
        frames : int, list, slice, or None, optional
            Frame selection for stacks. None loads all frames (default).
        axes : str, optional
            Dimension order override (e.g., 'TXYC'). Auto-detected if None.

        Returns
        -------
        np.ndarray
            Image data in internal format: (Y, X) or (Stack, Y, X).
        """
        with tifffile.TiffFile(self.file_path) as tif:
            series = tif.series[0]
            raw_data = series.asarray()

            # Determine or use provided axes order
            if axes is None:
                axes = self._determine_axes(series, tif)
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

            # Apply frame selection if specified
            if frames not in (None, 'all') and meta.n_stack > 1:
                data = data[frames]

            # Final cleanup and metadata updates
            data = data.squeeze()
            self.metadata.shape = data.shape  # shape after all processing
            self.metadata.size = data.shape[-2:]  # (height, width)
            self.save_metadata()

            return data

    @staticmethod
    def _determine_axes(series, tif: tifffile.TiffFile) -> str:
        """
        Return an upper-case axis string such as 'TCZYX', 'YXC', 'YX', …

        Raises
        ------
        ValueError
            if no reasonable guess is possible and the caller must supply
            the order manually.
        """
        # OME-TIFF
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                return root.find('.//{*}Image').attrib['DimensionOrder'].upper()
            except Exception:
                pass  # fall through to next strategy

        # ImageJ hyper-stack
        if tif.imagej_metadata:
            ij = tif.imagej_metadata
            order = ''
            if ij.get('frames', 1) > 1: order += 'T'
            if ij.get('slices', 1) > 1: order += 'Z'
            if ij.get('channels', 1) > 1: order += 'C'
            order += 'YX'
            return order

        # tifffile’s own guess
        if series.axes:
            axes = series.axes.upper().replace('S', 'C')  # S → C (samples)
            if 'Q' not in axes:  # ignore unknown axis
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
        Isolate the channel requested by ``self.channel`` and remove the
        channel axis from the array.

        Parameters
        ----------
        data
            Numpy array as it was read from disk (still in *source* order).
        axes
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
            • if the requested channel index is out of range
            • if ``self.metadata.channel`` is given but the image has no ``C`` axis
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
                    print(
                        f"Multi-channel image detected (n={n_chan}). "
                        f"Using channel 0 by default.  "
                        f"Pass   Structure(..., channel=<int>)   to override."
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
                print('Warning: ' + message)

        else:
            self.metadata.channel = None

        return data, axes

    def _harvest_metadata(self, series, tif, axes) -> ImageMetadata:
        """Collect metadata from tif and update the instance metadata object."""

        # pixel size
        px = None
        if tif.ome_metadata:
            try:
                root = ET.fromstring(tif.ome_metadata)
                px_elem = root.find('.//{*}Pixels')
                if px_elem is not None:
                    px = px_elem.get('PhysicalSizeX')
                    px = float(px) if px else None
            except Exception:
                pass

        if px is None and tif.imagej_metadata:
            ij = tif.imagej_metadata
            px = ij.get('pixel_width') or ij.get('PixelWidth')
            try:
                px = float(px) if px is not None else None
            except (TypeError, ValueError):
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
                except Exception:
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
            except Exception:
                pass

        if ft is None and tif.imagej_metadata:
            ij = tif.imagej_metadata
            ft = ij.get('finterval') or ij.get('Frame interval')
            if ft is None and (fps := ij.get('fps')):
                try:
                    ft = 1 / float(fps)
                except (ValueError, ZeroDivisionError):
                    pass

            if ts is None:
                ts = ij.get('timestamps')
                if isinstance(ts, str):
                    try:
                        ts = json.loads(ts)
                    except Exception:
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
                f"Please enter manually (e.g., Structure(filename, pixelsize=0.1))."
            )

        if self.metadata.pixelsize and not (0.01 <= self.metadata.pixelsize <= 0.5):
            message = (f"Pixel size {self.metadata.pixelsize} µm is outside reasonable range (0.01-0.5 µm). "
                       f"Please check your input or file metadata.")
            if not self.use_gui:
                raise MetaDataError(message)
            else:
                print('Warning: ' + message)

        if self.metadata.frametime is None and stack_len > 1:
            print('Warning: frametime could not be extracted from tif file. '
                  'Please enter manually if needed (e.g., Structure(file, frametime=0.1)).')

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

        # Save metadata if auto_save is enabled
        if self.auto_save:
            ImageMetadata.save_to_file(self.metadata, self.meta_file)

        return self.metadata

    @staticmethod
    def _validate_axes(axes: str) -> None:
        """
        Raise if `axes` is not a unique subset of {X, Y, T, C, Z}.
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
        Parameters
        ----------
        data : np.ndarray
            The image data as stored on disk.
        source_axes : str
            Axis string returned by `_determine_axes`.

        Returns
        -------
        np.ndarray
            Array permuted to (Stack, Y, X) or (Y, X).
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
        if perm:
            data = data.transpose(perm)

        return data

    def remove_intermediate_tiffs(self) -> None:
        """
        Removes intermediate TIFF files while preserving the original input.
        """
        targets = [
            self.file_zbands,
            self.file_mbands,
            self.file_orientation,
            self.file_cell_mask,
            self.file_sarcomere_mask,
            self.file_zbands_fast_movie,
        ]

        for path in targets:
            if os.path.exists(path):
                os.remove(path)
