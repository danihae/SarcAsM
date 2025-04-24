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


import datetime
import os
import shutil
from typing import Union, Literal, Dict, Any, List

import numpy as np
import tifffile
import torch

from sarcasm._version import __version__
from sarcasm.meta_data_handler import MetaDataHandler
from sarcasm.utils import Utils


class SarcAsM:
    """
    Base class for sarcomere structural and functional analysis.

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

    Attributes
    ----------
    filepath : str
        Absolute path to the input TIFF file.
    base_dir : str
        Base directory for analysis of the TIFF file.
    data_dir : str
        Directory for processed data storage.
    analysis_dir : str
        Directory for analysis results.
    device : torch.device
        Active computation device for PyTorch operations.

    Dynamic Attributes (loaded on demand):
    zbands : ndarray
        Z-band mask
    zbands_fast_movie : ndarray
        High-temporal resolution Z-band mask
    mbands : ndarray
        Sarcomere M-band mask
    orientation : ndarray
        Sarcomere orientation map
    cell_mask : ndarray
        Binary cell mask
    sarcomere_mask : ndarray
        Binary sarcomere mask
    """
    meta_data_handler: MetaDataHandler
    metadata: dict[str, Any]

    def __init__(
            self,
            filepath: Union[str, os.PathLike],
            restart: bool = False,
            channel: Union[int, None, Literal['RGB']] = None,
            auto_save: bool = True,
            use_gui: bool = False,
            device: Union[torch.device, Literal['auto', 'mps', 'cuda', 'cpu']] = 'auto',
            **info: Dict[str, Any]
    ):
        # Convert filename to absolute path (as a string)
        self.filepath = os.path.abspath(str(filepath))
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Input file not found: {self.filepath}")

        # Add version and analysis timestamp to metadata
        info['version'] = __version__
        info['timestamp_analysis'] = datetime.datetime.now().isoformat()

        # Configuration
        self.auto_save = auto_save
        self.channel = channel
        self.use_gui = use_gui
        self.restart = restart
        self.info = info

        # Directory structure: use the filename without extension as the base directory
        base_name = os.path.splitext(self.filepath)[0]
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

        # Initialize subsystems: metadata handler
        self.meta_data_handler = MetaDataHandler(self)

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
            'zbands': self.file_zbands,
            'zbands_fast_movie': self.file_zbands_fast_movie,
            'mbands': self.file_mbands,
            'orientation': self.file_orientation,
            'cell_mask': self.file_cell_mask,
            'sarcomere_mask': self.file_sarcomere_mask
        }

        if name in attr_map:
            import tifffile
            filepath = attr_map[name]
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Required analysis file missing: {os.path.basename(filepath)}\n"
                    f"Run the 'detect_sarcomeres' to create this file."
                )
            return tifffile.imread(filepath)

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
            f"║ SarcAsM Analysis v{self.info.get('version', 'unknown')}",
            f"║ ─────────────────────────────────────────────────────",
            f"║ File path: {os.path.basename(self.filepath)}",
            f"║ Base directory: {os.path.dirname(self.base_dir)}",
            f"║ Device: {self.device}",
            f"║ Pixel size: {round(self.metadata['pixelsize'], 5)} µm",
        ]

        # Add timestamp
        timestamp = self.info.get('timestamp_analysis', 'unknown')
        if timestamp != 'unknown':
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        summary.append(f"║ Analysis timestamp: {timestamp}")
        summary.append(f"╚══════════════════════════════════════════════════════")

        return "\n".join(summary)

    def open_base_dir(self):
        """Opens the base directory of the tiff file in the file explorer."""
        Utils.open_folder(self.base_dir)

    @property
    def model_dir(self) -> str:
        """
        Returns the path to the model directory.
        """
        current_file = os.path.abspath(__file__)
        # Move two directories up to get the parent directory, then append 'models'
        parent_dir = os.path.dirname(os.path.dirname(current_file))
        return os.path.join(parent_dir, "models") + os.sep

    def read_imgs(self, frames: Union[str, int, List[int]] = None):
        """Load tif file, and optionally select channel"""
        if frames is None or frames == 'all':
            data = tifffile.imread(self.filepath)
        else:
            data = tifffile.imread(self.filepath, key=frames)

        if self.channel is not None:
            if self.channel == 'RGB':
                # Convert RGB to grayscale
                if data.ndim == 3 and data.shape[2] == 3:  # Single RGB image
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
                elif data.ndim == 4 and data.shape[3] == 3:  # Stack of RGB images
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
            elif isinstance(self.channel, int):
                if data.ndim == 3:
                    data = data[:, :, self.channel]
                elif data.ndim == 4:
                    data = data[:, :, :, self.channel]
            else:
                raise Exception('Parameter "channel" must be either int or "RGB"')

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
