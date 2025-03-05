import datetime
import os
import shutil
from typing import Union, Literal, Dict, Any

import torch

from ._version import __version__
from .meta_data_handler import MetaDataHandler
from .structure import Structure
from .utils import Utils


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
    """
    meta_data_handler: MetaDataHandler
    metadata: dict[str, Any]
    structure: Structure

    def __init__(
            self,
            filepath: Union[str, os.PathLike],
            restart: bool = False,
            channel: Union[int, None, Literal['RGB']] = None,
            auto_save: bool = True,
            use_gui: bool = False,
            device: Union[torch.device, Literal['auto']] = 'auto',
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

        # File paths: determine the z-bands file based on legacy naming
        if os.path.exists(os.path.join(self.base_dir, "sarcomeres.tif")) and not os.path.exists(os.path.join(self.base_dir, "zbands.tif")):
            self.file_z_bands = os.path.join(self.base_dir, "sarcomeres.tif")
        else:
            self.file_z_bands = os.path.join(self.base_dir, "zbands.tif")

        self.file_z_bands_fast_movie = os.path.join(self.base_dir, "zbands_fast_movie.tif")
        self.file_midlines = os.path.join(self.base_dir, "midlines.tif")
        self.file_orientation = os.path.join(self.base_dir, "orientation.tif")
        self.file_cell_mask = os.path.join(self.base_dir, "cell_mask.tif")
        self.file_sarcomere_mask = os.path.join(self.base_dir, "sarcomere_mask.tif")

        # Initialize subsystems: metadata handler and structure
        self.meta_data_handler = MetaDataHandler(self)
        self.structure = Structure(self)

        # Device configuration: auto-detect or validate provided device
        if device == "auto":
            self.device = Utils.get_device(print_device=False)
        else:
            if not isinstance(device, torch.device):
                raise ValueError(f"Invalid device: {device}. Expected torch.device instance.")
            self.device = device

    @property
    def model_dir(self) -> str:
        """
        Returns the path to the model directory.
        """
        current_file = os.path.abspath(__file__)
        # Move two directories up to get the parent directory, then append 'models'
        parent_dir = os.path.dirname(os.path.dirname(current_file))
        return os.path.join(parent_dir, "models") + os.sep

    def remove_intermediate_tiffs(self) -> None:
        """
        Removes intermediate TIFF files while preserving the original input.
        """
        targets = [
            self.file_z_bands,
            self.file_z_bands_fast_movie,
            self.file_midlines,
            self.file_orientation,
            self.file_cell_mask,
            self.file_sarcomere_mask
        ]

        for path in targets:
            if os.path.exists(path):
                os.remove(path)
