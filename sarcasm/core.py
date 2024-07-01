import os
import pathlib
import shutil
from typing import Union

import torch

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from .utils import Utils
from .meta_data_handler import MetaDataHandler
from .structure import Structure


class SarcAsM:
    """
    Base class for sarcomere structural and functional analysis.

    Parameters
    ----------
    filename : str
        Filename of the TIFF file for analysis.
    restart : bool, optional
        If True, deletes existing analysis and starts fresh. Defaults to False.
    channel : int or None, optional
        Specifies the channel with sarcomeres in multicolor stacks. Defaults to None.
    auto_save : bool, optional
        If True, automatically saves analysis results. Defaults to True.
    use_gui : bool, optional
        Indicates if SarcAsM is used through the application. Defaults to False.
    device : torch.device, optional
        Device on which to run Pytorch computations.
        Defaults to 'auto', which selects CUDA or MPS device if available, else CPU.
    **info : dict
        Additional metadata for analysis as kwargs (e.g. cell_line='wt').

    Attributes
    ----------
    filename : str
        Path to the TIFF file for analysis.
    auto_save : bool
        Whether to save analysis results automatically.
    channel : int or None or str
        Channel containing sarcomeres in multichannel images/movies, or 'RGB' for RGB images.
    use_gui : bool
        Whether SarcAsM is used through GUI.
    info : dict
        Arbitrary keyword arguments for additional metadata.
    folder : str
        Main folder path where all analyses and data are stored.
    data_folder : str
        Data folder path
    analysis_folder : str
        Analysis results folder path.
    file_sarcomeres : str or None
        Path to the segmented z-bands file, if exists.
    file_cell_mask : str or None
        Path to the cell mask file, if exists.
    file_sarcomere_mask : str or None
        Path to the sarcomere mask file, if exists.
    """

    def __init__(self, filename: str, restart: bool = False, channel: Union[int, None, str] = None, auto_save: bool = True,
                 use_gui: bool = False, device: Union[torch.device, str] = 'auto', **info):
        """
        Initializes a SarcAsM object with specified parameters and directory structure.
        """
        if not os.path.exists(filename):
            raise FileExistsError(f'The file {filename} does not exist!')

        self.filename = filename
        self.auto_save = auto_save
        self.channel = channel
        self.use_gui = use_gui
        self.restart = restart
        self.info = info

        self.folder = os.path.splitext(filename)[0]
        self.data_folder = os.path.join(self.folder, 'data')
        self.analysis_folder = os.path.join(self.folder, 'analysis')

        if restart and os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.analysis_folder, exist_ok=True)

        self.file_sarcomeres = os.path.join(self.folder, 'sarcomeres.tif')
        self.file_cell_mask = os.path.join(self.folder, 'cell_mask.tif')
        self.file_sarcomere_mask = os.path.join(self.folder, 'sarcomere_mask.tif')

        # Initialize MetaDataHandler and Structure (Assuming these are defined elsewhere)
        self.meta_data_handler: MetaDataHandler = MetaDataHandler(self)
        self.metadata = self.meta_data_handler.metadata
        self.structure = Structure(self)

        # default path of models (U-Net, contraction CNN)
        self.model_dir = str(pathlib.Path(__file__).resolve().parent.parent / 'models/') + '/'

        # determines the most suitable device (CUDA, MPS, or CPU) for PyTorch operations.
        if device == 'auto':
            self.device = Utils.get_device(print_device=False)
        else:
            assert isinstance(device, torch.device), "Device must be of type 'torch.device', e.g. torch.device('cpu')"
            self.device = device
