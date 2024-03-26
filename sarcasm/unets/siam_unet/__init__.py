try:
    import torch
except ImportError:
    raise ImportError('Torch is not installed. Please install through https://pytorch.org/get-started/locally/')

# check if torch version >= 1.9.0
from packaging import version
if not version.parse(torch.__version__) >= version.parse('1.9.0'):
    raise ImportError(f'This package requires torch >=1.9.0 to run. You have version {torch.__version__}.Please install through https://pytorch.org/get-started/locally/')

from .data import DataProcess
from .losses import *
from .predict import Predict
from .siam_unet import Siam_UNet
from .train import Trainer
