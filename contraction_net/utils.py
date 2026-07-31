"""Shared helpers for ContractionNet.

Trace simulation lives in :mod:`contraction_net.simulation` and the morphological clean-up
of predicted masks in :func:`contraction_net.benchmark.postprocess`.
"""

import torch


def get_device(print_device=False):
    """
    Determines the most suitable device (CUDA, MPS, or CPU) for PyTorch operations.

    Returns:
    - A torch.device object representing the selected device.
    """
    if torch.backends.cuda.is_built():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_built():  # only for Apple M1/M2/...
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Warning: No CUDA or MPS device found. Calculations will run on the CPU, "
              "which might be slower.")
    if print_device:
        print(f"Using device: {device}")
    return device
