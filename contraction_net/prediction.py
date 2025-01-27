import numpy as np
import torch

from .contraction_net import ContractionNet
from .utils import get_device

# select device
device = get_device()


def predict_contractions(data, model, network=ContractionNet):
    """predict contraction intervals time-series with neural network

    Parameters
    ----------
    data : ndarray
        1D array with time-series of contraction
    model : str
        trained model weights (.pt file)
    network : nn.Module
        Network to predict contractions from time-series
    standard_normalizer : bool, ndarray
        If False, each data is normalized by its mean and std. If True, the mean and std from the training data set are
        applied. If ndarray, the data is normalized with the entered args [[input_mean, input_std], [vel_mean, vel_std]]
    """
    # load model state
    state_dict = torch.load(model, map_location=device)
    # initiate model
    model = network(state_dict['n_filter'], in_channels=state_dict['in_channels'],
                    out_channels=state_dict['out_channels']).to(device)
    model.load_state_dict(state_dict['state_dict'])
    # resize data
    len_data = data.shape[0]
    contr = np.pad(data, (0, 32 - np.mod(len_data, 32)), mode='reflect')
    # convert to torch
    data = torch.from_numpy(contr.astype('float32')).view(1, 1, -1).to(device)
    # predict data
    res = model(data)[0][0]
    # resize data and convert to numpy
    res = res.detach().cpu().numpy()
    return res[:, :len_data]
