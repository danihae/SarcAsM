import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import distance_transform


class DataProcess(Dataset):
    """
    Class to create training data object for ContractionNet training

    Parameters
    ----------
    source_dir : Tuple[str, str]
        Path of training data [images, labels]. Images need to be tif files.
    input_len : int
        Length of the input sequences
    normalize : bool
        Whether to normalize each time-series (standard normalizer)
    aug_factor : int
        Factor of image augmentation
    val_split : float
        Validation split for training
    noise_amp : float
        Amplitude of Gaussian noise for image augmentation

    """
    def __init__(self, source_dir, input_len=512, normalize=False, val_split=0.2, aug_factor=10, aug_p=0.5,
                 noise_amp=0.2, random_offset=0.25, random_outlier=0.5, random_drift=(0.01, 0.2), random_swap=0.5,
                 random_subsampling=None):

        self.source_dir = source_dir
        self.data = []
        self.is_real = []
        self.input_len = input_len
        self.val_split = val_split
        self.normalize = normalize
        self.aug_factor = aug_factor
        self.aug_p = aug_p
        self.noise_amp = noise_amp
        self.random_offset = random_offset
        self.random_drift = random_drift
        self.random_outlier = random_outlier
        self.random_subsampling = random_subsampling
        self.random_swap = random_swap
        self.mode = 'train'
        self.__load_and_edit()
        if self.aug_factor is not None:
            self.__augment()

    def __load_and_edit(self):
        files = glob.glob(self.source_dir + '*.txt')
        print(f'{len(files)} files found')
        files_input = [f for f in files if 'peaks' not in f and 'contr' not in os.path.basename(f)]
        for file_i in files_input:
            # input
            input_i = np.loadtxt(file_i)[: self.input_len]
            # normalize
            if self.normalize:
                input_i -= np.mean(input_i)
                input_i /= np.std(input_i)
            # systole
            start_end_systole_i = np.loadtxt(file_i[:-4] + '_contr.txt')[: self.input_len]
            if len(start_end_systole_i) == len(input_i):
                contr_i = start_end_systole_i
            else:
                contr_i = np.zeros_like(input_i)
                start_end_systole_i = np.clip(start_end_systole_i, a_min=0, a_max=self.input_len)
                if start_end_systole_i.size == 0:
                    pass
                elif len(start_end_systole_i.shape) == 1:
                    contr_i[int(start_end_systole_i[0]): int(start_end_systole_i[1])] = 1
                elif len(start_end_systole_i.shape) == 2:
                    for start, end in start_end_systole_i.T:
                        contr_i[int(start): int(end)] = 1
            # check if simulated or real data
            if 'sim' in file_i or 'noise' in file_i:
                is_real_i = False
            else:
                is_real_i = True
            # make stack and create normalized distance transform per connected region
            self.data.append([input_i, contr_i, distance_transform(input_i, contr_i)])
            self.is_real.append(is_real_i)
        self.data = np.asarray(self.data)

    def __augment(self):
        _data = self.data.copy()
        self.data = []
        for i, d_i in enumerate(_data):
            for j in range(self.aug_factor):
                d_ij = d_i.copy()
                std_ij = np.std(d_ij[0])
                d_ij[0] = d_ij[0] / std_ij
                # random_swap
                d_ij[0] = np.random.choice([-1, 1], p=(1-self.random_swap, self.random_swap)) * d_ij[0]
                # random subsampling
                if self.random_subsampling is not None:
                    if np.random.binomial(1, self.aug_p):
                        d_ij = np.zeros_like(d_ij)
                        d_ij_short = np.delete(d_i, np.arange(0, 512, np.random.randint(self.random_subsampling[0],
                                                                                        self.random_subsampling[1])),
                                               axis=1)
                        d_ij[:, :d_ij_short.shape[1]] = d_ij_short
                # random noise
                if not self.is_real[i]:
                    if np.random.binomial(1, self.aug_p):
                        d_ij[0] += np.random.normal(0, self.noise_amp, size=d_i.shape[1])
                # random offset
                if np.random.binomial(1, self.aug_p):
                    d_ij[0] += np.random.normal(0, self.random_offset, size=1)
                # random outliers
                if np.random.binomial(1, self.aug_p):
                    n_outliers = np.random.randint(0, 10)
                    t_outliers = np.random.randint(0, self.input_len, n_outliers)
                    d_ij[0, t_outliers] += np.random.normal(0, self.random_outlier, n_outliers)
                # random drift
                if np.random.binomial(1, self.aug_p):
                    d_ij[0] += self.random_drift[1] * np.cos(self.random_drift[0] * np.arange(self.input_len))
                d_ij[0] = d_ij[0] * std_ij
                self.data.append(d_ij)
        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_torch = torch.from_numpy(self.data[idx]).float()
        sample = {'input': data_torch[0], 'target': data_torch[1], 'distance': data_torch[2]}
        return sample
