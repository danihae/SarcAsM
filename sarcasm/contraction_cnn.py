import glob
import os

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter

from .plots import width_1p5cols

# select device
device = torch.device('cpu')


class DataProcess(Dataset):
    """Class for training data processing"""

    def __init__(self, source_dir, input_len=512, val_split=0.2, aug_factor=10, aug_p=0.5,
                 noise_amp=0.1, random_offset=0.25, random_outlier=1, random_drift=(0.01, 0.2), random_swap=0.5,
                 random_subsampling=None):
        """
        Create training data object for network training


        Parameters
        ----------
        source_dir : Tuple[str, str]
            Path of training data [images, labels]. Images need to be tif files.
        aug_factor : int
            Factor of image augmentation
        val_split : float
            Validation split for training
        noise_amp : float
            Amplitude of Gaussian noise for image augmentation

        """
        self.source_dir = source_dir
        self.data = []
        self.input_len = input_len
        self.val_split = val_split
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
        files_input = [f for f in files if 'peaks' not in f and 'contraction' not in os.path.basename(f)]
        for file_i in files_input:
            # input
            input_i = np.loadtxt(file_i)
            # normalize
            input_i -= np.mean(input_i)
            input_i /= np.std(input_i)
            # contraction
            start_end_contraction_i = np.loadtxt(file_i[:-4] + '_contraction.txt')
            if len(start_end_contraction_i) == len(input_i):
                contraction_i = start_end_contraction_i
            else:
                contraction_i = np.zeros_like(input_i)
                start_end_contraction_i = np.clip(start_end_contraction_i, a_min=0, a_max=self.input_len)
                if start_end_contraction_i.size == 0:
                    pass
                elif len(start_end_contraction_i.shape) == 1:
                    contraction_i[int(start_end_contraction_i[0]): int(start_end_contraction_i[1])] = 1
                elif len(start_end_contraction_i.shape) == 2:
                    for start, end in start_end_contraction_i.T:
                        contraction_i[int(start): int(end)] = 1
            self.data.append([input_i, contraction_i])
        self.data = np.asarray(self.data)

    def __augment(self):
        _data = self.data.copy()
        self.data = []
        for i, d_i in enumerate(_data):
            for j in range(self.aug_factor):
                d_ij = d_i.copy()
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
                self.data.append(d_ij)
        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'input': torch.from_numpy(self.data[idx, 0].astype('float32')),
                  'gt': torch.from_numpy(self.data[idx, 1].astype('float32'))}
        return sample


class Unet(nn.Module):
    def __init__(self, n_filter=32):
        """
        Neural network for semantic image segmentation U-Net (PyTorch),
        Reference:  Falk, T. et al. U-Net: deep learning for cell counting, detection, and morphometry. Nat Methods 16,
        67–70 (2019).

        Parameters
        ----------
        n_filter : int
            Number of convolutional filters (commonly 16, 32, or 64)
        """
        super().__init__()
        # encode
        self.encode1 = self.conv(1, n_filter)
        self.encode2 = self.conv(n_filter, n_filter)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encode3 = self.conv(n_filter, 2 * n_filter)
        self.encode4 = self.conv(2 * n_filter, 2 * n_filter)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encode5 = self.conv(2 * n_filter, 4 * n_filter)
        self.encode6 = self.conv(4 * n_filter, 4 * n_filter)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encode7 = self.conv(4 * n_filter, 8 * n_filter)
        self.encode8 = self.conv(8 * n_filter, 8 * n_filter)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # middle
        self.middle_conv1 = self.conv(8 * n_filter, 16 * n_filter)
        self.middle_conv2 = self.conv(16 * n_filter, 16 * n_filter, dropout=0.5)

        # decode
        self.up1 = nn.ConvTranspose1d(16 * n_filter, 8 * n_filter, kernel_size=2, stride=2)
        self.decode1 = self.conv(16 * n_filter, 8 * n_filter)
        self.decode2 = self.conv(8 * n_filter, 8 * n_filter)
        self.up2 = nn.ConvTranspose1d(8 * n_filter, 4 * n_filter, kernel_size=2, stride=2)
        self.decode3 = self.conv(8 * n_filter, 4 * n_filter)
        self.decode4 = self.conv(4 * n_filter, 4 * n_filter)
        self.up3 = nn.ConvTranspose1d(4 * n_filter, 2 * n_filter, kernel_size=2, stride=2)
        self.decode5 = self.conv(4 * n_filter, 2 * n_filter)
        self.decode6 = self.conv(2 * n_filter, 2 * n_filter)
        self.up4 = nn.ConvTranspose1d(2 * n_filter, 1 * n_filter, kernel_size=2, stride=2)
        self.decode7 = self.conv(2 * n_filter, 1 * n_filter)
        self.decode8 = self.conv(1 * n_filter, 1 * n_filter)
        self.decode9 = self.conv(1 * n_filter, 1)
        self.final = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, padding=0),
        )

    def conv(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        block = nn.Sequential(
            nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout)
        )
        return block

    def concat(self, x1, x2):
        if x1.shape == x2.shape:
            return torch.cat((x1, x2), 1)
        else:
            print(x1.shape, x2.shape)
            raise ValueError('concatenation failed: wrong dimensions')

    def forward(self, x):
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        m1 = self.maxpool1(e2)
        e3 = self.encode3(m1)
        e4 = self.encode4(e3)
        m2 = self.maxpool2(e4)
        e5 = self.encode5(m2)
        e6 = self.encode6(e5)
        m3 = self.maxpool3(e6)
        e7 = self.encode7(m3)
        e8 = self.encode8(e7)
        m4 = self.maxpool4(e8)

        mid1 = self.middle_conv1(m4)
        mid2 = self.middle_conv2(mid1)

        u1 = self.up1(mid2)
        c1 = self.concat(u1, e7)
        d1 = self.decode1(c1)
        d2 = self.decode2(d1)
        u2 = self.up2(d2)
        c2 = self.concat(u2, e5)
        d3 = self.decode3(c2)
        d4 = self.decode4(d3)
        u3 = self.up3(d4)
        c3 = self.concat(u3, e3)
        d5 = self.decode5(c3)
        d6 = self.decode6(d5)
        u4 = self.up4(d6)
        c4 = self.concat(u4, e1)
        d7 = self.decode7(c4)
        d8 = self.decode8(d7)
        d9 = self.decode9(d8)
        logits = self.final(d9)
        return torch.sigmoid(logits), logits


class Trainer:
    def __init__(self, dataset, num_epochs, network=Unet, batch_size=16, lr=1e-3, n_filter=64, val_split=0.2,
                 save_dir='./', save_name='model.pth', save_iter=False, loss_function='BCEDice',
                 loss_params=(1, 1)):
        """
        Class for training of neural network. Creates trainer object, training is started with .start().

        Parameters
        ----------
        dataset
            Training data, object of PyTorch Dataset class
        num_epochs : int
            Number of training epochs
        network
            Network class (Default Unet)
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        n_filter : int
            Number of convolutional filters in first layer
        val_split : float
            Validation split
        save_dir : str
            Path of directory to save trained networks
        save_name : str
            Base name for saving trained networks
        save_iter : bool
            If True, network state is save after each epoch
        load_weights : str, optional
            If not None, network state is loaded before training
        loss_function : str
            Loss function ('BCEDice', 'Tversky' or 'logcoshTversky')
        loss_params : Tuple[float, float]
            Parameter of loss function, depends on chosen loss function
        """
        self.network = network
        self.model = network(n_filter=n_filter).to(device)
        self.data = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.best_loss = torch.tensor(float('inf'))
        self.save_iter = save_iter
        self.loss_function = loss_function
        self.loss_params = loss_params
        self.n_filter = n_filter
        # split training and validation data
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        self.dim = dataset.input_len
        train_data, val_data = random_split(dataset, [num_train, num_val])
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        if loss_function == 'BCEDice':
            self.criterion = BCEDiceLoss(loss_params[0], loss_params[1])
        elif loss_function == 'Tversky':
            self.criterion = TverskyLoss(loss_params[0], loss_params[1])
        elif loss_function == 'logcoshTversky':
            self.criterion = logcoshTverskyLoss(loss_params[0], loss_params[1])
        else:
            raise ValueError(f'Loss "{loss_function}" not defined!')
        self.val_criterion = SoftDiceLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, factor=0.1)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = save_name

    def __iterate(self, epoch, mode):
        if mode == 'train':
            print('\nStarting training epoch %s ...' % epoch)
            for i, batch_i in enumerate(self.train_loader):
                x_i = batch_i['input'].view(self.batch_size, 1, self.dim).to(device)
                y_i = batch_i['gt'].view(self.batch_size, 1, self.dim).to(device)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred, y_logits = self.model(x_i)

                # Compute and print loss
                loss = self.criterion(y_logits, y_i)

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss

        elif mode == 'val':
            loss_list = []
            print('\nStarting validation epoch %s ...' % epoch)
            with torch.no_grad():
                for i, batch_i in enumerate(self.val_loader):
                    x_i = batch_i['input'].view(self.batch_size, 1, self.dim).to(device)
                    y_i = batch_i['gt'].view(self.batch_size, 1, self.dim).to(device)
                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred, y_logits = self.model(x_i)
                    loss = self.val_criterion(y_logits, y_i)
                    loss_list.append(loss.detach())
            val_loss = torch.stack(loss_list).mean()
            return val_loss

    def start(self):
        """
        Start network training. Optional: predict small test sample after each epoch.
        """
        train_loss = []
        val_loss = []
        for epoch in range(self.num_epochs):
            train_loss_i = self.__iterate(epoch, 'train')
            train_loss.append(train_loss_i)
            self.state = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr': self.lr,
                'loss_function': self.loss_function,
                'loss_params': self.loss_params,
                'n_filter': self.n_filter,
                'batch_size': self.batch_size,
                'augmentation': self.data.aug_factor,
                'noise_amp': self.data.noise_amp,
                'random_offset': self.data.random_offset,
                'random_drift': self.data.random_drift,
                'random_outlier': self.data.random_outlier,
                'random_subsampling': self.data.random_subsampling,
                'random_swap': self.data.random_swap,
            }
            with torch.no_grad():
                val_loss_i = self.__iterate(epoch, 'val')
                val_loss.append(val_loss_i)
                self.scheduler.step(val_loss_i)
            if val_loss_i < self.best_loss:
                print('\nValidation loss improved from %s to %s - saving model state' % (
                    round(self.best_loss.item(), 5), round(val_loss_i.item(), 5)))
                self.state['best_loss'] = self.best_loss = val_loss_i
                torch.save(self.state, self.save_dir + '/' + self.save_name)
            if self.save_iter:
                torch.save(self.state, self.save_dir + '/' + f'model_epoch_{epoch}.pth')


def predict_contractions(data, weights, standard_normalizer=False):
    """predict contractions from contraction time-series with CNN

    Parameters
    ----------
    data : ndarray
        1D array with time-series of contraction
    weights : str
        trained model weights (.pth file)
    standard_normalizer : bool, ndarray
        If False, each data is normalized by its mean and std. If True, the mean and std from the training data set are
        applied. If ndarray, the data is normalized with the entered args [[input_mean, input_std], [vel_mean, vel_std]]
    """
    # load model state
    state_dict = torch.load(weights, map_location=device)
    # initiate model
    model = Unet(state_dict['n_filter']).to(device)
    model.load_state_dict(state_dict['state_dict'])
    # resize data
    len_data = data.shape[0]
    contr = np.pad(data, (0, 32 - np.mod(len_data, 32)), mode='reflect')
    # normalize data
    if not standard_normalizer:
        contr -= np.mean(contr[contr != 0])
        contr /= np.std(contr[contr != 0])
    elif standard_normalizer:
        contr -= state_dict['standard_normalizer'][0, 0]
        contr /= state_dict['standard_normalizer'][0, 1]
    elif isinstance(standard_normalizer, np.ndarray):
        contr -= standard_normalizer[0, 0]
        contr /= standard_normalizer[0, 1]
    else:
        raise ValueError('standard_normalizer not valid, see docstring')
    # convert to torch
    data = torch.from_numpy(contr.astype('float32')).view(1, 1, -1).to(device)
    # predict data
    res = model(data)[0][0, 0]
    # resize data and convert to numpy
    res = res.detach().cpu().numpy()[:len_data]

    return res


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + self.beta * self.dice(logits, targets)


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class DiceLossVal(nn.Module):
    def __init__(self, smooth=0):
        super(DiceLossVal, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))


class logcoshDiceLoss(nn.Module):
    def __init__(self):
        super(logcoshDiceLoss, self).__init__()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        x = self.dice(logits, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class logcoshTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(logcoshTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return torch.log(torch.cosh(1 - Tversky))


def simulate_training_data(folder, input_len=512, n=100, freq_range=(0.04, 0.25), prob_zeros=0.5,
                           clip_thrs=(-0.75, 0.75), random_drift_amp_range=(0.005, 0.02),
                           random_drift_freq_range=(0, 0.05), noise_amp_range=(0, 0.25), plot=False):
    os.makedirs(folder, exist_ok=True)

    for i in range(n):

        freq = np.random.uniform(freq_range[0], freq_range[1])
        x_range = np.arange(input_len)
        amp_mod = 1 + np.abs(np.cos(x_range * np.random.uniform(0.01, 0.02)))
        if np.random.binomial(1, prob_zeros):
            freq = 0

        y_sim = np.clip(np.cos(x_range * freq), None, np.random.uniform(clip_thrs[0], clip_thrs[1]))
        y_sim -= np.max(y_sim)
        y_sim = amp_mod * y_sim

        # calculate contraction
        y_contraction = np.zeros_like(y_sim)
        y_contraction[y_sim != 0] = 1

        # add random drift
        random_drift = (np.random.uniform(random_drift_amp_range[0], random_drift_amp_range[1]),
                        np.random.uniform(random_drift_freq_range[0], random_drift_freq_range[1]))
        y_sim += random_drift[0] * np.cos(random_drift[1] * x_range)

        # add normal noise
        y_sim += np.random.normal(0, np.random.uniform(noise_amp_range[0], noise_amp_range[1]), size=input_len)

        if plot:
            plt.figure()
            plt.plot(x_range, y_sim)
            plt.plot(x_range, y_contraction)
            plt.show()

        np.savetxt(folder + f'simulated_{i}.txt', y_sim)
        np.savetxt(folder + f'simulated_{i}_contraction.txt', y_contraction)


def plot_selection_training_data(dataset, n_sample):
    selection = dataset.data[np.random.choice(dataset.data.shape[0], n_sample)]
    fig, axs = plt.subplots(figsize=(width_1p5cols, 2 * n_sample), nrows=n_sample)

    for i, d_i in enumerate(selection):
        axs[i].plot(d_i[0], c='k')
        axs[i].plot(d_i[1], c='r')

    plt.show()
