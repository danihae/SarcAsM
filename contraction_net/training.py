import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .losses import BCEDiceLoss, SmoothnessLoss
from .contraction_net import ContractionNet
from .utils import get_device

# select device
device = get_device()


class Trainer:
    """
    Class for training of ContractionNet. Creates Trainer object.


    Parameters
    ----------
    dataset
        Training data, object of PyTorch Dataset class
    num_epochs : int
        Number of training epochs
    network
        Network class (Default Unet)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
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
    def __init__(self, dataset, num_epochs, network=ContractionNet, in_channels=1, out_channels=2,
                 batch_size=16, lr=1e-3, n_filter=64, val_split=0.2,
                 save_dir='./', save_name='model.pt', save_iter=False, loss_function='BCEDice',
                 loss_params=(1, 1)):

        self.network = network
        self.model = network(n_filter=n_filter, in_channels=in_channels, out_channels=out_channels).to(device)
        self.data = dataset
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        self.train_data, self.val_data = random_split(dataset, [num_train, num_val])
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True, drop_last=True)
        if loss_function == 'BCEDice':
            self.criterion = BCEDiceLoss(loss_params)
        else:
            raise ValueError(f'Loss "{loss_function}" not defined!')
        self.smooth_loss = SmoothnessLoss(alpha=0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, factor=0.1)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = save_name

    def __iterate(self, epoch, mode):
        if mode == 'train':
            print('\nStarting training epoch %s ...' % epoch)
            for i, batch_i in tqdm(enumerate(self.train_loader), total=len(self.train_loader), unit='batch'):
                x_i = batch_i['input'].view(self.batch_size, self.in_channels, self.dim).to(device)
                y_i = batch_i['target'].view(self.batch_size, 1, self.dim).to(device)
                d_i = batch_i['distance'].view(self.batch_size, 1, self.dim).to(device)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred, y_logits = self.model(x_i)
                # Split the tensor into 2 chunks along the second dimension
                y_1, y_2 = torch.chunk(y_logits, chunks=2, dim=1)
                # Compute loss
                contr_loss = self.criterion(y_1, y_i)
                dist_loss = self.criterion(y_2, d_i)
                smooth_loss = self.smooth_loss(y_2)
                loss = contr_loss + dist_loss
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
                    x_i = batch_i['input'].view(self.batch_size, self.in_channels, self.dim).to(device)
                    y_i = batch_i['target'].view(self.batch_size, 1, self.dim).to(device)
                    d_i = batch_i['distance'].view(self.batch_size, 1, self.dim).to(device)
                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred, y_logits = self.model(x_i)

                    # Compute loss
                    loss = self.criterion(y_logits[:, 0], y_i[:, 0]) + self.criterion(y_logits[:, 1], d_i[:, 0])
                    loss_list.append(loss.detach())
            val_loss = torch.stack(loss_list).mean()
            return val_loss

    def start(self):
        """
        Start network training.
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
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
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
                torch.save(self.state, self.save_dir + '/' + f'model_epoch_{epoch}.pt')

