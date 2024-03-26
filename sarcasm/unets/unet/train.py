import glob
import os

import torch.optim as optim
from barbar import Bar
from torch.utils.data import DataLoader, random_split

from .losses import *
from .predict import Predict
from .unet import Unet
from .baby_unet import BabyUnet

# select device
if torch.has_cuda:
    device = torch.device('cuda:0')
elif hasattr(torch, 'has_mps'):  # only for apple m1/m2/...
    if torch.has_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cpu')


class Trainer:
    def __init__(self, dataset, num_epochs, network=Unet, batch_size=4, lr=1e-3, n_filter=64, val_split=0.2,
                 save_dir='./', save_name='model.pth', save_iter=False, load_weights=False, loss_function='BCEDice',
                 loss_params=(0.5, 0.5)):
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
        self.dim = dataset.dim_out
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, factor=0.1)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_name = save_name
        if load_weights:
            self.state = torch.load(self.save_dir + '/' + self.save_name)
            self.model.load_state_dict(self.state['state_dict'])

    def __iterate(self, epoch, mode):
        if mode == 'train':
            print('\nStarting training epoch %s ...' % epoch)
            for i, batch_i in enumerate(Bar(self.train_loader)):
                x_i = batch_i['image'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                y_i = batch_i['mask'].view(self.batch_size, 1, self.dim[0], self.dim[0]).to(device)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred, y_logits = self.model(x_i)

                # Compute and print loss
                loss = self.criterion(y_logits, y_i)

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        elif mode == 'val':
            loss_list = []
            print('\nStarting validation epoch %s ...' % epoch)
            with torch.no_grad():
                for i, batch_i in enumerate(Bar(self.val_loader)):
                    x_i = batch_i['image'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                    y_i = batch_i['mask'].view(self.batch_size, 1, self.dim[0], self.dim[1]).to(device)
                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred, y_logits = self.model(x_i)
                    loss = self.criterion(y_logits, y_i)
                    loss_list.append(loss.detach())
            val_loss = torch.stack(loss_list).mean()
            return val_loss

        torch.cuda.empty_cache()

    def start(self, test_data_path=None, result_path=None, test_resize_dim=(512, 512)):
        """
        Start network training. Optional: predict small test sample after each epoch.

        Parameters
        ----------
        test_data_path : str
            Path of folder with test tif files
        result_path : str
            Path for saving prediction results of test data
        test_resize_dim: tuple
            Resize dimensions for prediction of test movies
        """
        for epoch in range(self.num_epochs):
            self.__iterate(epoch, 'train')
            self.state = {
                'epoch': epoch,
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr': self.lr,
                'loss_function': self.loss_function,
                'loss_params': self.loss_params,
                'n_filter': self.n_filter,
                'augmentation': self.data.aug_factor,
                'clip_threshold': self.data.clip_threshold,
                'noise_amp': self.data.noise_amp,
                'brightness_contrast': self.data.brightness_contrast,
                'shiftscalerotate': self.data.shiftscalerotate,
            }
            with torch.no_grad():
                val_loss = self.__iterate(epoch, 'val')
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print('\nValidation loss improved from %s to %s - saving model state' % (
                    round(self.best_loss.item(), 5), round(val_loss.item(), 5)))
                self.state['best_loss'] = self.best_loss = val_loss
                torch.save(self.state, self.save_dir + '/' + self.save_name)
            if self.save_iter:
                torch.save(self.state, self.save_dir + '/' + f'model_epoch_{epoch}.pth')

            if test_data_path is not None:
                print('Predicting test data...')
                files = glob.glob(test_data_path + '*.tif')
                for i, file in enumerate(files):
                    Predict(file, result_path + os.path.basename(file) + f'epoch_{epoch}.tif',
                            self.save_dir + '/' + f'model_epoch_{epoch}.pth', self.network, resize_dim=test_resize_dim,
                            invert=False)
