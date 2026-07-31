import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .losses import BCEDiceLoss, SmoothnessLoss
from .contraction_net import ContractionNet, ContractionNetV2
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



class TrainerV2:
    """
    Trainer for :class:`~contraction_net.contraction_net.ContractionNetV2`.

    Fixes four problems in :class:`Trainer` that made its validation signal unreliable:

    - **The split is by source recording, not by sample.** :class:`Trainer` calls
      ``random_split`` *after* 10x augmentation, so ten near-copies of one trace land on
      both sides; and even without augmentation the 2002 real traces come from only 165
      recordings, dozens of rows per cell. Validation loss was therefore close to training
      loss, and it is what selected the bundled checkpoint.
    - **Batches are shuffled every epoch.** :class:`Trainer` never passes ``shuffle=True``,
      so every epoch sees the same fixed batch composition.
    - **Losses are epoch means and the metric is IoU**, not the last batch's loss. Model
      selection uses validation IoU, the quantity actually cared about.
    - **The smoothness term is either used or absent.** :class:`Trainer` computes one and
      then never adds it to the loss.

    Parameters
    ----------
    dataset : ContractionDataset
        Training data; must expose a ``groups`` list.
    num_epochs : int
        Number of training epochs.
    network : type, optional
        Network class. Default :class:`ContractionNetV2`.
    batch_size, lr, n_filter : optional
        Standard training knobs.
    val_split : float, optional
        Fraction of *recordings* (not samples) held out. Default is 0.2.
    save_dir, save_name : str, optional
        Where to write the best checkpoint.
    seed : int, optional
        Seed for the split and for shuffling. Default is 0.
    """

    def __init__(self, dataset, num_epochs, network=ContractionNetV2, batch_size=32,
                 lr=2e-3, n_filter=64, val_split=0.2, save_dir='./',
                 save_name='model_ContractionNetV2.pt', seed=0, num_workers=0):
        from torch.utils.data import DataLoader, Subset

        from .losses import MaskBoundaryLoss

        self.dataset = dataset
        self.network = network
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_filter = n_filter
        self.save_dir = save_dir
        self.save_name = save_name
        self.in_channels, self.out_channels = 2, 3

        self.model = network(n_filter=n_filter, in_channels=self.in_channels,
                             out_channels=self.out_channels).to(device)

        train_idx, val_idx = self._group_split(dataset.groups, val_split, seed)
        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                                       shuffle=True, drop_last=True, num_workers=num_workers)
        self.val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                                     shuffle=False, drop_last=False, num_workers=num_workers)
        self.n_train, self.n_val = len(train_idx), len(val_idx)

        self.criterion = MaskBoundaryLoss().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=lr, epochs=num_epochs,
            steps_per_epoch=max(1, len(self.train_loader)))
        self.best_iou = -1.0
        self.recommended_threshold = 0.5
        self.history = []
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _group_split(groups, val_split, seed):
        """Split indices so that no source recording appears on both sides."""
        import numpy as np

        unique = sorted(set(groups))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        n_val = max(1, int(round(len(unique) * val_split)))
        val_groups = set(unique[:n_val])
        train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
        val_idx = [i for i, g in enumerate(groups) if g in val_groups]
        return train_idx, val_idx

    def _run_epoch(self, loader, train):
        from .losses import iou_score

        self.model.train(train)
        total_loss, total_iou, n_batches = 0.0, 0.0, 0
        for batch in loader:
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            with torch.set_grad_enabled(train):
                _, logits = self.model(x)
                loss, _ = self.criterion(logits, y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            total_loss += loss.detach().item()
            total_iou += float(iou_score(logits.detach(), y))
            n_batches += 1
        n_batches = max(n_batches, 1)
        return total_loss / n_batches, total_iou / n_batches

    def start(self, verbose=True):
        """Train, keeping the checkpoint with the best validation IoU."""
        if verbose:
            print(f'train samples {self.n_train}, val samples {self.n_val} '
                  f'(split by recording)')
        for epoch in range(self.num_epochs):
            train_loss, train_iou = self._run_epoch(self.train_loader, True)
            with torch.no_grad():
                val_loss, val_iou = self._run_epoch(self.val_loader, False)
            self.history.append({'epoch': epoch, 'train_loss': train_loss,
                                 'train_iou': train_iou, 'val_loss': val_loss,
                                 'val_iou': val_iou})
            improved = val_iou > self.best_iou
            if improved:
                self.best_iou = val_iou
                torch.save(self._state(epoch), os.path.join(self.save_dir, self.save_name))
            if verbose:
                print(f'epoch {epoch:3d}  train loss {train_loss:.4f} IoU {train_iou:.3f}  '
                      f'| val loss {val_loss:.4f} IoU {val_iou:.3f}'
                      + ('  <- saved' if improved else ''))
        return self.history

    def _state(self, epoch):
        return {
            'epoch': epoch,
            # read back by prediction.predict_contractions to pick the architecture and the
            # input convention, so a checkpoint is self-describing
            'arch': self.network.__name__,
            'input_norm': getattr(self.network, 'input_norm', 'robust'),
            # operating point, re-tuned per model: the value that was right for the old
            # architecture is not right for this one, and callers should not have to know
            'recommended_threshold': self.recommended_threshold,
            'state_dict': self.model.state_dict(),
            'n_filter': self.n_filter,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'best_val_iou': self.best_iou,
            'history': self.history,
            'noise_params': getattr(self.dataset, 'noise_params', None),
        }
