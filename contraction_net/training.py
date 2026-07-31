import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .contraction_net import ContractionNet
from .utils import get_device

# select device
device = get_device()


class Trainer:
    """
    Trainer for :class:`~contraction_net.contraction_net.ContractionNet`.

    Four choices keep the validation signal meaningful:

    - **The split is by source recording, not by sample.** The annotated traces come from
      far fewer recordings than there are traces -- dozens of rows per cell -- so a random
      split over samples puts the same cell on both sides and validation loss collapses
      towards training loss. Holding out whole recordings is the only way the number means
      anything.
    - **Batches are shuffled every epoch**, so batch composition is not a fixed artefact of
      file order.
    - **Losses are epoch means, and model selection uses validation IoU** -- the quantity
      actually cared about -- rather than a single batch's loss.
    - **Checkpoints are self-describing:** architecture and tuned decision threshold are
      stored alongside the weights, so inference configures itself from the file.

    Parameters
    ----------
    dataset : ContractionDataset
        Training data; must expose a ``groups`` list.
    num_epochs : int
        Number of training epochs.
    network : type, optional
        Network class. Default :class:`ContractionNet`.
    batch_size, lr, n_filter : optional
        Standard training knobs.
    val_split : float, optional
        Fraction of *recordings* (not samples) held out. Default is 0.2.
    save_dir, save_name : str, optional
        Where to write the best checkpoint.
    seed : int, optional
        Seed for the split and for shuffling. Default is 0.
    """

    def __init__(self, dataset, num_epochs, network=ContractionNet, batch_size=32,
                 lr=2e-3, n_filter=64, val_split=0.2, save_dir='./',
                 save_name='model_ContractionNet.pt', seed=0, num_workers=0):
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
