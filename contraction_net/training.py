import copy
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .contraction_net import ContractionNet
from .utils import get_device

# select device
device = get_device()

#: Probability thresholds swept when selecting a model and tuning its operating point.
DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 3)


class Trainer:
    """
    Trainer for :class:`~contraction_net.contraction_net.ContractionNet`.

    Five choices keep the reported numbers meaningful:

    - **The split is by source recording, and stratified by data pool.** Annotated traces
      come from far fewer recordings than there are traces, so a random split over samples
      puts one cell on both sides. Splitting on groups alone is not enough either: with
      thousands of simulated traces the held-out set becomes almost entirely simulated and
      stops measuring real data, so the split is taken *within* each pool.
    - **Held-out recordings are split again into ``val`` and ``cal``.** Epochs are selected
      on ``val`` and the decision threshold is fitted on ``cal``; tuning the operating point
      on the same data that chose the epoch would overfit it.
    - **Selection sweeps the threshold** and uses the pool-weighted mean IoU at its best
      value, so a better model is not rejected for being calibrated differently, and
      simulated traces cannot outvote real ones.
    - Batches are shuffled every epoch, and the dataset's augmentation stream is advanced
      per epoch.
    - **Checkpoints are self-describing:** architecture, input convention and tuned
      threshold travel with the weights, so inference configures itself from the file.

    Parameters
    ----------
    dataset : ContractionDataset
        Training data; must expose ``groups`` and ``pools``.
    num_epochs : int
        Number of training epochs.
    network : type, optional
        Network class. Default :class:`ContractionNet`.
    batch_size, lr, n_filter : optional
        Standard training knobs.
    in_channels, out_channels : int, optional
        Input and output channel counts. Defaults 2 and 3.
    val_split : float, optional
        Fraction of recordings held out of training, per pool. Default is 0.2.
    cal_split : float, optional
        Fraction of the held-out recordings used to fit the threshold. Default is 0.5.
    save_dir, save_name : str, optional
        Where to write the best checkpoint.
    seed : int, optional
        Seed for the split and for shuffling. Default is 0.
    num_workers : int, optional
        DataLoader workers. Default is 4.
    thresholds : sequence of float or None, optional
        Probability grid to sweep. Default :data:`DEFAULT_THRESHOLDS`.
    pool_weights : dict or None, optional
        Weight per pool in the selection metric. Default weights every present pool equally.
    arch_kwargs : dict or None, optional
        Extra keyword arguments for the network constructor, stored in the checkpoint.
    """

    def __init__(self, dataset, num_epochs, network=ContractionNet, batch_size=32,
                 lr=2e-3, n_filter=64, in_channels=2, out_channels=3, val_split=0.2,
                 cal_split=0.5, save_dir='./', save_name='model_ContractionNet.pt',
                 seed=0, num_workers=4, thresholds=None, pool_weights=None,
                 arch_kwargs=None):
        from .losses import MaskBoundaryLoss

        self.dataset = dataset
        self.network = network
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_filter = n_filter
        self.save_dir = save_dir
        self.save_name = save_name
        self.seed = seed
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.arch_kwargs = dict(arch_kwargs or {})
        self.thresholds = np.asarray(DEFAULT_THRESHOLDS if thresholds is None
                                     else thresholds, dtype=float)
        self.pool_weights = pool_weights

        self.model = network(n_filter=n_filter, in_channels=self.in_channels,
                             out_channels=self.out_channels, **self.arch_kwargs).to(device)

        pools = getattr(dataset, 'pools', None)
        train_idx, val_idx, cal_idx = self._group_split(dataset.groups, val_split,
                                                        cal_split, seed, pools)
        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                                       shuffle=True, drop_last=True, num_workers=num_workers)
        self.val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
        self.cal_loader = DataLoader(Subset(dataset, cal_idx), batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
        self.n_train, self.n_val, self.n_cal = len(train_idx), len(val_idx), len(cal_idx)

        self.criterion = MaskBoundaryLoss().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=lr, epochs=num_epochs,
            steps_per_epoch=max(1, len(self.train_loader)))
        self.best_score = -1.0
        self.best_state = None
        self.recommended_threshold = 0.5
        self.history = []
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _group_split(groups, val_split, cal_split, seed, pools=None):
        """Split indices by group, stratified over data pools.

        Returns ``(train, val, cal)``; no source recording appears in more than one.
        """
        rng = np.random.default_rng(seed)
        pools = ['all'] * len(groups) if pools is None else list(pools)
        by_pool = {}
        for i, (group, pool) in enumerate(zip(groups, pools)):
            by_pool.setdefault(pool, {}).setdefault(group, []).append(i)

        train, val, cal = [], [], []
        for pool, mapping in sorted(by_pool.items()):
            unique = sorted(mapping)
            rng.shuffle(unique)
            n_hold = min(len(unique) - 1, max(1, int(round(len(unique) * val_split)))) \
                if len(unique) > 1 else 0
            held = unique[:n_hold]
            n_val = max(1, int(round(len(held) * (1 - cal_split)))) if held else 0
            for group in unique[n_hold:]:
                train += mapping[group]
            for group in held[:n_val]:
                val += mapping[group]
            for group in held[n_val:]:
                cal += mapping[group]
        # with a single held-out recording there is nothing left to calibrate on
        if not cal:
            cal = list(val)
        return train, val, cal

    def _train_epoch(self):
        from .losses import iou_score

        self.model.train(True)
        total_loss, total_iou, n_batches = 0.0, 0.0, 0
        for batch in self.train_loader:
            # in_channels < 2 drops the difference channel, keeping the level
            x = batch['input'][:, :self.in_channels].to(device)
            y = batch['target'].to(device)
            _, logits = self.model(x)
            loss, _ = self.criterion(logits, y)
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

    @torch.no_grad()
    def _evaluate(self, loader):
        """Loss and per-pool IoU across the threshold grid.

        Returns
        -------
        dict
            ``loss``, ``iou_by_pool`` mapping pool to an array over thresholds, and
            ``counts`` per pool.
        """
        from .losses import iou_at_thresholds

        self.model.train(False)
        sums, counts = {}, {}
        duty_pred = np.zeros(len(self.thresholds))
        duty_true, n_frames = 0.0, 0
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            # in_channels < 2 drops the difference channel, keeping the level
            x = batch['input'][:, :self.in_channels].to(device)
            y = batch['target'].to(device)
            _, logits = self.model(x)
            loss, _ = self.criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits[:, 0])
            for j, threshold in enumerate(self.thresholds):
                duty_pred[j] += float((probs > float(threshold)).sum())
            duty_true += float((y[:, 0] > 0.5).sum())
            n_frames += int(y[:, 0].numel())
            per_sample = iou_at_thresholds(logits, y, self.thresholds).cpu().numpy()
            for pool in set(batch['pool']):
                sel = np.asarray([p == pool for p in batch['pool']])
                sums[pool] = sums.get(pool, 0.0) + per_sample[:, sel].sum(axis=1)
                counts[pool] = counts.get(pool, 0) + int(sel.sum())
        iou_by_pool = {p: sums[p] / max(counts[p], 1) for p in sums}
        n_frames = max(n_frames, 1)
        return {'loss': total_loss / max(n_batches, 1), 'iou_by_pool': iou_by_pool,
                'counts': counts, 'duty_pred': duty_pred / n_frames,
                'duty_true': duty_true / n_frames}

    def _weighted(self, iou_by_pool):
        """Pool-weighted IoU across the threshold grid."""
        if not iou_by_pool:
            return np.zeros_like(self.thresholds)
        weights = self.pool_weights or {p: 1.0 for p in iou_by_pool}
        total = sum(weights.get(p, 0.0) for p in iou_by_pool) or 1.0
        return sum(iou_by_pool[p] * weights.get(p, 0.0) for p in iou_by_pool) / total

    def start(self, verbose=True):
        """Train, then fit the decision threshold on the calibration recordings."""
        if verbose:
            print(f'train {self.n_train}, val {self.n_val}, cal {self.n_cal} samples '
                  f'(split by recording, stratified by pool)')
        for epoch in range(self.num_epochs):
            self.dataset.set_epoch(epoch) if hasattr(self.dataset, 'set_epoch') else None
            train_loss, train_iou = self._train_epoch()
            val = self._evaluate(self.val_loader)
            curve = self._weighted(val['iou_by_pool'])
            best_at = int(np.argmax(curve))
            score = float(curve[best_at])

            entry = {'epoch': epoch, 'train_loss': train_loss, 'train_iou': train_iou,
                     'val_loss': val['loss'], 'val_iou': score,
                     'val_threshold': float(self.thresholds[best_at]),
                     'val_iou_by_pool': {p: float(v[best_at])
                                         for p, v in val['iou_by_pool'].items()}}
            self.history.append(entry)

            improved = score > self.best_score
            if improved:
                self.best_score = score
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.recommended_threshold = float(self.thresholds[best_at])
                torch.save(self._state(epoch), os.path.join(self.save_dir, self.save_name))
            if verbose:
                pools = '  '.join(f'{p} {v:.3f}' for p, v in
                                  sorted(entry['val_iou_by_pool'].items()))
                print(f'epoch {epoch:3d}  train loss {train_loss:.4f} IoU {train_iou:.3f}'
                      f'  | val loss {val["loss"]:.4f} IoU {score:.3f}'
                      f' @thr {entry["val_threshold"]:.2f}  [{pools}]'
                      + ('  <- saved' if improved else ''))

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.recommended_threshold = self.tune_threshold()
        torch.save(self._state(self.num_epochs - 1),
                   os.path.join(self.save_dir, self.save_name))
        if verbose:
            print(f'threshold tuned on held-out calibration recordings: '
                  f'{self.recommended_threshold:.2f}')
        return self.history

    def tune_threshold(self, iou_tolerance=0.005):
        """Best threshold on the calibration recordings, which never chose an epoch.

        IoU is nearly flat over a wide band of thresholds and barely registers a few per
        cent of bias in how long each contraction is called, so maximising it alone picks
        a point in that band arbitrarily -- and the point it picked ran contractions 3.7%
        short on real traces, which propagates straight into time_contr, time_to_peak and
        time_to_relax downstream. Among thresholds within ``iou_tolerance`` of the best,
        this takes the one whose predicted duty matches the target.
        """
        cal = self._evaluate(self.cal_loader)
        curve = self._weighted(cal['iou_by_pool'])
        if not np.isfinite(curve).any():
            return float(self.recommended_threshold)
        best = float(np.max(curve))
        self.cal_iou = best
        near = np.flatnonzero(curve >= best - float(iou_tolerance))
        target = cal.get('duty_true', 0.0) or 1.0
        bias = np.abs(cal['duty_pred'][near] / target - 1.0)
        chosen = int(near[int(np.argmin(bias))])
        self.cal_duty_bias = float(cal['duty_pred'][chosen] / target - 1.0)
        return float(self.thresholds[chosen])

    def _state(self, epoch):
        return {
            'epoch': epoch,
            # read back by prediction.predict_contractions to pick the architecture and
            # the input convention, so a checkpoint is self-describing
            'arch': self.network.__name__,
            'arch_kwargs': self.arch_kwargs,
            'input_convention': getattr(self.dataset, 'input_convention', 'q90'),
            'crop_len': getattr(self.dataset, 'crop_len', None),
            'recommended_threshold': self.recommended_threshold,
            'state_dict': self.model.state_dict(),
            'n_filter': self.n_filter,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'best_val_iou': self.best_score,
            'val_iou_by_pool': self.history[-1]['val_iou_by_pool'] if self.history else {},
            'history': self.history,
            'pool_weights': self.pool_weights,
            'noise_params': getattr(self.dataset, 'noise_params', None),
        }
