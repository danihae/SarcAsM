import copy
import dataclasses
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .contraction_net import ContractionNet
from .data import SIM_DEV_GROUP
from .polarity import PolarityRule
from .utils import get_device

# select device
device = get_device()

#: Probability thresholds swept when selecting a model and tuning its operating point.
DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 3)

#: How :class:`Trainer` chooses the weights it keeps.
SELECT = ('best', 'last')

#: Confidence peak the shared decoder starts from: a run above the threshold counts only if
#: it reaches this probability somewhere. Stored with the threshold as the checkpoint's
#: ``decoding``, where a peak tuned on held-out recordings can replace it.
DECODING_PEAK = 0.85

#: Polarity rule the inference path applies to a rest-high (``'q90'``) checkpoint before
#: the network reads a trace: windows in seconds, the quantile of the skewness, the margin
#: within which it abstains and the shortest trace it decides. Stored in the checkpoint so
#: the rule travels with the weights it was validated with.
POLARITY_RULE = dataclasses.asdict(PolarityRule())

#: Entries of a split file per pool; :meth:`Trainer._file_split` reads no others.
SPLIT_KEYS = {'drug': 'corpus', 'eht': 'eht'}


def seed_everything(seed, deterministic=False):
    """Seed Python, NumPy and PyTorch (CPU and every GPU) from one integer.

    Parameters
    ----------
    seed : int
        The seed.
    deterministic : bool, optional
        Also select deterministic cuDNN kernels and make PyTorch raise on an operation that
        has none. Default is False.
    """
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS needs this set before its first call to be deterministic
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


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

    Models that are compared with each other need one split and no selection noise, so two
    options replace the first three choices. With ``split``, every pool is split by a fixed
    file that no training seed changes, and its dev side serves as both ``val`` and
    ``cal``. Dev is scored as held-out traces are, without augmentation
    (:meth:`~contraction_net.data.ContractionDataset.held_out`), and its simulated share is
    simulated with the file's own seed, so no dev sample depends on the training seed.
    ``select='last'`` keeps the final epoch, only logs the validation, and tunes the
    threshold by duty match alone: in 15 runs of 60 epochs the best epoch was the 45th or
    later every time, and IoU is too flat over thresholds to choose one.

    The duty match counts every dev frame unless ``duty_pools`` names the pools to count.
    With the split the bundled model was trained on, 58% of the dev frames are simulated,
    41% corpus and 1% EHT, so the threshold follows the simulator's labels more than the
    corpus labels or the EHT annotations, which end at other levels. The checkpoint records
    the duty bias left on each pool (``duty_bias_by_pool``); ``duty_pools=('drug', 'eht')``
    matches the real dev recordings alone.

    Parameters
    ----------
    dataset : ContractionDataset
        Training data; must expose ``groups`` and ``pools``, and ``held_out`` with
        ``split``.
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
        Where to write the checkpoint ``select`` keeps.
    seed : int, optional
        Seed for the split, the weight initialisation, dropout and the batch order. Default
        is 0. The dataset's augmentation stream has its own seed
        (:class:`~contraction_net.data.ContractionDataset`), drawn per sample and epoch, so
        it does not depend on the number of workers.
    amp : bool or None, optional
        Train under bfloat16 autocast on the GPU's tensor cores, about 2.5x faster for
        this network; the loss and the validation stay in float32. None (the
        default) enables it on CUDA only.
    deterministic : bool, optional
        Also ask PyTorch for deterministic kernels, so a rerun on the same GPU reproduces
        the weights bit for bit. Slower; default is False, which reproduces everything
        except the last bits of the floating-point reductions on the GPU.
    num_workers : int, optional
        DataLoader workers. Default is 4.
    thresholds : sequence of float or None, optional
        Probability grid to sweep. Default :data:`DEFAULT_THRESHOLDS`.
    pool_weights : dict or None, optional
        Weight per pool in the selection metric. Default weights every present pool equally.
    arch_kwargs : dict or None, optional
        Extra keyword arguments for the network constructor, stored in the checkpoint.
    pool_repeat : dict or None, optional
        How often each training trace of a pool is drawn per epoch, e.g. ``{'eht': 20}``.
        A small pool of real recordings is otherwise a rounding error next to thousands
        of simulated traces. Every draw gets its own crop and augmentation, and only the
        training split is repeated.
    split : str, dict or None, optional
        A split file (JSON) or its content; see :meth:`_file_split`. The dataset must hold
        the file's simulated dev share (its ``sim_dev``) when it holds simulated traces.
        None (the default) splits by ``val_split`` and ``cal_split`` with ``seed``, as
        before.
    select : str, optional
        One of :data:`SELECT`. ``'best'`` (default) keeps the epoch with the best
        pool-weighted validation IoU and tunes the threshold by duty match among the
        thresholds near the best calibration IoU; ``'last'`` keeps the final epoch and tunes
        the threshold by duty match alone.
    dice_weight : float, optional
        Weight of the Dice term in
        :class:`~contraction_net.losses.MaskBoundaryLoss`. Default is 1.0.
    duty_pools : sequence of str or None, optional
        Pools whose calibration frames the threshold's duty match counts. None (the
        default), or none of them present, counts every pool, as before.
    """

    def __init__(self, dataset, num_epochs, network=ContractionNet, batch_size=32,
                 lr=2e-3, n_filter=64, in_channels=2, out_channels=3, val_split=0.2,
                 cal_split=0.5, save_dir='./', save_name='model_ContractionNet.pt',
                 seed=0, num_workers=4, thresholds=None, pool_weights=None,
                 arch_kwargs=None, pool_repeat=None, deterministic=False, amp=None,
                 split=None, select='best', dice_weight=1.0, duty_pools=None):
        from .losses import MaskBoundaryLoss

        if select not in SELECT:
            raise ValueError(f'select must be one of {SELECT}, got {select!r}')
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
        self.select = select
        self.duty_pools = None if duty_pools is None else list(duty_pools)
        self.duty_bias_by_pool = None
        if isinstance(split, (str, os.PathLike)):
            with open(split) as handle:
                split = json.load(handle)
        self.split = split

        # Seed before the network is built, so the initial weights -- and with them every
        # result -- follow from the seed; the split alone was seeded before.
        self.deterministic = bool(deterministic)
        seed_everything(seed, deterministic=self.deterministic)
        self.amp = device.type == 'cuda' and (amp is None or bool(amp))
        if device.type == 'cuda' and not self.deterministic:
            torch.backends.cudnn.benchmark = True   # fixed crop length: tune once, reuse
        self.model = network(n_filter=n_filter, in_channels=self.in_channels,
                             out_channels=self.out_channels, **self.arch_kwargs).to(device)

        pools = getattr(dataset, 'pools', None)
        held = dataset
        if self.split is None:
            train_idx, val_idx, cal_idx = self._group_split(dataset.groups, val_split,
                                                            cal_split, seed, pools)
        else:
            if 'sim' in (pools or ()) and \
                    getattr(dataset, 'sim_dev', None) != self.split.get('sim'):
                raise ValueError("the dataset does not hold the split file's simulated dev "
                                 "share; build it with sim_dev=split['sim']")
            train_idx, val_idx = self._file_split(dataset.groups, pools, self.split)
            cal_idx = val_idx
            held = dataset.held_out()
        self.pool_repeat = dict(pool_repeat or {})
        if self.pool_repeat and pools is not None:
            train_idx = [i for i in train_idx
                         for _ in range(int(self.pool_repeat.get(pools[i], 1)))]
        # the shuffle order has its own generator, so nothing else drawing from the global
        # torch stream can change it
        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                                       shuffle=True, drop_last=True, num_workers=num_workers,
                                       generator=torch.Generator().manual_seed(int(seed)))
        self.val_loader = DataLoader(Subset(held, val_idx), batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
        self.cal_loader = DataLoader(Subset(held, cal_idx), batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
        self.n_train, self.n_val, self.n_cal = len(train_idx), len(val_idx), len(cal_idx)

        self.criterion = MaskBoundaryLoss(dice_weight=dice_weight).to(device)
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

    @staticmethod
    def _file_split(groups, pools, split):
        """Split indices by a fixed split file, which no training seed changes.

        Returns ``(train, dev)``. Corpus traces (pool ``'drug'``) are placed by well under
        ``split['corpus']`` and EHT traces by recording under ``split['eht']``, each mapping
        ``'train'``, ``'dev'`` and ``'test'`` to group names. With a ``split['sim']``
        entry, every simulated trace trains except the dev share in the group
        :data:`~contraction_net.data.SIM_DEV_GROUP`, which the dataset simulates with
        ``split['sim']['seed']`` (its ``sim_dev``). Test groups are in neither list. A group
        the file does not place is an error, so a new recording is never trained on by
        accident; entries for other sets, which are never trained on, are not read.
        """
        pools = ['all'] * len(groups) if pools is None else list(pools)
        side = {}
        for pool, key in SPLIT_KEYS.items():
            for name, members in split.get(key, {}).items():
                if name not in ('train', 'dev', 'test'):
                    raise ValueError(f'split side must be train, dev or test, got {name!r}')
                for member in members:
                    group = f'well/{member}' if pool == 'drug' else member
                    if side.setdefault((pool, group), name) != name:
                        raise ValueError(f'{group!r} is on two sides of the split')
        if 'sim' in split:
            for group in {g for g, p in zip(groups, pools) if p == 'sim'}:
                side[('sim', group)] = 'dev' if group == SIM_DEV_GROUP else 'train'

        train, dev = [], []
        for i, (group, pool) in enumerate(zip(groups, pools)):
            name = side.get((pool, group))
            if name is None:
                raise ValueError(f'the split file does not place {group!r} (pool {pool!r})')
            if name == 'train':
                train.append(i)
            elif name == 'dev':
                dev.append(i)
        return train, dev

    def _train_epoch(self):
        from .losses import iou_score

        self.model.train(True)
        total_loss, total_iou, n_batches = 0.0, 0.0, 0
        for batch in self.train_loader:
            # in_channels < 2 drops the difference channel, keeping the level
            x = batch['input'][:, :self.in_channels].to(device)
            y = batch['target'].to(device)
            # 0 on the frames that pad a short trace to the crop length
            w = batch['weight'].to(device) if 'weight' in batch else None
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=self.amp):
                _, logits = self.model(x)
            logits = logits.float()
            loss, _ = self.criterion(logits, y, w)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.detach().item()
            total_iou += float(iou_score(logits.detach(), y, weight=w))
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
            ``counts`` per pool; ``duty_pred`` (over thresholds) and ``duty_true`` over
            every frame, and ``duty_by_pool`` mapping pool to the frames called at each
            threshold, the labelled frames and the frames it holds.
        """
        from .losses import iou_at_thresholds

        self.model.train(False)
        sums, counts, duty = {}, {}, {}
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            # in_channels < 2 drops the difference channel, keeping the level
            x = batch['input'][:, :self.in_channels].to(device)
            y = batch['target'].to(device)
            w = batch['weight'].to(device) if 'weight' in batch else None
            _, logits = self.model(x)
            loss, _ = self.criterion(logits, y, w)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits[:, 0])
            # duty is counted on real frames only, not on the padding of short traces
            real = torch.ones_like(probs, dtype=torch.bool) if w is None else w > 0
            called = torch.stack([((probs > float(threshold)) & real).sum(dim=1)
                                  for threshold in self.thresholds]).cpu().numpy()
            labelled = ((y[:, 0] > 0.5) & real).sum(dim=1).cpu().numpy()
            frames = real.sum(dim=1).cpu().numpy()
            per_sample = iou_at_thresholds(logits, y, self.thresholds,
                                           weight=w).cpu().numpy()
            for pool in set(batch['pool']):
                sel = np.asarray([p == pool for p in batch['pool']])
                sums[pool] = sums.get(pool, 0.0) + per_sample[:, sel].sum(axis=1)
                counts[pool] = counts.get(pool, 0) + int(sel.sum())
                tally = duty.setdefault(pool, [np.zeros(len(self.thresholds), int), 0, 0])
                tally[0] = tally[0] + called[:, sel].sum(axis=1)
                tally[1] += int(labelled[sel].sum())
                tally[2] += int(frames[sel].sum())
        iou_by_pool = {p: sums[p] / max(counts[p], 1) for p in sums}
        n_frames = max(sum(t[2] for t in duty.values()), 1)
        return {'loss': total_loss / max(n_batches, 1), 'iou_by_pool': iou_by_pool,
                'counts': counts,
                'duty_pred': sum((t[0] for t in duty.values()),
                                 np.zeros(len(self.thresholds), int)) / n_frames,
                'duty_true': sum(t[1] for t in duty.values()) / n_frames,
                'duty_by_pool': {p: tuple(t) for p, t in duty.items()}}

    def _weighted(self, iou_by_pool):
        """Pool-weighted IoU across the threshold grid."""
        if not iou_by_pool:
            return np.zeros_like(self.thresholds)
        weights = self.pool_weights or {p: 1.0 for p in iou_by_pool}
        total = sum(weights.get(p, 0.0) for p in iou_by_pool) or 1.0
        return sum(iou_by_pool[p] * weights.get(p, 0.0) for p in iou_by_pool) / total

    def start(self, verbose=True):
        """Train, then fit the decision threshold on the calibration recordings."""
        if verbose and self.split is None:
            print(f'train {self.n_train}, val {self.n_val}, cal {self.n_cal} samples '
                  f'(split by recording, stratified by pool)')
        elif verbose:
            print(f'train {self.n_train}, dev {self.n_val} samples (fixed split file; dev '
                  f'is logged each epoch and tunes the threshold)')
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
            saved = improved and self.select == 'best'
            if improved:
                self.best_score = score
            if saved:
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.recommended_threshold = float(self.thresholds[best_at])
                torch.save(self._state(epoch), os.path.join(self.save_dir, self.save_name))
            if verbose:
                pools = '  '.join(f'{p} {v:.3f}' for p, v in
                                  sorted(entry['val_iou_by_pool'].items()))
                print(f'epoch {epoch:3d}  train loss {train_loss:.4f} IoU {train_iou:.3f}'
                      f'  | val loss {val["loss"]:.4f} IoU {score:.3f}'
                      f' @thr {entry["val_threshold"]:.2f}  [{pools}]'
                      + ('  <- saved' if saved else ''))

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.recommended_threshold = self.tune_threshold(
            iou_tolerance=None if self.select == 'last' else 0.005, pools=self.duty_pools)
        torch.save(self._state(self.num_epochs - 1),
                   os.path.join(self.save_dir, self.save_name))
        if verbose:
            where = ('held-out calibration recordings' if self.split is None
                     else 'the dev split')
            bias = '  '.join(f'{p} {b:+.3f}' for p, b in (self.duty_bias_by_pool or {}).items()
                             if b is not None)
            print(f'threshold tuned on {where}'
                  + (f" ({', '.join(self.duty_pools)})" if self.duty_pools else '')
                  + f': {self.recommended_threshold:.2f}  [duty bias {bias}]')
        return self.history

    def tune_threshold(self, iou_tolerance=0.005, pools=None):
        """Best threshold on the calibration recordings, which never chose an epoch.

        IoU is nearly flat over a wide band of thresholds and barely registers a few per
        cent of bias in how long each contraction is called, so maximising it alone picks
        a point in that band arbitrarily -- and the point it picked ran contractions 3.7%
        short on real traces, which propagates straight into time_contr, time_to_peak and
        time_to_relax downstream. Among thresholds within ``iou_tolerance`` of the best,
        this takes the one whose predicted duty matches the target; ``iou_tolerance=None``
        drops the IoU stage and takes it from the whole grid. The duty is counted over the
        frames of ``pools``, or of every pool when it is None or names none present. The
        duty bias each pool is left with at the chosen threshold is kept as
        ``duty_bias_by_pool`` (None for a pool without a labelled frame).
        """
        cal = self._evaluate(self.cal_loader)
        curve = self._weighted(cal['iou_by_pool'])
        if not np.isfinite(curve).any():
            return float(self.recommended_threshold)
        best = float(np.max(curve))
        self.cal_iou = best
        near = (np.arange(curve.size) if iou_tolerance is None
                else np.flatnonzero(curve >= best - float(iou_tolerance)))
        duty_pred, duty_true = cal['duty_pred'], cal.get('duty_true', 0.0)
        tallies = [t for p, t in cal.get('duty_by_pool', {}).items()
                   if pools is not None and p in pools]
        if tallies:
            n_frames = max(sum(t[2] for t in tallies), 1)
            duty_pred = sum(t[0] for t in tallies) / n_frames
            duty_true = sum(t[1] for t in tallies) / n_frames
        target = duty_true or 1.0
        bias = np.abs(duty_pred[near] / target - 1.0)
        chosen = int(near[int(np.argmin(bias))])
        self.cal_duty_bias = float(duty_pred[chosen] / target - 1.0)
        self.duty_bias_by_pool = {p: float(t[0][chosen] / t[1] - 1.0) if t[1] else None
                                  for p, t in sorted(cal.get('duty_by_pool', {}).items())}
        return float(self.thresholds[chosen])

    def _state(self, epoch):
        state = {
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
            'seed': self.seed,
            'deterministic': self.deterministic,
            'amp': self.amp,
            'pool_repeat': self.pool_repeat,
            'noise_params': getattr(self.dataset, 'noise_params', None),
            'select': self.select,
            'split': self.split,
            # which dev pools the threshold's duty match counted, and the bias left on each
            'duty_pools': self.duty_pools,
            'duty_bias_by_pool': self.duty_bias_by_pool,
            'dice_weight': float(self.criterion.dice_weight),
            'negatives': getattr(self.dataset, 'negatives', None),
            'split_hole_frac': getattr(self.dataset, 'split_hole_frac', None),
            'augment_config': (self.dataset.augment_configs()
                               if hasattr(self.dataset, 'augment_configs') else None),
            # the shared decoder: runs above the threshold that reach the peak somewhere
            'decoding': {'threshold': float(self.recommended_threshold),
                         'peak': DECODING_PEAK},
        }
        if state['input_convention'] == 'q90':
            # only a rest-high network has its traces oriented by the rule
            state['polarity_rule'] = dict(POLARITY_RULE)
        return state
