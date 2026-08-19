import glob
import os
import re
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentConfig, augment

#: Data pools kept separate for splitting and for reporting.
POOLS = ('txt', 'drug', 'sim')


class ContractionDataset(Dataset):
    """
    Training data for :class:`~contraction_net.contraction_net.ContractionNet`.

    Combines hand-annotated traces, a corpus distilled from an already-analysed dataset
    and simulated traces. The synthetic part is
    not padding: real recordings of healthy cells cover neither the high duty cycles nor
    the coarse temporal sampling the model has to handle, and no augmentation of them can
    supply those regimes.

    Four properties keep the validation signal meaningful:

    - Traces keep their own length and are cropped at access time, so long recordings are
      not discarded and every epoch sees a different window.
    - Each sample carries a ``group`` naming its source recording, so
      :class:`~contraction_net.training.Trainer` can hold out whole recordings, and a
      ``pool`` so the split can be stratified across the three sources.
    - Augmentation is applied on the fly, so augmented copies of one trace cannot straddle
      the split, and simulated traces get a fresh noise draw each epoch.
    - Random number generation is per worker and per sample, so ``num_workers > 0`` does
      not replay one augmentation stream in every worker.

    Sign-flip augmentation is on by default and requires ``input_convention='symmetric'``.
    An earlier version deliberately omitted it, on the grounds that the direction of the
    excursion is the cue separating a long contraction from a long quiescent stretch. That
    holds only when rest really is the high side, which is false for Z-band position
    traces: bands on opposite sides of the contraction node move in opposite directions,
    and the pipeline feeds both to the network. Under polarity invariance the network must
    instead resolve rest from the waveform asymmetry, which is what the simulator's twitch
    shape provides.

    Parameters
    ----------
    source_dir : str or None, optional
        Directory of annotated ``.txt`` traces (``<name>.txt`` with ``<name>_contr.txt``).
    corpus : str or sequence of str or None, optional
        One or more ``.npz`` corpora of real traces; see :meth:`load_corpus`.
    n_synthetic : int, optional
        Number of simulated traces. Default is 8000.
    crop_len : int, optional
        Window length drawn per sample. Default is 1024. Shorter than the 513-frame
        receptive field plus a full cycle, a high-duty crop can contain no transition at
        all while still being labelled contracting, which teaches only a duty prior.
    seed : int, optional
        Base seed for simulation and augmentation. Default is 0.
    boundary_halfwidth : int, optional
        Half-width in frames of the tent on each onset and offset. Default is 2.
    augment : bool, optional
        Apply on-the-fly augmentation. Default is True.
    augment_config : AugmentConfig or None, optional
        Override the per-pool presets.
    resample_sim_noise : bool, optional
        Redraw the simulator's artefacts on every access. Default is True.
    input_convention : str, optional
        Conditioning passed to
        :func:`~contraction_net.prediction.prepare_robust_input`. Default ``'symmetric'``.
    sim_groups : int, optional
        Number of groups the simulated traces are spread over. Default is 40. One group
        per trace would leave the group split unable to allocate simulated data.
    """

    def __init__(self, source_dir=None, corpus=None, n_synthetic=8000, crop_len=1024,
                 seed=0, boundary_halfwidth=2, augment=True, augment_config=None,
                 resample_sim_noise=True, input_convention='symmetric', sim_groups=40):
        from .simulation import estimate_noise_params, simulate_dataset

        self.crop_len = int(crop_len)
        self.boundary_halfwidth = int(boundary_halfwidth)
        self.augment = bool(augment)
        self.augment_config = augment_config
        self.resample_sim_noise = bool(resample_sim_noise)
        self.input_convention = input_convention
        self.seed = int(seed)
        self.epoch = 0

        self.signals, self.labels, self.groups, self.pools = [], [], [], []
        self.trace_types, self.traces = [], {}

        real_traces = []
        if source_dir is not None:
            for signal, label, group in self._load_real(source_dir):
                self._add(signal, label, group, 'txt', 'txt')
                real_traces.append(signal)

        if corpus is not None:
            for path in ([corpus] if isinstance(corpus, (str, os.PathLike)) else corpus):
                for signal, label, group, trace_type in self._load_corpus(path):
                    self._add(signal, label, group, 'drug', trace_type)
                    real_traces.append(signal)

        # match the simulated noise to the real traces: sarcomere length is derived from
        # smoothly-moving band positions, so its noise is correlated frame to frame
        noise_rel, noise_rho = (estimate_noise_params(real_traces)
                                if real_traces else (0.05, 0.5))
        self.noise_params = (noise_rel, noise_rho)

        for i, trace in enumerate(simulate_dataset(n=n_synthetic, seed=seed + 1,
                                                   noise_rel=noise_rel,
                                                   noise_rho=noise_rho)):
            idx = len(self.signals)
            self._add(np.asarray(trace.signal, float), np.asarray(trace.label, bool),
                      f'sim/{i % max(1, int(sim_groups))}', 'sim', f'sim_{trace.regime}')
            if self.resample_sim_noise:
                self.traces[idx] = trace

    def _add(self, signal, label, group, pool, trace_type):
        self.signals.append(signal)
        self.labels.append(label)
        self.groups.append(group)
        self.pools.append(pool)
        self.trace_types.append(trace_type)

    @staticmethod
    def _load_real(source_dir):
        """Yield ``(signal, label, group)`` for each annotated trace in a directory.

        Traces whose annotation file holds no parseable data are skipped rather than read
        as all-quiescent: coercing an unannotated but visibly beating trace to a zero mask
        would teach the network that a beating trace contains no contractions. Genuine
        quiescent examples come from the ``*_noise_*`` files and from the simulator.
        """
        paths = sorted(glob.glob(os.path.join(source_dir, '*.txt')))
        inputs = [p for p in paths
                  if 'peaks' not in p and 'contr' not in os.path.basename(p)]
        skipped = []
        for path in inputs:
            label_path = path[:-4] + '_contr.txt'
            if not os.path.exists(label_path):
                continue
            signal = np.loadtxt(path).astype(float).ravel()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')      # blank files are handled below
                raw = np.loadtxt(label_path)
            if np.size(raw) == 0:
                skipped.append(os.path.basename(path))
                continue
            if raw.ndim == 1 and raw.size == signal.size:
                label = raw.astype(bool)
            else:
                # annotations stored as start/end pairs rather than a per-frame mask
                label = np.zeros(signal.size, dtype=bool)
                raw = np.clip(np.atleast_2d(raw), 0, signal.size)
                pairs = raw.T if raw.shape[0] == 2 else raw
                for start, end in np.asarray(pairs).reshape(-1, 2):
                    label[int(start):int(end)] = True
            if signal.size < 8:
                continue
            yield signal, label, ContractionDataset._group_of(path)
        if skipped:
            print(f'{len(skipped)} trace(s) skipped: annotation file present but empty '
                  f'(unannotated, not quiescent), e.g. {skipped[0]}')

    @staticmethod
    def load_corpus(path):
        """Read a corpus of real traces: flat signals plus an offset index.

        Written by whatever pipeline distils traces from analysed recordings; the layout
        is deliberately plain so the reader needs no pickle support. ``signals`` and
        ``labels`` are concatenated and split by ``offsets``; any other array is per-trace
        metadata carried through for grouping and reporting.
        """
        with np.load(path, allow_pickle=False) as data:
            offsets = data['offsets']
            sig, lab = data['signals'], data['labels']
            out = {'signals': [sig[a:b] for a, b in zip(offsets[:-1], offsets[1:])],
                   'labels': [lab[a:b].astype(bool)
                              for a, b in zip(offsets[:-1], offsets[1:])]}
            for key in data.files:
                if key not in ('signals', 'labels', 'offsets'):
                    out[key] = data[key]
        return out

    @staticmethod
    def _load_corpus(path):
        """Yield ``(signal, label, group, trace_type)`` from a distilled corpus.

        The group is the **well**, not the cell: cells from one well share a preparation
        and a drug exposure, so splitting on cells would leave correlated recordings on
        both sides.
        """
        data = ContractionDataset.load_corpus(path)
        n = len(data['signals'])
        wells = data.get('well_uid')
        cells = data.get('cell_uid')
        types = data.get('trace_type')
        for i in range(n):
            if wells is not None:
                group = f'well/{wells[i]}'
            elif cells is not None:
                group = f'cell/{cells[i]}'
            else:
                group = f'corpus/{os.path.basename(path)}/{i}'
            yield (np.asarray(data['signals'][i], float),
                   np.asarray(data['labels'][i], bool),
                   group,
                   str(types[i]) if types is not None else 'corpus')

    @staticmethod
    def _group_of(path):
        """Source recording a trace belongs to.

        Traces are named ``<recording>_slen_<i>`` / ``<recording>_z_pos_<i>``, so dozens of
        near-identical rows share one recording. Prefixes marking a derived view and
        suffixes naming the annotator are stripped too: those are different views or
        independent annotations of the same cell.
        """
        name = os.path.basename(path)[:-4]
        for prefix in ('line_', 'wavelet_', 'kymo_'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        for marker in ('_slen_', '_z_pos_'):
            if marker in name:
                return name.split(marker)[0]
        name = re.sub(r'_(daniel|lara)$', '', name)
        return re.sub(r'_\d+$', '', name)

    def __len__(self):
        return len(self.signals)

    def set_epoch(self, epoch):
        """Advance the augmentation stream; call once per epoch before iterating.

        Without this the draw depends only on the sample index, so every epoch would
        replay identical augmentation.
        """
        self.epoch = int(epoch)

    def _rng(self, idx):
        """Generator that differs per sample and per epoch, reproducibly.

        Seeded from ``(seed, epoch, idx)`` only, so the stream does not depend on how many
        workers the DataLoader runs. A single stateful generator on the dataset would be
        forked identically into every worker and replay one stream in each.
        """
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, getattr(self, 'epoch', 0), int(idx)]))

    def _config_for(self, pool):
        if self.augment_config is not None:
            return self.augment_config
        return AugmentConfig.for_pool(pool,
                                      sign_flip=self.input_convention == 'symmetric')

    def _boundaries(self, label):
        """Onset and offset channels: a tent of ``boundary_halfwidth`` on each transition."""
        onset = np.zeros(label.size, dtype=np.float32)
        offset = np.zeros(label.size, dtype=np.float32)
        edges = np.diff(label.astype(np.int8))
        w = self.boundary_halfwidth
        for target, positions in ((onset, np.where(edges > 0)[0] + 1),
                                  (offset, np.where(edges < 0)[0] + 1)):
            for pos in positions:
                lo, hi = max(0, pos - w), min(label.size, pos + w + 1)
                for i in range(lo, hi):
                    target[i] = max(target[i], 1.0 - abs(i - pos) / (w + 1.0))
        return onset, offset

    def __getitem__(self, idx):
        from .prediction import prepare_robust_input
        from .simulation import renoise

        rng = self._rng(idx)
        pool = self.pools[idx]
        label = np.asarray(self.labels[idx], dtype=bool)

        if pool == 'sim' and self.resample_sim_noise and idx in self.traces:
            signal = renoise(self.traces[idx], rng)
        else:
            signal = np.asarray(self.signals[idx], dtype=float)

        if self.augment:
            signal, label = augment(signal, label, rng, self._config_for(pool))

        n = signal.size
        if n > self.crop_len:
            start = int(rng.integers(0, n - self.crop_len + 1))
            signal = signal[start:start + self.crop_len]
            label = label[start:start + self.crop_len]
        elif n < self.crop_len:
            pad = self.crop_len - n
            signal = np.pad(signal, (0, pad), mode='reflect' if n > 1 else 'edge')
            label = np.pad(label, (0, pad), mode='reflect' if n > 1 else 'edge')

        inputs = prepare_robust_input(signal, convention=self.input_convention)
        onset, offset = self._boundaries(label)
        target = np.stack([label.astype(np.float32), onset, offset])
        return {'input': torch.from_numpy(inputs).float(),
                'target': torch.from_numpy(target).float(),
                'group': self.groups[idx],
                'pool': pool,
                'trace_type': self.trace_types[idx]}
