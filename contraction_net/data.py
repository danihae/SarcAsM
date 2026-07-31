import glob
import os
import re
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset



class ContractionDataset(Dataset):
    """
    Training data for :class:`~contraction_net.contraction_net.ContractionNet`.

    Combines hand-annotated real traces with simulated ones from
    :mod:`contraction_net.simulation`. The synthetic half is not padding: the real
    recordings all come from normally-beating cells, so they cover neither the high duty
    cycles nor the coarse temporal sampling where the previous model fails, and no amount
    of augmentation of them can supply those regimes.

    Four properties matter for getting an honest validation signal out of it:

    - Traces keep their own length and are **cropped at access time**, so long recordings
      are not discarded and every epoch sees different windows.
    - Each sample carries a ``group`` tag naming its source recording, so
      :class:`~contraction_net.training.Trainer` can hold out whole recordings. Splitting
      on individual traces would put dozens of near-identical rows from one cell on both
      sides, and the validation score would then measure memorisation.
    - Augmentation is applied on the fly rather than baked into a larger array, so
      augmented copies of one trace can never straddle the split.
    - No random sign flip. Making the network blind to the direction of the excursion
      would discard the one cue that separates a long contraction from a long quiescent
      stretch, and the input already carries a signed difference channel.

    Parameters
    ----------
    source_dir : str or None, optional
        Directory of annotated ``.txt`` traces, in the layout :class:`DataProcess` expects
        (``<name>.txt`` alongside ``<name>_contr.txt``). None loads no real data.
    n_synthetic : int, optional
        Number of simulated traces to generate. Default is 3000.
    crop_len : int, optional
        Window length drawn per sample. Default is 512.
    seed : int, optional
        Seed for both simulation and augmentation. Default is 0.
    boundary_halfwidth : int, optional
        Half-width in frames of the tent placed on each onset and offset. Default is 2.
    augment : bool, optional
        Apply on-the-fly augmentation to real traces. Default is True.
    """

    def __init__(self, source_dir=None, n_synthetic=3000, crop_len=512, seed=0,
                 boundary_halfwidth=2, augment=True):
        from .simulation import estimate_noise_params, simulate_dataset

        self.crop_len = int(crop_len)
        self.boundary_halfwidth = int(boundary_halfwidth)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

        self.signals, self.labels, self.groups = [], [], []

        real_traces = []
        if source_dir is not None:
            for signal, label, group in self._load_real(source_dir):
                self.signals.append(signal)
                self.labels.append(label)
                self.groups.append(group)
                real_traces.append(signal)

        # Match the simulated noise to the real traces rather than assuming white Gaussian:
        # sarcomere length is derived from smoothly-moving band positions, so its noise is
        # correlated frame to frame and a white-noise model under-prepares the network.
        noise_rel, noise_rho = (estimate_noise_params(real_traces)
                                if real_traces else (0.05, 0.5))
        self.noise_params = (noise_rel, noise_rho)

        for i, trace in enumerate(simulate_dataset(n=n_synthetic, seed=seed + 1,
                                                   noise_rel=noise_rel, noise_rho=noise_rho)):
            self.signals.append(np.asarray(trace.signal, dtype=float))
            self.labels.append(np.asarray(trace.label, dtype=bool))
            # every simulated trace is its own group: they are independent by construction
            self.groups.append(f'sim/{trace.regime}/{i}')

    @staticmethod
    def _load_real(source_dir):
        """Yield ``(signal, label, group)`` for each annotated trace in a directory.

        Traces whose annotation file holds no parseable data are **skipped**, not read as
        all-quiescent. In the bundled training set 45 files are blank, and 29% of those
        belong to traces that are unmistakably beating (robust spread 8-23x the noise,
        against ~22 for annotated beating traces and ~3 for pure noise). They were never
        annotated. Coercing them to a zero mask would teach the network that a beating
        trace contains no contractions. Genuine quiescent examples
        come from the ``*_noise_*`` files, which carry explicit full-length zero masks, and
        from the simulator, where the label is certain.
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
    def _group_of(path):
        """Source recording a trace belongs to.

        Traces are named ``<recording>_slen_<i>`` / ``<recording>_z_pos_<i>``, so dozens of
        near-identical rows share one recording. Splitting on individual traces leaks a
        recording across train and validation and makes the validation loss meaningless --
        which is what selected the bundled checkpoint.

        Prefixes marking a derived view (``line_``, ``wavelet_``) and suffixes naming the
        annotator (``_daniel``, ``_lara``) are stripped too: those are different
        representations or independent annotations *of the same recording*, so leaving them
        distinct would put the same underlying cell on both sides of the split.
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

    def _augment(self, signal):
        """Nuisances applied on the fly; the simulator already carries its own."""
        rng = self.rng
        spread = np.percentile(signal, 90) - np.percentile(signal, 10)
        if spread <= 0:
            spread = np.abs(signal).max() or 1.0
        out = signal.astype(float).copy()
        if rng.random() < 0.5:
            out = out + rng.normal(0, 0.05 * spread * rng.uniform(0.2, 2.0), out.size)
        if rng.random() < 0.3:
            out = out + 0.2 * spread * rng.uniform(-1, 1) * np.linspace(0, 1, out.size)
        if rng.random() < 0.3:
            n_out = int(rng.integers(1, 6))
            idx = rng.integers(0, out.size, n_out)
            out[idx] += rng.normal(0, 0.5 * spread, n_out)
        if rng.random() < 0.2:      # coarser sampling, by dropping frames
            step = int(rng.integers(2, 5))
            out = out[::step]
        return out

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

        signal = np.asarray(self.signals[idx], dtype=float)
        label = np.asarray(self.labels[idx], dtype=bool)

        if self.augment and not self.groups[idx].startswith('sim/'):
            n_before = signal.size
            signal = self._augment(signal)
            if signal.size != n_before:      # frame dropping also thins the labels
                label = label[::int(round(n_before / signal.size))][:signal.size]
                label = np.pad(label, (0, max(0, signal.size - label.size)))[:signal.size]

        # random crop, or pad up if the trace is shorter than the window
        n = signal.size
        if n > self.crop_len:
            start = int(self.rng.integers(0, n - self.crop_len + 1))
            signal = signal[start:start + self.crop_len]
            label = label[start:start + self.crop_len]
        elif n < self.crop_len:
            pad = self.crop_len - n
            signal = np.pad(signal, (0, pad), mode='reflect' if n > 1 else 'edge')
            label = np.pad(label, (0, pad), mode='reflect' if n > 1 else 'edge')

        inputs = prepare_robust_input(signal)
        onset, offset = self._boundaries(label)
        target = np.stack([label.astype(np.float32), onset, offset])
        return {'input': torch.from_numpy(inputs).float(),
                'target': torch.from_numpy(target).float(),
                'group': self.groups[idx]}
