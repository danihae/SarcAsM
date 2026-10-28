import copy
import glob
import os
import re
import warnings
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentConfig, augment

#: Data pools kept separate for splitting and for reporting.
POOLS = ('txt', 'eht', 'drug', 'sim')

#: Recordings of the annotated EHT set that never enter training, so that a model can be
#: scored on EHT recordings it has not seen. One or two recordings per annotator and plate,
#: 25 of the 119 annotated traces.
EHT_TEST_GROUPS = ('eht/20230608_annotator2_ACTN2/20230620a',
                   'eht/20230608_annotator2_TTNv22/20230707a',
                   'eht/20230608_annotator2_TTNv37/20230714a', 'eht/annotator4/20230322m',
                   'eht/annotator4/20230322p')

#: Group of the simulated dev share (``sim_dev`` of :class:`ContractionDataset`), which a
#: split file holds out as dev.
SIM_DEV_GROUP = 'simdev'


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
      ``pool`` so the split can be stratified across the data pools (:data:`POOLS`).
    - Augmentation is applied on the fly, so augmented copies of one trace cannot straddle
      the split, and simulated traces get a fresh noise draw each epoch.
    - Random number generation is per worker and per sample, so ``num_workers > 0`` does
      not replay one augmentation stream in every worker.

    A sample is built the way inference sees a trace. The whole augmented trace is
    conditioned by :func:`~contraction_net.prediction.prepare_robust_input`, so its
    reference is the trace's own and not the crop's, and the crop is cut from the result. A
    trace shorter than the crop gets the inference pad
    (:func:`~contraction_net.padding.pad_for_inference`) and then zeros in every channel --
    the rest level and zero slope under ``'q90'`` -- up to the crop length. The sample's
    ``weight`` is 1 on real frames and 0 on the padding, which carries no label, so the
    loss learns nothing from it. Reflecting signal and label to the crop length instead made
    about half the frames of a short trace time-reversed copies.

    Sign-flip augmentation is on by default and requires ``input_convention='symmetric'``.
    An earlier version deliberately omitted it, on the grounds that the direction of the
    excursion is the cue separating a long contraction from a long quiescent stretch. That
    holds only when rest really is the high side, which is false for Z-band position
    traces: bands on opposite sides of the contraction node move in opposite directions,
    and the pipeline feeds both to the network. Under polarity invariance the network must
    instead resolve rest from the waveform asymmetry, which is what the simulator's twitch
    shape provides. The bundled model is trained the other way, with
    ``input_convention='q90'``, ``rest_high=True`` and no sign flip: at inference the
    polarity rule (:mod:`contraction_net.polarity`) turns every trace rest-high before the
    network reads it, so the direction of the excursion stays a cue.

    Parameters
    ----------
    source_dir : str or None, optional
        Directory of annotated ``.txt`` traces (``<name>.txt`` with ``<name>_contr.txt``).
        Independent annotations of one trace are ``<name>_annotator<k>.txt``, numbered per
        annotator, and stay in one group.
    eht_dir : str or None, optional
        Directory of annotated engineered-heart-tissue pole traces, one sub-directory per
        plate, each trace as ``<recording>_<i>.txt`` with ``<recording>_<i>_contraction.txt``
        holding one start and one end frame per twitch. Kept as its own pool, ``'eht'``.
    eht_exclude : sequence of str, optional
        Groups of the EHT pool left out entirely. Default :data:`EHT_TEST_GROUPS`.
    rest_high : bool, optional
        Present every trace with rest on the high side, for a model that is told the
        polarity rather than inferring it. Real traces are flipped where their label puts
        the contraction above rest -- half the Z-band position tracks of the drug corpus
        -- and the simulator generates downward excursions only. Default is False, which
        keeps both polarities, as a polarity-invariant model needs.
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
        per trace would leave the group split unable to allocate simulated data. A split
        file trains on all of them and holds out ``sim_dev`` instead.
    negatives : str, optional
        ``'legacy'`` (default) or ``'structured'``; see
        :data:`~contraction_net.simulation.NEGATIVES`. ``'structured'`` also shows every
        augmented trace without a labelled contraction negated with probability 0.5, since
        the polarity rule at inference hands the network quiet traces of either sign.
    split_hole_frac : float, optional
        Width of the simulator's label cut between overlapping twitches, as a fraction of
        the shorter twitch; see :func:`~contraction_net.simulation.simulate_dataset`.
        Default 0 keeps the one-frame cut.
    sim_dev : dict or None, optional
        The simulated dev share of a split file, ``{'n_dev': n, 'seed': s}``: ``n`` more
        traces simulated like the others but with seed ``s`` rather than ``seed + 1``, in
        the group :data:`SIM_DEV_GROUP`, which
        :class:`~contraction_net.training.Trainer` holds out as dev. Its own seed keeps dev
        the same for every training seed, and its own duty sweep covers every duty. Default
        None simulates none.
    """

    def __init__(self, source_dir=None, corpus=None, n_synthetic=8000, crop_len=1024,
                 seed=0, boundary_halfwidth=2, augment=True, augment_config=None,
                 resample_sim_noise=True, input_convention='symmetric', sim_groups=40,
                 eht_dir=None, eht_exclude=EHT_TEST_GROUPS, rest_high=False,
                 negatives='legacy', split_hole_frac=0.0, sim_dev=None):
        from .simulation import estimate_noise_params, simulate_dataset

        self.crop_len = int(crop_len)
        self.boundary_halfwidth = int(boundary_halfwidth)
        self.augment = bool(augment)
        self.augment_config = augment_config
        self.resample_sim_noise = bool(resample_sim_noise)
        # False draws the central crop instead; see held_out()
        self.random_crop = True
        self.input_convention = input_convention
        self.seed = int(seed)
        self.epoch = 0
        self.rest_high = bool(rest_high)
        self.negatives = negatives
        self.split_hole_frac = float(split_hole_frac)
        self.sim_dev = None if sim_dev is None else dict(sim_dev)

        self.signals, self.labels, self.groups, self.pools = [], [], [], []
        self.trace_types, self.traces = [], {}

        real_traces = []
        if source_dir is not None:
            for signal, label, group in self._load_real(source_dir):
                self._add(signal, label, group, 'txt', 'txt')
                real_traces.append(signal)

        if eht_dir is not None:
            excluded = set(eht_exclude or ())
            for signal, label, group in self.load_eht(eht_dir):
                if group not in excluded:
                    self._add(signal, label, group, 'eht', 'eht')
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

        def simulate(n, sim_seed):
            return simulate_dataset(n=n, seed=sim_seed, noise_rel=noise_rel,
                                    noise_rho=noise_rho,
                                    polarity=1 if self.rest_high else None,
                                    negatives=negatives,
                                    split_hole_frac=self.split_hole_frac)

        for i, trace in enumerate(simulate(n_synthetic, seed + 1)):
            self._add_simulated(trace, f'sim/{i % max(1, int(sim_groups))}')
        if self.sim_dev is not None:
            for trace in simulate(int(self.sim_dev['n_dev']), int(self.sim_dev['seed'])):
                self._add_simulated(trace, SIM_DEV_GROUP)

    def _add_simulated(self, trace, group):
        idx = len(self.signals)
        self._add(np.asarray(trace.signal, float), np.asarray(trace.label, bool), group,
                  'sim', f'sim_{trace.regime}')
        # the dev share keeps the artefacts it was simulated with (see held_out), so only
        # the training traces need what renoise redraws them from
        if self.resample_sim_noise and group != SIM_DEV_GROUP:
            self.traces[idx] = trace

    def _add(self, signal, label, group, pool, trace_type):
        if self.rest_high and pool != 'sim':
            signal, label = np.asarray(signal, float), np.asarray(label, bool)
            if label.any() and not label.all() and \
                    np.nanmedian(signal[label]) > np.nanmedian(signal[~label]):
                signal = -signal
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
    def load_eht(eht_dir):
        """Yield ``(signal, label, group)`` for every annotated EHT trace under ``eht_dir``.

        The group is ``eht/<plate directory>/<recording>``: recording names repeat across
        plates, so the name alone would join unrelated tissues into one group.
        """
        paths = sorted(glob.glob(os.path.join(eht_dir, '**', '*.txt'), recursive=True))
        for path in paths:
            name = os.path.basename(path)
            if 'contr' in name or 'peaks' in name or 'systole' in name:
                continue
            label_path = path[:-4] + '_contraction.txt'
            if not os.path.exists(label_path):
                continue
            signal = np.loadtxt(path).astype(float).ravel()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                raw = np.loadtxt(label_path)
            label = np.zeros(signal.size, dtype=bool)
            if np.size(raw):
                raw = np.atleast_2d(raw)
                pairs = raw.T if raw.shape[0] == 2 else raw.reshape(-1, 2)
                for start, end in np.clip(pairs, 0, signal.size):
                    label[int(round(start)):int(round(end))] = True
            plate = os.path.relpath(os.path.dirname(path), eht_dir)
            yield signal, label, f'eht/{plate}/{re.sub(r"_[0-9]+$", "", name[:-4])}'

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
        near-identical rows share one recording. Prefixes marking a derived view and an
        ``_annotator<k>`` suffix are stripped too: those are different views or
        independent annotations of the same cell.
        """
        name = os.path.basename(path)[:-4]
        for prefix in ('line_', 'wavelet_', 'kymo_'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        for marker in ('_slen_', '_z_pos_'):
            if marker in name:
                return name.split(marker)[0]
        name = re.sub(r'_annotator\d+$', '', name)
        return re.sub(r'_\d+$', '', name)

    def __len__(self):
        return len(self.signals)

    def set_epoch(self, epoch):
        """Advance the augmentation stream; call once per epoch before iterating.

        Without this the draw depends only on the sample index, so every epoch would
        replay identical augmentation.
        """
        self.epoch = int(epoch)

    def held_out(self):
        """The same traces, drawn the way a held-out trace is scored.

        No augmentation, the artefacts a simulated trace was generated with instead of a
        fresh draw, and the central crop, so a sample is the same on every access whatever
        the epoch or the seed. :class:`~contraction_net.training.Trainer` scores the dev
        side of a split file through it.
        """
        view = copy.copy(self)
        view.augment, view.resample_sim_noise, view.random_crop = False, False, False
        return view

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

    def augment_configs(self):
        """Augmentation settings per pool as plain dicts, or None without augmentation.

        Stored in the checkpoint, so a model records the augmentation it was trained with.
        """
        if not self.augment:
            return None
        return {pool: asdict(self._config_for(pool)) for pool in sorted(set(self.pools))}

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
        from .padding import pad_for_inference
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
            # a quiet trace reaches the network in either sign, as the polarity rule hands
            # it over
            if self.negatives == 'structured' and not label.any() and rng.random() < 0.5:
                signal = -signal

        # condition the whole trace, as inference does, then cut the crop from the result
        inputs = prepare_robust_input(signal, convention=self.input_convention)
        onset, offset = self._boundaries(label)
        target = np.stack([label.astype(np.float32), onset, offset])
        weight = np.ones(label.size, dtype=np.float32)
        if label.size < self.crop_len:
            # the inference pad; its frames carry no label and no weight
            inputs, pad = pad_for_inference(inputs)
            target = np.pad(target, ((0, 0), (pad, pad)))
            weight = np.pad(weight, (pad, pad))
        n = weight.size
        if n > self.crop_len:
            start = (int(rng.integers(0, n - self.crop_len + 1)) if self.random_crop
                     else (n - self.crop_len) // 2)
            inputs = inputs[:, start:start + self.crop_len]
            target = target[:, start:start + self.crop_len]
            weight = weight[start:start + self.crop_len]
        elif n < self.crop_len:
            # zeros in every channel up to the crop length, weighted 0 as well
            fill = ((0, 0), (0, self.crop_len - n))
            inputs, target = np.pad(inputs, fill), np.pad(target, fill)
            weight = np.pad(weight, fill[1])
        return {'input': torch.from_numpy(np.ascontiguousarray(inputs)).float(),
                'target': torch.from_numpy(np.ascontiguousarray(target)).float(),
                'weight': torch.from_numpy(np.ascontiguousarray(weight)),
                'group': self.groups[idx],
                'pool': pool,
                'trace_type': self.trace_types[idx]}
