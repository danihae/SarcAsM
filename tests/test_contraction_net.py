"""Tests for the ContractionNet input conventions, architecture dispatch and training path.

The golden fixture ``tests/data/contraction_net_golden.npz`` pins the bundled checkpoint's
output so that changes to the conditioning cannot silently alter shipped results.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from contraction_net.augment import AugmentConfig, augment, time_warp, twitch_scale
from contraction_net.contraction_net import ContractionNet, SymmetrizedContractionNet
from contraction_net.benchmark import (boundary_report, polarity_cost_probe,
                                       polarity_invariance_report)
from contraction_net.data import ContractionDataset
from contraction_net.losses import MaskBoundaryLoss, iou_at_thresholds
from contraction_net.training import Trainer
from contraction_net.prediction import (INPUT_CONVENTIONS, _load_model, predict_contractions,
                                        prepare_robust_input, recommended_threshold)
from contraction_net.simulation import (SNR_RANGE, _ar1, make_stress_set, renoise,
                                        simulate_dataset,
                                        simulate_trace)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_MODEL = os.path.join(REPO, 'sarcasm', 'models', 'model_ContractionNet.pt')
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                      'contraction_net_golden.npz')

#: Conventions that condition the trace; 'raw' passes it through unchanged.
CONDITIONED = tuple(c for c in INPUT_CONVENTIONS if c != 'raw')


def _traces():
    rng = np.random.default_rng(0)
    return [
        np.cumsum(rng.normal(0, .01, 800)) + 1.8,
        1.9 - 0.15 * (np.sin(np.linspace(0, 40, 1200)) > 0.6),
        np.full(300, 1.75),
        np.r_[np.full(200, 1.8), np.full(800, 1.6)],
        rng.normal(1.8, .002, 64),
    ]


class TestInputConventions:
    def test_symmetric_is_exactly_odd(self):
        for x in _traces():
            a = prepare_robust_input(x, convention='symmetric')
            b = prepare_robust_input(-x, convention='symmetric')
            assert np.abs(a + b).max() < 1e-6

    def test_q90_matches_reference_formula(self):
        for x in _traces():
            got = prepare_robust_input(x, convention='q90')
            rest = np.percentile(x, 90)
            scale = float(np.percentile(x, 90) - np.percentile(x, 10))
            if scale <= 0:
                scale = float(np.abs(x - rest).max())
            if scale <= 0:
                scale = 1.0
            assert np.array_equal(got[0], ((x - rest) / scale).astype(np.float32))

    def test_q90_is_not_odd(self):
        # the asymmetry is what sign-flip augmentation would contradict
        x = _traces()[1]
        a = prepare_robust_input(x, convention='q90')
        b = prepare_robust_input(-x, convention='q90')
        assert np.abs(a + b).max() > 0.5

    def test_affine_invariance(self):
        # 'raw' is excluded on purpose: the pre-1.0 recipe did no conditioning, so its
        # comparison arm is scale dependent by construction
        x = _traces()[0]
        for convention in CONDITIONED:
            a = prepare_robust_input(x, convention=convention)
            b = prepare_robust_input(1000 * x + 7, convention=convention)
            assert np.abs(a - b).max() < 1e-4

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match='convention'):
            prepare_robust_input(_traces()[0], convention='median')


class TestShippedCheckpoint:
    def test_declares_its_conventions(self):
        net = _load_model(SHIPPED_MODEL, ContractionNet)
        assert type(net) is SymmetrizedContractionNet
        assert net.input_convention == 'symmetric'
        assert recommended_threshold(SHIPPED_MODEL) == pytest.approx(0.45)

    def test_is_exactly_polarity_invariant(self):
        # the property the shipped model exists to provide
        for x in _traces():
            a = predict_contractions(x, SHIPPED_MODEL)
            b = predict_contractions(-x, SHIPPED_MODEL)
            assert np.allclose(a, b, atol=1e-6)

    def test_reproduces_golden_prediction(self):
        # float32 inference differs by up to ~6e-4 between CPU, CUDA and MPS
        # backends; a changed checkpoint or conditioning moves outputs by O(0.1)
        golden = np.load(GOLDEN)
        for i, x in enumerate(_traces()):
            got = predict_contractions(x, SHIPPED_MODEL)
            assert np.allclose(got, golden[f'trace_{i}'], atol=2e-3)
    def test_checkpoint_without_arch_is_rejected(self, tmp_path):
        path = tmp_path / 'pre10.pt'
        torch.save({'n_filter': 64, 'in_channels': 1, 'out_channels': 2,
                    'state_dict': {}}, path)
        with pytest.raises(ValueError, match='predates the 1.0'):
            _load_model(str(path), ContractionNet)

    @pytest.mark.parametrize('cls', [ContractionNet, SymmetrizedContractionNet])
    def test_roundtrip(self, tmp_path, cls):
        net = cls(n_filter=8, in_channels=2, out_channels=3)
        path = tmp_path / f'{cls.__name__}.pt'
        torch.save({'arch': cls.__name__, 'n_filter': 8, 'in_channels': 2,
                    'out_channels': 3, 'input_convention': 'symmetric',
                    'recommended_threshold': 0.42,
                    'state_dict': net.state_dict()}, path)
        loaded = _load_model(str(path), ContractionNet)
        assert type(loaded) is cls
        assert loaded.input_convention == 'symmetric'
        assert recommended_threshold(str(path)) == pytest.approx(0.42)

    def test_arch_kwargs_roundtrip(self, tmp_path):
        net = ContractionNet(n_filter=8, in_channels=2, out_channels=3, norm='instance')
        path = tmp_path / 'instance.pt'
        torch.save({'arch': 'ContractionNet', 'n_filter': 8, 'in_channels': 2,
                    'out_channels': 3, 'arch_kwargs': {'norm': 'instance'},
                    'state_dict': net.state_dict()}, path)
        loaded = _load_model(str(path), ContractionNet)
        assert loaded.norm_kind == 'instance'


class TestArchitectures:
    def test_receptive_field(self):
        assert ContractionNet().receptive_field == 513

    def test_shipped_checkpoint_loads_into_its_declared_architecture(self):
        state = torch.load(SHIPPED_MODEL, map_location='cpu', weights_only=False)
        net = SymmetrizedContractionNet(n_filter=state['n_filter'],
                                        in_channels=state['in_channels'],
                                        out_channels=state['out_channels'],
                                        **state['arch_kwargs'])
        net.load_state_dict(state['state_dict'])

    @pytest.mark.parametrize('reduce', ['max', 'mean'])
    def test_symmetrized_is_exactly_invariant(self, reduce):
        net = SymmetrizedContractionNet(n_filter=8, reduce=reduce).eval()
        x = torch.randn(2, 2, 256)
        with torch.no_grad():
            a = net(x)[1]
            b = net(-x)[1]
        assert torch.equal(a, b)

    @pytest.mark.parametrize('norm', ['batch', 'instance', 'group', 'none'])
    def test_norm_variants_run(self, norm):
        net = ContractionNet(n_filter=8, norm=norm).eval()
        with torch.no_grad():
            probs, logits = net(torch.randn(2, 2, 128))
        assert probs.shape == (2, 3, 128) and logits.shape == (2, 3, 128)

    def test_attention_variant_runs(self):
        net = ContractionNet(n_filter=8, attention=True).eval()
        with torch.no_grad():
            probs, _ = net(torch.randn(2, 2, 128))
        assert probs.shape == (2, 3, 128)
        assert net.receptive_field is None

    @pytest.mark.parametrize('in_ch,out_ch', [(1, 1), (1, 3), (2, 1), (2, 2), (2, 3)])
    def test_channel_variants_run(self, in_ch, out_ch):
        net = ContractionNet(n_filter=8, in_channels=in_ch, out_channels=out_ch).eval()
        with torch.no_grad():
            probs, _ = net(torch.randn(2, in_ch, 128))
        assert probs.shape == (2, out_ch, 128)
    def test_polarity_mirrors_signal_and_keeps_label(self):
        down = simulate_trace('regular', duty=0.4, n_frames=512,
                              rng=np.random.default_rng(7), polarity=1)
        up = simulate_trace('regular', duty=0.4, n_frames=512,
                            rng=np.random.default_rng(7), polarity=-1)
        baseline = down.meta['baseline']
        assert np.allclose(up.clean, 2 * baseline - down.clean)
        assert np.array_equal(up.label, down.label)

    def test_polarity_is_mixed_and_points_both_ways(self):
        traces = simulate_dataset(n=300, seed=3)
        polarity = np.array([t.meta['polarity'] for t in traces])
        assert set(np.unique(polarity)) == {-1, 1}
        assert 0.3 < (polarity == 1).mean() < 0.7
        shifts = {}
        for sign in (-1, 1):
            deltas = [np.median(t.clean[t.label]) - np.median(t.clean[~t.label])
                      for t in traces
                      if t.meta['polarity'] == sign and t.label.any() and not t.label.all()]
            shifts[sign] = float(np.median(deltas))
        assert shifts[1] < 0 < shifts[-1]

    @staticmethod
    def _hf_noise(trace):
        """Robust high-frequency residual scale, blind to drift, steps and outliers."""
        d = np.diff(trace.signal - trace.clean)
        return 1.4826 * np.median(np.abs(d - np.median(d)))

    def test_snr_sets_the_noise_level(self):
        kw = dict(duty=0.3, n_frames=2048, amplitude=0.3, regime='regular')
        loud = simulate_trace(rng=np.random.default_rng(1), snr=30.0, **kw)
        quiet = simulate_trace(rng=np.random.default_rng(1), snr=2.0, **kw)
        assert self._hf_noise(quiet) > 4 * self._hf_noise(loud)

    def test_snr_decouples_noise_from_amplitude(self):
        # on the noise_rel path the ratio is pinned at 1/noise_rel whatever the amplitude
        kw = dict(duty=0.3, n_frames=2048, regime='regular', snr=8.0)
        small = simulate_trace(rng=np.random.default_rng(2), amplitude=0.05, **kw)
        large = simulate_trace(rng=np.random.default_rng(2), amplitude=0.5, **kw)
        assert self._hf_noise(large) > 5 * self._hf_noise(small)

    def test_snr_below_range_is_refused(self):
        with pytest.raises(ValueError, match='snr'):
            simulate_trace('regular', snr=SNR_RANGE[0] - 0.1)

    def test_bad_polarity_is_refused(self):
        with pytest.raises(ValueError, match='polarity'):
            simulate_trace('regular', polarity=0)

    def test_dataset_samples_low_snr(self):
        traces = simulate_dataset(n=400, seed=5, p_sampled_snr=0.5)
        snr = np.array([t.meta['snr'] for t in traces if t.meta['snr'] is not None])
        assert 0.35 < snr.size / 400 < 0.65
        assert snr.min() >= SNR_RANGE[0] and snr.max() <= SNR_RANGE[1]

    def test_renoise_keeps_ground_truth_and_varies(self):
        trace = simulate_dataset(n=20, seed=11)[5]
        rng = np.random.default_rng(0)
        a, b = renoise(trace, rng), renoise(trace, rng)
        assert a.shape == trace.signal.shape and np.isfinite(a).all()
        assert not np.allclose(a, b)
        assert not np.allclose(a, trace.signal)

    def test_ar1_is_unit_variance_and_correlated(self):
        rng = np.random.default_rng(0)
        white, corr = _ar1(20000, 0.0, rng), _ar1(20000, 0.9, rng)
        assert abs(white.std() - 1) < 0.05 and abs(corr.std() - 1) < 0.1
        assert np.corrcoef(corr[:-1], corr[1:])[0, 1] > 0.8


class TestAugmentation:
    def test_time_warp_transforms_signal_and_label_together(self):
        t = np.arange(2000)
        label = (t % 200) < 60
        signal = 1.8 - 0.2 * label
        rng = np.random.default_rng(0)
        out, out_lab = time_warp(signal, label, rng, 2.0, 2.0)
        assert out.size == out_lab.size
        assert abs(out.size - signal.size / 2) <= 2
        # duty survives resampling; a label left un-warped would not preserve it
        assert out_lab.mean() == pytest.approx(label.mean(), abs=0.03)

    def test_time_warp_keeps_short_traces(self):
        rng = np.random.default_rng(0)
        sig, lab = np.ones(20), np.zeros(20, bool)
        out, out_lab = time_warp(sig, lab, rng, 3.0, 3.0, min_len=32)
        assert out.size == 20 and out_lab.size == 20

    def test_twitch_scale_ignores_drift(self):
        t = np.arange(4000)
        label = (t % 200) < 60
        twitch = -0.2 * label
        drift = 3.0 * np.sin(2 * np.pi * t / 4000)
        clean = twitch + 1.8
        assert twitch_scale(clean, label) == pytest.approx(0.2, rel=0.3)
        # a 15x larger drift must not inflate the reference
        assert twitch_scale(clean + drift, label) == pytest.approx(0.2, rel=0.5)

    def test_sign_flip_preserves_the_label(self):
        t = np.arange(1000)
        label = (t % 200) < 60
        cfg = AugmentConfig(p_sign_flip=1.0, p_time_warp=0.0, p_noise=0.0, p_drift=0.0,
                            p_bleach=0.0, p_steps=0.0, p_gaps=0.0, p_quantise=0.0,
                            p_outliers=0.0)
        out, out_lab = augment(1.8 - 0.2 * label, label, np.random.default_rng(0), cfg)
        assert np.array_equal(out_lab, label)
        assert np.allclose(out, -(1.8 - 0.2 * label))

    def test_global_scale_and_offset_are_no_ops_after_conditioning(self):
        # the reason neither is implemented as an augmentation
        x = _traces()[0]
        for convention in CONDITIONED:
            base = prepare_robust_input(x, convention=convention)
            assert np.abs(prepare_robust_input(37 * x - 4, convention=convention)
                          - base).max() < 1e-4

    def test_augment_is_finite_and_label_aligned(self):
        rng = np.random.default_rng(0)
        t = np.arange(1500)
        label = (t % 150) < 40
        for i in range(25):
            out, out_lab = augment(1.8 - 0.2 * label, label,
                                   np.random.default_rng(i), AugmentConfig())
            assert out.size == out_lab.size
            assert np.isfinite(out).all()

    def test_sim_preset_leaves_noise_to_the_simulator(self):
        cfg = AugmentConfig.for_pool('sim')
        assert cfg.p_noise == 0 and cfg.p_drift == 0 and cfg.p_gaps == 0
        assert cfg.p_sign_flip > 0 and cfg.p_time_warp > 0
        assert AugmentConfig.for_pool('drug').p_noise > 0

    def test_sign_flip_disabled_without_symmetric_conditioning(self):
        assert AugmentConfig.for_pool('drug', sign_flip=False).p_sign_flip == 0


class TestDatasetAndTrainer:
    def test_rng_varies_per_epoch_and_not_per_worker(self):
        ds = ContractionDataset(n_synthetic=32, crop_len=256, seed=0)
        first = ds[3]['input'].numpy().copy()
        ds.set_epoch(1)
        second = ds[3]['input'].numpy().copy()
        ds.set_epoch(0)
        assert not np.allclose(first, second)
        assert np.allclose(first, ds[3]['input'].numpy())

    def test_dataloader_workers_do_not_change_the_stream(self):
        from torch.utils.data import DataLoader
        ds = ContractionDataset(n_synthetic=32, crop_len=256, seed=0)

        def run(workers):
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=workers)
            return torch.cat([b['input'] for b in loader]).numpy()

        assert np.allclose(run(0), run(2))

    def test_group_split_keeps_recordings_apart_and_covers_pools(self):
        groups = [f'g{i // 4}' for i in range(80)]
        pools = ['sim'] * 60 + ['drug'] * 20
        train, val, cal = Trainer._group_split(groups, 0.2, 0.5, 0, pools)
        assert not (set(train) & set(val)) and not (set(val) & set(cal))
        for a, b in ((train, val), (train, cal), (val, cal)):
            assert not ({groups[i] for i in a} & {groups[i] for i in b})
        # both pools must reach the held-out side, or it measures only the larger one
        assert {pools[i] for i in val} | {pools[i] for i in cal} == {'sim', 'drug'}

    def test_iou_at_thresholds_is_monotone_for_a_confident_model(self):
        logits = torch.full((2, 3, 100), -5.0)
        logits[:, 0, 20:40] = 5.0
        targets = torch.zeros(2, 3, 100)
        targets[:, 0, 20:40] = 1.0
        scores = iou_at_thresholds(logits, targets, [0.1, 0.5, 0.9]).numpy()
        assert scores.shape == (3, 2)
        assert np.allclose(scores, 1.0)

    @pytest.mark.parametrize('out_channels', [1, 2, 3])
    def test_loss_handles_every_head_count(self, out_channels):
        loss = MaskBoundaryLoss()
        logits = torch.randn(2, out_channels, 64)
        targets = torch.rand(2, 3, 64)
        value, terms = loss(logits, targets)
        assert torch.isfinite(value)
        assert (terms['boundary'] == 0.0) == (out_channels == 1)

    def test_checkpoint_records_its_conventions(self, tmp_path):
        ds = ContractionDataset(n_synthetic=48, crop_len=256, seed=0)
        trainer = Trainer(ds, num_epochs=1, n_filter=8, batch_size=8,
                          save_dir=str(tmp_path), num_workers=0)
        trainer.start(verbose=False)
        state = torch.load(tmp_path / 'model_ContractionNet.pt', map_location='cpu',
                           weights_only=False)
        assert state['arch'] == 'ContractionNet'
        assert state['input_convention'] == 'symmetric'
        assert state['crop_len'] == 256
        assert 0.0 < state['recommended_threshold'] < 1.0
        assert 'sim' in state['val_iou_by_pool']


class TestBenchmarkProbes:
    @staticmethod
    def _oracle_predictor(threshold_frac=0.5):
        """A perfect, polarity-blind detector, for probing the probes themselves."""
        def _predict(signal, frametime):
            x = np.asarray(signal, float)
            centre = 0.5 * (np.percentile(x, 10) + np.percentile(x, 90))
            spread = np.percentile(x, 90) - np.percentile(x, 10)
            return (np.abs(x - centre) > threshold_frac * spread * 0.5).astype(float)
        return _predict

    def test_polarity_report_flags_a_polarity_dependent_model(self):
        traces = make_stress_set(n=8, seed=99999, polarity=1)

        def biased(signal, frametime):
            x = np.asarray(signal, float)
            return (x < np.percentile(x, 50)).astype(float)

        report = polarity_invariance_report(biased, traces)
        assert report['disagreement'] > 0.5

    def test_polarity_report_clears_an_invariant_model(self):
        traces = make_stress_set(n=8, seed=99999, polarity=1)
        report = polarity_invariance_report(self._oracle_predictor(), traces)
        assert report['disagreement'] < 1e-9
        assert report['iou_upright'] == pytest.approx(report['iou_flipped'], abs=1e-9)

    def test_polarity_cost_probe_shape(self):
        probe = polarity_cost_probe(self._oracle_predictor(), duties=(0.3, 0.6))
        assert probe['duties'].shape == probe['iou_upright'].shape == (2,)
        assert np.allclose(probe['gap'], 0.0, atol=1e-9)
        assert probe['symmetric_control']['iou_upright'].shape == (2,)

    def test_boundary_report_finds_exact_transitions(self):
        traces = make_stress_set(n=6, seed=99999, polarity=1)

        def cheat(signal, frametime):
            # emit the true boundaries from the trace this predictor is handed
            for trace in traces:
                if trace.signal.shape == np.shape(signal) and np.allclose(trace.signal,
                                                                          signal):
                    edges = np.diff(trace.label.astype(np.int8))
                    out = np.zeros((3, trace.label.size))
                    out[1, np.flatnonzero(edges > 0) + 1] = 1.0
                    out[2, np.flatnonzero(edges < 0) + 1] = 1.0
                    return out
            return np.zeros((3, np.size(signal)))

        report = boundary_report(cheat, traces)
        assert report['onset_f1'] == pytest.approx(1.0)
        assert report['onset_median_error'] == 0.0


class TestCorpusIntegration:
    @staticmethod
    def _write_corpus(path, n=12, length=800):
        rng = np.random.default_rng(0)
        signals, labels, offsets = [], [], [0]
        wells, types = [], []
        for i in range(n):
            t = np.arange(length)
            label = (t % 200) < 70
            polarity = 1 if i % 2 else -1
            sig = 1.8 - polarity * 0.2 * label + rng.normal(0, 0.01, length)
            signals.append(sig.astype(np.float32))
            labels.append(label.astype(np.uint8))
            offsets.append(offsets[-1] + length)
            wells.append(f'well{i % 3}')
            types.append(('track_slen', 'track_z_pos', 'group_slen')[i % 3])
        np.savez_compressed(path, signals=np.concatenate(signals),
                            labels=np.concatenate(labels),
                            offsets=np.asarray(offsets, np.int64),
                            well_uid=np.asarray(wells, np.str_),
                            trace_type=np.asarray(types, np.str_))
        return str(path)

    def test_dataset_groups_corpus_traces_by_well(self, tmp_path):
        path = self._write_corpus(tmp_path / 'corpus.npz')
        ds = ContractionDataset(corpus=path, n_synthetic=16, crop_len=256, seed=0)
        assert len(ds) == 28
        drug = {g for g, p in zip(ds.groups, ds.pools) if p == 'drug'}
        assert drug == {'well/well0', 'well/well1', 'well/well2'}
        assert set(ds.pools) == {'drug', 'sim'}

    def test_trainer_reports_both_pools(self, tmp_path):
        path = self._write_corpus(tmp_path / 'corpus.npz', n=18)
        ds = ContractionDataset(corpus=path, n_synthetic=24, crop_len=256, seed=0)
        trainer = Trainer(ds, num_epochs=1, n_filter=8, batch_size=4,
                          save_dir=str(tmp_path), num_workers=0)
        history = trainer.start(verbose=False)
        # the held-out side must contain real traces, not only simulated ones
        assert set(history[-1]['val_iou_by_pool']) == {'drug', 'sim'}

    def test_corpus_traces_survive_conditioning_whatever_their_polarity(self, tmp_path):
        path = self._write_corpus(tmp_path / 'corpus.npz')
        ds = ContractionDataset(corpus=path, n_synthetic=0, crop_len=512, seed=0,
                                augment=False)
        for i in range(len(ds)):
            item = ds[i]
            assert torch.isfinite(item['input']).all()
            assert item['target'].shape[0] == 3
