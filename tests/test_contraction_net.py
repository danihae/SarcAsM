"""Tests for the ContractionNet input conventions, architecture dispatch and dataset RNG, and
for the inference path that orients 'q90' traces by the polarity rule.

The golden fixture ``tests/data/contraction_net_golden.npz`` pins the bundled checkpoint's
output so that changes to the conditioning cannot silently alter shipped results. The goldens
are CPU readings, the inference default, of the traces of :func:`_traces` at 1/30 s per
frame with rest declared high (``polarity=1``), as SarcAsM reads sarcomere length: left to
the polarity rule, four of the five traces are undecided, and their zeros would pin nothing.
"""

from __future__ import annotations

import dataclasses
import math
import os
import warnings

import numpy as np
import pytest
import torch

from contraction_net import prediction
from contraction_net.augment import AugmentConfig, augment, time_warp, twitch_scale
from contraction_net.contraction_net import ContractionNet, SymmetrizedContractionNet
from contraction_net.benchmark import (boundary_report, model_predictor, model_predictor_full,
                                       polarity_cost_probe, polarity_invariance_report)
from contraction_net.data import ContractionDataset
from contraction_net.decoding import decode_contractions, recommended_decoding
from contraction_net.losses import MaskBoundaryLoss, iou_at_thresholds
from contraction_net.polarity import PolarityRule
from contraction_net.training import Trainer
from contraction_net.prediction import (INPUT_CONVENTIONS, _load_model, polarity_rule,
                                        predict_contractions, prepare_robust_input,
                                        recommended_threshold)
from contraction_net.simulation import (SNR_RANGE, _ar1, make_stress_set, renoise,
                                        simulate_dataset,
                                        simulate_trace)
from tests.test_polarity import edge_cases, regression_set, twitch_trains

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The bundled checkpoint: a rest-high 'q90' ContractionNet with GroupNorm, which stores its
#: polarity rule and decoding. The inference tests below run behind it.
SHIPPED_MODEL = os.path.join(REPO, 'sarcasm', 'models', 'model_ContractionNet.pt')
#: Device the mirror tests run on. CPU unless set: a device is enabled for inference only
#: once these tests pass on it.
TEST_DEVICE = os.environ.get('CONTRACTION_NET_DEVICE', 'cpu')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
GOLDEN = os.path.join(DATA, 'contraction_net_golden.npz')
#: On the CPU the goldens move by 5e-7 with the thread count and by 1.2e-6 against the same
#: network run in float64; on CUDA the previous model differed by up to 8e-4, which is why
#: they are read on the CPU.
GOLDEN_ATOL = 1e-5

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
        # 'raw' is excluded on purpose: it passes the trace through unconditioned, so it is
        # scale dependent by construction
        x = _traces()[0]
        for convention in CONDITIONED:
            a = prepare_robust_input(x, convention=convention)
            b = prepare_robust_input(1000 * x + 7, convention=convention)
            assert np.abs(a - b).max() < 1e-4

    def test_raw_convention_is_passthrough_and_not_scale_invariant(self):
        x = _traces()[0]
        assert np.allclose(prepare_robust_input(x, convention='raw')[0], x)
        assert prepare_robust_input(x, convention='raw').shape == (1, x.size)
        assert not np.allclose(prepare_robust_input(1000 * x, convention='raw'),
                               prepare_robust_input(x, convention='raw'))

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match='convention'):
            prepare_robust_input(_traces()[0], convention='median')


class TestShippedCheckpoint:
    def test_declares_its_conventions(self):
        net = _load_model(SHIPPED_MODEL, ContractionNet)
        assert type(net) is ContractionNet
        assert (net.norm_kind, net.attn_position) == ('group', False)
        assert net.input_convention == 'q90'
        assert recommended_threshold(SHIPPED_MODEL) == pytest.approx(0.45)
        assert recommended_decoding(SHIPPED_MODEL) == {'threshold': 0.45, 'peak': 0.85}
        assert polarity_rule(SHIPPED_MODEL) == PolarityRule()

    def test_is_exactly_polarity_invariant(self):
        # x and -x reach the network as the same array, whether the rule decides the rest
        # side or it is declared
        for x in _traces():
            a = predict_contractions(x, SHIPPED_MODEL, frametime=1 / 30)
            b = predict_contractions(-x, SHIPPED_MODEL, frametime=1 / 30)
            assert np.array_equal(a, b)
            a = predict_contractions(x, SHIPPED_MODEL, frametime=1 / 30, polarity=1)
            b = predict_contractions(-x, SHIPPED_MODEL, frametime=1 / 30, polarity=-1)
            assert np.array_equal(a, b)

    def test_reproduces_golden_prediction(self):
        golden = np.load(GOLDEN)
        for i, x in enumerate(_traces()):
            got = predict_contractions(x, SHIPPED_MODEL, frametime=1 / 30, polarity=1)
            assert np.allclose(got, golden[f'trace_{i}'], atol=GOLDEN_ATOL)


class TestArchDispatch:
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
        net = ContractionNet(n_filter=state['n_filter'], in_channels=state['in_channels'],
                             out_channels=state['out_channels'], **state['arch_kwargs'])
        net.load_state_dict(state['state_dict'])

    @pytest.mark.parametrize('reduce', ['max', 'mean'])
    def test_symmetrized_is_exactly_invariant(self, reduce):
        net = SymmetrizedContractionNet(n_filter=8, reduce=reduce).eval()
        x = torch.randn(2, 2, 256)
        with torch.no_grad():
            a = net(x)[1]
            b = net(-x)[1]
        assert torch.equal(a, b)

    @pytest.mark.parametrize('norm', ['batch', 'layer', 'instance', 'group', 'none'])
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


class TestSimulator:
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


class TestCorpusFile:
    def test_corpus_roundtrip(self, tmp_path):
        sigs = [np.arange(10, dtype=np.float32), np.arange(4, dtype=np.float32)]
        labs = [np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], np.uint8),
                np.array([1, 1, 0, 0], np.uint8)]
        path = tmp_path / 'c.npz'
        np.savez_compressed(path, signals=np.concatenate(sigs),
                            labels=np.concatenate(labs),
                            offsets=np.array([0, 10, 14], np.int64),
                            trace_type=np.asarray(['track_slen', 'group_slen'], np.str_))
        got = ContractionDataset.load_corpus(str(path))
        assert len(got['signals']) == 2
        assert np.array_equal(got['signals'][1], sigs[1])
        assert got['labels'][0].dtype == bool and got['labels'][0].sum() == 4
        assert list(got['trace_type']) == ['track_slen', 'group_slen']


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
        # a trained checkpoint holds plain data only, so it loads without unpickling code
        state = torch.load(tmp_path / 'model_ContractionNet.pt', map_location='cpu',
                           weights_only=True)
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


# --------------------------------------------------------------------------------------
# inference behind the polarity rule
# --------------------------------------------------------------------------------------

#: CardioTrace's decoder settings, at the bundled checkpoint's stored threshold and peak:
#: contractions of 0.1 s or more, none within 5 frames of either end.
_CARDIOTRACE_DECODING = dict(threshold=0.45, peak=0.85, min_duration_s=0.1, border_frames=5)


def _same_score(a, b):
    """Scores agree, counting two NaNs (a guarded trace on both sides) as equal."""
    return a == b or (math.isnan(a) and math.isnan(b))


def _reference_reading(net, x):
    """One reading of ``x`` by ``net`` on the CPU, written out apart from the code under
    test: condition by the checkpoint's convention, pad 24 frames, run, crop. This is
    predict_contractions as it was before the polarity rule."""
    x = np.asarray(x, dtype=np.float64).ravel()
    prepared = prepare_robust_input(x, convention=net.input_convention)
    prepared = prepared[:net.expected_in_channels]
    mode = 'reflect' if 24 <= x.size - 1 else 'edge'
    prepared = np.pad(prepared, ((0, 0), (24, 24)), mode=mode)
    with torch.no_grad():
        out = net(torch.from_numpy(prepared).unsqueeze(0))[0][0].numpy()
    return out[:, 24:24 + x.size]


@pytest.fixture
def on_test_device(monkeypatch):
    monkeypatch.setattr(prediction, 'device', torch.device(TEST_DEVICE))


@pytest.mark.usefixtures('on_test_device')
class TestMirrorInvariance:
    """Behind the polarity rule, x and -x reach the network as the same array, so a 'q90'
    checkpoint answers both bit for bit, on float64 and on float32 input."""

    def test_mirror_is_bitwise(self):
        for name, x, ft in regression_set():
            for v in (x, x.astype(np.float32)):
                a, da = predict_contractions(v, SHIPPED_MODEL, frametime=ft,
                                             return_decision=True)
                b, db = predict_contractions(-v, SHIPPED_MODEL, frametime=ft,
                                             return_decision=True)
                assert np.array_equal(a, b), name
                assert (db.sign, db.reason) == (-da.sign, da.reason), name
                assert _same_score(db.score, -da.score), name
                assert (decode_contractions(a[0], ft, **_CARDIOTRACE_DECODING)
                        == decode_contractions(b[0], ft, **_CARDIOTRACE_DECODING)), name

    def test_joint_mirror_of_a_declared_polarity_is_bitwise(self):
        for name, x, ft in regression_set():
            for v in (x, x.astype(np.float32)):
                for p in (1, -1):
                    a, da = predict_contractions(v, SHIPPED_MODEL, frametime=ft, polarity=p,
                                                 return_decision=True)
                    b, db = predict_contractions(-v, SHIPPED_MODEL, frametime=ft, polarity=-p,
                                                 return_decision=True)
                    assert np.array_equal(a, b), (name, p)
                    assert (db.sign, db.reason, db.declared) == (-da.sign, da.reason, -p)


class TestOrientedReading:
    """What the network reads on the 'q90' path. The mirror tests hold for any mistake made
    the same way at both signs, such as reading the complement or padding differently, so
    these check the output against a reading made apart from the code under test, and
    against where the contractions are."""

    def test_output_is_one_reading_of_the_oriented_trace(self):
        # no second reading and no mixing: a decided trace gets one reading of sign * x,
        # the same as declaring that sign; an undecided one gets zeros
        net = _load_model(SHIPPED_MODEL, ContractionNet)
        for name, x, ft in regression_set():
            for v in (x, x.astype(np.float32)):
                got, d = predict_contractions(v, SHIPPED_MODEL, frametime=ft,
                                              return_decision=True)
                if not d.sign:
                    assert not got.any(), name
                    continue
                expected = _reference_reading(net, d.sign * v.astype(np.float64))
                assert np.array_equal(got, expected), name
                declared = predict_contractions(v, SHIPPED_MODEL, frametime=ft,
                                                polarity=d.sign)
                assert np.array_equal(declared, expected), name

    def test_the_contracted_frames_are_called_at_either_sign(self):
        # the twitch trains rest high: frames more than a quarter of the way down a twitch
        # are the contraction, at x and at -x alike, and the reading of the complement
        # would call the rests instead (measured on 118 decided readings of 120: median
        # 0.88 of frames agree, lowest 0.67; read upside down, median 0.34)
        agree = []
        for x in twitch_trains().values():
            contracted = x < -0.25
            for v in (x, -x):
                probs, d = predict_contractions(v, SHIPPED_MODEL, frametime=1 / 30,
                                                return_decision=True)
                if d.sign:
                    agree.append(np.mean((probs[0] > _CARDIOTRACE_DECODING['threshold'])
                                         == contracted))
        assert len(agree) >= 100 and np.median(agree) >= 0.8


class TestInferenceDevice:
    def test_defaults_to_the_cpu(self):
        # the mirror tests above are what any other device has to pass first
        assert prediction.device == torch.device('cpu')


class TestDegenerateInputs:
    """Through predict_contractions, degenerate traces never reach the network."""

    @pytest.mark.parametrize('name,reason', [
        ('n=1', 'too_short'), ('n=5', 'too_short'), ('n=31', 'too_short'),
        ('n=40', 'too_short'), ('n=59', 'too_short'), ('zeros', 'flat'), ('constant', 'flat'),
        ('zeros + outlier', 'flat'), ('zeros + 2-frame blips', 'flat'),
        ('two-level 91/9', 'flat')])
    def test_zeros_and_a_reason(self, name, reason):
        x = edge_cases()[name]
        for v in (x, -x):
            probs, d = predict_contractions(v, SHIPPED_MODEL, frametime=1 / 30,
                                            return_decision=True)
            assert probs.dtype == np.float32 and probs.shape == (3, x.size)
            assert not probs.any()
            assert (d.sign, d.reason) == (0, reason)
            assert decode_contractions(probs[0], 1 / 30, **_CARDIOTRACE_DECODING) == []

    def test_every_length_below_32_frames_is_too_short(self):
        for n in range(1, 32):
            x = np.random.default_rng(n).standard_normal(n)
            probs, d = predict_contractions(x, SHIPPED_MODEL, frametime=1.0,
                                            return_decision=True)
            assert d.reason == 'too_short' and not probs.any() and probs.shape == (3, n)

    def test_the_network_is_not_called(self, monkeypatch):
        net = ContractionNet(n_filter=8)

        def refuse(*args, **kwargs):
            raise AssertionError('a degenerate trace reached the network')

        monkeypatch.setattr(net, 'forward', refuse)
        for name in ('n=5', 'zeros', 'zeros + outlier', 'two-level 91/9'):
            probs = predict_contractions(edge_cases()[name], net, frametime=1 / 30)
            assert probs.shape == (3, edge_cases()[name].size) and not probs.any()

    @pytest.mark.parametrize('cls,kwargs,n_out', [
        (ContractionNet, {}, 3), (SymmetrizedContractionNet, {}, 3),
        (ContractionNet, {'in_channels': 1, 'out_channels': 2}, 2)])
    def test_a_bare_module_gets_zeros_of_its_own_channel_count(self, cls, kwargs, n_out):
        net = cls(n_filter=8, **kwargs)
        probs = predict_contractions(np.zeros(300), net, frametime=1 / 30)
        assert probs.shape == (n_out, 300) and not probs.any()

    def test_a_module_of_unknown_output_is_refused(self):
        with pytest.raises(ValueError, match='channels'):
            predict_contractions(np.zeros(300), torch.nn.Identity(), frametime=1 / 30)

    def test_symmetric_waves_give_zeros(self):
        # no polarity cue, so no contraction rather than a guess
        t = np.arange(1800) / 30
        for y in (np.sin(2 * np.pi * 1.3 * t), np.sign(np.sin(2 * np.pi * 1.5 * t))):
            probs, d = predict_contractions(y, SHIPPED_MODEL, frametime=1 / 30,
                                            return_decision=True)
            assert d.reason == 'ambiguous' and not probs.any()
            assert decode_contractions(probs[0], 1 / 30, **_CARDIOTRACE_DECODING) == []

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
    def test_non_finite_input_raises(self, bad):
        with pytest.raises(ValueError, match='NaN or infinite'):
            predict_contractions(np.r_[np.zeros(50), bad, np.ones(50)], SHIPPED_MODEL,
                                 frametime=1 / 30)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match='empty'):
            predict_contractions(np.zeros(0), SHIPPED_MODEL, frametime=1 / 30)


class TestCheckpointRule:
    """The rule and the decoding constants travel with the checkpoint."""

    @staticmethod
    def _save(tmp_path, **extra):
        net = ContractionNet(n_filter=8)
        path = tmp_path / 'q90.pt'
        torch.save({'arch': 'ContractionNet', 'n_filter': 8, 'in_channels': 2,
                    'out_channels': 3, 'input_convention': 'q90',
                    'state_dict': net.state_dict(), **extra}, path)
        return str(path)

    def test_the_shipped_checkpoint_stores_version_1(self):
        stored = torch.load(SHIPPED_MODEL, map_location='cpu',
                            weights_only=False)['polarity_rule']
        assert stored == dataclasses.asdict(PolarityRule())
        assert polarity_rule(_load_model(SHIPPED_MODEL, ContractionNet)) == PolarityRule()

    def test_checkpoints_without_a_rule_get_version_1(self, tmp_path):
        # as a 1.0 checkpoint, which stores neither and still loads by path
        path = self._save(tmp_path)
        assert polarity_rule(path) == PolarityRule()
        assert polarity_rule(_load_model(path, ContractionNet)) == PolarityRule()
        assert recommended_decoding(path) == {'threshold': 0.5, 'peak': 0.85}

    def test_a_stored_rule_is_used(self, tmp_path):
        path = self._save(tmp_path, polarity_rule=dataclasses.asdict(PolarityRule(margin=1.0)),
                          decoding={'threshold': 0.55, 'peak': 0.9})
        assert polarity_rule(path) == PolarityRule(margin=1.0)
        assert recommended_decoding(path) == {'threshold': 0.55, 'peak': 0.9}
        x = twitch_trains()['train a3 b6 r24']
        probs, d = predict_contractions(x, path, frametime=1 / 30, return_decision=True)
        assert (d.sign, d.reason, d.margin) == (0, 'ambiguous', 1.0) and not probs.any()
        # the margin argument still overrides the stored one
        _, d = predict_contractions(x, path, frametime=1 / 30, margin=0.1, return_decision=True)
        assert d.sign == 1

    def test_an_unknown_rule_version_is_refused(self, tmp_path):
        path = self._save(tmp_path, polarity_rule={'version': 2})
        with pytest.raises(ValueError, match='version'):
            predict_contractions(np.zeros(100), path, frametime=1 / 30)

    def test_missing_frametime_warns_and_assumes_30_fps(self):
        x = twitch_trains()['train a3 b6 r24']
        with pytest.warns(UserWarning, match='frametime'):
            probs = predict_contractions(x, SHIPPED_MODEL)
        assert np.array_equal(probs, predict_contractions(x, SHIPPED_MODEL, frametime=1 / 30))


class TestBenchmarkPredictors:
    """The benchmark's model wrappers hand the frametime they are called with, and a
    declared polarity, on to predict_contractions."""

    def test_the_frametime_reaches_the_polarity_rule(self):
        x = twitch_trains()['train a3 b6 r24'][:300]
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='.*frametime')
            assert model_predictor(SHIPPED_MODEL)(x, 1 / 30).any()
            # the same 300 frames last 1.5 s at 200 fps: too short to decide, so zeros
            assert not model_predictor(SHIPPED_MODEL)(x, 1 / 200).any()
            assert not model_predictor_full(SHIPPED_MODEL)(x, 1 / 200).any()

    def test_a_declared_polarity_is_passed_on(self):
        x = twitch_trains()['train a3 b6 r24']
        for p in (1, -1):
            expected = predict_contractions(x, SHIPPED_MODEL, frametime=1 / 30, polarity=p)
            assert np.array_equal(model_predictor_full(SHIPPED_MODEL, polarity=p)(x, 1 / 30),
                                  expected)
            assert np.array_equal(
                model_predictor(SHIPPED_MODEL, channel=1, polarity=p)(x, 1 / 30), expected[1])


class TestLegacyConventions:
    """'symmetric' and 'raw' checkpoints keep the path they had before the polarity rule."""

    def _check_unchanged(self, path, convention):
        net = _load_model(path, ContractionNet)
        assert net.input_convention == convention
        with warnings.catch_warnings():
            # no frametime, and no warning about it: these checkpoints do not use one
            warnings.filterwarnings('error', message='.*frametime')
            for x in _traces() + [np.ones(3), np.zeros(1)]:
                got, d = predict_contractions(x, path, return_decision=True)
                assert np.array_equal(got, _reference_reading(net, x))
                assert (d.sign, d.reason) == (1, f'rule not used: {convention!r} checkpoint')
                assert math.isnan(d.score) and d.declared is None
        with pytest.raises(ValueError, match='polarity'):
            predict_contractions(_traces()[0], path, polarity=1)

    def test_symmetric_checkpoints_are_unchanged(self, tmp_path):
        net = SymmetrizedContractionNet(n_filter=8, norm='group')
        path = tmp_path / 'symmetric.pt'
        torch.save({'arch': 'SymmetrizedContractionNet', 'n_filter': 8, 'in_channels': 2,
                    'out_channels': 3, 'arch_kwargs': {'norm': 'group'},
                    'input_convention': 'symmetric', 'state_dict': net.state_dict()}, path)
        self._check_unchanged(str(path), 'symmetric')

    def test_raw_checkpoints_are_unchanged(self, tmp_path):
        net = ContractionNet(n_filter=8, in_channels=1, out_channels=2)
        path = tmp_path / 'raw.pt'
        torch.save({'arch': 'ContractionNet', 'n_filter': 8, 'in_channels': 1,
                    'out_channels': 2, 'input_convention': 'raw',
                    'state_dict': net.state_dict()}, path)
        self._check_unchanged(str(path), 'raw')


class TestLoiChannelCombination:
    """``Motion.predict_contractions`` averages the Z-band and the length traces separately and
    thresholds the more confident channel mean, so a channel of pure noise cannot dilute a clear one."""

    @staticmethod
    def _fake_net(monkeypatch, prob_of):
        import sarcasm.motion as motion
        monkeypatch.setattr(motion, 'predict_contractions', lambda d, w: np.asarray([prob_of(d)]))

    def test_weak_length_channel_does_not_dilute_zband_channel(self, monkeypatch):
        from sarcasm.motion import Motion
        state = np.r_[np.zeros(20), np.ones(20), np.zeros(20)]
        # Z-band traces carry the contraction (probability 0.7), length traces only noise (0.1);
        # the mean over all traces (0.4) would fall below the threshold
        self._fake_net(monkeypatch, lambda d: 0.7 * state if d[0] > 5 else np.full(state.size, 0.1))
        z_pos = np.full((3, state.size), 10.0)
        slen = np.full((3, state.size), 1.8)
        got = Motion.predict_contractions(z_pos, slen, 'unused.pt', threshold=0.45)
        np.testing.assert_array_equal(got, state.astype(bool))

    def test_quiescent_channels_stay_quiescent(self, monkeypatch):
        from sarcasm.motion import Motion
        self._fake_net(monkeypatch, lambda d: np.full(d.size, 0.2))
        got = Motion.predict_contractions(np.full((2, 50), 10.0), np.full((2, 50), 1.8), 'unused.pt',
                                          threshold=0.45)
        assert not got.any()
