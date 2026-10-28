"""Tests for the polarity rule: which side of a trace is rest, decided once per trace.

Tests on real recordings skip when their files are not present; :func:`regression_set` uses
whatever is.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os

import numpy as np
import pytest
from scipy.signal import resample_poly

from contraction_net.data import EHT_TEST_GROUPS, ContractionDataset
from contraction_net.polarity import (PolarityRule, _local_skewness, _odd_frames,
                                      decide_polarity, polarity_score)
from contraction_net.simulation import _ar1, make_stress_set

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA = os.path.join(REPO, 'training_data')
CARDIOTRACE_PLATES = os.path.join(TRAINING_DATA, 'cta', 'cta_sim_plates.npz')
VIDEO = os.path.join(TRAINING_DATA, 'cta', 'video_traces.npz')
EHT_DIR = os.environ.get('CONTRACTION_EHT_DIR', os.path.join(TRAINING_DATA, 'eht'))
CORPUS = os.path.join(TRAINING_DATA, 'drug_corpus_v3.npz')
#: CardioTrace's real FOC48 plate: 48 wells, 60 s at 30 fps, pole distance (rest high).
FOC48 = os.environ.get('CONTRACTION_FOC48_CSV', os.path.join(TRAINING_DATA, 'cta', '20220603a_stim0.csv'))
FOC48_WELLS = [f'{row}{col}' for row in 'ABCDEFGH' for col in range(1, 7)]
#: Wells that do not beat, by eye and by the acquisition software's peak count.
FOC48_NON_BEATING = ('A1', 'A4', 'A5', 'B4', 'D5', 'E2', 'G6')
#: Odd wells left out of the clear ones; A3 shows upward excursions of unknown origin.
FOC48_AMBIGUOUS = ('A2', 'A3', 'C4', 'C6', 'F6')

MARGIN = PolarityRule().margin


# --------------------------------------------------------------------------------------
# trace sets, shared with test_contraction_net and test_decoding
# --------------------------------------------------------------------------------------

def edge_cases():
    """Degenerate and boundary traces, keyed by name; frametime 1/30 s."""
    rng = np.random.default_rng(0)
    cases = {f'n={n}': rng.standard_normal(n) for n in (1, 5, 31, 40, 59, 60, 61)}
    cases.update({
        'zeros': np.zeros(300),
        'constant': np.full(300, 7.0),
        'zeros + outlier': np.r_[np.zeros(150), 5.0, np.zeros(149)],
        'zeros + 2-frame blips': np.r_[np.zeros(100), 1.0, 1.0, np.zeros(100), 1.0, 1.0,
                                       np.zeros(96)],
        'two-level 91/9': np.r_[np.zeros(273), np.ones(27)],
        'quantised noise': np.round(rng.standard_normal(900) / 2.0 + 0.35) * 2.0,
    })
    return cases


def twitch_trains():
    """60 EHT-like twitch trains at 30 fps, rest high: raised-cosine onset over ``a`` frames,
    relaxation over ``b`` and flat rest for ``r``, 1800 frames each."""
    n = 1800
    out = {}
    for a in (3, 6, 12):
        for b in (3, 6, 12, 24):
            for r in (3, 6, 12, 24, 48):
                one = np.r_[-0.5 * (1 - np.cos(np.pi * np.arange(1, a + 1) / a)),
                            -0.5 * (1 + np.cos(np.pi * np.arange(1, b + 1) / b)), np.zeros(r)]
                period = a + b + r
                x = np.tile(one, n // period + 2)[period // 2:period // 2 + n]
                out[f'train a{a} b{b} r{r}'] = \
                    x + 0.01 * np.random.default_rng(0).standard_normal(n)
    return out


def structure_free():
    """24 traces of 900 frames without any contraction: AR(1) noise, drift, steps, outliers."""
    rng = np.random.default_rng(7)
    n = 900
    out = {}
    for rho in (0.0, 0.5, 0.9):
        for i in range(4):
            out[f'ar1 rho {rho} {i}'] = _ar1(n, rho, rng)
    t = np.linspace(0, 1, n)
    for i in range(4):
        base = _ar1(n, 0.5, rng)
        out[f'drift {i}'] = base + 3 * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t
                                              + rng.uniform(0, 6.3))
        steps = base.copy()
        for _ in range(3):
            steps[rng.integers(1, n):] += rng.choice([-1, 1]) * rng.uniform(1, 4)
        out[f'steps {i}'] = steps
        outliers = base.copy()
        outliers[rng.integers(0, n, 10)] += rng.normal(0, 8, 10)
        out[f'outliers {i}'] = outliers
    return out


@functools.lru_cache(maxsize=None)
def regression_set():
    """The mirror-test regression set, as a tuple of ``(name, trace, frametime)``.

    20 stress traces (4 quiescent), one simulated CardioTrace well per plate row, the first 900
    frames of the video, 5 EHT traces outside the held-out recordings, 15 corpus traces (5
    per type), the 60 twitch trains, the 24 structure-free traces and the edge cases. The
    real-data parts are included when their files are present.
    """
    out = []
    stress = make_stress_set(200, seed=99999)
    picked = ([tr for tr in stress if tr.label.any()][:16]
              + [tr for tr in stress if not tr.label.any()][:4])
    out += [(f'stress {i}', tr.signal, tr.frametime) for i, tr in enumerate(picked)]
    if os.path.exists(CARDIOTRACE_PLATES):
        plates = np.load(CARDIOTRACE_PLATES)
        wells = list(plates['well'])
        for row in 'ABCDEFGH':
            i = wells.index(f'{row}1')
            out.append((f'plate {row}1', plates['traces'][i], float(plates['frametime'][i])))
    if os.path.exists(VIDEO):
        video = np.load(VIDEO)
        out.append(('video, first 900 frames', video['whole'][:900], float(video['frametime'])))
    if os.path.isdir(EHT_DIR):
        eht = [s for s, _, g in ContractionDataset.load_eht(EHT_DIR) if g not in EHT_TEST_GROUPS]
        out += [(f'eht {i}', s, 1 / 30) for i, s in enumerate(eht[:5])]
    if os.path.exists(CORPUS):
        corpus = ContractionDataset.load_corpus(CORPUS)
        types = np.asarray(corpus['trace_type'])
        for kind in ('group_slen', 'track_slen', 'track_z_pos'):
            idx = np.flatnonzero(types == kind)
            for i in idx[np.linspace(0, idx.size - 1, 5).astype(int)]:
                out.append((f'corpus {kind} {i}', corpus['signals'][i],
                            float(corpus['frametime'][i])))
    out += [(name, x, 1 / 30) for name, x in twitch_trains().items()]
    out += [(name, x, 1 / 30) for name, x in structure_free().items()]
    out += [(f'edge {name}', x, 1 / 30) for name, x in edge_cases().items()]
    return tuple((name, np.asarray(x, dtype=np.float64), float(ft)) for name, x, ft in out)


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f'{path} not present')


@functools.lru_cache(maxsize=None)
def _plates():
    """Simulated CardioTrace plates: ``(trace, frametime, well, true beat count)`` per well."""
    d = np.load(CARDIOTRACE_PLATES)
    return [(np.asarray(x, float), float(ft), str(w), int(n))
            for x, ft, w, n in zip(d['traces'], d['frametime'], d['well'], d['true_n'])]


@functools.lru_cache(maxsize=None)
def _eht():
    return [np.asarray(s, float) for s, _, _ in ContractionDataset.load_eht(EHT_DIR)]


@functools.lru_cache(maxsize=None)
def _video():
    d = np.load(VIDEO)
    return {k: np.asarray(d[k], float) for k in ('whole', 'roi')}, float(d['frametime'])


@functools.lru_cache(maxsize=None)
def _foc48():
    a = np.loadtxt(FOC48, delimiter=',')
    return {w: a[:, 1 + j] for j, w in enumerate(FOC48_WELLS)}, float(np.diff(a[:, 0]).mean())


@functools.lru_cache(maxsize=None)
def _corpus_tally(kind):
    """``(beating traces, decided, decided wrong)`` for one corpus trace type; a trace's rest
    side is the side its unlabelled frames sit on."""
    corpus = ContractionDataset.load_corpus(CORPUS)
    n = decided = wrong = 0
    for i in np.flatnonzero(np.asarray(corpus['trace_type']) == kind):
        x = np.asarray(corpus['signals'][i], float)
        label = np.asarray(corpus['labels'][i], bool)
        if not 0 < label.mean() < 1:
            continue
        truth = 1 if np.median(x[label]) < np.median(x[~label]) else -1
        sign = decide_polarity(x, float(corpus['frametime'][i])).sign
        n += 1
        decided += sign != 0
        wrong += sign == -truth
    return n, decided, wrong


# --------------------------------------------------------------------------------------
# the rule and its windows
# --------------------------------------------------------------------------------------

class TestRule:
    def test_version_1_constants(self):
        assert dataclasses.asdict(PolarityRule()) == {
            'version': 1, 'reference_s': 2.0, 'smooth_s': 0.1, 'quantile': 10.0,
            'margin': 0.10, 'min_frames': 32, 'min_seconds': 2.0}

    def test_checkpoint_roundtrip_and_defaults(self):
        assert PolarityRule.from_checkpoint(None) == PolarityRule()
        stored = dataclasses.asdict(PolarityRule(margin=0.12))
        assert PolarityRule.from_checkpoint(stored) == PolarityRule(margin=0.12)

    @pytest.mark.parametrize('stored', [{'version': 2}, {'margin': 0.1},
                                        {'version': 1, 'reference_s': 3.0, 'window_s': 4.0}])
    def test_unknown_rules_are_refused(self, stored):
        with pytest.raises(ValueError, match='polarity_rule'):
            PolarityRule.from_checkpoint(stored)

    def test_rule_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PolarityRule().margin = 0.2


class TestWindows:
    @pytest.mark.parametrize('seconds,fps,frames', [(2.0, 30, 61), (2.0, 61, 123), (2.0, 15, 31),
                                                    (0.1, 30, 3), (0.1, 61, 7), (0.1, 15, 1)])
    def test_nearest_odd_frame_count(self, seconds, fps, frames):
        assert _odd_frames(seconds, 1 / fps, 10 ** 6) == frames

    @pytest.mark.parametrize('fps', [30, 93, 99, 186])
    def test_a_tie_takes_the_larger_count(self, fps):
        # 2 s at a whole frame rate lies halfway between two odd counts; at 93, 99 and 186
        # fps 2 / (1 / fps) rounds below 2 * fps, which must not decide the tie
        assert _odd_frames(2.0, 1 / fps, 10 ** 6) == 2 * fps + 1

    def test_capped_at_the_trace_and_at_least_one(self):
        assert _odd_frames(2.0, 1 / 30, 41) == 41
        assert _odd_frames(2.0, 1 / 30, 40) == 39
        assert _odd_frames(2.0, 1 / 30, 1) == 1
        assert _odd_frames(0.01, 1 / 30, 900) == 1


# --------------------------------------------------------------------------------------
# guards and the decision itself
# --------------------------------------------------------------------------------------

class TestGuards:
    """Degenerate traces abstain with a reason, and x and -x always meet the same guard."""

    @pytest.mark.parametrize('n', range(1, 32))
    def test_fewer_than_32_frames_is_too_short(self, n):
        x = np.random.default_rng(n).standard_normal(n)
        for v in (x, -x):
            # one frame per second, so only the frame count can stop these
            d = decide_polarity(v, 1.0)
            assert (d.sign, d.reason) == (0, 'too_short') and math.isnan(d.score)

    @pytest.mark.parametrize('n,fps,short', [(40, 30, True), (59, 30, True), (60, 30, False),
                                             (61, 30, False), (199, 100, True),
                                             (200, 100, False), (97, 49, True),
                                             # exactly 2 s, although n / fps rounds below it
                                             (98, 49, False), (206, 103, False)])
    def test_shorter_than_2_s_is_too_short(self, n, fps, short):
        x = np.random.default_rng(n).standard_normal(n)
        for v in (x, -x):
            assert (decide_polarity(v, 1 / fps).reason == 'too_short') == short

    @pytest.mark.parametrize('name', ['zeros', 'constant', 'zeros + outlier',
                                      'zeros + 2-frame blips', 'two-level 91/9'])
    def test_flat_traces(self, name):
        x = edge_cases()[name]
        for v in (x, -x):
            d = decide_polarity(v, 1 / 30)
            assert (d.sign, d.reason) == (0, 'flat') and math.isnan(d.score)

    def test_a_two_level_trace_with_spread_is_not_flat(self):
        d = decide_polarity(np.r_[np.zeros(255), np.ones(45)], 1 / 30)
        assert d.reason != 'flat'

    def test_guards_apply_to_a_declared_polarity(self):
        d = decide_polarity(np.zeros(300), 1 / 30, polarity=1)
        assert (d.sign, d.reason, d.declared) == (0, 'flat', 1)
        d = decide_polarity(np.ones(20), 1 / 30, polarity=-1)
        assert (d.sign, d.reason, d.declared) == (0, 'too_short', -1)

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
    def test_non_finite_input_raises(self, bad):
        x = np.r_[np.zeros(50), bad, np.ones(50)]
        with pytest.raises(ValueError, match='NaN or infinite'):
            decide_polarity(x, 1 / 30)
        with pytest.raises(ValueError, match='NaN or infinite'):
            polarity_score(x, 1 / 30)

    @pytest.mark.parametrize('frametime', [None, 0.0, -1 / 30, np.nan])
    def test_frametime_must_be_positive(self, frametime):
        with pytest.raises(ValueError, match='frametime'):
            decide_polarity(np.random.default_rng(0).standard_normal(300), frametime)


class TestDecision:
    TRAIN = 'train a3 b6 r24'

    def test_rest_high_is_positive_and_the_mirror_negative(self):
        x = twitch_trains()[self.TRAIN]
        up, down = decide_polarity(x, 1 / 30), decide_polarity(-x, 1 / 30)
        assert (up.sign, up.reason) == (1, 'decided') and up.score > MARGIN
        assert (down.sign, down.reason) == (-1, 'decided') and down.score == -up.score
        assert up.margin == MARGIN and up.declared is None

    def test_a_score_equal_to_the_margin_abstains_for_both_signs(self):
        x = twitch_trains()[self.TRAIN]
        s = polarity_score(x, 1 / 30)
        for v in (x, -x):
            d = decide_polarity(v, 1 / 30, margin=abs(s))
            assert (d.sign, d.reason) == (0, 'ambiguous')

    def test_margin_zero_decides_by_the_sign_of_the_score(self):
        for name, x in structure_free().items():
            s = polarity_score(x, 1 / 30)
            assert decide_polarity(x, 1 / 30, margin=0.0).sign == int(np.sign(s)), name
        # -y equals y reversed, and at 15 fps no boxcar runs, so the score is exactly 0
        v = np.random.default_rng(1).standard_normal(40)
        y = np.r_[v, -v[::-1]]
        assert polarity_score(y, 1 / 15) == 0.0
        d = decide_polarity(y, 1 / 15, margin=0.0)
        assert (d.sign, d.reason) == (0, 'ambiguous')

    def test_margin_defaults_to_the_rule(self):
        x = twitch_trains()[self.TRAIN]
        # |score| <= 1 always, so a rule with margin 1 abstains on everything
        d = decide_polarity(x, 1 / 30, rule=PolarityRule(margin=1.0))
        assert d.margin == 1.0 and d.sign == 0
        assert decide_polarity(x, 1 / 30, margin=0.0, rule=PolarityRule(margin=1.0)).sign == 1

    def test_declared_polarity_orients_and_reports_disagreement(self):
        x = twitch_trains()[self.TRAIN]
        s = polarity_score(x, 1 / 30)
        agree = decide_polarity(x, 1 / 30, polarity=1)
        assert (agree.sign, agree.reason, agree.declared, agree.score) == (1, 'declared', 1, s)
        disagree = decide_polarity(x, 1 / 30, polarity=-1)
        assert (disagree.sign, disagree.reason) == (-1, 'declared, data disagree')
        # the joint mirror: (-x, -p) gets the opposite sign and the same reason
        assert decide_polarity(-x, 1 / 30, polarity=1).reason == 'declared, data disagree'
        assert decide_polarity(-x, 1 / 30, polarity=-1).reason == 'declared'
        # an undecidable trace never disagrees beyond the margin
        noise = structure_free()['ar1 rho 0.0 0']
        assert abs(polarity_score(noise, 1 / 30)) <= MARGIN
        assert decide_polarity(noise, 1 / 30, polarity=-1).reason == 'declared'

    def test_invalid_arguments_raise(self):
        x = twitch_trains()[self.TRAIN]
        with pytest.raises(ValueError, match='polarity'):
            decide_polarity(x, 1 / 30, polarity=0)
        with pytest.raises(ValueError, match='margin'):
            decide_polarity(x, 1 / 30, margin=-0.1)


# --------------------------------------------------------------------------------------
# exactness and failing closed
# --------------------------------------------------------------------------------------

class TestExactness:
    def test_score_is_odd_and_the_antisymmetrisation_is_load_bearing(self):
        """The raw statistic is not exactly odd on some trace, the score on none, so nobody
        can 'simplify' the score back to g."""
        raw_not_odd = 0
        for name, x, ft in regression_set():
            for v in (x, x.astype(np.float32)):
                assert polarity_score(-v, ft) == -polarity_score(v, ft), name
            if x.size >= 32:
                rule = PolarityRule()
                raw_not_odd += _local_skewness(-x, ft, rule) != -_local_skewness(x, ft, rule)
        assert raw_not_odd >= 1

    def test_time_reversal_and_affine_maps(self):
        """Time reversal moves the score by at most 1e-12, an affine map by at most 1e-9. At a = +-1e-3 with b = 1e3 the offset dwarfs the scaled trace, so rounding
        ``a * x + b`` to float64 already changes the trace (a stress trace of 0.06 µm spread
        keeps about nine digits) and its score by up to 2e-8. For that map the score is
        compared with that of the trace the map carries, ``(y - b) / a`` (measured: within
        4e-16)."""
        rng = np.random.default_rng(4)
        walks = [(f'walk {i}', np.cumsum(rng.standard_normal(int(rng.integers(64, 3000)))), 1 / 30)
                 for i in range(300)]
        for name, x, ft in list(regression_set()) + walks:
            s = polarity_score(x, ft)
            assert abs(polarity_score(x[::-1], ft) - s) <= 1e-12, name
            for a in (1e-3, -1e-3, 1e3, -1e3):
                for b in (0.0, 7.0, 1e3):
                    y = a * x + b
                    carried = (polarity_score((y - b) / a, ft) if (abs(a), b) == (1e-3, 1e3)
                               else s)
                    ds = polarity_score(y, ft) - np.sign(a) * carried
                    assert abs(ds) <= 1e-9, (name, a, b)


class TestFailsClosed:
    """Half-wave-symmetric waves carry no polarity cue, so they must abstain."""

    T = np.arange(1800) / 30

    @staticmethod
    def _noise(f, phase, n):
        return np.random.default_rng(int(100 * f) * 10 + int(phase * 4)).standard_normal(n)

    @pytest.mark.parametrize('shape', ['sine', 'triangle'])
    def test_sine_and_triangle_waves_abstain(self, shape):
        for f in np.arange(0.5, 3.01, 0.1):
            for phase in (0.0, 0.25, 0.5, 0.75):
                u = 2 * np.pi * (f * self.T + phase)
                y = np.sin(u) if shape == 'sine' else 2 / np.pi * np.arcsin(np.sin(u))
                for noise in (0.0, 0.05, 0.1, 0.3):
                    d = decide_polarity(y + noise * self._noise(f, phase, y.size), 1 / 30)
                    assert (d.sign, d.reason) == (0, 'ambiguous'), (f, phase, noise, d.score)

    @pytest.mark.parametrize('f', [1.0, 1.5, 2.5])
    def test_square_waves_with_a_whole_frame_half_period_abstain(self, f):
        for phase in (0.0, 0.25, 0.5, 0.75):
            y = np.sign(np.sin(2 * np.pi * (f * self.T + phase)))
            for noise in (0.0, 0.05, 0.1, 0.3):
                d = decide_polarity(y + noise * self._noise(f, phase, y.size), 1 / 30)
                assert (d.sign, d.reason) == (0, 'ambiguous'), (phase, noise, d.score)


# --------------------------------------------------------------------------------------
# real recordings: stability and accuracy at the frozen margin
# --------------------------------------------------------------------------------------

def _perturbed(x, ft, kind, rng):
    if kind == '2x':
        return resample_poly(x - x.mean(), 2, 1) + x.mean(), ft / 2
    if kind == '0.5x':
        return x[::2].copy(), ft * 2
    if kind == 'first half':
        return x[:x.size // 2], ft
    if kind == 'second half':
        return x[x.size // 2:], ft
    if kind == '+15% noise':
        spread = np.percentile(x, 95) - np.percentile(x, 5)
        return x + 0.15 * spread * rng.standard_normal(x.size), ft
    if kind == 'shift 7':
        return x[7:], ft
    if kind == 'reverse':
        return x[::-1].copy(), ft
    raise KeyError(kind)


class TestStability:
    def test_no_perturbation_flips_a_decision(self):
        """Resampling, halving, noise, a shift or reversal can move a trace between decided
        and abstained, never to the other sign. The beating CardioTrace wells, EHT and the
        video, plus the FOC48 clear wells and 400 beating corpus traces when present."""
        for path in (CARDIOTRACE_PLATES, VIDEO, EHT_DIR):
            _require(path)
        traces = [(x, ft) for x, ft, _, n in _plates() if n > 0]
        traces += [(x, 1 / 30) for x in _eht()]
        video, vft = _video()
        traces += [(x, vft) for x in video.values()]
        if os.path.exists(FOC48):
            wells, ft = _foc48()
            traces += [(wells[w], ft) for w in FOC48_WELLS
                       if w not in FOC48_NON_BEATING + FOC48_AMBIGUOUS]
        if os.path.exists(CORPUS):
            corpus = ContractionDataset.load_corpus(CORPUS)
            beating = [i for i, label in enumerate(corpus['labels']) if 0 < label.mean() < 1]
            for i in np.random.default_rng(3).choice(beating, 400, replace=False):
                traces.append((np.asarray(corpus['signals'][i], float),
                               float(corpus['frametime'][i])))
        moved = total = 0
        for kind in ('2x', '0.5x', 'first half', 'second half', '+15% noise', 'shift 7',
                     'reverse'):
            for i, (x, ft) in enumerate(traces):
                before = decide_polarity(x, ft).sign
                xp, ftp = _perturbed(x, ft, kind, np.random.default_rng(100 + i))
                after = decide_polarity(xp, ftp).sign
                assert before * after != -1, (kind, i)
                moved += (before == 0) != (after == 0)
                total += 1
        assert moved <= 0.05 * total


class TestAccuracy:
    """The frozen rule on recordings whose rest side is known."""

    def test_simulated_plates(self):
        _require(CARDIOTRACE_PLATES)
        decisions = [(w, decide_polarity(x, ft)) for x, ft, w, n in _plates() if n > 0]
        assert not [w for w, d in decisions if d.sign == -1]
        # measured: D5 of one plate and H5 of both (2-3 beats) abstain
        assert sum(d.sign == 1 for _, d in decisions) >= len(decisions) - 3

    def test_eht(self):
        _require(EHT_DIR)
        signs = np.array([decide_polarity(x, 1 / 30).sign for x in _eht()])
        assert not (signs == -1).any()
        assert (signs == 1).mean() >= 0.95

    def test_video(self):
        _require(VIDEO)
        video, ft = _video()
        for x in video.values():
            d = decide_polarity(x, ft)
            assert d.sign == 1 and d.score >= 0.3

    def test_foc48_plate(self):
        _require(FOC48)
        wells, ft = _foc48()
        for w in FOC48_NON_BEATING:
            assert decide_polarity(wells[w], ft).sign == 0, w
        clear = [w for w in FOC48_WELLS if w not in FOC48_NON_BEATING + FOC48_AMBIGUOUS]
        assert len(clear) == 36
        assert all(decide_polarity(wells[w], ft).sign == 1 for w in clear)

    @pytest.mark.parametrize('kind,min_coverage', [('group_slen', 0.95), ('track_slen', 0.85),
                                                   ('track_z_pos', 0.95)])
    def test_corpus_coverage(self, kind, min_coverage):
        """Every beating corpus trace of one type, its rest side known from the labels.
        Measured: 0.965 / 0.913 / 0.971 decided."""
        _require(CORPUS)
        n, decided, _ = _corpus_tally(kind)
        assert decided / n >= min_coverage

    @pytest.mark.parametrize('kind,max_wrong', [('group_slen', 0.005), ('track_slen', 0.03),
                                                ('track_z_pos', 0.006)])
    def test_corpus_wrong_share(self, kind, max_wrong):
        """Wrong among decided. Measured: 0 / 1.6% / 0.53%.

        For track_z_pos all 1547 traces give 8 wrong in 1502 decided (0.53%), 6 of them at
        duty 0.68 or more, where a trace spends longer contracted than at rest; the bound is
        0.6%.
        """
        _require(CORPUS)
        _, decided, wrong = _corpus_tally(kind)
        assert wrong / decided <= max_wrong
