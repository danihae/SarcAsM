# -*- coding: utf-8 -*-
"""``sarcasm.analysis.heterogeneity`` — serial/mutual correlation and oscillation spectra.

The vectorized correlation is checked against a brute-force ``np.corrcoef`` loop,
and against the legacy reduction that paired the sarcomere index with the cycle
index (which inverted the static-vs-stochastic reading).
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.analysis import heterogeneity as het


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _labels(T, spans):
    lab = np.zeros(T, dtype=int)
    for k, (a, b) in enumerate(spans, start=1):
        lab[a:b] = k
    return lab


def _beats(N, K, L, static=0.0, stochastic=0.0, seed=0, pad=5):
    """N members x K cycles of length L, each cycle a shortening pulse.

    ``static`` adds a member-specific waveform (the same every beat), ``stochastic``
    a beat-specific one (different every beat), both as fractions of the pulse.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, L)
    pulse = -np.sin(np.pi * t) ** 2
    member_shape = rng.standard_normal((N, L))
    T = pad + K * (L + pad)
    x = np.zeros((N, T))
    lab = np.zeros(T, dtype=int)
    onsets = []
    for k in range(K):
        o = pad + k * (L + pad)
        onsets.append(o)
        lab[o:o + L] = k + 1
        beat_shape = rng.standard_normal((N, L))
        x[:, o:o + L] = pulse + static * member_shape + stochastic * beat_shape
    return x, lab, np.asarray(onsets), L


def _reference_serial_mutual(x, onsets, L):
    """Brute-force r(i, j, k, l) with np.corrcoef; serial = i==j, k!=l; mutual = i!=j, k==l."""
    N, K = x.shape[0], onsets.size
    r = np.full((N, N, K, K), np.nan)
    for i in range(N):
        for j in range(N):
            for k, ok in enumerate(onsets):
                for l, ol in enumerate(onsets):
                    a, b = x[i, ok:ok + L], x[j, ol:ol + L]
                    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and a.std() > 0 and b.std() > 0:
                        r[i, j, k, l] = np.corrcoef(a, b)[0, 1]
    serial = np.nanmean([r[i, i, k, l] for i in range(N) for k in range(K) for l in range(K) if k != l])
    mutual = np.nanmean([r[i, j, k, k] for i in range(N) for j in range(N) for k in range(K) if i != j])
    return serial, mutual, r


# ---------------------------------------------------------------------------
# cycle windows
# ---------------------------------------------------------------------------

def test_cycle_windows_uses_complete_cycles_and_drops_overrunning_windows():
    T = 60
    lab = _labels(T, [(0, 6), (15, 25), (30, 42), (54, 60)])
    onsets, L = het.cycle_windows(lab)
    assert L == 11                                   # median of the two complete cycles (10, 12)
    assert onsets.tolist() == [0, 15, 30]            # the last cycle's window overruns the end
    assert het.cycle_windows(np.zeros(T, int)) == (pytest.approx(np.zeros(0)), 0)
    assert het.cycle_windows(_labels(T, [(0, 60)]))[1] == 0      # no complete cycle


# ---------------------------------------------------------------------------
# serial / mutual correlation
# ---------------------------------------------------------------------------

def test_vectorized_correlation_matches_the_brute_force_loop():
    x, lab, onsets, L = _beats(N=5, K=4, L=20, static=0.4, stochastic=0.3, seed=1)
    x[2, onsets[1] + 3] = np.nan                     # one incomplete window
    x[4, onsets[3]:onsets[3] + L] = 1.0              # one constant window
    got = het.serial_mutual_correlation(x, onsets, L)
    serial, mutual, _ = _reference_serial_mutual(x, onsets, L)
    assert got['serial'] == pytest.approx(serial, abs=1e-12)
    assert got['mutual'] == pytest.approx(mutual, abs=1e-12)
    assert got['ratio_mutual_serial'] == pytest.approx(mutual / serial, abs=1e-12)
    assert got['n_members'] == 5 and got['n_cycles'] == 4


def test_static_heterogeneity_gives_r_below_one_and_stochastic_gives_one():
    static, lab, onsets, L = _beats(N=8, K=6, L=25, static=0.5, stochastic=0.0, seed=2)
    r_static = het.serial_mutual_correlation(static, onsets, L)
    assert r_static['serial'] == pytest.approx(1.0)              # every beat identical per member
    assert r_static['mutual'] < 0.95
    assert r_static['ratio_mutual_serial'] < 0.95

    stoch, lab, onsets, L = _beats(N=8, K=6, L=25, static=0.0, stochastic=0.5, seed=3)
    r_stoch = het.serial_mutual_correlation(stoch, onsets, L)
    assert r_stoch['serial'] < 0.95
    assert r_stoch['ratio_mutual_serial'] == pytest.approx(1.0, abs=0.1)


def test_mutual_is_the_same_cycle_diagonal_not_the_legacy_one():
    """The legacy reduction ``np.diagonal(r, axis1=1, axis2=2)`` paired sarcomere j
    with cycle k. On a fixture where the two differ, ours must equal the correct one."""
    x, lab, onsets, L = _beats(N=4, K=4, L=15, static=0.6, stochastic=0.4, seed=4)
    got = het.serial_mutual_correlation(x, onsets, L)
    _, mutual, r = _reference_serial_mutual(x, onsets, L)
    legacy = np.nanmean(np.diagonal(np.tril(r.transpose(2, 3, 0, 1), -1).transpose(2, 3, 0, 1),
                                    axis1=1, axis2=2))
    assert got['mutual'] == pytest.approx(mutual, abs=1e-12)
    assert abs(got['mutual'] - legacy) > 1e-3


def test_correlation_is_nan_without_enough_members_or_cycles():
    x, lab, onsets, L = _beats(N=1, K=3, L=10)
    r = het.serial_mutual_correlation(x, onsets, L)
    assert np.isnan(r['mutual']) and np.isfinite(r['serial'])
    x, lab, onsets, L = _beats(N=3, K=1, L=10)
    r = het.serial_mutual_correlation(x, onsets, L)
    assert np.isnan(r['serial']) and np.isfinite(r['mutual']) and np.isnan(r['ratio_mutual_serial'])
    assert np.isnan(het.serial_mutual_correlation(x, np.zeros(0, int), 0)['mutual'])


# ---------------------------------------------------------------------------
# kinematics + oscillations
# ---------------------------------------------------------------------------

def test_member_kinematics_equilibrium_and_velocity():
    T, ft = 200, 0.01
    t = np.arange(T) * ft
    slen = np.stack([1.9 - 0.2 * np.clip(np.sin(2 * np.pi * t), 0, None) + 0.05 * k for k in range(3)])
    contr = np.sin(2 * np.pi * t) > 0.1
    slen[1, 50:53] = np.nan                                 # short interior gap
    slen[2, :] = np.where(t > 1.5, np.nan, slen[2])         # trailing dropout
    kin = het.member_kinematics(slen, contr, ft)
    assert np.allclose(kin['equ'], [1.9, 1.95, 2.0], atol=0.01)
    assert kin['delta_slen'].shape == (3, T) and kin['vel'].shape == (3, T)
    assert np.all(np.isnan(kin['vel'][2, t > 1.5]))           # never extrapolated
    assert np.nanmin(kin['vel'][0]) < -0.5 and np.nanmax(kin['vel'][0]) > 0.5


def test_oscillation_spectrum_finds_beat_and_single_sarcomere_peaks():
    T, ft = 1000, 0.01
    t = np.arange(T) * ft
    beat = -0.2 * np.clip(np.sin(2 * np.pi * 1.0 * t), 0, None)
    rng = np.random.default_rng(0)
    members = np.stack([beat + 0.03 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
                        for _ in range(12)])
    contr = beat < -0.02
    osc = het.oscillation_spectrum(members, contr, ft, beating_rate=1.0)
    assert osc['frequencies'].shape == osc['magnitudes_avg'].shape == osc['magnitudes_single'].shape
    assert osc['peak_1_single'] == pytest.approx(1.0, rel=0.3)
    assert osc['peak_2_single'] == pytest.approx(6.0, rel=0.25)
    assert osc['amp_2_single'] < osc['amp_1_single']
    # the random phases cancel in the mean trace, so its high-frequency content is small
    assert osc['peak_avg'] == pytest.approx(1.0, rel=0.3)


def test_oscillation_spectrum_caps_members_reproducibly():
    T, ft = 300, 0.01
    rng = np.random.default_rng(1)
    t = np.arange(T) * ft
    members = np.stack([-0.2 * np.clip(np.sin(2 * np.pi * t), 0, None)
                        + 0.03 * np.sin(2 * np.pi * 5 * t + p) + 0.005 * rng.standard_normal(T)
                        for p in rng.uniform(0, 2 * np.pi, 50)])
    a = het.oscillation_spectrum(members, np.ones(T, bool), ft, 1.0, max_members=10)
    b = het.oscillation_spectrum(members, np.ones(T, bool), ft, 1.0, max_members=10)
    np.testing.assert_array_equal(a['magnitudes_single'], b['magnitudes_single'])
    full = het.oscillation_spectrum(members, np.ones(T, bool), ft, 1.0, max_members=None)
    assert np.corrcoef(full['magnitudes_single'], a['magnitudes_single'])[0, 1] > 0.9


# ---------------------------------------------------------------------------
# per-group driver
# ---------------------------------------------------------------------------

def test_analyze_groups_shapes_and_empty_group():
    x, lab, onsets, L = _beats(N=6, K=4, L=20, static=0.3, stochastic=0.3)
    slen = 1.9 + 0.2 * x
    gid = np.array([0, 0, 0, 1, 1, 1])
    contr = np.stack([lab > 0, lab > 0, np.zeros_like(lab, bool)])
    labels = np.stack([lab, lab, np.zeros_like(lab)])
    out = het.analyze_groups(slen, gid, 3, contr, labels, np.array([1.0, 1.0, np.nan]),
                             frametime=0.01, num_scales=20)
    assert set(out) == set(het.GROUP_KEYS)
    assert out['corr_delta_slen_serial'].shape == (3,)
    assert out['oscill_magnitudes_avg'].shape == (3, 20) and out['oscill_frequencies'].shape == (20,)
    assert np.all(np.isfinite(out['ratio_delta_slen_mutual_serial'][:2]))
    assert np.isnan(out['corr_delta_slen_serial'][2])            # group without members
    assert out['corr_n_cycles'][0] == 4


# ---------------------------------------------------------------------------
# end to end on real tracks (20 kPa reference store)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('by', ['pool', 'mband'])
def test_track_motion_writes_heterogeneity_keys(motion_file_path, by):
    from sarcasm import SarcAsM
    s = SarcAsM(motion_file_path)
    if 'motion.tracks.slen' not in s.data:
        pytest.skip('20 kPa store has no tracks')
    s.analyze_track_motion(by=by, reference_frame=0)
    n = int(s.data[f'motion.groups.{by}.n'])
    for key in het.GROUP_KEYS:
        assert f'motion.{by}.{key}' in s.data, key
    serial = np.asarray(s.data[f'motion.{by}.corr_delta_slen_serial'])
    mutual = np.asarray(s.data[f'motion.{by}.corr_delta_slen_mutual'])
    ratio = np.asarray(s.data[f'motion.{by}.ratio_delta_slen_mutual_serial'])
    assert ratio.shape == serial.shape == (n,)
    assert np.isfinite(ratio).any()
    ok = np.isfinite(ratio)
    assert np.all(serial[ok] > 0) and np.allclose(ratio[ok], mutual[ok] / serial[ok])
    assert np.all(np.isnan(ratio[np.isfinite(serial) & (serial <= 0)]))
    if by == 'pool':                                  # distinct sarcomeres: R in the paper's range
        assert 0.3 < serial[0] < 0.8 and 0.5 < ratio[0] < 1.2
    assert np.asarray(s.data[f'motion.{by}.oscill_magnitudes_single']).shape == (n, 60)
