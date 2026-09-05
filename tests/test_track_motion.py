# -*- coding: utf-8 -*-
"""``SarcAsM.get_track_motion`` and the LOI plots it feeds.

The synthetic-chain ``Motion`` returned by ``get_track_motion`` is the object behind
the per-fibre panel of the GUI and every ``Plots.plot_*`` LOI view. Its bugs this
cycle (cumulative ``z_pos`` blanking a chain, time-median anchoring, coverage holes,
the negative-index window) were all found by hand; these tests pin the contract.
"""
from __future__ import annotations

import types

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sarcasm import SarcAsM  # noqa: E402
from sarcasm.motion import Motion  # noqa: E402
from sarcasm.plotting.plots import Plots  # noqa: E402


# ---------------------------------------------------------------------------
# contraction-centred plot windows (no file IO)
# ---------------------------------------------------------------------------

def _fake_analyzed_motion(T=120, K=4, frametime=0.01):
    """A ``Motion`` carrying everything the LOI plots read, with a cycle at frame 0."""
    m = Motion.__new__(Motion)
    m.metadata = types.SimpleNamespace(frametime=frametime)
    t = np.arange(T) * frametime
    base = 1.9 - 0.2 * np.clip(np.sin(2 * np.pi * t / 0.5), 0, None)
    slen = np.stack([base + 0.01 * k for k in range(K)])
    z_pos = np.vstack([np.zeros(T), np.cumsum(slen, axis=0)])
    delta = slen - np.median(slen, axis=1, keepdims=True)
    vel = np.gradient(slen, frametime, axis=1)
    contr = base < 1.85
    onsets = np.flatnonzero(np.diff(np.r_[0, contr.astype(int)]) == 1)
    m.loi_data = {
        'time': t, 'z_pos': z_pos, 'slen': slen, 'synthetic': True,
        'n_sarcomeres': K, 'delta_slen': delta, 'delta_slen_avg': delta.mean(axis=0),
        'vel': vel, 'vel_avg': vel.mean(axis=0), 'contr': contr,
        'start_contr': onsets * frametime, 'start_contr_frame': onsets,
        'n_contr': int(onsets.size), 'contr_complete': np.ones(onsets.size),
    }
    return m


def test_contr_window_resolves_none_bounds_to_the_recording():
    m = _fake_analyzed_motion()
    T = len(m.loi_data['time'])
    (lo, hi), i0, i1 = Plots._contr_window(m, 0, (None, None))
    assert (lo, i0) == (0.0, 0)
    assert i1 == T and hi == pytest.approx(T * m.metadata.frametime)
    # a relative lower bound with an open upper bound
    (lo, hi), i0, i1 = Plots._contr_window(m, 1, (-0.05, None))
    assert lo == pytest.approx(m.loi_data['start_contr'][1] - 0.05)
    assert i1 == T and i0 < i1


def test_contr_window_clamps_a_negative_lead_in():
    m = _fake_analyzed_motion()
    onset = m.loi_data['start_contr'][0]
    assert onset < 0.1                                   # the first cycle begins near frame 0
    (lo, hi), i0, i1 = Plots._contr_window(m, 0, (-0.1, 0.3))
    assert lo < 0 and i0 == 0                            # never a negative slice start
    assert i1 == int((onset + 0.3) / m.metadata.frametime)


@pytest.mark.parametrize('plot', [Plots.plot_z_pos, Plots.plot_overlay_delta_slen,
                                  Plots.plot_overlay_velocity, Plots.plot_phase_space])
def test_centred_views_render_with_default_limits(plot):
    """``number_contr`` with the default ``t_lim`` used to raise on ``None`` bounds."""
    m = _fake_analyzed_motion()
    for number_contr in (None, 0, 1):
        ax = plt.figure().gca()
        plot(ax, m, number_contr=number_contr)
        assert ax.lines, f'{plot.__name__} drew nothing for number_contr={number_contr}'
        plt.close('all')


def test_full_recording_is_the_default_time_axis():
    m = _fake_analyzed_motion(T=300)
    ax = plt.figure().gca()
    Plots.plot_z_pos(ax, m)
    x0, x1 = ax.get_xlim()
    assert x0 == 0 and x1 >= m.loi_data['time'][-1]
    plt.close('all')


def test_plot_z_pos_has_no_kymograph_option():
    m = _fake_analyzed_motion()
    with pytest.raises(TypeError):
        Plots.plot_z_pos(plt.figure().gca(), m, show_kymograph=True)
    plt.close('all')


# ---------------------------------------------------------------------------
# get_track_motion on real tracks (20 kPa reference store)
# ---------------------------------------------------------------------------

class TestGetTrackMotion:

    @pytest.fixture(scope='class')
    def sarc(self, motion_file_path_class):
        s = SarcAsM(motion_file_path_class)
        if 'motion.tracks.slen' not in s.data or s.data.get('structure.myofibril.lines') is None:
            pytest.skip('20 kPa store has no tracks / myofibril lines')
        s.group_tracks(by='myofibril', reference_frame=0)
        return s

    @pytest.fixture(scope='class')
    def motion(self, sarc):
        return sarc.get_track_motion(0, analyze=True)

    @pytest.fixture(scope='class')
    def raw(self, sarc):
        """The chain before ``get_trajectories`` smooths ``slen`` in place."""
        return sarc.get_track_motion(0)

    def test_requires_a_chain_grouping(self, motion_file_path_class):
        s = SarcAsM(motion_file_path_class)
        s.group_tracks(by='pool')
        with pytest.raises(ValueError, match='myofibril'):
            s.get_track_motion(0)

    def test_chain_geometry(self, sarc, raw):
        ld = raw.loi_data
        K = int(np.count_nonzero(sarc.data['motion.tracks.group_id'] == 0))
        T = int(np.asarray(sarc.data['motion.tracks.slen']).shape[1])
        assert ld['synthetic'] is True
        assert ld['n_sarcomeres'] == K and K >= 3
        assert ld['slen'].shape == (K, T)
        assert ld['z_pos'].shape == (K + 1, T)
        assert np.nanmin(ld['z_pos']) == pytest.approx(0.0)          # legacy z_pos[0] == 0
        # member lengths are the tracks' own, not re-derived from z_pos
        members = np.flatnonzero(sarc.data['motion.tracks.group_id'] == 0)
        order = np.asarray(sarc.data['motion.tracks.group_order'])[members]
        tracks_slen = np.asarray(sarc.data['motion.tracks.slen'], float)[members[np.argsort(order)]]
        assert np.allclose(np.nan_to_num(ld['slen']), np.nan_to_num(tracks_slen), atol=1e-6)
        # a missing member blanks only its own row: NaN rows never exceed missing members
        nan_rows = np.isnan(ld['slen']).sum(axis=0)
        assert nan_rows.max() < K
        # consecutive boundaries advance by ~one sarcomere where both are defined
        spacing = np.diff(ld['z_pos'], axis=0)
        ok = np.isfinite(spacing) & np.isfinite(ld['slen'])
        assert np.nanmedian(np.abs(spacing[ok] - ld['slen'][ok])) < 0.15

    def test_analysis_keys(self, motion):
        ld = motion.loi_data
        for key in ('contr', 'start_contr', 'n_contr', 'n_contr_complete', 'beating_rate',
                    'delta_slen', 'delta_slen_avg', 'vel', 'vel_avg', 'time_contr'):
            assert key in ld, key
        assert ld['n_contr'] >= ld['n_contr_complete'] >= 0
        assert ld['n_contr'] >= 1
        assert ld['contr'].shape == ld['time'].shape

    def test_summary_figure_renders(self, motion, tmp_path):
        out = tmp_path / 'summary' / 'summary_loi.png'        # keep the reference store clean
        Plots.plot_loi_summary_motion(motion, file_path=str(out))
        assert out.exists() and out.stat().st_size > 0
        plt.close('all')

    def test_every_loi_plot_accepts_the_synthetic_chain(self, motion):
        for fn in (Plots.plot_z_pos, Plots.plot_delta_slen, Plots.plot_overlay_delta_slen,
                   Plots.plot_overlay_velocity, Plots.plot_phase_space, Plots.plot_slen):
            fn(plt.figure().gca(), motion)
        plt.close('all')


# ---------------------------------------------------------------------------
# per-track kinematics + raster (20 kPa reference store)
# ---------------------------------------------------------------------------

class TestTrackKinematics:

    @pytest.fixture(scope='class')
    def sarc(self, motion_file_path_class):
        s = SarcAsM(motion_file_path_class)
        if 'motion.tracks.slen' not in s.data:
            pytest.skip('20 kPa store has no tracks')
        s.analyze_track_motion(by='pool')                  # pooled cycles for the equilibrium / raster
        s.group_tracks(by='myofibril', reference_frame=0, min_group_size=6)
        return s

    def test_kinematics_shapes_and_equilibrium(self, sarc):
        kin = sarc.get_track_kinematics()
        n, T = np.asarray(sarc.data['motion.tracks.slen']).shape
        for key in ('slen', 'delta_slen', 'vel'):
            assert kin[key].shape == (n, T)
        assert kin['equ'].shape == kin['coverage'].shape == (n,)
        assert kin['contr'].shape == (T,) and kin['contr'].any()
        stored = np.asarray(sarc.data['motion.tracks.slen'], float)
        assert np.all(np.isnan(kin['slen'][np.isnan(stored)]))          # never invents beyond the tracks
        assert np.all(np.isnan(kin['delta_slen'][np.isnan(stored)]))
        assert np.nanmin(kin['slen']) >= 1.2 and np.nanmax(kin['slen']) <= 3.0
        full = kin['coverage'] > 0.9
        assert np.nanmedian(np.abs(kin['delta_slen'][full])) < 0.1        # centred on the resting length
        assert 1.5 < np.nanmedian(kin['equ'][full]) < 2.1

    @pytest.mark.parametrize('sort_by', ['time_to_peak', 'amplitude', 'group'])
    def test_cycle_raster(self, sarc, sort_by):
        ax = plt.figure().gca()
        im = Plots.plot_track_raster(ax, sarc, sort_by=sort_by)
        assert im is not None
        h, w = im.get_array().shape
        n_cycle = int(np.median(np.diff(
            np.flatnonzero(np.diff(np.r_[0, np.asarray(sarc.data['motion.pool.contr'])[0].astype(int)]) == 1))))
        assert w == n_cycle and h > 100
        plt.close('all')

    def test_full_recording_raster_and_errors(self, sarc):
        ax = plt.figure().gca()
        im = Plots.plot_track_raster(ax, sarc, cycle_average=False, value='vel')
        assert im.get_array().shape[1] == np.asarray(sarc.data['motion.tracks.slen']).shape[1]
        with pytest.raises(ValueError, match='value'):
            Plots.plot_track_raster(ax, sarc, value='nope')
        with pytest.raises(ValueError, match='sort_by'):
            Plots.plot_track_raster(ax, sarc, sort_by='nope')
        plt.close('all')
