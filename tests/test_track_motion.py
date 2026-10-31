# -*- coding: utf-8 -*-
"""The plots of one group (``sarc.motion.groups(kind)[i]``) and of the member tracks.

A group's plots read its stored results and its members' kinematics (SARC-28). Bugs of the
per-group view found by hand this cycle (a chain blanked by one undefined member, the negative-index
window, ...) are pinned here.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sarcasm import SarcAsM  # noqa: E402
from sarcasm.plotting.plots import Plots  # noqa: E402
from .test_grouped_motion import _fake_structure  # noqa: E402


# ---------------------------------------------------------------------------
# contraction-centred plot windows (synthetic beating group, no file IO)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def beating_group():
    """A myofibril group of a synthetic 20 s recording with 25 beats."""
    sarc = _fake_structure(n_tracks=6, T=400, frametime=0.05)
    sarc.data['structure.myofibril.lines'] = [[np.array([0, 1, 2, 3, 4, 5])]]
    g = sarc.motion.analyze('myofibril', reference_frame=0)[0]
    assert g.n_contr > 2 and g.ordered
    return g


def test_contr_window_resolves_none_bounds_to_the_recording(beating_group):
    g = beating_group
    T = np.asarray(g.tracks.slen).shape[1]
    onsets = Plots._series(g)['onsets'] * 0.05
    (lo, hi), i0, i1 = Plots._contr_window(g, 0, (None, None))
    assert (lo, i0) == (0.0, 0)
    assert i1 == T and hi == pytest.approx(T * 0.05)
    (lo, hi), i0, i1 = Plots._contr_window(g, 1, (-0.05, None))
    assert lo == pytest.approx(onsets[1] - 0.05)
    assert i1 == T and i0 < i1


def test_contr_window_clamps_a_negative_lead_in(beating_group):
    g = beating_group
    onset = Plots._series(g)['onsets'][0] * 0.05
    assert onset < 0.3                                   # the first cycle begins near frame 0
    (lo, hi), i0, i1 = Plots._contr_window(g, 0, (-0.3, 0.3))
    assert lo < 0 and i0 == 0                            # never a negative slice start


@pytest.mark.parametrize('plot', [Plots.plot_z_pos, Plots.plot_overlay_delta_slen,
                                  Plots.plot_overlay_velocity, Plots.plot_phase_space])
def test_centred_views_render_with_default_limits(plot, beating_group):
    for number_contr in (None, 0, 1):
        ax = plt.figure().gca()
        plot(ax, beating_group, number_contr=number_contr)
        assert ax.lines, f'{plot.__name__} drew nothing for number_contr={number_contr}'
        plt.close('all')


def test_full_recording_is_the_default_time_axis(beating_group):
    ax = plt.figure().gca()
    Plots.plot_z_pos(ax, beating_group)
    x0, x1 = ax.get_xlim()
    assert x0 == 0 and x1 >= Plots._series(beating_group)['time'][-1]
    plt.close('all')


def test_popping_map_has_one_row_per_member_and_one_column_per_cycle(beating_group):
    fig = Plots.plot_popping_events(beating_group)
    ax = fig.axes[0]
    assert ax.get_ylim()[1] == beating_group.n_members
    assert ax.get_xlim()[1] == beating_group.n_contr
    plt.close('all')


def test_z_pos_needs_an_ordered_group():
    sarc = _fake_structure(n_tracks=6)
    g = sarc.motion.analyze('pool')[0]
    with pytest.raises(ValueError, match='ordered kinds'):
        Plots.plot_z_pos(plt.figure().gca(), g)
    plt.close('all')


# ---------------------------------------------------------------------------
# a myofibril group on real tracks (20 kPa reference store)
# ---------------------------------------------------------------------------

class TestGroupOnRealTracks:

    @pytest.fixture(scope='class')
    def sarc(self, motion_file_path_class):
        # in memory only: the 20 kPa store is the shared reference and keeps its results
        s = SarcAsM(motion_file_path_class, auto_save=False)
        if 'motion.tracks.slen' not in s.data:
            pytest.skip('20 kPa store has no tracks')
        if 0 not in list(s.data.get('params.structure.analyze_myofibrils.frames', [])):
            s.structure.analyze_myofibrils(frames=0)                 # the store holds no myofibril lines
        s.motion.analyze('myofibril', reference_frame=0)
        return s

    @pytest.fixture(scope='class')
    def group(self, sarc):
        return sarc.motion.groups('myofibril')[0]

    def test_chain_geometry(self, sarc, group):
        K, T = group.n_members, int(np.asarray(sarc.data['motion.tracks.slen']).shape[1])
        assert K >= 3
        z_pos, slen = group.z_pos, np.asarray(group.tracks.slen, float)
        assert slen.shape == (K, T) and z_pos.shape == (K + 1, T)
        assert np.nanmin(z_pos) == pytest.approx(0.0)                # z_pos[0] == 0
        # members are head to tail along the line
        row_of = {int(t): i for i, t in enumerate(sarc.data['motion.tracks.ids'])}
        order = np.asarray(sarc.data['motion.groups.myofibril.track_group_order'])
        assert np.all(np.diff(order[[row_of[int(t)] for t in group.tracks.ids]]) > 0)
        # a missing member blanks only its own row
        assert np.isnan(slen).sum(axis=0).max() < K
        # consecutive boundaries advance by ~one sarcomere where both are defined
        spacing = np.diff(z_pos, axis=0)
        ok = np.isfinite(spacing) & np.isfinite(slen)
        assert np.nanmedian(np.abs(spacing[ok] - slen[ok])) < 0.15

    def test_group_results(self, group):
        assert group.n_contr >= group.n_contr_complete >= 1
        assert group.contr.shape == group.slen_avg.shape
        assert group.track_popping.shape == (group.n_members, group.popping_rate_contr.shape[0])

    def test_summary_figure_renders(self, group, tmp_path):
        out = tmp_path / 'summary.png'
        Plots.plot_group_summary(group, file_path=str(out))
        assert out.exists() and out.stat().st_size > 0
        plt.close('all')

    def test_every_group_plot_accepts_the_group(self, group):
        for fn in (Plots.plot_z_pos, Plots.plot_delta_slen, Plots.plot_overlay_delta_slen,
                   Plots.plot_overlay_velocity, Plots.plot_phase_space, Plots.plot_slen):
            fn(plt.figure().gca(), group)
        Plots.plot_delta_slen(plt.figure().gca(), group, stacked=True)
        Plots.plot_popping_events(group)
        plt.close('all')


# ---------------------------------------------------------------------------
# per-track kinematics + raster (20 kPa reference store)
# ---------------------------------------------------------------------------

class TestTrackKinematics:

    @pytest.fixture(scope='class')
    def sarc(self, motion_file_path_class):
        # in memory only: the 20 kPa store is the shared reference and keeps its results
        s = SarcAsM(motion_file_path_class, auto_save=False)
        if 'motion.tracks.slen' not in s.data:
            pytest.skip('20 kPa store has no tracks')
        s.motion.analyze(by='pool')                  # pooled cycles for the equilibrium / raster
        if 0 not in list(s.data.get('params.structure.analyze_myofibrils.frames', [])):
            s.structure.analyze_myofibrils(frames=0)                 # the store holds no myofibril lines
        s.motion.group_tracks(by='myofibril', reference_frame=0, min_group_size=6)
        return s

    def test_kinematics_shapes_and_equilibrium(self, sarc):
        kin = sarc.motion.kinematics()
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
        im = Plots.plot_track_raster(ax, sarc, sort_by=sort_by, kind='myofibril' if sort_by == 'group' else None)
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
