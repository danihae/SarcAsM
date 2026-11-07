# -*- coding: utf-8 -*-
"""``sarc.motion.groups(kind)``: one Group class for every grouping kind (SARC-28).

A Group's attributes are views of the stored results under the same names: its row of
``motion.<kind>.*``, its members' slice of ``motion.tracks.*``, and their kinematics against the
group's own contractions. The per-group view of 1.0.0b4 (``Motion``) decided contractions a
second time and could disagree with the group (SARC-21); a Group cannot.
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm import SarcAsM
from sarcasm.analysis import heterogeneity
from .test_grouped_motion import _fake_structure


def test_group_attributes_are_rows_of_the_stored_results():
    sarc = _fake_structure(n_tracks=6)
    groups = sarc.motion.analyze('mband', reference_frame=0)
    assert len(groups) == 2 and repr(groups) == '<GroupSet mband: 2 groups>'
    for g in groups:
        assert g.kind == 'mband' and not g.ordered and g.n_members == 3
        np.testing.assert_array_equal(g.contr, sarc.data['motion.mband.contr'][g.id])
        np.testing.assert_array_equal(g.slen_avg, sarc.data['motion.mband.slen_avg'][g.id])
        np.testing.assert_equal(g.beating_rate, sarc.data['motion.mband.beating_rate'][g.id])
        np.testing.assert_array_equal(g.oscill_frequencies, sarc.data['motion.mband.oscill_frequencies'])
        assert g.track_popping.shape[0] == g.n_members
    np.testing.assert_array_equal(groups[-1].tracks.ids, groups[1].tracks.ids)
    with pytest.raises(IndexError):
        groups[2]
    with pytest.raises(AttributeError, match='not a result'):
        groups[0].fibre_thing


def test_member_tracks_are_slices_of_the_track_results():
    sarc = _fake_structure(n_tracks=6)
    g = sarc.motion.analyze('mband', reference_frame=0)[1]
    np.testing.assert_array_equal(g.tracks.ids, [3, 4, 5])
    np.testing.assert_array_equal(g.tracks.slen, sarc.data['motion.tracks.slen'][3:6])
    kin = heterogeneity.member_kinematics(sarc.data['motion.tracks.slen'][3:6].astype(float),
                                          sarc.data['motion.mband.contr'][1].astype(bool), 0.01,
                                          filter_params=(13, 5), slen_lims=(1.0, 3.0))
    np.testing.assert_allclose(g.tracks.delta_slen, kin['delta_slen'])
    np.testing.assert_allclose(g.tracks.equ, kin['equ'])


def test_ordered_groups_carry_the_line():
    sarc = _fake_structure(n_tracks=6)
    sarc.data['structure.myofibril.lines'] = [[np.array([2, 0, 1]), np.array([3, 4, 5])]]
    groups = sarc.motion.group_tracks(by='myofibril', reference_frame=0)
    g = groups[[g.n_members for g in groups].index(3)]
    assert g.ordered and g.z_pos.shape == (4, 80)
    with pytest.raises(ValueError, match='ordered kinds'):
        sarc.motion.group_tracks(by='pool')[0].z_pos


def test_groups_need_a_stored_grouping_and_current_tracks():
    sarc = _fake_structure(n_tracks=6)
    with pytest.raises(ValueError, match="No 'mband' grouping"):
        sarc.motion.groups('mband')
    sarc.motion.group_tracks(by='mband', reference_frame=0)
    g = sarc.motion.groups('mband')[0]
    with pytest.raises(AttributeError, match="analyze\\('mband'\\)"):
        g.contr                                       # grouped, not analysed
    sarc.data['motion.tracks.ids'] = np.arange(10, 16)
    with pytest.raises(ValueError, match='Tracks changed'):
        sarc.motion.groups('mband')


def test_popping_events_per_member_and_cycle():
    T = 40
    labels = np.zeros(T, int)
    labels[5:15], labels[25:35] = 1, 2
    delta = np.zeros((3, T))
    delta[0, 8] = 0.3                                  # pops in cycle 1
    delta[1, 30] = 0.3                                 # pops in cycle 2
    delta[2, 20] = 0.5                                 # lengthens between cycles: not popping
    delta[2, 26:30] = np.nan
    events = heterogeneity.popping_events(delta, labels, np.array([True, False]), thres_popping=0.25)
    np.testing.assert_array_equal(events[:, 0], [1, 0, 0])
    assert np.isnan(events[:, 1]).all()                # an incomplete cycle is not scored


def _pulse(L):
    return -0.3 * np.sin(np.pi * np.linspace(0, 1, L)) ** 2


def test_window_states_per_member_and_cycle():
    labels = np.zeros(60, int)
    labels[5:25], labels[35:55] = 1, 2
    delta = np.zeros((5, 60))
    delta[:, 5:25] = delta[:, 35:55] = _pulse(20)
    delta[1, 20:25] = -0.15                            # cycle 1: still short at its end
    delta[2, 35] = 0.15                                # cycle 2: starts lengthened
    delta[3, 6:9] = 0.2                                # cycle 1: lengthens first, early
    delta[4, 5:25] = 0.05 * _pulse(20) / 0.3           # cycle 1: small shortening, then a late pop:
    delta[4, 21] = 0.3                                 # its first move is a lengthening, but late
    s = heterogeneity.window_states(delta, labels, np.array([True, True]))
    assert s == pytest.approx({'still_short_rate': 0.1, 'starts_long_rate': 0.1, 'lengthens_first_rate': 0.1})


def test_window_states_skip_cycles_holding_more_than_one_beat():
    labels = np.zeros(140, int)
    labels[5:25], labels[35:55], labels[65:85], labels[90:130] = 1, 2, 3, 4
    delta = np.zeros((3, 140))
    delta[:, 5:25] = delta[:, 35:55] = _pulse(20)
    delta[:, 65:85] = np.r_[_pulse(10), _pulse(10)]   # two beats merged into one cycle
    delta[:, 90:130] = _pulse(40)                      # 2 x the median cycle
    delta[0, 82:85] = delta[0, 127:130] = -0.15        # still short only in those two
    s = heterogeneity.window_states(delta, labels, np.ones(4, bool))
    assert s['still_short_rate'] == 0.0


class TestSarc21OnRealTracks:
    """SARC-21: the per-group view reports the group's own contractions, small groups included."""

    @pytest.fixture(scope='class')
    def sarc(self, motion_file_path_class):
        s = SarcAsM(motion_file_path_class, auto_save=False)
        if 'motion.tracks.slen' not in s.data:
            pytest.skip('20 kPa store has no tracks')
        if 0 not in list(s.data.get('params.structure.analyze_myofibrils.frames', [])):
            s.structure.analyze_myofibrils(frames=0)
        s.motion.analyze('myofibril', reference_frame=0)
        return s

    def test_every_group_shows_the_contractions_of_its_row(self, sarc):
        groups = sarc.motion.groups('myofibril')
        stored = np.asarray(sarc.data['motion.myofibril.contr'])
        small = 0
        for g in groups:
            np.testing.assert_array_equal(g.contr, stored[g.id])
            small += 0 < g.n_members < 4
        assert small > 0, 'the case SARC-21 was about: groups with fewer than 4 sarcomeres'
        beating = [g for g in groups if 0 < g.n_members < 4 and g.contr.any()]
        assert beating, 'small groups with contractions are reported as contracting'
