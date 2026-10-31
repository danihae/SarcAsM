# -*- coding: utf-8 -*-
"""Each grouping kind is stored once, beside the others, and every reader names its kind.

Until 1.0.0b4 ``group_tracks`` wrote the grouping "currently in effect" to single-valued
keys (``motion.groups.n``, ``motion.tracks.group_id``, ...) plus a per-kind copy, so what a
reader got depended on which call ran last (SARC-27). Now there is only
``motion.groups.<kind>.*``, and these tests pin that model.
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.export import Export
from .test_grouped_motion import _fake_structure


def _analyze_two_kinds(sarc):
    sarc.motion.analyze('pool')
    sarc.motion.analyze('mband', reference_frame=0)


def _exportable(sarc):
    """The synthetic fixture's metadata is a SimpleNamespace; Export needs to_dict."""
    sarc.file_path = 'synthetic.tif'
    sarc.metadata.to_dict = lambda: {'file_name': 'synthetic.tif'}
    return sarc


# ---------------------------------------------------------------------------
# storage: one grouping per kind, side by side
# ---------------------------------------------------------------------------

def test_each_kind_is_stored_beside_the_others():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.group_tracks(by='pool')
    sarc.motion.group_tracks(by='mband', reference_frame=0)

    assert sarc.data['motion.groups.pool.n'] == 1
    assert sarc.data['motion.groups.mband.n'] == 2
    assert np.all(sarc.data['motion.groups.pool.track_group_id'] == 0)
    assert sarc.data['motion.groups.pool.member_counts'].tolist() == [6]
    assert sarc.data['motion.groups.mband.member_counts'].tolist() == [3, 3]
    assert sarc.data['motion.groups.pool.hash'] != sarc.data['motion.groups.mband.hash']
    assert sarc.motion._grouped_kinds() == ['pool', 'mband']
    # no "current grouping" keys
    for key in ('motion.groups.kind', 'motion.groups.n', 'motion.tracks.group_id', 'motion.groups.analyzed_kind'):
        assert key not in sarc.data


def test_grouping_params_are_stored_per_kind():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.group_tracks(by='pool', min_coverage=0.3)
    sarc.motion.group_tracks(by='mband', reference_frame=0)
    assert sarc.data['params.motion.group_tracks.pool.min_coverage'] == 0.3
    assert sarc.data['params.motion.group_tracks.mband.min_coverage'] == 0.5


def test_regrouping_a_kind_drops_only_its_analysis():
    sarc = _fake_structure(n_tracks=6)
    _analyze_two_kinds(sarc)
    sarc.motion.group_tracks(by='mband', reference_frame=0, min_coverage=0.3)
    assert 'motion.mband.contr' not in sarc.data
    assert 'params.motion.analyze.mband.grouping_hash' not in sarc.data
    assert 'motion.pool.contr' in sarc.data
    sarc.motion._assert_track_motion_fresh('pool')


# ---------------------------------------------------------------------------
# analyze(by): the stored grouping, unless grouping arguments are given
# ---------------------------------------------------------------------------

def test_analyze_uses_the_stored_grouping():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.group_tracks(by='mband', reference_frame=0, min_coverage=0.3)
    sarc.motion.analyze('mband')
    assert sarc.data['params.motion.group_tracks.mband.min_coverage'] == 0.3


def test_analyze_with_grouping_arguments_regroups():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.group_tracks(by='mband', reference_frame=0, min_coverage=0.3)
    sarc.motion.analyze('mband', min_coverage=0.6)
    assert sarc.data['params.motion.group_tracks.mband.min_coverage'] == 0.6


def test_analyze_groups_a_kind_not_grouped_yet():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.analyze('pool')
    assert sarc.data['motion.groups.pool.n'] == 1


def test_analyze_custom_needs_its_own_grouping():
    sarc = _fake_structure(n_tracks=6)
    with pytest.raises(ValueError, match="group_tracks\\(by='custom'"):
        sarc.motion.analyze('custom')
    sarc.motion.group_tracks(by='custom', labels=np.array([0, 0, 1, 1, 2, 2]))
    with pytest.raises(ValueError, match='no grouping arguments'):
        sarc.motion.analyze('custom', min_coverage=0.2)
    sarc.motion.analyze('custom')
    assert sarc.data['motion.groups.custom.n'] == 3


def test_analyze_needs_a_kind():
    sarc = _fake_structure(n_tracks=6)
    with pytest.raises(TypeError):
        sarc.motion.analyze()
    with pytest.raises(ValueError, match='Unknown grouping'):
        sarc.motion.analyze('fibre')


def test_analysis_params_are_stored_per_kind():
    sarc = _fake_structure(n_tracks=6)
    _analyze_two_kinds(sarc)
    assert sarc.motion._analyzed_kinds() == ['pool', 'mband']
    assert sarc.data['params.motion.analyze.pool.n_groups'] == 1
    assert sarc.data['params.motion.analyze.mband.n_groups'] == 2
    assert sarc.data['params.motion.analyze.mband.grouping_hash'] == sarc.data['motion.groups.mband.hash']


def test_group_average_is_slen_avg():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.analyze('mband', reference_frame=0)
    assert sarc.data['motion.mband.slen_avg'].shape == (2, sarc.data['motion.tracks.slen'].shape[1])
    assert 'motion.mband.slen' not in sarc.data


# ---------------------------------------------------------------------------
# readers name their kind
# ---------------------------------------------------------------------------

def test_freshness_is_checked_per_kind():
    sarc = _fake_structure(n_tracks=6)
    _analyze_two_kinds(sarc)
    sarc.motion._assert_track_motion_fresh('pool')
    sarc.motion._assert_track_motion_fresh('mband')
    with pytest.raises(ValueError, match='myofibril'):
        sarc.motion._assert_track_motion_fresh('myofibril')


def test_export_reads_each_kind_with_its_own_group_count():
    sarc = _fake_structure(n_tracks=6)
    _analyze_two_kinds(sarc)
    _exportable(sarc)
    pool = Export.get_motion_dict_per_group(sarc, kind='pool')
    mband = Export.get_motion_dict_per_group(sarc, kind='mband')
    assert len(pool) == 1 and len(mband) == 2
    assert [r['kind'] for r in pool] == ['pool']
    assert pool[0]['group_member_count'] == 6
    assert [r['group_member_count'] for r in mband] == [3, 3]


def test_export_needs_an_analysed_kind():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.analyze('mband', reference_frame=0)
    _exportable(sarc)
    with pytest.raises(ValueError, match='Name the grouping'):
        Export.get_motion_dict_per_group(sarc)
    with pytest.raises(ValueError, match='pool'):
        Export.get_motion_dict_per_group(sarc, kind='pool')


def test_tracks_have_one_group_column_per_kind():
    sarc = _fake_structure(n_tracks=6)
    sarc.motion.group_tracks(by='mband', reference_frame=0)
    sarc.motion.group_tracks(by='pool')
    tracks = sarc.motion.tracks()
    assert set(tracks['group_mband']) == {0, 1}
    assert set(tracks['group_pool']) == {0}
    assert 'order_mband' not in tracks          # only the ordered kinds (myofibril, loi) have an order


# ---------------------------------------------------------------------------
# the store, and re-tracking
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_movie(tmp_path):
    """A tiny real .tif, so a real .ome.zarr store gets built (no test data needed)."""
    import tifffile
    T, size, frametime, pixelsize = 80, 48, 0.01, 0.1
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (T, size, size), dtype=np.uint8)
    path = tmp_path / 'synthetic.tif'
    tifffile.imwrite(path, frames, imagej=True,
                     resolution=(1 / pixelsize, 1 / pixelsize),
                     metadata={'axes': 'TYX', 'unit': 'um', 'finterval': frametime, 'fps': 1 / frametime})
    return str(path), T


def test_per_kind_keys_survive_the_store_round_trip(synthetic_movie):
    from sarcasm.core import SarcAsM as RealSarcAsM

    path, T = synthetic_movie
    sarc = RealSarcAsM(path)
    template = _fake_structure(n_tracks=6, T=T)
    sarc.data.update(template.data)
    _analyze_two_kinds(sarc)
    sarc.commit()

    reopened = RealSarcAsM(path)
    assert reopened.motion._analyzed_kinds() == ['pool', 'mband']
    assert int(reopened.data['motion.groups.pool.n']) == 1
    assert np.asarray(reopened.data['motion.groups.mband.member_counts']).tolist() == [3, 3]
    reopened.motion._assert_track_motion_fresh('pool')
    assert set(reopened.motion.tracks()['group_mband']) == {0, 1}
    assert len(Export.get_motion_dict_per_group(reopened, kind='mband')) == 2


def test_retracking_clears_every_grouping_and_its_analysis():
    sarc = _fake_structure(n_tracks=6)
    _analyze_two_kinds(sarc)
    sarc.motion.set_lois([np.array([[0, 20], [40, 20]], float)])

    sarc.motion._invalidate_groupings()

    leftovers = [k for k in sarc.data if k.startswith(
        ('motion.groups.', 'motion.pool.', 'motion.mband.', 'params.motion.group_tracks.', 'params.motion.analyze.'))]
    assert leftovers == []
    assert 'motion.lois.lines' in sarc.data        # LOI lines are input, not a grouping


# ---------------------------------------------------------------------------
# full_analysis: the motion pipeline after detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('by, prerequisite', [('pool', None), ('mband', None), ('myofibril', 'analyze_myofibrils'),
                                              ('domain', 'analyze_domains'), ('loi', 'detect_lois')])
def test_full_analysis_runs_the_steps_the_grouping_needs(monkeypatch, by, prerequisite):
    sarc = _fake_structure(n_tracks=6)
    calls = []
    structure, motion = sarc.structure, sarc.motion
    monkeypatch.setattr(sarc, 'commit', lambda: calls.append('commit'), raising=False)
    monkeypatch.setattr(structure, 'analyze_vectors', lambda frames: calls.append('analyze_vectors'))
    monkeypatch.setattr(motion, 'track', lambda frames, motion_predictor: calls.append('track'))

    def myofibrils(frames):
        calls.append('analyze_myofibrils')
        sarc.data['structure.myofibril.lines'] = [[np.array([0, 1, 2]), np.array([3, 4, 5])]]

    def domains(frames):
        calls.append('analyze_domains')
        sarc.data['structure.domain.mask'] = [np.ones((128, 128), np.uint8)]
        sarc.data['structure.domain.n'] = np.array([1])
        sarc.data['structure.domain.members'] = [None]
        sarc.data['params.structure.analyze_domains.frames'] = [0]

    def lois(frame):
        calls.append('detect_lois')
        motion.set_lois([np.array([[0.0, 20.0], [60.0, 20.0]])])

    monkeypatch.setattr(structure, 'analyze_myofibrils', myofibrils)
    monkeypatch.setattr(structure, 'analyze_domains', domains)
    monkeypatch.setattr(motion, 'detect_lois', lois)
    monkeypatch.setattr(motion, 'commit', lambda: calls.append('commit'))

    motion.full_analysis(by=by, frames='all', reference_frame=0)
    expected = ['analyze_vectors', 'track'] + ([prerequisite] if prerequisite else []) + ['commit']
    assert calls == expected
    assert motion._analyzed_kinds() == [by]
