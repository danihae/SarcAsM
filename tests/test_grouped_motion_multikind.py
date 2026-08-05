# -*- coding: utf-8 -*-
"""Several groupings off one tracking must stay independently readable.

``group_tracks`` / ``analyze_track_motion`` write single-valued keys
(``n_groups``, ``group_member_counts``, ``track_group_id``, ...) describing the
grouping *currently* in effect, so a second grouping overwrote the first. The
per-group export then iterated the wrong group count and read ``<kind>_<suffix>``
out of range — silently returning wrong-length data rather than failing. These
tests pin the per-kind mirrors that fix it.
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.export import Export
from .test_grouped_motion import _fake_structure


def _analyze_three_kinds(sarc):
    """pool (1 group) -> mband (2 groups), the ordering that used to clobber."""
    sarc.analyze_track_motion(by='pool')
    sarc.analyze_track_motion(by='mband', reference_frame=0)


def _exportable(sarc):
    """The synthetic fixture's metadata is a SimpleNamespace; Export needs to_dict."""
    sarc.file_path = 'synthetic.tif'
    sarc.metadata.to_dict = lambda: {'file_name': 'synthetic.tif'}
    return sarc


# ---------------------------------------------------------------------------
# group_tracks writes per-kind mirrors
# ---------------------------------------------------------------------------

def test_group_tracks_mirrors_survive_a_later_grouping():
    sarc = _fake_structure(n_tracks=6)
    sarc.group_tracks(by='pool')
    sarc.group_tracks(by='mband', reference_frame=0)

    # the unsuffixed keys track the current grouping ...
    assert sarc.data['group_kind'] == 'mband'
    assert sarc.data['n_groups'] == 2
    # ... while both per-kind mirrors persist
    assert sarc.data['n_groups_pool'] == 1
    assert sarc.data['n_groups_mband'] == 2
    assert np.all(sarc.data['track_group_id_pool'] == 0)
    assert sarc.data['group_member_counts_pool'].tolist() == [6]
    assert sarc.data['group_member_counts_mband'].tolist() == [3, 3]
    assert sarc.data['grouping_hash_pool'] != sarc.data['grouping_hash_mband']


def test_mirrors_carry_chain_order():
    sarc = _fake_structure(n_tracks=6)
    sarc.group_tracks(by='mband', reference_frame=0)
    assert 'track_group_order_mband' in sarc.data
    np.testing.assert_array_equal(sarc.data['track_group_order_mband'],
                                  sarc.data['track_group_order'])


# ---------------------------------------------------------------------------
# analyze_track_motion provenance
# ---------------------------------------------------------------------------

def test_track_motion_kinds_accumulates_in_call_order():
    sarc = _fake_structure(n_tracks=6)
    _analyze_three_kinds(sarc)
    assert sarc.data['track_motion_kinds'] == ['pool', 'mband']
    assert sarc.data['track_motion_kind'] == 'mband'          # last one still wins
    assert sarc.data['params.analyze_track_motion.n_groups_pool'] == 1
    assert sarc.data['params.analyze_track_motion.n_groups_mband'] == 2


def test_re_analyzing_a_kind_does_not_duplicate_it():
    sarc = _fake_structure(n_tracks=6)
    sarc.analyze_track_motion(by='pool')
    sarc.analyze_track_motion(by='pool')
    assert sarc.data['track_motion_kinds'] == ['pool']


def test_freshness_guard_accepts_an_earlier_kind():
    sarc = _fake_structure(n_tracks=6)
    _analyze_three_kinds(sarc)
    # the earlier kind is still valid against the current tracks ...
    sarc._assert_track_motion_fresh('pool')
    sarc._assert_track_motion_fresh('mband')
    # ... and the unsuffixed check still describes the current grouping
    sarc._assert_track_motion_fresh()


def test_freshness_guard_rejects_a_kind_never_analyzed():
    sarc = _fake_structure(n_tracks=6)
    sarc.analyze_track_motion(by='pool')
    with pytest.raises(ValueError, match="myofibril"):
        sarc._assert_track_motion_fresh('myofibril')


# ---------------------------------------------------------------------------
# The export bug this whole patch exists for
# ---------------------------------------------------------------------------

def test_export_reads_each_kind_with_its_own_group_count():
    sarc = _fake_structure(n_tracks=6)
    _analyze_three_kinds(sarc)
    _exportable(sarc)

    pool = Export.get_motion_dict_per_group(sarc, kind='pool')
    mband = Export.get_motion_dict_per_group(sarc, kind='mband')

    # Before the fix both used the last grouping's n_groups (2), so 'pool'
    # returned two records, the second one reading past the end of pool_*.
    assert len(pool) == 1
    assert len(mband) == 2
    assert [r['kind'] for r in pool] == ['pool']
    assert pool[0]['group_member_count'] == 6
    assert [r['group_member_count'] for r in mband] == [3, 3]


def test_export_defaults_to_the_last_analyzed_kind():
    sarc = _fake_structure(n_tracks=6)
    _analyze_three_kinds(sarc)
    _exportable(sarc)
    records = Export.get_motion_dict_per_group(sarc)
    assert [r['kind'] for r in records] == ['mband', 'mband']


def test_export_raises_for_a_kind_never_analyzed():
    """Asking for an unanalysed kind is an error, not an empty result.

    The whole bug was a silent wrong-length read, so the export refuses to guess
    from the unsuffixed keys.
    """
    sarc = _fake_structure(n_tracks=6)
    sarc.analyze_track_motion(by='mband', reference_frame=0)
    _exportable(sarc)
    with pytest.raises(ValueError, match='pool'):
        Export.get_motion_dict_per_group(sarc, kind='pool')


# ---------------------------------------------------------------------------
# get_tracks must be able to name the grouping it labels by
# ---------------------------------------------------------------------------

def test_get_tracks_labels_the_requested_kind():
    sarc = _fake_structure(n_tracks=6)
    sarc.group_tracks(by='mband', reference_frame=0)
    sarc.group_tracks(by='pool')

    # default keeps the historical meaning: whatever grouping is in effect
    assert set(sarc.get_tracks()['group_id']) == {0}
    # naming the kind reaches back to the earlier grouping
    assert set(sarc.get_tracks(kind='mband')['group_id']) == {0, 1}
    assert set(sarc.get_tracks(kind='pool')['group_id']) == {0}


def test_get_tracks_raises_for_a_kind_never_grouped():
    sarc = _fake_structure(n_tracks=6)
    sarc.group_tracks(by='pool')
    with pytest.raises(ValueError, match='myofibril'):
        sarc.get_tracks(kind='myofibril')


# ---------------------------------------------------------------------------
# The mirrors have to survive the store, not just live in self.data
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
                     metadata={'unit': 'um', 'finterval': frametime, 'fps': 1 / frametime})
    return str(path), T


def test_per_kind_keys_survive_the_store_round_trip(synthetic_movie):
    """Reopening the store must give back every grouping, not just the last.

    The mirrors are only useful if results_store routes and persists them; a
    synthetic self.data dict cannot show that.
    """
    from sarcasm.structure import SarcAsM as RealSarcAsM

    path, T = synthetic_movie
    sarc = RealSarcAsM(path)
    n_tracks = 6
    template = _fake_structure(n_tracks=n_tracks, T=T)
    sarc.data.update(template.data)
    sarc.analyze_track_motion(by='pool')
    sarc.analyze_track_motion(by='mband', reference_frame=0)
    sarc.store_structure_data()

    reopened = RealSarcAsM(path)
    data = reopened.data
    assert data['track_motion_kinds'] == ['pool', 'mband']
    assert int(data['n_groups_pool']) == 1
    assert int(data['n_groups_mband']) == 2
    assert np.asarray(data['group_member_counts_mband']).tolist() == [3, 3]
    assert np.all(np.asarray(data['track_group_id_pool']) == 0)
    assert data['grouping_hash_pool'] != data['grouping_hash_mband']

    # and the readers work off the reopened store
    reopened._assert_track_motion_fresh('pool')
    assert set(reopened.get_tracks(kind='mband')['group_id']) == {0, 1}
    assert len(Export.get_motion_dict_per_group(reopened, kind='pool')) == 1
    assert len(Export.get_motion_dict_per_group(reopened, kind='mband')) == 2


# ---------------------------------------------------------------------------
# Re-tracking must not leave a mirror behind
# ---------------------------------------------------------------------------

def test_retracking_clears_every_per_kind_key():
    sarc = _fake_structure(n_tracks=6)
    _analyze_three_kinds(sarc)
    assert 'n_groups_pool' in sarc.data

    sarc._invalidate_groupings()

    leftovers = [k for k in sarc.data
                 if k.startswith(('track_group_id', 'track_group_order', 'n_groups',
                                  'group_member_counts', 'grouping_hash',
                                  'track_ids_snapshot', 'group_n_vectors'))
                 or k == 'track_motion_kinds'
                 or k.startswith('params.analyze_track_motion.grouping_hash')]
    assert leftovers == []
    assert 'group_kind' not in sarc.data
