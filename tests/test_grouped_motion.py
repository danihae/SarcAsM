# -*- coding: utf-8 -*-
"""Phase-1 tests: grouped-motion engine + group_tracks / analyze_track_motion."""
from __future__ import annotations

import types

import numpy as np
import pytest

from sarcasm.structure import Structure
from sarcasm.structure_modules import grouped_motion
from sarcasm.utils import Utils


# ---------------------------------------------------------------------------
# Pure engine: aggregate_group_slen
# ---------------------------------------------------------------------------

def test_aggregate_group_slen_basic():
    # 4 tracks, 3 frames; tracks {0,1}->group 0, {2,3}->group 1.
    tracks_slen = np.array([
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0],
    ], dtype=float)
    gid = np.array([0, 0, 1, 1])
    out = grouped_motion.aggregate_group_slen(tracks_slen, gid, n_groups=2, aggregate='nanmean')
    assert out['slen_timeseries'].shape == (2, 3)
    assert np.allclose(out['slen_timeseries'][0], 1.5)   # mean(2,1)
    assert np.allclose(out['slen_timeseries'][1], 2.0)   # mean(3,1)
    assert np.all(out['n_members_timeseries'] == 2)


def test_aggregate_group_slen_median_nan_and_slen_lims():
    tracks_slen = np.array([
        [2.0, np.nan, 2.0],
        [2.2, 2.2, 5.0],     # 5.0 is outside slen_lims -> dropped
        [1.8, 1.8, np.nan],
    ], dtype=float)
    gid = np.array([0, 0, 0])
    out = grouped_motion.aggregate_group_slen(
        tracks_slen, gid, n_groups=1, aggregate='nanmedian', slen_lims=(1.0, 3.0))
    # frame 0: median(2.0, 2.2, 1.8) = 2.0 ; members = 3
    assert out['slen_timeseries'][0, 0] == pytest.approx(2.0)
    assert out['n_members_timeseries'][0, 0] == 3
    # frame 2: only 2.0 finite & in-range (5.0 clipped to NaN, 1.8->nan) -> 1 member
    assert out['n_members_timeseries'][0, 2] == 1
    assert out['slen_timeseries'][0, 2] == pytest.approx(2.0)


def test_aggregate_group_slen_unassigned_excluded():
    tracks_slen = np.array([[2.0, 2.0], [9.0, 9.0]], dtype=float)
    gid = np.array([0, -1])   # second track unassigned
    out = grouped_motion.aggregate_group_slen(tracks_slen, gid, n_groups=1, aggregate='nanmean')
    assert np.allclose(out['slen_timeseries'][0], 2.0)
    assert np.all(out['n_members_timeseries'][0] == 1)


def test_run_cycle_engine_empty_groups():
    out = grouped_motion.run_cycle_engine(np.zeros((0, 20)), frametime=0.01, model_path='unused')
    assert out['domain_contr'].shape == (0, 20)
    assert out['domain_n_contr'].shape == (0,)


# ---------------------------------------------------------------------------
# Structure.group_tracks / analyze_track_motion on a synthetic data dict
# ---------------------------------------------------------------------------

def _fake_structure(n_tracks=6, T=80, frametime=0.01, seed=0):
    """A Structure with a synthetic tracker output, no file IO."""
    rng = np.random.default_rng(seed)
    sarc = Structure.__new__(Structure)
    sarc.auto_save = False
    sarc.metadata = types.SimpleNamespace(frametime=frametime, pixelsize=0.1, n_stack=T, size=(128, 128))
    sarc.model_dir = Utils.get_models_dir()

    t = np.arange(T)
    # beating-like signal: baseline 1.95 µm with periodic contractions to ~1.72
    base = 1.95 - 0.22 * np.clip(np.sin(2 * np.pi * t / 16.0), 0, None)
    slen = np.stack([base + 0.01 * rng.standard_normal(T) for _ in range(n_tracks)])
    snapped = np.ones((n_tracks, T), bool)
    # two M-bands: tracks 0,1,2 -> midline 5 ; tracks 3,4,5 -> midline 9
    mids = np.where(np.arange(n_tracks)[:, None] < n_tracks // 2, 5, 9).astype(np.int32)
    mids = np.broadcast_to(mids, (n_tracks, T)).copy()
    pos = np.zeros((n_tracks, T, 2), np.float32)
    pos[:, :, 0] = (10 + 10 * np.arange(n_tracks))[:, None]
    pos[:, :, 1] = 20.0

    sarc.data = {
        'n_tracks': n_tracks,
        'track_ids': np.arange(n_tracks),
        'track_start_frame': np.zeros(n_tracks, int),
        'track_lengths': snapped.sum(axis=1),
        'tracks_slen': slen.astype(np.float32),
        'tracks_positions_px': pos,
        'tracks_positions_um': pos * 0.1,
        'tracks_snapped': snapped,
        'tracks_midline_id': mids,
        # track i is snapped to detection (vector) index i at every frame
        'tracks_detection_id': np.broadcast_to(
            np.arange(n_tracks, dtype=np.int32)[:, None], (n_tracks, T)).copy(),
        'tracks_orientations': np.zeros((n_tracks, T), np.float32),
        'params.track_sarcomere_vectors.frames': list(range(T)),
    }
    return sarc


def test_group_tracks_pool():
    sarc = _fake_structure()
    sarc.group_tracks(by='pool')
    assert sarc.data['group_kind'] == 'pool'
    assert sarc.data['n_groups'] == 1
    assert np.all(sarc.data['track_group_id'] == 0)
    assert 'grouping_hash' in sarc.data


def test_group_tracks_mband():
    sarc = _fake_structure(n_tracks=6)
    sarc.group_tracks(by='mband', reference_frame=0)
    gid = sarc.data['track_group_id']
    assert sarc.data['n_groups'] == 2
    # tracks sharing an M-band id land in the same group
    assert gid[0] == gid[1] == gid[2]
    assert gid[3] == gid[4] == gid[5]
    assert gid[0] != gid[3]
    assert sarc.data['group_member_counts'].tolist() == [3, 3]


def test_group_tracks_custom_and_min_coverage():
    sarc = _fake_structure(n_tracks=4)
    # make track 3 low coverage so min_coverage drops it
    snp = sarc.data['tracks_snapped'].copy()
    snp[3, 10:] = False
    sarc.data['tracks_snapped'] = snp
    sarc.data['track_lengths'] = snp.sum(axis=1)
    labels = np.array([0, 0, 1, 1])
    sarc.group_tracks(by='custom', labels=labels, min_coverage=0.5)
    gid = sarc.data['track_group_id']
    assert gid[0] == gid[1]
    assert gid[3] == -1               # dropped by min_coverage
    assert sarc.data['n_groups'] == 2  # labels 0 and 1 both still present (track 2)


def test_group_tracks_custom_requires_labels():
    sarc = _fake_structure()
    with pytest.raises(ValueError):
        sarc.group_tracks(by='custom')


def test_group_tracks_unknown_and_missing_prereq_raise():
    sarc = _fake_structure()
    # unknown level
    with pytest.raises(ValueError):
        sarc.group_tracks(by='nonsense')
    # implemented levels raise ValueError (not NotImplementedError) on missing prereqs
    with pytest.raises(ValueError):
        sarc.group_tracks(by='domain')      # needs analyze_sarcomere_domains
    with pytest.raises(ValueError):
        sarc.group_tracks(by='myofibril')   # needs analyze_myofibrils


def test_analyze_track_motion_pool_end_to_end():
    sarc = _fake_structure()
    sarc.analyze_track_motion(by='pool')   # front door: groups + analyzes
    # legacy-mirrored keys exist under the 'pool' prefix
    for k in ['pool_slen_timeseries', 'pool_contr', 'pool_n_contr',
              'pool_beating_rate', 'pool_equ', 'pool_contr_max']:
        assert k in sarc.data, k
    assert sarc.data['pool_slen_timeseries'].shape == (1, 80)
    assert sarc.data['pool_contr'].shape == (1, 80)
    assert sarc.data['track_motion_kind'] == 'pool'
    assert sarc.data['params.analyze_track_motion.grouping_hash'] == sarc.data['grouping_hash']
    # freshness guard passes right after analysis
    sarc._assert_track_motion_fresh()


def test_analyze_track_motion_mband_shapes():
    sarc = _fake_structure(n_tracks=6)
    sarc.analyze_track_motion(by='mband', reference_frame=0)
    assert sarc.data['mband_slen_timeseries'].shape == (2, 80)
    assert sarc.data['mband_contr'].shape == (2, 80)


def test_stale_grouping_hard_raises():
    sarc = _fake_structure()
    sarc.analyze_track_motion(by='pool')
    sarc._assert_track_motion_fresh()           # fresh
    sarc.group_tracks(by='mband')               # re-group, do NOT re-analyze
    with pytest.raises(ValueError):
        sarc._assert_track_motion_fresh()       # stale -> hard raise


def test_analyze_refuses_stale_track_ids():
    sarc = _fake_structure()
    sarc.group_tracks(by='pool')
    # simulate re-tracking that changed the track set
    sarc.data['track_ids'] = np.arange(99)
    with pytest.raises(ValueError):
        sarc.analyze_track_motion()


def test_get_tracks_group_columns_after_grouping():
    sarc = _fake_structure(n_tracks=6)
    df0 = sarc.get_tracks()
    assert 'group_id' not in df0.columns        # before grouping
    assert 'ref_midline_id' in df0.columns
    sarc.group_tracks(by='mband', reference_frame=0)
    df1 = sarc.get_tracks()
    assert 'group_id' in df1.columns
    assert set(df1['group_id'].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# Phase 2: domain grouping (frozen reference-frame mask) + legacy-key emit
# ---------------------------------------------------------------------------

def _fake_structure_domain(n_per=3, T=80, frametime=0.01):
    """Fake structure with a stored 3-label domain mask; group 3 has no tracks."""
    n = 2 * n_per
    sarc = _fake_structure(n_tracks=n, T=T, frametime=frametime)
    H = W = 140
    sarc.metadata.size = (H, W)
    # group A around px-row 20 (mask label 1), group B around px-row 90 (label 2)
    rows = np.concatenate([np.full(n_per, 20.0), np.full(n_per, 90.0)])
    pos = np.zeros((n, T, 2), np.float32)
    pos[:, :, 0] = rows[:, None]
    pos[:, :, 1] = 30.0
    sarc.data['tracks_positions_px'] = pos
    sarc.data['tracks_positions_um'] = pos * sarc.metadata.pixelsize  # pixelsize 0.1
    mask = np.zeros((H, W), np.uint8)
    mask[0:50] = 1
    mask[60:110] = 2
    mask[115:140] = 3          # label 3 region has no tracks -> empty group preserved
    sarc.data['domain_mask'] = [mask]
    sarc.data['n_domains'] = np.array([3])
    sarc.data['domains'] = [None]
    sarc.data['params.analyze_sarcomere_domains.frames'] = [0]
    return sarc


def test_group_tracks_domain_label_alignment_and_empty_group():
    sarc = _fake_structure_domain(n_per=3)
    sarc.group_tracks(by='domain', reference_frame=0)
    gid = sarc.data['track_group_id']
    # mask label 1 region -> group 0 ; label 2 region -> group 1
    assert gid[:3].tolist() == [0, 0, 0]
    assert gid[3:].tolist() == [1, 1, 1]
    # label space preserved: n_groups == n_domains (3) even though group 2 is empty
    assert sarc.data['n_groups'] == 3
    assert sarc.data['group_member_counts'].tolist() == [3, 3, 0]


def test_analyze_track_motion_domain_emits_legacy_keys():
    sarc = _fake_structure_domain(n_per=3)
    sarc.analyze_track_motion(by='domain', reference_frame=0)
    # exact legacy domain_* schema (what plot_domain_timeseries / feature_dict / export read)
    legacy = [
        'domain_slen_timeseries', 'domain_slen_median_timeseries', 'domain_slen_std_timeseries',
        'domain_slen_q25_timeseries', 'domain_slen_q75_timeseries', 'domain_n_vectors_timeseries',
        'domain_contr', 'domain_n_contr', 'domain_labels_contr', 'domain_beating_rate',
        'domain_beating_rate_variability', 'domain_equ', 'domain_contr_max', 'domain_elong_max',
        'domain_vel_contr_max', 'domain_vel_elong_max', 'domain_time_to_peak', 'domain_time_to_relax',
        'domain_time_contr',
    ]
    for k in legacy:
        assert k in sarc.data, k
    assert sarc.data['domain_slen_timeseries'].shape == (3, 80)
    assert sarc.data['domain_contr'].shape == (3, 80)
    assert sarc.data['domain_n_contr'].shape == (3,)
    # the empty domain (group 2) yields no contractions
    assert sarc.data['domain_n_contr'][2] == 0
    assert sarc.data['track_motion_kind'] == 'domain'


def test_analyze_domain_motion_deprecation_delegates():
    sarc = _fake_structure_domain(n_per=3)
    with pytest.warns(DeprecationWarning):
        sarc.analyze_domain_motion(reference_frame=0)
    # delegated to the track-based path
    assert sarc.data['track_motion_kind'] == 'domain'
    assert 'domain_slen_timeseries' in sarc.data
    assert sarc.data['domain_slen_timeseries'].shape == (3, 80)


def test_domain_front_door_and_unknown_level():
    sarc = _fake_structure_domain(n_per=3)
    sarc.analyze_track_motion(by='domain', reference_frame=0)   # front door works
    assert sarc.data['group_kind'] == 'domain'
    with pytest.raises(ValueError):
        sarc.group_tracks(by='nonsense')


# ---------------------------------------------------------------------------
# Phase 3: myofibril grouping (order) + LOI chain synthesis + Motion view
# ---------------------------------------------------------------------------

def test_synthesize_loi_chain_diff_and_nan():
    member = np.array([
        [1.8, 1.8, np.nan, 1.8, 1.8],
        [2.0, np.nan, 2.0, 2.0, 2.0],
        [1.9, 1.9, 1.9, np.nan, 1.9],
    ])
    z, slen, time = grouped_motion.synthesize_loi_chain(member, frametime=0.01)
    assert z.shape == (4, 5)
    assert slen.shape == (3, 5)
    # by construction diff(z_pos) == slen, and NaNs are interpolated away
    assert np.allclose(np.diff(z, axis=0), slen)
    assert not np.isnan(slen).any()
    assert np.allclose(z[0], 0.0)
    assert np.allclose(time, np.arange(5) * 0.01)


def test_group_tracks_myofibril_ordering():
    sarc = _fake_structure(n_tracks=6)
    # two fibres; fibre A lists vectors in reversed order [2,1,0]
    sarc.data['myof_lines'] = [[np.array([2, 1, 0]), np.array([3, 4, 5])]]
    sarc.group_tracks(by='myofibril', reference_frame=0)
    gid = sarc.data['track_group_id']
    order = sarc.data['track_group_order']
    assert sarc.data['n_groups'] == 2
    # fibre A: tracks {0,1,2} share a group; ordering follows the vector order [2,1,0]
    assert gid[0] == gid[1] == gid[2]
    assert gid[3] == gid[4] == gid[5]
    assert gid[0] != gid[3]
    assert order[2] == 0 and order[1] == 1 and order[0] == 2


def test_group_tracks_myofibril_requires_lines():
    sarc = _fake_structure(n_tracks=6)
    with pytest.raises(ValueError):
        sarc.group_tracks(by='myofibril', reference_frame=0)


def test_analyze_track_motion_myofibril_coarse():
    sarc = _fake_structure(n_tracks=6)
    sarc.data['myof_lines'] = [[np.array([0, 1, 2]), np.array([3, 4, 5])]]
    sarc.analyze_track_motion(by='myofibril', reference_frame=0)
    assert sarc.data['myofibril_slen_timeseries'].shape == (2, 80)
    assert 'myofibril_contr' in sarc.data
    assert sarc.data['track_motion_kind'] == 'myofibril'


def test_plot_track_myofibrils_smoke():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sarcasm.plots import Plots
    # stub the image background (needs file IO)
    orig_z, orig_img = Plots.plot_z_bands, Plots.plot_image
    Plots.plot_z_bands = staticmethod(lambda ax, *a, **k: None)
    Plots.plot_image = staticmethod(lambda ax, *a, **k: None)
    try:
        sarc = _fake_structure(n_tracks=6)
        sarc.data['myof_lines'] = [[np.array([0, 1, 2]), np.array([3, 4, 5])]]
        sarc.group_tracks(by='myofibril', reference_frame=0)
        for cb in ('group', 'slen'):
            Plots.plot_track_myofibrils(plt.figure().gca(), sarc, frame=0, color_by=cb, scalebar=False)
        # requires a myofibril grouping
        sarc.group_tracks(by='pool')
        with pytest.raises(ValueError):
            Plots.plot_track_myofibrils(plt.figure().gca(), sarc)
    finally:
        Plots.plot_z_bands, Plots.plot_image = orig_z, orig_img
        plt.close('all')


def test_motion_from_loi_data_runs_loi_engine():
    """Motion.from_loi_data + the LOI engine on a synthesized fibre (needs a cell tif)."""
    from pathlib import Path
    from sarcasm.motion import Motion
    fp = Path(__file__).parent.parent / 'test_data/high_speed_single_ACTN2-citrine_CM/20kPa.tif'
    if not fp.exists():
        pytest.skip('test data not found')
    T = 80
    t = np.arange(T)
    base = 1.95 - 0.2 * np.clip(np.sin(2 * np.pi * t / 16.0), 0, None)
    member = np.stack([base + 0.01 * np.sin(t / 3.0 + k) for k in range(4)])
    z, slen, time = grouped_motion.synthesize_loi_chain(member, frametime=0.01)
    m = Motion.from_loi_data(
        str(fp), 'track_myofibril_test',
        {'z_pos': z, 'z_pos_raw': z.copy(), 'slen': slen, 'time': time, 'n_sarcomeres': 4},
        auto_save=False, frametime=0.01)
    assert m.loi_data['synthetic'] is True
    assert m.metadata.frametime is not None   # from override or stored/embedded metadata
    # the battle-tested LOI engine runs unchanged on the synthesized chain
    m.detect_analyze_contractions()
    m.get_trajectories()
    assert m.loi_data['slen'].shape[0] == 4
    assert 'delta_slen' in m.loi_data


# ---------------------------------------------------------------------------
# group_tracks(by='loi') — curated fibre lines from detect_lois
# ---------------------------------------------------------------------------

def test_group_tracks_loi_assigns_and_orders():
    sarc = _fake_structure(n_tracks=6)
    T = sarc.data['tracks_slen'].shape[1]
    # two well-separated triplets of tracks (rows ~10-12 and ~50-52, col 20)
    pos = np.zeros((6, T, 2), np.float32)
    rows = np.array([12, 10, 11, 52, 50, 51], float)   # shuffled so order!=index
    pos[:, :, 0] = rows[:, None]; pos[:, :, 1] = 20.0
    sarc.data['tracks_positions_px'] = pos
    sarc.data['tracks_positions_um'] = pos * 0.1
    # curated LOI polylines (px), each passing through one triplet
    line0 = np.array([[10, 20], [11, 20], [12, 20]], float)
    line1 = np.array([[50, 20], [51, 20], [52, 20]], float)
    sarc.data['loi_data'] = {'loi_lines': [line0, line1]}

    sarc.group_tracks(by='loi', reference_frame=0)
    gid = sarc.data['track_group_id']
    assert sarc.data['group_kind'] == 'loi'
    assert sarc.data['n_groups'] == 2
    assert gid[[1, 2, 0]].tolist() == [0, 0, 0]    # rows 10,11,12 -> line 0
    assert gid[[4, 5, 3]].tolist() == [1, 1, 1]    # rows 50,51,52 -> line 1
    # order is rank along the line (by arc length / row), independent of track index
    order = sarc.data['track_group_order']
    assert order[1] == 0 and order[2] == 1 and order[0] == 2
    assert order[4] == 0 and order[5] == 1 and order[3] == 2


def test_group_tracks_loi_far_tracks_unassigned():
    sarc = _fake_structure(n_tracks=3)
    T = sarc.data['tracks_slen'].shape[1]
    pos = np.zeros((3, T, 2), np.float32)
    pos[:, :, 0] = np.array([10, 11, 90], float)[:, None]  # track 2 far away
    pos[:, :, 1] = 20.0
    sarc.data['tracks_positions_px'] = pos
    sarc.data['loi_data'] = {'loi_lines': [np.array([[10, 20], [11, 20]], float)]}
    sarc.group_tracks(by='loi')
    gid = sarc.data['track_group_id']
    assert gid[0] == 0 and gid[1] == 0
    assert gid[2] == -1            # > 0.5*slen from the line -> dropped


def test_group_tracks_loi_missing_prereq_raises():
    sarc = _fake_structure()
    with pytest.raises(ValueError, match='LOI lines not available'):
        sarc.group_tracks(by='loi')
