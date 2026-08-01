# -*- coding: utf-8 -*-
"""Phase-1 tests: grouped-motion engine + group_tracks / analyze_track_motion."""
from __future__ import annotations

import types

import numpy as np
import pytest

from sarcasm.structure import SarcAsM
from sarcasm.analysis import contraction_analysis, grouped_motion
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
    assert out['domain_n_contr_complete'].shape == (0,)
    assert out['domain_contr_complete'].shape == (0, 1)


# ---------------------------------------------------------------------------
# Incomplete (edge-truncated) contraction cycles
# ---------------------------------------------------------------------------

_FT = 0.01          # frametime
_BUF = 3            # buffer_frames


def _labelled_cycles(T, spans):
    """Label array with one cycle per (start, stop) span (stop exclusive)."""
    labels = np.zeros(T, dtype=np.int32)
    for i, (a, b) in enumerate(spans, start=1):
        labels[a:b] = i
    return labels


def _v_trace(T, spans, equ=1.95, depth=0.2):
    """Baseline trace with a symmetric V-shaped dip inside each span."""
    slen = np.full(T, equ, dtype=float)
    for a, b in spans:
        n = b - a
        ramp = np.abs(np.linspace(-1.0, 1.0, n))     # 1 -> 0 -> 1, min in the interior
        slen[a:b] = equ - depth * (1.0 - ramp)
    return slen


def test_cycle_truncation_flags_classifies_edges():
    T = 60
    spans = [(0, 6), (15, 25), (30, 40), (54, 60)]
    labels = _labelled_cycles(T, spans)
    trunc_start, trunc_end = contraction_analysis.cycle_truncation_flags(labels, 4, _BUF)
    assert trunc_start.tolist() == [True, False, False, False]
    assert trunc_end.tolist() == [False, False, False, True]
    # a cycle that merely comes *close* to the edge (within the buffer) also counts
    labels2 = _labelled_cycles(20, [(2, 8)])
    ts2, te2 = contraction_analysis.cycle_truncation_flags(labels2, 1, _BUF)
    assert ts2.tolist() == [True] and te2.tolist() == [False]


def test_incomplete_cycles_have_nan_duration_but_keep_complete_ones():
    T = 60
    spans = [(0, 6), (15, 25), (30, 40), (54, 60)]
    slen = _v_trace(T, spans)[None, :]
    labels = _labelled_cycles(T, spans)[None, :]
    out = contraction_analysis.analyze_contraction_parameters(
        slen, labels, np.array([4]), frametime=_FT, buffer_frames=_BUF)
    tc = out['domain_time_contr'][0]
    assert np.isnan(tc[0]) and np.isnan(tc[3])          # truncated at start / end
    assert tc[1] == pytest.approx(10 * _FT)            # complete cycles keep their duration
    assert tc[2] == pytest.approx(10 * _FT)


def test_incomplete_cycles_keep_the_timing_half_they_can_support():
    """A start-truncated cycle has no onset (no time_to_peak) but a measurable
    peak->offset; an end-truncated cycle is the mirror image."""
    T = 60
    spans = [(0, 6), (15, 25), (54, 60)]
    slen = _v_trace(T, spans)[None, :]
    labels = _labelled_cycles(T, spans)[None, :]
    out = contraction_analysis.analyze_contraction_parameters(
        slen, labels, np.array([3]), frametime=_FT, buffer_frames=_BUF)
    ttp, ttr = out['domain_time_to_peak'][0], out['domain_time_to_relax'][0]
    assert np.isnan(ttp[0]) and np.isfinite(ttr[0])     # start-truncated
    assert np.isfinite(ttp[1]) and np.isfinite(ttr[1])  # complete
    assert np.isfinite(ttp[2]) and np.isnan(ttr[2])     # end-truncated
    # the V dip is real and inside the recording, so the amplitude survives
    assert np.all(np.isfinite(out['domain_contr_max'][0][:3]))


def test_extremum_on_a_truncated_boundary_is_dropped():
    """A cycle still shortening when the recording stops never reached its peak."""
    T = 60
    slen = np.full(T, 1.95)
    slen[54:60] = np.linspace(1.93, 1.75, 6)   # monotonically falling into the last frame
    labels = _labelled_cycles(T, [(54, 60)])
    out = contraction_analysis.analyze_contraction_parameters(
        slen[None, :], labels[None, :], np.array([1]), frametime=_FT, buffer_frames=_BUF)
    assert np.isnan(out['domain_contr_max'][0, 0])     # min sits on the truncated edge
    assert np.isnan(out['domain_time_to_peak'][0, 0])
    assert np.isnan(out['domain_time_to_relax'][0, 0])
    assert np.isnan(out['domain_time_contr'][0, 0])


def test_equilibrium_excludes_edge_contraction_frames():
    """Edge cycles stay in the mask, so they no longer pollute the quiet baseline.

    Uses a long leading contraction (18 of 40 frames) — the median is robust, so the
    bias only becomes visible once the edge cycle covers a sizeable share of the
    trace, which is exactly the short-recording case this change protects.
    """
    T = 40
    spans = [(0, 18), (25, 33)]
    slen = _v_trace(T, spans)
    contr = np.zeros(T, dtype=bool)
    for a, b in spans:
        contr[a:b] = True
    equ = grouped_motion.equilibrium_over_quiet(slen, contr)
    # identical to explicitly masking every contracting frame
    assert equ == pytest.approx(np.nanmedian(slen[~contr]))
    # and strictly above the biased value you get when the edge cycle counts as quiet
    contr_edge_dropped = contr.copy()
    contr_edge_dropped[0:18] = False
    assert equ > grouped_motion.equilibrium_over_quiet(slen, contr_edge_dropped)


def test_motion_period_durations_nan_at_edges():
    from sarcasm.motion import Motion
    m = Motion.__new__(Motion)
    m.metadata = types.SimpleNamespace(frametime=_FT)
    labels = _labelled_cycles(60, [(0, 6), (15, 25), (54, 60)])
    durations = m._period_durations(labels, 3, _BUF)
    assert np.isnan(durations[0]) and np.isnan(durations[2])
    assert durations[1] == pytest.approx(10 * _FT)


# ---------------------------------------------------------------------------
# SarcAsM.group_tracks / analyze_track_motion on a synthetic data dict
# ---------------------------------------------------------------------------

def _fake_structure(n_tracks=6, T=80, frametime=0.01, seed=0):
    """A SarcAsM with a synthetic tracker output, no file IO."""
    rng = np.random.default_rng(seed)
    sarc = SarcAsM.__new__(SarcAsM)
    sarc.auto_save = False
    sarc.metadata = types.SimpleNamespace(frametime=frametime, pixelsize=0.1, n_stack=T, size=(128, 128))
    sarc.model_dir = Utils.get_models_dir()

    t = np.arange(T)
    # beating-like signal: baseline 1.95 µm with periodic contractions to ~1.72
    base = 1.95 - 0.22 * np.clip(np.sin(2 * np.pi * t / 16.0), 0, None)
    slen = np.stack([base + 0.01 * rng.standard_normal(T) for _ in range(n_tracks)])
    observed = np.ones((n_tracks, T), bool)
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
        'track_lengths': observed.sum(axis=1),
        'tracks_slen': slen.astype(np.float32),
        'tracks_positions_px': pos,
        'tracks_positions_um': pos * 0.1,
        'tracks_observed': observed,
        'tracks_midline_id': mids,
        # track i is matched to detection (vector) index i at every frame
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
    snp = sarc.data['tracks_observed'].copy()
    snp[3, 10:] = False
    sarc.data['tracks_observed'] = snp
    sarc.data['track_lengths'] = snp.sum(axis=1)
    labels = np.array([0, 0, 1, 1])
    sarc.group_tracks(by='custom', labels=labels, min_coverage=0.5)
    gid = sarc.data['track_group_id']
    assert gid[0] == gid[1]
    assert gid[3] == -1               # dropped by min_coverage
    assert sarc.data['n_groups'] == 2  # labels 0 and 1 both still present (track 2)


def test_group_tracks_min_group_size_drops_and_renumbers():
    """Under-sized groups are unassigned and the survivors renumbered contiguously."""
    sarc = _fake_structure(n_tracks=6)
    # unbalanced M-bands: tracks 0-4 -> midline 5, track 5 alone -> midline 9
    mids = np.full((6, sarc.data['tracks_slen'].shape[1]), 5, np.int32)
    mids[5] = 9
    sarc.data['tracks_midline_id'] = mids

    sarc.group_tracks(by='mband', reference_frame=0, min_group_size=1)
    assert sarc.data['n_groups'] == 2
    assert sarc.data['group_member_counts'].tolist() == [5, 1]

    sarc.group_tracks(by='mband', reference_frame=0, min_group_size=2)
    gid = sarc.data['track_group_id']
    assert gid[5] == -1                                   # the 1-track group is dropped
    assert gid[:5].tolist() == [0] * 5                    # survivor renumbered to 0
    assert sarc.data['n_groups'] == 1
    assert sarc.data['group_member_counts'].tolist() == [5]
    assert sarc.data['params.group_tracks.min_group_size'] == 2


def test_group_tracks_min_group_size_keeps_fixed_label_space():
    """'domain' keeps its mask label space; an under-sized domain just empties out."""
    sarc = _fake_structure_domain(n_per=3)
    pos = sarc.data['tracks_positions_px'].copy()
    pos[4:, :, 0] = 55.0        # rows 50-60 are between masks -> tracks 4,5 unassigned
    sarc.data['tracks_positions_px'] = pos
    sarc.data['tracks_positions_um'] = pos * sarc.metadata.pixelsize

    sarc.group_tracks(by='domain', reference_frame=0, min_group_size=1)
    assert sarc.data['group_member_counts'].tolist() == [3, 1, 0]

    sarc.group_tracks(by='domain', reference_frame=0, min_group_size=2)
    gid = sarc.data['track_group_id']
    assert gid[:3].tolist() == [0, 0, 0]                  # label 1 -> group 0 unchanged
    assert gid[3] == -1                                   # the 1-track domain is dropped
    assert sarc.data['n_groups'] == 3                     # label space preserved
    assert sarc.data['group_member_counts'].tolist() == [3, 0, 0]


def test_analyze_track_motion_forwards_min_group_size():
    sarc = _fake_structure(n_tracks=6)
    mids = np.full((6, sarc.data['tracks_slen'].shape[1]), 5, np.int32)
    mids[5] = 9
    sarc.data['tracks_midline_id'] = mids
    sarc.analyze_track_motion(by='mband', reference_frame=0, min_group_size=2)
    assert sarc.data['n_groups'] == 1
    assert np.asarray(sarc.data['mband_slen_timeseries']).shape[0] == 1


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
    # all gaps here are interior, so they interpolate away and diff(z) == slen
    assert np.allclose(np.diff(z, axis=0), slen)
    assert not np.isnan(slen).any()
    assert np.allclose(z[0], 0.0)
    assert np.allclose(time, np.arange(5) * 0.01)


def test_synthesize_loi_chain_edge_nan_never_constant():
    # A member must never be extended with a held-constant length where it has
    # no observation: leading/trailing gaps stay NaN, only interior gaps fill.
    member = np.array([
        [np.nan, np.nan, 1.8, 1.8, 1.8],   # appears at t=2 -> leading-edge NaN
        [2.0, 2.0, 2.0, np.nan, np.nan],   # lost after t=2 -> trailing-edge NaN
        [1.9, 1.9, np.nan, 1.9, 1.9],      # interior gap -> interpolated
    ])
    z, slen, time = grouped_motion.synthesize_loi_chain(member, frametime=0.01)
    # leading/trailing edges are NaN, not the constant last value
    assert np.isnan(slen[0, 0]) and np.isnan(slen[0, 1])
    assert np.isnan(slen[1, 3]) and np.isnan(slen[1, 4])
    # interior gap interpolated (here both anchors are 1.9)
    assert not np.isnan(slen[2, 2]) and np.isclose(slen[2, 2], 1.9)
    # a member keeps its own length even when a sibling is undefined that frame
    assert np.isclose(slen[0, 2], 1.8)
    # z_pos boundary 0 is always the origin; undefined members propagate NaN into
    # the boundaries below them (no fabricated arc-length).
    assert np.allclose(z[0], 0.0)
    assert np.isnan(z[1, 0])  # member 0 undefined at t=0 -> all boundaries below NaN


def _straight_chain_pos(K, T, slen=1.9, curve=0.0):
    """(K, T, 2) positions of K sarcomeres spaced one slen apart along a fibre."""
    pos = np.zeros((K, T, 2))
    for k in range(K):
        pos[k, :, 0] = curve * k ** 2      # 0 -> straight fibre, >0 -> curved
        pos[k, :, 1] = k * slen
    return pos


def test_synthesize_loi_chain_from_positions_matches_lengths():
    """With measured positions, z_pos is built per member yet still reproduces the
    member lengths on a clean, evenly-spaced chain."""
    K, T, SL = 4, 6, 1.9
    member = np.full((K, T), SL)
    pos = _straight_chain_pos(K, T, SL)
    z, slen, _ = grouped_motion.synthesize_loi_chain(member, 0.01, member_pos=pos)
    assert z.shape == (K + 1, T)
    assert np.allclose(np.diff(z, axis=0), slen)
    assert np.allclose(z[-1] - z[0], K * SL)      # fibre spans K sarcomeres


def test_synthesize_loi_chain_gap_blanks_only_its_own_boundary():
    """The regression: a member that drops out must not blank the boundaries below
    it. Accumulating the lengths did exactly that."""
    K, T, SL = 4, 6, 1.9
    member = np.full((K, T), SL)
    pos = _straight_chain_pos(K, T, SL)
    member[0, 4:] = np.nan          # head member lost from t=4 (worst case)
    pos[0, 4:, :] = np.nan
    z, slen, _ = grouped_motion.synthesize_loi_chain(member, 0.01, member_pos=pos)

    assert np.isnan(slen[0, 4:]).all()            # honest: the member is undefined
    assert np.isfinite(slen[1:, 4:]).all()        # its siblings are unaffected
    assert np.isnan(z[0, 4:]).all()               # only its own boundary is blank
    assert np.isfinite(z[1:, 4:]).all()           # every boundary below survives

    # the legacy accumulate path (no positions) still propagates - kept as fallback
    z_cumsum, _, _ = grouped_motion.synthesize_loi_chain(member, 0.01)
    assert np.isnan(z_cumsum[1:, 4:]).all()


def test_synthesize_loi_chain_does_not_bridge_long_dropouts():
    """A long dropout must stay NaN: interpolating it draws a straight line that can
    span a whole contraction and silently smooth it away. Short flicker still fills."""
    K, T, SL = 3, 80, 1.9
    member = np.full((K, T), SL)
    pos = _straight_chain_pos(K, T, SL)
    member[1, 10:12] = np.nan            # 2 frames  = 0.02 s -> bridged
    pos[1, 10:12] = np.nan
    member[1, 30:60] = np.nan            # 30 frames = 0.30 s -> must stay NaN
    pos[1, 30:60] = np.nan
    z, slen, _ = grouped_motion.synthesize_loi_chain(
        member, 0.01, member_pos=pos)     # default cap 0.05 s = 5 frames

    assert np.isfinite(slen[1, 10:12]).all()
    assert not np.isfinite(slen[1, 30:60]).any()
    assert not np.isfinite(z[1, 30:60]).any()     # z_pos inherits the gap
    assert np.isfinite(slen[0]).all() and np.isfinite(slen[2]).all()   # siblings intact


def test_interp_gap_cap_is_a_physical_duration():
    """The cap is in seconds, so the same real dropout is bridged at any frame rate."""
    K, T, SL = 2, 80, 1.9
    member = np.full((K, T), SL)
    member[1, 30:60] = np.nan                     # 30 frames
    pos = _straight_chain_pos(K, T, SL)
    pos[1, 30:60] = np.nan
    # 0.01 s/frame -> 30 frames is 0.30 s, far over the 0.05 s cap
    _, slow, _ = grouped_motion.synthesize_loi_chain(member, 0.01, member_pos=pos)
    assert not np.isfinite(slow[1, 30:60]).any()
    # 0.001 s/frame -> the same 30 frames is only 0.03 s, under the cap
    _, fast, _ = grouped_motion.synthesize_loi_chain(member, 0.001, member_pos=pos)
    assert np.isfinite(fast[1, 30:60]).all()
    # opting out restores the old fill-everything behaviour
    _, off, _ = grouped_motion.synthesize_loi_chain(
        member, 0.01, member_pos=pos, max_interp_seconds=None)
    assert np.isfinite(off[1, 30:60]).all()


def test_synthesize_loi_chain_starts_at_zero():
    """z_pos must not go negative: the arc origin sits at the first member's
    *centre*, so its leading edge would fall below 0 and be clipped away by plots
    that assume a non-negative Z-band position (plot_z_pos sets ylim(0, None))."""
    K, T, SL = 4, 6, 1.9
    member = np.full((K, T), SL)
    pos = _straight_chain_pos(K, T, SL)
    z, slen, _ = grouped_motion.synthesize_loi_chain(member, 0.01, member_pos=pos)
    assert np.nanmin(z) == pytest.approx(0.0)
    assert (z[np.isfinite(z)] >= -1e-9).all()
    assert np.allclose(np.diff(z, axis=0), slen)          # spacing preserved


def test_group_tracks_chain_keeps_partially_tracked_sarcomeres():
    """A chain represents the fibre's geometry, so a sarcomere tracked only part of
    the time must stay in it — dropping it punches a hole into z_pos. Pooled
    groupings keep the strict floor, where the filter removes noise from a mean."""
    sarc = _fake_structure(n_tracks=4)
    T = sarc.data['tracks_slen'].shape[1]
    # member 2 is only tracked 30% of the time -> below the 0.5 pooled floor
    snp = sarc.data['tracks_observed'].copy()
    snp[2, int(0.3 * T):] = False
    sarc.data['tracks_observed'] = snp
    sarc.data['track_lengths'] = snp.sum(axis=1)
    pos = np.zeros((4, T, 2), np.float32)
    pos[:, :, 0] = (np.arange(4) * _SLEN_PX)[:, None]
    pos[:, :, 1] = 20.0
    sarc.data['tracks_positions_px'] = pos
    sarc.data['tracks_positions_um'] = pos * 0.1
    sarc.data['loi_data'] = {
        'loi_lines': [np.array([[0, 20], [3 * _SLEN_PX, 20]], float)]}

    sarc.group_tracks(by='loi', reference_frame=0)          # chain default (0.1)
    assert int((sarc.data['track_group_id'] >= 0).sum()) == 4, 'chain lost a sarcomere'

    sarc.group_tracks(by='pool')                            # pooled default (0.5)
    assert sarc.data['track_group_id'][2] == -1
    # an explicit value always wins over the per-kind default
    sarc.group_tracks(by='loi', reference_frame=0, min_coverage=0.5)
    assert sarc.data['track_group_id'][2] == -1


def test_synthesize_loi_chain_arc_is_monotone_on_a_curved_fibre():
    K, T, SL = 5, 4, 1.9
    member = np.full((K, T), SL)
    pos = _straight_chain_pos(K, T, SL, curve=0.4)      # bent fibre
    z, _, _ = grouped_motion.synthesize_loi_chain(member, 0.01, member_pos=pos)
    assert np.all(np.diff(z[:, 0]) > 0)                 # arc increases along the chain


def _fake_motion(loi_data, frametime=0.01):
    from sarcasm.motion import Motion
    m = Motion.__new__(Motion)
    m.metadata = types.SimpleNamespace(frametime=frametime)
    m.loi_data = loi_data
    return m


def test_member_slen_prefers_the_stored_series_for_a_synthetic_chain():
    """A synthesized chain carries the honest per-member lengths; re-deriving them
    from diff(z_pos) would blank every member below an undefined one."""
    K, T, SL = 4, 6, 1.9
    member = np.full((K, T), SL)
    pos = _straight_chain_pos(K, T, SL)
    member[0, 4:] = np.nan
    pos[0, 4:, :] = np.nan
    z, slen, _ = grouped_motion.synthesize_loi_chain(member, 0.01, member_pos=pos)

    m = _fake_motion({'z_pos': z, 'slen': slen, 'synthetic': True})
    got = m._member_slen()
    assert np.array_equal(np.isnan(got), np.isnan(slen))
    assert np.count_nonzero(~np.isnan(got[:, 5])) == K - 1     # only the lost member is gone
    got[:] = 0                                                 # must be a copy
    assert not np.allclose(np.asarray(m.loi_data['slen']), 0)


def test_member_slen_falls_back_to_diff_for_a_legacy_loi():
    z = np.cumsum(np.full((4, 5), 1.8), axis=0)
    m = _fake_motion({'z_pos': z})                     # no 'synthetic' flag
    assert np.allclose(m._member_slen(), np.diff(z, axis=0))
    # a stored slen on a non-synthetic LOI is still ignored (legacy semantics)
    m2 = _fake_motion({'z_pos': z, 'slen': np.full((3, 5), 99.0)})
    assert np.allclose(m2._member_slen(), np.diff(z, axis=0))


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
    from sarcasm.plotting.plots import Plots
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


def test_plot_tracks_lines_smoke():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from sarcasm.plotting.plots import Plots
    # stub the image background (needs file IO)
    orig_z, orig_img = Plots.plot_z_bands, Plots.plot_image
    Plots.plot_z_bands = staticmethod(lambda ax, *a, **k: None)
    Plots.plot_image = staticmethod(lambda ax, *a, **k: None)
    try:
        sarc = _fake_structure(n_tracks=6)
        # give the trajectories a little motion so the lines are non-degenerate
        pos = np.asarray(sarc.data['tracks_positions_px'], dtype=np.float32)
        pos[:, :, 1] += np.linspace(0, 2, pos.shape[1])[None, :]
        sarc.data['tracks_positions_px'] = pos
        # color_by='group' requires a grouping first
        with pytest.raises(ValueError):
            Plots.plot_tracks(plt.figure().gca(), sarc, color_by='group')
        # 'coverage' / 'slen' draw one trajectory LineCollection
        for cb in ('coverage', 'slen'):
            ax = plt.figure().gca()
            Plots.plot_tracks(ax, sarc, color_by=cb, scalebar=False)
            assert any(isinstance(c, LineCollection) for c in ax.collections)
        # 'group' colouring works once grouped
        sarc.group_tracks(by='pool')
        ax = plt.figure().gca()
        Plots.plot_tracks(ax, sarc, color_by='group', scalebar=False)
        assert any(isinstance(c, LineCollection) for c in ax.collections)
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

# A sarcomere is ~1.95 µm at pixelsize 0.1 in _fake_structure -> ~19.5 px. Consecutive
# sarcomeres along a fibre are therefore ~20 px apart, not adjacent pixels.
_SLEN_PX = 19.5


def test_group_tracks_loi_assigns_and_orders():
    sarc = _fake_structure(n_tracks=6)
    T = sarc.data['tracks_slen'].shape[1]
    # two well-separated fibres, each a chain of 3 sarcomeres one slen apart
    pos = np.zeros((6, T, 2), np.float32)
    rows = np.array([2 * _SLEN_PX, 0.0, _SLEN_PX,              # fibre 0, shuffled
                     150 + 2 * _SLEN_PX, 150.0, 150 + _SLEN_PX], float)
    pos[:, :, 0] = rows[:, None]; pos[:, :, 1] = 20.0
    sarc.data['tracks_positions_px'] = pos
    sarc.data['tracks_positions_um'] = pos * 0.1
    # curated LOI polylines (px), each running along one fibre
    line0 = np.array([[0, 20], [2 * _SLEN_PX, 20]], float)
    line1 = np.array([[150, 20], [150 + 2 * _SLEN_PX, 20]], float)
    sarc.data['loi_data'] = {'loi_lines': [line0, line1]}

    sarc.group_tracks(by='loi', reference_frame=0)
    gid = sarc.data['track_group_id']
    assert sarc.data['group_kind'] == 'loi'
    assert sarc.data['n_groups'] == 2
    assert gid[[1, 2, 0]].tolist() == [0, 0, 0]
    assert gid[[4, 5, 3]].tolist() == [1, 1, 1]
    # order is rank along the line (by arc length / row), independent of track index
    order = sarc.data['track_group_order']
    assert order[1] == 0 and order[2] == 1 and order[0] == 2
    assert order[4] == 0 and order[5] == 1 and order[3] == 2


def test_group_tracks_loi_is_a_thread_not_a_band():
    """Tracks stacked laterally at the same position along the line belong to
    *parallel* fibres. An LOI is one 1D thread, so only one per step is kept."""
    sarc = _fake_structure(n_tracks=6)
    T = sarc.data['tracks_slen'].shape[1]
    pos = np.zeros((6, T, 2), np.float32)
    # 3 arc positions x 2 lateral neighbours (cols 20 and 26 — under 0.5 slen away)
    pos[:, :, 0] = np.repeat([0.0, _SLEN_PX, 2 * _SLEN_PX], 2)[:, None]
    pos[:, :, 1] = np.tile([20.0, 26.0], 3)[:, None]
    sarc.data['tracks_positions_px'] = pos
    sarc.data['tracks_positions_um'] = pos * 0.1
    sarc.data['loi_data'] = {
        'loi_lines': [np.array([[0, 20], [2 * _SLEN_PX, 20]], float)]}

    sarc.group_tracks(by='loi', reference_frame=0)
    gid = sarc.data['track_group_id']
    assert int((gid == 0).sum()) == 3          # one per step, not all 6
    assert gid[[0, 2, 4]].tolist() == [0, 0, 0]   # the on-line ones (col 20) win
    assert gid[[1, 3, 5]].tolist() == [-1, -1, -1]
    assert sorted(sarc.data['track_group_order'][gid == 0]) == [0, 1, 2]


def test_group_tracks_loi_uses_the_detection_chain_when_available():
    """detect_lois keeps each LOI's ordered detection chain; grouping must follow it
    (identical mechanism to by='myofibril'), not re-derive membership from geometry."""
    sarc = _fake_structure(n_tracks=4)
    T = sarc.data['tracks_slen'].shape[1]
    # scatter the positions so a geometric assignment could not produce this order
    pos = np.zeros((4, T, 2), np.float32)
    pos[:, :, 0] = np.array([0.0, 300.0, 150.0, 900.0], float)[:, None]
    pos[:, :, 1] = 20.0
    sarc.data['tracks_positions_px'] = pos
    sarc.data['tracks_positions_um'] = pos * 0.1
    # tracks_detection_id maps track i -> detection i in _fake_structure
    sarc.data['loi_data'] = {
        'loi_lines': [np.array([[0, 20], [300, 20]], float)],
        'loi_index_lines': [np.array([2, 0, 1])],      # chain: track 2 -> 0 -> 1
    }

    sarc.group_tracks(by='loi', reference_frame=0)
    gid, order = sarc.data['track_group_id'], sarc.data['track_group_order']
    assert gid.tolist() == [0, 0, 0, -1]           # track 3 is not on the chain
    assert order[2] == 0 and order[0] == 1 and order[1] == 2   # chain order, not geometry


def test_group_tracks_loi_far_tracks_unassigned():
    sarc = _fake_structure(n_tracks=3)
    T = sarc.data['tracks_slen'].shape[1]
    pos = np.zeros((3, T, 2), np.float32)
    pos[:, :, 0] = np.array([0.0, _SLEN_PX, 900.0], float)[:, None]  # track 2 far away
    pos[:, :, 1] = 20.0
    sarc.data['tracks_positions_px'] = pos
    sarc.data['loi_data'] = {'loi_lines': [np.array([[0, 20], [_SLEN_PX, 20]], float)]}
    sarc.group_tracks(by='loi')
    gid = sarc.data['track_group_id']
    assert gid[0] == 0 and gid[1] == 0
    assert gid[2] == -1            # > 0.5*slen from the line -> dropped


def test_group_tracks_loi_missing_prereq_raises():
    sarc = _fake_structure()
    with pytest.raises(ValueError, match='LOI lines not available'):
        sarc.group_tracks(by='loi')
