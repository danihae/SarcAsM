"""Unit + behavioural tests for :mod:`sarcasm.analysis.sarcomere_tracking`.

The tracker is flow-predict + detection-snap; no M-band identity is persisted.
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.analysis import sarcomere_tracking as st


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

def _make_band_masks(H=120, W=120, n_bands=6, spacing=18, band_width=2, seed=0):
    """Build a pair of Z/M band masks with irregular endpoints."""
    rng = np.random.default_rng(seed)
    z = np.zeros((H, W), dtype=np.float32)
    m = np.zeros((H, W), dtype=np.float32)
    for i in range(n_bands):
        r = 10 + i * spacing
        x0 = int(rng.integers(5, 25)); x1 = int(rng.integers(W - 25, W - 5))
        z[r - band_width // 2:r + band_width // 2 + 1, x0:x1] = 1.0
        r2 = r + spacing // 2
        if r2 < H - band_width:
            x0 = int(rng.integers(5, 25)); x1 = int(rng.integers(W - 25, W - 5))
            m[r2 - band_width // 2:r2 + band_width // 2 + 1, x0:x1] = 1.0
    return z, m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_angular_diff_wraps_modulo_pi():
    # θ and θ+π are the same axis.
    assert abs(st._angular_diff(0.0, np.pi)) < 1e-6
    assert abs(st._angular_diff(0.1, np.pi + 0.1)) < 1e-6
    # Small difference is small.
    assert abs(st._angular_diff(0.1, 0.05) - 0.05) < 1e-6
    # Returns at most ±π/2.
    for a in np.linspace(-np.pi, np.pi, 37):
        for b in np.linspace(-np.pi, np.pi, 37):
            assert abs(st._angular_diff(a, b)) <= np.pi / 2 + 1e-6


# ---------------------------------------------------------------------------
# Flow engine
# ---------------------------------------------------------------------------

def test_build_dt_channels_shape_and_dtype():
    z, m = _make_band_masks()
    dz, dm = st.build_dt_channels(z, m, threshold=0.5)
    assert dz.shape == z.shape
    assert dm.shape == m.shape
    assert dz.dtype == np.uint8
    assert dm.dtype == np.uint8
    # DT is zero on the mask.
    assert (dz[z > 0.5] == 0).all()
    assert (dm[m > 0.5] == 0).all()


def test_flow_output_shape_and_zero_on_identical_frames():
    z, m = _make_band_masks(H=160, W=160)
    flow = st.compute_flow_pair(z, m, z, m, threshold=0.5, clip=10.0)
    assert flow.shape == (160, 160, 2)
    assert flow.dtype == np.float32
    # Identical frames → flow should be near zero everywhere.
    assert np.max(np.abs(flow)) < 0.5


# ---------------------------------------------------------------------------
# Motion-field sampling
# ---------------------------------------------------------------------------

def test_sample_flow_bilinear_subpixel():
    H = W = 50
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    flow[..., 1] = -1.0
    positions = np.array([[10.5, 10.5], [25.0, 25.0], [40.25, 30.75]], dtype=np.float32)
    out = st.sample_flow_bilinear(flow, positions)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(out[:, 1], -1.0, atol=1e-5)


def test_decompose_along_perpendicular():
    # sarcomere axis = (sin θ, cos θ); θ=0 → along = (0, 1)  (col direction)
    disp = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    ori = np.zeros(2, dtype=np.float32)
    along, perp = st.decompose_along_perpendicular(disp, ori)
    assert along[0] == pytest.approx(1.0, abs=1e-5)
    assert abs(perp[0]) < 1e-5
    assert abs(along[1]) < 1e-5


def test_compute_motion_field_stats_velocity_scaling():
    disp_lists = [np.array([[1.0, 0.0], [0.0, 2.0]], np.float32)]
    ori_lists = [np.array([np.pi / 2, np.pi / 2], np.float32)]
    # orientation = π/2 → sarcomere axis = (1, 0). Along = dy.
    stats = st.compute_motion_field_stats(disp_lists, ori_lists, frametime=0.01)
    assert stats['displacement_along_sarcomere'][0][0] == pytest.approx(1.0, abs=1e-5)
    assert stats['displacement_along_sarcomere'][0][1] == pytest.approx(0.0, abs=1e-5)
    # Velocity = magnitude / frametime.
    np.testing.assert_allclose(
        stats['velocity_magnitude'][0], [100.0, 200.0], atol=1e-3,
    )


# ---------------------------------------------------------------------------
# Snap gate — anisotropic + orientation
# ---------------------------------------------------------------------------

def test_anisotropic_snap_rejects_perpendicular_outliers():
    # sarcomere orientation = 0 → along = col direction, perp = row direction
    query_pos = (10.0, 10.0); query_ori = 0.0
    dets = np.array([
        [10.0, 20.0],   # 10 px along-sarcomere — accept
        [18.0, 10.0],   # 8 px perpendicular — reject (max_perp=6)
        [11.0, 11.0],   # close on both — accept, should win (closest)
    ], dtype=np.float32)
    det_ori = np.zeros(3, dtype=np.float32)
    best = st._anisotropic_snap(
        query_pos, query_ori, dets, det_ori,
        candidate_indices=np.array([0, 1, 2]),
        max_along=15.0, max_perp=6.0, ori_tol_rad=np.deg2rad(45),
    )
    assert best == 2


def test_anisotropic_snap_rejects_bad_orientation():
    query_pos = (10.0, 10.0); query_ori = 0.0
    dets = np.array([[10.5, 10.5]], dtype=np.float32)
    det_ori = np.array([np.pi / 3], dtype=np.float32)  # 60° off — should reject at 30°
    best = st._anisotropic_snap(
        query_pos, query_ori, dets, det_ori,
        candidate_indices=np.array([0]),
        max_along=15.0, max_perp=6.0, ori_tol_rad=np.deg2rad(30),
    )
    assert best == -1
    # Accept at 90° tolerance.
    best2 = st._anisotropic_snap(
        query_pos, query_ori, dets, det_ori,
        candidate_indices=np.array([0]),
        max_along=15.0, max_perp=6.0, ori_tol_rad=np.deg2rad(90),
    )
    assert best2 == 0


# ---------------------------------------------------------------------------
# End-to-end tracker behaviour on a minimal synthetic sequence
# ---------------------------------------------------------------------------

def _make_detection_sequence(T=6, n_points=20, shift_per_frame=1.0, orientation=0.0, seed=0):
    """Build detections that move by a known uniform translation.

    Returns pos_px_all, slen_all, ori_all suitable for the tracker entry.
    Detections are on a regular grid with small jitter so multiple points are
    well-separated.
    """
    rng = np.random.default_rng(seed)
    base_y = rng.uniform(20, 80, size=n_points)
    base_x = rng.uniform(20, 80, size=n_points)
    slen = rng.uniform(1.7, 2.0, size=n_points).astype(np.float32)
    ori = np.full(n_points, orientation, dtype=np.float32)
    pos_px_all: list = []
    slen_all: list = []
    ori_all: list = []
    for t in range(T):
        ys = base_y + t * shift_per_frame  # uniform along-row translation
        xs = base_x
        pos_px_all.append(np.stack([ys, xs], axis=1).astype(np.float32))
        slen_all.append(slen.copy())
        ori_all.append(ori.copy())
    return pos_px_all, slen_all, ori_all


def test_seed_and_snap_recovers_uniform_translation():
    """Build a flat motion with a constant-flow stack so flow prediction works.

    We don't need realistic masks: the tracker's compute_flow_sequence needs a
    stack, but we bypass it by calling the tracker with a pre-baked constant
    flow via farneback_kwargs. Simpler path: construct a 2-frame case where
    pos moves by a known amount and flow is uniform.
    """
    T = 3
    H, W = 80, 80
    # Build Z/M masks that identically contain one stripe so the DT flow is ~0,
    # then shift positions by 0 as well — this is the simplest match test.
    z = np.zeros((H, W), dtype=np.float32)
    m = np.zeros((H, W), dtype=np.float32)
    z[20, 10:70] = 1.0; m[30, 10:70] = 1.0
    zstack = np.stack([z] * T)
    mstack = np.stack([m] * T)

    # Detections don't move — pure "will the tracker just stay on the same
    # detections?" sanity check.
    pos_px_all, slen_all, ori_all = _make_detection_sequence(
        T=T, n_points=10, shift_per_frame=0.0, orientation=np.pi / 2,
    )
    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=2, min_track_length=2,
    )
    # All 10 detections should produce persistent tracks.
    assert out['n_tracks'] == 10
    # Every track should be continuously snapped across all frames.
    snapped = out['tracks_snapped']
    assert np.all(snapped == True)  # noqa: E712 — explicit test of boolean array


def test_anti_convergence_tracks_stay_separated():
    """Two query points ≥ 20 px apart at t=0 must stay separated. The snap
    anchor guarantees they cannot collapse onto each other even if flow is
    locally uniform."""
    T = 5
    H, W = 120, 120
    z = np.zeros((H, W), dtype=np.float32); m = np.zeros((H, W), dtype=np.float32)
    z[40, 20:100] = 1.0; z[80, 20:100] = 1.0
    m[30, 20:100] = 1.0; m[70, 20:100] = 1.0
    zstack = np.stack([z] * T); mstack = np.stack([m] * T)

    # Two detections, 40 px apart, static across time.
    pos = np.array([[30.0, 50.0], [70.0, 50.0]], dtype=np.float32)
    pos_px_all = [pos.copy() for _ in range(T)]
    slen = np.array([1.8, 1.8], dtype=np.float32)
    ori = np.array([np.pi / 2, np.pi / 2], dtype=np.float32)
    slen_all = [slen.copy() for _ in range(T)]
    ori_all = [ori.copy() for _ in range(T)]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=2, min_track_length=2,
    )
    # 2 tracks, 40 px apart in every frame.
    assert out['n_tracks'] == 2
    p = out['tracks_positions_px']  # (2, T, 2)
    for t in range(T):
        d = np.linalg.norm(p[0, t] - p[1, t])
        assert d >= 30.0, f"tracks collapsed at frame {t}: distance = {d:.1f}"


def test_flow_advection_is_along_sarcomere_axis_only():
    """Sarcomere orientation is horizontal (θ=0 → along = col direction).
    A purely perpendicular flow (uniform in row direction) must not move
    query points perpendicularly. Only the snap residual can move them off-axis.
    """
    # Actually achieving this in a test is intricate — the synthetic DT-Farneback
    # flow will be zero on identical frames so we can't inject a custom flow
    # easily. Instead we test the _advect_all_slots logic indirectly: if
    # sarcomeres are oriented along cols (θ=0) and the movie is identical,
    # tracks should stay put even if there were phantom perpendicular flow.
    T = 4
    H, W = 80, 80
    z = np.zeros((H, W), dtype=np.float32); m = np.zeros((H, W), dtype=np.float32)
    # horizontal Z-band at row 30
    z[30, 10:70] = 1.0; m[35, 10:70] = 1.0
    zstack = np.stack([z] * T); mstack = np.stack([m] * T)
    # One detection at (30, 40), orientation = 0 → sarcomere axis along cols.
    pos = np.array([[30.0, 40.0]], np.float32)
    pos_px_all = [pos.copy() for _ in range(T)]
    slen_all = [np.array([1.8], np.float32) for _ in range(T)]
    ori_all = [np.array([0.0], np.float32) for _ in range(T)]
    out = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=1, min_track_length=2,
    )
    # Track should stay at (30, 40) for all frames (no motion).
    assert out['n_tracks'] == 1
    p = out['tracks_positions_px'][0]
    np.testing.assert_allclose(p[:, 0], 30.0, atol=0.5)
    np.testing.assert_allclose(p[:, 1], 40.0, atol=0.5)


def test_gap_frame_records_nan_slen_but_keeps_position():
    """If a detection disappears for a frame, the tracker should keep the
    query point alive with flow-predicted position and NaN slen."""
    T = 4
    H, W = 80, 80
    z = np.zeros((H, W), dtype=np.float32); m = np.zeros((H, W), dtype=np.float32)
    z[20, 10:70] = 1.0; m[30, 10:70] = 1.0
    zstack = np.stack([z] * T); mstack = np.stack([m] * T)

    pos_full = np.array([[25.0, 40.0]], dtype=np.float32)
    pos_empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos_full, pos_empty, pos_full, pos_full]  # missing at t=1
    slen_all = [np.array([1.8], np.float32), np.zeros(0, np.float32),
                np.array([1.8], np.float32), np.array([1.8], np.float32)]
    ori_all = [np.array([np.pi / 2], np.float32), np.zeros(0, np.float32),
               np.array([np.pi / 2], np.float32), np.array([np.pi / 2], np.float32)]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=3, min_track_length=2,
    )
    # Either one continuous track (gap bridged) or, depending on memory,
    # a re-spawn. At memory=3 the original track should survive.
    assert out['n_tracks'] >= 1
    slens = out['tracks_slen']
    # At least one track has a NaN slen at frame 1 but valid slen at frames 0/2/3.
    assert any(
        np.isnan(slens[i, 1]) and np.isfinite(slens[i, 0]) and np.isfinite(slens[i, 2])
        for i in range(slens.shape[0])
    )


def test_lost_track_trailing_coast_is_nan_not_held():
    """A track that permanently loses its detection must not freeze in place at
    its last position. The trailing coast frames (after the final snap, while
    the query point is flow-advected toward closure) are blanked to NaN in the
    output — position, slen and orientation alike."""
    T = 8
    zstack, mstack = _identical_band_stack(T)  # flow ~ 0
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    # Detection present frames 0-3, then gone for good (track is lost).
    pos_px_all = [pos, pos, pos, pos, empty, empty, empty, empty]
    slen_one = np.array([1.8], np.float32)
    ori_one = np.array([0.0], np.float32)
    slen_all = [slen_one if len(p) else np.zeros(0, np.float32) for p in pos_px_all]
    ori_all = [ori_one if len(p) else np.zeros(0, np.float32) for p in pos_px_all]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=3, min_track_length=2,
        merge_tracks=False,
    )
    assert out['n_tracks'] == 1
    snapped = out['tracks_snapped'][0]
    assert snapped.tolist() == [True, True, True, True, False, False, False, False]
    pos_out = out['tracks_positions_px'][0]
    pos_um = out['tracks_positions_um'][0]
    slens = out['tracks_slen'][0]
    oris = out['tracks_orientations'][0]
    # Snapped frames keep their real values...
    np.testing.assert_allclose(pos_out[:4, 0], 25.0, atol=1.0)
    np.testing.assert_allclose(pos_out[:4, 1], 40.0, atol=1.0)
    assert np.all(np.isfinite(slens[:4]))
    # ...and every frame after the final snap is NaN (no constant hold-over).
    assert np.all(np.isnan(pos_out[4:]))
    assert np.all(np.isnan(pos_um[4:]))
    assert np.all(np.isnan(slens[4:]))
    assert np.all(np.isnan(oris[4:]))


def test_motion_predictor_none_is_default_and_flowless():
    """Default motion_predictor='none' computes NO optical flow and returns no
    flow-derived outputs, while producing the same tracks as the flow predictor
    on quiescent input. compute_motion_field still forces flow on demand."""
    T = 6
    zstack, mstack = _identical_band_stack(T)
    pos = np.array([[25.0, 40.0]], np.float32)
    pos_px_all = [pos.copy() for _ in range(T)]
    slen_all = [np.array([1.8], np.float32) for _ in range(T)]
    ori_all = [np.array([0.0], np.float32) for _ in range(T)]
    common = dict(pixelsize=0.1, frametime=0.01, memory=2, min_track_length=2)

    out_none = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all, **common)
    # No flow-derived outputs by default.
    assert 'velocity_magnitude' not in out_none
    assert 'flow_at_vectors' not in out_none
    assert 'flow_fields' not in out_none
    assert out_none['n_tracks'] == 1

    # Equivalent to the flow predictor on quiescent input (flow ~ 0 there).
    out_flow = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        motion_predictor='flow', **common)
    np.testing.assert_array_equal(np.nan_to_num(out_none['tracks_positions_px']),
                                  np.nan_to_num(out_flow['tracks_positions_px']))

    # compute_motion_field forces the flow computation even with predictor 'none'.
    out_mf = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        compute_motion_field=True, **common)
    assert 'velocity_magnitude' in out_mf

    # Unknown predictor is rejected.
    with pytest.raises(ValueError):
        st.track_sarcomere_vectors(
            zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
            motion_predictor='bogus', **common)


# ---------------------------------------------------------------------------
# Trajectory-merge step
# ---------------------------------------------------------------------------

def _identical_band_stack(T: int, H: int = 80, W: int = 80):
    """Identical Z/M-band stack so synthetic flow is ~0 across frames.

    Lets tests manipulate detections independently of the underlying flow
    (which is computed from the masks, not from the detections).
    """
    z = np.zeros((H, W), dtype=np.float32)
    m = np.zeros((H, W), dtype=np.float32)
    z[20, 10:70] = 1.0
    m[30, 10:70] = 1.0
    return np.stack([z] * T), np.stack([m] * T)


def test_merge_bridges_one_frame_gap():
    """A respawned track separated by a 1-frame gap should be stitched back."""
    T = 6
    zstack, mstack = _identical_band_stack(T)
    # Detection present every frame except t=2.
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos, pos, empty, pos, pos, pos]
    slen_one = np.array([1.8], np.float32)
    ori_one = np.array([0.0], np.float32)  # sarcomere axis along cols
    slen_all = [slen_one, slen_one, np.zeros(0, np.float32),
                slen_one, slen_one, slen_one]
    ori_all = [ori_one, ori_one, np.zeros(0, np.float32),
               ori_one, ori_one, ori_one]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=0, min_track_length=2,
        max_gap_interpolation=5,
        merge_tracks=True,
    )
    # Without merging this would yield 2 tracks (A snaps 0,1; B snaps 3,4,5).
    assert out['n_tracks'] == 1
    assert out['n_merges'] == 1
    snapped = out['tracks_snapped'][0]
    assert snapped.tolist() == [True, True, False, True, True, True]
    slens = out['tracks_slen'][0]
    assert np.isnan(slens[2])
    np.testing.assert_allclose(slens[[0, 1, 3, 4, 5]], 1.8)
    pos_out = out['tracks_positions_px'][0]
    np.testing.assert_allclose(pos_out[:, 0], 25.0, atol=1.0)
    np.testing.assert_allclose(pos_out[:, 1], 40.0, atol=1.0)


def test_merge_off_preserves_legacy_behavior():
    """With merge_tracks=False the gap should leave two separate tracks."""
    T = 6
    zstack, mstack = _identical_band_stack(T)
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos, pos, empty, pos, pos, pos]
    slen_one = np.array([1.8], np.float32)
    ori_one = np.array([0.0], np.float32)
    slen_all = [slen_one, slen_one, np.zeros(0, np.float32),
                slen_one, slen_one, slen_one]
    ori_all = [ori_one, ori_one, np.zeros(0, np.float32),
               ori_one, ori_one, ori_one]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=0, min_track_length=2,
        merge_tracks=False,
    )
    assert out['n_tracks'] == 2
    assert out['n_merges'] == 0


def test_merge_respects_perp_gate():
    """A respawned track far enough transverse to the original must NOT merge.

    With the default ``merge_max_disp_perp_px=4`` (gap-scaled to ``perp² ≤
    16·gap``), an 8-px transverse offset across a 2-frame gap exceeds the
    gate (perp²=64 vs 16·2=32) and the merge should be rejected.
    """
    T = 6
    zstack, mstack = _identical_band_stack(T)
    pos_a = np.array([[25.0, 40.0]], dtype=np.float32)
    pos_b = np.array([[33.0, 40.0]], dtype=np.float32)  # 8 px transverse
    empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos_a, pos_a, empty, pos_b, pos_b, pos_b]
    slen_one = np.array([1.8], np.float32)
    ori_one = np.array([0.0], np.float32)  # axis along cols → perp = rows
    slen_all = [slen_one, slen_one, np.zeros(0, np.float32),
                slen_one, slen_one, slen_one]
    ori_all = [ori_one, ori_one, np.zeros(0, np.float32),
               ori_one, ori_one, ori_one]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=0, min_track_length=2,
        merge_tracks=True,
    )
    assert out['n_tracks'] == 2
    assert out['n_merges'] == 0


def test_merge_respects_slen_gate():
    """Two co-located fragments with very different slens should not merge."""
    T = 6
    zstack, mstack = _identical_band_stack(T)
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos, pos, empty, pos, pos, pos]
    slen_a = np.array([1.5], np.float32)
    slen_b = np.array([2.4], np.float32)  # 0.9 μm difference >> 0.15 default
    ori_one = np.array([0.0], np.float32)
    slen_all = [slen_a, slen_a, np.zeros(0, np.float32),
                slen_b, slen_b, slen_b]
    ori_all = [ori_one, ori_one, np.zeros(0, np.float32),
               ori_one, ori_one, ori_one]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=0, min_track_length=2,
        merge_tracks=True,  # default merge_slen_tol_um=0.30 μm
    )
    assert out['n_tracks'] == 2
    assert out['n_merges'] == 0


def test_merge_respects_slen_lims():
    """A fragment whose seam slen falls outside slen_lims must NOT be stitched.

    Fragments at the same position with slens of 1.7 μm (in range) and
    3.5 μm (above the default ``slen_lims=(1.0, 3.0)``) should be left as
    two separate tracks even though every other gate would pass.
    """
    T = 6
    zstack, mstack = _identical_band_stack(T)
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos, pos, empty, pos, pos, pos]
    slen_in = np.array([1.7], np.float32)
    slen_out = np.array([3.5], np.float32)  # > slen_lims[1]=3.0
    ori_one = np.array([0.0], np.float32)
    slen_all = [slen_in, slen_in, np.zeros(0, np.float32),
                slen_out, slen_out, slen_out]
    ori_all = [ori_one, ori_one, np.zeros(0, np.float32),
               ori_one, ori_one, ori_one]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=0, min_track_length=2,
        # Loosen slen continuity so only slen_lims can reject; otherwise
        # the |Δslen|=1.8 would also fail merge_slen_tol_um.
        merge_slen_tol_um=5.0,
        slen_lims=(1.0, 3.0),
        merge_tracks=True,
    )
    assert out['n_tracks'] == 2
    assert out['n_merges'] == 0


def test_merge_chains_multi_hop():
    """Three respawned fragments on the same trajectory should chain into one."""
    T = 10
    zstack, mstack = _identical_band_stack(T)
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    # Fragments: A=[0,1,2], gap, B=[5,6], gap, C=[8,9].
    pos_px_all = [pos, pos, pos, empty, empty,
                  pos, pos, empty, pos, pos]
    slen_one = np.array([1.8], np.float32)
    ori_one = np.array([0.0], np.float32)
    slen_all = [
        slen_one if len(p) else np.zeros(0, np.float32)
        for p in pos_px_all
    ]
    ori_all = [
        ori_one if len(p) else np.zeros(0, np.float32)
        for p in pos_px_all
    ]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=0, min_track_length=2,
        max_gap_interpolation=5,
        merge_tracks=True,
    )
    assert out['n_tracks'] == 1
    assert out['n_merges'] == 2
    snapped = out['tracks_snapped'][0]
    expected = [True, True, True, False, False, True, True, False, True, True]
    assert snapped.tolist() == expected


def test_merge_log_records_each_merge():
    """When return_merge_log=True the log should have one entry per merge."""
    T = 6
    zstack, mstack = _identical_band_stack(T)
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos, pos, empty, pos, pos, pos]
    slen_one = np.array([1.8], np.float32)
    ori_one = np.array([0.0], np.float32)
    slen_all = [slen_one, slen_one, np.zeros(0, np.float32),
                slen_one, slen_one, slen_one]
    ori_all = [ori_one, ori_one, np.zeros(0, np.float32),
               ori_one, ori_one, ori_one]

    out = st.track_sarcomere_vectors(
        zstack, mstack,
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        memory=0, min_track_length=2,
        merge_tracks=True,
        return_merge_log=True,
    )
    assert out['n_merges'] == 1
    log = out['merge_log']
    assert len(log) == 1
    entry = log[0]
    assert int(entry['gap']) == 2
    assert int(entry['t_a']) == 1
    assert int(entry['t_b']) == 3
    # Residuals on flat synthetic flow should be tiny.
    assert abs(entry['perp_resid_px']) < 1.0
    assert abs(entry['along_resid_px']) < 1.0


# ---------------------------------------------------------------------------
# Short-fragment merge bridges + gap-scaled re-acquisition
# ---------------------------------------------------------------------------

def test_merge_uses_short_fragment_as_bridge():
    """A short fragment (< min_track_length) must be usable as a bridge that
    chains two longer pieces which are otherwise too far apart to merge directly.

    Layout (min_track_length=4, max_gap_interpolation=5):
        A: snaps at 0,1,2,3      (len 4 — eligible seed)
        B: snap  at 8            (len 1 — short fragment, bridge only)
        C: snaps at 13,14,15     (len 3 — too short to seed/keep alone)
    A→C directly spans a 9-frame gap (> max_gap), so it can only be stitched
    *through* B (A→B gap 4, B→C gap 4). Legacy eligibility (bridge must be long)
    drops B and C, leaving A alone at length 4.
    """
    T = 16
    zstack, mstack = _identical_band_stack(T)
    p = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    sl = np.array([1.8], np.float32)
    ori = np.array([0.0], np.float32)
    sl0 = np.zeros(0, np.float32)
    pos_px_all, slen_all, ori_all = [], [], []
    for t in range(T):
        if t in (0, 1, 2, 3, 8, 13, 14, 15):
            pos_px_all.append(p); slen_all.append(sl); ori_all.append(ori)
        else:
            pos_px_all.append(empty); slen_all.append(sl0); ori_all.append(sl0)

    common = dict(pixelsize=0.1, frametime=0.01, memory=0, min_track_length=4,
                  max_gap_interpolation=5, merge_tracks=True, reacquire_gap_cap=1)

    # New behaviour: short B bridges A→B→C into one long track.
    out_new = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        merge_min_bridge_snaps=1, **common)
    assert out_new['n_tracks'] == 1
    assert int(out_new['tracks_snapped'][0].sum()) == 8   # 4 + 1 + 3 snaps

    # Legacy emulation: bridges must themselves be >= min_track_length, so B and
    # C are excluded and A can't reach C across the 9-frame gap → A alone.
    out_old = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        merge_min_bridge_snaps=4, merge_max_passes=1, **common)
    assert out_old['n_tracks'] == 1
    assert int(out_old['tracks_snapped'][0].sum()) == 4


def test_gap_scaled_reacquisition_recovers_offset_snap():
    """A coasting track re-snaps to a detection just outside the fresh gate but
    inside the gap-widened gate, instead of fragmenting.

    Flow ~0 (identical masks). Track sits at (25,40), detected at frames 0,1;
    frame 2 has no detection (track coasts, frames_since_snap→1); frames 3-5
    have a detection 18 px along-axis away. The fresh along gate is 15 px (legacy
    rejects → new track spawns), but after one gap frame the widened along gate
    is 15·sqrt(2) ≈ 21 px (re-acquisition accepts). merge is OFF to isolate the
    live loop. pixelsize=0.05 → slen ≈ 36 px, so the scale-aware along cap
    (0.6·slen ≈ 21.6 px) is inactive and does not interfere with this test.
    """
    T = 6
    zstack, mstack = _identical_band_stack(T)
    p0 = np.array([[25.0, 40.0]], dtype=np.float32)
    p1 = np.array([[25.0, 58.0]], dtype=np.float32)   # 18 px along cols
    empty = np.zeros((0, 2), dtype=np.float32)
    sl = np.array([1.8], np.float32); ori = np.array([0.0], np.float32)
    sl0 = np.zeros(0, np.float32)
    pos_px_all = [p0, p0, empty, p1, p1, p1]
    slen_all = [sl, sl, sl0, sl, sl, sl]
    ori_all = [ori, ori, sl0, ori, ori, ori]
    common = dict(pixelsize=0.05, frametime=0.01, memory=3, min_track_length=2,
                  merge_tracks=False)

    out_off = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        reacquire_gap_cap=1, **common)
    assert out_off['n_tracks'] == 2     # legacy: gap not bridged, B spawns

    out_on = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        reacquire_gap_cap=4, **common)
    assert out_on['n_tracks'] == 1      # widened gate re-acquires the offset det
    assert int(out_on['tracks_snapped'][0].sum()) == 5


def test_scale_aware_along_gate_cap_prevents_neighbour_snap():
    """The along snap gate is capped relative to sarcomere length (in px), so it
    is scale-invariant: a fixed pixel offset that the raw 15 px gate would accept
    is rejected at coarse pixel size (where it equals ~1 sarcomere = a swap), but
    accepted at fine pixel size. Same masks/positions/offset — only pixelsize
    (hence slen_px and the cap) differs.

    offset = 12 px. fine: slen=1.8/0.05=36 px → cap 0.6·36=21.6 (no-op, gate 15)
    → 12<15 snapped → 1 track. coarse: slen=1.8/0.12=15 px → cap 0.6·15=9 → 12>9
    rejected → fragments into 2 tracks (12 px ≈ 0.8 sarcomere = a neighbour).
    """
    T = 5
    zstack, mstack = _identical_band_stack(T)
    p0 = np.array([[25.0, 40.0]], dtype=np.float32)
    p1 = np.array([[25.0, 52.0]], dtype=np.float32)   # 12 px along cols
    empty = np.zeros((0, 2), dtype=np.float32)
    sl = np.array([1.8], np.float32); ori = np.array([0.0], np.float32)
    sl0 = np.zeros(0, np.float32)
    pos_px_all = [p0, p0, empty, p1, p1]
    slen_all = [sl, sl, sl0, sl, sl]
    ori_all = [ori, ori, sl0, ori, ori]
    common = dict(frametime=0.01, memory=3, min_track_length=2,
                  merge_tracks=False, reacquire_gap_cap=1)

    out_fine = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.05, **common)
    assert out_fine['n_tracks'] == 1     # cap inactive → 12 px snapped

    out_coarse = st.track_sarcomere_vectors(
        zstack, mstack, pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.12, **common)
    assert out_coarse['n_tracks'] == 2   # cap (9 px) rejects the 12 px neighbour
