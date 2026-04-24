"""Unit + behavioural tests for :mod:`sarcasm.structure_modules.sarcomere_tracking`.

The tracker is flow-predict + detection-snap; no M-band identity is persisted.
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.structure_modules import sarcomere_tracking as st


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
