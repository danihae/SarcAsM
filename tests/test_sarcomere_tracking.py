"""Unit + behavioural tests for :mod:`sarcasm.analysis.sarcomere_tracking`.

The tracker consumes only per-frame sarcomere vectors (position, length,
orientation) — no image data — and persists no M-band identity.
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.analysis import sarcomere_tracking as st


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

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
# End-to-end tracker behaviour
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


def test_seed_and_match_recovers_uniform_translation():
    """Static detections must simply stay on the same detections."""
    T = 3
    pos_px_all, slen_all, ori_all = _make_detection_sequence(
        T=T, n_points=10, shift_per_frame=0.0, orientation=np.pi / 2,
    )
    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        min_track_duration_s=0.02,
    )
    # All 10 detections should produce persistent tracks.
    assert out['motion.tracks.n'] == 10
    # Every track should be continuously observed across all frames.
    observed = out['motion.tracks.observed']
    assert np.all(observed == True)  # noqa: E712 — explicit test of boolean array


def test_anti_convergence_tracks_stay_separated():
    """Two query points 40 px apart must stay separated: each detection is
    matched at most once, so they cannot collapse onto each other."""
    T = 5

    # Two detections, 40 px apart, static across time.
    pos = np.array([[30.0, 50.0], [70.0, 50.0]], dtype=np.float32)
    pos_px_all = [pos.copy() for _ in range(T)]
    slen = np.array([1.8, 1.8], dtype=np.float32)
    ori = np.array([np.pi / 2, np.pi / 2], dtype=np.float32)
    slen_all = [slen.copy() for _ in range(T)]
    ori_all = [ori.copy() for _ in range(T)]

    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01,
        min_track_duration_s=0.02,
    )
    # 2 tracks, 40 px apart in every frame.
    assert out['motion.tracks.n'] == 2
    p = out['motion.tracks.positions_px']  # (2, T, 2)
    for t in range(T):
        d = np.linalg.norm(p[0, t] - p[1, t])
        assert d >= 30.0, f"tracks collapsed at frame {t}: distance = {d:.1f}"


def test_gap_frame_is_not_observed_and_carries_no_measured_slen():
    """If a detection disappears for a frame, the query point stays alive at its
    predicted position, and that frame is marked as not observed — it carries no
    *measured* sarcomere length. With gap interpolation disabled its slen is NaN;
    the interpolated value that the default fills in is a convenience for
    continuous traces, and ``tracks_observed`` remains the record of what is real."""
    T = 4

    pos_full = np.array([[25.0, 40.0]], dtype=np.float32)
    pos_empty = np.zeros((0, 2), dtype=np.float32)
    pos_px_all = [pos_full, pos_empty, pos_full, pos_full]  # missing at t=1
    slen_all = [np.array([1.8], np.float32), np.zeros(0, np.float32),
                np.array([1.8], np.float32), np.array([1.8], np.float32)]
    ori_all = [np.array([np.pi / 2], np.float32), np.zeros(0, np.float32),
               np.array([np.pi / 2], np.float32), np.array([np.pi / 2], np.float32)]

    common = dict(pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02)
    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        max_gap_interpolation_s=0, **common)
    assert out['motion.tracks.n'] >= 1
    slens = out['motion.tracks.slen']
    observed = out['motion.tracks.observed']
    # At least one track has a NaN slen at frame 1 but valid slen at frames 0/2/3,
    # and frame 1 is not counted as an observation.
    assert any(
        np.isnan(slens[i, 1]) and np.isfinite(slens[i, 0]) and np.isfinite(slens[i, 2])
        and not observed[i, 1]
        for i in range(slens.shape[0])
    )
    # The position is kept (predicted), not blanked, since the gap is interior.
    assert np.all(np.isfinite(out['motion.tracks.positions_px'][0, 1]))


def test_frames_after_last_observation_are_nan():
    """A track that permanently loses its detection must not freeze in place at
    its last position: every frame after the last observation is blanked to NaN —
    position, slen and orientation alike."""
    T = 8
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    # Detection present frames 0-3, then gone for good (track is lost).
    pos_px_all = [pos, pos, pos, pos, empty, empty, empty, empty]
    slen_one = np.array([1.8], np.float32)
    ori_one = np.array([0.0], np.float32)
    slen_all = [slen_one if len(p) else np.zeros(0, np.float32) for p in pos_px_all]
    ori_all = [ori_one if len(p) else np.zeros(0, np.float32) for p in pos_px_all]

    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02,
    )
    assert out['motion.tracks.n'] == 1
    observed = out['motion.tracks.observed'][0]
    assert observed.tolist() == [True, True, True, True, False, False, False, False]
    pos_out = out['motion.tracks.positions_px'][0]
    pos_um = out['motion.tracks.positions_um'][0]
    slens = out['motion.tracks.slen'][0]
    oris = out['motion.tracks.orientations'][0]
    # Observed frames keep their real values...
    np.testing.assert_allclose(pos_out[:4, 0], 25.0, atol=1.0)
    np.testing.assert_allclose(pos_out[:4, 1], 40.0, atol=1.0)
    assert np.all(np.isfinite(slens[:4]))
    # ...and every frame after the last observation is NaN (no constant hold-over).
    assert np.all(np.isnan(pos_out[4:]))
    assert np.all(np.isnan(pos_um[4:]))
    assert np.all(np.isnan(slens[4:]))
    assert np.all(np.isnan(oris[4:]))


def test_optimal_assignment_handles_a_shifted_1px_row():
    """The decisive case for the assignment.

    Sarcomere vectors are a ~1 px sampling along each M-band, so a whole ordered
    row of them shifts together while its ends gain and lose samples. Here a row
    of 40 detections at 1 px lateral spacing moves 1 px along the sarcomere axis
    between frames; one sample is lost at one end and one gained at the other.

    Every one of the 39 surviving samples must keep its identity. A greedy claim
    ordered by raw Euclidean distance cannot guarantee this — it orphans a track
    at the end of the row, which then dies and respawns as a duplicate.
    """
    T = 4
    n = 40
    lateral = np.arange(n, dtype=np.float32)      # 1 px apart along the row (rows)
    pos_px_all, slen_all, ori_all = [], [], []
    for t in range(T):
        # the row translates 1 px per frame along the sarcomere axis (cols);
        # drop the first sample and add one past the far end each frame
        ys = lateral[1:] + 0.0
        xs = np.full(n - 1, 40.0 + t, np.float32)
        pos_px_all.append(np.stack([ys, xs], axis=1).astype(np.float32))
        slen_all.append(np.full(n - 1, 1.8, np.float32))
        ori_all.append(np.zeros(n - 1, np.float32))   # axis along cols
    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.06, frametime=0.01, min_track_duration_s=0.02)
    observed = out['motion.tracks.observed']
    # One track per surviving sample, each observed in every frame: no duplicates.
    assert out['motion.tracks.n'] == n - 1, f"expected {n - 1} tracks, got {out['motion.tracks.n']}"
    assert observed.all(), 'every sample must stay matched through the shift'
    # And each track stayed on its own lateral row (no swap with a neighbour).
    rows = out['motion.tracks.positions_px'][:, :, 0]
    assert np.allclose(rows, rows[:, :1]), 'a track changed lateral position'


def test_track_survives_a_gap_far_longer_than_the_old_memory_horizon():
    """Identity now survives a dropout of any length, with no merge pass: the
    unmatched track keeps its anchor and re-acquires the same detection. The gap
    frames stay honest — observed False and slen NaN, never a fabricated length."""
    T = 40
    pos = np.array([[25.0, 40.0]], np.float32)
    empty = np.zeros((0, 2), np.float32)
    sl, ori = np.array([1.8], np.float32), np.array([0.0], np.float32)
    sl0 = np.zeros(0, np.float32)
    gap = slice(5, 35)                       # 30 frames absent (0.3 s at 0.01 s)
    pos_px_all = [pos.copy() for _ in range(T)]
    slen_all = [sl.copy() for _ in range(T)]
    ori_all = [ori.copy() for _ in range(T)]
    for t in range(gap.start, gap.stop):
        pos_px_all[t] = empty; slen_all[t] = sl0; ori_all[t] = sl0

    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02)
    assert out['motion.tracks.n'] == 1, 'the 30-frame gap must not split the trajectory'
    observed = out['motion.tracks.observed'][0]
    assert observed[:5].all() and observed[35:].all()
    assert not observed[gap].any()
    # gap frames carry no fabricated length
    assert np.all(np.isnan(out['motion.tracks.slen'][0][gap]))


def test_a_gap_never_widens_the_match_gate():
    """Waiting longer does not license a longer jump: a detection outside the
    single-frame gate is never claimed, however long the track has been unmatched.
    (The previous design widened the gate with the gap, which measurably traded
    identity for continuity.)"""
    T = 12
    p0 = np.array([[25.0, 40.0]], np.float32)
    # reappears 14 px along cols — outside the 1 µm gate at pixelsize 0.1 (10 px)
    p1 = np.array([[25.0, 54.0]], np.float32)
    empty = np.zeros((0, 2), np.float32)
    sl, ori = np.array([1.8], np.float32), np.array([0.0], np.float32)
    sl0 = np.zeros(0, np.float32)
    pos_px_all = [p0, p0] + [empty] * 8 + [p1, p1]
    slen_all = [sl, sl] + [sl0] * 8 + [sl, sl]
    ori_all = [ori, ori] + [sl0] * 8 + [ori, ori]
    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02)
    # two separate tracks: the far reappearance is a different sarcomere
    assert out['motion.tracks.n'] == 2


def test_unmatched_track_advection_is_along_the_axis_only():
    """An unmatched track is carried by its neighbourhood, but only along its own
    sarcomere axis: perpendicular motion can come solely from a match residual,
    which the perpendicular gate hard-caps. Here the neighbours move purely
    perpendicular to the axis, so the unmatched track must not follow them."""
    T = 6
    # neighbours: a block of detections moving +2 px per frame in rows (= perp,
    # since orientation 0 puts the sarcomere axis along cols)
    nb0 = np.stack([np.full(8, 60.0), np.linspace(30, 100, 8)], axis=1)
    target = np.array([[100.0, 65.0]], np.float32)   # present only at t=0..1
    pos_px_all, slen_all, ori_all = [], [], []
    for t in range(T):
        nb = nb0 + np.array([2.0 * t, 0.0])
        pts = np.vstack([nb, target]) if t < 2 else nb
        pos_px_all.append(pts.astype(np.float32))
        slen_all.append(np.full(len(pts), 1.8, np.float32))
        ori_all.append(np.zeros(len(pts), np.float32))
    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02)
    # find the track that owned the target and check it never moved in rows
    pos_out = out['motion.tracks.positions_px']
    which = [i for i in range(out['motion.tracks.n'])
             if np.isfinite(pos_out[i, 0]).all() and abs(pos_out[i, 0, 0] - 100.0) < 1.0]
    assert which, 'the target detection was not tracked'
    row = pos_out[which[0], :, 0]
    finite = row[np.isfinite(row)]
    assert np.allclose(finite, 100.0, atol=0.5), (
        f'unmatched track drifted perpendicular to its axis: {finite}')


def test_scale_aware_along_gate_cap_prevents_neighbour_match():
    """The along match gate is a fixed physical distance (max_disp_along_um), so
    tracking is pixel-size invariant: the SAME 12 px offset is accepted at fine
    pixel size (where 12 px is well under 1 µm) and rejected at coarse pixel size
    (where 12 px exceeds the 1 µm gate and would reach a neighbour). Same
    masks/positions/offset — only pixelsize (hence the gate in px) differs.

    offset = 12 px. fine (pixelsize 0.05): gate 1.0 µm = 20 px -> 12<20 matched ->
    1 track. coarse (pixelsize 0.12): gate 1.0 µm = 8.3 px -> 12>8.3 rejected ->
    fragments into 2 tracks (12 px ~ 1.4 µm = a neighbouring sarcomere).
    """
    T = 5
    p0 = np.array([[25.0, 40.0]], dtype=np.float32)
    p1 = np.array([[25.0, 52.0]], dtype=np.float32)   # 12 px along cols
    empty = np.zeros((0, 2), dtype=np.float32)
    sl = np.array([1.8], np.float32); ori = np.array([0.0], np.float32)
    sl0 = np.zeros(0, np.float32)
    pos_px_all = [p0, p0, empty, p1, p1]
    slen_all = [sl, sl, sl0, sl, sl]
    ori_all = [ori, ori, sl0, ori, ori]
    common = dict(frametime=0.01, min_track_duration_s=0.02)

    out_fine = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.05, **common)
    assert out_fine['motion.tracks.n'] == 1     # cap inactive -> 12 px matched

    out_coarse = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.12, **common)
    assert out_coarse['motion.tracks.n'] == 2   # cap (9 px) rejects the 12 px neighbour


def test_short_interior_gaps_are_interpolated_but_not_marked_observed():
    """``max_gap_interpolation_s`` fills slen/orientation across brief flicker so the
    per-track traces have no holes — but the filled frames must stay False in
    ``tracks_observed``, so coverage and every real-observation metric still count
    only genuine detections. Gaps longer than the limit stay NaN, and nothing is
    ever extrapolated past the last observation."""
    T = 12
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    sl_a, sl_b = np.array([1.6], np.float32), np.array([2.0], np.float32)
    ori = np.array([0.0], np.float32)
    sl0 = np.zeros(0, np.float32)
    # obs 0,1 -> 2-frame gap -> obs 4 -> 5-frame gap -> obs 10, 11
    present = [True, True, False, False, True, False, False, False, False, False, True, True]
    pos_px_all = [pos.copy() if p else empty for p in present]
    slen_all = [(sl_a if t < 5 else sl_b) if present[t] else sl0 for t in range(T)]
    ori_all = [ori.copy() if p else sl0 for p in present]

    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02,
        max_gap_interpolation_s=0.03)                    # 3 frames at 100 fps
    assert out['motion.tracks.n'] == 1
    observed = out['motion.tracks.observed'][0]
    slen = out['motion.tracks.slen'][0]

    assert observed.tolist() == present, 'interpolation must not fabricate observations'
    # the 2-frame gap is filled, and monotonically between its anchors
    assert np.all(np.isfinite(slen[2:4]))
    assert slen[1] <= slen[2] <= slen[3] <= slen[4]
    # the 5-frame gap exceeds the limit and stays NaN
    assert np.all(np.isnan(slen[5:10]))
    assert out['motion.tracks.n_interpolated_gap_frames'] == 2

    # disabling it leaves every gap frame NaN
    off = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02,
        max_gap_interpolation_s=0)
    assert np.all(np.isnan(off['motion.tracks.slen'][0][~off['motion.tracks.observed'][0]]))
    assert off['motion.tracks.n_interpolated_gap_frames'] == 0


def test_interpolated_orientation_is_axial():
    """Orientations are undirected axes, so they must be interpolated in the
    double-angle representation: between 0.1 rad and π-0.1 rad the midpoint is
    ~0 (the short way round the axis), not ~π/2."""
    T = 3
    pos = np.array([[25.0, 40.0]], dtype=np.float32)
    empty = np.zeros((0, 2), dtype=np.float32)
    sl = np.array([1.8], np.float32)
    sl0 = np.zeros(0, np.float32)
    pos_px_all = [pos, empty, pos]
    slen_all = [sl, sl0, sl]
    ori_all = [np.array([0.1], np.float32), sl0, np.array([np.pi - 0.1], np.float32)]
    out = st.track_sarcomere_vectors(
        pos_px_all, [None] * T, slen_all, ori_all,
        pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02,
        max_gap_interpolation_s=0.03)
    mid = float(out['motion.tracks.orientations'][0][1])
    assert abs(st._angular_diff(mid, 0.0)) < 0.05, mid
