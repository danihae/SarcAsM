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


# ---------------------------------------------------------------------------
# Motion predictor hook (image optical flow at coarse frame rates)
# ---------------------------------------------------------------------------

def _beyond_gate_sequence(T=6, n_points=20, step=14.0):
    """Along-axis translation of ``step`` px per frame — beyond the 10 px along gate
    at pixelsize 0.1 (min(1.0 µm, 0.6·slen)), so hold-position prediction cannot follow."""
    return _make_detection_sequence(T=T, n_points=n_points, shift_per_frame=step, orientation=np.pi / 2)


def test_without_predictor_a_step_beyond_the_gate_breaks_every_track():
    T = 6
    pos, slen, ori = _beyond_gate_sequence(T=T)
    out = st.track_sarcomere_vectors(pos, [None] * T, slen, ori, pixelsize=0.1, frametime=0.2,
                                     min_track_duration_s=0.2)
    observed = out['motion.tracks.observed']
    assert not observed.all(axis=1).any()          # no track survives the whole movie
    assert out['motion.tracks.n'] > 20             # identities fragment into new tracks


def test_exact_predictor_carries_every_track_through_the_step():
    T = 6
    pos, slen, ori = _beyond_gate_sequence(T=T)
    exact = lambda t, yx: np.tile(np.array([14.0, 0.0], np.float32), (len(yx), 1))
    out = st.track_sarcomere_vectors(pos, [None] * T, slen, ori, pixelsize=0.1, frametime=0.2,
                                     min_track_duration_s=0.2, predictor=exact)
    assert out['motion.tracks.n'] == 20
    assert out['motion.tracks.observed'].all()
    # positions are the matched detections, not the predictions
    assert np.allclose(out['motion.tracks.positions_px'][:, -1], pos[-1])


def test_predictor_perpendicular_component_is_ignored_when_small():
    """The flow's lateral component is noisy on striations: below half the perpendicular
    gate it is dropped, so a small spurious lateral prediction does not push a query
    point off its own detection."""
    T = 6
    pos, slen, ori = _beyond_gate_sequence(T=T)
    noisy = lambda t, yx: np.tile(np.array([14.0, 0.8], np.float32), (len(yx), 1))   # 0.8 px < 0.5 * 2 px perp gate
    out = st.track_sarcomere_vectors(pos, [None] * T, slen, ori, pixelsize=0.1, frametime=0.2,
                                     min_track_duration_s=0.2, predictor=noisy)
    assert out['motion.tracks.n'] == 20 and out['motion.tracks.observed'].all()


def test_predictor_with_wrong_length_raises():
    T = 3
    pos, slen, ori = _make_detection_sequence(T=T, n_points=10, shift_per_frame=0.0)
    bad = lambda t, yx: np.zeros((len(yx) - 1, 2), np.float32)
    with pytest.raises(ValueError, match='predictor returned'):
        st.track_sarcomere_vectors(pos, [None] * T, slen, ori, pixelsize=0.1, frametime=0.01,
                                   min_track_duration_s=0.02, predictor=bad)


def test_predictor_none_is_the_default_path():
    T = 6
    pos, slen, ori = _make_detection_sequence(T=T, n_points=15, shift_per_frame=1.0, orientation=np.pi / 2)
    a = st.track_sarcomere_vectors(pos, [None] * T, slen, ori, pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02)
    b = st.track_sarcomere_vectors(pos, [None] * T, slen, ori, pixelsize=0.1, frametime=0.01, min_track_duration_s=0.02,
                                   predictor=None)
    for key in ('motion.tracks.positions_px', 'motion.tracks.slen', 'motion.tracks.observed', 'motion.tracks.detection_id'):
        assert np.array_equal(a[key], b[key], equal_nan=True)


# ---------------------------------------------------------------------------
# Length refinement along tracks (refine_lengths)
# ---------------------------------------------------------------------------

def test_track_axis_orientations_is_axial_constant_and_ignores_gaps():
    # alternating just above 0 and just below π: the same axis, so the median is ~0 (or ~π), never π/2
    T = 40
    ori = np.where(np.arange(T) % 2 == 0, 0.03, np.pi - 0.03)[None, :].astype(float)
    observed = np.ones_like(ori, dtype=bool)
    observed[0, 10:13] = False
    ori[0, 10:13] = 1.2                      # values on unobserved frames must not leak in
    out = st.track_axis_orientations(ori, observed)
    dev = np.abs((out[0, observed[0]] + np.pi / 2) % np.pi - np.pi / 2)
    assert np.nanmax(dev) < 0.05             # a wrap error would give ~π/2
    assert np.ptp(out[0, observed[0]]) == 0  # one axis per track
    assert np.all(np.isnan(out[0, 10:13]))   # gaps stay gaps
    # a window of 1 returns the observed input unchanged (wrapped to [0, π))
    np.testing.assert_allclose(st.track_axis_orientations(ori, observed, 1)[0, :10], ori[0, :10])


def test_track_axis_running_median_follows_slow_turn_but_not_contractions():
    # the tissue turns slowly by 0.2 rad; every beat of 20 frames tilts the Z-bands by 0.1 rad for 8 frames
    T = 400
    t = np.arange(T)
    slow = 0.2 * t / T
    ori = (slow + np.where(t % 20 < 8, 0.1, 0.0))[None, :]
    observed = np.ones_like(ori, dtype=bool)
    running = st.track_axis_orientations(ori, observed, window_frames=61)
    assert np.max(np.abs(running[0, 30:-30] - slow[30:-30])) < 0.02      # vs the 0.1-rad tilt of each beat
    assert np.ptp(st.track_axis_orientations(ori, observed)[0]) == 0


def _stripe_zbands(shape=(80, 200), period_px=30.0, sigma_px=2.5, x0=10.0, shear=0.0, row0=40.0):
    """Z-band map of straight Z-bands perpendicular to x (sarcomere axis along x, orientation 0). ``shear``
    shifts every row along x by ``shear * (row - row0)``, which tilts the Z-bands but keeps their spacing along x
    (the sarcomere axis), as when neighbouring myofibrils shorten by different amounts."""
    y, x = np.mgrid[:shape[0], :shape[1]].astype(float)
    d = np.mod(x - shear * (y - row0) - x0 + period_px / 2, period_px) - period_px / 2   # along x
    return np.exp(-0.5 * (d / np.sqrt(1 + shear ** 2) / sigma_px) ** 2).astype(np.float32)


def test_refine_track_lengths_recovers_spacing_from_jittered_orientation():
    pixelsize, period = 0.06, 30.0
    truth = period * pixelsize
    z = _stripe_zbands(period_px=period)
    n, T = 6, 60
    pos = np.zeros((n, T, 2))
    pos[:, :, 0] = (20.0 + 8.0 * np.arange(n))[:, None]
    pos[:, :, 1] = (85.0 + period * (np.arange(n) % 3))[:, None]          # on M-bands, between two Z-bands
    observed = np.ones((n, T), bool)
    rng = np.random.default_rng(1)
    jitter = np.deg2rad(rng.normal(0, 10, size=(n, T)))
    exact, _, _ = st.refine_track_lengths(pos, np.zeros((n, T)), observed, lambda k: z, pixelsize, window_frames=1)
    assert np.nanmax(np.abs(exact - truth)) < 0.005
    raw, _, _ = st.refine_track_lengths(pos, jitter, observed, lambda k: z, pixelsize, window_frames=1)
    axis, ori_a, _ = st.refine_track_lengths(pos, jitter, observed, lambda k: z, pixelsize)
    assert np.nanmean(np.abs(axis - truth)) < 0.5 * np.nanmean(np.abs(raw - truth))
    axial_dev = np.abs((ori_a + np.pi / 2) % np.pi - np.pi / 2)          # distance to the true axis 0, mod π
    assert np.nanmean(axial_dev) < 0.25 * np.nanmean(np.abs(jitter))


def test_refine_track_lengths_measures_along_the_axis_not_normal_to_sheared_zbands():
    # relaxed frames, then contraction frames in which a shear tilts the Z-bands by 10° (spacing along the
    # axis unchanged); the detection follows the Z-band normal. Normal to the tilted bands the spacing is
    # shorter by cos 10° (27 nm here); along the track's axis it is the true length.
    pixelsize, period, tilt = 0.06, 30.0, np.deg2rad(10.0)
    truth = period * pixelsize
    relaxed, sheared = _stripe_zbands(period_px=period), _stripe_zbands(period_px=period, shear=np.tan(tilt))
    n, T = 4, 30
    contraction = np.arange(T) >= 20
    pos = np.zeros((n, T, 2))
    pos[:, :, 0] = 40.0                                                    # on the row the shear pivots about
    pos[:, :, 1] = (85.0 + period * (np.arange(n) % 3))[:, None]           # on M-bands of that row
    detected = np.where(contraction, -tilt, 0.0)[None, :].repeat(n, 0)    # normal of the tilted Z-bands
    observed = np.ones((n, T), bool)
    read = lambda k: sheared if contraction[k] else relaxed                # noqa: E731
    normal, _, _ = st.refine_track_lengths(pos, detected, observed, read, pixelsize, window_frames=1)
    axis, _, _ = st.refine_track_lengths(pos, detected, observed, read, pixelsize)
    np.testing.assert_allclose(normal[:, contraction], truth * np.cos(tilt), atol=0.006)
    np.testing.assert_allclose(axis, truth, atol=0.005)


def test_refine_track_lengths_moves_positions_to_the_zband_midpoint():
    # the tracker follows integer M-band pixels; a track sitting 1-2 px off the midpoint between its Z-bands
    # (x0 + period/2 + multiples of the period) is moved onto it, its length is the spacing either way
    pixelsize, period, x0 = 0.06, 30.0, 10.0
    z = _stripe_zbands(period_px=period, x0=x0)
    n, T = 5, 4
    mid = x0 + period / 2 + period * 2                                       # 85: midpoint between two Z-bands
    off = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    pos = np.zeros((n, T, 2))
    pos[:, :, 0] = (20.0 + 8.0 * np.arange(n))[:, None]
    pos[:, :, 1] = (mid + off)[:, None]
    observed = np.ones((n, T), bool)
    observed[0, -1] = False
    pos[0, -1] = np.nan
    slen, _, moved = st.refine_track_lengths(pos, np.zeros((n, T)), observed, lambda k: z, pixelsize, window_frames=1)
    np.testing.assert_allclose(slen[observed], period * pixelsize, atol=0.005)
    cols = moved[observed][:, 1]
    assert np.ptp(cols) < 0.05                                               # the +-2 px quantisation is gone
    np.testing.assert_allclose(cols, mid, atol=0.1)
    np.testing.assert_allclose(moved[observed][:, 0], pos[observed][:, 0])    # rows (across the axis): untouched
    assert np.all(np.isnan(moved[0, -1]))                                    # unobserved frame left as given


def test_orientation_window_warns_about_too_few_frames(caplog, monkeypatch):
    import logging
    import types
    from sarcasm import SarcAsM
    # caplog listens on the root logger, and SarcAsM.__init__ (run by any earlier store-backed test) turns the
    # package logger's propagation off
    monkeypatch.setattr(logging.getLogger('sarcasm'), 'propagate', True)
    sarc = SarcAsM.__new__(SarcAsM)
    # 5 fps: 0.1 s holds half a frame -> warning, one axis per track instead of a running median
    sarc.metadata = types.SimpleNamespace(frametime=0.2, pixelsize=0.1)
    with caplog.at_level(logging.WARNING):
        assert sarc._orientation_window_frames(0.1) is None
    assert 'holds 0.50 frames' in caplog.text and 'at least 5' in caplog.text
    assert sarc._orientation_window_frames(None) is None
    assert sarc._orientation_window_frames(3.0) == 15
    sarc.metadata = types.SimpleNamespace(frametime=None, pixelsize=0.1)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert sarc._orientation_window_frames(2.0) is None
    assert 'frametime is unknown' in caplog.text
