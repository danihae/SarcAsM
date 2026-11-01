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
    assert abs((mid + np.pi / 2) % np.pi - np.pi / 2) < 0.05, mid      # axial difference to 0


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
        assert sarc.motion._orientation_window_frames(0.1) is None
    assert 'holds 0.50 frames' in caplog.text and 'at least 5' in caplog.text
    assert sarc.motion._orientation_window_frames(None) is None
    assert sarc.motion._orientation_window_frames(3.0) == 15
    sarc.metadata = types.SimpleNamespace(frametime=None, pixelsize=0.1)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert sarc.motion._orientation_window_frames(2.0) is None
    assert 'frametime is unknown' in caplog.text


# ---------------------------------------------------------------------------
# Compiled kernels: parity with the scipy / numpy code they replaced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('integer_positions', [True, False])
def test_candidate_pairs_equal_the_kdtree_ball_query(integer_positions):
    """Same candidates, in the same order, as cKDTree.query_ball_point (sorted per query point)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(2)
    dets = rng.uniform(0, 120, (3000, 2))
    if integer_positions:                                     # M-band skeleton pixels, unique
        dets = np.unique(np.floor(dets), axis=0)
    dets = dets.astype(np.float32)
    query = (dets[rng.choice(len(dets), 800)] + rng.normal(0, 2, (800, 2))).astype(np.float32)
    radius = 16.3

    counts, det = st._candidate_pairs(dets, query, radius)

    ref = cKDTree(dets).query_ball_point(query, r=radius)
    np.testing.assert_array_equal(counts, [len(r) for r in ref])
    np.testing.assert_array_equal(det, np.concatenate([np.asarray(r, np.int64) for r in ref]))


def _assign_reference(cost, qp, det, n_qp, n_det, max_component):
    """The former _assign_optimal: scipy connected components, per-component np.unique matrices."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = n_qp + n_det
    adj = coo_matrix((np.ones(len(qp)), (qp, det + n_qp)), shape=(n, n))
    _, labels = connected_components(adj + adj.T, directed=False)
    order = np.argsort(labels[qp], kind='stable')
    qp, det, cost = qp[order], det[order], cost[order]
    pairs = set()
    for sl in np.split(np.arange(len(qp)), np.flatnonzero(np.diff(labels[qp])) + 1):
        q_ids, q_inv = np.unique(qp[sl], return_inverse=True)
        d_ids, d_inv = np.unique(det[sl], return_inverse=True)
        if max(len(q_ids), len(d_ids)) > max_component:
            claimed_q, claimed_d = set(), set()
            for i in np.argsort(cost[sl], kind='stable'):
                if q_inv[i] not in claimed_q and d_inv[i] not in claimed_d:
                    claimed_q.add(q_inv[i]); claimed_d.add(d_inv[i])
                    pairs.add((q_ids[q_inv[i]], d_ids[d_inv[i]]))
            continue
        sentinel = 2.0 * min(len(q_ids), len(d_ids)) + 1.0
        mat = np.full((len(q_ids), len(d_ids)), sentinel)
        np.minimum.at(mat, (q_inv, d_inv), cost[sl])
        rows, cols = linear_sum_assignment(mat)
        ok = mat[rows, cols] < sentinel
        pairs.update(zip(q_ids[rows[ok]], d_ids[cols[ok]]))
    return pairs


@pytest.mark.parametrize('max_component', [1500, 6])
def test_assignment_matches_the_scipy_component_reference(monkeypatch, max_component):
    """Identical matches to the per-component scipy implementation it replaced, ties and duplicate pairs
    included; max_component=6 sends the larger components through the greedy fallback."""
    monkeypatch.setattr(st, '_LAP_MAX_COMPONENT', max_component)
    rng = np.random.default_rng(4)
    n_qp, n_det = 400, 380
    qp = rng.integers(0, n_qp, 1500)
    det = np.clip(qp + rng.integers(-3, 4, 1500), 0, n_det - 1)            # row-like components
    cost = rng.choice([0.25, 0.5, 1.0, 1.5], 1500).astype(np.float32)       # many ties
    qp, det, cost = np.r_[qp, qp[:50]], np.r_[det, det[:50]], np.r_[cost, cost[:50] + 0.5]   # duplicate pairs

    q_acc, d_acc = st._assign_optimal(cost, qp, det, n_qp, n_det)

    assert set(zip(q_acc, d_acc)) == _assign_reference(cost, qp, det, n_qp, n_det, max_component)


def test_nan_skipping_median_equals_numpy_nanmedian():
    rng = np.random.default_rng(6)
    for k in (1, 4, 7, 16):
        values = rng.normal(size=(300, k, 2))
        values[rng.random(values.shape) < 0.4] = np.nan
        values[0] = np.nan                                                    # nothing in range
        with np.errstate(all='ignore'), pytest.warns(RuntimeWarning):
            ref = np.nanmedian(values, axis=1)
        np.testing.assert_array_equal(st._nan_skipping_median(values), ref)


def test_progress_bars_are_labelled_and_on_by_default(capsys):
    """track() reports progress in API use by default, as two labelled bars (tracking, length refinement)."""
    import inspect
    from bio_image_unet.progress import ProgressNotifier
    from sarcasm.motion.tracking import TrackingMixin
    assert inspect.signature(TrackingMixin.track).parameters['progress_notifier'].default is not None

    notifier = ProgressNotifier.progress_notifier_tqdm()
    pos_px_all, slen_all, ori_all = _make_detection_sequence(T=4, n_points=10)
    st.track_sarcomere_vectors(pos_px_all, [None] * 4, slen_all, ori_all, pixelsize=0.1, frametime=0.01,
                               min_track_duration_s=0.02, progress_notifier=notifier)
    pos = np.full((2, 4, 2), 40.0)
    st.refine_track_lengths(pos, np.zeros((2, 4)), np.ones((2, 4), bool), lambda k: _stripe_zbands(), 0.06,
                            progress_notifier=notifier)
    err = capsys.readouterr().err
    assert 'Tracking' in err and 'Refining lengths' in err


# ---------------------------------------------------------------------------
# Rest frames, rest position, gap filling, quality rules
# ---------------------------------------------------------------------------

def _beating_trace(T=400, period=100, contraction=60):
    """Length trace that contracts for most of each cycle, so its median lies between rest and peak."""
    phase = np.arange(T) % period
    return 1.9 - 0.3 * np.sin(np.pi * np.minimum(phase, contraction) / contraction)


def test_rest_frames_use_the_most_occupied_level():
    trace = _beating_trace()
    rest = st.rest_frames_from_trace(trace)
    plateau = (np.arange(400) % 100) >= 62
    assert abs(float(np.median(trace)) - 1.9) > 0.01      # the median misses the plateau
    assert rest[plateau].mean() > 0.9
    assert not rest[(np.arange(400) % 100 > 5) & (np.arange(400) % 100 < 55)].any()


@pytest.mark.parametrize('trace', [np.random.default_rng(0).normal(1.8, 0.01, 200), np.full(50, np.nan), np.full(50, 1.8),
                                   np.where(np.arange(50) % 2, np.arange(50) * 0.01, np.nan)])
def test_no_rest_frames_without_a_beat_above_the_noise(trace):
    assert not st.rest_frames_from_trace(trace).any()


@pytest.mark.parametrize('fps', [10, 25, 50, 100])
def test_rest_frames_do_not_depend_on_the_frame_rate(fps):
    t = np.arange(20 * fps) / fps
    phase = t % 1.0
    trace = 1.9 - 0.3 * np.sin(np.pi * np.minimum(phase, 0.6) / 0.6) + np.random.default_rng(0).normal(0, 0.005, t.size)
    rest = st.rest_frames_from_trace(trace)
    assert rest[phase >= 0.65].mean() > 0.8
    assert not rest[(phase > 0.15) & (phase < 0.45)].any()


def test_rest_frames_with_a_single_beat_in_a_long_movie():
    trace = 1.8 + np.random.default_rng(0).normal(0, 0.002, 6000)
    trace[3000:3030] -= 0.3 * np.sin(np.pi * np.arange(30) / 30)
    assert st.rest_frames_from_trace(trace).mean() > 0.7


def _hopping_beats(T=80, pitch=30.0, jump=20.0, relax=5.0, px=0.06):
    """M-bands one pitch apart along x. Each beat throws them by ``jump`` in one frame, nearer to the
    neighbour's position than to their own, and they relax in small steps: the bare tracker follows the neighbour."""
    columns = np.arange(40.0, 200.0, pitch)
    rows = np.arange(40.0, 60.0)
    base = np.array([(r, c) for c in columns for r in rows], dtype=np.float32)
    rng = np.random.default_rng(0)
    pos_all, slen_all, ori_all, mid_all = [], [], [], []
    for t in range(T):
        since = (t - 20) % 40
        shift = jump - relax * since if since * relax < jump else 0.0
        pos_all.append(np.round(base + np.array([0.0, shift], dtype=np.float32)))
        length = 1.8 - (0.25 if shift > 0 else 0.0) + rng.normal(0, 0.002)
        slen_all.append(np.full(len(base), length, dtype=np.float32))
        ori_all.append(np.zeros(len(base), dtype=np.float32))
        mid_all.append(np.zeros(len(base), dtype=np.int32))
    return pos_all, mid_all, slen_all, ori_all, px


def test_track_returns_to_its_sarcomere_after_a_beat():
    pos_all, mid_all, slen_all, ori_all, px = _hopping_beats()
    kwargs = dict(pixelsize=px, frametime=0.02, min_track_duration_s=0.0)
    anchored = st.track_sarcomere_vectors(pos_all, mid_all, slen_all, ori_all, **kwargs)
    bare = st.track_sarcomere_vectors(pos_all, mid_all, slen_all, ori_all, max_disp_along_rest_um=None, **kwargs)

    def moved(out):
        p, o = out['motion.tracks.positions_px'], out['motion.tracks.observed']
        both = o[:, 15] & o[:, 75]
        return np.abs(p[both, 75, 1] - p[both, 15, 1])
    rest = anchored['motion.tracks.rest_frames']
    assert rest[:19].all() and not rest[19:24].any()     # a frame next to a beat is not at rest
    assert moved(bare).max() >= 25                       # the bare tracker has changed sarcomere for good
    assert moved(anchored).max() <= 1


def _lattice(px, shift_px, T=80):
    """Six M-bands 1.8 um apart with a beat every 40 frames from frame 20; ``shift_px(t)`` moves all of them along the axis."""
    base = np.array([(r, 40 + c * 1.8 / px) for c in range(6) for r in np.arange(40.0, 60.0)], dtype=np.float32)
    rng = np.random.default_rng(0)
    lengths = [1.8 - (0.25 if (t - 20) % 40 < 5 else 0.0) + rng.normal(0, 0.002) for t in range(T)]
    return ([np.round(base + np.array([0.0, shift_px(t)], dtype=np.float32)) for t in range(T)],
            [np.zeros(len(base), dtype=np.int32) for _ in range(T)],
            [np.full(len(base), length, dtype=np.float32) for length in lengths],
            [np.zeros(len(base), dtype=np.float32) for _ in range(T)])


@pytest.mark.parametrize('px', [0.06, 0.12])
def test_rest_gate_limits_a_step_on_a_rest_frame(px):
    def thrown(jump_um, gate_um):
        pos, mid, slen, ori = _lattice(px, lambda t: jump_um / px if t == 50 else 0.0)
        out = st.track_sarcomere_vectors(pos, mid, slen, ori, pixelsize=px, frametime=0.02,
                                         max_disp_along_rest_um=gate_um, min_track_duration_s=0.0)
        seed = out['motion.tracks.start_frame'] == 0
        return out['motion.tracks.observed'][seed, 50], out['motion.tracks.positions_px'][seed]
    assert thrown(0.3, 0.5)[0].all()
    assert not thrown(0.7, 0.5)[0].any()
    observed, p = thrown(0.95, 5.0)                      # the half-sarcomere cap binds: the neighbour is 0.85 um away
    assert (np.abs(p[observed, 50, 1] - p[observed, 49, 1]) * px <= 0.9).all()


def test_stored_rest_frames_are_not_the_contracted_end():
    px, T = 0.06, 400
    phase = np.arange(T) % 100
    rng = np.random.default_rng(0)

    def stored_rest(trace):
        pos, mid, _, ori = _lattice(px, lambda t: 0.0, T=T)
        slen = [np.full(len(pos[0]), trace[t] + rng.normal(0, 0.002), dtype=np.float32) for t in range(T)]
        return st.track_sarcomere_vectors(pos, mid, slen, ori, pixelsize=px, frametime=0.02,
                                          min_track_duration_s=0.0)['motion.tracks.rest_frames']
    # 70 frames contracted, 20 relaxed: the most occupied level is the contracted one
    assert not stored_rest(1.8 - 0.3 * np.clip(np.minimum(phase, 80 - phase) / 5, 0, 1)).any()
    # readings below rest, then above it, as on low-count movies: the level is rest
    rest = stored_rest(1.8 - 0.1 * (phase < 12) + 0.25 * ((phase >= 12) & (phase < 21)))
    assert rest[phase >= 25].mean() > 0.9 and not rest[phase < 21].any()


def test_track_keeps_following_tissue_that_does_not_return():
    px = 0.06
    pos, mid, slen, ori = _lattice(px, lambda t: 22.0 * np.clip((t - 20) / 4, 0, 1))       # 1.3 um during the first beat, for good
    out = st.track_sarcomere_vectors(pos, mid, slen, ori, pixelsize=px, frametime=0.02, min_track_duration_s=0.0)
    p, o = out['motion.tracks.positions_px'], out['motion.tracks.observed']
    assert out['motion.tracks.n'] == len(pos[0])
    assert o.mean() > 0.99
    np.testing.assert_allclose(p[:, 75, 1] - p[:, 15, 1], 22.0, atol=1.0)


def _pair_profile(heights=(1.0, 1.0), third=None, length=120, px=0.06, spacing=30):
    x = np.arange(length, dtype=float)
    centre = (length - 1) / 2
    y = heights[0] * np.exp(-0.5 * ((x - centre + spacing / 2) / 2.5) ** 2)
    y += heights[1] * np.exp(-0.5 * ((x - centre - spacing / 2) / 2.5) ** 2)
    if third is not None:
        y += third[1] * np.exp(-0.5 * ((x - centre - third[0]) / 2.5) ** 2)
    return y


def test_grade_profile_pairs_reads_symmetry_and_a_third_peak():
    from sarcasm.utils import Utils
    px = 0.06
    profiles = np.stack([_pair_profile(), _pair_profile((1.0, 0.4)), _pair_profile(third=(30.0, 0.8)),
                         _pair_profile(third=(0.0, 0.7)), np.zeros(120), _pair_profile(third=(45.0, 0.8))])
    lengths, offsets = np.full(6, 30 * px), np.zeros(6)
    lengths[4] = np.nan
    asymmetry, third = Utils.grade_profile_pairs(profiles, px, lengths, offsets)
    assert asymmetry[0] == pytest.approx(0, abs=0.01) and third[0] == 0
    assert asymmetry[1] == pytest.approx(0.6 / 1.4, abs=0.02) and third[1] == 0
    assert third[2] > 0.5                                   # too close outside the pair for a regular lattice
    assert third[3] == pytest.approx(0.7, abs=0.05)         # between the pair
    assert np.isnan(asymmetry[4]) and np.isnan(third[4])
    assert third[5] == 0                                    # one length outside the pair: the next Z-band


def test_grade_profile_pairs_does_not_read_a_broad_band_as_a_third_peak():
    from sarcasm.utils import Utils
    px, x = 0.05, np.arange(120.0)
    band = lambda at, height=1.0: height * np.exp(-0.5 * ((x - at) / 1.5) ** 2)
    profile = (band(42) + band(46, 0.9) + band(78))[None]    # one band with two tops 0.2 um apart
    lengths, offsets = Utils.process_profiles_batch(profile, px, slen_lims=(1, 3), interp_factor=4)
    assert Utils.grade_profile_pairs(profile, px, lengths, offsets)[1][0] == 0
    assert np.isfinite(Utils.grade_profile_pairs(profile, px, lengths, offsets, interp_factor=0)[0][0])


def _fill_scene(third_at=None):
    """One sarcomere between two stripes 30 px apart, three tracks: observed except a gap, sparse, and complete."""
    T, px = 12, 0.06
    obs = np.ones((3, T), bool)
    obs[0, 4:7] = False
    obs[1, 2:11] = False
    pos = np.full((3, T, 2), 0.0, dtype=np.float32)
    pos[:, :, 0] = 40.0
    pos[:, :, 1] = 55.0                                     # the midpoint of the stripes
    pos[0, 4:7, 1] = 58.0                                   # predictions drift off on the gap frames ...
    slen = np.where(obs, 1.8, np.nan).astype(np.float32)
    axis = np.where(obs, 0.0, np.nan)

    def read(k):
        stripes = _stripe_zbands(period_px=30.0, x0=10.0)
        if third_at == k:
            stripes = stripes + 0.8 * np.exp(-0.5 * ((np.arange(stripes.shape[1]) - 55.0) / 2.5) ** 2)[None]
        return stripes
    return pos, axis, obs, slen, read, px


def test_fill_measures_gaps_at_the_track_position_and_leaves_real_frames_alone():
    pos, axis, obs, slen, read, px = _fill_scene()
    filled_slen, filled, out_pos = st.fill_track_gaps(pos, axis, obs, slen, pos.copy(), np.zeros(12, bool), read, px)
    assert filled[0, 4:7].all() and filled.sum() == 3       # the sparse track is left alone
    assert np.isfinite(filled_slen[0]).all()
    assert filled_slen[0, 5] == pytest.approx(1.8, abs=0.02)
    np.testing.assert_array_equal(filled_slen[0, obs[0]], slen[0, obs[0]])
    assert not np.isfinite(filled_slen[1, 2:11]).any()
    assert abs(out_pos[0, 5, 1] - 55.0) < 1.0               # measured midpoint, not the prediction


def test_fill_where_the_line_lengths_round_differently():
    px, n = 0.065, 30                                       # half a line is exactly 30 px, so rounding decides its samples
    obs = np.ones((n, 4), bool)
    obs[:, 1] = False
    pos = np.full((n, 4, 2), 0.0, dtype=np.float32)
    pos[:, :, 0] = 40.0
    pos[:, :, 1] = 55.0
    slen = np.where(obs, 1.95, np.nan).astype(np.float32)
    axis = np.where(obs, np.random.default_rng(0).normal(0, 0.02, n)[:, None], np.nan)
    read = lambda k: _stripe_zbands(period_px=30.0, x0=10.0)
    filled = st.fill_track_gaps(pos, axis, obs, slen, pos.copy(), np.zeros(4, bool), read, px)[1]
    assert filled[:, 1].all()


def test_fill_corrects_the_prediction_and_fills_interior_gaps_only():
    pos, axis, obs, slen, read, px = _fill_scene()
    pos[0, 4:7, 1] = [66.0, 72.0, 78.0]                     # far from the track's next observation at 55
    obs[2, :3] = obs[2, 9:] = obs[2, 5:7] = False
    slen = np.where(obs, slen, np.nan)
    filled_slen, filled, out_pos = st.fill_track_gaps(pos, axis, obs, slen, pos.copy(), np.zeros(12, bool), read, px, min_coverage=0.3)
    np.testing.assert_allclose(out_pos[0, 4:7, 1], 55.0, atol=1.0)
    np.testing.assert_allclose(filled_slen[0, 4:7], 1.8, atol=0.02)
    assert np.flatnonzero(filled[2]).tolist() == [5, 6]


def test_fill_rejects_asymmetric_bands():
    pos, axis, obs, slen, read, px = _fill_scene()
    dim = lambda k: read(k) * np.where(np.arange(read(k).shape[1]) > 55, 0.45, 1.0)[None]
    assert not st.fill_track_gaps(pos, axis, obs, slen, pos.copy(), np.zeros(12, bool), dim, px)[1].any()


def test_fill_rejects_a_third_peak_and_a_length_off_the_rest_level():
    pos, axis, obs, slen, read, px = _fill_scene(third_at=5)
    _, filled, _ = st.fill_track_gaps(pos, axis, obs, slen, pos.copy(), np.zeros(12, bool), read, px)
    assert filled[0, 4] and not filled[0, 5] and filled[0, 6]

    pos, axis, obs, slen, read, px = _fill_scene()
    slen = slen - 0.5                                       # real frames 0.5 um shorter than the map
    rest = np.ones(12, bool)
    assert not st.fill_track_gaps(pos, axis, obs, slen, pos.copy(), rest, read, px)[1].any()
    assert st.fill_track_gaps(pos, axis, obs, slen, pos.copy(), np.zeros(12, bool), read, px)[1][0, 4:7].all()


def test_assess_tracks_applies_each_rule_on_real_frames():
    T = 200
    rest = (np.arange(T) % 50) >= 10                        # four rest periods of 40 frames, beats in between
    rng = np.random.default_rng(0)
    slen = 1.8 + rng.normal(0, 0.005, (6, T))
    obs = np.ones((6, T), bool)
    slen[1, np.arange(T) % 50 == 30] += 0.4                 # flicker in the middle of each rest period
    slen[2, 100:] += 0.3                                    # level shift
    slen[3, ~rest] += 0.8                                   # large excursions during beats: popping is not read
    obs[4, 40:] = False                                     # too few real frames
    obs[5, :] = False
    obs[5, 10:19] = True                                    # never two rest levels
    filled = np.where(obs, slen, np.nan)
    q = st.assess_tracks(slen, obs, filled, rest)
    assert q['rest_flicker'].tolist() == [False, True, False, False, False, False]
    assert q['level_shift'].tolist() == [False, False, True, False, False, False]
    assert q['judged'].tolist() == [True, True, True, True, False, False]
    assert q['passed'].tolist() == [True, False, False, True, False, False]
    # the same sparse track passes once its gaps are filled, if it has enough real frames
    obs2 = obs.copy()
    obs2[4] = np.arange(T) % 5 != 0
    obs2[4, 100:] = np.arange(100, T) % 2 == 0
    q2 = st.assess_tracks(slen, obs2, np.where(obs2, slen, np.nan), rest)
    assert not q2['passed'][4]
    assert st.assess_tracks(slen, obs2, slen, rest)['passed'][4]


def test_assess_tracks_ignores_filled_values_edges_of_rest_and_slow_change():
    T = 200
    rest = (np.arange(T) % 50) >= 10
    slen = 1.8 + np.random.default_rng(0).normal(0, 0.005, (5, T))
    obs = np.ones((5, T), bool)
    filled = slen.copy()
    gap = np.isin(np.arange(T) % 50, [29, 30, 31, 32])
    obs[1, gap] = False
    filled[1, gap] += 0.4                                   # off level, but filled
    slen[2, np.isin(np.arange(T) % 50, [10, 11, 12, 13])] += 0.4       # at the start of each rest period
    filled[2] = slen[2]
    obs[3] = np.arange(T) % 5 < 2                           # 40 % real, 100 % after filling
    slen[4] += 0.15 * (np.arange(T) // 50)                  # small steps add up, but each stays under the limit
    filled[4] = slen[4]
    q = st.assess_tracks(slen, obs, filled, rest)
    assert not q['rest_flicker'].any() and not q['level_shift'].any()
    assert q['passed'].tolist() == [True, True, True, False, True]


def test_assess_tracks_reads_the_shortest_rest_period_in_seconds():
    rest = (np.arange(200) % 20) >= 14                      # rest runs of 6 frames
    slen = np.full((1, 200), 1.8)
    slen[0, 100:] += 0.4
    obs = np.ones((1, 200), bool)
    assert st.assess_tracks(slen, obs, slen, rest, frametime=0.05)['level_shift'][0]       # 0.3 s of rest at 20 fps
    assert not st.assess_tracks(slen, obs, slen, rest, frametime=0.01)['judged'][0]        # 0.06 s at 100 fps


def test_assess_tracks_without_rest_judges_nothing():
    slen = np.full((2, 50), 1.8)
    obs = np.ones((2, 50), bool)
    q = st.assess_tracks(slen, obs, slen, np.zeros(50, bool))
    assert not q['judged'].any() and q['passed'].all()

