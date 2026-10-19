"""Unit tests for helpers in :mod:`sarcasm.analysis.sarcomere_vectors`."""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.analysis import sarcomere_vectors as sv
from sarcasm.utils import Utils


def _field_from_angle(angle: float, H: int = 8, W: int = 8) -> np.ndarray:
    """Build a (2, H, W) orientation field with uniform angle."""
    return np.stack(
        [np.full((H, W), np.cos(angle), dtype=np.float32),
         np.full((H, W), np.sin(angle), dtype=np.float32)],
        axis=0,
    )


def _axial_angle(field: np.ndarray) -> np.ndarray:
    """Recover the axial angle [0, π) from a (T, 2, H, W) field."""
    a = np.arctan2(field[:, 1], field[:, 0])
    a = (a + 2 * np.pi) % (2 * np.pi)
    return np.where(a > np.pi, a - np.pi, a)


def test_smooth_orientation_zero_sigma_is_identity():
    stack = np.stack([_field_from_angle(0.1), _field_from_angle(0.2)])
    out = sv.smooth_orientation_field_temporal(stack, sigma=0.0)
    assert out is stack or np.array_equal(out, stack)


def test_smooth_orientation_constant_field_is_unchanged():
    """A constant-orientation stack must be invariant under temporal smoothing."""
    field = _field_from_angle(0.7)  # some arbitrary axial angle
    stack = np.stack([field.copy() for _ in range(10)])
    out = sv.smooth_orientation_field_temporal(stack, sigma=1.5)
    angles_in = _axial_angle(stack)
    angles_out = _axial_angle(out)
    np.testing.assert_allclose(angles_out, angles_in, atol=1e-5)


def test_smooth_orientation_handles_sign_flip_axially():
    """The U-Net may output (cos θ, sin θ) in one frame and (-cos θ, -sin θ) in
    the next for the *same* physical axis. Axial-correct smoothing must
    recover the common angle, not average to zero."""
    angle = 0.4
    T = 8
    stack = np.empty((T, 2, 4, 4), dtype=np.float32)
    for t in range(T):
        sign = 1.0 if t % 2 == 0 else -1.0
        stack[t, 0] = sign * np.cos(angle)
        stack[t, 1] = sign * np.sin(angle)
    out = sv.smooth_orientation_field_temporal(stack, sigma=1.0)
    angles_out = _axial_angle(out)
    # Every pixel should recover the original axial angle (mod π).
    diff = (angles_out - angle + np.pi / 2) % np.pi - np.pi / 2
    assert np.max(np.abs(diff)) < 1e-4


def test_smooth_orientation_reduces_temporal_variance():
    """Smoothing should suppress frame-to-frame noise on the axial angle."""
    rng = np.random.default_rng(0)
    true_angle = 1.2
    T, H, W = 30, 16, 16
    stack = np.empty((T, 2, H, W), dtype=np.float32)
    noise = rng.normal(0.0, 0.15, size=T)  # per-frame angular noise (rad)
    for t in range(T):
        a = true_angle + noise[t]
        stack[t, 0] = np.cos(a); stack[t, 1] = np.sin(a)

    angles_raw = _axial_angle(stack).mean(axis=(1, 2))
    var_raw = float(np.var(angles_raw))

    out = sv.smooth_orientation_field_temporal(stack, sigma=1.5)
    angles_smooth = _axial_angle(out).mean(axis=(1, 2))
    var_smooth = float(np.var(angles_smooth))

    assert var_smooth < 0.5 * var_raw, (
        f"expected variance reduction; raw={var_raw:.4f}, smooth={var_smooth:.4f}"
    )


def test_smooth_orientation_rejects_wrong_shape():
    import pytest
    # Single-frame shape (2, H, W) — not a valid (T, 2, H, W) input.
    single = _field_from_angle(0.5)
    with pytest.raises(ValueError):
        sv.smooth_orientation_field_temporal(single, sigma=1.0)


def test_smooth_orientation_scipy_and_torch_agree():
    """scipy and torch backends must produce essentially identical output
    on the same input (both implement the same mathematical operation)."""
    rng = np.random.default_rng(1)
    T, H, W = 12, 8, 8
    angles = rng.uniform(0, np.pi, size=(T, H, W))
    stack = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    out_scipy = sv.smooth_orientation_field_temporal(stack, sigma=1.0, backend='scipy')
    out_torch = sv.smooth_orientation_field_temporal(stack, sigma=1.0, backend='torch', device='cpu')
    # The two backends truncate the Gaussian at slightly different radii
    # (scipy 4σ default vs torch 3σ) and use different conv implementations;
    # a few‑mrad angular agreement is more than enough for physical use.
    np.testing.assert_allclose(out_scipy, out_torch, atol=5e-3)


# ---------------------------------------------------------------------------
# Profile peak detection
# ---------------------------------------------------------------------------

def _gauss_peak(x, center, sigma=0.1):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _synthetic_profile(slen_um, pixelsize=0.06, n=40, noise=0.0, seed=0):
    """Build a 1-D intensity profile with two Gaussian Z-band peaks separated
    by ``slen_um`` centred on the profile."""
    rng = np.random.default_rng(seed)
    x = np.arange(n) * pixelsize
    center = (x[-1] + x[0]) * 0.5
    left = center - 0.5 * slen_um
    right = center + 0.5 * slen_um
    y = _gauss_peak(x, left, sigma=0.08) + _gauss_peak(x, right, sigma=0.08)
    y = y + rng.normal(0.0, noise, size=n)
    return y.astype(np.float32)


def test_process_profiles_batch_recovers_known_slen():
    slen_true = 1.85
    profiles = [_synthetic_profile(slen_true) for _ in range(5)]
    slens, _ = Utils.process_profiles_batch(
        profiles, pixelsize=0.06, interp_factor=4, interpolation_method='akima',
        prominence=0.5,
    )
    np.testing.assert_allclose(slens, slen_true, atol=0.05)


def _profile_with_spurious_peak(slen_um, bump, offset=0.35, pixelsize=0.06, n=40, seed=0):
    """Two real Z-band peaks plus a faint third one between the profile centre
    and the true left peak, placed so that *prominence* is the filter that
    decides its fate.

    scipy applies height, then distance, then prominence — so the spurious peak
    must sit farther than ``min_dist`` from both true peaks, or the distance rule
    removes it before prominence is ever consulted (the tests below pass
    ``min_dist=0.3`` for that reason). If prominence lets it through it becomes
    the 'last peak below centre' and the measured length collapses from
    ``slen_um`` to ``slen_um / 2 + offset``.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(n) * pixelsize
    center = (x[-1] + x[0]) * 0.5
    y = (_gauss_peak(x, center - 0.5 * slen_um, sigma=0.08)
         + _gauss_peak(x, center + 0.5 * slen_um, sigma=0.08)
         + bump * _gauss_peak(x, center - offset, sigma=0.08))
    return (y + rng.normal(0.0, 0.01, size=n)).astype(np.float32)


_PROM_KW = dict(pixelsize=0.06, interp_factor=4, interpolation_method='akima',
                min_dist=0.3)


def test_prominence_rejects_spurious_peaks():
    """Prominence, and only prominence, must reject a sub-threshold peak.

    Every other setting is held fixed, so this fails if ``prominence`` is ignored
    or hard-coded. The previous version of this test varied ``interp_factor`` at
    the same time and passed even when the prominence argument was discarded
    entirely.
    """
    slen_true, offset = 1.85, 0.35
    profiles = [
        _profile_with_spurious_peak(slen_true, bump=0.3, offset=offset, seed=i)
        for i in range(64)
    ]
    strict, _ = Utils.process_profiles_batch(profiles, prominence=0.5, **_PROM_KW)
    loose, _ = Utils.process_profiles_batch(profiles, prominence=0.05, **_PROM_KW)

    # Strict: the 0.3-high bump lacks the prominence to qualify, so the true
    # Z-band pair survives and the length is right.
    np.testing.assert_allclose(np.nanmedian(strict), slen_true, atol=0.05)
    # Loose: the bump qualifies and displaces the left peak, so the measurement
    # collapses towards ``slen_true / 2 + offset``. The centre-of-mass window
    # still drags it a little towards the true peak, so assert the direction and
    # magnitude rather than the exact value.
    assert np.nanmedian(loose) < slen_true - 0.3, (
        f'loose prominence should admit the spurious peak and shorten the '
        f'measurement (got median {np.nanmedian(loose):.4f} µm)'
    )


def test_prominence_argument_is_actually_used():
    """Guard against the threshold being ignored or hard-coded downstream."""
    profiles = [
        _profile_with_spurious_peak(1.85, bump=0.3, seed=i) for i in range(32)
    ]
    a, _ = Utils.process_profiles_batch(profiles, prominence=0.05, **_PROM_KW)
    b, _ = Utils.process_profiles_batch(profiles, prominence=0.5, **_PROM_KW)
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b)), (
        'changing prominence must change the result'
    )


def test_process_profiles_batch_prominence_is_scale_invariant():
    """Peak selection must not depend on the profile's absolute intensity scale.

    ``process_profiles_batch`` min-max normalises each profile before applying
    the prominence threshold, so scaling a profile by any positive constant must
    leave the measured length unchanged. The deleted LOI path failed exactly this
    — it applied a hard-coded absolute prominence to un-normalised data, so it
    only behaved as intended on [0, 1] probability maps.
    """
    rng = np.random.default_rng(11)
    profiles = [
        _synthetic_profile(1.85, noise=0.05, seed=int(rng.integers(1e9)))
        for _ in range(64)
    ]
    base, _ = Utils.process_profiles_batch(profiles, pixelsize=0.06)
    for scale in (0.01, 255.0):
        scaled = [(p * scale).astype(np.float32) for p in profiles]
        got, _ = Utils.process_profiles_batch(scaled, pixelsize=0.06)
        np.testing.assert_allclose(got, base, rtol=1e-6, equal_nan=True,
                                   err_msg=f'scale={scale}')


def test_orientation_at_points_matches_full_angle_map_in_interior():
    """The sparse gather must equal the full-frame disk median away from borders.

    ``orientation_at_points`` replaced a whole-image median filter that was 59%
    of ``analyze_sarcomere_vectors`` runtime while ~98% of its output was
    discarded. Interior results must be identical; at the border they
    legitimately differ, because the full-frame filter zero-pads and zero is not
    a valid unit vector, whereas the gather replicates the edge.
    """
    rng = np.random.default_rng(7)
    height, width, radius = 64, 96, 4

    # Smooth-ish field plus noise, so the median actually has work to do.
    yy, xx = np.mgrid[0:height, 0:width]
    angle = 0.6 * np.sin(xx / 11.0) + 0.4 * np.cos(yy / 7.0)
    angle += rng.normal(0, 0.25, angle.shape)
    field = np.stack([np.cos(angle), np.sin(angle)]).astype(np.float32)

    full = Utils.get_orientation_angle_map(field, use_median_filter=True, radius=radius)

    rows = rng.integers(radius, height - radius, 400)
    cols = rng.integers(radius, width - radius, 400)
    sparse = sv.orientation_at_points(field, rows, cols, radius)

    np.testing.assert_allclose(sparse, full[rows, cols], atol=1e-12)


def test_orientation_at_points_empty_input():
    field = _field_from_angle(0.3, H=16, W=16)
    out = sv.orientation_at_points(field, np.array([], dtype=int),
                                   np.array([], dtype=int), 3)
    assert out.shape == (0,)


def test_batched_peak_finder_matches_scipy_find_peaks():
    """The numba peak kernel must reproduce scipy's find_peaks selection.

    Guards the filter *order* in particular: scipy applies height, then
    distance, then prominence, so a peak dropped by the distance rule is never
    prominence-tested and a low-prominence peak can still suppress a neighbour.
    Getting that order wrong changes which sarcomeres are measured.
    """
    from scipy.signal import find_peaks
    from sarcasm.utils import _slen_from_profiles

    rng = np.random.default_rng(3)
    n_profiles, length = 300, 220
    x_interp = np.linspace(0.0, 4.0, length)
    profiles = np.empty((n_profiles, length), dtype=np.float64)
    for i in range(n_profiles):
        # Several bumps of varied height/width plus noise -> plenty of marginal
        # peaks near the height/prominence/distance thresholds.
        y = np.zeros(length)
        for _ in range(rng.integers(2, 6)):
            c = rng.uniform(0, 4.0)
            y += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((x_interp - c) / rng.uniform(0.05, 0.3)) ** 2)
        profiles[i] = y + rng.normal(0, 0.03, length)

    thres, distance, prominence, window = 0.25, 12, 0.3, 8
    center = (x_interp[-1] + x_interp[0]) * 0.5
    flat = np.zeros(n_profiles, dtype=bool)

    got_slen, got_off = _slen_from_profiles(
        profiles, x_interp, flat, thres, distance, prominence,
        window, center, 0.2, 3.0,
    )

    for i in range(n_profiles):
        y = profiles[i]
        peaks_idx, _ = find_peaks(y, height=thres, distance=distance, prominence=prominence)
        if len(peaks_idx) < 2:
            assert np.isnan(got_slen[i])
            continue
        coms = np.empty(len(peaks_idx))
        for j, idx in enumerate(peaks_idx):
            s, e = max(0, idx - window), min(length, idx + window + 1)
            w = y[s:e] - y[s:e].min()
            coms[j] = np.dot(x_interp[s:e], w) / w.sum() if w.sum() > 0 else x_interp[idx]
        left, right = coms < center, coms >= center
        if not (left.any() and right.any()):
            assert np.isnan(got_slen[i])
            continue
        expected = coms[right][0] - coms[left][-1]
        if 0.2 <= expected <= 3.0:
            assert np.isclose(got_slen[i], expected, atol=1e-9), f'profile {i}'
        else:
            assert np.isnan(got_slen[i])

    assert np.isfinite(got_slen).sum() > 0.5 * n_profiles, 'test data exercises too few peaks'


def test_get_sarcomere_vectors_recovers_known_slen():
    """End-to-end on a synthetic two-Z-band field with a known spacing.

    Z-bands are horizontal, so the sarcomere axis is vertical and the profile
    must be cast along rows. ``get_sarcomere_vectors`` builds the profile
    direction as ``(sin θ, cos θ)`` against ``(row, col)`` positions, so that
    requires θ = π/2, i.e. the orientation field's *y* channel set to 1.
    """
    H, W = 48, 48
    pixelsize = 0.1
    zb = np.zeros((H, W), dtype=np.float32)
    mb = np.zeros((H, W), dtype=np.float32)
    zb[16, 8:40] = 1.0; zb[32, 8:40] = 1.0; mb[24, 8:40] = 1.0
    ori_field = np.zeros((2, H, W), dtype=np.float32)
    ori_field[1] = 1.0  # sin θ = 1 → θ = π/2, profile along rows

    *_, slen, _orient, n_mbands = sv.get_sarcomere_vectors(
        zb, mb > 0.25, ori_field, pixelsize=pixelsize,
    )
    assert n_mbands == 1
    assert np.isfinite(slen).mean() > 0.9, 'most midline points should yield a length'
    # Z-bands sit 16 px apart → 1.6 µm.
    np.testing.assert_allclose(np.nanmedian(slen), 16 * pixelsize, atol=0.05)


def test_smooth_orientation_single_frame_noop():
    """A single-frame stack (T=1) has nothing to smooth; return unchanged."""
    one = _field_from_angle(0.5)[np.newaxis]  # (1, 2, H, W)
    out = sv.smooth_orientation_field_temporal(one, sigma=2.0)
    np.testing.assert_array_equal(out, one)


# --- neighbourhood outlier filter -------------------------------------------

def _comb(n_bands: int = 3, n_points: int = 20, theta: float = np.pi / 2,
          spacing: float = 0.1, band_gap: float = 1.8):
    """Vectors on ``n_bands`` parallel M-bands, one per pixel along each band.

    Positions are in µm: points ``spacing`` apart along a band (one midline
    pixel each, like the real output), bands ``band_gap`` apart. Every vector
    carries the same axial orientation ``theta``.
    """
    rows = np.repeat(np.arange(n_bands) * band_gap, n_points)
    cols = np.tile(np.arange(n_points) * spacing, n_bands)
    pos = np.stack([rows, cols], axis=1)
    return pos, np.full(len(pos), theta)


def test_neighbor_filter_keeps_a_parallel_comb():
    pos, theta = _comb()
    dev, n_nb = sv.neighbor_orientation_deviation(pos, theta, radius=1.0)
    assert n_nb.min() >= 1 and np.nanmax(dev) < 1e-9
    assert sv.filter_vectors_by_neighbors(pos, theta).all()


def test_neighbor_filter_removes_the_one_vector_pointing_across_its_neighbours():
    pos, theta = _comb()
    theta = theta.copy()
    theta[10] += np.radians(60)  # one point of the first band points 60° off
    keep = sv.filter_vectors_by_neighbors(pos, theta, neighbor_radius=1.0, max_neighbor_deviation=30.0)
    assert not keep[10]
    assert keep.sum() == len(pos) - 1, 'its neighbours must survive the outlier in their mean'


def test_neighbor_filter_keeps_vectors_without_neighbours():
    """A vector with nobody inside the radius cannot be judged and is kept, whatever it points at."""
    pos, theta = _comb()
    pos = np.vstack([pos, [[10.0, 10.0]]])
    theta = np.append(theta, 0.0)  # 90° off the comb, but 10 µm away from it
    dev, n_nb = sv.neighbor_orientation_deviation(pos, theta, radius=1.0)
    assert n_nb[-1] == 0 and np.isnan(dev[-1])
    assert sv.filter_vectors_by_neighbors(pos, theta)[-1]


def test_neighbor_deviation_is_axial():
    """θ and θ + π are the same sarcomere axis; the deviation runs 0–90°."""
    pos, theta = _comb(n_bands=1, theta=0.3)
    flipped = theta.copy(); flipped[5] += np.pi
    dev, _ = sv.neighbor_orientation_deviation(pos, flipped, radius=1.0)
    assert dev[5] < 1e-9
    crossed = theta.copy(); crossed[5] += np.pi / 2
    dev, _ = sv.neighbor_orientation_deviation(pos, crossed, radius=1.0)
    np.testing.assert_allclose(dev[5], 90.0, atol=1e-9)


def test_neighbor_radius_is_in_micrometres():
    pos = np.array([[0.0, 0.0], [0.0, 0.9]])
    theta = np.array([0.0, np.pi / 2])
    _, n_nb = sv.neighbor_orientation_deviation(pos, theta, radius=1.0)
    assert n_nb.tolist() == [1, 1]
    _, n_nb = sv.neighbor_orientation_deviation(pos, theta, radius=0.5)
    assert n_nb.tolist() == [0, 0]
    # both vectors point 90° across each other: at 1 µm both go, at 0.5 µm both stay
    assert not sv.filter_vectors_by_neighbors(pos, theta, neighbor_radius=1.0).any()
    assert sv.filter_vectors_by_neighbors(pos, theta, neighbor_radius=0.5).all()


def test_neighbor_filter_degenerate_inputs():
    assert sv.filter_vectors_by_neighbors(np.empty((0, 2)), np.empty(0)).shape == (0,)
    assert sv.filter_vectors_by_neighbors(np.zeros((1, 2)), np.zeros(1)).tolist() == [True]
    # a vector with an undefined orientation neither counts for its neighbours nor is dropped
    pos, theta = _comb(n_bands=1)
    theta = theta.copy(); theta[3] = np.nan
    dev, n_nb = sv.neighbor_orientation_deviation(pos, theta, radius=1.0)
    assert np.isnan(dev[3]) and n_nb[3] == 0
    assert np.isfinite(dev[[2, 4]]).all() and np.nanmax(dev) < 1e-9
    assert sv.filter_vectors_by_neighbors(pos, theta).all()


def test_frame_without_mbands_gives_empty_arrays_of_the_same_dimensionality():
    """A plain empty list next to (n, 2) position arrays made the results store fall back to writing the whole
    per-frame position list as an inline JSON attribute."""
    shape = (32, 32)
    field = np.stack([np.ones(shape), np.zeros(shape)])
    out = sv.get_sarcomere_vectors(np.zeros(shape, np.float32), np.zeros(shape, bool), field, pixelsize=0.1)
    pos_px, pos, mid, mid_len, slen, ori, n_mbands = out
    assert n_mbands == 0
    assert pos_px.shape == (0, 2) and pos.shape == (0, 2)
    assert mid.shape == mid_len.shape == slen.shape == ori.shape == (0,)


# --- Z-band registry --------------------------------------------------------

def _row(n: int = 6, slen: float = 1.8, theta: float = 0.0, y: float = 0.0,
         x0: float = 0.0, lengths=None):
    """``n`` sarcomeres end to end along one myofibril, sharing Z-discs, laid out along the columns (axis
    ``theta = 0``: ``t = (sin θ, cos θ) = (0, 1)``); the first centre at ``x0``. ``lengths`` overrides the per-sarcomere lengths (the stored ``slen`` stays ``slen`` so that the
    lattice of every vector is the nominal one, as in a real recording with a local-median length)."""
    L = np.full(n, slen) if lengths is None else np.asarray(lengths, float)
    ends = x0 - L[0] / 2 + np.r_[0, np.cumsum(L)]
    centres = 0.5 * (ends[:-1] + ends[1:])
    pos = np.stack([np.full(n, y), centres], axis=1)              # (row, col) µm; axis along the columns
    return pos, np.full(n, theta), np.full(n, slen)


def _two_rows(shift: float, slen: float = 1.8, n: int = 6, gap: float = 2.0):
    """Two parallel myofibrils ``gap`` µm apart; the second shifted by ``shift`` µm along the axis."""
    p1, o1, l1 = _row(n, slen)
    p2, o2, l2 = _row(n, slen, y=gap, x0=shift)
    return np.vstack([p1, p2]), np.concatenate([o1, o2]), np.concatenate([l1, l2])


def test_registry_lateral_is_one_for_level_zdiscs():
    pos, ori, L = _two_rows(shift=0.0)
    lat, ax = sv.zband_registry(pos, ori, L)
    np.testing.assert_allclose(lat[np.isfinite(lat)], 1.0, atol=1e-6)
    assert np.isfinite(lat).sum() == len(pos)


def test_registry_lateral_is_minus_one_for_zdiscs_at_the_mband():
    pos, ori, L = _two_rows(shift=0.9)                            # half a sarcomere
    lat, _ = sv.zband_registry(pos, ori, L)
    np.testing.assert_allclose(lat[np.isfinite(lat)], -1.0, atol=1e-6)


def test_registry_lateral_is_zero_for_a_quarter_shift():
    pos, ori, L = _two_rows(shift=0.45)                           # a quarter sarcomere: cos(π/2)
    lat, _ = sv.zband_registry(pos, ori, L)
    np.testing.assert_allclose(lat[np.isfinite(lat)], 0.0, atol=1e-6)


def test_registry_axial_is_one_along_a_regular_row():
    pos, ori, L = _row(8)
    _, ax = sv.zband_registry(pos, ori, L)
    np.testing.assert_allclose(ax[np.isfinite(ax)], 1.0, atol=1e-6)
    assert np.isfinite(ax).all()


def test_registry_axial_drops_when_the_spacing_drifts():
    """Sarcomeres 1.25 L long put the Z-discs two sarcomeres down the row at the M-band positions."""
    pos, ori, L = _row(7, lengths=np.full(7, 1.25 * 1.8))
    _, ax = sv.zband_registry(pos, ori, L)
    assert np.nanmean(ax) < 0.0


def test_registry_excludes_vectors_without_neighbours_instead_of_scoring_zero():
    pos, ori, L = _row(6)                                         # one myofibril: nothing beside it
    lat, ax = sv.zband_registry(pos, ori, L)
    assert np.isnan(lat).all() and np.isfinite(ax).all()
    lone = sv.zband_registry(np.array([[0.0, 0.0]]), np.array([0.0]), np.array([1.8]))
    assert all(np.isnan(v).all() for v in lone)


def test_registry_is_axially_flip_invariant():
    pos, ori, L = _two_rows(shift=0.3)
    rng = np.random.default_rng(0)
    flipped = ori + np.where(rng.random(len(ori)) < 0.5, np.pi, 0.0)
    for variant in (flipped, np.mod(flipped, np.pi)):
        a = sv.zband_registry(pos, ori, L)
        b = sv.zband_registry(pos, variant, L)
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.nan_to_num(x, nan=9.0), np.nan_to_num(y, nan=9.0), atol=1e-9)


def test_registry_ignores_an_oblique_row_neighbour_but_not_an_oblique_lateral_one():
    pos, ori, L = _two_rows(shift=0.0)
    tilted = ori.copy(); tilted[8] += np.radians(30)              # one vector of the second row, 30° off
    lat, ax = sv.zband_registry(pos, tilted, L)
    # its row neighbours no longer count it (beyond max_angle), its lateral partner in row 1 still does
    assert np.isfinite(lat[2])
    lat0, ax0 = sv.zband_registry(pos, ori, L)
    assert np.isfinite(ax0).all()


def test_registry_parameters_are_in_micrometres_and_degrees():
    pos, ori, L = _two_rows(shift=0.0, gap=2.0)
    lat_in, _ = sv.zband_registry(pos, ori, L, lateral_band=(1.5, 2.5))
    lat_out, _ = sv.zband_registry(pos, ori, L, lateral_band=(2.5, 3.5))   # the neighbour is 2 µm away
    assert np.isfinite(lat_in).all() and np.isnan(lat_out).all()
    pos, ori, L = _row(8)
    _, ax1 = sv.zband_registry(pos, ori, L, axial_range=1)
    _, ax3 = sv.zband_registry(pos, ori, L, axial_range=3)
    np.testing.assert_allclose(ax1[np.isfinite(ax1)], 1.0); np.testing.assert_allclose(ax3[np.isfinite(ax3)], 1.0)


def test_registry_degenerate_inputs():
    lat, ax = sv.zband_registry(np.empty((0, 2)), np.empty(0), np.empty(0))
    assert lat.shape == (0,) and ax.shape == (0,)
    pos, ori, L = _two_rows(shift=0.0)
    L = L.copy(); L[0] = np.nan
    lat, ax = sv.zband_registry(pos, ori, L)
    assert np.isnan(lat[0]) and np.isnan(ax[0])
    assert np.isnan(lat[6]), 'its lateral partner was the dropped vector'
    assert np.isfinite(np.delete(lat, [0, 6])).all() and np.isfinite(ax[1:]).all()
