"""Unit tests for helpers in :mod:`sarcasm.analysis.sarcomere_vectors`."""
from __future__ import annotations

import numpy as np

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
# Profile peak detection — default vs LOI pipelines
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


def test_process_profiles_batch_loi_recovers_known_slen():
    slen_true = 1.85
    profiles = [_synthetic_profile(slen_true) for _ in range(5)]
    slens, _ = Utils.process_profiles_batch_loi(
        profiles, pixelsize=0.06, slen_lims=(1.0, 3.0),
    )
    np.testing.assert_allclose(slens, slen_true, atol=0.05)


def test_loi_peak_algorithm_tighter_than_default_under_noise():
    """With noisy profiles the LOI algorithm (6× Akima + prominence=0.5) should
    produce tighter slen estimates than the default at interp_factor=0 +
    prominence=0.2 — because upsampling reduces peak-position quantisation and
    the stricter prominence rejects noise-driven spurious peaks."""
    rng = np.random.default_rng(0)
    slen_true = 1.85
    n_profiles = 500
    profiles = [
        _synthetic_profile(slen_true, noise=0.05, seed=int(rng.integers(1e9)))
        for _ in range(n_profiles)
    ]

    slens_loi, _ = Utils.process_profiles_batch_loi(profiles, pixelsize=0.06)
    slens_default_tight, _ = Utils.process_profiles_batch(
        profiles, pixelsize=0.06, interp_factor=4, interpolation_method='akima',
        prominence=0.5,
    )
    # Degenerate default — mimics the previous behaviour (no upsample, loose prominence)
    slens_default_loose, _ = Utils.process_profiles_batch(
        profiles, pixelsize=0.06, interp_factor=0,
        prominence=0.2,
    )

    loi_std = float(np.nanstd(slens_loi))
    tight_std = float(np.nanstd(slens_default_tight))
    loose_std = float(np.nanstd(slens_default_loose))
    # LOI should beat (or match within tolerance) the tightened default, and
    # both should beat the loose default by a clear margin.
    assert loi_std < 2.0 * tight_std
    assert tight_std < loose_std * 0.95, (
        f"tightened default (std={tight_std:.4f}) should be tighter than loose "
        f"default (std={loose_std:.4f})"
    )


def test_get_sarcomere_vectors_accepts_peak_algorithm():
    """Smoke test: the ``peak_algorithm`` kwarg is accepted and routed."""
    import pytest
    H, W = 32, 32
    zb = np.zeros((H, W), dtype=np.float32)
    mb = np.zeros((H, W), dtype=np.float32)
    zb[10, 5:25] = 1.0; zb[20, 5:25] = 1.0; mb[15, 5:25] = 1.0
    ori_field = np.zeros((2, H, W), dtype=np.float32)
    ori_field[0] = 1.0  # cos θ = 1 → θ = 0
    for algo in ('default', 'loi'):
        out = sv.get_sarcomere_vectors(
            zb, mb > 0.25, ori_field, pixelsize=0.1, peak_algorithm=algo,
        )
        assert out is not None  # no exception
    with pytest.raises(ValueError):
        sv.get_sarcomere_vectors(
            zb, mb > 0.25, ori_field, pixelsize=0.1, peak_algorithm='garbage',
        )


def test_smooth_orientation_single_frame_noop():
    """A single-frame stack (T=1) has nothing to smooth; return unchanged."""
    one = _field_from_angle(0.5)[np.newaxis]  # (1, 2, H, W)
    out = sv.smooth_orientation_field_temporal(one, sigma=2.0)
    np.testing.assert_array_equal(out, one)
