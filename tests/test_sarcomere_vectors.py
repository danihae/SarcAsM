"""Unit tests for helpers in :mod:`sarcasm.structure_modules.sarcomere_vectors`."""
from __future__ import annotations

import numpy as np

from sarcasm.structure_modules import sarcomere_vectors as sv


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


def test_smooth_orientation_single_frame_noop():
    """A single-frame stack (T=1) has nothing to smooth; return unchanged."""
    one = _field_from_angle(0.5)[np.newaxis]  # (1, 2, H, W)
    out = sv.smooth_orientation_field_temporal(one, sigma=2.0)
    np.testing.assert_array_equal(out, one)
