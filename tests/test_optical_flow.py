"""Tests for :mod:`sarcasm.analysis.optical_flow` — the image-flow motion predictor."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from sarcasm.analysis import optical_flow as of


def _texture(H=256, W=256, seed=0):
    """Striations (28 px period, tilted) modulated by smooth blobs, so both flow
    components are determined (pure striations leave the lateral component free)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    stripes = 0.5 + 0.5 * np.cos(2 * np.pi * (xx * np.cos(0.2) + yy * np.sin(0.2)) / 28.0)
    blobs = ndimage.gaussian_filter(rng.random((H, W)), 6)
    blobs = (blobs - blobs.min()) / np.ptp(blobs)
    return (1000.0 * stripes * blobs + 50.0).astype(np.float32)


def test_frame_to_uint8_clips_to_the_window():
    img = np.array([[-5.0, 0.0, 50.0, 100.0, 1e6]], dtype=np.float32)
    u = of.frame_to_uint8(img, 0.0, 100.0)
    assert u.dtype == np.uint8
    assert u.tolist() == [[0, 0, 127, 255, 255]]
    lo, hi = of.intensity_range(np.zeros((4, 4)))
    assert hi > lo  # degenerate frame still gives a usable window


@pytest.mark.parametrize('shift', [(0.0, 0.0), (0.5, -0.5), (3.0, 2.0), (12.0, -6.0)])
def test_dis_flow_recovers_a_known_shift(shift):
    """flow[y, x] = (dy, dx) such that a feature at (y, x) in ``a`` sits at (y+dy, x+dx) in ``b``."""
    a = _texture()
    b = ndimage.shift(a, shift, order=1, mode='nearest')
    lo, hi = of.intensity_range(a)
    flow = of.dis_flow(of.frame_to_uint8(a, lo, hi), of.frame_to_uint8(b, lo, hi))
    assert flow.shape == a.shape + (2,) and flow.dtype == np.float32
    inner = flow[40:-40, 40:-40].reshape(-1, 2)          # away from the border
    med = np.median(inner, axis=0)
    assert np.allclose(med, shift, atol=0.5), (med, shift)


def test_block_mean_and_sampling_reproduce_a_linear_field():
    H, W, ds = 64, 96, 4
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    field = np.stack([yy / 10.0, -xx / 20.0], axis=-1)
    small = of.block_mean(field, ds)
    assert small.shape == (H // ds, W // ds, 2)
    pts = np.array([[10.0, 20.0], [31.5, 47.25], [50.0, 80.0]])
    got = of.sample_flow(small, pts, ds)
    expect = np.column_stack([pts[:, 0] / 10.0, -pts[:, 1] / 20.0])
    assert np.allclose(got, expect, atol=0.02)
    assert of.sample_flow(small, np.zeros((0, 2)), ds).shape == (0, 2)
    assert of.block_mean(field, 1) is not None and of.block_mean(field, 1).shape == field.shape


def test_image_flow_predictor_predicts_each_step_and_reads_each_frame_once():
    a = _texture(); step = (4.0, -2.0)
    frames = [ndimage.shift(a, (k * step[0], k * step[1]), order=1, mode='nearest') for k in range(3)]
    reads = []

    def read_frame(k):
        reads.append(k)
        return frames[k][None]          # a (1, H, W) stack slice is accepted

    pred = of.ImageFlowPredictor(read_frame, downsample=4)
    pts = np.array([[100.0, 100.0], [120.0, 150.0], [160.0, 90.0]])
    d0 = pred(0, pts)
    assert d0.shape == (3, 2) and np.allclose(d0, step, atol=0.5), d0
    d0_again = pred(0, pts[:1])          # cached field, no re-read
    assert np.allclose(d0_again, d0[:1])
    d1 = pred(1, pts)
    assert np.allclose(d1, step, atol=0.5), d1
    assert sorted(set(reads)) == [0, 1, 2] and len(reads) == 3


def test_image_flow_predictor_rejects_a_non_2d_frame():
    pred = of.ImageFlowPredictor(lambda k: np.zeros((2, 8, 8)))
    with pytest.raises(ValueError, match='2D frame'):
        pred(0, np.zeros((1, 2)))
