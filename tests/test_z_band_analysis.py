"""Lateral Z-band connections of ``z_band_analysis.analyze_z_bands`` on synthetic Z-band maps.

Z-bands are vertical (along the rows), the sarcomere axis runs along x (columns); pixel size 0.1 µm.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.draw import line

from sarcasm.analysis import z_band_analysis as zba

PX = 0.1
PARAMS = dict(min_length=0.2, threshold=0.5, median_filter_radius=0.2, theta_phi_min=0.4, a_min=0.3, d_max=3.0,
              d_min=0.0)


def _zband_map(segments=(), arcs=(), shape=(300, 300)):
    """Binary Z-band map from straight segments ((r0, c0), (r1, c1)) and arcs (center, radius, start_deg, stop_deg)
    in px, three pixels wide."""
    img = np.zeros(shape, bool)
    for (r0, c0), (r1, c1) in segments:
        rr, cc = line(r0, c0, r1, c1)
        img[rr, cc] = True
    for (rc, cc0), radius, a0, a1 in arcs:
        t = np.radians(np.linspace(a0, a1, 400))
        img[np.round(rc - radius * np.sin(t)).astype(int), np.round(cc0 + radius * np.cos(t)).astype(int)] = True
    return ndimage.binary_dilation(img).astype(np.float32)


def _analyse(zbands, **kw):
    field = np.stack([np.ones_like(zbands), np.zeros_like(zbands)])      # sarcomere axis along x
    labels, skel = zba.segment_z_bands(zbands, threshold=0.5)
    res = zba.analyze_z_bands(zbands, labels, skel, zbands, field, pixelsize=PX, **PARAMS, **kw)
    ends = np.asarray(res[13], float) / PX                                 # (n_z, 2 ends, row/col) px
    links = np.asarray(res[12])
    pairs = {tuple(sorted((int(i), int(j)))) for i, _, j, _ in links.T} if links.size else set()
    return ends, pairs, np.asarray(res[16], float)


def _band_at(ends, row, col):
    """Index of the Z-band whose mean end position is closest to (row, col) px."""
    return int(np.argmin(np.hypot(ends[:, :, 0].mean(1) - row, ends[:, :, 1].mean(1) - col)))


def test_collinear_bands_are_linked_into_one_group():
    zb = _zband_map(segments=[((50, 150), (100, 150)), ((115, 150), (165, 150))])
    ends, pairs, group_lengths = _analyse(zb)
    assert pairs == {(0, 1)}
    assert group_lengths.size == 1
    assert abs(group_lengths[0] - 11.5) < 0.6           # 5 µm + 1.5 µm gap + 5 µm


def test_axial_offset_limit_rejects_staggered_link():
    # lateral gap 2 µm, offset 1 µm along the sarcomere axis: well aligned (A = 0.8) but beyond 0.9 µm
    zb = _zband_map(segments=[((50, 150), (100, 150)), ((120, 160), (170, 160))])
    _, pairs_off, _ = _analyse(zb, max_axial_offset=None)
    _, pairs_on, _ = _analyse(zb, max_axial_offset=0.9)
    assert pairs_off == {(0, 1)}
    assert pairs_on == set()
    _, pairs_loose, _ = _analyse(zb, max_axial_offset=1.2)
    assert pairs_loose == {(0, 1)}


def test_link_over_a_third_zband_is_rejected():
    a, b = ((50, 150), (100, 150)), ((125, 150), (175, 150))
    _, pairs, _ = _analyse(_zband_map(segments=[a, b]))
    assert pairs == {(0, 1)}
    zb = _zband_map(segments=[a, b, ((112, 141), (112, 159))])          # a short band across the gap
    ends, pairs, _ = _analyse(zb)
    ia, ib = _band_at(ends, 75, 150), _band_at(ends, 150, 150)
    assert tuple(sorted((ia, ib))) not in pairs


def test_two_ends_of_one_zband_are_never_linked():
    # a C-shaped band whose ends face each other across a 1-µm gap, and a second band elsewhere (n_z = 2: the old
    # same-band mask paired rows (2k, 2k + 1) of the Fortran-order end list, missed these ends and linked the band to
    # itself here)
    zb = _zband_map(segments=[((200, 60), (260, 60))], arcs=[((100, 150), 15, 20, 340)])
    ends, pairs, _ = _analyse(zb, max_axial_offset=None)
    assert len(ends) == 2
    assert all(i != j for i, j in pairs)
