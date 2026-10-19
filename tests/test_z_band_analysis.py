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
    ends = np.asarray(res['ends'], float) / PX                             # (n_z, 2 ends, row/col) px
    links = np.asarray(res['links'])
    pairs = {tuple(sorted((int(i), int(j)))) for i, _, j, _ in links.T} if links.size else set()
    return ends, pairs, np.asarray(res['lat_length_groups'], float)


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


def _cell_scene(px, cols_um=(15.0,), size_um=30.0):
    """Vertical Z-bands (10 µm long, 0.3 µm wide) at x = cols_um in a cell (5-25 µm square): Z-bands 300, cell
    pixels within 1.5 µm of a Z-band 150, farther cell pixels 400, medium outside the cell 100."""
    n = int(round(size_um / px))
    yy, xx = np.mgrid[0:n, 0:n] * px
    zb = np.zeros((n, n), bool)
    for c in cols_um:
        zb |= (np.abs(xx - c) <= 0.15) & (yy >= 10) & (yy <= 20)
    cell = (yy >= 5) & (yy <= 25) & (xx >= 5) & (xx <= 25)
    near = ndimage.distance_transform_edt(~zb) * px <= 1.5
    image = np.where(cell, np.where(near, 150.0, 400.0), 100.0)
    image[zb] = 300.0
    return zb.astype(np.float32), image, cell


def _contrast(px=PX, cols_um=(15.0,)):
    zbands, image, cell = _cell_scene(px, cols_um)
    field = np.stack([np.ones_like(zbands), np.zeros_like(zbands)])
    labels, skel = zba.segment_z_bands(zbands, threshold=0.5)
    res = zba.analyze_z_bands(zbands, labels, skel, image, field, pixelsize=px, **PARAMS,
                              background=zba.estimate_background(image, cell), cell_mask=cell)
    return np.asarray(res['contrast'], float)


def test_contrast_is_the_background_corrected_ratio_to_the_surrounding_pixels():
    np.testing.assert_allclose(_contrast(), [(300 - 100) / (150 - 100)])


def test_contrast_does_not_depend_on_the_pixel_size():
    np.testing.assert_allclose(_contrast(px=0.05), [4.0])


def test_neighbouring_zband_is_not_part_of_the_ring():
    np.testing.assert_allclose(_contrast(cols_um=(14.5, 15.5)), [4.0, 4.0])


def test_wider_ring_reaches_the_far_cytoplasm():
    zbands, image, _ = _cell_scene(PX)
    labels = (zbands > 0.5).astype(int)
    contrast, surround = zba.z_band_contrast(labels, zbands > 0.5, image, PX, 100.0, ring_width=2.0)
    assert contrast[0] < 3.0 and surround[0] > 50.0


def test_background_is_the_median_outside_the_cell():
    image = np.arange(100.0).reshape(10, 10)
    assert zba.estimate_background(image, image >= 50) == np.median(image[image < 50])
    assert np.isnan(zba.estimate_background(image, np.ones((10, 10), bool)))     # no free background
    assert np.isnan(zba.estimate_background(image, None))


def test_contrast_falls_back_to_a_low_percentile_without_background():
    zbands, image, _ = _cell_scene(PX)
    field = np.stack([np.ones_like(zbands), np.zeros_like(zbands)])
    labels, skel = zba.segment_z_bands(zbands, threshold=0.5)
    res = zba.analyze_z_bands(zbands, labels, skel, image, field, pixelsize=PX, **PARAMS, background=np.nan)
    assert res['background'] == np.percentile(image, 1)


def test_no_zbands_gives_an_empty_contrast():
    contrast, surround = zba.z_band_contrast(np.zeros((20, 20), int), np.zeros((20, 20), bool), np.ones((20, 20)),
                                             PX, 0.0)
    assert contrast.size == 0 and surround.size == 0


def test_ring_stays_inside_the_cell():
    # a Z-band 0.5 µm from the cell edge: its ring would reach the bright medium outside the cell
    zbands, image, cell = _cell_scene(PX, cols_um=(5.5,))
    image[~cell] = 600.0
    labels = (zbands > 0.5).astype(int)
    contrast, surround = zba.z_band_contrast(labels, zbands > 0.5, image, PX, 100.0, cell_mask=cell)
    np.testing.assert_allclose(surround, [50.0])
    np.testing.assert_allclose(contrast, [4.0])
