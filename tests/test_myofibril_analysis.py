"""Unit tests for the myofibril line growth in :mod:`sarcasm.analysis.myofibril_analysis`."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
from scipy.spatial import cKDTree

from sarcasm.analysis import myofibril_analysis as ma

PX = 0.1


def _grow(seed, pts, slen, theta, threshold_distance=0.5, persistence=1):
    return ma.grow_line(seed, pts, slen, theta, cKDTree(pts), threshold_distance, PX, persistence)


def _grow_in_subprocess(q, *args):
    q.put(_grow(*args).tolist())


def _grow_with_timeout(*args, timeout=20.0):
    """Run grow_line in a child process so that a growth that never terminates fails instead of hanging the suite."""
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_grow_in_subprocess, args=(q, *args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        raise AssertionError('grow_line did not terminate')
    return np.asarray(q.get(timeout=5))


def test_grow_line_follows_a_straight_chain():
    """Points one sarcomere apart along the axis are collected in order, each once.

    The growth step from a point is ``(sin θ, cos θ)`` in (row, col) times the sarcomere length,
    so θ = π/2 walks along rows."""
    n = 12
    pts = np.stack([np.arange(n) * 10.0, np.full(n, 50.0)], axis=1)      # rows 0, 10, ..., 110; one column
    slen = np.full(n, 10 * PX)
    theta = np.full(n, np.pi / 2)
    line = _grow_with_timeout(5, pts, slen, theta)
    assert line.tolist() == list(range(n))


def _hexagon():
    """Six points on a regular hexagon (side 10 px), each oriented towards the next corner, so that
    growth walks round the ring: the old loop never terminated on such a cycle."""
    k = np.arange(6)
    pts = np.stack([10 * np.sin(np.pi / 3 * k), 10 * np.cos(np.pi / 3 * k)], axis=1)
    step = np.roll(pts, -1, axis=0) - pts
    theta = np.arctan2(step[:, 0], step[:, 1])          # growth step is (sin θ, cos θ) in (row, col)
    return pts, np.full(6, 10 * PX), theta


def test_grow_line_stops_instead_of_cycling():
    """A closed ring of points sent the old growth round and round forever (this hung 4 of 864
    movies of the chronic plate). A line never passes a point twice: growth stops when the next
    point is already part of it."""
    pts, slen, theta = _hexagon()
    line = _grow_with_timeout(0, pts, slen, theta)
    assert len(line) == len(set(line.tolist())), 'a point appears twice in the line'
    assert sorted(line.tolist()) == list(range(6))


def test_line_growth_returns_unique_point_lines():
    """The same ring through the public entry point, every point as a seed."""
    pts, slen, theta = _hexagon()
    out = ma.line_growth(pts, slen, theta, np.full(6, 1.0), PX, ratio_seeds=1.0, persistence=1,
                         threshold_distance=0.5, n_min=6, random_seed=0)
    assert len(out['lines']) == 6
    for line in out['lines']:
        assert len(line) == len(set(line.tolist())) == 6


def test_line_growth_is_reproducible_with_a_seed_and_leaves_the_global_random_state_alone():
    """The start vectors are drawn at random: a seed gives the same lines (and myofibril groups) every run,
    without seeding Python's global random generator for the caller."""
    import random
    pts = np.column_stack([np.repeat(np.arange(10) * 19.0, 10), np.tile(np.arange(10) * 19.0, 10)])
    slen, theta = np.full(100, 1.9), np.zeros(100)

    def lines(seed):
        out = ma.line_growth(pts.T, slen, theta, np.full(100, 1.0), PX, ratio_seeds=0.2, persistence=1,
                             threshold_distance=0.5, n_min=2, random_seed=seed)
        return [tuple(line.tolist()) for line in out['lines']]

    random.seed(123)
    expected_next = random.random()
    random.seed(123)
    first = lines(42)
    assert random.random() == expected_next          # the caller's random sequence is untouched
    assert len(first) > 0 and lines(42) == first


class TestLoiClustering:
    """LOI clustering works on Hausdorff distances in µm, so the result does not depend on the pixel size."""

    @staticmethod
    def _lines(rng, n=12):
        return [np.cumsum(rng.normal(0, 1, (rng.integers(3, 30), 2)), axis=0) + rng.uniform(0, 40, 2)
                for _ in range(n)]

    def test_hausdorff_matrix_matches_scipy(self):
        from scipy.spatial.distance import directed_hausdorff
        from sarcasm.analysis.loi_detection import hausdorff_distance_lois
        lines = self._lines(np.random.default_rng(0))
        for mode, combine in (('max', max), ('min', min)):
            got = hausdorff_distance_lois(lines, symmetry_mode=mode)
            want = np.array([[combine(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]) for b in lines]
                             for a in lines])
            np.testing.assert_allclose(got, want, atol=1e-12)

    def test_clusters_do_not_depend_on_pixel_size(self):
        from sarcasm.analysis.loi_detection import cluster_lois, hausdorff_distance_lois
        lines_um = self._lines(np.random.default_rng(1), n=20)
        labels = []
        for pixelsize in (0.06, 0.26):
            lines_px = [v / pixelsize for v in lines_um]
            # what SarcAsM._hausdorff_distance_lois does: positions in px times the pixel size
            h = hausdorff_distance_lois([v * pixelsize for v in lines_px])
            labels.append(cluster_lois(h, distance_threshold=2.5)[0])
        np.testing.assert_array_equal(labels[0], labels[1])
