"""Tests for the track-vector overlay (Plots.plot_track_vectors) and the movie API (Animations)."""
from __future__ import annotations

import os
import types
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from sarcasm import Animations, Plots, SarcAsM, Utils  # noqa: E402

STORE_TIF = Path(__file__).parent.parent / 'test_data' / 'high_speed_single_ACTN2-citrine_CM' / '20kPa.tif'


def _store_available(path: Path) -> bool:
    """False without the data: a clone without ``git lfs pull`` (CI) leaves a pointer file in place of the tif,
    so ``exists()`` alone would hand the pointer to tifffile."""
    if not path.exists():
        return False
    with open(path, 'rb') as fh:
        return not fh.read(64).startswith(b'version https://git-lfs')


needs_store = pytest.mark.skipif(not _store_available(STORE_TIF), reason='tracked 20kPa test store not available')


def _fake_tracks(n_tracks=5, T=10, slen=1.8, pixelsize=0.1, theta=0.0):
    """A SarcAsM with a synthetic tracker output (no file IO), vectors at orientation ``theta``."""
    sarc = SarcAsM.__new__(SarcAsM)
    sarc.auto_save = False
    sarc.metadata = types.SimpleNamespace(frametime=0.02, pixelsize=pixelsize, n_stack=T, size=(64, 128))
    sarc.model_dir = Utils.get_models_dir()
    pos = np.zeros((n_tracks, T, 2), np.float32)
    pos[:, :, 0] = 30.0
    pos[:, :, 1] = (20 + 20 * np.arange(n_tracks))[:, None]
    observed = np.ones((n_tracks, T), bool)
    observed[0, 3] = False  # one gap frame
    sarc.data = {
        'motion.tracks.n': n_tracks,
        'motion.tracks.ids': np.arange(n_tracks),
        'motion.tracks.start_frame': np.zeros(n_tracks, int),
        'motion.tracks.n_frames': observed.sum(axis=1),
        'motion.tracks.slen': np.full((n_tracks, T), slen, np.float32),
        'motion.tracks.positions_px': pos,
        'motion.tracks.positions_um': pos * pixelsize,
        'motion.tracks.observed': observed,
        'motion.tracks.orientations': np.full((n_tracks, T), theta, np.float32),
        'params.track_sarcomere_vectors.frames': list(range(T)),
    }
    return sarc


def test_track_vector_segments_geometry():
    sarc = _fake_tracks(theta=0.0)
    seg = Plots.track_vector_segments(sarc, frame=0)
    assert seg.shape == (5, 2, 2)
    # orientation 0: the line runs along x, centred on the track position, length slen / pixelsize
    length_px = 1.8 / 0.1
    np.testing.assert_allclose(seg[:, 1, 0] - seg[:, 0, 0], length_px)
    np.testing.assert_allclose(seg[:, 1, 1], seg[:, 0, 1])
    np.testing.assert_allclose(seg[:, :, 0].mean(axis=1), 20 + 20 * np.arange(5))
    # orientation pi/2: the line runs along y (image rows)
    seg90 = Plots.track_vector_segments(_fake_tracks(theta=np.pi / 2), frame=0)
    np.testing.assert_allclose(seg90[:, 1, 1] - seg90[:, 0, 1], length_px, atol=1e-5)
    np.testing.assert_allclose(seg90[:, 1, 0], seg90[:, 0, 0], atol=1e-5)


def test_track_vector_mask_hides_gap_frames():
    sarc = _fake_tracks()
    assert Plots.track_vector_mask(sarc, frame=3).sum() == 4
    assert Plots.track_vector_mask(sarc, frame=3, only_observed=False).sum() == 5


def test_plot_track_vectors_returns_collection():
    sarc = _fake_tracks()
    fig, ax = plt.subplots()
    lc = Plots.plot_track_vectors(ax, sarc, frame=0, color_by='slen', scalebar=False)
    assert isinstance(lc, LineCollection)
    assert len(lc.get_segments()) == 5
    np.testing.assert_allclose(lc.get_array(), 1.8)
    lc.set_segments(Plots.track_vector_segments(sarc, 3))
    assert len(lc.get_segments()) == 4
    plt.close(fig)


def test_resolve_frames_and_fps():
    sarc = _fake_tracks(T=10)
    np.testing.assert_array_equal(Animations.resolve_frames(sarc, None), np.arange(10))
    np.testing.assert_array_equal(Animations.resolve_frames(sarc, (0.0, 0.06)), [0, 1, 2, 3])
    np.testing.assert_array_equal(Animations.resolve_frames(sarc, [2, 5, 40]), [2, 5])
    assert Animations.playback_fps(sarc) == pytest.approx(50.0)
    assert Animations.playback_fps(sarc, fps=25, slow_motion=2.5) == pytest.approx(10.0)


def test_renderers_map_tracked_positions_to_movie_frames():
    """A tracking that starts at movie frame 3: the renderers are driven by positions in the tracked frame
    list, so position p must show movie frame 3 + p (it used to read movie frame p and fail on the tracks)."""
    from sarcasm.plotting.animation import _ImageLayer, _TrackVectorLayer
    T, start = 10, 3
    sarc = _fake_tracks(T=T)
    sarc.metadata.n_stack = start + T
    sarc.data['params.track_sarcomere_vectors.frames'] = list(range(start, start + T))
    sarc.read_imgs = lambda frames: (np.full((64, 128), frames, float) if np.ndim(frames) == 0
                                     else np.stack([np.full((64, 128), f, float) for f in frames]))
    image = _ImageLayer(sarc, np.arange(T))
    assert image.read(0)[0, 0] == start
    image.preload()
    assert image.read(2)[0, 0] == start + 2
    fig, ax = plt.subplots()
    vectors = _TrackVectorLayer(sarc, colorbar=False)
    vectors.setup(ax, 0)
    vectors.update(3)                        # position 3 holds the gap of track 0
    assert len(vectors.lc.get_segments()) == 4
    plt.close(fig)


@pytest.fixture(scope='module')
def tracked_store():
    """The tracked 20 kPa test store (tracks, pooled analysis, M-band grouping); read only."""
    return SarcAsM(str(STORE_TIF))


@needs_store
def test_animate_image_gif(tracked_store, tmp_path):
    out = Animations.animate_image(tracked_store, str(tmp_path / 'raw.gif'), frames=[0, 1, 2], dpi=50)
    assert os.path.getsize(out) > 1000


@needs_store
def test_animate_track_vectors_gif(tracked_store, tmp_path):
    out = Animations.animate_track_vectors(tracked_store, str(tmp_path / 'dsl.gif'), color_by='delta_slen',
                                           frames=[0, 1, 2], background='zbands', dpi=50)
    assert os.path.getsize(out) > 1000


@needs_store
def test_animate_results_gif(tracked_store, tmp_path):
    out = Animations.animate_results(tracked_store, str(tmp_path / 'results.gif'), frames=[0, 1, 2],
                                     timeseries=('pool', 'mband:auto'), dpi=50, snapshot=True)
    assert os.path.getsize(out) > 1000
    assert (tmp_path / 'results_frame0.png').exists()


@needs_store
def test_auto_group_members_fully_observed(tracked_store):
    frames = np.arange(20)
    g = Animations.auto_group(tracked_store, 'mband', frames)
    kind, gid, members = Animations._parse_group(tracked_store, f'mband:{g}', frames)
    obs = np.asarray(tracked_store.data['motion.tracks.observed'])[members][:, frames]
    assert kind == "mband" and gid == g and members.size >= 8 and obs.mean(axis=1).min() >= 0.8
