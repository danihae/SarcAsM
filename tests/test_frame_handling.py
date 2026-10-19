"""Frame selection is spelling-independent and results stay paired with their own frame.

``detect_sarcomeres(frames=[k])`` used to store ``(1, H, W)`` masks while ``frames=k`` stored ``(H, W)``;
``_read_mask`` then returned ``(1, 1, H, W)`` and ``analyze_cell_mask`` crashed. Around it: an explicit frame
number after a single-frame detection was filed under position 0 (so ``analyze_myofibrils(frames=k)`` failed),
ratios divided by the cell area of another frame, the fast-movie Z-bands of frame 0 were paired with the
M-bands of the detected frame, invalid frames ran silently or failed deep inside zarr, and the app's frame
field silently fell back to frame 0. These tests pin the behaviour of the shared frame helpers.

Every result is filed under its frame number.
"""
import shutil

import matplotlib
import numpy as np
import pytest
import tifffile

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from sarcasm import SarcAsM, Plots  # noqa: E402
from sarcasm._internal.frames import canonical_frame, canonical_frames, parse_frames_text  # noqa: E402

SINGLE_FRAME_SPELLINGS = [1, np.int64(1), [1], (1,), range(1, 2), np.array([1]), np.array(1)]
DETECT = dict(normalization_mode='single', max_patch_size=(512, 512))   # frame-independent U-Net input


# --------------------------------------------------------------------------------------------- pure helpers
class TestCanonicalFrames:

    @pytest.mark.parametrize('frames', SINGLE_FRAME_SPELLINGS)
    def test_every_single_frame_spelling_gives_one_list(self, frames):
        assert canonical_frames(frames, 4) == [1]
        assert canonical_frame(frames, 4) == 1

    @pytest.mark.parametrize('frames, expected', [
        (None, 'all'), ('all', 'all'), (' ALL ', 'all'),
        ([3, 1], [1, 3]), ([1, 3, 3], [1, 3]), (np.array([[0, 2]]), [0, 2]), ((2, 0), [0, 2]),
        (range(0, 4), [0, 1, 2, 3]), (np.arange(2, dtype=np.uint8), [0, 1]),
    ])
    def test_canonical_form(self, frames, expected):
        result = canonical_frames(frames, 4)
        assert result == expected
        if isinstance(result, list):
            assert all(type(f) is int for f in result)

    @pytest.mark.parametrize('frames, error, match', [
        (4, ValueError, r'out of range for a movie of 4 frames'),
        ([0, 4], ValueError, r'\[4\] out of range'),
        (-1, ValueError, r'out of range'),
        ([], ValueError, 'empty'),
        (1.0, TypeError, 'integers'),
        (np.float64(1), TypeError, 'integers'),
        (True, TypeError, 'bool'),
        ([0, True], TypeError, 'bool'),
        ('1', TypeError, "'all'"),
        ([[0, 1]], TypeError, 'integers'),
    ])
    def test_invalid_frames_are_refused_clearly(self, frames, error, match):
        with pytest.raises(error, match=match):
            canonical_frames(frames, 4)

    def test_a_single_image_has_frame_0_only(self):
        assert canonical_frames(0, 1) == [0]
        assert canonical_frames(None, None) == 'all'
        with pytest.raises(ValueError, match='single image'):
            canonical_frames([1], 1)

    def test_one_frame_means_one_frame(self):
        with pytest.raises(ValueError, match='one frame'):
            canonical_frame([0, 1], 4)
        for bad in (None, 'all'):
            with pytest.raises(TypeError):
                canonical_frame(bad, 4)


class TestParseFramesText:
    """The app's frame field: invalid input used to fall back to frame 0 and '1, 3' dropped frame 3."""

    @pytest.mark.parametrize('text, expected', [
        ('all', 'all'), (' ALL ', 'all'), ('', None), ('   ', None),
        ('1', [1]), (' 1', [1]), ('1,', [1]), ('1, 3', [1, 3]), ('3,1', [1, 3]), ('1,1', [1]),
        ('0 4 9', [0, 4, 9]), ('0;4', [0, 4]),
    ])
    def test_valid(self, text, expected):
        assert parse_frames_text(text) == expected

    @pytest.mark.parametrize('text', ['a', '-1', '1-3', '1.5', '0,2,x', ',', '٣'])
    def test_invalid_is_an_error_not_frame_0(self, text):
        with pytest.raises(ValueError):
            parse_frames_text(text)


# ---------------------------------------------------------------------------------- store level (no U-Net)
@pytest.fixture
def synthetic_movie(tmp_path):
    """A 3-frame 32x32 movie whose masks are written by hand (no prediction)."""
    rng = np.random.default_rng(0)
    path = tmp_path / 'synthetic.tif'
    tifffile.imwrite(path, rng.integers(0, 1000, (3, 32, 32), dtype=np.uint16), imagej=True,
                     resolution=(10, 10), metadata={'axes': 'TYX', 'unit': 'um'})
    sarc = SarcAsM(str(path), log_level='ERROR')
    sarc.read_imgs()
    return sarc


class TestReadMaskIsRankAware:
    """Masks of one detected frame may be stored with a frame axis ((1, H, W), the [k] spelling and every new
    store) or without ((H, W), stores written by frames=k before the fix); reads must not depend on it."""

    @pytest.mark.parametrize('paged', [True, False])
    def test_single_detected_frame(self, synthetic_movie, paged):
        sarc = synthetic_movie
        z = np.full((32, 32), 7, np.float32)
        o = np.stack([np.ones((32, 32)), np.zeros((32, 32))]).astype(np.float32)
        sarc.store.write_mask('zbands', z[None] if paged else z)
        sarc.store.write_mask('orientation', o[None] if paged else o)
        sarc.data.update({'params.detect_sarcomeres.frames': [2]})
        for spelling in (2, np.int64(2), np.array(2)):
            assert sarc._read_mask('zbands', frames=spelling).shape == (32, 32)
            assert sarc._read_mask('orientation', frames=spelling).shape == (2, 32, 32)
        for spelling in ([2], (2,), range(2, 3), np.array([2])):
            assert sarc._read_mask('zbands', frames=spelling).shape == (1, 32, 32)
            assert sarc._read_mask('orientation', frames=spelling).shape == (1, 2, 32, 32)
        # frame 0 was not detected: no position reading
        for frame in (0, 1):
            with pytest.raises(ValueError, match=r'not detected \(detected frames: \[2\]\)'):
                sarc._read_mask('zbands', frames=frame)
        with pytest.raises(ValueError, match='not detected'):
            sarc._read_mask('zbands', frames=[0, 2])

    def test_partial_detection_maps_frames_to_pages(self, synthetic_movie):
        sarc = synthetic_movie
        pages = np.stack([np.full((32, 32), v, np.float32) for v in (10, 30)])
        sarc.store.write_mask('zbands', pages)
        sarc.data.update({'params.detect_sarcomeres.frames': [1, 2]})
        assert sarc._read_mask('zbands', frames=2)[0, 0] == 30
        assert list(sarc._read_mask('zbands', frames=[2, 1])[:, 0, 0]) == [30, 10]
        with pytest.raises(ValueError, match='not detected'):
            sarc._read_mask('zbands', frames=0)

    def test_load_mask_full_stack_places_every_rank_at_its_frame(self, synthetic_movie):
        sarc = synthetic_movie
        sarc.store.write_mask('zbands', np.ones((1, 32, 32), np.float32))
        sarc.store.write_mask('zbands_fast_movie', np.ones((3, 32, 32), np.float32))
        sarc.data.update({'params.detect_sarcomeres.frames': [2]})
        full = sarc.load_mask_full_stack('zbands')
        assert full.shape == (3, 32, 32) and full[2].all() and not full[:2].any()
        # the fast-movie Z-bands cover every frame: no (n, n, H, W) blow-up after a single-frame detection
        assert sarc.load_mask_full_stack('zbands_fast_movie').shape == (3, 32, 32)

    def test_read_imgs_of_a_single_image_refuses_other_frames(self, tmp_path):
        path = tmp_path / 'single.tif'
        tifffile.imwrite(path, np.zeros((16, 16), np.uint16), imagej=True, resolution=(10, 10),
                         metadata={'unit': 'um'})
        sarc = SarcAsM(str(path), log_level='ERROR')
        assert sarc.read_imgs(frames=0).shape == (16, 16)
        assert sarc.read_imgs(frames=np.array(0)).shape == (16, 16)
        with pytest.raises(ValueError, match='single image'):
            sarc.read_imgs(frames=1)


# ------------------------------------------------------------------------------------- pipeline (U-Net)
@pytest.fixture(scope='class')
def crop_copy(structure_crop_file_path_class, tmp_path_factory):
    """Factory: a fresh copy of the 2-frame crop in its own directory (its store does not touch test_data)."""
    def make(name):
        d = tmp_path_factory.mktemp(name)
        dst = d / 'crop.tif'
        shutil.copyfile(structure_crop_file_path_class, dst)
        return str(dst)
    return make


def _run_chain(sarc, frames):
    sarc.analyze_z_bands(frames=frames)
    sarc.analyze_sarcomere_vectors(frames=frames)
    sarc.analyze_myofibrils(frames=frames)
    sarc.analyze_sarcomere_domains(frames=frames)


def _finite_at(sarc, key):
    return [i for i, v in enumerate(np.asarray(sarc.data[key], dtype=float)) if np.isfinite(v)]


class TestSingleFrameDetection:
    """detect_sarcomeres on one frame of a time-lapse, in every spelling."""

    @pytest.mark.parametrize('frames', [[1], np.array(1), 1], ids=['list', '0d-array', 'int'])
    def test_detection_stores_one_page_and_analyzes_the_cell_mask(self, crop_copy, frames):
        sarc = SarcAsM(crop_copy('single_' + type(frames).__name__), log_level='ERROR')
        sarc.detect_sarcomeres(frames=frames, **DETECT)                    # used to raise IndexError for [1]
        assert sarc.store.read_mask('zbands').shape == (1, 512, 512)
        assert sarc.store.read_mask('orientation').shape == (1, 2, 512, 512)
        assert list(sarc.data['params.detect_sarcomeres.frames']) == [1]
        assert _finite_at(sarc, 'structure.cell.mask_area') == [1]

    @pytest.fixture(scope='class')
    def by_number(self, crop_copy):
        """The app/batch flow: the same frame number passed to every step."""
        sarc = SarcAsM(crop_copy('by_number'), log_level='ERROR')
        sarc.detect_sarcomeres(frames=[1], **DETECT)
        _run_chain(sarc, [1])                                     # analyze_myofibrils(frames=1) used to raise
        return sarc

    def test_every_step_files_the_frame_under_its_number(self, by_number):
        for step in ('analyze_cell_mask', 'analyze_z_bands', 'analyze_sarcomere_vectors', 'analyze_myofibrils',
                     'analyze_sarcomere_domains'):
            assert list(by_number.data[f'params.{step}.frames']) == [1], step
        for key in ('structure.zbands.n', 'structure.sarcomere.n_vectors', 'structure.domain.n'):
            assert _finite_at(by_number, key) == [1], key

    def test_ratios_use_the_cell_area_of_the_same_frame(self, by_number):
        d = by_number.data
        cell_area = d['structure.cell.mask_area'][1]
        assert np.isclose(d['structure.zbands.mask_area_ratio'][1], d['structure.zbands.mask_area'][1] / cell_area)
        assert np.isclose(d['structure.sarcomere.area_ratio'][1], d['structure.sarcomere.area'][1] / cell_area)

    @pytest.mark.parametrize('frames', [np.int64(1), (1,), range(1, 2), np.array([1])])
    def test_other_spellings_give_identical_results(self, crop_copy, by_number, frames):
        sarc = SarcAsM(crop_copy('spelling'), log_level='ERROR')
        sarc.detect_sarcomeres(frames=frames, **DETECT)
        sarc.analyze_z_bands(frames=frames)
        sarc.analyze_sarcomere_vectors(frames=frames)
        for key in ('structure.cell.mask_intensity', 'structure.zbands.n', 'structure.zbands.intensity_mean',
                    'structure.sarcomere.n_vectors', 'structure.sarcomere.slen_mean'):
            np.testing.assert_array_equal(np.asarray(sarc.data[key]), np.asarray(by_number.data[key]), key)

    @pytest.mark.parametrize('frame', [1, np.int64(1), [1], np.array(1)])
    def test_plots_accept_every_spelling(self, by_number, frame):
        for plot in (Plots.plot_z_bands, Plots.plot_cell_mask, Plots.plot_z_segmentation,
                     Plots.plot_sarcomere_vectors, Plots.plot_sarcomere_domains, Plots.plot_myofibril_lines):
            fig, ax = plt.subplots()
            plot(ax, by_number, frame=frame)
            plt.close(fig)

    def test_all_after_a_single_frame_detection_files_the_frame_number(self, crop_copy, by_number):
        """frames='all' analyzes the detected frame and files it under its number, like an explicit frame;
        frame 0 (the old position reading) is refused because it was not detected."""
        sarc = SarcAsM(crop_copy('all_single'), log_level='ERROR')
        sarc.detect_sarcomeres(frames=1, **DETECT)
        _run_chain(sarc, 'all')
        for step in ('analyze_z_bands', 'analyze_sarcomere_vectors', 'analyze_myofibrils',
                     'analyze_sarcomere_domains'):
            assert list(sarc.data[f'params.{step}.frames']) == [1], step
        for key in ('structure.zbands.intensity_mean', 'structure.sarcomere.n_vectors', 'structure.sarcomere.area_ratio'):
            np.testing.assert_array_equal(np.asarray(sarc.data[key]), np.asarray(by_number.data[key]), key)
        for call in (sarc.analyze_cell_mask, sarc.analyze_z_bands, sarc.analyze_sarcomere_vectors):
            with pytest.raises(ValueError, match=r'not detected.*\[1\]'):
                call(frames=0)

    def test_errors_name_the_problem(self, crop_copy):
        sarc = SarcAsM(crop_copy('errors'), log_level='ERROR')
        with pytest.raises(ValueError, match='out of range'):
            sarc.detect_sarcomeres(frames=2)
        with pytest.raises(TypeError, match='integers'):
            sarc.detect_sarcomeres(frames=1.0)
        with pytest.raises(ValueError, match='empty'):
            sarc.detect_sarcomeres(frames=[])
        sarc.detect_sarcomeres(frames=1, **DETECT)
        with pytest.raises(ValueError, match=r'not detected.*\[1\]'):
            sarc.analyze_z_bands(frames=[0, 1])


class TestPartialDetectionOfAMovie:
    """Several frames of a 3-frame movie (crop frames 0, 1 and frame 0 mirrored)."""

    @pytest.fixture(scope='class')
    def movie3(self, structure_crop_file_path_class, tmp_path_factory):
        with tifffile.TiffFile(structure_crop_file_path_class) as tif:
            crop = tif.series[0].asarray()
            xres = tif.pages[0].tags['XResolution'].value
        px = xres[1] / xres[0]
        path = tmp_path_factory.mktemp('movie3') / 'movie3.tif'
        tifffile.imwrite(path, np.stack([crop[0], crop[1], crop[0][:, ::-1]]), imagej=True,
                         resolution=(1 / px, 1 / px), metadata={'axes': 'TYX', 'unit': 'um'})
        return str(path)

    def test_unsorted_duplicate_selection_is_canonical(self, movie3):
        sarc = SarcAsM(movie3, restart=True, log_level='ERROR')
        sarc.detect_sarcomeres(frames=[2, 1, 2], **DETECT)
        assert list(sarc.data['params.detect_sarcomeres.frames']) == [1, 2]
        assert sarc.store.read_mask('zbands').shape == (2, 512, 512)
        # 'all' = the detected frames, filed under their numbers
        sarc.analyze_z_bands(frames='all')
        sarc.analyze_sarcomere_vectors(frames='all')
        d = sarc.data
        assert list(d['params.analyze_z_bands.frames']) == [1, 2]
        for frame in [1, 2]:
            cell_area = d['structure.cell.mask_area'][frame]
            assert np.isclose(d['structure.zbands.mask_area_ratio'][frame], d['structure.zbands.mask_area'][frame] / cell_area)
            assert np.isclose(d['structure.sarcomere.area_ratio'][frame], d['structure.sarcomere.area'][frame] / cell_area)
        # explicit frame numbers are filed under their numbers
        sarc.analyze_sarcomere_vectors(frames=[2, 1])
        assert list(d['params.analyze_sarcomere_vectors.frames']) == [1, 2]
        sarc.analyze_myofibrils(frames=2)
        assert list(d['params.analyze_myofibrils.frames']) == [2]
        with pytest.raises(ValueError, match='not detected'):
            sarc.analyze_z_bands(frames=0)


def test_fast_movie_z_bands_are_paired_with_their_own_frame(crop_copy):
    """analyze_sarcomere_vectors after a single-frame detection used to combine the fast-movie Z-bands of
    frame 0 with the M-bands of the detected frame."""
    from sarcasm.analysis import sarcomere_vectors as sv
    sarc = SarcAsM(crop_copy('fast_movie'), log_level='ERROR')
    sarc.detect_z_bands_fast_movie()
    sarc.detect_sarcomeres(frames=1, **DETECT)
    sarc.analyze_sarcomere_vectors(frames='all')
    z = sarc._read_mask('zbands_fast_movie', frames=1)
    m = sarc._read_mask('mbands', frames=1) > 0.25
    o = sarc._read_mask('orientation', frames=1)
    out = sv.get_sarcomere_vectors(z, m, o, pixelsize=sarc.metadata.pixelsize, interpolation_method='akima')
    keep = sv.filter_vectors_by_neighbors(out[1], out[5], neighbor_radius=1.0, max_neighbor_deviation=30.0)
    assert _finite_at(sarc, 'structure.sarcomere.n_vectors') == [1]
    assert sarc.data['structure.sarcomere.n_vectors'][1] == int(np.sum(keep))
    assert sarc.load_mask_full_stack('zbands_fast_movie').shape == (2, 512, 512)
