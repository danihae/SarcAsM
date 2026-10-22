"""Headless CLI (``sarcasm.cli``) and the input checks it relies on.

Synthetic TIFFs only: no shared test data, no model inference.
"""

import argparse
import json

import numpy as np
import pytest
import tifffile

from sarcasm import SarcAsM
from sarcasm.cli import main, parse_frames
from sarcasm.exceptions import MetaDataError


def _tif(path, shape, axes, pixelsize=0.1, **ij):
    data = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    tifffile.imwrite(path, data, imagej=True, resolution=(1 / pixelsize, 1 / pixelsize),
                     metadata={'axes': axes, 'unit': 'um', **ij})
    return str(path)


def _err(err):
    """The JSON error object: the last line of stderr, after any log lines."""
    return json.loads(err.strip().splitlines()[-1])


def _run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


# --------------------------------------------------------------------------- #
# frames parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('text, expected', [
    ('all', 'all'), ('3', [3]), ('4,0, 4', [0, 4]), ('2:5', [2, 3, 4]),
])
def test_parse_frames(text, expected):
    assert parse_frames(text) == expected


@pytest.mark.parametrize('text', ['5:2', 'a:3', '-1', '1.5'])
def test_parse_frames_rejects(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_frames(text)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def test_features_is_json_and_filterable(capsys):
    code, out, _ = _run(capsys, 'features', '--find', 'slen_mean')
    doc = json.loads(out)
    assert code == 0
    assert 'structure.sarcomere.slen_mean' in doc['structure']
    assert all('slen_mean' in k for k in doc['structure'])
    assert doc['warnings'] == []


def test_info_reports_calibration(tmp_path, capsys):
    f = _tif(tmp_path / 'movie.tif', (5, 64, 64), 'TYX', pixelsize=0.1, finterval=0.02)
    code, out, _ = _run(capsys, 'info', f)
    doc = json.loads(out)
    assert code == 0
    assert doc['axes'] == 'TYX' and doc['n_frames'] == 5
    assert doc['pixelsize_um'] == pytest.approx(0.1)
    assert doc['frametime_s'] == pytest.approx(0.02)


def test_missing_file_is_input_error(tmp_path, capsys):
    code, out, err = _run(capsys, 'info', str(tmp_path / 'nope.tif'))
    assert code == 3 and out == ''
    assert _err(err)['error'] == 'FileNotFoundError'


def test_motion_without_frametime_is_input_error(tmp_path, capsys):
    f = _tif(tmp_path / 'movie.tif', (5, 64, 64), 'TYX')
    code, out, err = _run(capsys, 'motion', f)
    assert code == 3 and out == ''
    assert '--frametime' in _err(err)['hint']


def test_results_suggests_close_keys(tmp_path, capsys):
    f = _tif(tmp_path / 'img.tif', (64, 64), 'YX')
    sarc = SarcAsM(f)
    sarc.data['structure.sarcomere.slen_mean'] = np.array([1.8])
    sarc.commit()
    code, out, _ = _run(capsys, 'results', f, 'structure.sarcomere.slen_mean')
    assert code == 0
    assert json.loads(out)['values']['structure.sarcomere.slen_mean'] == [1.8]
    code, _, err = _run(capsys, 'results', f, 'structure.sarcomere.slen_maen')
    assert code == 3
    assert 'structure.sarcomere.slen_mean' in _err(err)['hint']


# --------------------------------------------------------------------------- #
# input checks in SarcAsMBase
# --------------------------------------------------------------------------- #
def test_lfs_pointer_is_rejected(tmp_path):
    f = tmp_path / 'pointer.tif'
    f.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n')
    with pytest.raises(ValueError, match='Git LFS pointer'):
        SarcAsM(str(f))


def test_many_channels_mean_mislabelled_axis(tmp_path):
    f = _tif(tmp_path / 'movie.tif', (30, 64, 64), 'CYX')
    with pytest.raises(MetaDataError, match='30 channels'):
        SarcAsM(f)
    assert SarcAsM(f, axes='TYX').metadata.n_stack == 30


def test_explicit_channel_allows_many_channels(tmp_path):
    f = _tif(tmp_path / 'spectral.tif', (8, 64, 64), 'CYX')
    assert SarcAsM(f, channel=2).metadata.channel == 2


def test_conflicting_calibration_on_reopen_raises(tmp_path):
    f = _tif(tmp_path / 'img.tif', (64, 64), 'YX', pixelsize=0.1)
    SarcAsM(f)
    SarcAsM(f, pixelsize=0.1)                       # same value: fine
    with pytest.raises(MetaDataError, match='restart=True'):
        SarcAsM(f, pixelsize=0.2)
    assert SarcAsM(f, pixelsize=0.2, restart=True).metadata.pixelsize == pytest.approx(0.2)


def test_missing_frametime_is_filled_on_reopen(tmp_path):
    f = _tif(tmp_path / 'movie.tif', (5, 64, 64), 'TYX')
    assert SarcAsM(f).metadata.frametime is None
    assert SarcAsM(f, frametime=0.05).metadata.frametime == pytest.approx(0.05)
    sarc = SarcAsM(f)
    assert sarc.metadata.frametime == pytest.approx(0.05)
    assert sarc.metadata.time[-1] == pytest.approx(0.2)
