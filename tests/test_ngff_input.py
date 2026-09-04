"""SarcAsM reading a third-party OME-Zarr (NGFF) store as input.

The fixtures build small NGFF stores by hand rather than through a writer API, so the
assertions do not depend on which zarr/ome-zarr version happens to be installed, and so
a v0.5-style store (OME block nested under ``ome``) can be exercised alongside v0.4.

What matters here is the calibration: without it ``pixelsize`` is None and every analysis
has to be told the pixel size by hand — and pixel size is what the generalist model is
calibrated on.
"""
from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from sarcasm import SarcAsM
from sarcasm.exceptions import MetaDataError


def _write_ngff(path, shape, axes, scale, units, nested=False, ms_scale=None):
    """Minimal NGFF image group: one dataset '0' plus a multiscales block."""
    g = zarr.open_group(str(path), mode="w")
    g.create_array("0", shape=shape, dtype="uint16")
    g["0"][:] = (np.random.default_rng(0).random(shape) * 1000).astype("uint16")
    ms = {
        "version": "0.5" if nested else "0.4",
        "name": path.name,
        "axes": [{"name": a.lower(), "type": t, **({"unit": u} if u else {})}
                 for a, t, u in zip(axes, _axis_types(axes), units)],
        "datasets": [{"path": "0",
                      "coordinateTransformations": [{"type": "scale", "scale": list(scale)}]}],
    }
    if ms_scale is not None:
        ms["coordinateTransformations"] = [{"type": "scale", "scale": list(ms_scale)}]
    g.attrs["ome" if nested else "multiscales"] = ({"multiscales": [ms], "version": "0.5"}
                                                   if nested else [ms])
    return path


def _axis_types(axes):
    return ["time" if a == "T" else "channel" if a == "C" else "space" for a in axes]


@pytest.fixture
def tyx_store(tmp_path):
    return _write_ngff(tmp_path / "movie.ome.zarr", (4, 32, 48), "TYX",
                       [0.25, 0.114, 0.114], [None, "micrometer", "micrometer"])


def test_pixelsize_and_frametime_come_from_the_store(tyx_store):
    s = SarcAsM(str(tyx_store))
    assert s.metadata.pixelsize == pytest.approx(0.114)
    assert s.metadata.frametime == pytest.approx(0.25)
    assert s.metadata.axes == "TYX"
    assert s.metadata.n_stack == 4


def test_frame_selection_returns_one_frame(tyx_store):
    s = SarcAsM(str(tyx_store))
    assert np.asarray(s.read_imgs(frames=0)).shape == (32, 48)
    assert np.asarray(s.read_imgs()).shape == (4, 32, 48)


def test_constructor_arguments_win_over_store_metadata(tyx_store):
    s = SarcAsM(str(tyx_store), pixelsize=0.2, frametime=1.5)
    assert s.metadata.pixelsize == pytest.approx(0.2)
    assert s.metadata.frametime == pytest.approx(1.5)


def test_nanometers_are_converted(tmp_path):
    p = _write_ngff(tmp_path / "nm.ome.zarr", (2, 16, 16), "TYX",
                    [1.0, 114.0, 114.0], [None, "nanometer", "nanometer"])
    assert SarcAsM(str(p)).metadata.pixelsize == pytest.approx(0.114)


def test_milliseconds_are_converted(tmp_path):
    p = _write_ngff(tmp_path / "ms.ome.zarr", (2, 16, 16), "TYX",
                    [40.0, 0.114, 0.114], [ "millisecond", "micrometer", "micrometer"])
    assert SarcAsM(str(p)).metadata.frametime == pytest.approx(0.04)


def test_multiscales_level_scale_is_folded_in(tmp_path):
    # a global transform on the multiscales block multiplies the per-dataset one
    p = _write_ngff(tmp_path / "both.ome.zarr", (2, 16, 16), "TYX",
                    [1.0, 0.057, 0.057], [None, "micrometer", "micrometer"],
                    ms_scale=[1.0, 2.0, 2.0])
    assert SarcAsM(str(p)).metadata.pixelsize == pytest.approx(0.114)


def test_v05_nested_ome_attrs(tmp_path):
    p = _write_ngff(tmp_path / "v05.ome.zarr", (3, 16, 16), "TYX",
                    [0.5, 0.108, 0.108], [None, "micrometer", "micrometer"], nested=True)
    s = SarcAsM(str(p))
    assert s.metadata.pixelsize == pytest.approx(0.108)
    assert s.metadata.frametime == pytest.approx(0.5)


def test_no_time_axis_leaves_frametime_unset(tmp_path):
    p = _write_ngff(tmp_path / "yx.ome.zarr", (24, 24), "YX",
                    [0.16, 0.16], ["micrometer", "micrometer"])
    s = SarcAsM(str(p))
    assert s.metadata.pixelsize == pytest.approx(0.16)
    assert s.metadata.frametime is None


def test_anisotropic_xy_uses_the_mean(tmp_path, caplog):
    p = _write_ngff(tmp_path / "aniso.ome.zarr", (16, 16), "YX",
                    [0.10, 0.12], ["micrometer", "micrometer"])
    assert SarcAsM(str(p)).metadata.pixelsize == pytest.approx(0.11)


def test_a_group_without_multiscales_is_rejected(tmp_path):
    p = tmp_path / "bare.ome.zarr"
    g = zarr.open_group(str(p), mode="w")
    g.create_array("0", shape=(8, 8), dtype="uint16")
    with pytest.raises(MetaDataError, match="multiscales"):
        SarcAsM(str(p))


def test_an_hcs_plate_root_is_rejected_with_a_useful_message(tmp_path):
    p = tmp_path / "plate.ome.zarr"
    g = zarr.open_group(str(p), mode="w")
    g.attrs["plate"] = {"columns": [{"name": "1"}], "rows": [{"name": "A"}],
                        "wells": [{"path": "A/1"}]}
    with pytest.raises(MetaDataError, match="plate"):
        SarcAsM(str(p))


def test_tiff_input_is_unaffected(tmp_path):
    import tifffile
    f = tmp_path / "stack.tif"
    tifffile.imwrite(f, (np.random.default_rng(1).random((3, 24, 24)) * 500).astype("uint16"))
    s = SarcAsM(str(f), pixelsize=0.114, frametime=0.1, restart=True)
    assert s.metadata.pixelsize == pytest.approx(0.114)
    assert np.asarray(s.read_imgs(frames=0)).shape == (24, 24)
