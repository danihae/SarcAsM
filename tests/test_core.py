"""Construction-time guards in :class:`sarcasm.core.SarcAsMBase`.

Regression coverage for the pre-1.0 migration hazard: the manual-LOI workflow was
entered as ``Motion(file_path, loi_name)``, and that positional slot now holds
``restart``. A LOI-name string is truthy, so without a guard the constructor would
silently take the ``restart=True`` path and delete the analysis store.
"""
import numpy as np
import pytest
import tifffile

from sarcasm import Motion, SarcAsM


# synthetic TIFFs carry no calibration; supply it so metadata validation passes
META = {"pixelsize": 0.1, "frametime": 0.01}


def _movie(tmp_path, name="movie.tif", n=4):
    """Write a small synthetic multi-frame TIFF and return its path."""
    p = tmp_path / name
    tifffile.imwrite(p, np.zeros((n, 32, 32), np.uint16), metadata=None, ome=False)
    return p


def _store_of(path):
    return path.with_suffix(".ome.zarr")


def test_loi_name_as_second_positional_raises(tmp_path):
    """The pre-1.0 ``Motion(file, loi_name)`` call must fail loudly, not silently."""
    p = _movie(tmp_path)
    with pytest.raises(TypeError) as exc:
        Motion(p, "loi_0.json")
    msg = str(exc.value)
    assert "restart" in msg
    # the message has to point at the replacement API, not just complain about a type
    assert "get_track_motion" in msg


def test_loi_name_does_not_delete_the_store(tmp_path):
    """The guard must fire before any destructive work."""
    p = _movie(tmp_path)
    SarcAsM(p, **META)  # creates the sibling .ome.zarr store eagerly
    store = _store_of(p)
    assert store.exists()
    before = sorted(f.name for f in store.rglob("*"))

    with pytest.raises(TypeError):
        Motion(p, "loi_0.json")

    assert store.exists(), "store was deleted by a rejected constructor call"
    assert sorted(f.name for f in store.rglob("*")) == before


@pytest.mark.parametrize("value", ["loi_0.json", None, ["a"], 1.5])
def test_non_bool_restart_rejected(tmp_path, value):
    p = _movie(tmp_path)
    with pytest.raises(TypeError, match="restart"):
        SarcAsM(p, value)


@pytest.mark.parametrize("value", [False, 0])
def test_falsy_bool_restart_accepted(tmp_path, value):
    """bool and int stay valid — the guard must not break ordinary usage."""
    p = _movie(tmp_path)
    sarc = SarcAsM(p, value, **META)
    assert sarc.restart == value


def test_restart_true_still_recreates_the_store(tmp_path):
    """Positive control: the behaviour the guard protects is itself unchanged."""
    p = _movie(tmp_path)
    SarcAsM(p, **META)
    store = _store_of(p)
    assert store.exists()
    marker = store / "sarcasm" / "_marker.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("stale")

    SarcAsM(p, restart=True, **META)

    assert store.exists(), "restart=True must leave a fresh store behind"
    assert not marker.exists(), "restart=True must clear the previous store contents"
