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


# --------------------------------------------------------------------------- #
# object summary: repr / str
# --------------------------------------------------------------------------- #
def test_str_on_fresh_object_does_not_raise(tmp_path):
    """print(sarc) used to crash: __str__ read metadata.version, which never existed."""
    sarc = SarcAsM(_movie(tmp_path), **META)
    text = str(sarc)
    assert "movie.tif" in text
    assert "0.1 µm/px" in text
    assert "0 keys" in text
    assert "no analysis yet" in text


def test_str_on_analysed_object(tmp_path):
    """Steps and counts come from the manifest — no detection, no array reads."""
    sarc = SarcAsM(_movie(tmp_path), **META)
    sarc.data.update({
        "params.detect_sarcomeres.frames": [0, 1, 2, 3],
        "params.analyze_sarcomere_vectors.frames": [0, 1, 2, 3],
        "params.track_sarcomere_vectors.frames": [0, 1, 2, 3],
        "structure.sarcomere.n_vectors": np.array([40.0, 41.0, 39.0, 40.0]),
        "motion.tracks.n": 12,
        "motion.groups.n": 3,
        "motion.groups.kind": "pool",
    })
    sarc.commit()
    text = str(sarc)
    assert "detect_sarcomeres" in text and "track_sarcomere_vectors" in text
    # pipeline order, not alphabetical
    assert text.index("detect_sarcomeres") < text.index("track_sarcomere_vectors")
    assert "12 tracks" in text and "3 groups (pool)" in text
    assert "~40 vectors/frame" in text


def test_repr_is_one_line(tmp_path):
    sarc = SarcAsM(_movie(tmp_path), **META)
    text = repr(sarc)
    assert "\n" not in text
    assert text.startswith("<SarcAsM") and "movie.tif" in text


def test_repr_survives_partially_constructed_object():
    """pytest reprs objects on assertion failure; a raising __repr__ hides the diff."""
    assert repr(SarcAsM.__new__(SarcAsM))


def test_str_without_store(tmp_path):
    """The store can be removed underneath the object; printing must still work."""
    p = _movie(tmp_path)
    sarc = SarcAsM(p, **META)
    import shutil
    shutil.rmtree(_store_of(p))
    text = str(sarc)
    assert "not created" in text


def test_str_on_motion_object(tmp_path):
    p = _movie(tmp_path)
    m = Motion(p, **META)
    assert "movie.tif" in str(m)               # no loi_data yet

    t = np.arange(4) * 0.01
    m2 = Motion.from_loi_data(p, "loi_0", {
        "z_pos": np.zeros((3, 4)), "slen": np.full((2, 4), 1.8), "time": t,
    }, frametime=0.01)
    text = str(m2)
    assert "2 sarcomeres × 4 frames" in text
    assert "synthetic" in text


def test_results_property_removed():
    """1.0 has a single accessor; `sarc.results` must not come back."""
    assert not hasattr(SarcAsM, "results")


def test_data_key_is_its_path(tmp_path):
    """`sarc.data['a.b.c']` and `sarc.data.a.b.c` are one value, two spellings."""
    sarc = SarcAsM(_movie(tmp_path), **META)
    sarc.data["structure.sarcomere.oop"] = np.array([0.7, 0.8, 0.9])
    assert sarc.data["structure.sarcomere.oop"] is sarc.data.structure.sarcomere.oop
    assert dir(sarc.data.structure) == ["sarcomere"]
    # tab completion offers the namespaces that exist, not every key
    assert "structure" in dir(sarc.data)
    assert "motion" not in dir(sarc.data)          # nothing tracked yet
    sarc.data["motion.tracks.n"] = 12
    assert "motion" in dir(sarc.data)
