"""Construction of the :class:`sarcasm.core.SarcAsM` recording: ``restart``, old stores, the summary.

``restart`` is keyword-only: the pre-1.0 manual-LOI workflow was entered as
``Motion(file_path, loi_name)``, and a truthy string in a positional ``restart`` slot would
have deleted the analysis store.
"""
import numpy as np
import pytest
import tifffile

from sarcasm import SarcAsM


# synthetic TIFFs carry no calibration; supply it so metadata validation passes
META = {"pixelsize": 0.1, "frametime": 0.01}


def _movie(tmp_path, name="movie.tif", n=4):
    """Write a small synthetic multi-frame TIFF and return its path."""
    p = tmp_path / name
    tifffile.imwrite(p, np.zeros((n, 32, 32), np.uint16), metadata=None, ome=False)
    return p


def _store_of(path):
    return path.with_suffix(".ome.zarr")


def test_restart_is_keyword_only(tmp_path):
    """A second positional argument (e.g. the pre-1.0 Motion(file, loi_name)) cannot become restart=True,
    which would delete the analysis store."""
    p = _movie(tmp_path)
    SarcAsM(p, **META)  # creates the sibling .ome.zarr store eagerly
    store = _store_of(p)
    before = sorted(f.name for f in store.rglob("*"))
    with pytest.raises(TypeError, match="positional"):
        SarcAsM(p, "loi_0.json")
    assert sorted(f.name for f in store.rglob("*")) == before


def test_restart_true_still_recreates_the_store(tmp_path):
    """restart=True replaces the store."""
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


def test_store_from_before_the_namespaces_warns(tmp_path, caplog, monkeypatch):
    """A store written by 1.0.0b4 or earlier names its params by the old steps (SARC-27)."""
    import logging
    p = _movie(tmp_path)
    sarc = SarcAsM(p, **META)
    sarc.data['params.analyze_sarcomere_vectors.frames'] = [0]
    sarc.commit()
    reopened = SarcAsM(p, **META)        # its constructor sets up logging, so capture afterwards
    monkeypatch.setattr(logging.getLogger('sarcasm'), 'propagate', True)
    with caplog.at_level(logging.WARNING, logger='sarcasm'):
        reopened._load_structure_data()
    assert 'SarcAsM 1.0.0b4 or earlier (analyze_sarcomere_vectors)' in caplog.text


def test_restart_leaves_the_folder_named_after_the_movie(tmp_path):
    """restart=True deletes only the store, not a ``<name>/`` folder beside the movie (SARC-26).

    That folder may be a pre-1.0 analysis, which the warning promises to leave in place, or an
    unrelated folder of the user's that happens to share the name.
    """
    p = _movie(tmp_path)
    folder = tmp_path / "movie"
    (folder / "data").mkdir(parents=True)
    (folder / "data" / "notes.txt").write_text("keep me")

    SarcAsM(p, restart=True, **META)

    assert (folder / "data" / "notes.txt").read_text() == "keep me"


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
        "params.structure.analyze_vectors.frames": [0, 1, 2, 3],
        "params.motion.track.frames": [0, 1, 2, 3],
        "structure.sarcomere.n_vectors": np.array([40.0, 41.0, 39.0, 40.0]),
        "motion.tracks.n": 12,
        "motion.groups.pool.track_group_id": np.zeros(12, int),
        "motion.groups.pool.n": 3,
    })
    sarc.commit()
    text = str(sarc)
    assert "detect_sarcomeres" in text and "motion.track" in text
    # pipeline order, not alphabetical
    assert text.index("detect_sarcomeres") < text.index("motion.track")
    assert "12 tracks" in text and "groups: 3 pool" in text
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
