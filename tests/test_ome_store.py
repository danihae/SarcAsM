# -*- coding: utf-8 -*-
"""Tests for the single-file OME-Zarr analysis container (sarcasm.io.ome_store)."""
from __future__ import annotations

import numpy as np
import pytest
import zarr

from sarcasm.io.ome_store import (
    OmeZarrStore,
    detect_legacy_layout,
    legacy_layout_message,
    remove_tree,
    store_path_for,
)


def _img():
    rng = np.random.default_rng(0)
    return (rng.random((5, 48, 64)) * 1000).astype(np.uint16)  # (T, Y, X)


# --------------------------------------------------------------------------- #
# path mapping + legacy detection
# --------------------------------------------------------------------------- #
def test_store_path_for():
    assert store_path_for("/x/movie.tif").name == "movie.ome.zarr"
    assert store_path_for("/x/movie.ome.tif").name == "movie.ome.zarr"
    assert store_path_for("/x/movie.tiff").name == "movie.ome.zarr"
    # an .ome.zarr input is its own store (analyze in place)
    assert str(store_path_for("/x/movie.ome.zarr")) == "/x/movie.ome.zarr"


def test_remove_tree_basic_and_missing(tmp_path):
    d = tmp_path / "store.ome.zarr"
    (d / "sarcasm" / "masks").mkdir(parents=True)
    (d / "sarcasm" / "masks" / "zbands").write_bytes(b"x")
    remove_tree(d)
    assert not d.exists()
    remove_tree(d)  # a missing tree is not an error


def test_remove_tree_stages_the_tree_out_of_the_watchers_path(tmp_path, monkeypatch):
    """The macOS Finder failure mode: something keeps writing .DS_Store into the tree
    while rmtree walks it. The tree must be renamed away *before* deletion, so the
    delete runs on a path nothing else is touching."""
    from sarcasm.io import ome_store

    d = tmp_path / "store.ome.zarr"
    (d / "sarcasm").mkdir(parents=True)
    (d / "sarcasm" / "data").write_bytes(b"x")

    # record the path rmtree is called with, then delete for real
    seen = {}
    real = ome_store.shutil.rmtree

    def recording_rmtree(path, *a, **kw):
        seen["path"] = str(path)
        return real(path, *a, **kw)

    monkeypatch.setattr(ome_store.shutil, "rmtree", recording_rmtree)
    remove_tree(d)
    assert not d.exists()
    # deleted via a staging name, not the original path the watcher is holding
    assert seen["path"].endswith(".deleting")
    assert seen["path"] != str(d)
    assert not list(tmp_path.glob("*.deleting*"))


def test_remove_tree_retries_when_staging_is_impossible(tmp_path, monkeypatch):
    """If the rename cannot happen, fall back to deleting in place with retries."""
    from sarcasm.io import ome_store

    d = tmp_path / "store.ome.zarr"
    (d / "sarcasm").mkdir(parents=True)
    (d / "sarcasm" / "data").write_bytes(b"x")

    monkeypatch.setattr(ome_store.os, "replace",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("no rename here")))
    real = ome_store.shutil.rmtree
    calls = {"n": 0}

    def flaky_rmtree(path, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(66, "Directory not empty", str(path))
        return real(path, *a, **kw)

    monkeypatch.setattr(ome_store.shutil, "rmtree", flaky_rmtree)
    remove_tree(d)
    assert not d.exists()
    assert calls["n"] == 2          # retried exactly once, in place


def test_remove_tree_raises_a_clear_error_when_it_never_succeeds(tmp_path, monkeypatch):
    from sarcasm.io import ome_store

    d = tmp_path / "store.ome.zarr"
    d.mkdir()

    def always_fail(path, *a, **kw):
        raise OSError(66, "Directory not empty", str(path))

    monkeypatch.setattr(ome_store.shutil, "rmtree", always_fail)
    with pytest.raises(OSError, match="close any program"):
        remove_tree(d, attempts=2)


def test_detect_legacy_layout(tmp_path):
    tif = tmp_path / "old.tif"
    tif.write_bytes(b"")
    assert detect_legacy_layout(tif) is None
    data = tmp_path / "old" / "data"
    data.mkdir(parents=True)
    (data / "structure.json").write_text("{}")
    assert detect_legacy_layout(tif) == tmp_path / "old"
    assert "sarcasm==0.5" in legacy_layout_message(tmp_path / "old")


# --------------------------------------------------------------------------- #
# create / ingest image + OME metadata
# --------------------------------------------------------------------------- #
def test_create_and_read_image(tmp_path):
    img = _img()
    s = OmeZarrStore.create(tmp_path / "m.ome.zarr", img, axes="TYX",
                            pixelsize=0.65, frametime=0.1)
    assert s.exists
    assert np.array_equal(s.read_image(), img)
    assert s.read_image(2).shape == (48, 64)            # lazy single frame
    assert isinstance(s.image_handle(), zarr.Array)
    assert s.axes == "TYX"


def test_ome_multiscales_metadata(tmp_path):
    OmeZarrStore.create(tmp_path / "m.ome.zarr", _img(), axes="TYX",
                        pixelsize=0.65, frametime=0.1)
    ome = dict(zarr.open_group(str(tmp_path / "m.ome.zarr"), mode="r").attrs)["ome"]
    ms = ome["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["t", "y", "x"]
    assert ms["axes"][0]["type"] == "time"
    scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [0.1, 0.65, 0.65]                   # t=frametime, y/x=pixelsize


def test_create_refuses_overwrite(tmp_path):
    p = tmp_path / "m.ome.zarr"
    OmeZarrStore.create(p, _img(), axes="TYX")
    with pytest.raises(FileExistsError):
        OmeZarrStore.create(p, _img(), axes="TYX")
    OmeZarrStore.create(p, _img(), axes="TYX", overwrite=True)  # ok


# --------------------------------------------------------------------------- #
# masks / flow coexist with the image
# --------------------------------------------------------------------------- #
def test_masks_label_and_float(tmp_path):
    img = _img()
    s = OmeZarrStore.create(tmp_path / "m.ome.zarr", img, axes="TYX")
    cell = (img > 500).astype(np.uint8)
    zbands = (img / 1000.0).astype(np.float32)
    s.write_mask("cell_mask", cell, as_label=True)
    s.write_mask("zbands", zbands, as_label=False)
    assert np.array_equal(s.read_mask("cell_mask"), cell)
    assert np.allclose(s.read_mask("zbands"), zbands)
    assert s.has_mask("cell_mask") and s.has_mask("zbands")
    assert not s.has_mask("nope")
    assert set(s.mask_names()) == {"cell_mask", "zbands"}
    # OME labels list registered for napari
    root = zarr.open_group(str(s.path), mode="r")
    assert root["labels"].attrs["labels"] == ["cell_mask"]
    # the raw image survived the mask writes
    assert np.array_equal(s.read_image(), img)


def test_flow_roundtrip(tmp_path):
    s = OmeZarrStore.create(tmp_path / "m.ome.zarr", _img(), axes="TYX")
    flow = np.random.default_rng(1).random((4, 48, 64, 2)).astype(np.float32)
    s.write_flow(flow)
    assert np.allclose(s.read_flow(), flow)


# --------------------------------------------------------------------------- #
# analysis results nested under sarcasm/
# --------------------------------------------------------------------------- #
def test_results_nested_and_isolated(tmp_path):
    img = _img()
    s = OmeZarrStore.create(tmp_path / "m.ome.zarr", img, axes="TYX")
    rd = s.results_dict()
    rd["tracks_slen"] = np.full((30, 5), 1.8, np.float32)
    rd["n_tracks"] = 30
    rd.flush()
    # writing analysis did not disturb the image or masks
    assert np.array_equal(s.read_image(), img)
    # grouped lazy view through the store
    v = s.results_view()
    assert v.tracks.slen.shape == (30, 5)
    assert v["n_tracks"] == 30


def test_metadata_roundtrip(tmp_path):
    s = OmeZarrStore.create(tmp_path / "m.ome.zarr", _img(), axes="TYX",
                            pixelsize=0.65, frametime=0.1,
                            metadata={"channel": 1, "frametime": np.float64(0.1)})
    meta = s.read_metadata()
    assert meta["axes"] == "TYX"
    assert meta["pixelsize"] == 0.65
    assert meta["channel"] == 1


def test_everything_coexists(tmp_path):
    """One store: image + label + float mask + flow + tracks + metadata, all readable."""
    img = _img()
    s = OmeZarrStore.create(tmp_path / "m.ome.zarr", img, axes="TYX",
                            pixelsize=0.65, frametime=0.1)
    s.write_mask("cell_mask", (img > 500).astype(np.uint8), as_label=True)
    s.write_mask("zbands", (img / 1000).astype(np.float32))
    s.write_flow(np.zeros((4, 48, 64, 2), np.float32))
    rd = s.results_dict()
    rd["n_tracks"] = 7
    rd.flush()

    s2 = OmeZarrStore(tmp_path / "m.ome.zarr")           # fresh handle
    assert np.array_equal(s2.read_image(), img)
    assert s2.has_mask("cell_mask") and s2.has_mask("zbands")
    assert s2.read_flow().shape == (4, 48, 64, 2)
    assert s2.results_view()["n_tracks"] == 7
    assert s2.read_metadata()["pixelsize"] == 0.65
    # napari/Fiji-visible top level is a valid OME image with labels
    root = zarr.open_group(str(s2.path), mode="r")
    assert "ome" in dict(root.attrs)
    assert "0" in set(root.array_keys())
    assert "cell_mask" in root["labels"].attrs["labels"]
