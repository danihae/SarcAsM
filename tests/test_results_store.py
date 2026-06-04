# -*- coding: utf-8 -*-
"""Tests for the Zarr results store + lazy accessor (sarcasm.io.results_store)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import zarr
from scipy import sparse

from sarcasm.io.ioutils import IOUtils
from sarcasm.io.results_store import (
    Results,
    ResultsDict,
    export_to_json,
    write_results,
    _route,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _sample_data():
    """A dict spanning every value kind the real structure.json holds."""
    rng = np.random.default_rng(0)
    live = rng.random((40, 30)) > 0.4
    slen = np.where(live, rng.normal(1.8, 0.1, (40, 30)).astype(np.float32), np.nan)
    return {
        "n_tracks": 40,
        "motionfield_source": "tracker",                 # string scalar
        "tracks_slen": slen,                              # (n,T) float32 + NaN
        "tracks_positions_um": rng.random((40, 30, 2)).astype(np.float32),
        "tracks_snapped": live,                           # bool
        "tracks_detection_id": rng.integers(-1, 50, (40, 30)).astype(np.int32),
        "displacement_magnitude": rng.random((40, 30)).astype(np.float32),
        "sarcomere_oop": np.float64(0.73),                # numpy scalar
        "sarcomere_length_mean": np.array([1.7, 1.8, np.nan]),  # small array w/ NaN
        "sarcomere_length_vectors": [np.array([1.8, 1.9]), None, np.array([2.0, 2.1, 2.2])],
        "domain_slen_std": [np.asarray(0.1), None, np.asarray(0.2)],  # 0-d ragged
        "domain_mask": [sparse.random(15, 15, density=0.2, random_state=1).tocoo(), None,
                        sparse.eye(15).tocoo()],          # per-frame sparse
        "params.track_sarcomere_vectors.frames": list(range(30)),
        "params.track_sarcomere_vectors.max_disp_along_px": 15.0,
        "params.detect_sarcomeres.model": "model_v3",
    }


def _equal(a, b):
    if sparse.issparse(a) or sparse.issparse(b):
        a = a.toarray() if sparse.issparse(a) else np.asarray(a)
        b = b.toarray() if sparse.issparse(b) else np.asarray(b)
        return a.shape == b.shape and np.allclose(a, b, equal_nan=True)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            return np.allclose(a.astype(float), b.astype(float), equal_nan=True)
        return np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "data.zarr"
    write_results(_sample_data(), p)
    return p


# --------------------------------------------------------------------------- #
# round-trip / fidelity
# --------------------------------------------------------------------------- #
def test_roundtrip_all_kinds(store):
    data = _sample_data()
    r = Results(store)
    assert set(r.keys()) == set(data)
    for k in data:
        assert _equal(data[k], r[k]), f"mismatch for {k}"


def test_dtype_and_nan_preserved(store):
    r = Results(store)
    assert r["tracks_slen"].dtype == np.float32
    assert r["tracks_detection_id"].dtype == np.int32
    assert r["tracks_snapped"].dtype == bool
    assert np.isnan(r["tracks_slen"]).any()
    assert isinstance(r["sarcomere_oop"], np.floating)


def test_ragged_rank_preserved(store):
    r = Results(store)
    # 1-d per-frame vectors keep rank 1; 0-d scalars stay 0-d; None stays None
    assert r["sarcomere_length_vectors"][1] is None
    assert r["sarcomere_length_vectors"][0].shape == (2,)
    assert r["domain_slen_std"][0].shape == ()
    assert r["domain_slen_std"][1] is None


def test_sparse_sequence(store):
    r = Results(store)
    dm = r["domain_mask"]
    assert dm[1] is None
    assert sparse.issparse(dm[0]) and dm[0].shape == (15, 15)


# --------------------------------------------------------------------------- #
# routing / native group tree
# --------------------------------------------------------------------------- #
def test_routing():
    assert _route("tracks_slen") == ("tracks", "slen")
    assert _route("track_ids") == ("tracks", "ids")
    assert _route("displacement_magnitude") == ("motion", "displacement_magnitude")
    assert _route("sarcomere_oop") == ("structure/sarcomere", "oop")
    assert _route("params.detect_sarcomeres.model") == ("params/detect_sarcomeres", "model")


def test_native_groups_on_disk(store):
    root = zarr.open_group(str(store), mode="r")
    groups = set(root.group_keys())
    assert {"tracks", "motion", "structure", "params"} <= groups
    assert "slen" in set(root["tracks"].array_keys())


def test_attribute_access(store):
    r = Results(store)
    assert _equal(r.tracks.slen[:], r["tracks_slen"])
    assert r.structure.sarcomere.oop == r["sarcomere_oop"]
    assert r.params.track_sarcomere_vectors.max_disp_along_px == 15.0
    assert r.params.detect_sarcomeres.model == "model_v3"


def test_lazy_array_handle(store):
    r = Results(store)
    h = r.tracks.slen
    assert isinstance(h, zarr.Array)         # not materialised
    assert h.shape == (40, 30)
    assert _equal(h[5], r["tracks_slen"][5])  # single-row slice


# --------------------------------------------------------------------------- #
# ResultsDict: dict interface + lazy + staged writes
# --------------------------------------------------------------------------- #
def test_resultsdict_dict_interface(store):
    rd = ResultsDict(store)
    assert "n_tracks" in rd
    assert rd["n_tracks"] == 40
    assert rd.get("missing", "DEF") == "DEF"
    assert len(rd) == len(_sample_data())
    assert set(rd.keys()) == set(_sample_data())
    # update + pop + del
    rd.update({"new_scalar": 7})
    assert rd["new_scalar"] == 7
    assert rd.pop("new_scalar") == 7
    assert "new_scalar" not in rd


def test_resultsdict_staged_then_flush(tmp_path):
    p = tmp_path / "data.zarr"
    rd = ResultsDict(p)                      # no store yet
    rd["tracks_slen"] = np.arange(60, dtype=np.float32).reshape(20, 3)
    rd["n_tracks"] = 20
    assert rd["n_tracks"] == 20              # readable before flush
    rd.flush()
    assert p.exists()
    rd2 = ResultsDict(p)                      # fresh handle reads from disk
    assert rd2["n_tracks"] == 20
    assert _equal(rd2["tracks_slen"], np.arange(60, dtype=np.float32).reshape(20, 3))


def test_incremental_only_touches_changed_group(store):
    rd = ResultsDict(store)

    def snap():
        out = {}
        for dp, _, fs in os.walk(store):
            for f in fs:
                ap = os.path.join(dp, f)
                out[os.path.relpath(ap, store)] = os.path.getmtime(ap)
        return out

    before = snap()
    import time
    time.sleep(0.02)
    rd["pool_slen_timeseries"] = np.ones((3, 30), np.float32)  # -> structure/pool (new)
    rd.flush()
    changed = {k for k, v in snap().items() if v != before.get(k)}
    assert any(c.startswith("structure/pool/") for c in changed)
    assert not any(c.startswith("tracks/") for c in changed)
    assert not any(c.startswith("motion/") for c in changed)


def test_overwrite_existing_key(store):
    rd = ResultsDict(store)
    rd["tracks_slen"] = np.zeros((40, 30), np.float32)
    rd.flush()
    assert np.all(ResultsDict(store)["tracks_slen"] == 0)


def test_delete_persists(store):
    rd = ResultsDict(store)
    del rd["n_tracks"]
    rd.flush()
    assert "n_tracks" not in ResultsDict(store)


def test_view_reflects_flushed_state(store):
    rd = ResultsDict(store)
    rd["mband_slen_timeseries"] = np.full((2, 30), 1.9, np.float32)
    v = rd.view()                            # flushes, returns Results
    assert _equal(v.structure.mband.slen_timeseries[:], rd["mband_slen_timeseries"])


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_json_legacy_roundtrip(store, tmp_path):
    rd = ResultsDict(store)
    out = tmp_path / "structure.json"
    export_to_json(rd, out)
    back = IOUtils.json_deserialize(out)
    assert set(back.keys()) == set(rd.keys())
    assert _equal(back["tracks_slen"], rd["tracks_slen"])


def test_export_json_skip_arrays(store, tmp_path):
    rd = ResultsDict(store)
    out = tmp_path / "scalars.json"
    export_to_json(rd, out, include_arrays=False)
    back = IOUtils.json_deserialize(out)
    assert "tracks_slen" not in back           # large array skipped
    assert "n_tracks" in back                  # scalar kept
