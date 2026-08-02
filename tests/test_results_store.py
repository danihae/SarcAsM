# -*- coding: utf-8 -*-
"""Tests for the Zarr results store + accessor (sarcasm.io.results_store)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import zarr
from scipy import sparse

from sarcasm.features import MOTION_KINDS
from sarcasm.io.ioutils import IOUtils
from sarcasm.io.results_store import (
    Results,
    TOP_GROUPS,
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
        "motion.tracks.n": 40,
        "motion.tracks.slen": slen,                              # (n,T) float32 + NaN
        "motion.tracks.positions_um": rng.random((40, 30, 2)).astype(np.float32),
        "motion.tracks.observed": live,                           # bool
        "motion.tracks.detection_id": rng.integers(-1, 50, (40, 30)).astype(np.int32),
        "structure.sarcomere.oop": np.float64(0.73),                # numpy scalar
        "structure.sarcomere.slen_mean": np.array([1.7, 1.8, np.nan]),  # small array w/ NaN
        "structure.sarcomere.slen": [np.array([1.8, 1.9]), None, np.array([2.0, 2.1, 2.2])],
        "structure.domain.slen_std": [np.asarray(0.1), None, np.asarray(0.2)],  # 0-d ragged
        "structure.domain.mask": [sparse.random(15, 15, density=0.2, random_state=1).tocoo(), None,
                        sparse.eye(15).tocoo()],          # per-frame sparse
        "params.track_sarcomere_vectors.frames": list(range(30)),
        "params.track_sarcomere_vectors.max_disp_along_um": 1.0,
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
    assert r["motion.tracks.slen"].dtype == np.float32
    assert r["motion.tracks.detection_id"].dtype == np.int32
    assert r["motion.tracks.observed"].dtype == bool
    assert np.isnan(r["motion.tracks.slen"]).any()
    assert isinstance(r["structure.sarcomere.oop"], np.floating)


def test_tracks_prefix_maps_to_zarr_member_names(store):
    """`tracks_<x>` is stored as the zarr member `tracks/<x>`.

    The physical name is derived by stripping the prefix, so renaming a flat key
    silently renames the on-disk array with no change to results_store.py.
    Pin it, or a rename lands in the store unnoticed.
    """
    members = set(zarr.open_group(store)["motion/tracks"])
    assert {"slen", "positions_um", "observed", "detection_id"} <= members


def test_ragged_rank_preserved(store):
    r = Results(store)
    # 1-d per-frame vectors keep rank 1; 0-d scalars stay 0-d; None stays None
    assert r["structure.sarcomere.slen"][1] is None
    assert r["structure.sarcomere.slen"][0].shape == (2,)
    assert r["structure.domain.slen_std"][0].shape == ()
    assert r["structure.domain.slen_std"][1] is None


def test_sparse_sequence(store):
    r = Results(store)
    dm = r["structure.domain.mask"]
    assert dm[1] is None
    assert sparse.issparse(dm[0]) and dm[0].shape == (15, 15)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def test_route_is_the_path():
    """Routing is a split, not a table: the key already says where it lives."""
    assert _route("motion.tracks.slen") == ("motion/tracks", "slen")
    assert _route("motion.pool.slen") == ("motion/pool", "slen")
    assert _route("structure.domain.slen") == ("structure/domain", "slen")
    assert _route("structure.sarcomere.oop") == ("structure/sarcomere", "oop")
    assert _route("params.detect_sarcomeres.model") == ("params/detect_sarcomeres", "model")


def test_structure_and_motion_never_collide():
    """The same subject on both branches stays distinct — this is why paths."""
    assert _route("structure.domain.slen") != _route("motion.domain.slen")
    assert _route("structure.myofibril.length") != _route("motion.myofibril.slen")
    assert _route("motion.loi.data") == ("motion/loi", "data")
    for kind in ("pool", "mband", "myofibril", "domain", "loi", "custom"):
        assert _route(f"motion.{kind}.equ") == (f"motion/{kind}", "equ")



def test_native_groups_on_disk(store):
    root = zarr.open_group(str(store), mode="r")
    groups = set(root.group_keys())
    assert {"structure", "motion", "params"} <= groups
    assert "slen" in set(root["motion/tracks"].array_keys())


# --------------------------------------------------------------------------- #
# the three access styles
# --------------------------------------------------------------------------- #
def test_attribute_access(store):
    r = Results(store)
    assert _equal(r.motion.tracks.slen, r["motion.tracks.slen"])
    assert r.structure.sarcomere.oop == r["structure.sarcomere.oop"]
    assert type(r.structure.sarcomere.oop) is type(r["structure.sarcomere.oop"])
    assert r.params.track_sarcomere_vectors.max_disp_along_um == 1.0
    assert r.params.detect_sarcomeres.model == "model_v3"



def test_three_access_styles_return_the_same_object_kind(store):
    r = Results(store)
    for key in ("motion.tracks.slen", "structure.sarcomere.oop", "structure.domain.mask"):
        node = r
        for seg in key.split("."):
            node = getattr(node, seg)
        assert node is r[key], f"{key}: dotted key and attribute path must be one value"


@pytest.mark.parametrize("n", [100, 1000])
def test_attribute_type_is_storage_independent(tmp_path, n):
    """Small values live inline in group attrs, big ones become zarr arrays.

    Attribute access must not leak that difference: before, the same logical key
    was an ndarray on a short recording and a (non-arithmetic) zarr.Array on a
    long one, so `oop * 2` worked on one dataset and raised on the next.
    """
    p = tmp_path / f"data_{n}.zarr"
    oop = np.linspace(0.5, 0.9, n)
    write_results({"structure.sarcomere.oop": oop}, p)
    r = Results(p)
    for got in (r.structure.sarcomere.oop, r["structure.sarcomere.oop"]):
        assert isinstance(got, np.ndarray)
        assert not isinstance(got, zarr.Array)
        assert np.allclose(got * 2, oop * 2)          # arithmetic, not just aggregation


def test_grouped_view_sees_staged_writes_without_flush(store):
    """The namespace is derived from the keys, so it needs no round trip."""
    r = Results(store)
    r["motion.mband.slen"] = np.full((2, 30), 1.9, np.float32)
    assert _equal(r.motion.mband.slen, r["motion.mband.slen"])
    assert "motion.mband.slen" in r.keys()
    r.flush()
    assert _equal(Results(store).motion.mband.slen, np.full((2, 30), 1.9, np.float32))


def test_masks_group_is_not_exposed(store):
    """`masks/` shares the sarcasm group but is reached as sarc.zbands etc."""
    zarr.open_group(str(store), mode="a").create_group("masks")
    r = Results(store)
    assert "masks" not in dir(r)
    with pytest.raises(AttributeError):
        r.masks


def test_namespaces_have_no_public_methods(store):
    """Any public name on a namespace is a name a result key could never take."""
    r = Results(store)
    assert dir(r.motion.tracks) == sorted(["slen", "positions_um", "observed",
                                           "detection_id", "n"])


# --------------------------------------------------------------------------- #
# discoverability
# --------------------------------------------------------------------------- #
def test_keys_are_ordered_not_a_set(store):
    r = Results(store)
    assert r.keys() == list(_sample_data())
    assert list(iter(r)) == r.keys()
    r["structure.cell.mask_area"] = 1
    assert r.keys()[-1] == "structure.cell.mask_area"


def test_dir_lists_keys_and_groups(store):
    r = Results(store)
    names = dir(r)
    # keys are dotted paths, so tab completion offers the namespaces, not 181 keys
    assert set(TOP_GROUPS) <= set(names)
    assert "keys" in names and "find" in names
    assert not any("." in n for n in names)
    assert dir(r.motion) == ["tracks"]


def test_find_substring_glob_group_and_empty(store):
    r = Results(store)
    assert set(r.find("slen")) == {"motion.tracks.slen", "structure.sarcomere.slen",
                                   "structure.sarcomere.slen_mean",
                                   "structure.domain.slen_std"}
    assert set(r.find("SLEN")) == set(r.find("slen"))          # case-insensitive
    assert set(r.find("motion.tracks.*")) == set(r.find(group="motion/tracks"))
    assert set(r.find(group="tracks")) == {"motion.tracks.slen", "motion.tracks.positions_um",
                                           "motion.tracks.observed", "motion.tracks.detection_id",
                                           "motion.tracks.n"}
    assert set(r.find(r"\.n$", regex=True)) == {"motion.tracks.n"}
    empty = r.find("definitely_absent")
    assert list(empty) == []
    assert "no keys match" in repr(empty)


def test_find_by_group_last_segment_spans_both_branches(store):
    """`domain` names a group on both branches, so the short form is a union."""
    r = Results(store)
    r["motion.domain.slen"] = 1.0
    assert set(r.find(group="domain")) == {"structure.domain.slen_std",
                                           "structure.domain.mask",
                                           "motion.domain.slen"}
    assert set(r.find(group="motion/domain")) == {"motion.domain.slen"}


def test_find_returns_usable_keys(store):
    r = Results(store)
    hits = r.find("tracks")
    assert hits and all(k in r for k in hits)
    assert _equal(r[hits[0]], r[hits[0]])


def test_describe_structure_and_motion_and_params(store):
    r = Results(store)
    r["motion.pool.beating_rate"] = np.array([1.2])
    assert "sarcomere length" in r.describe("motion.tracks.slen").name.lower()
    info = r.describe("motion.pool.beating_rate")
    assert info.registry == "motion" and "beating" in info.description.lower()
    assert r.describe("params.detect_sarcomeres.model").registry == "params"


@pytest.mark.parametrize("kind", MOTION_KINDS)
def test_describe_resolves_every_grouping_kind(store, kind):
    """`<kind>_<suffix>` must resolve for all six kinds, not just `domain`.

    `domain_*` additionally has its own structure-registry entries, which take
    precedence; either registry is a documented answer.
    """
    r = Results(store)
    key = f"motion.{kind}.beating_rate"
    r[key] = np.array([1.0])
    info = r.describe(key)
    assert info.registry in ("motion", "structure")
    assert "beating" in info.description.lower()


def test_describe_undocumented_key_does_not_raise(store):
    r = Results(store)
    r["structure.cell.totally_new_metric"] = np.arange(5)
    info = r.describe("structure.cell.totally_new_metric")
    assert info.registry is None
    assert "not documented" in repr(info)


def test_describe_absent_key_raises_with_suggestion(store):
    r = Results(store)
    with pytest.raises(KeyError, match="motion.tracks.slen"):
        r.describe("tracks_slne")


def test_unknown_attribute_suggests_close_match(store):
    r = Results(store)
    with pytest.raises(AttributeError, match="motion.tracks.slen"):
        r.tracks_slne


def test_repr_is_bounded_and_names_groups(store):
    r = Results(store)
    text = repr(r)
    assert "tracks" in text and "sarcomere" in text and "params" in text
    assert str(len(r)) in text
    assert text.count("\n") < 30
    # namespaces read in pipeline order, not alphabetically
    assert text.index("structure") < text.index("motion") < text.index("params")
    assert repr(r.motion.tracks).startswith("<data.motion.tracks")


def test_repr_html_contains_keys(store):
    html = Results(store)._repr_html_()
    assert "<table" in html and "motion.tracks.slen" in html


def test_repr_of_empty_store(tmp_path):
    p = tmp_path / "empty.zarr"
    write_results({}, p)
    assert "empty" in repr(Results(p))


# --------------------------------------------------------------------------- #
# write protection / key validity
# --------------------------------------------------------------------------- #
def test_attribute_assignment_raises(store):
    r = Results(store)
    with pytest.raises(AttributeError, match=r"sarc\.data\['tpyo'\]"):
        r.tpyo = 1
    with pytest.raises(AttributeError):
        r.motion = 1
    with pytest.raises(AttributeError):
        r.motion.tracks.slen = np.zeros((2, 2))
    assert "tpyo" not in r


@pytest.mark.parametrize("bad", ["slen", "keys", "tracks", "bogus.slen", "motion", ""])
def test_key_must_be_a_dotted_path_under_a_known_namespace(tmp_path, store, bad):
    """A bare name could shadow an accessor method; an unknown head invents a
    namespace. Both are refused at write time, in both write paths."""
    r = Results(store)
    with pytest.raises((KeyError, TypeError)):
        r[bad] = 1
    with pytest.raises((KeyError, TypeError)):
        write_results({bad: 1}, tmp_path / "bad.zarr")


def test_non_string_key_rejected(store):
    r = Results(store)
    with pytest.raises(TypeError):
        r[7] = 1


# --------------------------------------------------------------------------- #
# handle(): the explicit lazy escape hatch
# --------------------------------------------------------------------------- #
def test_handle_returns_lazy_zarr_array(store):
    r = Results(store)
    h = r.handle("motion.tracks.slen")
    assert isinstance(h, zarr.Array)             # not materialised
    assert h.shape == (40, 30)
    assert _equal(h[5], r["motion.tracks.slen"][5])     # single-row slice


def test_handle_rejects_inline_key(store):
    r = Results(store)
    with pytest.raises(TypeError, match="inline"):
        r.handle("structure.sarcomere.oop")


def test_handle_rejects_staged_key(store):
    r = Results(store)
    r["motion.tracks.slen"] = np.zeros((40, 30), np.float32)
    with pytest.raises(RuntimeError, match="commit"):
        r.handle("motion.tracks.slen")


def test_handle_unknown_key_raises_keyerror(store):
    with pytest.raises(KeyError):
        Results(store).handle("nope")


# --------------------------------------------------------------------------- #
# dict interface + staged writes + persistence
# --------------------------------------------------------------------------- #
def test_dict_interface(store):
    rd = Results(store)
    assert "motion.tracks.n" in rd
    assert rd["motion.tracks.n"] == 40
    assert rd.get("missing", "DEF") == "DEF"
    assert len(rd) == len(_sample_data())
    assert set(rd.keys()) == set(_sample_data())
    # update + pop + del
    rd.update({"structure.cell.mask_area": 7})
    assert rd["structure.cell.mask_area"] == 7
    assert rd.pop("structure.cell.mask_area") == 7
    assert "structure.cell.mask_area" not in rd


def test_staged_then_flush(tmp_path):
    p = tmp_path / "data.zarr"
    rd = Results(p)                          # no store yet
    rd["motion.tracks.slen"] = np.arange(60, dtype=np.float32).reshape(20, 3)
    rd["motion.tracks.n"] = 20
    assert rd["motion.tracks.n"] == 20              # readable before flush
    rd.flush()
    assert p.exists()
    rd2 = Results(p)                         # fresh handle reads from disk
    assert rd2["motion.tracks.n"] == 20
    assert _equal(rd2["motion.tracks.slen"], np.arange(60, dtype=np.float32).reshape(20, 3))


def test_opening_does_not_create_a_store(tmp_path):
    p = tmp_path / "never.zarr"
    r = Results(p)
    assert r.keys() == [] and len(r) == 0
    repr(r)                                   # display must not create it either
    assert not p.exists()


def test_incremental_only_touches_changed_group(store):
    rd = Results(store)

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
    rd["motion.pool.slen"] = np.ones((3, 30), np.float32)  # -> motion/pool (new)
    rd.flush()
    changed = {k for k, v in snap().items() if v != before.get(k)}
    assert any(c.startswith("motion/pool/") for c in changed)
    assert not any(c.startswith("motion/tracks/") for c in changed)


def test_overwrite_existing_key(store):
    rd = Results(store)
    rd["motion.tracks.slen"] = np.zeros((40, 30), np.float32)
    rd.flush()
    assert np.all(Results(store)["motion.tracks.slen"] == 0)


def test_delete_persists(store):
    rd = Results(store)
    del rd["motion.tracks.n"]
    rd.flush()
    assert "motion.tracks.n" not in Results(store)
    assert "motion.tracks.n" not in dir(Results(store))


def test_manifest_pins_the_physical_member(tmp_path):
    """The manifest, not the key, decides where an existing value is rewritten.

    A store may hold a member whose name does not match what _route would pick
    today. Reads still resolve through the manifest, and an overwrite lands on
    the same member rather than orphaning it.
    """
    p = tmp_path / "odd.zarr"
    root = zarr.open_group(str(p), mode="w")
    root.create_group("structure").create_group("sarcomere").attrs["legacy_oop"] = 0.5
    root.attrs["_manifest"] = {
        "structure.sarcomere.oop": ["structure/sarcomere", "legacy_oop", "attr"]}

    r = Results(p)
    assert r["structure.sarcomere.oop"] == 0.5
    assert r.structure.sarcomere.oop == 0.5        # namespace comes from the key
    r["structure.sarcomere.oop"] = 0.9
    r.flush()
    stored = zarr.open_group(str(p), mode="r")["structure/sarcomere"].attrs
    assert stored["legacy_oop"] == 0.9             # overwrote in place
    assert "oop" not in stored                     # no orphan created


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_json_legacy_roundtrip(store, tmp_path):
    rd = Results(store)
    out = tmp_path / "structure.json"
    export_to_json(rd, out)
    back = IOUtils.json_deserialize(out)
    assert set(back.keys()) == set(rd.keys())
    assert _equal(back["motion.tracks.slen"], rd["motion.tracks.slen"])


def test_export_json_skip_arrays(store, tmp_path):
    rd = Results(store)
    out = tmp_path / "scalars.json"
    export_to_json(rd, out, include_arrays=False)
    back = IOUtils.json_deserialize(out)
    assert "motion.tracks.slen" not in back           # large array skipped
    assert "motion.tracks.n" in back                  # scalar kept
