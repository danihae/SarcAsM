# -*- coding: utf-8 -*-
# Copyright (c) 2025 University Medical Center Göttingen, Germany.
# All rights reserved.
#
# Patent Pending: DE 10 2024 112 939.5
# SPDX-License-Identifier: LicenseRef-Proprietary-See-LICENSE
#
# This software is licensed under a custom license. See the LICENSE file
# in the root directory for full details.
#
# **Commercial use is prohibited without a separate license.**
# Contact MBM ScienceBridge GmbH (https://sciencebridge.de/en/) for licensing.

"""Single Zarr store for SarcAsM analysis + motion results.

A self-describing Zarr group tree — used standalone as ``data.zarr`` and, in
production, re-parented under the ``sarcasm/`` group of the OME-Zarr container
(see :mod:`sarcasm.io.ome_store`).

**A result key is its path.** ``'motion.pool.slen'`` is member ``slen`` of group
``motion/pool``, so the logical name and the physical location are the same
thing, routing is a split rather than a table of prefix rules, and two keys can
never claim the same home. There are three namespaces::

    data.zarr/
      structure/                per-frame / single-frame morphology
        cell/ zbands/ sarcomere/ myofibril/ domain/
      motion/                   everything derived from the sarcomere tracks
        tracks/                 dense per-track block, row-chunked + zstd
        groups/                 which grouping was built, and from what
        pool/ mband/ myofibril/ domain/ loi/ custom/   per-group contraction metrics
                                (loi/ also holds the LOI geometry, loi.data)
      params/<step>/  (attrs)   what each analysis step ran with

Small things (scalars, params, strings) live as JSON-shaped Zarr **attributes**
(human-readable inside each group's ``zarr.json``); large numeric arrays live as
binary Zarr arrays next to them. JSON becomes an explicit *export*, not the
storage format.

Access (see :class:`Results`) — one name, two spellings::

    r = Results("…/<name>.ome.zarr/sarcasm")
    r['motion.tracks.slen']                  # dict
    r.motion.tracks.slen                     # attribute path — the same object
    r.structure.sarcomere.oop
    r.params.track_sarcomere_vectors.max_disp_along_um

Both materialise, so a value's type never depends on whether it happened to be
small enough to be stored inline. :meth:`Results.handle` is the explicit opt-in
for a lazy ``zarr.Array`` when you want one row of a large ``(n_tracks, T)``
block without loading it all.

The namespace is derived from the keys, not from the physical group tree, so it
also covers staged, not-yet-flushed writes, and a store whose members were
written under a different layout still presents the current namespace (the
manifest pins each key's physical home). Stores written before the dotted-key
rename hold the old names and must be regenerated.

Image pixel data and segmentation masks live alongside these groups in the same
OME-Zarr container (:mod:`sarcasm.io.ome_store`); image metadata is mirrored into
the root attrs.
"""

from __future__ import annotations

import difflib
import fnmatch
import logging
import re
import textwrap
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import zarr
from scipy import sparse

from sarcasm.features import describe_key, pretty_name

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
_INLINE_MAX = 256          # ndarrays with <= this many elements go inline (attrs)
_ROW_CHUNK = 512           # chunk size along axis 0 for big arrays (lazy per-row reads)
_KIND = "_kind"            # subgroup attr marking ragged/sparse leaves
_MANIFEST = "_manifest"    # root attr: flat_key -> [group, member, kind]
_VERSION = "_format_version"


# --------------------------------------------------------------------------- #
# routing: a key IS its path
# --------------------------------------------------------------------------- #
#: The three top-level namespaces. ``structure`` is per-frame / single-frame
#: morphology, ``motion`` is everything derived from the sarcomere tracks, and
#: ``params`` records the arguments each analysis step ran with.
TOP_GROUPS = ("structure", "motion", "params")

#: Subgroup display order within each namespace — pipeline order, so the repr
#: reads the way the analysis runs. Unlisted subgroups sort after, alphabetically.
_GROUP_ORDER = {
    "structure": ("cell", "zbands", "sarcomere", "myofibril", "domain"),
    "motion": ("tracks", "groups", "pool", "mband", "myofibril", "domain", "loi",
               "custom"),
}


def _route(key: str) -> Tuple[str, str]:
    """Split a result key into its group path and member name.

    A key is a dotted path — ``'motion.pool.slen'`` is member ``slen`` of group
    ``motion/pool`` — so routing is a split, not a table of prefix rules, and two
    keys can never claim the same home.

    Parameters
    ----------
    key : str
        Dotted result key (e.g. ``'motion.tracks.slen'``).

    Returns
    -------
    tuple of (str, str)
        The ``(group_path, member_name)`` where the value is stored.
    """
    group, _, member = key.rpartition(".")
    return group.replace(".", "/"), member


def _check_key(key: str) -> None:
    """Reject keys that are not dotted paths under a known namespace.

    This is what makes attribute access safe: a key always has at least one dot,
    so it can never be a bare identifier shadowing an accessor method, and it
    always starts with a known namespace, so it can never invent a top-level
    group. Checked at write time, where the traceback still points at the
    analysis code that produced the key.

    Parameters
    ----------
    key : str
        Result key.

    Raises
    ------
    TypeError
        If ``key`` is not a non-empty string.
    KeyError
        If ``key`` is not ``<namespace>.<...>.<member>`` with a known namespace.
    """
    if not isinstance(key, str) or not key:
        raise TypeError(f"result keys must be non-empty strings, got {key!r}")
    head, _, rest = key.partition(".")
    if head not in TOP_GROUPS or not rest:
        raise KeyError(
            f"{key!r} is not a valid result key: keys are dotted paths under "
            f"{', '.join(TOP_GROUPS)} (e.g. 'motion.pool.slen').")


# --------------------------------------------------------------------------- #
# attr (JSON-shaped) encoding for small / non-array values
# --------------------------------------------------------------------------- #
def _to_attr(v: Any) -> Any:
    """Encode a small/non-array value as a JSON-shaped Zarr attribute.

    Parameters
    ----------
    v : Any
        Value to encode (ndarray, np.generic, dict, list/tuple, or plain type).

    Returns
    -------
    Any
        JSON-serializable representation, tagging ndarrays and numpy scalars
        for round-trip.
    """
    if sparse.issparse(v):
        raise TypeError("sparse matrix reached the attr path; handle as a bulk type")
    if isinstance(v, np.ndarray):
        return {"__nd__": True, "dtype": str(v.dtype), "data": v.tolist()}
    if isinstance(v, np.generic):
        return {"__sc__": True, "dtype": v.dtype.name, "data": v.item()}
    if isinstance(v, dict):
        return {k: _to_attr(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_attr(x) for x in v]
    return v


def _from_attr(v: Any) -> Any:
    """Decode a value produced by :func:`_to_attr`.

    Parameters
    ----------
    v : Any
        Encoded attribute value.

    Returns
    -------
    Any
        Reconstructed value, restoring ndarrays and numpy scalars.
    """
    if isinstance(v, dict):
        if v.get("__nd__"):
            return np.array(v["data"], dtype=v["dtype"])
        if v.get("__sc__"):
            return np.dtype(v["dtype"]).type(v["data"])
        return {k: _from_attr(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_from_attr(x) for x in v]
    return v


# --------------------------------------------------------------------------- #
# array / ragged / sparse helpers
# --------------------------------------------------------------------------- #
def _chunks(shape: tuple) -> Optional[tuple]:
    """Row-chunk a shape along axis 0 for lazy per-row reads.

    Parameters
    ----------
    shape : tuple
        Array shape.

    Returns
    -------
    tuple or None
        Chunk shape with axis 0 capped at ``_ROW_CHUNK``, or None for scalars.
    """
    if len(shape) == 0:
        return None
    c = list(shape)
    c[0] = min(shape[0], _ROW_CHUNK)
    return tuple(c)


def _write_array(grp: "zarr.Group", member: str, arr: np.ndarray,
                 row_chunk: bool = False) -> None:
    """Write a numpy array into a Zarr group.

    Parameters
    ----------
    grp : zarr.Group
        Target group.
    member : str
        Array name.
    arr : np.ndarray
        Array to write.
    row_chunk : bool, optional
        If True, row-chunk along axis 0 (for ``(n_tracks, T)`` arrays so a
        single track reads one chunk); otherwise internal ragged/sparse arrays
        auto-chunk to avoid tiny-file blow-up. Default is False.
    """
    if arr.size and row_chunk:
        chunks = _chunks(arr.shape)
    elif arr.size:
        chunks = "auto"
    else:
        chunks = None
    a = grp.create_array(member, shape=arr.shape, dtype=arr.dtype, chunks=chunks)
    if arr.size:
        a[...] = arr


def _is_ragged(v: Any) -> bool:
    """Return True if ``v`` is a non-empty list/tuple containing any ndarray."""
    return isinstance(v, (list, tuple)) and len(v) > 0 and any(
        isinstance(x, np.ndarray) for x in v
    )


def _write_ragged(parent: "zarr.Group", member: str, frames: list) -> None:
    """Write a per-frame list of arrays as a flat CSR-like subgroup.

    Stored as ``values`` + ``offsets`` + ``none_mask`` (rank-preserving),
    tagged ``_kind = 'ragged'``.

    Parameters
    ----------
    parent : zarr.Group
        Parent group.
    member : str
        Subgroup name.
    frames : list
        Per-frame list of ndarrays (or None) with homogeneous element ndim.

    Raises
    ------
    ValueError
        If elements have heterogeneous ndim.
    """
    arrs = [None if f is None else np.asarray(f) for f in frames]
    present = [a for a in arrs if a is not None]
    ndims = {a.ndim for a in present}
    if len(ndims) > 1:
        raise ValueError(f"heterogeneous element ndim {ndims}")
    elem_ndim = ndims.pop() if ndims else 1
    lengths = np.array([0 if a is None else (a.shape[0] if a.ndim else 1) for a in arrs],
                       dtype=np.int64)
    none_mask = np.array([a is None for a in arrs], dtype=bool)
    nonempty = [a for a in arrs if a is not None and a.size > 0]
    if nonempty:
        values = np.concatenate([np.atleast_1d(a) for a in nonempty], axis=0)
    else:
        ref = present[0] if present else None
        trailing = ref.shape[1:] if (ref is not None and ref.ndim > 1) else ()
        values = np.zeros((0,) + trailing, dtype=ref.dtype if ref is not None else np.float32)
    offsets = np.empty(lengths.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    g = parent.create_group(member)
    g.attrs[_KIND] = "ragged"
    g.attrs["elem_ndim"] = int(elem_ndim)
    _write_array(g, "values", values)
    _write_array(g, "offsets", offsets)
    _write_array(g, "none_mask", none_mask)


def _read_ragged(g: "zarr.Group") -> List[Optional[np.ndarray]]:
    """Read a ragged subgroup back into a per-frame list of arrays.

    Parameters
    ----------
    g : zarr.Group
        Subgroup written by :func:`_write_ragged`.

    Returns
    -------
    list of (np.ndarray or None)
        Per-frame arrays, with None in the slots flagged by ``none_mask``.
    """
    values = g["values"][...]
    offsets = g["offsets"][...]
    none_mask = g["none_mask"][...]
    elem_ndim = int(g.attrs.get("elem_ndim", 1))
    out: List[Optional[np.ndarray]] = []
    for i in range(none_mask.shape[0]):
        if none_mask[i]:
            out.append(None)
        elif elem_ndim == 0:
            out.append(np.asarray(values[offsets[i]]))
        else:
            out.append(values[offsets[i]:offsets[i + 1]])
    return out


def _write_sparse(parent: "zarr.Group", member: str, mat) -> None:
    """Write a sparse matrix as a COO subgroup tagged ``_kind = 'sparse'``.

    Parameters
    ----------
    parent : zarr.Group
        Parent group.
    member : str
        Subgroup name.
    mat : scipy.sparse.spmatrix
        Matrix to store.
    """
    coo = mat.tocoo()
    g = parent.create_group(member)
    g.attrs[_KIND] = "sparse"
    g.attrs["shape"] = list(coo.shape)
    _write_array(g, "data", coo.data)
    _write_array(g, "row", coo.row.astype(np.int64))
    _write_array(g, "col", coo.col.astype(np.int64))


def _read_sparse(g: "zarr.Group"):
    """Read a sparse subgroup back into a ``scipy.sparse.coo_matrix``.

    Parameters
    ----------
    g : zarr.Group
        Subgroup written by :func:`_write_sparse`.

    Returns
    -------
    scipy.sparse.coo_matrix
        Reconstructed matrix.
    """
    return sparse.coo_matrix(
        (g["data"][...], (g["row"][...], g["col"][...])),
        shape=tuple(g.attrs["shape"]),
    )


def _is_sparse_seq(v: Any) -> bool:
    """Return True if ``v`` is a non-empty list/tuple containing any sparse matrix."""
    return isinstance(v, (list, tuple)) and len(v) > 0 and any(
        sparse.issparse(x) for x in v
    )


def _write_sparse_seq(parent: "zarr.Group", member: str, seq: list) -> None:
    """Write a per-frame list of sparse matrices as one flat COO.

    Stored as concatenated ``data``/``row``/``col`` plus a ``frame`` index and
    ``none_mask``, tagged ``_kind = 'sparse_seq'``.

    Parameters
    ----------
    parent : zarr.Group
        Parent group.
    member : str
        Subgroup name.
    seq : list
        Per-frame list of sparse matrices (or None).
    """
    coos = [None if x is None else x.tocoo() for x in seq]
    present = [c for c in coos if c is not None]
    shape = present[0].shape if present else (0, 0)
    none_mask = np.array([c is None for c in coos], dtype=bool)
    datas, rows, cols, frames = [], [], [], []
    for i, c in enumerate(coos):
        if c is None:
            continue
        datas.append(c.data)
        rows.append(c.row.astype(np.int64))
        cols.append(c.col.astype(np.int64))
        frames.append(np.full(c.nnz, i, dtype=np.int64))
    def cat(xs, dt):
        return np.concatenate(xs) if xs else np.zeros(0, dtype=dt)

    g = parent.create_group(member)
    g.attrs[_KIND] = "sparse_seq"
    g.attrs["shape"] = list(shape)
    g.attrs["n"] = len(coos)
    _write_array(g, "data", cat(datas, np.float64))
    _write_array(g, "row", cat(rows, np.int64))
    _write_array(g, "col", cat(cols, np.int64))
    _write_array(g, "frame", cat(frames, np.int64))
    _write_array(g, "none_mask", none_mask)


def _read_sparse_seq(g: "zarr.Group") -> List[Optional[Any]]:
    """Read a sparse_seq subgroup back into a per-frame list of matrices.

    Parameters
    ----------
    g : zarr.Group
        Subgroup written by :func:`_write_sparse_seq`.

    Returns
    -------
    list of (scipy.sparse.coo_matrix or None)
        Per-frame matrices, with None in the slots flagged by ``none_mask``.
    """
    data, row, col = g["data"][...], g["row"][...], g["col"][...]
    frame, none_mask = g["frame"][...], g["none_mask"][...]
    shape = tuple(g.attrs["shape"])
    out: List[Optional[Any]] = []
    for i in range(int(g.attrs["n"])):
        if none_mask[i]:
            out.append(None)
            continue
        m = frame == i
        out.append(sparse.coo_matrix((data[m], (row[m], col[m])), shape=shape))
    return out


def _ensure_group(root: "zarr.Group", path: str) -> "zarr.Group":
    """Get or create the subgroup at ``path`` (slash-separated) under ``root``.

    Parameters
    ----------
    root : zarr.Group
        Root group.
    path : str
        Slash-separated subgroup path.

    Returns
    -------
    zarr.Group
        The (possibly newly created) group.
    """
    g = root
    for seg in path.split("/"):
        try:
            g = g[seg]
        except KeyError:
            g = g.create_group(seg)
    return g


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
def _write_value(grp: "zarr.Group", member: str, val: Any) -> str:
    """Write one value into a group, dispatching on its type.

    Parameters
    ----------
    grp : zarr.Group
        Target group.
    member : str
        Member name.
    val : Any
        Value to store (large ndarray, sparse matrix, sparse_seq, ragged list,
        or small attr-encodable value).

    Returns
    -------
    str
        Kind tag: one of ``'array'``, ``'sparse'``, ``'sparse_seq'``,
        ``'ragged'`` or ``'attr'``.
    """
    if isinstance(val, np.ndarray) and val.size > _INLINE_MAX:
        _write_array(grp, member, val, row_chunk=True)
        return "array"
    if sparse.issparse(val):
        _write_sparse(grp, member, val)
        return "sparse"
    if _is_sparse_seq(val):
        _write_sparse_seq(grp, member, list(val))
        return "sparse_seq"
    if _is_ragged(val):
        try:
            _write_ragged(grp, member, list(val))
            return "ragged"
        except Exception as e:
            logger.debug("ragged encode failed for %s (%s); inlining", member, e)
    grp.attrs[member] = _to_attr(val)
    return "attr"


def _remove_member(grp: "zarr.Group", member: str) -> None:
    """Delete an existing array/subgroup/attr member, if present (for overwrite).

    Parameters
    ----------
    grp : zarr.Group
        Group to remove from.
    member : str
        Member name.
    """
    try:
        del grp[member]
    except (KeyError, AttributeError):
        pass
    if member in grp.attrs:
        del grp.attrs[member]


def _write_key(root: "zarr.Group", key: str, val: Any,
               manifest: Dict[str, list], used: Dict[str, set],
               overwrite: bool = False) -> None:
    """Route, write and register one flat key in the store and manifest.

    Parameters
    ----------
    root : zarr.Group
        Store root.
    key : str
        Flat result key.
    val : Any
        Value to store.
    manifest : dict of str to list
        Flat-key -> ``[group, member, kind]`` map, updated in place.
    used : dict of str to set
        Per-group set of taken member names, to avoid clashes; updated in place.
    overwrite : bool, optional
        If True, remove any existing member before writing. Default is False.
    """
    _check_key(key)
    group, member = _route(key)
    if key in manifest:                            # keep a stable overwrite target
        group, member = manifest[key][0], manifest[key][1]
    else:
        taken = used.setdefault(group, set())
        if member in taken:                        # avoid clobbering on rare clash
            member = key
        taken.add(member)
    grp = _ensure_group(root, group)
    if overwrite:
        _remove_member(grp, member)
    manifest[key] = [group, member, _write_value(grp, member, val)]


def _read_entry(root: "zarr.Group", entry: list) -> Any:
    """Materialise one manifest entry into its python/numpy value.

    Parameters
    ----------
    root : zarr.Group
        Store root.
    entry : list
        Manifest entry ``[group, member, kind]``.

    Returns
    -------
    Any
        The materialised value (ndarray, ragged list, sparse matrix,
        sparse_seq list, or attr value), depending on ``kind``.
    """
    group, member, kind = entry
    if kind == "array":
        return root[f"{group}/{member}"][...]
    if kind == "ragged":
        return _read_ragged(root[f"{group}/{member}"])
    if kind == "sparse":
        return _read_sparse(root[f"{group}/{member}"])
    if kind == "sparse_seq":
        return _read_sparse_seq(root[f"{group}/{member}"])
    return _from_attr(root[group].attrs[member])


def write_results(data: Dict[str, Any], store_path: Union[str, Path]) -> None:
    """Serialize a results dict into a fresh native-group Zarr store.

    Arrays/ragged/sparse go to Zarr; scalars/params/strings to group attributes.
    A manifest in the root attrs records each flat key's home for exact,
    O(1) dict-style read-back. Round-trips dtype and NaN.

    Parameters
    ----------
    data : dict of str to Any
        Flat results mapping to serialize.
    store_path : str or Path
        Destination ``data.zarr`` path; overwritten if it exists.
    """
    root = zarr.open_group(str(Path(store_path)), mode="w")
    manifest: Dict[str, list] = {}
    used: Dict[str, set] = {}
    for key, val in data.items():
        _write_key(root, key, val, manifest, used)
    root.attrs[_MANIFEST] = manifest
    root.attrs[_VERSION] = FORMAT_VERSION
    logger.debug("wrote %d keys to %s", len(data), store_path)


def update_results(store_path: Union[str, Path], mapping: Dict[str, Any],
                   deleted: Iterable[str] = ()) -> None:
    """Incrementally update an existing (or new) store.

    Rewrites only the affected members plus the root manifest.

    Parameters
    ----------
    store_path : str or Path
        Path to the ``data.zarr`` store; created if missing.
    mapping : dict of str to Any
        Flat keys to write or overwrite.
    deleted : iterable of str, optional
        Flat keys to drop. Default is ``()``.
    """
    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="a" if store_path.exists() else "w")
    manifest = dict(root.attrs.get(_MANIFEST, {}))
    used: Dict[str, set] = {}
    for g, m, _k in manifest.values():
        used.setdefault(g, set()).add(m)
    for key in deleted:
        entry = manifest.pop(key, None)
        if entry is not None:
            g, m, _k = entry
            _remove_member(_ensure_group(root, g), m)
            used.get(g, set()).discard(m)
    for key, val in mapping.items():
        _write_key(root, key, val, manifest, used, overwrite=True)
    root.attrs[_MANIFEST] = manifest
    root.attrs[_VERSION] = FORMAT_VERSION


def export_to_json(data, path: Union[str, Path], *,
                   keys: Optional[Iterable[str]] = None,
                   include_arrays: bool = True) -> Path:
    """Project a results mapping to a JSON file.

    Reuses :meth:`IOUtils.json_serialize`, so the output is in the
    ``structure.json`` format and is reloadable by :meth:`IOUtils.json_deserialize`.

    Parameters
    ----------
    data : Mapping
        Results mapping (e.g. a :class:`Results` accessor or a plain dict).
    path : str or Path
        Output JSON path.
    keys : iterable of str or None, optional
        Subset of keys to export (e.g. one group's keys). Default is None
        (all keys).
    include_arrays : bool, optional
        If False, skip large arrays (readable params/scalars dump). Default
        is True.

    Returns
    -------
    Path
        The written JSON path.
    """
    from sarcasm.io.ioutils import IOUtils
    keys = list(data.keys()) if keys is None else list(keys)
    out = {}
    for k in keys:
        v = data[k]
        if not include_arrays and isinstance(v, np.ndarray) and v.size > _INLINE_MAX:
            continue
        out[k] = v
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    IOUtils.json_serialize(out, str(path))
    return Path(path)


# --------------------------------------------------------------------------- #
# logical namespace
# --------------------------------------------------------------------------- #
def _tree_from_keys(keys: Iterable[str]) -> Dict[str, Any]:
    """Build the grouped namespace from flat keys via :func:`_route`.

    The tree is derived from the routing schema, not from the physical Zarr
    tree, so it is identical across store vintages, covers staged (unflushed)
    keys, and cannot expose non-result groups such as ``masks/``. Leaves hold
    the **flat key**, so every read goes through the single materialising path
    :meth:`Results.__getitem__`.

    Parameters
    ----------
    keys : iterable of str
        Flat result keys.

    Returns
    -------
    dict
        Nested ``{segment: {member: flat_key | subtree}}`` mapping.
    """
    root: Dict[str, Any] = {}
    for key in keys:
        group, member = _route(key)
        node = root
        for seg in group.split("/"):
            child = node.setdefault(seg, {})
            if not isinstance(child, dict):
                logger.warning("namespace clash at %r; %r stays reachable via data[%r]",
                               seg, key, key)
                node = None
                break
            node = child
        if node is None:
            continue
        if isinstance(node.get(member), dict):
            logger.warning("%r is shadowed by subgroup %r; use data[%r]",
                           key, member, key)
            continue
        node[member] = key
    return root


class _KeyInfo(NamedTuple):
    """Cheap, metadata-only description of one stored key."""

    key: str
    group: str
    kind: str
    shape: Optional[tuple]
    dtype: Optional[str]
    staged: bool


_DTYPE_ABBREV = {"float64": "f64", "float32": "f32", "float16": "f16",
                 "int64": "i64", "int32": "i32", "int16": "i16", "int8": "i8",
                 "uint64": "u64", "uint32": "u32", "uint16": "u16", "uint8": "u8",
                 "bool": "bool"}


def _fmt_dtype(dtype: Optional[str]) -> str:
    """Abbreviate a dtype name for compact table output."""
    if not dtype:
        return ""
    return _DTYPE_ABBREV.get(dtype, dtype)


def _fmt_shape(shape: Optional[tuple]) -> str:
    """Render a shape tuple compactly (``''`` for scalars/unknown)."""
    if shape is None or len(shape) == 0:
        return ""
    return "(" + ",".join(str(d) for d in shape) + ")"


def _fmt_spec(info: "_KeyInfo") -> str:
    """One compact ``shape dtype`` cell, disambiguating the storage kinds.

    Ragged and sparse-sequence values are per-frame lists, so their leading
    dimension is a frame count, not an array axis — render them as
    ``ragged[500]`` rather than ``(500)``.
    """
    dtype = _fmt_dtype(info.dtype)
    n = info.shape[0] if info.shape else None
    if info.kind == "ragged":
        return f"ragged[{n}] {dtype}".strip()
    if info.kind == "sparse_seq":
        return f"sparse[{n}] {dtype}".strip()
    if info.kind == "sparse":
        return f"sparse{_fmt_shape(info.shape)} {dtype}".strip()
    return f"{_fmt_shape(info.shape)} {dtype}".strip()


def _describe_value(val: Any) -> Tuple[str, Optional[tuple], Optional[str]]:
    """Infer ``(kind, shape, dtype)`` from an in-memory (staged) value."""
    if isinstance(val, np.ndarray):
        return ("array" if val.size > _INLINE_MAX else "attr"), val.shape, val.dtype.name
    if sparse.issparse(val):
        return "sparse", tuple(val.shape), val.dtype.name
    if _is_sparse_seq(val):
        return "sparse_seq", (len(val),), None
    if _is_ragged(val):
        first = next((np.asarray(x) for x in val if x is not None), None)
        return "ragged", (len(val),), (first.dtype.name if first is not None else None)
    if isinstance(val, (list, tuple)):
        return "attr", (len(val),), None
    if isinstance(val, np.generic):
        return "attr", (), val.dtype.name
    return "attr", (), type(val).__name__


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def _html_escape(text: Any) -> str:
    """Minimal HTML escaping for repr tables."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _display_name(key: str) -> str:
    """Registry label for ``key``, blank when it just restates the key.

    ``params.group_tracks.by`` is labelled ``by``, which adds nothing next to
    the key itself — drop it rather than print the same word twice.
    """
    name = pretty_name(key)
    return "" if name == key or key.endswith(name) else name


class KeyTable(list):
    """A list of result keys that prints as a table.

    Subclasses :class:`list`, so it iterates and indexes as plain key strings
    (``for k in sarc.data.find('slen'): sarc.data[k]``) while rendering a
    readable key/kind/shape/dtype/name table at the REPL and in Jupyter.
    """

    def __init__(self, keys: Iterable[str], results: "Results" = None,
                 pattern: str = "", searched: int = 0):
        super().__init__(keys)
        self._results = results
        self._pattern = pattern
        self._searched = searched

    def _rows(self) -> List[_KeyInfo]:
        if self._results is None:
            return []
        return [self._results._key_info(k) for k in self]

    def __repr__(self) -> str:
        if not self:
            what = f" match {self._pattern!r}" if self._pattern else ""
            return f"no keys{what} ({self._searched} keys searched)"
        rows = self._rows()
        if not rows:
            return super().__repr__()
        wk = max(len(r.key) for r in rows)
        ws = max(len(_fmt_spec(r)) for r in rows)
        out = []
        for r in rows:
            spec = _fmt_spec(r)
            out.append(f"  {r.key:<{wk}}  {spec:<{ws}}  {_display_name(r.key)}".rstrip())
        head = f"{len(self)} key{'s' if len(self) != 1 else ''}"
        if self._pattern:
            head += f" matching {self._pattern!r}"
        return head + "\n" + "\n".join(out)

    def _repr_html_(self) -> str:
        if not self:
            what = f" match <code>{_html_escape(self._pattern)}</code>" if self._pattern else ""
            return f"<p><em>no keys{what} ({self._searched} keys searched)</em></p>"
        cells = "".join(
            "<tr>"
            f"<td><code>{_html_escape(r.key)}</code></td>"
            f"<td>{_html_escape(r.kind)}</td>"
            f"<td><code>{_html_escape(_fmt_spec(r))}</code></td>"
            f"<td>{_html_escape(_display_name(r.key))}</td>"
            "</tr>"
            for r in self._rows())
        return ("<table><thead><tr><th>key</th><th>kind</th><th>shape</th>"
                "<th>name</th></tr></thead>"
                f"<tbody>{cells}</tbody></table>")


class FeatureInfo:
    """What one result key is, where it lives, and how it is stored.

    Returned by :meth:`Results.describe`. ``registry`` is None when the key is
    written by the pipeline but not documented in :mod:`sarcasm.features`.
    """

    __slots__ = ("key", "name", "description", "function", "registry",
                 "kind", "group", "shape", "dtype")

    def __init__(self, key, info: _KeyInfo, entry: Optional[dict]):
        entry = entry or {}
        self.key = key
        self.name = entry.get("name", key)
        self.description = entry.get("description")
        self.function = entry.get("function")
        self.registry = entry.get("registry")
        self.kind = info.kind
        self.group = info.group
        self.shape = info.shape
        self.dtype = info.dtype

    def __repr__(self) -> str:
        spec = _fmt_spec(_KeyInfo(self.key, self.group, self.kind, self.shape,
                                  self.dtype, False))
        lines = [f"{self.key}",
                 f"  stored     {self.group} · {spec or self.kind}"]
        if self.registry:
            lines.append(f"  name       {self.name}")
            lines.append(f"  written by {self.function}")
            lines.append("  " + _wrap_field("about", self.description or ""))
        else:
            lines.append("  registry   no entry — this key is written by the pipeline but "
                         "not documented in sarcasm.features")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        spec = _fmt_spec(_KeyInfo(self.key, self.group, self.kind, self.shape,
                                  self.dtype, False))
        rows = [("stored", f"<code>{_html_escape(self.group)}</code> · "
                           f"{_html_escape(spec or self.kind)}")]
        if self.registry:
            rows += [("name", _html_escape(self.name)),
                     ("written by", f"<code>{_html_escape(self.function)}</code>"),
                     ("about", _html_escape(self.description or ""))]
        else:
            rows.append(("registry", "<em>no entry in sarcasm.features</em>"))
        body = "".join(f"<tr><th align='left'>{k}</th><td>{v}</td></tr>" for k, v in rows)
        return (f"<p><code><b>{_html_escape(self.key)}</b></code></p>"
                f"<table><tbody>{body}</tbody></table>")


def _wrap_field(label: str, text: str, width: int = 88) -> str:
    """Wrap ``text`` under a fixed-width label, continuation lines aligned.

    The caller prefixes two spaces, so continuation lines are indented by
    ``2 + 9 + 2`` to line up under the start of ``text``.
    """
    body = textwrap.fill(text, width=width, initial_indent="",
                         subsequent_indent=" " * 13)
    return f"{label:<9}  {body}"


# --------------------------------------------------------------------------- #
# accessor
# --------------------------------------------------------------------------- #
class _NodeView:
    """One node of the grouped namespace (e.g. ``sarc.data.structure.sarcomere``).

    Read-only and deliberately free of public methods: every public name here
    would be a name a future result key could not use. All helpers
    (``keys``, ``find``, ``describe``, …) live on :class:`Results`.
    """

    __slots__ = ("_res", "_node", "_path")

    def __init__(self, res: "Results", node: Dict[str, Any], path: str):
        object.__setattr__(self, "_res", res)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_path", path)

    def __getitem__(self, name: str) -> Any:
        child = self._node[name]
        if isinstance(child, dict):
            return _NodeView(self._res, child, f"{self._path}.{name}")
        return self._res[child]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self._path}' has no member {name!r}. Available: "
                f"{', '.join(self.__dir__())}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"'{self._path}' is read-only. Write with "
            f"sarc.data['<flat_key>'] = ... (then sarc.commit()).")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"'{self._path}' is read-only; use del sarc.data['<flat_key>'].")

    def __contains__(self, name: str) -> bool:
        return name in self._node

    def __iter__(self):
        return iter(sorted(self._node))

    def __len__(self) -> int:
        return len(self._node)

    def __dir__(self) -> List[str]:
        return sorted(self._node)

    def _flat_keys(self) -> List[str]:
        """All flat keys at or below this node, in namespace order."""
        out: List[str] = []
        for name in sorted(self._node):
            child = self._node[name]
            if isinstance(child, dict):
                out.extend(_NodeView(self._res, child, f"{self._path}.{name}")._flat_keys())
            else:
                out.append(child)
        return out

    def _direct_keys(self) -> List[str]:
        """Flat keys of this node itself, excluding its subgroups."""
        return [self._node[n] for n in sorted(self._node)
                if not isinstance(self._node[n], dict)]

    def _subgroup_summary(self) -> List[str]:
        """``name (n)`` for each subgroup, so drilling down is guided."""
        out = []
        for name in sorted(n for n, v in self._node.items() if isinstance(v, dict)):
            n_keys = len(_NodeView(self._res, self._node[name],
                                   f"{self._path}.{name}")._flat_keys())
            out.append(f"{name} ({n_keys})")
        return out

    def __repr__(self) -> str:
        # only this node's own keys — subgroups are listed, not expanded, so
        # `sarc.data.structure` stays readable instead of dumping 165 rows
        direct = self._direct_keys()
        subs = self._subgroup_summary()
        head = f"<{self._path} · {len(self._flat_keys())} keys>"
        parts = [head]
        if subs:
            parts.append(f"  subgroups: {', '.join(subs)}")
        if direct:
            parts.append(repr(KeyTable(direct, self._res)))
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        subs = self._subgroup_summary()
        head = f"<p><code><b>{_html_escape(self._path)}</b></code></p>"
        if subs:
            head += f"<p>subgroups: <code>{_html_escape(', '.join(subs))}</code></p>"
        direct = self._direct_keys()
        return head + (KeyTable(direct, self._res)._repr_html_() if direct else "")


class Results(MutableMapping):
    """Lazy, Zarr-backed results of one recording — ``SarcAsM.data``.

    A key is its path, so there is one name with two spellings::

        sarc.data['motion.tracks.slen']  ==  sarc.data.motion.tracks.slen
        sarc.data['structure.sarcomere.oop']
        sarc.data['params.detect_sarcomeres.frames']

    ``structure`` holds per-frame morphology, ``motion`` everything derived from
    the sarcomere tracks, and ``params`` what each analysis step ran with.

    Reads materialise on first access and are cached; writes are staged in
    memory and persisted incrementally by :meth:`flush` (only changed members
    are rewritten). **Attributes are read-only** — write with
    ``sarc.data['key'] = value`` followed by ``sarc.commit()``. No read ever
    touches the store on disk.

    To explore: ``print(sarc.data)``, ``sarc.data.keys()``,
    :meth:`find`, :meth:`describe`, and tab completion (``sarc.data.<TAB>``).
    For chunk-wise access to a large array without materialising it, use
    :meth:`handle`.

    Parameters
    ----------
    store_path : str or Path
        Path to the results store (the ``sarcasm/`` group of the OME-Zarr
        container). It need not exist yet.
    initial : dict or None, optional
        Key/value pairs staged on construction. Default is None.
    """

    def __init__(self, store_path: Union[str, Path], *, initial: Optional[dict] = None):
        self._path = Path(store_path)
        self._cache: Dict[str, Any] = {}
        self._staged: Dict[str, Any] = {}
        self._dirty: set = set()
        self._deleted: set = set()
        self._tree_cache: Dict[str, Any] = {}
        self._tree_stamp: Optional[tuple] = None
        self._open()
        if initial:
            for k, v in dict(initial).items():
                self[k] = v

    def _open(self) -> None:
        """(Re)open the backing store read-only and load its manifest."""
        # shape/dtype of persisted keys can only change through a flush, which
        # comes back through here, so the info cache is invalidated for free
        self._info_cache: Dict[str, _KeyInfo] = {}
        if self._path.exists():
            self._root = zarr.open_group(str(self._path), mode="r")
            self._manifest = dict(self._root.attrs.get(_MANIFEST, {}))
        else:
            self._root = None
            self._manifest = {}

    # -- key sets ---------------------------------------------------------- #
    def _live_keys(self) -> set:
        """Currently visible keys (persisted + staged, minus deleted)."""
        return (set(self._manifest) | set(self._staged)) - self._deleted

    def _ordered_keys(self) -> List[str]:
        """Visible keys in a stable order: manifest (write) order, then staged."""
        deleted = self._deleted
        out = [k for k in self._manifest if k not in deleted]
        seen = set(out)
        out += [k for k in self._staged if k not in seen and k not in deleted]
        return out

    def _tree(self) -> Dict[str, Any]:
        """The grouped namespace, rebuilt when the key set changes."""
        stamp = tuple(self._ordered_keys())
        if self._tree_stamp != stamp:
            self._tree_cache = _tree_from_keys(stamp)
            self._tree_stamp = stamp
        return self._tree_cache

    # -- MutableMapping ----------------------------------------------------- #
    def __getitem__(self, key: str) -> Any:
        if key in self._deleted:
            raise KeyError(key)
        if key in self._staged:
            return self._staged[key]
        if key in self._cache:
            return self._cache[key]
        if key in self._manifest:
            val = _read_entry(self._root, self._manifest[key])
            self._cache[key] = val
            return val
        raise KeyError(key)

    def __setitem__(self, key: str, val: Any) -> None:
        _check_key(key)
        self._staged[key] = val
        self._dirty.add(key)
        self._deleted.discard(key)
        self._cache.pop(key, None)

    def __delitem__(self, key: str) -> None:
        if key not in self._live_keys():
            raise KeyError(key)
        self._deleted.add(key)
        self._staged.pop(key, None)
        self._dirty.discard(key)
        self._cache.pop(key, None)

    def __iter__(self):
        return iter(self._ordered_keys())

    def __len__(self) -> int:
        return len(self._ordered_keys())

    def __contains__(self, key: str) -> bool:
        return key in self._live_keys()

    def keys(self) -> List[str]:
        """Flat result keys, in write order (never a scrambled set)."""
        return self._ordered_keys()

    def to_dict(self) -> Dict[str, Any]:
        """Materialise every key into a plain flat-key dict."""
        return {k: self[k] for k in self._ordered_keys()}

    # -- attribute access --------------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        # Keys are dotted paths, so they are never bare identifiers and can never
        # shadow a method here; only namespaces resolve as attributes.
        if name.startswith("_"):
            raise AttributeError(name)
        node = self._tree().get(name)
        if isinstance(node, dict):
            return _NodeView(self, node, f"data.{name}")
        raise AttributeError(self._attr_error(name))

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"results are read-only as attributes. Write with "
            f"sarc.data[{name!r}] = ... (then sarc.commit()).")

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        raise AttributeError(f"results are read-only as attributes; "
                             f"use del sarc.data[{name!r}].")

    def _attr_error(self, name: str) -> str:
        """Build an AttributeError message with close-match suggestions."""
        paths = sorted({p for p, _ in self._group_rows()})
        close = difflib.get_close_matches(name, paths + self._ordered_keys(),
                                          n=3, cutoff=0.5)
        msg = f"no result namespace {name!r} in this store"
        if close:
            msg += f". Did you mean: {', '.join(repr(c) for c in close)}?"
        return (msg + f" — sarc.data holds {', '.join(sorted(self._tree()))}; "
                      f"sarc.data.find({name!r}) searches all {len(self)} keys.")

    def __dir__(self) -> List[str]:
        base = [n for n in super().__dir__() if not n.startswith("_")]
        groups = [g for g, v in self._tree().items() if isinstance(v, dict)]
        return sorted(set(base + groups))

    # -- introspection ------------------------------------------------------ #
    def _key_info(self, key: str) -> _KeyInfo:
        """Describe a key from store metadata alone — never reads a chunk."""
        if key in self._staged:
            kind, shape, dtype = _describe_value(self._staged[key])
            return _KeyInfo(key, _route(key)[0], kind, shape, dtype, True)
        cached = self._info_cache.get(key)
        if cached is not None:
            return cached
        entry = self._manifest.get(key)
        if entry is None:
            raise KeyError(key)
        group, member, kind = entry
        group = _route(key)[0]        # logical home, not a stale physical one
        shape: Optional[tuple] = None
        dtype: Optional[str] = None
        try:
            if kind == "array":
                arr = self._root[f"{group}/{member}"]
                shape, dtype = tuple(arr.shape), arr.dtype.name
            elif kind == "ragged":
                sub = self._root[f"{group}/{member}"]
                shape = (int(sub["none_mask"].shape[0]),)
                dtype = sub["values"].dtype.name
            elif kind in ("sparse", "sparse_seq"):
                sub = self._root[f"{group}/{member}"]
                shape = tuple(sub.attrs.get("shape", ()))
                if kind == "sparse_seq":
                    shape = (int(sub.attrs.get("n", 0)),) + shape
                dtype = sub["data"].dtype.name
            else:
                _kind, shape, dtype = _describe_value(
                    _from_attr(self._root[group].attrs[member]))
        except (KeyError, AttributeError, TypeError):
            pass
        info = _KeyInfo(key, group, kind, shape, dtype, False)
        self._info_cache[key] = info
        return info

    def handle(self, key: str) -> "zarr.Array":
        """Lazy Zarr handle for a chunked array, for row-wise reads.

        Attribute and dict access always materialise, so their type never
        depends on how large the value happened to be. When you explicitly
        want to read one row of a big ``(n_tracks, T)`` array without loading
        all of it, ask for the handle::

            sarc.data.handle('motion.tracks.slen')[5]

        Parameters
        ----------
        key : str
            Flat result key.

        Returns
        -------
        zarr.Array
            The on-disk array (nothing is read until you index it).

        Raises
        ------
        KeyError
            If ``key`` is not in the store.
        TypeError
            If the value is stored inline (small values live in group attrs and
            have no handle) — read it with ``sarc.data[key]``.
        RuntimeError
            If ``key`` has unflushed staged writes — call ``sarc.commit()``
            first. Reading never writes to disk.
        """
        if key not in self._live_keys():
            raise KeyError(key)
        if key in self._staged:
            raise RuntimeError(
                f"{key!r} has unflushed writes; call sarc.commit() before asking "
                f"for a handle (reads never write to disk).")
        group, member, kind = self._manifest[key]
        if kind != "array":
            raise TypeError(
                f"{key!r} is stored inline (kind={kind!r}) and has no zarr handle; "
                f"read it with sarc.data[{key!r}].")
        return self._root[f"{group}/{member}"]

    def find(self, pattern: str = "", *, group: Optional[str] = None,
             regex: bool = False) -> KeyTable:
        """Search the result keys.

        Parameters
        ----------
        pattern : str, optional
            Case-insensitive substring, or a glob if it contains ``*``, ``?``
            or ``[``. Default is ``''`` (every key).
        group : str or None, optional
            Restrict to one namespace, given as a full path
            (``'motion/pool'``) or its last segment (``'pool'``). Default is
            None. Note that ``domain``, ``myofibril`` and ``loi`` name a group
            on *both* branches, so the last-segment form returns the union —
            use the full path to pick one.
        regex : bool, optional
            Treat ``pattern`` as a regular expression. Default is False.

        Returns
        -------
        KeyTable
            A list of matching keys that prints as a table.

        Examples
        --------
        >>> sarc.data.find('slen')                    # doctest: +SKIP
        >>> sarc.data.find('motion.*.beating_rate')   # doctest: +SKIP
        >>> sarc.data.find(group='motion/tracks')     # doctest: +SKIP
        """
        keys = self._ordered_keys()
        searched = len(keys)
        if group:
            want = group.strip("/")
            keys = [k for k in keys
                    if _route(k)[0] == want or _route(k)[0].split("/")[-1] == want]
        if regex:
            rx = re.compile(pattern, re.IGNORECASE)
            keys = [k for k in keys if rx.search(k)]
        elif any(c in pattern for c in "*?["):
            low = pattern.lower()
            keys = [k for k in keys if fnmatch.fnmatchcase(k.lower(), low)]
        elif pattern:
            low = pattern.lower()
            keys = [k for k in keys if low in k.lower()]
        return KeyTable(keys, self, pattern, searched)

    def describe(self, key: Optional[str] = None) -> Union[FeatureInfo, KeyTable]:
        """Explain what a result key means, how it is stored and who wrote it.

        Parameters
        ----------
        key : str or None, optional
            Flat result key. Default is None, which tables every key.

        Returns
        -------
        FeatureInfo or KeyTable
            Description of ``key``, or a table of all keys.

        Raises
        ------
        KeyError
            If ``key`` is not in this store.
        """
        if key is None:
            return self.find("")
        if key not in self._live_keys():
            raise KeyError(self._attr_error(key))
        return FeatureInfo(key, self._key_info(key), describe_key(key))

    # -- persistence -------------------------------------------------------- #
    def flush(self) -> None:
        """Persist staged writes and deletes to the store (incremental)."""
        if not self._dirty and not self._deleted:
            return
        mapping = {k: self._staged[k] for k in self._dirty}
        update_results(self._path, mapping, deleted=self._deleted)
        self._cache.update(mapping)
        self._staged = {k: v for k, v in self._staged.items() if k not in self._dirty}
        self._dirty.clear()
        self._deleted.clear()
        self._open()

    def ensure_store(self) -> None:
        """Make sure a (possibly empty) store exists on disk."""
        self.flush()
        if not self._path.exists():
            write_results({}, self._path)
            self._open()

    def set_root_attr(self, name: str, value: Any) -> None:
        """Write a root-level store attribute (e.g. mirrored metadata).

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            JSON-serializable attribute value.
        """
        self.ensure_store()
        root = zarr.open_group(str(self._path), mode="a")
        root.attrs[name] = value

    # -- display ------------------------------------------------------------ #
    def _group_rows(self) -> List[Tuple[str, List[str]]]:
        """Flatten the namespace to ``(dotted_group_path, [keys])`` rows.

        Ordered for reading rather than alphabetically: the two result
        namespaces first (``structure``, then ``motion``), provenance last, and
        within each the subgroups in pipeline order.
        """
        rows: List[Tuple[str, List[str]]] = []

        def rank(top: str, name: str):
            order = _GROUP_ORDER.get(top, ())
            return (order.index(name) if name in order else len(order)), name

        def walk(node: Dict[str, Any], path: str, top: str) -> None:
            leaves = [v for v in node.values() if isinstance(v, str)]
            if leaves:
                rows.append((path, leaves))
            for name in sorted((k for k, v in node.items() if isinstance(v, dict)),
                               key=lambda n: rank(top, n)):
                walk(node[name], f"{path}.{name}", top)

        root = self._tree()
        for top in sorted(root, key=lambda x: (TOP_GROUPS.index(x) if x in TOP_GROUPS
                                               else len(TOP_GROUPS), x)):
            if isinstance(root[top], dict):
                walk(root[top], top, top)
        return rows

    def __repr__(self) -> str:
        try:
            return self._render()
        except Exception as e:                                  # never unprintable
            logger.debug("Results repr failed: %s", e)
            return f"<Results {len(self._manifest)} keys @ {self._path}>"

    def _render(self, max_shown: int = 3) -> str:
        n_staged = len(self._dirty)
        head = (f"<Results · {len(self)} keys"
                + (f" ({n_staged} unsaved)" if n_staged else "")
                + f" · {self._path.parent.name}/{self._path.name}>")
        rows = self._group_rows()
        if not rows:
            return head + "\n  (empty — run an analysis, e.g. sarc.detect_sarcomeres())"
        body = [(q, ks) for q, ks in rows if q.partition(".")[0] != "params"]
        params = [(q, ks) for q, ks in rows if q.partition(".")[0] == "params"]
        width = max([len(q.partition(".")[2] or q) for q, _ in body] + [6])
        lines, seen = [head], None
        for path, keys in body:
            top, _, sub = path.partition(".")
            if top != seen:                       # one header per namespace
                n_top = sum(len(k) for q, k in body if q.partition(".")[0] == top)
                lines.append(f"  {top}  ·  {n_top} keys")
                seen = top
            shown = [f"{k.rpartition('.')[2]} {_fmt_spec(self._key_info(k))}".strip()
                     for k in keys[:max_shown]]
            more = f", +{len(keys) - max_shown}" if len(keys) > max_shown else ""
            lines.append(f"    {sub or '·':<{width}}  {len(keys):>4}  "
                         f"{', '.join(shown)}{more}")
        if params:
            steps = [q.partition(".")[2] for q, _ in params]
            n_keys = sum(len(ks) for _, ks in params)
            more = f", +{len(steps) - max_shown}" if len(steps) > max_shown else ""
            lines.append(f"  params  ·  {n_keys} keys")
            lines.append(f"    {len(steps)} steps: {', '.join(steps[:max_shown])}{more}")
        lines.append("  sarc.data.find('slen')  ·  "
                     "sarc.data.describe('structure.sarcomere.oop')")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        blocks, seen = [], None
        for path, keys in self._group_rows():
            top, _, sub = path.partition(".")
            if top != seen:                        # one heading per namespace
                blocks.append(f"<p><b>{_html_escape(top)}</b></p>")
                seen = top
            opened = "" if top == "params" else " open"
            table = KeyTable(keys, self)._repr_html_()
            blocks.append(f"<details{opened}><summary><code>{_html_escape(sub or top)}"
                          f"</code> — {len(keys)} keys</summary>{table}</details>")
        head = (f"<p><b>Results</b> · {len(self)} keys · "
                f"<code>{_html_escape(self._path.parent.name)}/"
                f"{_html_escape(self._path.name)}</code></p>")
        hint = ("<p><em>A key is its path: "
                "<code>sarc.data['motion.tracks.slen']</code> and "
                "<code>sarc.data.motion.tracks.slen</code> are the same value. "
                "Search with <code>sarc.data.find('slen')</code>.</em></p>")
        return head + "".join(blocks) + hint
