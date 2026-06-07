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

"""Single Zarr store for SarcAsM analysis + track results (proof of concept).

This replaces the giant text ``structure.json`` with one self-describing Zarr
store, ``<data_dir>/data.zarr``. The physical layout mirrors the logical
structure, so the store is browsable with any Zarr tool::

    data.zarr/
      params/<step>/        (attrs)   analysis parameters, one subgroup per step
      tracks/                         dense per-track block, row-chunked + zstd
        slen positions_um positions_px orientations snapped detection_id
        midline_id ids start_frame lengths group_id ...   each (n_tracks, T) | (n_tracks,)
      motion/                         per-track/-vector motion field
        displacement_magnitude displacement_along_sarcomere ... velocity_magnitude flow_at_vectors
      structure/                      morphology / per-frame analysis
        sarcomere/  vectors/  domain/  myofibril/  pool/  mband/

Small things (scalars, params, strings) live as JSON-shaped Zarr **attributes**
(human-readable inside each group's ``zarr.json``); large numeric arrays live as
binary Zarr arrays next to them. JSON becomes an explicit *export*, not the
storage format.

Access (see :class:`Results`)::

    r = Results("…/data.zarr")
    r.tracks.slen               # lazy zarr array — nothing read yet
    r.tracks.slen[5]            # one track, reads a single chunk
    r.structure.sarcomere.oop   # eager value
    r.params.track_sarcomere_vectors.max_disp_along_px
    r['tracks_slen']            # legacy flat-key dict access (materialised)

Scope of this POC: analysis + track data (the contents of the old
``structure.json``). Image masks / flow / metadata migrate in later phases.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import zarr
from scipy import sparse

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
_INLINE_MAX = 256          # ndarrays with <= this many elements go inline (attrs)
_ROW_CHUNK = 512           # chunk size along axis 0 for big arrays (lazy per-row reads)
_KIND = "_kind"            # subgroup attr marking ragged/sparse leaves
_MANIFEST = "_manifest"    # root attr: flat_key -> [group, member, kind]
_VERSION = "_format_version"


# --------------------------------------------------------------------------- #
# routing schema: flat result key -> (group path, member name)
# --------------------------------------------------------------------------- #
def _route(key: str) -> Tuple[str, str]:
    """Map a legacy flat key to its home in the store.

    Parameters
    ----------
    key : str
        Legacy flat result key (e.g. ``'tracks_slen'``, ``'sarcomere_oop'``).

    Returns
    -------
    tuple of (str, str)
        The ``(group_path, member_name)`` where the value is stored.
    """
    if key.startswith("params."):
        parts = key.split(".")
        step = parts[1]
        member = ".".join(parts[2:]) if len(parts) > 2 else step
        return f"params/{step}", member
    if key.startswith("tracks_"):
        return "tracks", key[len("tracks_"):]
    if key.startswith("track_"):
        return "tracks", key[len("track_"):]
    if key in ("n_tracks", "n_merges", "n_groups", "group_kind", "grouping_hash"):
        return "tracks", key
    # motion-field keys keep their full descriptive names (avoid magnitude clashes)
    if key.startswith(("displacement_", "velocity_", "flow_at", "motionfield")):
        return "motion", key
    if key.startswith("sarcomere_"):
        return "structure/sarcomere", key[len("sarcomere_"):]
    if key.startswith(("pos_vectors", "midline_")):
        return "structure/vectors", key
    if key == "domains" or key.startswith("domain_"):
        return "structure/domain", key[len("domain_"):] if key.startswith("domain_") else key
    if key.startswith(("myof_", "myofibril_")):
        return "structure/myofibril", key.split("_", 1)[1]
    if key.startswith("pool_"):
        return "structure/pool", key[len("pool_"):]
    if key.startswith("mband_"):
        return "structure/mband", key[len("mband_"):]
    return "structure", key


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
        Legacy flat result key.
    val : Any
        Value to store.
    manifest : dict of str to list
        Flat-key -> ``[group, member, kind]`` map, updated in place.
    used : dict of str to set
        Per-group set of taken member names, to avoid clashes; updated in place.
    overwrite : bool, optional
        If True, remove any existing member before writing. Default is False.
    """
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
        Flat results mapping (legacy keys) to serialize.
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
    """Project a results mapping to a legacy-format JSON file.

    Reuses :meth:`IOUtils.json_serialize`, so the output is byte-compatible with
    the old ``structure.json`` and reloadable by old code.

    Parameters
    ----------
    data : Mapping
        Results mapping (e.g. :class:`Results` or :class:`ResultsDict`).
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
    IOUtils.json_serialize(out, str(path))
    return Path(path)


# --------------------------------------------------------------------------- #
# accessor
# --------------------------------------------------------------------------- #
def _resolve_member(grp: "zarr.Group", name: str):
    """Resolve a member name within a group to a value or namespace.

    Parameters
    ----------
    grp : zarr.Group
        Group to look in.
    name : str
        Member name (array, subgroup or attr).

    Returns
    -------
    Any
        A lazy zarr array, a materialised ragged/sparse value, a nested
        :class:`_GroupView`, or an attr value.

    Raises
    ------
    KeyError
        If no member ``name`` exists.
    """
    if name in grp.array_keys():
        return grp[name]                          # lazy zarr array
    if name in grp.group_keys():
        sub = grp[name]
        kind = sub.attrs.get(_KIND)
        if kind == "ragged":
            return _read_ragged(sub)
        if kind == "sparse":
            return _read_sparse(sub)
        if kind == "sparse_seq":
            return _read_sparse_seq(sub)
        return _GroupView(sub)                    # nested namespace
    if name in grp.attrs:
        return _from_attr(grp.attrs[name])
    raise KeyError(name)


class _GroupView:
    """Attribute view over one Zarr group.

    Subgroups resolve to nested namespaces, arrays to lazy handles, and attrs
    to values.

    Parameters
    ----------
    grp : zarr.Group
        The group to wrap.
    """

    __slots__ = ("_grp",)

    def __init__(self, grp: "zarr.Group"):
        object.__setattr__(self, "_grp", grp)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return _resolve_member(self._grp, name)
        except KeyError:
            raise AttributeError(
                f"group {self._grp.path!r} has no member {name!r}") from None

    def __getitem__(self, name: str):
        return _resolve_member(self._grp, name)

    def __dir__(self):
        members = (list(self._grp.array_keys()) + list(self._grp.group_keys())
                   + [k for k in self._grp.attrs if not k.startswith("_")])
        return sorted(members)

    def __repr__(self):
        return f"<group {self._grp.path or '/'}: {self.__dir__()}>"


class Results:
    """Lazy attribute-and-dict accessor over a ``data.zarr`` results store.

    Read-only. Supports both grouped attribute access (``r.tracks.slen``,
    nested namespaces, lazy zarr arrays) and legacy flat-key dict access
    (``r['tracks_slen']``, materialised numpy).

    Parameters
    ----------
    store_path : str or Path
        Path to the ``data.zarr`` store.
    """

    def __init__(self, store_path: Union[str, Path]):
        self._root = zarr.open_group(str(Path(store_path)), mode="r")
        self._manifest: Dict[str, list] = dict(self._root.attrs.get(_MANIFEST, {}))
        self._view = _GroupView(self._root)

    # -- dict interface (legacy flat keys, materialised numpy) ------------- #
    def __getitem__(self, key: str) -> Any:
        """Materialise the value for a legacy flat ``key``.

        Parameters
        ----------
        key : str
            Legacy flat result key.

        Returns
        -------
        Any
            The materialised value.

        Raises
        ------
        KeyError
            If ``key`` is not in the manifest.
        """
        if key not in self._manifest:
            raise KeyError(key)
        return _read_entry(self._root, self._manifest[key])

    @property
    def metadata(self) -> Optional[dict]:
        """Image metadata mirrored into the store root attrs, if present."""
        meta = self._root.attrs.get("metadata")
        return _from_attr(meta) if meta is not None else None

    def __contains__(self, key: str) -> bool:
        """Return True if ``key`` is a known legacy flat key."""
        return key in self._manifest

    def keys(self):
        """Return the list of legacy flat keys in the store."""
        return list(self._manifest.keys())

    def get(self, key: str, default=None):
        """Return the value for ``key``, or ``default`` if absent.

        Parameters
        ----------
        key : str
            Legacy flat key.
        default : Any, optional
            Value returned if ``key`` is absent. Default is None.

        Returns
        -------
        Any
            The materialised value or ``default``.
        """
        return self[key] if key in self._manifest else default

    def to_dict(self) -> Dict[str, Any]:
        """Materialise the whole store into a plain flat-key dict.

        Returns
        -------
        dict of str to Any
            All keys mapped to their materialised values.
        """
        return {k: self[k] for k in self._manifest}

    # -- attribute / namespace interface (native group tree) --------------- #
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return _resolve_member(self._root, name)
        except KeyError:
            raise AttributeError(f"Results has no group/key {name!r}") from None

    def __dir__(self):
        return self._view.__dir__() + ["keys", "to_dict", "get"]

    def __repr__(self):
        return (f"<Results {len(self._manifest)} keys, "
                f"groups={sorted(self._root.group_keys())}>")


class ResultsDict(MutableMapping):
    """Lazy, dict-compatible backing for ``SarcAsM.data`` over ``data.zarr``.

    Reads materialise (numpy/objects) on first access and cache; writes stage
    in memory and persist incrementally via :meth:`flush` (only the changed
    members are rewritten). Implements the full ``dict`` surface the codebase
    uses (``[]``, ``get``, ``update``, ``keys``, ``pop``, ``in``, iteration), so
    it is a drop-in for the old plain dict. For the ergonomic grouped/lazy view
    use :meth:`view` (``SarcAsM.results``).

    Parameters
    ----------
    store_path : str or Path
        Path to the ``data.zarr`` store (may not yet exist on disk).
    initial : dict or None, optional
        Initial key/value pairs to stage on construction. Default is None.
    """

    def __init__(self, store_path: Union[str, Path], *, initial: Optional[dict] = None):
        self._path = Path(store_path)
        self._cache: Dict[str, Any] = {}
        self._staged: Dict[str, Any] = {}
        self._dirty: set = set()
        self._deleted: set = set()
        self._open()
        if initial:
            for k, v in dict(initial).items():
                self[k] = v

    def _open(self) -> None:
        """(Re)open the backing store read-only and load its manifest."""
        if self._path.exists():
            self._root = zarr.open_group(str(self._path), mode="r")
            self._manifest = dict(self._root.attrs.get(_MANIFEST, {}))
        else:
            self._root = None
            self._manifest = {}

    def _live_keys(self) -> set:
        """Return the currently visible keys (persisted + staged, minus deleted)."""
        return (set(self._manifest) | set(self._staged)) - self._deleted

    # -- MutableMapping abstract methods ----------------------------------- #
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
        return iter(self._live_keys())

    def __len__(self) -> int:
        return len(self._live_keys())

    def __contains__(self, key: str) -> bool:
        return key in self._live_keys()

    # -- persistence ------------------------------------------------------- #
    def flush(self) -> None:
        """Persist staged writes/deletes to the store (incremental)."""
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

    def view(self) -> "Results":
        """Flush and return a read-only grouped/lazy :class:`Results` view.

        Returns
        -------
        Results
            A read-only view of the current store contents.
        """
        self.ensure_store()
        return Results(self._path)

    def __repr__(self):
        return (f"<ResultsDict {len(self)} keys ({len(self._dirty)} dirty) "
                f"@ {self._path.name}>")
