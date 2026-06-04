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

"""Single ``<name>.ome.zarr`` container for a complete SarcAsM analysis.

One store holds *everything*: the raw image (OME-Zarr multiscales), derived
masks (integer → OME-Zarr ``labels/``; float prob maps → ``sarcasm/masks/``),
the optical-flow field, the analysis/track results (the
:mod:`sarcasm.io.results_store` groups, re-parented under ``sarcasm/``) and the
metadata. Bioimage tools (napari, Fiji, vizarr) read the image + labels; the
``sarcasm/`` namespace is an extra group they ignore.

Layout::

    movie.ome.zarr/
      zarr.json            attrs.ome.multiscales  (axes + scale from pixelsize/frametime)
      0/  (1/ 2/ …)        raw image pixels (optional pyramid)
      labels/<name>/0      integer masks (cell_mask, sarcomere_mask) + image-label metadata
      sarcasm/
        masks/<name>       float prob maps (zbands, mbands, orientation, distance, …)
        flow/0             optical flow (T-1, H, W, 2)
        tracks/ motion/ structure/ params/   ← results_store groups
        zarr.json          attrs: metadata, _manifest

This module is storage-only: it takes/returns numpy arrays and plain dicts and
has no dependency on the analysis classes, so ``SarcAsMBase`` can build on it
without a circular import. It does **not** read or write the legacy
``base_dir/`` + ``structure.json`` layout — see :func:`detect_legacy_layout`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import zarr

from sarcasm.io.results_store import ResultsDict, Results

logger = logging.getLogger(__name__)

OME_VERSION = "0.5"
IMAGE = "0"                 # full-resolution image array (multiscales level 0)
SARCASM = "sarcasm"        # SarcAsM namespace group
MASKS = f"{SARCASM}/masks"
FLOW = f"{SARCASM}/flow"
LABELS = "labels"
_META = "metadata"

# OME axis descriptors by SarcAsM axis letter
_AXIS_TYPE = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}


# --------------------------------------------------------------------------- #
# path helpers / legacy detection
# --------------------------------------------------------------------------- #
def store_path_for(input_path: Union[str, os.PathLike]) -> Path:
    """The ``<name>.ome.zarr`` store path for an input image.

    ``.ome.zarr`` inputs are their own store (analyze in place); ``.tif`` /
    ``.ome.tif`` / ``.tiff`` map to a sibling ``<name>.ome.zarr``.
    """
    p = Path(input_path)
    name = p.name
    if name.endswith(".ome.zarr") or name.endswith(".zarr"):
        return p
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if name.lower().endswith(suffix):
            return p.with_name(name[: -len(suffix)] + ".ome.zarr")
    return p.with_name(p.stem + ".ome.zarr")


def detect_legacy_layout(input_path: Union[str, os.PathLike]) -> Optional[Path]:
    """Return the pre-1.0 ``base_dir`` if an old-style analysis is present, else None.

    The old layout was ``<name>/`` (sibling dir) holding ``data/structure.json``
    (or ``data.zarr``) and mask ``.tif`` files.
    """
    p = Path(input_path)
    base = p.with_suffix("") if p.suffix else p
    if base.is_dir() and (
        (base / "data" / "structure.json").exists()
        or (base / "data" / "data.zarr").exists()
        or (base / "zbands.tif").exists()
    ):
        return base
    return None


def legacy_layout_message(base_dir: Union[str, os.PathLike]) -> str:
    """The user-facing message for old-style data (clean break in >=1.0)."""
    return (
        f"Pre-1.0 SarcAsM analysis detected at '{base_dir}'. SarcAsM >=1.0 reads only "
        f"a single '<name>.ome.zarr' store. Install the last pre-1.0 release to open it "
        f"(`pip install \"sarcasm==0.5.*\"`), or re-run the analysis to create a fresh "
        f".ome.zarr store."
    )


# --------------------------------------------------------------------------- #
# small zarr helpers
# --------------------------------------------------------------------------- #
def _ensure_group(root: "zarr.Group", path: str) -> "zarr.Group":
    g = root
    for seg in path.split("/"):
        try:
            g = g[seg]
        except KeyError:
            g = g.create_group(seg)
    return g


def _image_chunks(shape: Sequence[int]) -> tuple:
    """Chunk a frame at a time along the leading (T/Z) axis, full Y/X plane."""
    c = list(shape)
    if len(c) >= 3:
        for i in range(len(c) - 2):
            c[i] = 1
    return tuple(c)


def _ome_axes(axes: str) -> List[dict]:
    return [{"name": a, "type": _AXIS_TYPE.get(a.lower(), "space")} for a in axes.lower()]


def _ome_scale(axes: str, pixelsize: Optional[float], frametime: Optional[float]) -> List[float]:
    scale = []
    for a in axes.lower():
        if a == "x" or a == "y":
            scale.append(float(pixelsize) if pixelsize else 1.0)
        elif a == "t":
            scale.append(float(frametime) if frametime else 1.0)
        else:
            scale.append(1.0)
    return scale


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
class OmeZarrStore:
    """Read/write façade over a single ``<name>.ome.zarr`` analysis store."""

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = Path(path)

    # -- existence / creation --------------------------------------------- #
    @property
    def exists(self) -> bool:
        return self.path.exists()

    def _root(self, mode: str = "r") -> "zarr.Group":
        return zarr.open_group(str(self.path), mode=mode)

    @classmethod
    def create(cls, path: Union[str, os.PathLike], image: np.ndarray, axes: str, *,
               pixelsize: Optional[float] = None, frametime: Optional[float] = None,
               metadata: Optional[dict] = None, overwrite: bool = False) -> "OmeZarrStore":
        """Create the store and ingest ``image`` as the OME-Zarr level-0 image."""
        store = cls(path)
        if store.exists and not overwrite:
            raise FileExistsError(f"store already exists: {store.path}")
        root = zarr.open_group(str(store.path), mode="w")
        if len(axes) != image.ndim:
            raise ValueError(f"axes {axes!r} does not match image ndim {image.ndim}")
        arr = root.create_array(IMAGE, shape=image.shape, dtype=image.dtype,
                                chunks=_image_chunks(image.shape))
        arr[...] = image
        root.attrs["ome"] = {
            "version": OME_VERSION,
            "multiscales": [{
                "name": Path(path).name,
                "axes": _ome_axes(axes),
                "datasets": [{
                    "path": IMAGE,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": _ome_scale(axes, pixelsize, frametime)}],
                }],
            }],
        }
        store.write_metadata({"axes": axes, "pixelsize": pixelsize,
                              "frametime": frametime, **(metadata or {})})
        return store

    # -- raw image -------------------------------------------------------- #
    def read_image(self, frames=None) -> np.ndarray:
        arr = self._root("r")[IMAGE]
        return arr[...] if frames is None else arr[frames]

    def image_handle(self) -> "zarr.Array":
        """Lazy zarr handle for the raw image (slice without loading)."""
        return self._root("r")[IMAGE]

    @property
    def axes(self) -> Optional[str]:
        return (self.read_metadata() or {}).get("axes")

    # -- masks ------------------------------------------------------------ #
    def write_mask(self, name: str, arr: np.ndarray, *, as_label: bool = False) -> None:
        """Store a derived mask. ``as_label`` puts integer masks under OME
        ``labels/`` (napari label layers); otherwise float prob maps go under
        ``sarcasm/masks/``."""
        root = self._root("a")
        if as_label:
            grp = _ensure_group(root, f"{LABELS}/{name}")
            a = grp.create_array(IMAGE, shape=arr.shape, dtype=arr.dtype,
                                 chunks=_image_chunks(arr.shape), overwrite=True)
            a[...] = arr
            grp.attrs["image-label"] = {"version": OME_VERSION}
            labels_grp = _ensure_group(root, LABELS)
            listed = list(labels_grp.attrs.get("labels", []))
            if name not in listed:
                listed.append(name)
            labels_grp.attrs["labels"] = listed
        else:
            grp = _ensure_group(root, MASKS)
            a = grp.create_array(name, shape=arr.shape, dtype=arr.dtype,
                                 chunks=_image_chunks(arr.shape), overwrite=True)
            a[...] = arr

    def read_mask(self, name: str) -> np.ndarray:
        root = self._root("r")
        if name in list(_ensure_group_ro(root, LABELS)):
            return root[f"{LABELS}/{name}/{IMAGE}"][...]
        return root[f"{MASKS}/{name}"][...]

    def has_mask(self, name: str) -> bool:
        root = self._root("r")
        try:
            return (name in list(root[LABELS].group_keys())) or (name in list(root[MASKS].array_keys()))
        except KeyError:
            return False

    def mask_names(self) -> List[str]:
        root = self._root("r")
        out = []
        for grp, getter in ((LABELS, "group_keys"), (MASKS, "array_keys")):
            try:
                out += list(getattr(root[grp], getter)())
            except KeyError:
                pass
        return out

    # -- flow ------------------------------------------------------------- #
    def write_flow(self, flow: np.ndarray) -> None:
        grp = _ensure_group(self._root("a"), SARCASM)
        a = grp.create_array("flow", shape=flow.shape, dtype=flow.dtype,
                             chunks=_image_chunks(flow.shape), overwrite=True)
        a[...] = flow

    def read_flow(self) -> Optional[np.ndarray]:
        try:
            return self._root("r")[FLOW][...]
        except KeyError:
            return None

    # -- analysis results (nested results_store) -------------------------- #
    @property
    def results_path(self) -> Path:
        return self.path / SARCASM

    def results_dict(self) -> ResultsDict:
        """The lazy, dict-compatible analysis store (``Structure.data`` backing)."""
        return ResultsDict(self.results_path)

    def results_view(self) -> Results:
        """The grouped, lazy, read-only results view (``Structure.results``)."""
        ResultsDict(self.results_path).ensure_store()
        return Results(self.results_path)

    # -- metadata --------------------------------------------------------- #
    def write_metadata(self, meta: dict) -> None:
        grp = _ensure_group(self._root("a"), SARCASM)
        grp.attrs[_META] = _jsonable(meta)

    def read_metadata(self) -> Optional[dict]:
        try:
            return dict(self._root("r")[SARCASM].attrs.get(_META, {})) or None
        except KeyError:
            return None

    def __repr__(self):
        return f"<OmeZarrStore {self.path.name} exists={self.exists}>"


def _ensure_group_ro(root: "zarr.Group", path: str):
    """Best-effort read-only group access; returns an empty list-able on miss."""
    try:
        return list(root[path].group_keys())
    except KeyError:
        return []


def _jsonable(v: Any) -> Any:
    """Make metadata JSON/attr-safe (numpy scalars/arrays -> python)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
