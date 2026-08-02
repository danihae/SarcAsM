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
the analysis/track results (the
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
        structure/ motion/ params/           ← results_store groups
        zarr.json          attrs: metadata, _manifest

This module is storage-only: it takes/returns numpy arrays and plain dicts and
has no dependency on the analysis classes, so ``SarcAsMBase`` can build on it
without a circular import. It does **not** read or write the legacy
``base_dir/`` + ``structure.json`` layout — see :func:`detect_legacy_layout`.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import zarr

from sarcasm.io.results_store import Results

logger = logging.getLogger(__name__)

OME_VERSION = "0.5"
IMAGE = "0"                 # full-resolution image array (multiscales level 0)
SARCASM = "sarcasm"        # SarcAsM namespace group
MASKS = f"{SARCASM}/masks"
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

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to the input image (``.tif``/``.ome.tif``/``.tiff`` or
        ``.ome.zarr``/``.zarr``).

    Returns
    -------
    Path
        The store path.
    """
    p = Path(input_path)
    name = p.name
    if name.endswith(".ome.zarr") or name.endswith(".zarr"):
        return p
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if name.lower().endswith(suffix):
            return p.with_name(name[: -len(suffix)] + ".ome.zarr")
    return p.with_name(p.stem + ".ome.zarr")


def remove_tree(path: Union[str, os.PathLike], attempts: int = 3) -> None:
    """Delete a directory tree, even while another program writes into it.

    ``shutil.rmtree`` walks a directory and then ``rmdir``s it, so an entry created
    in between makes the final step fail with ``OSError: [Errno 66] Directory not
    empty`` even though nothing is in use. The trigger in practice is macOS Finder
    writing ``.DS_Store`` into a folder the user has open, which makes
    ``SarcAsM(..., restart=True)`` fail on a store the user is merely *looking* at.

    Retrying alone does not fix this, because a watcher that keeps writing wins
    every race. So the tree is first **renamed out of the way**: ``os.replace`` is
    atomic, so the moment it
    returns, whatever is watching the original name can no longer reach the tree we
    are deleting, and the delete proceeds unopposed. Retries remain as a fallback for
    the case where the rename itself is not possible.

    Deliberately never uses ``ignore_errors=True``: that would leave a partially
    deleted store behind and let it be silently reused as if it were fresh.

    Parameters
    ----------
    path : str or os.PathLike
        Directory to remove. A missing directory is not an error.
    attempts : int, optional
        How many delete attempts before giving up. Default is 3.

    Raises
    ------
    OSError
        If the tree could not be deleted, with a message naming the likely cause.

    Notes
    -----
    The original path may be *recreated* (empty) by the watcher right after the
    rename — harmless, since a fresh store is written into it anyway.
    """
    path = Path(path)
    if not path.exists():
        return

    # Move the tree aside first (atomic), so the watcher's writes no longer land in it.
    target = path
    staged = path.with_name(path.name + ".deleting")
    suffix = 0
    while staged.exists():
        suffix += 1
        staged = path.with_name(f"{path.name}.deleting{suffix}")
    try:
        os.replace(path, staged)
        target = staged
    except OSError:
        pass  # can't stage (permissions, exotic FS) — delete in place with retries

    last_exc: Optional[OSError] = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(target)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                # Let whatever recreated the entry finish before the next walk.
                time.sleep(0.05)
    if not target.exists():
        return
    staged_note = (f" A partial copy was moved to '{target}' and can be deleted manually."
                   if target != path else "")
    raise OSError(
        f"Could not delete '{path}' after {attempts} attempts ({last_exc}). "
        f"Something is recreating files inside it or holding it open — close any "
        f"program using the folder (Finder, a file browser, napari, Fiji) and retry."
        f"{staged_note}"
    ) from last_exc


def detect_legacy_layout(input_path: Union[str, os.PathLike]) -> Optional[Path]:
    """Return the pre-1.0 ``base_dir`` if an old-style analysis is present.

    The old layout was ``<name>/`` (sibling dir) holding ``data/structure.json``
    (or ``data.zarr``) and mask ``.tif`` files.

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to the input image.

    Returns
    -------
    Path or None
        The legacy ``base_dir`` if detected, else None.
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
    """Build the warning shown when old-style data is found but not read.

    Parameters
    ----------
    base_dir : str or os.PathLike
        The detected legacy ``base_dir``.

    Returns
    -------
    str
        A user-facing warning that analysis starts fresh in the new store.
    """
    return (
        f"Pre-1.0 SarcAsM analysis detected at '{base_dir}'. SarcAsM >=1.0 does not read it; "
        f"starting fresh in the new '<name>.ome.zarr' store (the old analysis is left untouched). "
        f"To open the old results, install the last pre-1.0 release (`pip install \"sarcasm==0.5.*\"`)."
    )


# --------------------------------------------------------------------------- #
# small zarr helpers
# --------------------------------------------------------------------------- #
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


def _image_chunks(shape: Sequence[int]) -> tuple:
    """Chunk a frame at a time along the leading (T/Z) axis, full Y/X plane.

    Parameters
    ----------
    shape : sequence of int
        Image shape.

    Returns
    -------
    tuple
        Chunk shape: 1 along all leading axes, full extent on the last two.
    """
    c = list(shape)
    if len(c) >= 3:
        for i in range(len(c) - 2):
            c[i] = 1
    return tuple(c)


def _ome_axes(axes: str) -> List[dict]:
    """Build OME-Zarr axis descriptors from a SarcAsM axis string.

    Parameters
    ----------
    axes : str
        Axis letters (e.g. ``'tyx'``).

    Returns
    -------
    list of dict
        One ``{'name', 'type'}`` descriptor per axis.
    """
    return [{"name": a, "type": _AXIS_TYPE.get(a.lower(), "space")} for a in axes.lower()]


def _ome_scale(axes: str, pixelsize: Optional[float], frametime: Optional[float]) -> List[float]:
    """Build the OME-Zarr scale coordinate transform for the given axes.

    Parameters
    ----------
    axes : str
        Axis letters (e.g. ``'tyx'``).
    pixelsize : float or None
        Spatial pixel size for x/y axes; 1.0 if None.
    frametime : float or None
        Time step for the t axis; 1.0 if None.

    Returns
    -------
    list of float
        Per-axis scale factors.
    """
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
    """Read/write façade over a single ``<name>.ome.zarr`` analysis store.

    Holds the raw image, derived masks, analysis/track results
    and metadata in one OME-Zarr container.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the ``<name>.ome.zarr`` store (need not yet exist).

    Attributes
    ----------
    path : Path
        The store path.
    """

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = Path(path)

    # -- existence / creation --------------------------------------------- #
    @property
    def exists(self) -> bool:
        """bool: Whether the store exists on disk."""
        return self.path.exists()

    def size_bytes(self, *, max_files: int = 200_000) -> Optional[int]:
        """Total size of the store on disk, for display purposes.

        A Zarr store is a directory of chunk files, so this walks the tree.

        Parameters
        ----------
        max_files : int, optional
            Give up (returning None) once this many files have been visited, so
            printing an object can never become expensive. Default is 200000.

        Returns
        -------
        int or None
            Size in bytes, or None if the store is absent or too large to walk.
        """
        if not self.exists:
            return None
        total = 0
        seen = 0
        for dirpath, _dirnames, filenames in os.walk(self.path):
            for name in filenames:
                seen += 1
                if seen > max_files:
                    return None
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total

    def _root(self, mode: str = "r") -> "zarr.Group":
        """Open the store root group.

        Parameters
        ----------
        mode : str, optional
            Zarr open mode (``'r'``, ``'a'``, ``'w'``). Default is ``'r'``.

        Returns
        -------
        zarr.Group
            The root group.
        """
        return zarr.open_group(str(self.path), mode=mode)

    @classmethod
    def create(cls, path: Union[str, os.PathLike], image: np.ndarray, axes: str, *,
               pixelsize: Optional[float] = None, frametime: Optional[float] = None,
               metadata: Optional[dict] = None, overwrite: bool = False) -> "OmeZarrStore":
        """Create the store and ingest an image as the OME-Zarr level-0 image.

        Parameters
        ----------
        path : str or os.PathLike
            Store path to create.
        image : np.ndarray
            Raw image pixels.
        axes : str
            Axis letters matching ``image.ndim`` (e.g. ``'tyx'``).
        pixelsize : float or None, optional
            Spatial pixel size. Default is None.
        frametime : float or None, optional
            Time step between frames. Default is None.
        metadata : dict or None, optional
            Extra metadata to store. Default is None.
        overwrite : bool, optional
            If True, replace an existing store; otherwise raise. Default is
            False.

        Returns
        -------
        OmeZarrStore
            The created store.

        Raises
        ------
        FileExistsError
            If the store exists and ``overwrite`` is False.
        """
        store = cls(path)
        if store.exists and not overwrite:
            raise FileExistsError(f"store already exists: {store.path}")
        if store.exists and overwrite:
            remove_tree(store.path)
        store.ingest_image(image, axes, pixelsize=pixelsize, frametime=frametime,
                           metadata=metadata)
        return store

    def ingest_image(self, image: np.ndarray, axes: str, *,
                     pixelsize: Optional[float] = None, frametime: Optional[float] = None,
                     metadata: Optional[dict] = None) -> None:
        """Write an image as the level-0 OME-Zarr image (additive write).

        Existing ``labels/`` / ``sarcasm/`` siblings are left intact.

        Parameters
        ----------
        image : np.ndarray
            Raw image pixels.
        axes : str
            Axis letters matching ``image.ndim``.
        pixelsize : float or None, optional
            Spatial pixel size. Default is None.
        frametime : float or None, optional
            Time step between frames. Default is None.
        metadata : dict or None, optional
            Extra metadata to store. Default is None.

        Raises
        ------
        ValueError
            If ``len(axes)`` does not match ``image.ndim``.
        """
        if len(axes) != image.ndim:
            raise ValueError(f"axes {axes!r} does not match image ndim {image.ndim}")
        root = self._root("a")
        if IMAGE in set(root.array_keys()):
            del root[IMAGE]
        arr = root.create_array(IMAGE, shape=image.shape, dtype=image.dtype,
                                chunks=_image_chunks(image.shape))
        arr[...] = image
        root.attrs["ome"] = {
            "version": OME_VERSION,
            "multiscales": [{
                "name": self.path.name,
                "axes": _ome_axes(axes),
                "datasets": [{
                    "path": IMAGE,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": _ome_scale(axes, pixelsize, frametime)}],
                }],
            }],
        }
        self.write_metadata({"axes": axes, "pixelsize": pixelsize,
                             "frametime": frametime, **(metadata or {})})

    # -- raw image -------------------------------------------------------- #
    def has_image(self) -> bool:
        """Return True if a level-0 raw image is stored."""
        if not self.exists:
            return False
        try:
            return IMAGE in set(self._root("r").array_keys())
        except (KeyError, FileNotFoundError):
            return False

    def read_image(self, frames=None) -> np.ndarray:
        """Read the raw image (optionally a subset of frames).

        Parameters
        ----------
        frames : int, slice, or array-like, optional
            Index/slice along the leading axis. Default is None (whole image).

        Returns
        -------
        np.ndarray
            The (sub)image pixels.
        """
        arr = self._root("r")[IMAGE]
        return arr[...] if frames is None else arr[frames]

    def image_handle(self) -> "zarr.Array":
        """Return a lazy zarr handle for the raw image (slice without loading).

        Returns
        -------
        zarr.Array
            Lazy handle to the level-0 image array.
        """
        return self._root("r")[IMAGE]

    @property
    def axes(self) -> Optional[str]:
        """str or None: The stored axis-letter string, if any."""
        return (self.read_metadata() or {}).get("axes")

    # -- masks ------------------------------------------------------------ #
    def write_mask(self, name: str, arr: np.ndarray, *, as_label: bool = False) -> None:
        """Store a derived mask.

        Parameters
        ----------
        name : str
            Mask name (e.g. ``'cell_mask'``, ``'zbands'``).
        arr : np.ndarray
            Mask pixels.
        as_label : bool, optional
            If True, store an integer mask under OME ``labels/`` (napari label
            layers); otherwise store a float prob map under ``sarcasm/masks/``.
            Default is False.
        """
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

    def create_mask(self, name: str, shape, dtype) -> Any:
        """Allocate an empty mask array and return its handle for region writes.

        Lets a caller fill a mask block by block instead of holding the whole
        stack in memory first -- for a multi-head prediction over a large movie
        the assembled result is several times the size of the input.

        Parameters
        ----------
        name : str
            Mask name (e.g. ``'zbands'``).
        shape : sequence of int
            Full shape of the mask.
        dtype : dtype
            Element type.

        Returns
        -------
        zarr.Array
            Writable handle; assign into it with ordinary slicing.
        """
        grp = _ensure_group(self._root("a"), MASKS)
        return grp.create_array(name, shape=tuple(shape), dtype=dtype,
                                chunks=_image_chunks(tuple(shape)), overwrite=True)

    def read_mask(self, name: str, frames=None) -> np.ndarray:
        """Read a derived mask by name (from ``labels/`` or ``sarcasm/masks/``).

        Parameters
        ----------
        name : str
            Mask name.
        frames : int, list, slice, or None, optional
            Frame selection along axis 0. Indexing the zarr array directly loads
            only the requested chunks (masks are chunked one frame per chunk)
            rather than materialising the whole stack. None reads all frames.

        Returns
        -------
        np.ndarray
            The mask pixels.
        """
        root = self._root("r")
        if name in list(_ensure_group_ro(root, LABELS)):
            arr = root[f"{LABELS}/{name}/{IMAGE}"]
        else:
            arr = root[f"{MASKS}/{name}"]
        return arr[...] if frames is None else arr[frames]

    def has_mask(self, name: str) -> bool:
        """Return True if a mask ``name`` exists in either mask group.

        Parameters
        ----------
        name : str
            Mask name.

        Returns
        -------
        bool
            Whether the mask is present.
        """
        if not self.exists:
            return False
        root = self._root("r")
        # Check each group independently: a missing labels/ or masks/ group must
        # not short-circuit the other (e.g. when only float prob maps are stored).
        for grp, getter in ((LABELS, "group_keys"), (MASKS, "array_keys")):
            try:
                if name in list(getattr(root[grp], getter)()):
                    return True
            except KeyError:
                continue
        return False

    def mask_names(self) -> List[str]:
        """List all stored mask names across ``labels/`` and ``sarcasm/masks/``.

        Returns
        -------
        list of str
            Mask names.
        """
        if not self.exists:
            return []
        root = self._root("r")
        out = []
        for grp, getter in ((LABELS, "group_keys"), (MASKS, "array_keys")):
            try:
                out += list(getattr(root[grp], getter)())
            except KeyError:
                pass
        return out

    def delete_mask(self, name: str) -> bool:
        """Delete a stored mask (from ``labels/`` or ``sarcasm/masks/``) if present.

        Parameters
        ----------
        name : str
            Mask name.

        Returns
        -------
        bool
            True if a mask was removed, False if it was not present.
        """
        if not self.exists:
            return False
        root = self._root("a")
        try:
            labels_grp = root[LABELS]
        except KeyError:
            labels_grp = None
        if labels_grp is not None and name in list(labels_grp.group_keys()):
            del labels_grp[name]
            labels_grp.attrs["labels"] = [
                n for n in labels_grp.attrs.get("labels", []) if n != name]
            return True
        try:
            masks_grp = root[MASKS]
        except KeyError:
            masks_grp = None
        if masks_grp is not None and name in list(masks_grp.array_keys()):
            del masks_grp[name]
            return True
        return False

    # -- analysis results (nested results_store) -------------------------- #
    @property
    def results_path(self) -> Path:
        """Path: The ``sarcasm/`` results subgroup path inside the store."""
        return self.path / SARCASM

    def results(self) -> Results:
        """Open the lazy analysis results accessor (``SarcAsM.data`` backing).

        Opening never creates the store on disk.

        Returns
        -------
        Results
            The results mapping / namespace accessor.
        """
        return Results(self.results_path)

    # -- metadata --------------------------------------------------------- #
    def write_metadata(self, meta: dict) -> None:
        """Write image/analysis metadata to the ``sarcasm/`` group attrs.

        Parameters
        ----------
        meta : dict
            Metadata mapping (made JSON/attr-safe before writing).
        """
        grp = _ensure_group(self._root("a"), SARCASM)
        grp.attrs[_META] = _jsonable(meta)

    def read_metadata(self) -> Optional[dict]:
        """Read the stored metadata, or None if absent.

        Returns
        -------
        dict or None
            The metadata mapping, or None when no metadata is stored.
        """
        try:
            return dict(self._root("r")[SARCASM].attrs.get(_META, {})) or None
        except (KeyError, FileNotFoundError):
            return None

    def __repr__(self):
        return f"<OmeZarrStore {self.path.name} exists={self.exists}>"


def _ensure_group_ro(root: "zarr.Group", path: str):
    """List a group's subgroup keys read-only, returning ``[]`` on a miss.

    Parameters
    ----------
    root : zarr.Group
        Root group.
    path : str
        Subgroup path.

    Returns
    -------
    list of str
        Subgroup names, or an empty list if the group is absent.
    """
    try:
        return list(root[path].group_keys())
    except KeyError:
        return []


def _jsonable(v: Any) -> Any:
    """Make metadata JSON/attr-safe by converting numpy types to python.

    Parameters
    ----------
    v : Any
        Value to convert (ndarray, np.generic, dict, list/tuple, or plain type).

    Returns
    -------
    Any
        JSON/attr-safe value.
    """
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
