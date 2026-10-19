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

"""One canonical form for every ``frames`` / ``frame`` argument.

The analysis steps take ``frames`` as ``'all'``, one frame number or any sequence of frame numbers; plots,
``reference_frame`` and ``detect_lois`` take a single frame. These helpers turn every spelling into one value
and reject everything else with an error that says what is valid, so that ``k``, ``np.int64(k)``, ``[k]``,
``(k,)``, ``range(k, k + 1)``, ``np.array([k])`` and ``np.array(k)`` select the same frame everywhere.

Reading primitives (``read_imgs``, ``_read_mask``) keep the caller's order and do not use them.
"""
from typing import List, Union

import numpy as np

__all__ = ['canonical_frames', 'canonical_frame', 'parse_frames_text']


def _as_int(value, name: str) -> int:
    """A frame number as a Python int; bools, floats and other types are refused."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name}: frame numbers must be integers, got the bool {value!r}.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"{name}: frame numbers must be integers, got {type(value).__name__} {value!r}.")


def canonical_frames(frames, n_stack, *, name: str = 'frames') -> Union[str, List[int]]:
    """``'all'`` or the sorted list of unique frame numbers that ``frames`` selects.

    Parameters
    ----------
    frames : None, 'all', int, numpy integer, 0-d array, or a sequence of them (list, tuple, range, array)
        None and ``'all'`` select every frame.
    n_stack : int or None
        Number of frames of the image (None or 1 for a single image).
    name : str, optional
        Argument name used in error messages. Default is 'frames'.

    Returns
    -------
    str or list of int
        ``'all'``, or the selected frames as a sorted list of unique Python ints.

    Raises
    ------
    TypeError
        For non-integer frame numbers (floats, bools, strings other than 'all').
    ValueError
        For an empty selection or a frame outside ``0 .. n_stack - 1``.
    """
    n = int(n_stack) if n_stack else 1
    if frames is None:
        return 'all'
    if isinstance(frames, str):
        if frames.strip().lower() == 'all':
            return 'all'
        raise TypeError(f"{name}: expected 'all', a frame number or a sequence of frame numbers, "
                        f"got the string {frames!r}.")
    if isinstance(frames, np.ndarray):
        values = frames.ravel().tolist()
    elif isinstance(frames, (list, tuple, range)):
        values = list(frames)
    else:
        values = [frames]
    ints = [_as_int(v, name) for v in values]
    if not ints:
        raise ValueError(f"{name}: the selection is empty; pass 'all', a frame number or frame numbers.")
    bad = sorted({f for f in ints if f < 0 or f >= n})
    if bad:
        what = 'a single image (frame 0 only)' if n == 1 else f'a movie of {n} frames, numbered 0..{n - 1}'
        raise ValueError(f"{name}: frame(s) {bad} out of range for {what}.")
    return sorted(set(ints))


def canonical_frame(frame, n_stack, *, name: str = 'frame') -> int:
    """One frame number as a Python int, from an int, numpy integer, 0-d array or one-element sequence.

    Raises
    ------
    TypeError
        For None, strings and non-integer values.
    ValueError
        For more than one frame or a frame outside ``0 .. n_stack - 1``.
    """
    if frame is None or isinstance(frame, str):
        raise TypeError(f"{name}: expected one frame number, got {frame!r}.")
    frames = canonical_frames(frame, n_stack, name=name)
    if len(frames) != 1:
        raise ValueError(f"{name}: expected one frame number, got {len(frames)} frames {frames}.")
    return frames[0]


def parse_frames_text(text) -> Union[None, str, List[int]]:
    """Parse a typed frame selection: ``'all'``, one frame number, or frame numbers separated by commas,
    semicolons or spaces (frames start at 0). A trailing separator is ignored.

    Returns
    -------
    None, 'all' or list of int
        None for an empty field, ``'all'``, or the sorted unique frame numbers.

    Raises
    ------
    ValueError
        For anything else (negative numbers, ranges, decimals, letters), naming the offending entry.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.lower() == 'all':
        return 'all'
    tokens = [t for t in s.replace(';', ',').replace(' ', ',').split(',') if t.strip()]
    if not tokens:
        raise ValueError(f"{text!r} holds no frame number.")
    frames = []
    for t in tokens:
        t = t.strip()
        if not (t.isascii() and t.isdigit()):
            raise ValueError(f"{t!r} is not a frame number. Use 'all', one frame number (e.g. 3) or "
                             "frame numbers separated by commas (e.g. 0, 4, 9); frames start at 0.")
        frames.append(int(t))
    return sorted(set(frames))
