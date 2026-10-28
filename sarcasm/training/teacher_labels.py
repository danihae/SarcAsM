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


"""Teacher labels for real high-speed movies, from the 2D U-Net.

Real movies have no Z-band labels. The 2D generalist model (the "teacher") labels every
frame, and two steps make its labels better than what it gives on single frames:

1. **Denoise the teacher's input.** Each frame is replaced by the mean of itself and its
   ``k`` neighbours on either side, after warping those onto it with optical flow. The
   teacher thus sees more signal than any single frame holds, without motion blur.
2. **Consolidate in time.** The teacher's Z-band probabilities are averaged over the same
   flow-aligned neighbours, which removes frame-to-frame flicker.

On synthetic clips with exact labels this beat the raw 2D labels at low photon counts
(Dice 0.41 vs 0.29 at 10-50 photons) and matched them otherwise.

:func:`clips_from_movie` cuts a movie into short clips with these labels, ready for
:mod:`sarcasm.training.unet3d_trainer`.
"""

import json
import os

import numpy as np
import tifffile
from bio_image_unet.progress import ProgressNotifier
from scipy import ndimage

from sarcasm.analysis.detection import detect_sarcomeres_unet
from sarcasm.analysis.optical_flow import dis_flow, frame_to_uint8, intensity_range

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
TARGET = 'zbands_fast_movie'   # folder / head name the 3D U-Net is trained on


def zbands_2d(movie: np.ndarray, pixelsize: float, model: str = 'auto') -> np.ndarray:
    """Per-frame Z-band probability from the 2D U-Net, (T, H, W) float32.

    ``model='auto'`` uses the bundled 2D ``'generalist'``, as ``detect_sarcomeres`` does; the shipped 3D v2
    model was trained on teacher labels from ``'legacy'`` below 0.08 µm/px.
    """
    out = detect_sarcomeres_unet(movie, model_path=model, model_dir=MODEL_DIR, pixelsize=pixelsize,
                                 progress_notifier=ProgressNotifier.silent_notifier())
    return np.asarray(out['zbands'], np.float32).reshape(movie.shape)


def _neighbour_flows(movie, k):
    """Optical flow from every frame t to its neighbours t+j, |j| <= k: {(t, j): (H, W, 2)}."""
    smooth = np.stack([ndimage.gaussian_filter(f.astype(np.float32), 1.0) for f in movie])
    lo, hi = intensity_range(smooth[len(smooth) // 2])
    frames_u8 = np.stack([frame_to_uint8(f, lo, hi) for f in smooth])
    T = len(movie)
    return {(t, j): dis_flow(frames_u8[t], frames_u8[t + j])
            for t in range(T) for j in range(-k, k + 1) if j and 0 <= t + j < T}


def _aligned_mean(stack, flows, k):
    """Mean of each frame and its neighbours, the neighbours warped onto it with the flow."""
    T, H, W = stack.shape
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.empty(stack.shape, np.float32)
    for t in range(T):
        total, n = stack[t].astype(np.float32).copy(), 1
        for j in range(-k, k + 1):
            if j and 0 <= t + j < T:
                f = flows[(t, j)]
                total += ndimage.map_coordinates(stack[t + j].astype(np.float32),
                                                 [rows + f[..., 0], cols + f[..., 1]],
                                                 order=1, mode='nearest')
                n += 1
        out[t] = total / n
    return out


def teacher_labels(movie: np.ndarray, pixelsize: float, k: int = 2, model: str = 'auto') -> np.ndarray:
    """Z-band teacher labels (T, H, W) in [0, 1] for a short movie clip.

    Parameters
    ----------
    movie : np.ndarray
        (T, H, W) clip of a high-speed movie.
    pixelsize : float
        µm per pixel; selects the 2D model.
    k : int
        Neighbours on either side used for denoising and consolidation; 0 gives the plain
        per-frame 2D labels.
    model : str
        2D model ('auto', 'generalist', 'legacy' or a path).
    """
    if k == 0:
        return zbands_2d(movie, pixelsize, model)
    flows = _neighbour_flows(movie, k)
    denoised = _aligned_mean(movie.astype(np.float32), flows, k)
    return _aligned_mean(zbands_2d(denoised, pixelsize, model), flows, k)


def _read_frames(path, start, n):
    with tifffile.TiffFile(path) as tf:
        if len(tf.pages) >= start + n and tf.pages[0].ndim == 2:
            return np.stack([tf.pages[i].asarray() for i in range(start, start + n)])
    return np.asarray(tifffile.memmap(path)[start:start + n])


def _to_uint8(frames):
    """8-bit copy of a clip: integer data <= 255 as is, anything else percentile-scaled.

    The 3D U-Net normalises every clip by its percentiles anyway, and the quantisation
    step is far below the shot noise of a high-speed movie.
    """
    if frames.min() >= 0 and frames.max() <= 255 and np.allclose(frames[:2], np.round(frames[:2])):
        return frames.astype(np.uint8)
    lo, hi = np.percentile(frames, [0.05, 99.95])
    return np.clip((frames.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def _densest_crops(labels, size, n):
    """Up to n non-overlapping size x size crops with the highest mean label."""
    _, H, W = labels.shape
    h, w = min(size, H), min(size, W)
    mean = labels.mean(0)
    step = max(16, size // 8)
    candidates = sorted(((mean[y:y + h, x:x + w].mean(), y, x)
                         for y in range(0, H - h + 1, step) for x in range(0, W - w + 1, step)),
                        reverse=True)
    chosen = []
    for _, y, x in candidates:
        if all(abs(y - y2) >= h or abs(x - x2) >= w for y2, x2 in chosen):
            chosen.append((y, x))
        if len(chosen) == n:
            break
    return [(y, x, h, w) for y, x in chosen]


def clips_from_movie(movie_path: str, out_dir: str, pixelsize: float, name: str,
                     n_windows: int = 3, crops_per_window: int = 2, n_frames: int = 64,
                     crop_size: int = 256, frame_range=None, k: int = 2, label_levels: int = 16,
                     seed: int = 0) -> list:
    """Cut teacher-labelled training clips out of one high-speed movie.

    For each of ``n_windows`` time windows (spread over ``frame_range``) the teacher runs on
    the full frames, then the ``crops_per_window`` crops with the most Z-bands are written:

    * ``out_dir/volume/<name>_w<i>_c<j>.tif`` -- uint8 movie clip
    * ``out_dir/zbands_fast_movie/<name>_w<i>_c<j>.tif`` -- uint8 label (``label_levels`` steps)
    * ``out_dir/clips.jsonl`` -- one line of provenance per clip

    Returns the list of written file names.
    """
    for sub in ('volume', TARGET):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    with tifffile.TiffFile(movie_path) as tf:
        n_total = tf.series[0].shape[0]
    first, last = frame_range if frame_range else (0, n_total)
    last = min(last, n_total)
    if last - first < n_frames:
        raise ValueError(f'{movie_path}: {last - first} frames, need at least {n_frames}')
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    edges = np.linspace(first, last - n_frames, n_windows + 1)
    written = []
    with open(os.path.join(out_dir, 'clips.jsonl'), 'a') as log:
        for w in range(n_windows):
            start = int(rng.uniform(edges[w], max(edges[w] + 1, edges[w + 1])))
            frames = _to_uint8(_read_frames(movie_path, start, n_frames))
            labels = teacher_labels(frames, pixelsize, k=k)
            q = np.round(np.clip(labels, 0, 1) * (label_levels - 1)) / (label_levels - 1)
            labels_u8 = np.round(q * 255).astype(np.uint8)
            for c, (y, x, h, wd) in enumerate(_densest_crops(labels, crop_size, crops_per_window)):
                fname = f'{name}_w{w}_c{c}.tif'
                region = np.s_[:, y:y + h, x:x + wd]
                tifffile.imwrite(os.path.join(out_dir, 'volume', fname), frames[region], compression='zlib')
                tifffile.imwrite(os.path.join(out_dir, TARGET, fname), labels_u8[region], compression='zlib')
                log.write(json.dumps(dict(file=fname, movie=str(movie_path), first_frame=start, y=y, x=x,
                                          pixelsize=pixelsize, teacher_k=k)) + '\n')
                written.append(fname)
    return written
