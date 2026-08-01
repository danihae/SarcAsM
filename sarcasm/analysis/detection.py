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

"""Neural network-based detection module for sarcomeres and Z-bands."""

import logging
import os
from typing import Tuple, Union
import numpy as np
import torch
from bio_image_unet import multi_output_unet3d as unet3d
from bio_image_unet.multi_output_unet.multi_output_nested_unet import MultiOutputNestedUNet_3Levels
from bio_image_unet.multi_output_unet.predict import Predict as Predict_UNet
from bio_image_unet.progress import ProgressNotifier

from sarcasm.utils import Utils

logger = logging.getLogger(__name__)


def _resize_xy_back(arr: np.ndarray, target_xy: Tuple[int, int]) -> np.ndarray:
    """
    Resize an array's trailing (Y, X) dims to ``target_xy``.

    Uses nearest-neighbour interpolation and preserves range, restoring a
    rescaled mask to its original XY resolution.

    Parameters
    ----------
    arr : np.ndarray
        Input array whose trailing two dimensions are (Y, X).
    target_xy : tuple of int
        Target ``(Y, X)`` shape.

    Returns
    -------
    np.ndarray
        Array resized along its trailing (Y, X) dimensions.
    """
    from skimage.transform import resize
    if tuple(arr.shape[-2:]) == tuple(target_xy):
        return arr
    flat = arr.reshape(-1, *arr.shape[-2:])
    out = np.empty((flat.shape[0],) + tuple(target_xy), dtype=arr.dtype)
    for i in range(flat.shape[0]):
        out[i] = resize(flat[i], target_xy, order=0, preserve_range=True,
                        anti_aliasing=False).astype(arr.dtype)
    return out.reshape(arr.shape[:-2] + tuple(target_xy))


def _stack_clip_values(images, clip_thres: Tuple[float, float], block: int) -> Tuple[float, float]:
    """Intensity limits equivalent to ``normalization_mode='all'``, computed block-wise.

    ``np.percentile`` over a whole movie needs the movie in memory. For integer
    data an exact answer comes from a histogram accumulated block by block; for
    float data the percentiles are estimated from evenly spaced sample frames.

    Parameters
    ----------
    images : array-like
        Stack supporting ``.shape``, ``.dtype`` and slicing along axis 0.
    clip_thres : tuple of float
        Lower and upper percentiles.
    block : int
        Frames to read at a time.

    Returns
    -------
    tuple of float
        ``(low, high)`` absolute intensities.
    """
    n = images.shape[0]
    lo_pct, hi_pct = clip_thres

    if np.issubdtype(images.dtype, np.integer):
        lo_val = hi_val = None
        for start in range(0, n, block):
            chunk = np.asarray(images[start:start + block])
            c_lo, c_hi = int(chunk.min()), int(chunk.max())
            lo_val = c_lo if lo_val is None else min(lo_val, c_lo)
            hi_val = c_hi if hi_val is None else max(hi_val, c_hi)
        n_bins = int(hi_val - lo_val) + 1
        if n_bins <= (1 << 22):  # exact: one bin per representable value
            counts = np.zeros(n_bins, dtype=np.int64)
            for start in range(0, n, block):
                chunk = np.asarray(images[start:start + block]).ravel().astype(np.int64) - lo_val
                counts += np.bincount(chunk, minlength=n_bins)
            cum = np.cumsum(counts)
            total = cum[-1]
            edges = np.searchsorted(cum, [total * lo_pct / 100.0, total * hi_pct / 100.0])
            return float(lo_val + edges[0]), float(lo_val + min(edges[1], n_bins - 1))

    sample_idx = np.unique(np.linspace(0, n - 1, min(n, 32)).astype(int))
    sample = np.asarray(images[sample_idx] if n > 1 else images[:])
    return float(np.nanpercentile(sample, lo_pct)), float(np.percentile(sample, hi_pct))


def _blocks(n_frames: int, per_frame_bytes: int, budget_bytes: int, block_frames=None):
    """``(start, stop)`` frame ranges sized to stay inside a memory budget."""
    if block_frames is None:
        block_frames = int(np.clip(budget_bytes // max(per_frame_bytes, 1), 1, n_frames))
    block_frames = max(1, min(int(block_frames), n_frames))
    return [(start, min(start + block_frames, n_frames))
            for start in range(0, n_frames, block_frames)]


class _StackProgress(ProgressNotifier):
    """One progress report for a stack predicted in several blocks.

    Each block runs its own ``Predict``, and each asks its notifier for an
    iterator over that block's batches -- which on its own draws one progress bar
    per block. This hands out per-block iterators that all drive a single outer
    iterator instead, counted in frames.

    Parameters
    ----------
    outer : ProgressNotifier
        The notifier the caller passed in; supplies the one real progress display.
    n_frames : int
        Total frames across all blocks, the unit the outer report counts in.
    """

    def __init__(self, outer: ProgressNotifier, n_frames: int):
        super().__init__()
        self._outer = iter(outer.iterator(range(n_frames))) if outer is not None else None
        self._credit = 0.0
        self._block_frames = 0

    def set_block(self, n_frames: int):
        """Declare how many frames the next block covers."""
        self._block_frames = n_frames

    def _advance(self, frames: float):
        self._credit += frames
        while self._credit >= 1.0 and self._outer is not None:
            self._credit -= 1.0
            try:
                next(self._outer)
            except StopIteration:
                self._outer = None

    def iterator(self, iterable, total=None):
        steps = total if total is not None else len(iterable)
        per_step = (self._block_frames / steps) if steps else 0.0

        def tracked():
            for item in iterable:
                yield item
                self._advance(per_step)

        return tracked()

    def finish(self):
        """Run the outer report out to 100% and close it."""
        if self._outer is not None:
            for _ in self._outer:
                pass
            self._outer = None


def detect_sarcomeres_unet(images, model_path: str, model_dir: str,
                          pixelsize: float, max_patch_size: Union[Tuple[int, int], str] = 'auto',
                          normalization_mode: str = 'all', clip_thres: Tuple[float, float] = (0., 99.98),
                          rescale_factor: float = 1.0, device: Union[torch.device, str] = 'auto',
                          batch_size: Union[int, str] = 'auto', block_frames: int = None,
                          memory_budget_gb: float = 2.0, prune_level: int = None, make_sink=None,
                          info: dict = None,
                          progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
    """
    Predict sarcomeres (Z-bands, mbands, distance, orientation) with U-Net.

    Parameters
    ----------
    images : array-like
        Images to process. Anything with ``.shape``, ``.dtype`` and slicing along
        axis 0 works, including a lazy zarr handle -- passing one keeps the raw
        stack out of memory.
    model_path : str
        Path of trained network weights for U-Net. If None or 'generalist',
        the bundled generalist model is used.
    model_dir : str
        Directory containing model files.
    pixelsize : float
        Pixel size in micrometers.
    max_patch_size : tuple of int or 'auto', optional
        Maximal patch dimensions ``(n_x, n_y)`` for the network. 'auto' derives
        them from free device memory and the model. Default is 'auto'.
    normalization_mode : str, optional
        Intensity normalization mode for 3D stacks ('single': each image
        individually, 'all': histogram of full stack, 'first': histogram of
        first image). Default is 'all'.
    clip_thres : tuple of float, optional
        Clip threshold (lower, upper) for intensity normalization.
        Default is (0., 99.98).
    rescale_factor : float, optional
        Factor to rescale input images in XY before prediction (e.g. 0.5 halves
        XY resolution); outputs are rescaled back to original resolution.
        Default is 1.0 (no rescaling).
    device : torch.device or str, optional
        Device on which PyTorch kernels are executed. Default is 'auto'.
    batch_size : int or 'auto', optional
        Patches per forward pass. 'auto' sizes from free GPU memory on CUDA and
        uses 1 elsewhere. Default is 'auto'.
    block_frames : int, optional
        Frames predicted per pass. None derives it from ``memory_budget_gb``.
        Default is None.
    memory_budget_gb : float, optional
        Rough ceiling on the working set of one block. The five output heads make
        the assembled result about twelve times the size of a uint16 input, so a
        long movie is predicted in blocks rather than all at once.
        Default is 2.0.
    prune_level : int, optional
        Stop the nested U-Net at this nesting depth (1, 2 or 3) and read the
        matching deep-supervision head. Level 2 roughly halves the compute; the
        masks differ from the full model, so validate before relying on it. None
        uses the full model. Default is None.
    info : dict, optional
        Filled in with what the run actually resolved to (``'patch_size'``,
        ``'tiles'``, ``'blocks'``), so an ``'auto'`` patch size can be recorded and
        reproduced later. Default is None.
    make_sink : callable, optional
        ``make_sink(name, shape, dtype)`` returning a sliceable, writable array
        (e.g. a zarr array). When given, each block is written to the sink as it
        is predicted and the sinks are returned instead of in-memory arrays.
        Default is None.
    progress_notifier : ProgressNotifier, optional
        Progress notifier for inclusion in GUI.
        Default is ProgressNotifier.progress_notifier_tqdm().

    Returns
    -------
    dict of str to np.ndarray
        Predicted masks keyed by name (``zbands``, ``mbands``, ``orientation``,
        ``cell_mask``, ``sarcomere_mask``) as float probability maps at the
        original XY resolution. The caller writes these into the OME-Zarr store,
        unless ``make_sink`` already did.
    """
    max_patch_size = Utils.check_and_round_max_patch_size(max_patch_size)

    if len(images.shape) < 2:
        raise ValueError("Images must be at least 2D (Y,X) to have XY dimensions for rescaling.")
    if len(images.shape) > 3:
        raise ValueError(f"Unsupported image dimensionality: {len(images.shape)}D. Expected 2D or 3D.")
    original_xy_shape = tuple(images.shape[-2:])
    is_stack = len(images.shape) == 3
    n_frames = images.shape[0] if is_stack else 1

    logger.info('Predicting sarcomeres ...')
    if model_path is None or model_path == 'generalist':
        model_path = os.path.join(model_dir, 'model_sarcomeres_generalist.pt')
        if pixelsize is not None and pixelsize < 0.1:
            logger.warning(
                f"Pixel size ({round(pixelsize, 3)} µm) is smaller than the optimal range "
                f"(0.1-0.35 µm) for generalist model. Pixelsize might be too small. "
                f"Consider rescale_factor={_suggested_rescale(pixelsize)} for optimal results.")
        elif pixelsize is not None and pixelsize > 0.35:
            logger.warning(
                f"Pixel size ({round(pixelsize, 3)} µm) is larger than the optimal range "
                f"(0.1-0.35 µm) for generalist model. Pixelsize might be too large. "
                f"Consider rescale_factor={_suggested_rescale(pixelsize)} for optimal results.")
        logger.info(f"Using default model: {model_path}")

    # Size the blocks from the whole working set of one block, not just the
    # result: the float32 input, the extracted patches, the float16 patch
    # predictions and the float32 stitch canvas are all live at once, which comes
    # to roughly twelve float32 planes per frame.
    out_channels = 6
    per_frame_bytes = int(np.prod(original_xy_shape)) * 4 * (2 * out_channels)
    budget_bytes = int(memory_budget_gb * (1 << 30))

    clip_values = None
    if is_stack and n_frames > 1 and normalization_mode in ('all', 'first'):
        # Normalize against the whole stack once, then hand the absolute limits to
        # each block so block-wise prediction matches a whole-stack run exactly.
        probe = images if normalization_mode == 'all' else images[:1]
        block_for_stats = int(np.clip(budget_bytes // max(per_frame_bytes, 1), 1, n_frames))
        clip_values = _stack_clip_values(probe, clip_thres, block_for_stats)

    blocks = _blocks(n_frames, per_frame_bytes, budget_bytes, block_frames)
    # One progress report for the whole stack; without this each block's Predict
    # would draw its own bar.
    stack_progress = _StackProgress(progress_notifier, n_frames) if len(blocks) > 1 else None

    sinks = {}
    for start, stop in blocks:
        block = np.asarray(images[start:stop] if is_stack else images[:])
        block_n = stop - start
        if stack_progress is not None:
            stack_progress.set_block(block_n)

        if rescale_factor != 1.0:
            from skimage.transform import rescale
            scale_vector = ((1.0, rescale_factor, rescale_factor) if block.ndim == 3
                            else (rescale_factor, rescale_factor))
            block = rescale(block, scale_vector, order=0, mode='reflect',
                            preserve_range=True, channel_axis=None).astype(block.dtype)

        pred = Predict_UNet(block, model_params=model_path, result_path=None,
                            max_patch_size=max_patch_size, normalization_mode=normalization_mode,
                            network=MultiOutputNestedUNet_3Levels,
                            clip_threshold=clip_thres, clip_values=clip_values,
                            batch_size=batch_size, device=device, prune_level=prune_level,
                            progress_notifier=stack_progress or progress_notifier)
        block_result = pred.result
        if info is not None and 'patch_size' not in info:
            info['patch_size'] = tuple(int(v) for v in pred.patch_size)
            info['tiles'] = (int(pred.N_x), int(pred.N_y))
            info['blocks'] = len(blocks)
            logger.info(f"Patch {info['patch_size']}, {info['tiles'][0]}x{info['tiles'][1]} tiles "
                        f"per frame, {len(blocks)} block(s)")
        del pred

        if rescale_factor != 1.0:
            block_result = {k: _resize_xy_back(v, original_xy_shape) for k, v in block_result.items()}

        for name, arr in block_result.items():
            # Predict squeezes singleton axes; restore the block axis so a
            # one-frame block writes like any other.
            channels = arr.size // (block_n * int(np.prod(original_xy_shape)))
            arr = arr.reshape((block_n, channels) + original_xy_shape)
            full_shape = ((n_frames,) if is_stack else ()) + \
                         ((channels,) if channels > 1 else ()) + original_xy_shape
            if name not in sinks:
                sinks[name] = (make_sink(name, full_shape, np.float32) if make_sink is not None
                               else np.empty(full_shape, dtype=np.float32))
            target = arr if channels > 1 else arr[:, 0]
            if is_stack:
                sinks[name][start:stop] = target
            else:
                sinks[name][...] = target[0]
        del block_result

    if stack_progress is not None:
        stack_progress.finish()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sinks


def _suggested_rescale(pixelsize: float, target: float = 0.2) -> float:
    """Rescale factor bringing ``pixelsize`` to the middle of the model's trained range."""
    return round(pixelsize / target, 2)


def detect_z_bands_fast_movie_unet(images: np.ndarray, model_path: str, model_dir: str,
                                  max_patch_size: Union[Tuple[int, int, int], str] = 'auto',
                                  normalization_mode: str = 'all',
                                  clip_thres: Tuple[float, float] = (0., 99.8),
                                  device: Union[torch.device, str] = 'auto',
                                  batch_size: Union[int, str] = 'auto', info: dict = None,
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> dict:
    """
    Predict sarcomere z-bands with 3D U-Net for high-speed movies for improved temporal consistency.

    Parameters
    ----------
    images : np.ndarray
        Images to process.
    model_path : str
        Path of trained network weights for 3D U-Net. If None, the bundled
        z-band model is used.
    model_dir : str
        Directory containing model files.
    max_patch_size : tuple of int or 'auto', optional
        Maximal patch dimensions ``(n_frames, n_x, n_y)`` for the network;
        dimensions must be divisible by 16. 'auto' derives them from free device
        memory and the model. Default is 'auto'.
    normalization_mode : str, optional
        Intensity normalization mode for 3D stacks ('single': each image
        individually, 'all': histogram of full stack, 'first': histogram of
        first image). Default is 'all'.
    clip_thres : tuple of float, optional
        Clip threshold (lower, upper) for intensity normalization.
        Default is (0., 99.8).
    device : torch.device or str, optional
        Device on which PyTorch kernels are executed. Default is 'auto'.
    progress_notifier : ProgressNotifier, optional
        Progress notifier for inclusion in GUI.
        Default is ProgressNotifier.progress_notifier_tqdm().

    Returns
    -------
    dict of str to np.ndarray
        ``{'zbands_fast_movie': ndarray}`` — the caller writes it into the store.
    """
    logger.info('Predicting sarcomere z-bands ...')

    if model_path is None:
        model_path = os.path.join(model_dir, 'model_z_bands_unet3d.pt')
    max_patch_size = Utils.check_and_round_max_patch_size(max_patch_size)
    if not isinstance(max_patch_size, str) and len(max_patch_size) != 3:
        raise ValueError('patch size for prediction has to be be (frames, x, y)')
    pred = unet3d.Predict(images, model_params=model_path, result_path=None,
                          max_patch_size=max_patch_size, normalization_mode=normalization_mode,
                          device=device, clip_threshold=clip_thres, batch_size=batch_size,
                          progress_notifier=progress_notifier)
    if info is not None:
        info['patch_size'] = tuple(int(v) for v in pred.patch_size)
        info['tiles'] = (int(pred.N_z), int(pred.N_y), int(pred.N_x))
        logger.info(f"Patch {info['patch_size']}, {info['tiles']} tiles")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred.result
