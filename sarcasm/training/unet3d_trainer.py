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


"""Training of the 3D Z-band U-Net used by ``MotionAnalysis.detect_z_bands_fast_movie``.

Training data are folders of clips, each with two sub-folders of same-named TIFFs:
``volume/`` (movie clips, any integer type) and ``zbands_fast_movie/`` (uint8 labels,
probability * 255). :mod:`sarcasm.training.synthetic_motion` and
:mod:`sarcasm.training.teacher_labels` write this layout.

Each folder is one *domain* with a sampling weight. Patches are drawn on the fly, so a
small domain (say, one collaborator's movie) is seen as often as its weight says and
is not drowned by thousands of synthetic clips.

What differs from ``bio_image_unet.multi_output_unet3d.Trainer``, and why:

* **One sigmoid.** The network is trained on logits (BCE-with-logits + soft Dice on the
  sigmoid). The package trainer applies the sigmoid in the model *and* again in the loss,
  so the loss can never reach its minimum.
* **No temporal-consistency loss by default.** An L1 penalty between consecutive
  predictions rewards smoothing exactly where motion is fast.

The saved checkpoint has the format ``bio_image_unet`` expects, so it can be passed to
``MotionAnalysis.detect_z_bands_fast_movie(model_path=...)`` directly.
"""

import glob
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from bio_image_unet.multi_output_unet3d.multi_output_unet3d import MultiOutputUnet3D
from torch.utils.data import DataLoader, Dataset

TARGET = 'zbands_fast_movie'
CLIP_PERCENTILES = (0.0, 99.8)   # normalisation used at inference by motion.detect_z_bands_fast_movie


@dataclass
class Domain:
    """One training folder and how often to sample from it."""
    name: str
    folder: str
    weight: float = 1.0
    real: bool = False   #: real movies get temporal subsampling and dose-reduction noise


def load_clips(folder, max_clips=None):
    """All (movie, label) pairs of a clip folder, as lists of arrays."""
    files = sorted(glob.glob(os.path.join(folder, 'volume', '*.tif')))[:max_clips]
    if not files:
        raise FileNotFoundError(f'no clips in {folder}/volume')
    movies, labels = [], []
    for f in files:
        # a clip truncated by an interrupted copy reads as a single frame; skip it instead of crashing a worker
        try:
            movie = tifffile.imread(f)
            label = tifffile.imread(os.path.join(folder, TARGET, os.path.basename(f)))
        except Exception as e:
            print(f'skipping {f}: {e}', flush=True)
            continue
        if movie.ndim != 3 or movie.shape != label.shape:
            print(f'skipping {f}: movie {movie.shape} vs label {label.shape}', flush=True)
            continue
        movies.append(movie)
        labels.append(label)
    return movies, labels


def normalize(movie):
    """Clip to percentiles and scale to [0, 1], like the prediction does."""
    movie = movie.astype(np.float32)
    lo, hi = np.percentile(movie, CLIP_PERCENTILES)
    return np.clip((movie - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


class PatchSampler(Dataset):
    """Random training patches from weighted domains, augmented on the fly.

    Augmentations: flips, transpose, time reversal, gamma; for real movies also temporal
    subsampling by 2 (a high-speed movie at half the frame rate is still high-speed) and
    dose reduction (fresh Poisson noise at a lower photon count, so clean real movies
    also teach the model about dim ones).
    """

    def __init__(self, domains, patch=(48, 128, 128), patches_per_epoch=8000, seed=0):
        self.data = []
        for d in domains:
            movies, labels = load_clips(d.folder)
            self.data.append((d, movies, labels))
            print(f'domain {d.name}: {len(movies)} clips, weight {d.weight}, real={d.real}', flush=True)
        weights = np.array([d.weight for d in domains], float)
        self.p = weights / weights.sum()
        self.patch = patch
        self.n = patches_per_epoch
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, idx, time.time_ns() % 2**31])
        domain, movies, labels = self.data[rng.choice(len(self.data), p=self.p)]
        k = rng.integers(len(movies))
        movie, label = movies[k], labels[k]
        T, P, Q = self.patch

        stride = 2 if domain.real and rng.random() < 0.25 and movie.shape[0] >= 2 * T else 1
        if movie.shape[0] < stride * (T - 1) + 1:
            raise ValueError(f'{domain.name}: clips need at least {T} frames')
        if movie.shape[1] < P or movie.shape[2] < Q:
            pad = ((0, 0), (0, max(0, P - movie.shape[1])), (0, max(0, Q - movie.shape[2])))
            movie, label = np.pad(movie, pad, mode='reflect'), np.pad(label, pad, mode='reflect')
        t0 = rng.integers(0, movie.shape[0] - stride * (T - 1))
        y0 = rng.integers(0, movie.shape[1] - P + 1)
        x0 = rng.integers(0, movie.shape[2] - Q + 1)
        window = np.s_[t0:t0 + stride * (T - 1) + 1:stride, y0:y0 + P, x0:x0 + Q]
        x = normalize(movie[window])
        y = label[window].astype(np.float32) / 255

        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        if P == Q and rng.random() < 0.5:
            x, y = x.transpose(0, 2, 1), y.transpose(0, 2, 1)
        if rng.random() < 0.5:
            x, y = x[::-1], y[::-1]
        if rng.random() < 0.5:
            x = x ** np.exp(rng.uniform(-0.35, 0.35))
        if domain.real and rng.random() < 0.3:
            photons = np.exp(rng.uniform(np.log(8), np.log(200)))
            x = normalize(rng.poisson(x * photons) / photons)

        x = np.ascontiguousarray(x, np.float32)[None]
        y = np.ascontiguousarray(y, np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(y)


def bce_dice_loss(logits, target, time_weight=0.0):
    """BCE on logits + soft Dice on probabilities (+ optional temporal L1 penalty)."""
    loss = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    loss = loss + 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    if time_weight:
        loss = loss + time_weight * (prob[:, :, 1:] - prob[:, :, :-1]).abs().mean()
    return loss


def dice_scores(pred, label, threshold=0.5):
    """(Dice on all frames, Dice on the fastest-moving quarter of frames).

    "Fast" frames are those where the label changes most from the previous frame; a model
    that smooths over time loses there first.
    """
    p, g = pred > threshold, label > threshold

    def dice(a, b):
        s = a.sum() + b.sum()
        return 2 * (a & b).sum() / s if s else np.nan

    change = 1 - (g[1:] & g[:-1]).sum((1, 2)) / np.maximum((g[1:] | g[:-1]).sum((1, 2)), 1)
    fast = np.where(change >= np.quantile(change, 0.75))[0] + 1
    return dice(p, g), dice(p[fast], g[fast])


@torch.no_grad()
def validate(model, val_sets, device):
    """Mean Dice and fast-frame Dice per validation set; whole clips, full frames."""
    model.eval()
    scores = {}
    for name, (movies, labels) in val_sets.items():
        all_, fast = [], []
        for movie, label in zip(movies, labels):
            T, H, W = (s // 8 * 8 for s in movie.shape)     # the U-Net halves each axis 3x
            x = torch.from_numpy(normalize(movie[:T, :H, :W]))[None, None].to(device)
            pred = torch.sigmoid(model(x)[TARGET]).float().cpu().numpy()[0, 0]
            a, f = dice_scores(pred, label[:T, :H, :W] / 255)
            all_.append(a)
            fast.append(f)
        scores[name] = dict(dice=float(np.nanmean(all_)), fast=float(np.nanmean(fast)))
    model.train()
    return scores


def make_checkpoint(model, n_filter, epoch, scores, settings):
    """Checkpoint in the format bio_image_unet's Predict (and so SarcAsM) loads."""
    return {
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'in_channels': 1,
        'n_filter': n_filter,
        'use_interpolation': True,
        'output_heads': {TARGET: {'channels': 1, 'activation': 'sigmoid'}},
        'clip_threshold': CLIP_PERCENTILES,
        'epoch': epoch,
        'validation': scores,
        'training_settings': settings,
    }


def train(domains, out_dir, val_sets=None, epochs=60, iters_per_epoch=1000, batch_size=8,
          patch=(48, 128, 128), n_filter=16, lr=5e-4, time_weight=0.0, init=None,
          workers=8, device=None):
    """Train the 3D U-Net and write ``best.pt``, ``last.pt`` and ``log.jsonl`` to ``out_dir``.

    Parameters
    ----------
    domains : list of Domain
        Training folders with sampling weights.
    out_dir : str
        Output folder.
    val_sets : dict of str to folder, optional
        Validation folders (same layout). The checkpoint with the best mean
        (Dice + fast-frame Dice) over all sets is kept as ``best.pt``.
    epochs, iters_per_epoch, batch_size : int
        An epoch is ``iters_per_epoch`` batches of ``batch_size`` random patches.
    patch : (int, int, int)
        Patch size (frames, rows, cols); each divisible by 8.
    n_filter : int
        Filters in the first layer (16 = the bundled model; cost grows ~n_filter^2).
    lr : float
        Peak learning rate of the one-cycle schedule.
    time_weight : float
        Weight of the temporal-consistency penalty; 0 (default) switches it off.
    init : str, optional
        Checkpoint to start from (fine-tuning).
    workers : int
        Data-loading processes.
    device : str, optional
        'cuda', 'mps' or 'cpu'; default picks the best available.
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device)
    settings = dict(domains=[vars(d) for d in domains], val_sets=val_sets, epochs=epochs,
                    iters_per_epoch=iters_per_epoch, batch_size=batch_size, patch=list(patch),
                    n_filter=n_filter, lr=lr, time_weight=time_weight, init=init)
    json.dump(settings, open(os.path.join(out_dir, 'settings.json'), 'w'), indent=1)

    sampler = PatchSampler(domains, patch, iters_per_epoch * batch_size)
    loader = DataLoader(sampler, batch_size=batch_size, num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    val = {name: load_clips(folder, max_clips=24) for name, folder in (val_sets or {}).items()}

    # the head outputs logits while training; the checkpoint says 'sigmoid' for prediction
    model = MultiOutputUnet3D(in_channels=1, n_filter=n_filter, use_interpolation=True,
                              output_heads={TARGET: {'channels': 1, 'activation': None}}).to(device)
    if init:
        model.load_state_dict(torch.load(init, map_location='cpu', weights_only=False)['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.05,
                                                   total_steps=epochs * iters_per_epoch)
    use_amp = device.type == 'cuda'
    best = -np.inf
    with open(os.path.join(out_dir, 'log.jsonl'), 'a') as log:
        for epoch in range(epochs):
            start, losses = time.time(), []
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(x)[TARGET]
                loss = bce_dice_loss(logits.float(), y, time_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                losses.append(loss.item())
            scores = validate(model, val, device) if val else {}
            score = np.mean([s['dice'] + s['fast'] for s in scores.values()]) if scores else -np.mean(losses)
            record = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(time.time() - start),
                          score=float(score), **{f'{n}_{m}': v for n, s in scores.items() for m, v in s.items()})
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
            checkpoint = make_checkpoint(model, n_filter, epoch, scores, settings)
            torch.save(checkpoint, os.path.join(out_dir, 'last.pt'))
            if score > best:
                best = score
                torch.save(checkpoint, os.path.join(out_dir, 'best.pt'))
    return os.path.join(out_dir, 'best.pt')
