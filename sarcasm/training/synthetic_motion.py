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


"""Synthetic high-speed movies made from single 2D training images.

The 3D U-Net needs movies with Z-band labels in every frame. Hand-labelling movies is not
feasible, but the 2D training set already has curated labels. This module animates one
2D image together with its label, so the label stays exact in every frame.

How a clip is made
------------------
1. **Scale.** The image is resized so that the sarcomere period lands in
   ``period_range`` px (default 10-32 px: high-magnification recordings).
2. **Motion.** A displacement field ``u(x, t)`` moves image and label together:

   * contraction: local shortening along the myofibril axis ``n(x)`` (taken from the
     orientation label) towards a random centre, driven by a twitch waveform. Its
     amplitude varies smoothly in space and can drop to zero or turn negative (passive or
     elongating regions), and its onset is delayed a little from place to place, so
     sarcomeres never contract uniformly;
   * slow drift and a smooth elastic wobble;
   * high-frequency jitter: uncorrelated frame-to-frame shifts (heavy-tailed, sometimes
     only in short bursts) and non-rigid local jitter. Because the label follows every
     jump, a model that smooths over time is penalised whenever motion is fast.
3. **Optics.** Defocus blur that breathes with the beat, and motion blur from averaging
   sub-frames across the exposure time.
4. **Camera.** Photon counts with Poisson noise, read noise, fixed row/column offsets,
   hot pixels, bleaching and illumination flicker.

All rates are in frames and all lengths in pixels. The 3D U-Net is meant for high-speed
recordings only, so a contraction always spans at least 6 frames.

Example
-------
>>> clip = make_clip(image, zbands, orientation, sarcomere_mask, n_frames=64)
>>> clip.movie.shape, clip.label.shape        # (64, 192, 192) uint16, uint8 (label * 255)
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class SyntheticClip:
    """A synthetic movie, its Z-band label and the parameters that made it."""
    movie: np.ndarray        #: (T, H, W) uint16 camera counts
    label: np.ndarray        #: (T, H, W) uint8, Z-band probability * 255
    params: dict = field(default_factory=dict)


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _smooth_noise(rng, shape, sigma):
    """Gaussian random field with unit standard deviation and correlation length ~sigma."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode='reflect')
    return f / (f.std() + 1e-8)


def sarcomere_period_px(zbands: np.ndarray) -> float:
    """Dominant Z-band spacing in px, from the radial power spectrum of a Z-band label.

    Thin Z-band lines put much power into the second harmonic, so the result can be half
    the true period. It is only used to bring the training images to a common range of
    scales, where a factor of two is absorbed by the random target period.
    """
    z = zbands.astype(np.float32) - zbands.mean()
    power = np.abs(np.fft.rfft2(z)) ** 2
    k = np.hypot(np.fft.fftfreq(z.shape[0])[:, None], np.fft.rfftfreq(z.shape[1])[None, :])
    bins = np.linspace(1 / 60, 1 / 3, 300)
    idx = np.digitize(k.ravel(), bins)
    profile = np.bincount(idx, power.ravel(), minlength=len(bins) + 1)[1:len(bins)]
    counts = np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)]
    return float(1 / bins[np.argmax(profile / np.maximum(counts, 1))])


def myofibril_axis(orientation: np.ndarray, sarcomere_mask: np.ndarray, sigma: float = 8):
    """Smooth unit vector field ``(n_row, n_col)`` along the myofibrils.

    The orientation label stores the angle in the convention of the analysis: the axis
    for an angle ``o`` is ``(sin o, -cos o)`` in (row, col). Angles are averaged as doubled
    angles, which fills the gaps between sarcomeres and ignores the 180° ambiguity.
    """
    valid = np.isfinite(orientation) & (sarcomere_mask > 0)
    o = np.where(valid, orientation, 0)
    cos2 = ndimage.gaussian_filter(np.where(valid, np.cos(2 * o), 0), sigma)
    sin2 = ndimage.gaussian_filter(np.where(valid, np.sin(2 * o), 0), sigma)
    if not valid.any():          # no orientation at all: one axis for the whole image
        cos2, sin2 = np.ones_like(o), np.zeros_like(o)
    o_smooth = 0.5 * np.arctan2(sin2, cos2)
    return np.sin(o_smooth), -np.cos(o_smooth)


def twitch(t, t_contraction):
    """Twitch waveform: 0 at rest, 1 at peak shortening (after ~0.35 t_contraction)."""
    s = np.maximum(t, 0) / t_contraction / 0.35
    return np.where(t > 0, s ** 2.5 * np.exp(2.5 * (1 - s)), 0.0)


def beat_train(t, t_contraction, beat_period, phase):
    """A regular train of twitches, one every ``beat_period`` frames."""
    out = np.zeros_like(np.asarray(t, dtype=float))
    onset = phase - beat_period * np.ceil((phase + 3 * t_contraction) / beat_period)
    while onset < np.max(t) + 1:
        out = out + twitch(t - onset, t_contraction)
        onset += beat_period
    return np.clip(out, 0, 1.2)


def sample_params(rng: np.random.Generator) -> dict:
    """Draw one random set of motion, optics and camera parameters.

    The ranges cover high-speed recordings of beating cardiomyocytes, from clean
    high-magnification movies to dim, noisy ones.
    """
    t_c = _loguniform(rng, 6, 60)
    p = dict(
        # motion
        t_contraction=t_c,                              # frames from onset to relaxation
        beat_period=t_c * rng.uniform(2.0, 8.0),        # frames between beats
        phase=rng.uniform(0, 8 * t_c),
        strain=rng.uniform(0.0, 0.16) if rng.random() > 0.1 else 0.0,  # peak shortening
        heterogeneity=rng.uniform(0.0, 0.9),            # spatial spread of the strain
        heterogeneity_scale=_loguniform(rng, 15, 120),  # px
        delay=rng.uniform(0, 0.25) * t_c,               # frames, spatial spread of the onset
        drift=rng.uniform(0, 4),                        # px over the clip
        wobble=rng.uniform(0, 1.5),                     # px, smooth elastic deformation
        jitter=0.0,                                     # px per frame, rigid, uncorrelated
        jitter_in_bursts=rng.random() < 0.4,
        local_jitter=_loguniform(rng, 0.2, 2.5) if rng.random() < 0.4 else 0.0,  # px
        local_jitter_scale=rng.uniform(10, 40),         # px
        # optics
        defocus=rng.uniform(0, 1.0),                    # px, blur sigma at rest
        defocus_beat=rng.uniform(0, 1.5) if rng.random() < 0.5 else 0.0,  # extra at peak
        exposure=rng.uniform(0, 1.0),                   # fraction of the frame interval
        # camera
        peak_photons=_loguniform(rng, 5, 2000),
        background=_loguniform(rng, 0.02, 0.5),         # fraction of peak
        read_noise=rng.uniform(0, 3),
        fixed_pattern=rng.uniform(0, 1.0) if rng.random() < 0.3 else 0.0,
        hot_pixel_fraction=_loguniform(rng, 1e-6, 3e-4) if rng.random() < 0.3 else 0.0,
        bleaching=rng.uniform(0, 0.3),                  # fraction lost over the clip
        flicker=rng.uniform(0, 0.03),
        smooth_source=rng.random() < 0.5,               # hide the source image's own noise
    )
    if rng.random() < 0.6:   # bursts may shake harder than continuous jitter
        p['jitter'] = _loguniform(rng, 0.2, 4.0 if p['jitter_in_bursts'] else 2.0)
    return p


def make_clip(image: np.ndarray, zbands: np.ndarray, orientation: np.ndarray,
              sarcomere_mask: np.ndarray, n_frames: int = 64, size=(192, 192),
              period_range=(10, 32), rng: np.random.Generator = None,
              params: dict = None) -> SyntheticClip:
    """Animate one 2D training image into a synthetic high-speed movie.

    Parameters
    ----------
    image, zbands, orientation, sarcomere_mask : np.ndarray
        One sample of the 2D training set: image, Z-band label (0-1), orientation label
        (rad) and sarcomere mask.
    n_frames : int
        Frames in the clip.
    size : (int, int)
        Frame size (H, W) in px.
    period_range : (float, float)
        The sarcomere period in the clip is drawn log-uniformly from this range (px).
    rng : np.random.Generator, optional
        Random generator; pass a seeded one for reproducible clips.
    params : dict, optional
        Parameters from :func:`sample_params` to use instead of random ones.

    Returns
    -------
    SyntheticClip
    """
    rng = np.random.default_rng() if rng is None else rng
    p = sample_params(rng) if params is None else dict(params)
    H, W = size

    # 1. scale: bring the sarcomere period into period_range
    period = sarcomere_period_px(zbands)
    scale = float(np.clip(_loguniform(rng, *period_range) / period, 0.15, 1.7))
    p.update(source_period_px=period, scale=scale, period_px=period * scale)

    # crop a source region (with margin for the motion), preferring one with Z-bands
    margin = 24
    src_h = int(np.ceil((H + 2 * margin) / scale)) + 2
    src_w = int(np.ceil((W + 2 * margin) / scale)) + 2
    arrays = [image.astype(np.float32), zbands.astype(np.float32), orientation, sarcomere_mask]
    if image.shape[0] < src_h or image.shape[1] < src_w:
        ph, pw = max(0, src_h - image.shape[0]), max(0, src_w - image.shape[1])
        pad = ((ph // 2 + 1, ph - ph // 2 + 1), (pw // 2 + 1, pw - pw // 2 + 1))
        arrays = [np.pad(a, pad, mode='reflect') for a in arrays]
    img, lab, ori, smask = arrays
    for _ in range(20):
        y0 = rng.integers(0, img.shape[0] - src_h + 1)
        x0 = rng.integers(0, img.shape[1] - src_w + 1)
        if lab[y0:y0 + src_h, x0:x0 + src_w].mean() > 0.01:
            break
    crop = np.s_[y0:y0 + src_h, x0:x0 + src_w]
    img, lab = img[crop], lab[crop]
    axis_r, axis_c = myofibril_axis(ori[crop], smask[crop])
    if p['smooth_source']:
        img = ndimage.gaussian_filter(img, 0.8)
    lo, hi = np.percentile(img, [0.5, 99.9])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    # output grid (with margin) in source coordinates; the rescale lives in this grid
    yy, xx = np.meshgrid(np.arange(-margin, H + margin) / scale,
                         np.arange(-margin, W + margin) / scale, indexing='ij')
    gh, gw = yy.shape
    n_r = ndimage.map_coordinates(axis_r, [yy, xx], order=1, mode='nearest')
    n_c = ndimage.map_coordinates(axis_c, [yy, xx], order=1, mode='nearest')

    # 2. motion fields
    amplitude = 1 + p['heterogeneity'] * _smooth_noise(rng, (gh, gw), p['heterogeneity_scale'])
    delay = p['delay'] * _smooth_noise(rng, (gh, gw), p['heterogeneity_scale'])
    centre = rng.uniform(-0.3, 1.3) * gh, rng.uniform(-0.3, 1.3) * gw
    rows, cols = np.mgrid[0:gh, 0:gw].astype(np.float32)
    along_axis = n_r * (rows - centre[0]) + n_c * (cols - centre[1])   # (x - c) . n, px

    t_all = np.arange(n_frames, dtype=float)
    drift = np.stack([ndimage.gaussian_filter1d(rng.standard_normal(n_frames), n_frames / 6)
                      for _ in range(2)], 1)
    drift *= p['drift'] / (np.abs(drift).max() + 1e-8)
    jitter = rng.standard_t(3, (n_frames, 2)) * p['jitter'] / np.sqrt(3)   # t(3) has variance 3
    if p['jitter_in_bursts']:
        shaking = np.zeros(n_frames, bool)
        for _ in range(rng.integers(1, 4)):
            start = rng.integers(0, n_frames)
            shaking[start:start + rng.integers(2, max(3, n_frames // 4))] = True
        jitter[~shaking] = 0
    wobble_fields = [_smooth_noise(rng, (gh, gw), 40) for _ in range(4)]
    wobble_t = np.stack([ndimage.gaussian_filter1d(rng.standard_normal(n_frames),
                                                   max(p['t_contraction'] / 2, 1)) for _ in range(2)], 1)
    wobble_t *= p['wobble'] / (np.abs(wobble_t).max() + 1e-8)
    local_fields = [_smooth_noise(rng, (gh, gw), p['local_jitter_scale']) for _ in range(2)]
    local_t = rng.standard_normal((n_frames, 2)) * p['local_jitter']

    def displacement(t):
        """(u_row, u_col) at time t in output px."""
        strain = p['strain'] * amplitude * beat_train(t - delay, p['t_contraction'],
                                                      p['beat_period'], p['phase'])
        u_r = -strain * n_r * along_axis
        u_c = -strain * n_c * along_axis
        k = int(np.clip(round(t), 0, n_frames - 1))
        u_r = u_r + drift[k, 0] + jitter[k, 0] + wobble_t[k, 0] * wobble_fields[0] \
            + wobble_t[k, 1] * wobble_fields[1] + local_t[k, 0] * local_fields[0]
        u_c = u_c + drift[k, 1] + jitter[k, 1] + wobble_t[k, 0] * wobble_fields[2] \
            + wobble_t[k, 1] * wobble_fields[3] + local_t[k, 1] * local_fields[1]
        return u_r, u_c

    # 3. render frames: sub-frames across the exposure give motion blur
    n_sub = 1 if p['exposure'] < 0.05 else 4
    sub_times = np.linspace(-0.5, 0.5, n_sub) * p['exposure'] if n_sub > 1 else np.zeros(1)
    beat = beat_train(t_all, p['t_contraction'], p['beat_period'], p['phase'])
    movie = np.empty((n_frames, H, W), np.float32)
    label = np.empty((n_frames, H, W), np.float32)
    inner = np.s_[margin:margin + H, margin:margin + W]
    for i, t in enumerate(t_all):
        frame = np.zeros((gh, gw), np.float32)
        for j, dt in enumerate(sub_times):
            u_r, u_c = displacement(t + dt)
            # backward mapping: output pixel x shows the source point x - u
            coords = [yy - u_r / scale, xx - u_c / scale]
            frame += ndimage.map_coordinates(img, coords, order=1, mode='reflect')
            if j == n_sub // 2:
                label[i] = ndimage.map_coordinates(lab, coords, order=1, mode='reflect')[inner]
        frame /= n_sub
        blur = p['defocus'] + p['defocus_beat'] * beat[i]
        if blur > 0.15:
            frame = ndimage.gaussian_filter(frame, blur)
        movie[i] = frame[inner]

    # 4. camera
    bleach = np.exp(np.log(1 - p['bleaching']) * t_all / max(n_frames - 1, 1))
    flicker = 1 + p['flicker'] * rng.standard_normal(n_frames)
    photons = p['peak_photons'] * (movie * (bleach * flicker)[:, None, None] + p['background'])
    counts = rng.poisson(np.maximum(photons, 0)).astype(np.float32)
    if p['read_noise'] > 0:
        counts += rng.normal(0, p['read_noise'], counts.shape)
    if p['fixed_pattern'] > 0:
        counts += p['fixed_pattern'] * (rng.standard_normal((1, H, 1)) + rng.standard_normal((1, 1, W)))
    if p['hot_pixel_fraction'] > 0:
        hot = rng.random((H, W)) < p['hot_pixel_fraction']
        counts[:, hot] += p['peak_photons'] * rng.uniform(0.5, 3, hot.sum())
    camera_offset = 100.0
    movie_u16 = np.clip(np.round(counts + camera_offset), 0, 65535).astype(np.uint16)
    label_u8 = np.clip(np.round(label * 255), 0, 255).astype(np.uint8)
    return SyntheticClip(movie_u16, label_u8, p)
