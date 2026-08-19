"""Label-consistent augmentation for ContractionNet training traces.

Two properties decide what belongs here.

The input conditioning in :func:`~contraction_net.prediction.prepare_robust_input` is
affine-equivariant, so a global rescaling or a constant offset of a trace is cancelled
exactly and augmenting with either is wasted. What is not cancelled is anything that
changes the *ratio* of the twitch to the noise, drift, steps and quantisation around it,
which is what the transforms below vary.

The sign flip is only sound under the ``'symmetric'`` convention. Under ``'q90'`` a
flipped trace arrives with its rest level at ``-1`` instead of ``0``, so mixing the two
would train the network on two contradictory conventions rather than on one invariance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

__all__ = ['AugmentConfig', 'twitch_scale', 'time_warp', 'augment']


@dataclass
class AugmentConfig:
    """Probabilities and ranges for :func:`augment`.

    Fractions are relative to the trace's own twitch amplitude from
    :func:`twitch_scale`, so every setting is scale free.
    """

    p_sign_flip: float = 0.5
    p_time_warp: float = 0.5
    warp_range: Tuple[float, float] = (0.4, 3.0)
    p_nonuniform_warp: float = 0.15
    p_noise: float = 0.7
    noise_frac: Tuple[float, float] = (0.02, 0.35)
    noise_rho: Tuple[float, float] = (0.0, 0.9)
    p_drift: float = 0.4
    drift_frac: float = 0.6
    p_bleach: float = 0.3
    bleach_frac: float = 0.4
    p_steps: float = 0.25
    n_steps: Tuple[int, int] = (1, 4)
    step_frac: Tuple[float, float] = (0.1, 0.6)
    p_gaps: float = 0.35
    gap_frac: Tuple[float, float] = (0.01, 0.20)
    p_quantise: float = 0.25
    quantise_frac: Tuple[float, float] = (0.02, 0.20)
    p_outliers: float = 0.3
    n_outliers: Tuple[int, int] = (1, 9)
    outlier_scale: Tuple[float, float] = (0.3, 1.2)
    min_len: int = 32
    max_len: int = 8192

    @classmethod
    def for_pool(cls, pool: str, sign_flip: bool = True) -> 'AugmentConfig':
        """Preset for a data pool.

        ``'sim'`` gets only the geometric transforms: the simulator applies its own
        nuisances at generation time and :func:`~contraction_net.simulation.renoise`
        redraws them per epoch, so adding a second layer would double-count them.
        """
        cfg = cls()
        if pool == 'sim':
            cfg = replace(cfg, p_noise=0.0, p_drift=0.0, p_bleach=0.0, p_steps=0.0,
                          p_gaps=0.0, p_quantise=0.0, p_outliers=0.0,
                          p_time_warp=0.35)
        if not sign_flip:
            cfg = replace(cfg, p_sign_flip=0.0)
        return cfg


def _odd(n: int) -> int:
    n = int(max(5, n))
    return n if n % 2 else n + 1


def twitch_scale(signal: np.ndarray, label: Optional[np.ndarray] = None) -> float:
    """Robust amplitude of the twitch, with slow drift removed.

    The raw 10-90 spread is not usable as a reference here: on a Z-band position row it
    is dominated by drift, which can exceed the twitch by an order of magnitude, so
    nuisances sized against it would swamp the signal.
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 16:
        spread = float(np.ptp(x))
        return spread if spread > 0 else 1.0
    n_events = 1
    if label is not None:
        lab = np.asarray(label, bool)
        n_events = max(int(np.count_nonzero(np.diff(lab.astype(np.int8)) > 0)), 1)
    window = min(_odd(max(11, 1.5 * x.size / n_events)), _odd(x.size - 1))
    resid = x - savgol_filter(x, window, 1) if window < x.size else x - x.mean()
    spread = float(np.percentile(resid, 90) - np.percentile(resid, 10))
    if spread <= 0:
        spread = float(np.std(resid)) or float(np.ptp(x)) or 1.0
    return spread


def time_warp(signal: np.ndarray, label: np.ndarray, rng: np.random.Generator,
              lo: float = 0.4, hi: float = 3.0, nonuniform: bool = False,
              min_len: int = 32, max_len: int = 8192):
    """Resample a trace and its label onto a stretched or compressed time base.

    Covers acquisition rate directly: a contraction resolved by 400 frames becomes one
    resolved by 130, which integer frame-dropping cannot reach.

    Returns
    -------
    tuple of np.ndarray
        The resampled signal and label. Unchanged if the result would be too short.
    """
    x = np.asarray(signal, dtype=float)
    lab = np.asarray(label, bool)
    n = x.size
    if n < min_len:
        return x, lab
    rate = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if nonuniform:
        m = int(min(max_len, max(min_len, n / rate * 1.5)))
        u = np.linspace(0, 1, m)
        wobble = (np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * u + rng.uniform(0, 2 * np.pi))
                  + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * u
                                 + rng.uniform(0, 2 * np.pi)))
        steps = rate * (1 + 0.35 * wobble / 1.5)
        pos = np.cumsum(np.clip(steps, 0.05, None))
        pos = pos[pos < n - 1]
    else:
        pos = np.arange(0, n - 1, rate)
    if pos.size < min_len:
        return x, lab
    pos = pos[:max_len]
    grid = np.arange(n, dtype=float)
    out = np.interp(pos, grid, x)
    # nearest neighbour on the label: interpolating a boolean would invent partial states
    out_lab = lab[np.clip(np.rint(pos).astype(int), 0, n - 1)]
    return out, out_lab


def _ar1_noise(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    from .simulation import _ar1
    return _ar1(n, rho, rng)


def augment(signal: np.ndarray, label: np.ndarray, rng: np.random.Generator,
            cfg: AugmentConfig = AugmentConfig()):
    """Apply the augmentation chain to one trace.

    Parameters
    ----------
    signal, label : np.ndarray
        Trace and its per-frame contraction state.
    rng : np.random.Generator
        Per-sample generator.
    cfg : AugmentConfig, optional
        Probabilities and ranges.

    Returns
    -------
    tuple of np.ndarray
        Augmented signal and the label transformed with it.
    """
    x = np.asarray(signal, dtype=float).copy()
    lab = np.asarray(label, bool).copy()

    if rng.random() < cfg.p_sign_flip:
        x = -x

    if rng.random() < cfg.p_time_warp:
        x, lab = time_warp(x, lab, rng, *cfg.warp_range,
                           nonuniform=rng.random() < cfg.p_nonuniform_warp,
                           min_len=cfg.min_len, max_len=cfg.max_len)

    # reference every nuisance to the twitch measured before any of them are added
    amp = twitch_scale(x, lab)
    n = x.size

    if rng.random() < cfg.p_noise:
        sigma = amp * rng.uniform(*cfg.noise_frac)
        x = x + sigma * _ar1_noise(n, rng.uniform(*cfg.noise_rho), rng)

    if rng.random() < cfg.p_drift:
        t = np.linspace(0, 1, n)
        # periods of at least half the trace, so wander cannot mimic a twitch
        x = x + amp * rng.uniform(0, cfg.drift_frac) * (
            np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t + rng.uniform(0, 2 * np.pi))
            + 0.5 * np.sin(2 * np.pi * rng.uniform(0.3, 2.0) * t
                           + rng.uniform(0, 2 * np.pi))) / 1.5

    if rng.random() < cfg.p_bleach:
        x = x + amp * rng.uniform(-cfg.bleach_frac, cfg.bleach_frac) * np.linspace(0, 1, n)

    if rng.random() < cfg.p_steps:
        for _ in range(int(rng.integers(*cfg.n_steps))):
            i = int(rng.integers(1, max(2, n)))
            x[i:] += amp * rng.uniform(*cfg.step_frac) * rng.choice([-1.0, 1.0])

    if rng.random() < cfg.p_outliers:
        k = int(rng.integers(*cfg.n_outliers))
        idx = rng.integers(0, n, k)
        x[idx] += amp * rng.normal(0, rng.uniform(*cfg.outlier_scale), k)

    if rng.random() < cfg.p_quantise:
        step = amp * rng.uniform(*cfg.quantise_frac)
        if step > 0:
            x = np.round(x / step) * step

    if rng.random() < cfg.p_gaps:
        x = _gap_and_fill(x, rng, rng.uniform(*cfg.gap_frac))

    return x, lab


def _gap_and_fill(x: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    """Blank runs of frames and fill them linearly, as the inference path does."""
    n = x.size
    n_gap = int(round(frac * n))
    if n_gap < 1 or n < 8:
        return x
    mask = np.zeros(n, dtype=bool)
    remaining = n_gap
    while remaining > 0:
        length = int(min(remaining, rng.integers(2, max(3, n // 20))))
        start = int(rng.integers(0, max(1, n - length)))
        mask[start:start + length] = True
        remaining -= length
    if not mask.any() or mask.all():
        return x
    idx = np.arange(n)
    out = x.copy()
    out[mask] = np.interp(idx[mask], idx[~mask], out[~mask])
    return out
