# -*- coding: utf-8 -*-
"""Measure per-frame measurement noise in ``analyze_sarcomere_vectors`` output.

Answers "did this change actually make the trajectories less noisy, without
biasing them or losing detections?" — which a timing benchmark cannot.

Run against a store that has already been analysed *and tracked*, e.g.::

    python _bench/bench_vector_noise.py test_data/high_speed_single_ACTN2-citrine_CM/20kPa.ome.zarr

Metrics, computed only on frames where the tracker actually snapped to a
detection (interpolated gap frames are excluded — they are smooth by
construction and would flatter the noise figure):

``slen_noise_nm``
    Median over tracks of the std of ``tracks_slen`` minus its 5-frame running
    median. The running median removes contraction, which is smooth on that
    scale even at peak shortening velocity, so what is left is measurement
    noise. This is the headline number.
``ori_jitter_deg``
    The same statistic on ``tracks_orientations``, unwrapped mod π first because
    the orientation is axial and +89°/−89° are the same axis.
``coupling``
    ``corr(|orientation residual|, slen residual)`` within each track. Tests
    whether orientation jitter is inflating length via cosine projection — a
    line cast δ off-axis measures ``L/cos δ``, so the coupling is one-sided
    positive if that mechanism is live. Near zero means orientation is not the
    limiting factor and smoothing it further will not help.
``mean_slen_um`` / ``valid_pct``
    Guard rails. A change that lowers noise while shifting the mean or dropping
    detections is trading accuracy for smoothness, not improving it.

A caveat worth knowing before inventing new metrics here: neighbouring sarcomere
vectors are one pixel apart on the same M-band midline, i.e. they measure the
*same* sarcomere. Their residuals correlate at ~+0.89 whether the residual is
real motion or pure noise, so spatial correlation cannot separate signal from
noise. The temporal residual above can.
"""

import argparse
import sys

import numpy as np
import zarr
from scipy.ndimage import median_filter

DEFAULT_STORE = 'test_data/high_speed_single_ACTN2-citrine_CM/20kPa.ome.zarr'


def _unwrap_axial(angles: np.ndarray) -> np.ndarray:
    """Unwrap an axial (mod π) angle trace into a continuous one."""
    steps = np.diff(angles)
    steps = (steps + np.pi / 2) % np.pi - np.pi / 2
    return np.concatenate([[angles[0]], angles[0] + np.cumsum(steps)])


def _residuals(trace: np.ndarray, measured: np.ndarray, window: int) -> np.ndarray:
    """Trace minus its running median, with gaps filled so the filter is defined."""
    filled = trace.copy()
    idx = np.arange(len(trace))
    filled[~measured] = np.interp(idx[~measured], idx[measured], trace[measured])
    return filled - median_filter(filled, size=window, mode='nearest')


def measure(store_path: str, min_coverage: float = 0.9, window: int = 5,
            min_frames: int = 40) -> dict:
    root = zarr.open(store_path, mode='r')
    try:
        tracks = root['sarcasm/tracks']
    except KeyError:
        raise SystemExit(
            f"{store_path} has no 'sarcasm/tracks' group — run track_sarcomere_vectors() first."
        )

    slen = np.asarray(tracks['slen'][:], dtype=np.float64)
    ori = np.asarray(tracks['orientations'][:], dtype=np.float64)
    snapped = np.asarray(tracks['snapped'][:], dtype=bool)

    n_tracks, n_frames = slen.shape
    keep = snapped.mean(axis=1) >= min_coverage
    slen, ori, snapped = slen[keep], ori[keep], snapped[keep]

    slen_noise, ori_jitter, coupling = [], [], []
    for i in range(slen.shape[0]):
        measured = snapped[i] & np.isfinite(slen[i]) & np.isfinite(ori[i])
        if measured.sum() < min_frames:
            continue

        res_s = _residuals(slen[i], measured, window)
        unwrapped = np.full(n_frames, np.nan)
        unwrapped[measured] = _unwrap_axial(ori[i][measured])
        res_o = _residuals(unwrapped, measured, window)

        slen_noise.append(np.std(res_s[measured]))
        ori_jitter.append(np.std(res_o[measured]))

        a, b = np.abs(res_o[measured]), res_s[measured]
        if a.std() > 0 and b.std() > 0:
            coupling.append(np.corrcoef(a, b)[0, 1])

    if not slen_noise:
        raise SystemExit(
            f'No track reached {min_coverage:.0%} coverage and {min_frames} measured frames.'
        )

    return {
        'store': store_path,
        'n_tracks_total': int(n_tracks),
        'n_tracks_used': len(slen_noise),
        'n_frames': int(n_frames),
        'slen_noise_nm': float(np.median(slen_noise) * 1e3),
        'slen_noise_p90_nm': float(np.percentile(slen_noise, 90) * 1e3),
        'ori_jitter_deg': float(np.degrees(np.median(ori_jitter))),
        'coupling': float(np.mean(coupling)) if coupling else float('nan'),
        'mean_slen_um': float(np.nanmean(slen[snapped])),
        'valid_pct': float(np.isfinite(slen[snapped]).mean() * 100),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('stores', nargs='*', default=[DEFAULT_STORE],
                        help='paths to analysed+tracked .ome.zarr stores')
    parser.add_argument('--min-coverage', type=float, default=0.9,
                        help='minimum snapped fraction for a track to count (default 0.9)')
    parser.add_argument('--window', type=int, default=5,
                        help='running-median window in frames (default 5)')
    args = parser.parse_args(argv)

    header = (f"{'store':<46s} {'tracks':>7s} {'slen noise':>11s} {'ori jitter':>11s} "
              f"{'coupling':>9s} {'mean slen':>10s} {'valid':>7s}")
    print(header)
    print('-' * len(header))
    for store in args.stores:
        r = measure(store, min_coverage=args.min_coverage, window=args.window)
        name = store if len(store) <= 46 else '…' + store[-45:]
        print(f"{name:<46s} {r['n_tracks_used']:>7d} {r['slen_noise_nm']:>9.2f}nm "
              f"{r['ori_jitter_deg']:>9.2f}° {r['coupling']:>+9.3f} "
              f"{r['mean_slen_um']:>9.4f}µm {r['valid_pct']:>6.1f}%")
    return 0


if __name__ == '__main__':
    sys.exit(main())
