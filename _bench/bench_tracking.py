#!/usr/bin/env python
"""Empirical benchmark for sarcomere-vector tracking: track length & continuity.

Runs the *pure* ``sarcomere_tracking.track_sarcomere_vectors`` function directly
on the real 20kPa high-speed dataset (no disk writes, does NOT touch the shared
structure.json fixture). Flow is cached per (window, flow-param hash) so snap /
merge experiments are fast.

Usage:
    .venv/bin/python _bench/bench_tracking.py [--frames N] [--overrides JSON] [--no-cache-flow] [--label NAME]

Quality note: without ground truth we report coverage + length metrics AND
quality proxies (within-track slen std, max position jump) so that a change
which merely inflates coverage by joining wrong sarcomeres is visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sarcasm.structure import Structure
from sarcasm.structure_modules import sarcomere_tracking as stk

TIF = "test_data/high_speed_single_ACTN2-citrine_CM/20kPa.tif"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
os.makedirs(CACHE, exist_ok=True)

# flow params that affect the cached flow stack
FLOW_KEYS = ("threshold_zbands", "threshold_mbands", "dt_clip", "farneback_kwargs")


def _load_inputs(n_frames):
    sarc = Structure(TIF, restart=False)
    px = sarc.metadata.pixelsize
    ft = sarc.metadata.frametime
    frames = list(range(n_frames))
    zb = tifffile.imread(sarc.file_zbands, key=range(0, n_frames))
    mb = tifffile.imread(sarc.file_mbands, key=range(0, n_frames))
    pos = [np.asarray(sarc.data["pos_vectors_px"][t], np.int32) for t in frames]
    mid = [np.asarray(sarc.data["midline_id_vectors"][t], np.int64) for t in frames]
    slen = [np.asarray(sarc.data["sarcomere_length_vectors"][t], np.float32) for t in frames]
    ori = [np.asarray(sarc.data["sarcomere_orientation_vectors"][t], np.float32) for t in frames]
    return px, ft, zb, mb, pos, mid, slen, ori


def _flow_cache_key(n_frames, ov):
    fp = {k: ov.get(k) for k in FLOW_KEYS}
    h = hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(CACHE, f"flow_{n_frames}_{h}.npy")


def _get_flow(n_frames, zb, mb, ov, use_cache=True):
    key = _flow_cache_key(n_frames, ov)
    if use_cache and os.path.exists(key):
        return np.load(key)
    flows = stk.compute_flow_sequence(
        zb, mb,
        threshold=ov.get("threshold_zbands", 0.5),
        threshold_m=ov.get("threshold_mbands", 0.25),
        clip=ov.get("dt_clip", 20.0),
        farneback_kwargs=ov.get("farneback_kwargs"),
    )
    if use_cache:
        np.save(key, flows)
    return flows


def metrics(res, n_det_total, T):
    snapped = res["tracks_snapped"]          # (n, T) bool
    pos = res["tracks_positions_px"]         # (n, T, 2)
    slen = res["tracks_slen"]                # (n, T)
    n = res["n_tracks"]
    out = {"n_tracks": int(n), "n_merges": int(res.get("n_merges", 0))}
    if n == 0:
        return out
    snaps = snapped.sum(axis=1)              # per-track snap count
    # span = last - first + 1 (over snapped frames)
    first = np.argmax(snapped, axis=1)
    last = T - 1 - np.argmax(snapped[:, ::-1], axis=1)
    span = (last - first + 1).astype(float)
    fill = snaps / np.maximum(span, 1)
    total_snaps = int(snaps.sum())
    out.update({
        "total_snaps": total_snaps,
        "coverage_pct": round(100.0 * total_snaps / max(n_det_total, 1), 2),
        "len_mean": round(float(snaps.mean()), 2),
        "len_median": int(np.median(snaps)),
        "len_p90": int(np.percentile(snaps, 90)),
        "len_max": int(snaps.max()),
        "span_mean": round(float(span.mean()), 2),
        "fill_ratio_mean": round(float(fill.mean()), 3),
        "n_full_span": int((span >= T).sum()),
        "n_internal_gaps": int((span - snaps).sum()),
    })
    # quality proxies (lower = purer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slen_std = np.nanstd(np.where(snapped, slen, np.nan), axis=1)
    out["slen_std_mean"] = round(float(np.nanmean(slen_std)), 4)
    # max along-axis position jump between consecutive snapped frames
    jumps = []
    for i in range(min(n, 4000)):
        idx = np.flatnonzero(snapped[i])
        if idx.size < 2:
            continue
        p = pos[i, idx]
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dt = np.diff(idx)
        jumps.append((d / dt).max())
    out["max_step_px_p99"] = round(float(np.percentile(jumps, 99)), 2) if jumps else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--overrides", type=str, default="{}")
    ap.add_argument("--no-cache-flow", action="store_true")
    ap.add_argument("--label", type=str, default="run")
    args = ap.parse_args()
    ov = json.loads(args.overrides)

    px, ft, zb, mb, pos, mid, slen, ori = _load_inputs(args.frames)
    T = args.frames
    n_det_total = sum(len(p) for p in pos[1:])  # detections eligible to be claimed (frames 1..T-1)

    flows = _get_flow(args.frames, zb, mb, ov, use_cache=not args.no_cache_flow)

    # Monkeypatch compute_flow_sequence to return the cached flow (so we don't
    # recompute inside the tracker) while still exercising the real tracker.
    orig = stk.compute_flow_sequence
    stk.compute_flow_sequence = lambda *a, **k: flows
    try:
        t0 = time.time()
        kw = {k: v for k, v in ov.items() if k not in FLOW_KEYS}
        res = stk.track_sarcomere_vectors(
            zb, mb, pos, mid, slen, ori,
            pixelsize=px, frametime=ft,
            compute_motion_field=False,
            **kw,
        )
        dt = time.time() - t0
    finally:
        stk.compute_flow_sequence = orig

    m = metrics(res, n_det_total, T)
    m["track_secs"] = round(dt, 2)
    print(json.dumps({"label": args.label, "frames": T, "n_det_total": n_det_total,
                      "overrides": ov, "metrics": m}, indent=2))


if __name__ == "__main__":
    main()
