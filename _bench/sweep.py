#!/usr/bin/env python
"""Run many tracker configs over the SAME loaded inputs+flow and tabulate.

Usage: .venv/bin/python _bench/sweep.py [--frames N]
Edit CONFIGS below. Flow is computed once per distinct flow-param set and reused.
"""
from __future__ import annotations
import argparse, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bench_tracking import _load_inputs, _get_flow, metrics, FLOW_KEYS
from sarcasm.structure_modules import sarcomere_tracking as stk

# (label, overrides)
CONFIGS = [
    # Faithful pre-change emulation (verified == git-HEAD baseline at 150f):
    ("LEGACY", {"threshold_mbands": 0.5, "reacquire_gap_cap": 1,
                "merge_min_bridge_snaps": 5, "merge_max_passes": 1,
                }),
    ("DEFAULT", {}),                                   # all improvements on
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    args = ap.parse_args()
    T = args.frames
    px, ft, zb, mb, pos, mid, slen, ori = _load_inputs(T)
    n_det_total = sum(len(p) for p in pos[1:])

    # group configs by flow params
    flow_cache = {}
    rows = []
    orig = stk.compute_flow_sequence
    try:
        for label, ov in CONFIGS:
            fkey = json.dumps({k: ov.get(k) for k in FLOW_KEYS}, sort_keys=True)
            if fkey not in flow_cache:
                flow_cache[fkey] = _get_flow(T, zb, mb, ov, use_cache=True)
            flows = flow_cache[fkey]
            stk.compute_flow_sequence = lambda *a, **k: flows
            kw = {k: v for k, v in ov.items() if k not in FLOW_KEYS}
            t0 = time.time()
            res = stk.track_sarcomere_vectors(
                zb, mb, pos, mid, slen, ori, pixelsize=px, frametime=ft,
                compute_motion_field=False, **kw)
            m = metrics(res, n_det_total, T)
            m["secs"] = round(time.time() - t0, 1)
            rows.append((label, m))
    finally:
        stk.compute_flow_sequence = orig

    cols = ["n_tracks", "len_mean", "len_median", "n_full_span", "coverage_pct",
            "fill_ratio_mean", "n_internal_gaps", "n_merges", "slen_std_mean",
            "max_step_px_p99", "secs"]
    med_det = int(np.median([len(p) for p in pos]))
    print(f"\n# frames={T}  median_det/frame={med_det}  n_det_total(f1..)={n_det_total}")
    print(f"# fragmentation = n_tracks/median_det (lower=better; ideal~1.0)\n")
    hdr = f"{'config':<14}" + "".join(f"{c:>14}" for c in cols) + f"{'frag':>8}"
    print(hdr); print("-" * len(hdr))
    for label, m in rows:
        line = f"{label:<14}"
        for c in cols:
            v = m.get(c, "")
            line += f"{v:>14}"
        frag = round(m["n_tracks"] / med_det, 2) if m.get("n_tracks") else ""
        line += f"{frag:>8}"
        print(line)


if __name__ == "__main__":
    main()
