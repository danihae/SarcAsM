#!/usr/bin/env python
"""Scale-invariance test: does the tracker (LEGACY vs DEFAULT) hold up under
different pixel size / time resolution WITHOUT retuning parameters?

Builds variants of the real 20kPa movie:
  - temporal kx  : take every k-th frame  -> frametime *= k, per-frame disp *= k
  - spatial sx   : downscale masks+positions by s -> pixelsize *= s, slen_px /= s

Reports SCALE-INVARIANT metrics so variants are comparable:
  frag   = n_tracks / median_detections_per_frame   (ideal ~1; lower=better)
  cov%   = fraction of detections assigned to a kept track
  fill   = mean within-track snapped-fraction of span
  nstep  = max_step_px_p99 / median_slen_px  (>~0.5 ⇒ candidate neighbour swap)

Usage: .venv/bin/python _bench/scale_test.py [--frames N]
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from bench_tracking import _load_inputs
from sarcasm.structure_modules import sarcomere_tracking as stk

LEGACY = {"threshold_mbands": 0.5, "reacquire_gap_cap": 1,
          "merge_min_bridge_snaps": 5}
DEFAULT = {}


def make_variant(zb, mb, pos, mid, slen, ori, px, ft, spatial=1, temporal=1):
    if temporal > 1:
        zb, mb = zb[::temporal], mb[::temporal]
        pos, mid = pos[::temporal], mid[::temporal]
        slen, ori = slen[::temporal], ori[::temporal]
        ft = ft * temporal
    if spatial != 1:
        H, W = int(round(zb.shape[1] / spatial)), int(round(zb.shape[2] / spatial))
        interp = cv2.INTER_AREA if spatial > 1 else cv2.INTER_LINEAR
        zb = np.stack([cv2.resize(f, (W, H), interpolation=interp) for f in zb])
        mb = np.stack([cv2.resize(f, (W, H), interpolation=interp) for f in mb])
        pos = [p.astype(np.float32) / spatial for p in pos]
        px = px * spatial
    return zb, mb, pos, mid, slen, ori, px, ft


def metrics(res, n_det_total, T, median_slen_px):
    snp = res["tracks_snapped"]; ppx = res["tracks_positions_px"]; n = res["n_tracks"]
    if n == 0:
        return dict(n_tracks=0)
    snaps = snp.sum(axis=1)
    first = np.argmax(snp, axis=1); last = T - 1 - np.argmax(snp[:, ::-1], axis=1)
    span = (last - first + 1).astype(float)
    total = int(snaps.sum())
    jumps = []
    for i in range(min(n, 4000)):
        idx = np.flatnonzero(snp[i])
        if idx.size < 2:
            continue
        d = np.linalg.norm(np.diff(ppx[i, idx], axis=0), axis=1) / np.diff(idx)
        jumps.append(d.max())
    nstep = (np.percentile(jumps, 99) / median_slen_px) if jumps else 0.0
    return dict(n_tracks=int(n), cov=round(100 * total / max(n_det_total, 1), 1),
                fill=round(float((snaps / np.maximum(span, 1)).mean()), 3),
                len_frac=round(float(snaps.mean()) / T, 3),
                nstep=round(float(nstep), 3), merges=int(res.get("n_merges", 0)))


def run(zb, mb, pos, mid, slen, ori, px, ft, ov):
    T = len(zb)
    res = stk.track_sarcomere_vectors(zb, mb, pos, mid, slen, ori, pixelsize=px,
                                      frametime=ft, compute_motion_field=False, **ov)
    n_det_total = sum(len(p) for p in pos[1:])
    med_slen_px = float(np.nanmedian(np.concatenate([s for s in slen if len(s)]))) / px
    med_det = int(np.median([len(p) for p in pos]))
    m = metrics(res, n_det_total, T, med_slen_px)
    m["frag"] = round(m["n_tracks"] / med_det, 2) if m.get("n_tracks") else None
    m["slen_px"] = round(med_slen_px, 1)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()
    px0, ft0, zb0, mb0, pos0, mid0, slen0, ori0 = _load_inputs(args.frames)
    zb0 = np.asarray(zb0); mb0 = np.asarray(mb0)

    variants = [
        ("1x (base)",       dict(spatial=1, temporal=1)),
        ("spatial 0.5x fine", dict(spatial=0.5, temporal=1)),
        ("temporal 2x",     dict(spatial=1, temporal=2)),
        ("temporal 3x",     dict(spatial=1, temporal=3)),
        ("spatial 2x",      dict(spatial=2, temporal=1)),
        ("spat2 + temp2",   dict(spatial=2, temporal=2)),
    ]
    cols = ["frag", "cov", "fill", "len_frac", "nstep", "n_tracks", "merges", "slen_px"]
    print(f"\n# scale-invariance test (frames={args.frames}); px0={px0:.4f} ft0={ft0:.4f}")
    print(f"# frag=n_tracks/med_det  cov=%det assigned  nstep=maxstep/slen_px (>~0.5 = swap risk)\n")
    hdr = f"{'variant':<16}{'cfg':<9}" + "".join(f"{c:>10}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for vname, vk in variants:
        zb, mb, pos, mid, slen, ori, px, ft = make_variant(
            zb0, mb0, pos0, mid0, slen0, ori0, px0, ft0, **vk)
        for cname, ov in (("LEGACY", LEGACY), ("DEFAULT", DEFAULT)):
            m = run(zb, mb, pos, mid, slen, ori, px, ft, ov)
            line = f"{vname:<16}{cname:<9}" + "".join(f"{str(m.get(c,'')):>10}" for c in cols)
            print(line)
        print()


if __name__ == "__main__":
    main()
