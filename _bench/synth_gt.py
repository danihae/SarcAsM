#!/usr/bin/env python
"""Synthetic GROUND-TRUTH benchmark for the snap+merge identity logic.

Strategy: bypass the optical-flow engine by feeding the tracker an ANALYTIC
dense flow field that is exactly consistent with the ground-truth motion. This
isolates the snap/merge/assignment logic (where identity swaps happen) under
perfect prediction, with full ground truth, so we can measure:

  - fragments per GT sarcomere   (ideal: 1)
  - track PURITY                 (fraction of a track's snaps on its dominant GT id; 1.0 = no swap)
  - n_impure tracks (purity<0.95) = identity swaps
  - GT coverage                  (fraction of GT detections captured by kept tracks)
  - mean track length

Detection dropout is the dominant fragmentation driver (mimics transient U-Net
misses); jitter + burst dropouts + crossing/adjacent myofibrils stress the
identity logic.

Usage: .venv/bin/python _bench/synth_gt.py            # runs a config table
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sarcasm.structure_modules import sarcomere_tracking as stk

PX = 0.06117          # um/px (matches real dataset)
FT = 0.0164


def build_scene(T=120, n_myo=14, n_sarc=18, seed=0):
    """Generate GT sarcomere trajectories + a consistent analytic flow stack.

    Returns gt_pos (T, G, 2), gt_ori (G,), gt_slen_px (T, G), flows (T-1,H,W,2),
    H, W.  Sarcomere axis convention: (sin theta, cos theta) in (row, col)."""
    rng = np.random.default_rng(seed)
    H, W = max(220, 18 * n_myo + 40), 1024
    x0 = W * 0.5                       # contraction anchor (x)
    # myofibrils: nearly parallel & horizontal (axis ~ +x => theta ~ 0) so chains
    # never cross / overlap -> a clean, separable ground truth (no coincident
    # sarcomeres). Tiny orientation noise keeps the along/perp gates exercised.
    thetas = rng.uniform(-0.03, 0.03, n_myo).astype(np.float32)
    rows = np.linspace(20, H - 20, n_myo) + rng.uniform(-2, 2, n_myo)
    L0 = 30.0                          # rest sarcomere length px (~1.8um)
    # build rest positions per myofibril
    centers0 = []   # (G,2)
    gt_ori = []     # (G,)
    myo_of = []     # (G,)
    for m in range(n_myo):
        th = thetas[m]; s, c = np.sin(th), np.cos(th)
        # chain centered on the field, spaced L0 along axis
        offs = (np.arange(n_sarc) - (n_sarc - 1) / 2.0) * L0
        cy = rows[m] + offs * s
        cx = (W * 0.5) + offs * c
        for k in range(n_sarc):
            centers0.append((cy[k], cx[k])); gt_ori.append(th); myo_of.append(m)
    centers0 = np.array(centers0, np.float64)         # (G,2)
    gt_ori = np.array(gt_ori, np.float32)
    G = len(centers0)

    # time-varying contraction strength k(t): sarcomeres contract toward x0
    period = 40.0
    drift = rng.uniform(-0.3, 0.3, 2)                 # global px/frame drift
    gt_pos = np.zeros((T, G, 2), np.float64)
    flows = np.zeros((T - 1, H, W, 2), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def field(t):
        # contraction amplitude oscillates; sway in y
        k = 0.015 * np.sin(2 * np.pi * t / period)            # fractional contraction
        sway = 1.2 * np.sin(2 * np.pi * t / (period * 1.3))   # px lateral
        return k, sway

    pos = centers0.copy()
    gt_pos[0] = pos
    for t in range(T - 1):
        k, sway = field(t)
        # displacement field [dy,dx] applied this step
        dx = -k * (xx - x0) + drift[1]
        dy = np.full_like(xx, sway * 0.04) + drift[0]
        flows[t, ..., 0] = dy
        flows[t, ..., 1] = dx
        # advance GT positions by the same field (sampled at their location)
        py = pos[:, 0]; pxx = pos[:, 1]
        d_x = -k * (pxx - x0) + drift[1]
        d_y = np.full(G, sway * 0.04) + drift[0]
        pos = pos + np.column_stack([d_y, d_x])
        gt_pos[t + 1] = pos

    # GT slen per frame: local neighbour spacing along axis (approx via L0*(1-k cum))
    # simpler: derive from instantaneous contraction state -> all ~ L0 scaled
    gt_slen_px = np.full((T, G), L0, np.float32)
    for t in range(T):
        k, _ = field(t)
        gt_slen_px[t] = L0 * (1.0 - 0.6 * k)   # mild slen modulation
    return gt_pos, gt_ori, gt_slen_px, flows, H, W, np.array(myo_of)


def make_detections(gt_pos, gt_ori, gt_slen_px, p_drop=0.15, burst=0.04,
                    jitter=0.6, ori_noise=0.05, slen_noise=0.03, seed=1):
    """Per-frame detections from GT with dropout + jitter. Returns lists +
    per-frame det->GT-id map."""
    rng = np.random.default_rng(seed)
    T, G, _ = gt_pos.shape
    pos_all, ori_all, slen_all, mid_all, detgt_all = [], [], [], [], []
    for t in range(T):
        keep = rng.random(G) > p_drop
        # burst: occasionally drop a contiguous stretch of a myofibril
        if rng.random() < burst:
            a = rng.integers(0, G - 5); keep[a:a + rng.integers(2, 6)] = False
        idx = np.flatnonzero(keep)
        p = gt_pos[t, idx] + rng.normal(0, jitter, (idx.size, 2))
        o = gt_ori[idx] + rng.normal(0, ori_noise, idx.size)
        sl = (gt_slen_px[t, idx] * PX) + rng.normal(0, slen_noise, idx.size)
        pos_all.append(p.astype(np.float32))
        ori_all.append(o.astype(np.float32))
        slen_all.append(sl.astype(np.float32))
        mid_all.append(idx.astype(np.int64))          # midline id ~ unused here
        detgt_all.append(idx.copy())                   # det j in frame t -> GT id idx[j]
    return pos_all, ori_all, slen_all, mid_all, detgt_all


def evaluate(res, detgt_all, G, T):
    snapped = res["tracks_snapped"]
    detid = res["tracks_detection_id"]      # (n,T) index into pos of that frame
    n = res["n_tracks"]
    out = {"n_tracks": int(n), "n_merges": int(res.get("n_merges", 0))}
    if n == 0:
        return out
    # per track: gather GT ids of its snaps
    purities = []
    frags = {}   # gt -> set of track rows (dominant)
    lengths = snapped.sum(axis=1)
    covered = np.zeros((T, G), bool)
    for i in range(n):
        fr = np.flatnonzero(snapped[i])
        if fr.size == 0:
            continue
        gids = []
        for t in fr:
            j = detid[i, t]
            if j >= 0 and j < len(detgt_all[t]):
                g = detgt_all[t][j]; gids.append(g); covered[t, g] = True
        if not gids:
            continue
        gids = np.array(gids)
        vals, cnts = np.unique(gids, return_counts=True)
        dom = vals[cnts.argmax()]
        purities.append(cnts.max() / gids.size)
        frags.setdefault(int(dom), []).append(i)
    purities = np.array(purities)
    # fragments per GT that is covered by >=1 track
    frag_counts = np.array([len(v) for v in frags.values()]) if frags else np.array([0])
    # GT detection coverage
    total_gt_dets = sum(len(d) for d in detgt_all)
    out.update({
        "n_gt": G,
        "len_mean": round(float(lengths.mean()), 2),
        "len_median": int(np.median(lengths)),
        "purity_mean": round(float(purities.mean()), 4),
        "n_impure(<0.95)": int((purities < 0.95).sum()),
        "n_swap(<0.80)": int((purities < 0.80).sum()),
        "gt_covered": int(len(frags)),
        "frags_per_gt_mean": round(float(frag_counts.mean()), 3),
        "frags_per_gt_max": int(frag_counts.max()),
        "det_coverage_pct": round(100.0 * covered.sum() / max(total_gt_dets, 1), 2),
    })
    return out


def run(overrides, scene, dets, seedlabel=""):
    gt_pos, gt_ori, gt_slen_px, flows, H, W, myo = scene
    pos_all, ori_all, slen_all, mid_all, detgt_all = dets
    T, G, _ = gt_pos.shape
    orig = stk.compute_flow_sequence
    stk.compute_flow_sequence = lambda *a, **k: flows
    try:
        # masks unused (flow monkeypatched) but shape needed for H,W,T
        zb = np.zeros((T, H, W), np.float32); mb = zb
        res = stk.track_sarcomere_vectors(
            zb, mb, pos_all, mid_all, slen_all, ori_all,
            pixelsize=PX, frametime=FT, compute_motion_field=False, **overrides)
    finally:
        stk.compute_flow_sequence = orig
    return evaluate(res, detgt_all, G, T)


CONFIGS = [
    ("LEGACY", {"reacquire_gap_cap": 1,
                "merge_min_bridge_snaps": 5}),
    ("DEFAULT", {}),
]

N_MYO, N_SARC, T, P_DROP, SEEDS = 10, 16, 120, 0.15, (1, 2, 3)


def main():
    scene = build_scene(T=T, n_myo=N_MYO, n_sarc=N_SARC, seed=0)
    G = scene[0].shape[1]
    # average key metrics over detection seeds (dropout noise)
    keys = ["n_tracks", "len_mean", "purity_mean", "n_swap(<0.80)",
            "frags_per_gt_mean", "det_coverage_pct", "n_merges"]
    print(f"\n# synthetic GT: T={T} G={G} n_myo={N_MYO} p_drop={P_DROP} seeds={SEEDS}")
    hdr = f"{'config':<14}" + "".join(f"{k:>18}" for k in keys)
    print(hdr); print("-" * len(hdr))
    for label, ov in CONFIGS:
        acc = {k: [] for k in keys}
        for sd in SEEDS:
            dets = make_detections(scene[0], scene[1], scene[2], p_drop=P_DROP, seed=sd)
            m = run(ov, scene, dets)
            for k in keys:
                acc[k].append(m.get(k, 0))
        line = f"{label:<14}"
        for k in keys:
            v = np.mean(acc[k])
            line += f"{round(float(v), 3):>18}"
        print(line)


if __name__ == "__main__":
    main()
