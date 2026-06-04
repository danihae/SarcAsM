# -*- coding: utf-8 -*-
"""Synthetic ground-truth regression tests for the 2D sarcomere tracker.

The optical-flow engine is bypassed by feeding ``track_sarcomere_vectors`` an
ANALYTIC dense flow field that is exactly consistent with the ground-truth
motion. This isolates the snap / merge / assignment logic (where identity swaps
happen) under perfect prediction, with full ground truth, so we can assert on:

  - track PURITY            (fraction of a track's snaps on its dominant GT id)
  - identity swaps          (tracks whose purity < 0.8)
  - fragments per GT track   (ideal 1)
  - GT detection coverage

Detection dropout is the dominant fragmentation driver; jitter + burst dropouts
stress the identity logic. A coarse-spacing scene additionally guards the
scale-invariant gate caps (the along gate must never reach a neighbouring
sarcomere even when the sarcomere is only ~14 px).

Fast (<1 s/test), deterministic (fixed RNG seeds), and dependency-light: no
ContractionNet / no cv2 Farneback (flow is monkeypatched).
"""
from __future__ import annotations

import numpy as np
import pytest

from sarcasm.analysis import sarcomere_tracking as stk

PX = 0.06117      # µm/px (matches the real high-speed dataset)
FT = 0.0164       # s/frame


# ---------------------------------------------------------------------------
# Synthetic ground-truth scene + detections
# ---------------------------------------------------------------------------

def _build_scene(T=80, n_myo=8, n_sarc=14, L0=30.0, seed=0):
    """GT sarcomere trajectories + a consistent analytic flow stack.

    Myofibrils are near-parallel and well separated so no two sarcomeres ever
    coincide (a clean, separable ground truth). ``L0`` is the rest sarcomere
    spacing in px; small ``L0`` emulates coarse pixel size for the scale test.
    """
    rng = np.random.default_rng(seed)
    H = int(max(160, 2.2 * L0 * 1 + 20 * n_myo + 40))
    W = int(max(512, n_sarc * L0 + 120))
    x0 = W * 0.5
    thetas = rng.uniform(-0.03, 0.03, n_myo).astype(np.float32)
    rows = np.linspace(20, H - 20, n_myo) + rng.uniform(-2, 2, n_myo)
    centers0, gt_ori = [], []
    for m in range(n_myo):
        th = thetas[m]; s, c = np.sin(th), np.cos(th)
        offs = (np.arange(n_sarc) - (n_sarc - 1) / 2.0) * L0
        cy = rows[m] + offs * s
        cx = x0 + offs * c
        for k in range(n_sarc):
            centers0.append((cy[k], cx[k])); gt_ori.append(th)
    centers0 = np.array(centers0, np.float64)
    gt_ori = np.array(gt_ori, np.float32)
    G = len(centers0)

    period = 40.0
    drift = rng.uniform(-0.3, 0.3, 2)
    gt_pos = np.zeros((T, G, 2), np.float64)
    flows = np.zeros((T - 1, H, W, 2), np.float32)
    _, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def field(t):
        k = 0.015 * np.sin(2 * np.pi * t / period)
        sway = 1.2 * np.sin(2 * np.pi * t / (period * 1.3))
        return k, sway

    pos = centers0.copy()
    gt_pos[0] = pos
    for t in range(T - 1):
        k, sway = field(t)
        flows[t, ..., 0] = np.full_like(xx, sway * 0.04) + drift[0]
        flows[t, ..., 1] = -k * (xx - x0) + drift[1]
        d_x = -k * (pos[:, 1] - x0) + drift[1]
        d_y = np.full(G, sway * 0.04) + drift[0]
        pos = pos + np.column_stack([d_y, d_x])
        gt_pos[t + 1] = pos

    gt_slen_px = np.full((T, G), L0, np.float32)
    for t in range(T):
        k, _ = field(t)
        gt_slen_px[t] = L0 * (1.0 - 0.6 * k)
    return gt_pos, gt_ori, gt_slen_px, flows, H, W


def _make_detections(gt_pos, gt_ori, gt_slen_px, p_drop=0.15, burst=0.04,
                     jitter=0.6, ori_noise=0.05, slen_noise=0.03, seed=1):
    """Per-frame detections from GT with dropout + jitter; also returns the
    per-frame detection->GT-id map used to score purity/coverage."""
    rng = np.random.default_rng(seed)
    T, G, _ = gt_pos.shape
    pos_all, ori_all, slen_all, mid_all, detgt_all = [], [], [], [], []
    for t in range(T):
        keep = rng.random(G) > p_drop
        if rng.random() < burst:
            a = rng.integers(0, G - 5); keep[a:a + rng.integers(2, 6)] = False
        idx = np.flatnonzero(keep)
        p = gt_pos[t, idx] + rng.normal(0, jitter, (idx.size, 2))
        o = gt_ori[idx] + rng.normal(0, ori_noise, idx.size)
        sl = (gt_slen_px[t, idx] * PX) + rng.normal(0, slen_noise, idx.size)
        pos_all.append(p.astype(np.float32))
        ori_all.append(o.astype(np.float32))
        slen_all.append(sl.astype(np.float32))
        mid_all.append(idx.astype(np.int64))
        detgt_all.append(idx.copy())          # det j in frame t -> GT id idx[j]
    return pos_all, ori_all, slen_all, mid_all, detgt_all


def _evaluate(res, detgt_all, G, T):
    snapped = res["tracks_snapped"]
    detid = res["tracks_detection_id"]
    n = res["n_tracks"]
    if n == 0:
        return {"n_tracks": 0, "purity_mean": 0.0, "n_swap": 0,
                "frags_per_gt_mean": 0.0, "det_coverage_pct": 0.0}
    purities, frags = [], {}
    covered = np.zeros((T, G), bool)
    for i in range(n):
        fr = np.flatnonzero(snapped[i])
        gids = []
        for t in fr:
            j = detid[i, t]
            if 0 <= j < len(detgt_all[t]):
                g = detgt_all[t][j]; gids.append(g); covered[t, g] = True
        if not gids:
            continue
        gids = np.array(gids)
        vals, cnts = np.unique(gids, return_counts=True)
        purities.append(cnts.max() / gids.size)
        frags.setdefault(int(vals[cnts.argmax()]), []).append(i)
    purities = np.array(purities)
    frag_counts = np.array([len(v) for v in frags.values()]) if frags else np.array([0])
    total_gt = sum(len(d) for d in detgt_all)
    return {
        "n_tracks": int(n),
        "purity_mean": float(purities.mean()),
        "n_swap": int((purities < 0.80).sum()),
        "frags_per_gt_mean": float(frag_counts.mean()),
        "det_coverage_pct": 100.0 * covered.sum() / max(total_gt, 1),
    }


def _track(scene, dets, **overrides):
    gt_pos, gt_ori, gt_slen_px, flows, H, W = scene
    pos_all, ori_all, slen_all, mid_all, detgt_all = dets
    T, G, _ = gt_pos.shape
    orig = stk.compute_flow_sequence
    stk.compute_flow_sequence = lambda *a, **k: flows
    try:
        zb = np.zeros((T, H, W), np.float32)
        res = stk.track_sarcomere_vectors(
            zb, zb, pos_all, mid_all, slen_all, ori_all,
            pixelsize=PX, frametime=FT, compute_motion_field=False, **overrides)
    finally:
        stk.compute_flow_sequence = orig
    return _evaluate(res, detgt_all, G, T)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_continuity_and_purity():
    """Default tracker: high purity, few swaps, low fragmentation, full coverage
    on a normal-scale dropout scene."""
    scene = _build_scene(seed=0)
    G = scene[0].shape[1]
    dets = _make_detections(scene[0], scene[1], scene[2], p_drop=0.15, seed=1)
    m = _track(scene, dets)
    assert m["purity_mean"] >= 0.97
    assert m["n_swap"] <= max(3, int(0.04 * G))      # identity swaps stay rare
    assert m["frags_per_gt_mean"] < 2.0              # not heavily fragmented
    assert m["det_coverage_pct"] > 95.0              # almost all detections kept


def test_default_beats_legacy_on_fragmentation():
    """The continuity features (re-acquisition + short-fragment merge bridges)
    reduce fragmentation vs. the legacy gates, without raising swaps."""
    scene = _build_scene(seed=0)
    dets = _make_detections(scene[0], scene[1], scene[2], p_drop=0.15, seed=1)
    legacy = _track(scene, dets, reacquire_gap_cap=1, merge_min_bridge_snaps=5)
    default = _track(scene, dets)
    assert default["frags_per_gt_mean"] <= legacy["frags_per_gt_mean"] + 1e-6
    assert default["n_swap"] <= legacy["n_swap"] + 1   # no purity regression


def test_scale_invariant_no_neighbour_swaps_at_coarse_pixelsize():
    """Scale-invariance guard: with a ~14 px sarcomere spacing the raw 15 px
    along gate would exceed one sarcomere — the slen-relative gate caps must
    keep the snap/merge from reaching a neighbour, so swaps stay rare."""
    scene = _build_scene(L0=14.0, seed=0)
    G = scene[0].shape[1]
    dets = _make_detections(scene[0], scene[1], scene[2], p_drop=0.15, seed=1)
    m = _track(scene, dets)
    assert m["purity_mean"] >= 0.95
    assert m["n_swap"] <= max(3, int(0.05 * G))
    assert m["det_coverage_pct"] > 95.0
