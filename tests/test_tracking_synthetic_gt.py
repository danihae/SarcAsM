# -*- coding: utf-8 -*-
"""Synthetic ground-truth regression tests for the 2D sarcomere tracker.

The tracker consumes only per-frame sarcomere vectors, so these scenes generate
ground-truth trajectories plus the detections they would produce, and score:

  - track PURITY            (fraction of a track's observations on its dominant GT id)
  - identity swaps          (tracks whose purity < 0.8)
  - fragments per GT track   (ideal 1)
  - GT detection coverage

Two regimes are covered. The SPARSE scenes (`_build_scene`, `_build_drift_scene`)
place one detection per sarcomere and stress the gates under dropout, drift and
coarse pixel size. The DENSE scene (`_build_row_scene`) reproduces what the
detector actually emits — ~1 px-spaced vectors along each M-band midline — which
is the regime that drives real fragmentation and to which the sparse scenes are
blind (they saturate at 1.000 fragments per GT sarcomere).

Fast (<1 s/test), deterministic (fixed RNG seeds), no models, no image data.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from sarcasm.analysis import sarcomere_tracking as stk

PX = 0.06117      # µm/px (matches the real high-speed dataset)
FT = 0.0164       # s/frame


# ---------------------------------------------------------------------------
# Synthetic ground-truth scene + detections
# ---------------------------------------------------------------------------

def _build_scene(T=80, n_myo=8, n_sarc=14, L0=30.0, seed=0):
    """GT sarcomere trajectories (the returned flow stack is unused by the
    tracker, which reads no image data; it is kept so the tuple shape is stable).

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
    flows = None          # the tracker reads no image data; kept for tuple shape

    def field(t):
        k = 0.015 * np.sin(2 * np.pi * t / period)
        sway = 1.2 * np.sin(2 * np.pi * t / (period * 1.3))
        return k, sway

    pos = centers0.copy()
    gt_pos[0] = pos
    for t in range(T - 1):
        k, sway = field(t)
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
    observed = res["tracks_observed"]
    detid = res["tracks_detection_id"]
    n = res["n_tracks"]
    if n == 0:
        return {"n_tracks": 0, "purity_mean": 0.0, "n_swap": 0,
                "frags_per_gt_mean": 0.0, "det_coverage_pct": 0.0}
    purities, frags = [], {}
    covered = np.zeros((T, G), bool)
    for i in range(n):
        fr = np.flatnonzero(observed[i])
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
    res = stk.track_sarcomere_vectors(
        pos_all, mid_all, slen_all, ori_all,
        pixelsize=PX, frametime=FT, **overrides)
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


def _build_drift_scene(T=120, n_myo=8, n_sarc=14, L0=30.0, seed=0, drift_px=0.9):
    """Like :func:`_build_scene` but with a strong coherent translation along the
    fibre axis. Over a dropout of g frames the tissue moves g*drift_px, which is
    what carries a neighbouring sarcomere into a stale anchor's gate — the
    identity-swap mechanism the neighbour advection removes."""
    rng = np.random.default_rng(seed)
    H = int(max(160, 2.2 * L0 + 20 * n_myo + 40))
    W = int(max(512, n_sarc * L0 + 220))
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
    gt_pos = np.zeros((T, G, 2), np.float64)
    flows = None          # the tracker reads no image data; kept for tuple shape
    pos = centers0.copy()
    gt_pos[0] = pos
    for t in range(T - 1):
        k = 0.015 * np.sin(2 * np.pi * t / 40.0)
        pos = pos + np.column_stack([np.full(G, 0.05 * drift_px),
                                     -k * (pos[:, 1] - x0) + drift_px])
        gt_pos[t + 1] = pos
    gt_slen_px = np.full((T, G), L0, np.float32)
    for t in range(T):
        gt_slen_px[t] = L0 * (1.0 - 0.6 * (0.015 * np.sin(2 * np.pi * t / 40.0)))
    return gt_pos, gt_ori, gt_slen_px, flows, H, W


def test_track_drift_flags_a_swapped_track():
    """compute_track_drift must single out a track that leaves its neighbourhood."""
    T, N = 60, 40
    rng = np.random.default_rng(0)
    pos = np.zeros((N, T, 2))
    pos[:, :, 0] = rng.uniform(0, 200, N)[:, None]
    pos[:, :, 1] = rng.uniform(0, 200, N)[:, None]
    # everyone translates coherently by 0.5 px/frame in x ...
    pos[:, :, 1] += 0.5 * np.arange(T)[None, :]
    # ... except track 0, which additionally walks 30 px away (~1 slen at L0=30)
    pos[0, :, 1] += np.linspace(0, 30, T)
    observed = np.ones((N, T), bool)
    drift = stk.compute_track_drift(pos, observed, median_slen_px=30.0, pixelsize=PX)
    assert np.isfinite(drift).all()
    assert drift[0] > 5 * np.median(drift[1:])       # the walker stands out
    assert drift[0] > 25 * PX                        # ~30 px of extra travel


def test_scale_invariant_no_neighbour_swaps_at_coarse_pixelsize():
    """Scale-invariance guard: with a ~14 px sarcomere spacing the raw 15 px
    along gate would exceed one sarcomere — the slen-relative gate caps must
    keep the match/merge from reaching a neighbour, so swaps stay rare."""
    scene = _build_scene(L0=14.0, seed=0)
    G = scene[0].shape[1]
    dets = _make_detections(scene[0], scene[1], scene[2], p_drop=0.15, seed=1)
    m = _track(scene, dets)
    assert m["purity_mean"] >= 0.95
    assert m["n_swap"] <= max(3, int(0.05 * G))
    assert m["det_coverage_pct"] > 95.0


def test_unmatched_track_advection_is_load_bearing():
    """Carrying an unmatched track with its neighbourhood is what keeps its anchor
    valid through a dropout. Disable it and tracks start departing from their
    neighbourhood, which is what an identity change looks like on real data
    (measured there: tracks drifting >1 sarcomere rise from 0.04 % to 1.65 %)."""
    scene = _build_drift_scene(seed=0, drift_px=0.9)
    dets = _make_detections(scene[0], scene[1], scene[2],
                            p_drop=0.40, burst=0.50, seed=1)
    pos_all, ori_all, slen_all, mid_all, _ = dets

    def _drift_p90():
        res = stk.track_sarcomere_vectors(
            pos_all, mid_all, slen_all, ori_all,
            pixelsize=PX, frametime=FT)
        d = np.asarray(res['track_drift_um'], float)
        d = d[np.isfinite(d)]
        return float(np.percentile(d, 90)) if d.size else 0.0

    with_advection = _drift_p90()
    orig = stk._neighbor_displacement
    stk._neighbor_displacement = lambda q, *a, **k: np.zeros((len(q), 2), np.float32)
    try:
        without = _drift_p90()
    finally:
        stk._neighbor_displacement = orig
    assert without > with_advection, (
        f'advection should reduce identity drift: {without:.4f} vs {with_advection:.4f}')


# ---------------------------------------------------------------------------
# Dense 1-px M-band rows — the regime the real data is actually in
# ---------------------------------------------------------------------------

def _build_row_scene(T=250, n_mid=12, extent_um=8.5, slen_um=1.715,
                     beat_period=100.0, contract_px=18.0, seed=0):
    """GT sites sampled ~1 px apart along each M-band midline.

    The sparse scenes above place ONE detection per sarcomere, which is not what
    the detector produces: sarcomere vectors are skeleton pixels of the M-band
    midlines, ~1 px apart, so one midline carries tens of them and the
    perpendicular match gate spans several lateral neighbours. Calibrated to the
    real 20 kPa movie: |along| motion per frame p50/p90/p99 = 0.010/0.057/0.111 µm,
    lateral p90 0.013 µm, in-frame nearest-neighbour distance exactly 1 px, and a
    ~24 % beat-locked swing in the number of detections.

    Returns ``(gt_pos, gt_ori, gt_slen_um, mid_of_site, phase, H, W)``.
    """
    rng = np.random.default_rng(seed)
    K = int(round(extent_um / PX))            # sites per midline, 1 px apart
    slen_px = slen_um / PX
    thetas = rng.uniform(-0.04, 0.04, n_mid)
    x0 = 60.0 + np.arange(n_mid) * slen_px
    y0 = np.full(n_mid, 20.0 + K / 2.0)
    W = int(x0[-1] + 80)
    H = int(K + 60)

    centres, gt_ori, mid_of_site = [], [], []
    for m in range(n_mid):
        th = thetas[m]
        perp = np.array([-np.cos(th), np.sin(th)])   # the midline runs perp to the axis
        off = (np.arange(K) - (K - 1) / 2.0)[:, None] * perp[None, :]
        centres.append(np.array([y0[m], x0[m]])[None, :] + off)
        gt_ori.append(np.full(K, th))
        mid_of_site.append(np.full(K, m))
    centres = np.concatenate(centres)
    gt_ori = np.concatenate(gt_ori).astype(np.float32)
    mid_of_site = np.concatenate(mid_of_site)
    G = len(centres)

    amp = contract_px * (0.6 + 0.4 * np.cos(
        np.pi * (x0 - x0.mean()) / (np.ptp(x0) + 1e-9)))
    phase = np.zeros(T)
    gt_pos = np.zeros((T, G, 2))
    gt_slen = np.zeros((T, G), np.float32)
    axis = np.column_stack([np.sin(gt_ori), np.cos(gt_ori)])
    for t in range(T):
        # short systole, long diastolic rest: most frames are nearly still, which
        # is what makes the median per-frame step sub-pixel as measured
        ph = (t % beat_period) / beat_period
        s = 0.5 * (1.0 - np.cos(2 * np.pi * ph / 0.25)) if ph < 0.25 else 0.0
        phase[t] = s
        gt_pos[t] = centres + (amp[mid_of_site] * s)[:, None] * axis
        gt_slen[t] = slen_um * (1.0 - 0.10 * s)
    return gt_pos, gt_ori, gt_slen, mid_of_site, phase, H, W


def _make_row_detections(scene, p_drop=0.05, end_walk=1.5, jitter_px=0.25, seed=1):
    """Detections from a row scene: beat-locked dropout + row-end random walk.

    Two properties matter and are easy to get wrong. (1) Detections are skeleton
    pixels, so they are unique within a frame and their lateral order is exact —
    jittering both coordinates and rounding would let neighbouring sites collide
    onto one pixel, which no skeleton produces. (2) The wobble is coherent along a
    row (an M-band is a connected ridge that shifts as a unit), not i.i.d. per
    site; i.i.d. noise makes neighbouring sites swap rank and destroys ground-truth
    identifiability.
    """
    gt_pos, gt_ori, gt_slen, mid_of_site, phase, H, W = scene
    rng = np.random.default_rng(seed)
    T, G, _ = gt_pos.shape
    n_mid = int(mid_of_site.max()) + 1
    lat = np.zeros(G, int)
    for m in range(n_mid):
        s = np.flatnonzero(mid_of_site == m)
        lat[s] = np.arange(len(s))
    K = lat.max() + 1
    max_trim = 0.15 * K
    axis = np.column_stack([np.sin(gt_ori), np.cos(gt_ori)])
    static = rng.normal(0, jitter_px, G)
    lo = np.zeros(n_mid)
    hi = np.zeros(n_mid)
    drop = np.zeros(0, bool)

    pos_all, ori_all, slen_all, mid_all, detgt_all = [], [], [], [], []
    for t in range(T):
        lo = np.clip(lo + rng.normal(0, end_walk, n_mid), 0, max_trim)
        hi = np.clip(hi + rng.normal(0, end_walk, n_mid), 0, max_trim)
        inside = (lat >= lo[mid_of_site]) & (lat < K - hi[mid_of_site])
        # dropout is beat-locked and persists through the contraction window
        p = p_drop * (0.5 + phase[t])
        if phase[t] > 0:
            drop = (rng.random(G) < p) if not len(drop) else (drop | (rng.random(G) < p * 0.15))
        else:
            drop = np.zeros(0, bool)
        lost = drop if len(drop) else (rng.random(G) < p_drop * 0.3)
        idx = np.flatnonzero(inside & ~lost)
        shared = rng.normal(0, jitter_px, n_mid)[mid_of_site]
        w = shared[idx] + static[idx] + rng.normal(0, 0.05, idx.size)
        p_xy = np.rint(gt_pos[t, idx] + w[:, None] * axis[idx])
        _, uniq = np.unique(p_xy, axis=0, return_index=True)
        if len(uniq) != len(idx):
            uniq = np.sort(uniq)
            idx, p_xy = idx[uniq], p_xy[uniq]
        pos_all.append(p_xy.astype(np.float32))
        ori_all.append(gt_ori[idx] + rng.normal(0, 0.03, idx.size).astype(np.float32))
        slen_all.append(gt_slen[t, idx] + rng.normal(0, 0.02, idx.size).astype(np.float32))
        mid_all.append(mid_of_site[idx].astype(np.int64))
        detgt_all.append(idx.copy())
    return pos_all, ori_all, slen_all, mid_all, detgt_all


def _track_rows(scene, dets, **overrides):
    pos_all, ori_all, slen_all, mid_all, detgt_all = dets
    return stk.track_sarcomere_vectors(
        pos_all, mid_all, slen_all, ori_all,
        pixelsize=PX, frametime=0.01, **overrides)


def test_dense_row_scene_matches_the_real_detection_statistics():
    """Guard on the scene itself: if it does not reproduce the real regime, no
    conclusion drawn from it counts. The three properties that make the assignment
    hard are 1-px nearest-neighbour spacing, unique pixels, and a beat-locked
    swing in the number of detections."""
    scene = _build_row_scene(seed=0)
    pos_all, *_ = _make_row_detections(scene, seed=1)
    counts = np.array([len(p) for p in pos_all])
    swing = (counts.max() - counts.min()) / np.median(counts)
    assert 0.15 < swing < 0.45, f'population swing {swing:.2f}, real is ~0.24'
    for t in (0, len(pos_all) // 2):
        d, _ = cKDTree(pos_all[t]).query(pos_all[t], k=2)
        assert np.isclose(np.median(d[:, 1]), 1.0), 'lateral sampling must be 1 px'
        assert len(np.unique(pos_all[t], axis=0)) == len(pos_all[t]), (
            'skeleton pixels must be unique within a frame')


def test_dense_rows_are_tracked_without_fragmenting():
    """On the dense-row regime the tracker must keep close to one track per GT
    site, stay on the correct M-band, and claim essentially every detection."""
    scene = _build_row_scene(seed=0)
    dets = _make_row_detections(scene, seed=1)
    G = scene[0].shape[1]
    T = scene[0].shape[0]
    res = _track_rows(scene, dets)
    m = _evaluate(res, dets[4], G, T)

    assert m['frags_per_gt_mean'] <= 1.3, m
    assert m['det_coverage_pct'] >= 98.0, m
    # one track per GT site, within 10 %
    assert 0.9 * G <= res['n_tracks'] <= 1.1 * G, (res['n_tracks'], G)
    assert res['fragmentation_ratio'] <= 1.35

    # Identity: a track may slide by a lateral sample or two (the sites are only
    # 1 px = 0.06 µm apart, so that is sub-resolution), but it must never move to
    # another M-band, which would be a real swap onto a different myofibril.
    mid_of_site = scene[3]
    observed = res['tracks_observed']
    detid = res['tracks_detection_id']
    crossed = 0
    scored = 0
    for i in range(res['n_tracks']):
        mids = [mid_of_site[dets[4][t][detid[i, t]]]
                for t in np.flatnonzero(observed[i]) if 0 <= detid[i, t] < len(dets[4][t])]
        if len(mids) > 1:
            scored += 1
            crossed += len(set(mids)) > 1
    frac = crossed / max(scored, 1)
    assert frac <= 0.005, f'{crossed}/{scored} tracks crossed to another M-band'
