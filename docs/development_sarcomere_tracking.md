# 2D full-field sarcomere tracking — flow-predict + detection-snap

Location: [`sarcasm/structure_modules/sarcomere_tracking.py`](../sarcasm/structure_modules/sarcomere_tracking.py)

User-facing wrappers on [`Structure`](../sarcasm/structure.py):

- `Structure.track_sarcomere_vectors(...)` — full pipeline
- `Structure.compute_motion_field(...)` — flow + sampling without tracking

## What it does, and how it differs from the LOI tracker

The LOI tracker ([`sarcasm.motion`](../sarcasm/motion.py)) follows Z-band peaks along a user-drawn 1D line. This module does the same job in 2D across the whole image, automatically, without user input. The two coexist:

| | LOI (1D) | 2D full-field |
|---|---|---|
| Input | Manual line | Automatic |
| Output | Per-sarcomere length-vs-time on one fiber | Full dense `(n_tracks, T)` arrays across all sarcomeres |
| Strength | High SNR on a chosen fiber | Spatial heterogeneity, strain maps |

## Model — sarcomeres as samples of a field

The key reframing: individual sarcomere detections are **samples of an underlying continuous sarcomere-state field**, discretised only because the segmentation masks live on a pixel grid. We do not track identity-bearing particles; we track **query points** that sample that field.

Each query point:

1. Represents one sarcomere, seeded from a detection in frame 0 (or whenever a fresh detection appears that isn't claimed by an existing query point).
2. Each subsequent frame is **flow-advected** (Lagrangian prediction via bilinear-interpolated Farneback flow).
3. At every frame the query point tries to **snap** to the nearest sarcomere detection that is *consistent* with its prediction under:
   - An **anisotropic position gate** (`max_disp_along_px` along the sarcomere axis, `max_disp_perp_px` perpendicular). Motion along the sarcomere axis (contraction) is allowed generously; motion perpendicular is kept tight.
   - An **orientation gate** (`ori_tol_deg`, compared modulo π since sarcomeres are undirected).
4. **Hard assignment.** Each detection is claimed by at most one query point per frame — greedy by ascending distance over all passing (qp, det) pairs. This is the anti-convergence mechanism: two query points cannot collapse onto the same detection and merge.
5. If no consistent unclaimed detection is found, the query point keeps its flow-predicted position and records NaN for `slen`/`orientation` that frame. After `memory` frames without a snap the query point is closed.
6. **Unclaimed detections spawn new query points** (appearance).

No M-band identity, no arc-position, no slots — just per-sarcomere flow prediction + hard-assigned detection snap.

## Why this design

- **Anti-convergence by construction.** Detections sit at physical sarcomere centres (~1 sarcomere ≈ 18 px apart). Hard assignment means neighbouring query points stay anchored to *different* detections and cannot drift onto each other even if flow is locally uniform.
- **Coverage decouples from M-band topology.** Previously an M-band fragmentation killed all vector tracks on it. Here the only thing that matters is whether *some* consistent detection lies near the query point's prediction — a much weaker condition.
- **Physical plausibility.** The anisotropic + orientation guards prevent the "phantom jump" failure mode (multi-µm per-frame displacements from cross-M-band mismatches) that an isotropic position threshold couldn't catch.

## Key parameters

```python
sarc.track_sarcomere_vectors(
    frames='all',
    threshold_mbands=0.25, threshold_zbands=0.5, dt_clip=20.0,
    max_disp_along_px=15.0,   # sarcomere-axis motion tolerance
    max_disp_perp_px=6.0,     # perpendicular motion tolerance
    ori_tol_deg=45.0,         # orientation tolerance (wraps modulo π)
    memory=5,                 # max gap frames before a query point closes
    min_track_length=5,       # min actual snaps to keep a track
    max_gap_interpolation=5,  # max NaN run post-hoc interpolatable
    compute_motion_field=True,
    store_flow_fields=False,
)
```

## Outputs in `self.data`

Dense arrays shape `(n_tracks, T)` or `(n_tracks, T, 2)`:

- `tracks_positions_px`, `tracks_positions_um` — pixel / µm positions. NaN before the track's `start_frame` and after close.
- `tracks_slen`, `tracks_orientations` — NaN additionally on gap frames (no snap).
- `tracks_snapped` — bool mask: True where a real detection was snapped, False on predicted-position gap frames.
- `track_ids`, `track_start_frame`, `track_lengths`.

Motion-field outputs (unchanged semantics):

- `flow_at_vectors` — per-frame displacement sampled at detection positions (µm).
- `displacement_magnitude`, `displacement_along_sarcomere`, `displacement_perpendicular`, `velocity_magnitude`.

Parameters prefixed `params.track_sarcomere_vectors.*`.

## Performance (100 frames, 200×1024, ~4400 vectors/frame)

- Detection + vector analysis: ~60 s
- Tracker: ~90–180 s (dominated by per-frame KD-tree radius queries)
- ≥ 95 % of frame-0 seeds are continuously tracked through frame 99.
- Coverage over 80 % across all tracks: see test run in `tmp_track_plots/`.

## Unit tests

[`tests/test_sarcomere_tracking.py`](../tests/test_sarcomere_tracking.py) — 11 tests, all pass. Cover:

- Flow engine: DT channel shape, zero-flow on identical frames.
- Bilinear sampling correctness.
- Anisotropic decomposition (along/perp).
- Snap gate: rejects perpendicular outliers, rejects bad orientation.
- End-to-end: recovers stationary detections with 100 % snap rate.
- **Anti-convergence test**: two query points 40 px apart stay ≥ 30 px apart across a synthetic sequence — directly tests the anti-collapse property.
- Gap frame: NaN slen recorded, position kept via flow prediction.

## Related

- LOI tracker: [`sarcasm/motion.py`](../sarcasm/motion.py) — also received an opt-in topological-ordering constraint (`enforce_topological_order=True`) in `Motion._track_z_bands_lap`.
- Domain motion: [`sarcasm/structure_modules/domain_motion.py`](../sarcasm/structure_modules/domain_motion.py) — can consume `displacement_along_sarcomere` from the new tracker's output for per-domain aggregates.
