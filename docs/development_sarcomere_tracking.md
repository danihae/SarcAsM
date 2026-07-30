# 2D full-field sarcomere tracking — neighbour-predict + optimal detection assignment

Location: [`sarcasm/analysis/sarcomere_tracking.py`](../sarcasm/analysis/sarcomere_tracking.py)

User-facing wrappers on [`Structure`](../sarcasm/structure.py):

- `SarcAsM.track_sarcomere_vectors(...)` — the whole tracker

## What it does, and how it differs from the LOI tracker

The LOI tracker ([`sarcasm.motion`](../sarcasm/motion.py)) follows Z-band peaks along a user-drawn 1D line. This module does the same job in 2D across the whole image, automatically, without user input. The two coexist:

| | LOI (1D) | 2D full-field |
|---|---|---|
| Input | Manual line | Automatic |
| Output | Per-sarcomere length-vs-time on one fiber | Full dense `(n_tracks, T)` arrays across all sarcomeres |
| Strength | High SNR on a chosen fiber | Spatial heterogeneity across the whole cell |

## Model — sarcomeres as samples of a field

The key reframing: individual sarcomere detections are **samples of an underlying continuous sarcomere-state field**, discretised only because the segmentation masks live on a pixel grid. We do not track identity-bearing particles; we track **query points** that sample that field.

Each query point:

1. Represents one sarcomere vector, seeded from a detection in frame 0 (or whenever a fresh detection appears that no existing query point matched).
2. A query point that matched last frame keeps its own fresh position. One that did *not* is advected by the **local coherent motion of its neighbours** (median step of the nearby tracks that matched in both of the last two frames), projected onto its own sarcomere axis. No optical flow is involved — the tracker reads no image data.
3. Its candidate detections are those passing an **anisotropic position gate** (`max_disp_along_um` along the sarcomere axis, `max_disp_perp_um` perpendicular; motion along the axis is contraction and gets a generous budget, perpendicular is kept tight) and an **orientation gate** (`ori_tol_deg`, modulo π since sarcomeres are undirected).
4. **Optimal assignment.** The candidate pairs form a bipartite graph; each connected component is solved exactly — minimum-cost, maximum-cardinality — with the gate-normalised cost `along²/along_budget + perp²/perp_budget`. Each detection is matched at most once, which is the anti-convergence mechanism.
5. If nothing consistent is available, the query point records an **honest gap frame**: its predicted position, `snapped=False`, and NaN `slen`/`orientation` (a length is never fabricated). It keeps its identity and re-enters the assignment later, so a dropout of any length no longer ends the trajectory; by default tracks never retire.
6. **Unmatched detections spawn new query points** (appearance).

No M-band identity, no arc-position, no slots, no post-hoc fragment stitching.

## Why this design

- **Anti-convergence by construction.** Each detection is matched at most once, so neighbouring query points stay anchored to *different* detections and cannot collapse onto each other.
- **The assignment must be joint, not greedy.** Sarcomere vectors are a ~1 px sampling along each M-band midline, so one midline carries tens of them and the perpendicular gate spans several lateral neighbours — the graph components are effectively the midline rows. Ranking candidates by raw Euclidean distance would let a lateral neighbour 1 px away outrank the correct detection 2 px along the axis, and a one-sided greedy claim orphans a track whenever a row shifts, which then spawns a duplicate. Solving each row jointly removes both effects (measured: fragmentation 2.1–3.4 → 1.2–1.4 across three movies).
- **Identity is decoupled from detection continuity.** A material sarcomere exists whether or not the U-Net fires on it in a given frame, so an unmatched track waits instead of dying. This is what the removed `memory` / re-acquisition / merge machinery was approximating.
- **Physical plausibility.** The anisotropic + orientation guards prevent the "phantom jump" failure mode (multi-µm per-frame displacements from cross-M-band mismatches) that an isotropic position threshold couldn't catch.

## Key parameters

```python
sarc.track_sarcomere_vectors(
    frames='all',
    max_disp_along_um=1.0,     # sarcomere-axis snap gate (µm)
    max_disp_perp_um=0.2,      # perpendicular snap gate (µm)
    ori_tol_deg=45.0,          # orientation tolerance (wraps modulo π)
    retire_after_s=None,       # None = tracks never retire (identity survives any gap)
    min_track_duration_s=0.08, # min accumulated real observation time to keep a track
)
```

All gates are physical (µm / seconds / degrees), so the same defaults hold across
pixel sizes and frame rates without retuning.

Candidate matches are resolved by an exact minimum-cost, maximum-cardinality
assignment per connected component of the gated candidate graph, using the
gate-normalised anisotropic cost `along²/along_budget + perp²/perp_budget`. This
matters because sarcomere vectors are a ~1 px sampling along each M-band midline:
the perpendicular gate spans several lateral neighbours, so the components are
effectively the midline rows, and ranking by raw Euclidean distance (or claiming
greedily) would reshuffle a row whenever it shifts. An unmatched track records an
honest gap frame and keeps its identity, which is why no post-hoc fragment
stitching is needed.

## Outputs in `self.data`

Dense arrays shape `(n_tracks, T)` or `(n_tracks, T, 2)`:

- `tracks_positions_px`, `tracks_positions_um` — pixel / µm positions. NaN before the track's `start_frame` and after close.
- `tracks_slen`, `tracks_orientations` — NaN additionally on gap frames (no snap).
- `tracks_snapped` — bool mask: True where a real detection was snapped, False on predicted-position gap frames.
- `track_ids`, `track_start_frame`, `track_lengths`.

Quality scalars:

- `fragmentation_ratio` — tracks per median detections-per-frame. **Ideal 1.0**; the headline continuity number.
- `track_drift_um` — per-track departure from the coherent motion of its neighbours. A track drifting ~one sarcomere length has almost certainly changed identity; `group_tracks` drops those from chain groupings by default.
- `n_tracks_retired` — 0 unless `retire_after_s` is set.

Parameters prefixed `params.track_sarcomere_vectors.*`.

## Performance (500 frames, 200×1024, ~2300–4200 vectors/frame)

- Detection + vector analysis: ~7 min
- Tracker: **~8–14 s** (no image data is read; cost is the per-frame KD-tree query plus the per-component assignment)

## Measured quality (three 500-frame movies, identical detection settings)

| | 10 kPa | 20 kPa | 30 kPa |
|---|---|---|---|
| `fragmentation_ratio` | 1.27 | 1.23 | 1.38 |
| median real snaps / T | 0.95 | 0.97 | 0.86 |
| detection coverage | 99.99 % | 99.99 % | 99.98 % |
| `track_drift_um` p90 | 0.15 | 0.12 | 0.16 |

Mean fill (snaps per track span) stays ~0.72–0.81: the detector misses each vector
~15 % of frames, phase-locked to contraction. That is a **detection** ceiling, not a
tracking one — lifting it needs a temporal M-band model, the analogue of
`detect_z_bands_fast_movie` for Z-bands.

## Tests

- [`tests/test_sarcomere_tracking.py`](../tests/test_sarcomere_tracking.py) — unit + behavioural: anti-convergence, gap-frame NaN slen, frames-after-final-snap blanking, a gap never widening the gate, unmatched-track advection staying on-axis, the scale-aware gate cap, and the decisive `test_optimal_assignment_handles_a_shifted_1px_row` (a 1-px-spaced row shifted by 1 px with one sample lost and one gained — greedy-by-Euclidean fails it).
- [`tests/test_tracking_synthetic_gt.py`](../tests/test_tracking_synthetic_gt.py) — ground-truth scenes: sparse (dropout, drift, coarse pixel size) and **dense 1-px rows** calibrated to the real detection statistics, with a guard test asserting the scene actually reproduces them.

## Related

- LOI tracker: [`sarcasm/motion.py`](../sarcasm/motion.py) — also received an opt-in topological-ordering constraint (`enforce_topological_order=True`) in `Motion._track_z_bands_lap`.
- Contraction dynamics: [`sarcasm/analysis/contraction_analysis.py`](../sarcasm/analysis/contraction_analysis.py) — consumes `tracks_slen` (sarcomere length and its rate of change), which is the meaningful motion readout.
