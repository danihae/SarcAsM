# Result key migration (1.0)

In 1.0 a result key **is its path**: `sarc.data['motion.pool.slen_avg']` and
`sarc.data.motion.pool.slen_avg` are the same value. Keys live in three namespaces —
`structure` (per-frame / single-frame morphology), `motion` (everything derived from
the sarcomere tracks) and `params` (what each analysis step ran with).

There is **no runtime translation layer** and no store migration: an analysis written
before 1.0 holds the old keys and must be regenerated. Exported CSV / DataFrame column
names change with the keys, so downstream analyses need re-pointing against this table.

A parameter key is the **call path** of the step that wrote it plus the parameter's name in the
method signature: `params.detect_sarcomeres.<name>` for the recording's own step,
`params.structure.<step>.<name>` and `params.motion.<step>.<name>` for the analyses, and
`params.motion.group_tracks.<kind>.<name>` / `params.motion.analyze.<kind>.<name>` for the grouping
of each kind and its analysis. For example `params.analyze_sarcomere_vectors.frames` (1.0.0b4) is
`params.structure.analyze_vectors.frames`, and `params.analyze_track_motion.reading` is
`params.motion.analyze.<kind>.reading`. Four names had drifted from the signature:

| old | new |
|---|---|
| `params.detect_sarcomeres.model` | `params.detect_sarcomeres.model_path` |
| `params.detect_sarcomeres.clip_threshold` | `params.detect_sarcomeres.clip_thres` |
| `params.detect_z_bands_fast_movie.model` | `params.motion.detect_z_bands_fast_movie.model_path` |
| `params.detect_z_bands_fast_movie.clip_threshold` | `params.motion.detect_z_bands_fast_movie.clip_thres` |

## `StructureAnalysis.analyze_cell_mask`

| old | new |
|---|---|
| `cell_mask_area` | `structure.cell.mask_area` |
| `cell_mask_area_ratio` | `structure.cell.mask_area_ratio` |
| `cell_mask_intensity` | `structure.cell.mask_intensity` |

## `StructureAnalysis.analyze_z_bands`

The `z_` prefix is replaced by the group name.

Not computed any more in 1.0 (the rows below keep their names for renaming 0.5 tables):
`structure.zbands.straightness` (+ `_mean`, `_std`), which measured Z-band length rather than
curvature (area / convex-hull area falls with length), and `structure.zbands.lat_size_groups`
(+ `_mean`, `_std`), which duplicates the lateral neighbours: the mean group size is
`2 / (2 - structure.zbands.lat_neighbors_mean)` and the size of each group is the length of its
entry in `structure.zbands.lat_groups`. The same holds for `structure.myofibril.straightness`
(+ `_mean`, `_std`) below; `structure.myofibril.bending` stays.

| old | new |
|---|---|
| `n_zbands` | `structure.zbands.n` |
| `z_ends` | `structure.zbands.ends` |
| `z_intensity` | `structure.zbands.intensity` |
| `z_intensity_mean` | `structure.zbands.intensity_mean` |
| `z_intensity_std` | `structure.zbands.intensity_std` |
| `z_labels` | `structure.zbands.labels` |
| `z_lat_alignment` | — (removed in 1.0, see `structure.sarcomere.registry_*`) |
| `z_lat_alignment_groups` | — (removed in 1.0, see `structure.sarcomere.registry_*`) |
| `z_lat_alignment_groups_mean` | — (removed in 1.0, see `structure.sarcomere.registry_*`) |
| `z_lat_alignment_groups_std` | — (removed in 1.0, see `structure.sarcomere.registry_*`) |
| `z_lat_alignment_mean` | — (removed in 1.0, see `structure.sarcomere.registry_*`) |
| `z_lat_alignment_std` | — (removed in 1.0, see `structure.sarcomere.registry_*`) |
| `z_lat_dist` | `structure.zbands.lat_dist` |
| `z_lat_dist_mean` | `structure.zbands.lat_dist_mean` |
| `z_lat_dist_std` | `structure.zbands.lat_dist_std` |
| `z_lat_groups` | `structure.zbands.lat_groups` |
| `z_lat_length_groups` | `structure.zbands.lat_length_groups` |
| `z_lat_length_groups_mean` | `structure.zbands.lat_length_groups_mean` |
| `z_lat_length_groups_std` | `structure.zbands.lat_length_groups_std` |
| `z_lat_links` | `structure.zbands.lat_links` |
| `z_lat_neighbors` | `structure.zbands.lat_neighbors` |
| `z_lat_neighbors_mean` | `structure.zbands.lat_neighbors_mean` |
| `z_lat_neighbors_std` | `structure.zbands.lat_neighbors_std` |
| `z_lat_size_groups` | `structure.zbands.lat_size_groups` |
| `z_lat_size_groups_mean` | `structure.zbands.lat_size_groups_mean` |
| `z_lat_size_groups_std` | `structure.zbands.lat_size_groups_std` |
| `z_length` | `structure.zbands.length` |
| `z_length_max` | `structure.zbands.length_max` |
| `z_length_mean` | `structure.zbands.length_mean` |
| `z_length_std` | `structure.zbands.length_std` |
| `z_mask_area` | `structure.zbands.mask_area` |
| `z_mask_area_ratio` | `structure.zbands.mask_area_ratio` |
| `z_mask_intensity` | `structure.zbands.mask_intensity` |
| `z_oop` | `structure.zbands.oop` |
| `z_orientation` | `structure.zbands.orientation` |
| `z_straightness` | `structure.zbands.straightness` |
| `z_straightness_mean` | `structure.zbands.straightness_mean` |
| `z_straightness_std` | `structure.zbands.straightness_std` |

## `StructureAnalysis.analyze_vectors`

The per-vector arrays and their per-frame summaries now live in one group, so `slen` (per vector) sits next to `slen_mean` (per frame). The `_vectors` suffix is dropped, and sarcomere length is `slen` everywhere — `length` is reserved for the length of the object a group is about (`structure.zbands.length`, `structure.myofibril.length`).

| old | new |
|---|---|
| `midline_id_vectors` | `structure.sarcomere.midline_id` |
| `midline_length_vectors` | `structure.sarcomere.midline_length` |
| `n_mbands` | `structure.sarcomere.n_mbands` |
| `n_vectors` | `structure.sarcomere.n_vectors` |
| `pos_vectors` | `structure.sarcomere.pos` |
| `pos_vectors_px` | `structure.sarcomere.pos_px` |
| `sarcomere_area` | `structure.sarcomere.area` |
| `sarcomere_area_ratio` | `structure.sarcomere.area_ratio` |
| `sarcomere_length_mean` | `structure.sarcomere.slen_mean` |
| `sarcomere_length_std` | `structure.sarcomere.slen_std` |
| `sarcomere_length_vectors` | `structure.sarcomere.slen` |
| `sarcomere_oop` | `structure.sarcomere.oop` |
| `sarcomere_orientation_mean` | `structure.sarcomere.orientation_mean` |
| `sarcomere_orientation_std` | `structure.sarcomere.orientation_std` |
| `sarcomere_orientation_vectors` | `structure.sarcomere.orientation` |
| — (new) | `structure.sarcomere.registry_lateral`, `structure.sarcomere.registry_axial` (per vector) |
| — (new) | `structure.sarcomere.registry_lateral_mean`, `structure.sarcomere.registry_axial_mean`, `structure.sarcomere.registry_mean` (per frame) |

## `StructureAnalysis.analyze_myofibrils`

| old | new |
|---|---|
| `myof_bending` | `structure.myofibril.bending` |
| `myof_bending_mean` | `structure.myofibril.bending_mean` |
| `myof_bending_std` | `structure.myofibril.bending_std` |
| `myof_length` | `structure.myofibril.length` |
| `myof_length_max` | `structure.myofibril.length_max` |
| `myof_length_mean` | `structure.myofibril.length_mean` |
| `myof_length_std` | `structure.myofibril.length_std` |
| `myof_lines` | `structure.myofibril.lines` |
| `myof_straightness` | `structure.myofibril.straightness` |
| `myof_straightness_mean` | `structure.myofibril.straightness_mean` |
| `myof_straightness_std` | `structure.myofibril.straightness_std` |

## `StructureAnalysis.analyze_domains`

| old | new |
|---|---|
| `domain_area` | `structure.domain.area` |
| `domain_area_mean` | `structure.domain.area_mean` |
| `domain_area_std` | `structure.domain.area_std` |
| `domain_mask` | `structure.domain.mask` |
| `domain_oop` | `structure.domain.oop` |
| `domain_oop_mean` | `structure.domain.oop_mean` |
| `domain_oop_std` | `structure.domain.oop_std` |
| `domain_orientation` | `structure.domain.orientation` |
| `domain_slen` | `structure.domain.slen` |
| `domain_slen_mean` | `structure.domain.slen_mean` |
| `domain_slen_std` | `structure.domain.slen_std` |
| `domains` | `structure.domain.members` |
| `n_domains` | `structure.domain.n` |
| — (new) | `structure.domain.registry_lateral`, `structure.domain.registry_axial` (per domain) |
| — (new) | `structure.domain.registry_lateral_mean`, `structure.domain.registry_axial_mean` (per frame) |

## `MotionAnalysis.detect_lois`

| old | new |
|---|---|
| `loi_data` (dict) | `motion.lois.lines` (px), `motion.lois.lengths` (µm), `motion.lois.index_lines` |

The grown candidate lines, their Hausdorff distances and clusters are no longer stored.

## `MotionAnalysis.track`

`track_lengths` becomes `n_frames` — it is a duration in frames, not a length.

| old | new |
|---|---|
| `fragmentation_ratio` | `motion.tracks.fragmentation_ratio` |
| `n_interpolated_gap_frames` | `motion.tracks.n_interpolated_gap_frames` |
| `n_tracks` | `motion.tracks.n` |
| `n_tracks_retired` | `motion.tracks.n_retired` |
| `track_drift_um` | `motion.tracks.drift_um` |
| `track_ids` | `motion.tracks.ids` |
| `track_lengths` | `motion.tracks.n_frames` |
| `track_start_frame` | `motion.tracks.start_frame` |
| `tracks_detection_id` | `motion.tracks.detection_id` |
| `tracks_midline_id` | `motion.tracks.midline_id` |
| `tracks_observed` | `motion.tracks.observed` |
| `tracks_orientations` | `motion.tracks.orientations` |
| `tracks_positions_px` | `motion.tracks.positions_px` |
| `tracks_positions_um` | `motion.tracks.positions_um` |
| `tracks_slen` | `motion.tracks.slen` |

## `MotionAnalysis.group_tracks`

Each grouping kind is stored **once**, beside the others, under `motion.groups.<kind>.<leaf>`;
grouping a kind again replaces it and drops its analysis. There is no "grouping currently in
effect": every reader names its kind, and `sarc.motion.tracks()` has one `group_<kind>` column per
stored kind (plus `order_<kind>` for myofibril and loi).

| old | new |
|---|---|
| `group_kind` | — (the kind is part of every key) |
| `group_member_counts` | `motion.groups.<kind>.member_counts` |
| `group_n_vectors_in_long_tracks` | `motion.groups.<kind>.n_vectors_in_long_tracks` |
| `group_n_vectors_total` | `motion.groups.<kind>.n_vectors_total` |
| `grouping_hash` | `motion.groups.<kind>.hash` |
| `n_groups` | `motion.groups.<kind>.n` |
| `track_group_id` | `motion.groups.<kind>.track_group_id` |
| `track_group_order` | `motion.groups.<kind>.track_group_order` |
| `track_ids_snapshot` | `motion.groups.<kind>.track_ids` |

## `MotionAnalysis.analyze`

Written once per grouping `kind` in pool / mband / myofibril / domain / loi / custom. The `_timeseries` suffix is dropped — nearly every array in the store is per-frame, so it marked nothing.

| old | new |
|---|---|
| `track_motion_kind` | — (a kind is analysed when `params.motion.analyze.<kind>.*` exists) |
| `<kind>_slen_timeseries` | `motion.<kind>.slen_avg` (the group average; the members' lengths are `motion.tracks.slen`) |
| `<kind>_slen_median_timeseries` | `motion.<kind>.slen_median` |
| `<kind>_slen_std_timeseries` | `motion.<kind>.slen_std` |
| `<kind>_slen_q25_timeseries` | `motion.<kind>.slen_q25` |
| `<kind>_slen_q75_timeseries` | `motion.<kind>.slen_q75` |
| `<kind>_n_members_timeseries` | `motion.<kind>.n_members` |
| `<kind>_contr` | `motion.<kind>.contr` |
| `<kind>_labels_contr` | `motion.<kind>.labels_contr` |
| `<kind>_n_contr` | `motion.<kind>.n_contr` |
| `<kind>_n_contr_complete` | `motion.<kind>.n_contr_complete` |
| `<kind>_contr_complete` | `motion.<kind>.contr_complete` |
| `<kind>_beating_rate` | `motion.<kind>.beating_rate` |
| `<kind>_beating_rate_variability` | `motion.<kind>.beating_rate_variability` |
| `<kind>_equ` | `motion.<kind>.equ` |
| `<kind>_contr_max` | `motion.<kind>.contr_max` |
| `<kind>_elong_max` | `motion.<kind>.elong_max` |
| `<kind>_vel_contr_max` | `motion.<kind>.vel_contr_max` |
| `<kind>_vel_elong_max` | `motion.<kind>.vel_elong_max` |
| `<kind>_time_to_peak` | `motion.<kind>.time_to_peak` |
| `<kind>_time_to_relax` | `motion.<kind>.time_to_relax` |
| `<kind>_time_contr` | `motion.<kind>.time_contr` |

The domain-only `domain_n_vectors_timeseries` spelling is gone: every kind now
writes `n_members`.

**New in 1.0, no pre-1.0 counterpart** — the per-group heterogeneity of the member
tracks, written by the same call for every kind (see
`sarcasm.analysis.heterogeneity`): `motion.<kind>.corr_delta_slen_serial`,
`corr_delta_slen_mutual`, `ratio_delta_slen_mutual_serial`, `corr_vel_serial`,
`corr_vel_mutual`, `ratio_vel_mutual_serial`, `corr_n_cycles`, `oscill_frequencies`,
`oscill_magnitudes_avg`, `oscill_magnitudes_single`, `oscill_peak_avg`, `oscill_amp_avg`,
`oscill_peak_1_single`, `oscill_amp_1_single`, `oscill_peak_2_single`, `oscill_amp_2_single`.
The 0.5 `Motion.analyze_correlations` / `analyze_oscillations` results lived only in
a manual LOI's `loi_data`; the 4-D correlation matrices (`corr_delta_slen`, `corr_vel`)
and the raw wavelet coefficients (`oscill_cfs*`) are not stored any more. Note that
the 0.5 *mutual* correlation was reduced over the wrong matrix axes (it paired the
sarcomere index with the cycle index); 1.0 values follow eq. (1) of Haertter et al.

---

232 keys total: 105 named individually, plus 126 per-group motion keys (21 members x 6 grouping kinds).
