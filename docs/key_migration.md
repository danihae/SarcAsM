# Result key migration (1.0)

In 1.0 a result key **is its path**: `sarc.data['motion.pool.slen']` and
`sarc.data.motion.pool.slen` are the same value. Keys live in three namespaces —
`structure` (per-frame / single-frame morphology), `motion` (everything derived from
the sarcomere tracks) and `params` (what each analysis step ran with).

There is **no runtime translation layer** and no store migration: an analysis written
before 1.0 holds the old keys and must be regenerated. Exported CSV / DataFrame column
names change with the keys, so downstream analyses need re-pointing against this table.

`params.<step>.<name>` keys are **unchanged**.

## `SarcAsM.analyze_cell_mask`

| old | new |
|---|---|
| `cell_mask_area` | `structure.cell.mask_area` |
| `cell_mask_area_ratio` | `structure.cell.mask_area_ratio` |
| `cell_mask_intensity` | `structure.cell.mask_intensity` |

## `SarcAsM.analyze_z_bands`

The `z_` prefix is replaced by the group name.

| old | new |
|---|---|
| `n_zbands` | `structure.zbands.n` |
| `z_ends` | `structure.zbands.ends` |
| `z_intensity` | `structure.zbands.intensity` |
| `z_intensity_mean` | `structure.zbands.intensity_mean` |
| `z_intensity_std` | `structure.zbands.intensity_std` |
| `z_labels` | `structure.zbands.labels` |
| `z_lat_alignment` | `structure.zbands.lat_alignment` |
| `z_lat_alignment_groups` | `structure.zbands.lat_alignment_groups` |
| `z_lat_alignment_groups_mean` | `structure.zbands.lat_alignment_groups_mean` |
| `z_lat_alignment_groups_std` | `structure.zbands.lat_alignment_groups_std` |
| `z_lat_alignment_mean` | `structure.zbands.lat_alignment_mean` |
| `z_lat_alignment_std` | `structure.zbands.lat_alignment_std` |
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

## `SarcAsM.analyze_sarcomere_vectors`

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

## `SarcAsM.analyze_myofibrils`

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

## `SarcAsM.analyze_sarcomere_domains`

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

## `SarcAsM.detect_lois`

| old | new |
|---|---|
| `loi_data` | `motion.loi.data` |

## `SarcAsM.track_sarcomere_vectors`

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

## `SarcAsM.group_tracks`

`track_motion_kind` (written by `analyze_track_motion`) becomes `motion.groups.analyzed_kind`, next to the `kind` the grouping was built with.

The keys below describe **the grouping currently in effect** and are overwritten by the
next `group_tracks` call. Each is additionally mirrored under `motion.groups.<kind>.<leaf>`
(`motion.groups.pool.n`, `motion.groups.mband.member_counts`, ...), so grouping one tracking
several ways keeps every grouping independently readable. `analyze_track_motion` accumulates
the kinds it has run in `motion.groups.analyzed_kinds`, and readers that name a kind
(`get_tracks(kind=...)`, `Export.get_motion_dict_per_group`) read that kind's mirror and
raise if it is absent, rather than silently falling back to the current grouping.

| current | per-kind mirror |
|---|---|
| `motion.tracks.group_id` | `motion.groups.<kind>.track_group_id` |
| `motion.tracks.group_order` | `motion.groups.<kind>.track_group_order` |
| `motion.groups.n` | `motion.groups.<kind>.n` |
| `motion.groups.member_counts` | `motion.groups.<kind>.member_counts` |
| `motion.groups.n_vectors_total` | `motion.groups.<kind>.n_vectors_total` |
| `motion.groups.n_vectors_in_long_tracks` | `motion.groups.<kind>.n_vectors_in_long_tracks` |
| `motion.groups.track_ids` | `motion.groups.<kind>.track_ids` |
| `motion.groups.hash` | `motion.groups.<kind>.hash` |

| old | new |
|---|---|
| `group_kind` | `motion.groups.kind` |
| `group_member_counts` | `motion.groups.member_counts` |
| `group_n_vectors_in_long_tracks` | `motion.groups.n_vectors_in_long_tracks` |
| `group_n_vectors_total` | `motion.groups.n_vectors_total` |
| `grouping_hash` | `motion.groups.hash` |
| `n_groups` | `motion.groups.n` |
| `track_group_id` | `motion.tracks.group_id` |
| `track_group_order` | `motion.tracks.group_order` |
| `track_ids_snapshot` | `motion.groups.track_ids` |

## `SarcAsM.analyze_track_motion`

Written once per grouping `kind` in pool / mband / myofibril / domain / loi / custom. The `_timeseries` suffix is dropped — nearly every array in the store is per-frame, so it marked nothing.

| old | new |
|---|---|
| `track_motion_kind` | `motion.groups.analyzed_kind` |
| `<kind>_slen_timeseries` | `motion.<kind>.slen` |
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

---

232 keys total: 105 named individually, plus 126 per-group motion keys (21 members x 6 grouping kinds).
