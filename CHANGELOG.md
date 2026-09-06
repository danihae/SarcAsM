# Changelog

All notable changes to SarcAsM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0b1] — unreleased

A breaking major release. Analyses produced by 0.5.x cannot be read by 1.0 — install
`sarc-asm==0.5.*` to open them, or recompute. Result keys changed throughout; see
[`docs/key_migration.md`](docs/key_migration.md) for the full old → new table.

### Requirements

- Python **3.12 or 3.13** (was ≥ 3.10).
- `scipy` and `tqdm` are now declared dependencies; `pandas ≥ 2.1`, `napari ≥ 0.5`,
  `numpy < 3`, `bio-image-unet ≥ 1.2.2`.

### Breaking changes

- **`Structure` is now `SarcAsM`**, the single main class for morphology, tracking and
  motion; the former base class is `SarcAsMBase`. No aliases.
- **The manual line-of-interest (LOI) motion workflow is removed**: `Motion(file, loi_name)`,
  `full_analysis_loi`, `detekt_peaks`, `track_z_bands`, `get_list_lois`, the kymograph
  module and `Export.MultiLOIAnalysis`. Motion is analysed from 2D sarcomere tracks
  (below); `Motion` objects are obtained through `SarcAsM.get_track_motion(group)`.
  `Motion`'s second positional argument is now `restart` and rejects non-bool values, as
  `restart=True` deletes the analysis store.
- **`MultiStructureAnalysis` is now `BatchExport`** (it exports already-analysed features).
- **Package layout**: `sarcasm.analysis` (vectors, detection, tracking, grouping,
  contraction, heterogeneity, myofibrils, domains, LOI lines), `sarcasm.plotting`,
  `sarcasm.io`; the top level exposes `SarcAsM`, `Motion`, `Export`, `BatchExport`,
  `Plots`, `PlotUtils`, `Utils`, `TrainingDataGenerator`.
- **One `.ome.zarr` store per recording** replaces the `<name>/` folder of TIFF masks and
  `structure.json`: raw pixels, every mask, and all results/parameters live in
  `<name>.ome.zarr`. Legacy layouts are detected and refused with a message.
  `export_json()` still writes the legacy JSON on demand.
- **A result key is its dotted path**: `sarc.data['motion.pool.slen']` and
  `sarc.data.motion.pool.slen` are the same value, in the namespaces `structure`,
  `motion` and `params`. Flat keys, `sarc.results` and flat-attribute access are gone.
  `params.<step>.<name>` always uses the parameter's own name (`model_path`, `clip_thres`).
- **Tracking and grouping replace `analyze_domain_motion`**: `track_sarcomere_vectors` →
  `group_tracks(by='pool' | 'mband' | 'myofibril' | 'domain' | 'loi' | 'custom')` →
  `analyze_track_motion`, all writing `motion.<kind>.*` with identical members. All
  tracker gates are physical (µm, degrees, seconds); `max_gap_interpolation_s` (seconds)
  replaces the frame count.
- **Contraction cycles touching the recording edges are kept** (flagged, excluded from
  durations): `n_contr` counts them, `n_contr_complete` does not.
- **Equilibrium length** (`equ`, and therefore `delta_slen`, `contr_max`, `elong_max`) is
  the median over the non-contracting frames, as the plots always showed it.
- **ContractionNet retrained** (polarity-invariant, duty-cycle and sampling robust); the
  operating threshold is read from the checkpoint. Pre-1.0 checkpoints are rejected.
- **Sarcomere U-Net checkpoint chosen by pixel size** (`model_path='auto'`): the
  scale-augmented `generalist` (v1) at ≥ 0.08 µm/px, the previous checkpoint (`legacy`)
  below, where v1 fragments Z-/M-band lines. Either can be forced. The two are not
  interchangeable within one study (cell-mask area shifts).
- The app no longer exposes the 3D U-Net ("Z-band detection (high-speed)"); it remains in
  the Python API as `detect_z_bands_fast_movie` / `analyze_sarcomere_vectors(use_fast_movie_zbands=)`.
- `detect_sarcomeres` computes the cell-mask features (`structure.cell.*`) itself;
  `analyze_cell_mask` remains for re-evaluating with another threshold. The app's
  "Analyze cell mask" step and batch checkbox are gone.
- `analyze_sarcomere_vectors`: `peak_algorithm` and `smooth_zbands_sigma` removed,
  `peak_prominence` default 0.4.
- Plot defaults: overlays draw no image background unless asked (`show_image` /
  `show_z_bands`, with `invert_*`); every `t_lim` defaults to the full recording
  (`(0, None)`); `plot_z_pos(show_kymograph=)` removed.
- The app's "Open folder" button (and `SarcAsMBase.open_base_dir` / `Utils.open_folder`)
  are removed: with the single `.ome.zarr` store there is no human-readable folder to open.
- `Motion.analyze_correlations` / `analyze_oscillations` are re-implemented over
  tracks (see Added) and no longer store the 4-D correlation matrices or raw wavelet
  coefficients. The 0.5 *mutual* correlation was reduced over the wrong axes; 1.0 follows
  eq. (1) of Haertter et al.

### Added

- **2D full-field sarcomere tracking** (`SarcAsM.track_sarcomere_vectors`): every
  sarcomere vector followed through the movie with exact per-frame assignment, honest
  gap frames (`motion.tracks.observed`) and a fragmentation-ratio QC number.
- **Track grouping** at six levels and **per-group contraction analysis** with a shared
  engine; `get_track_motion(group)` turns a myofibril/LOI chain into a `Motion` view so
  every LOI plot and analysis applies to tracked fibres.
- **Per-group heterogeneity** (`sarcasm.analysis.heterogeneity`): serial/mutual
  correlation of ΔSL and velocity across cycles (`corr_*`, `ratio_*_mutual_serial`) and
  wavelet oscillation spectra (`oscill_*`), for every grouping kind and in the
  `get_track_motion(analyze=True)` chain.
- `group_tracks(min_group_size=, max_drift_slen=)`; `by='loi'` builds 1-D chains.
- 3D fast-movie Z-band prediction is used automatically for motion when available.
- OME-Zarr input from third-party tools, with pixel size and frame time read from it;
  TIFF `I`/`Q` stack-axis detection.
- `Export.tabular_frame`, `BatchExport.load_motion_data`; per-group motion export.
- napari app: Motion tab rebuilt on track → group → analyze with a per-fibre detail
  panel, LOI drawing, drop-to-import and "Open .ome.zarr"; batch runs use the tuned LOI
  parameters and expose `min_group_size`. Tracked sarcomeres are shown as coloured dots at
  their current position (ΔSL / SL / velocity / group / coverage; only the current frame is
  held by the layer) plus a **Groups** layer of labelled fibre paths; clicking a sarcomere or
  a fibre path selects its group and opens a time-series panel (SL / ΔSL / velocity overlay
  of the group with the clicked sarcomere highlighted, zoom/pan toolbar).
- `SarcAsM.get_track_kinematics()` (per-track ΔSL / velocity / resting length) and
  `Plots.plot_track_raster` (cycle-averaged sarcomere × time raster sorted by
  time-to-peak or amplitude, or the full recording by group).
- Documentation: `docs/key_migration.md`, the tracking tutorial, a rewritten quickstart;
  a CI test workflow gates PyPI publishing and standalone builds.

### Fixed

- `restart=True` no longer fails on macOS when the folder is open in Finder.
- Synthesized fibre chains: a missing member blanks only its own row; chain geometry is
  anchored on the grouping's reference frame; `min_coverage` no longer punches holes in
  chains; `z_pos[0] == 0` again.
- Tracks drifting onto a neighbour during coasting; identity swaps on coarse pixel sizes.
- `myofibril_analysis` random seed `0` was ignored; `midline_length_vectors` misaligned
  after NaN filtering.
- Contraction-centred plot windows near frame 0 produced empty slices; plots with a
  `None` time bound raised.
- Tabular exports no longer replicate the per-frame `time` vector into every row.
- `analyze_myofibrils` / `analyze_sarcomere_domains` with `frames='all'` after a partial
  detection (e.g. `detect_sarcomeres(frames=0)` then `full_analysis_structure()`) no longer
  raise; `'all'` means every frame that carries sarcomere vectors.
- Deprecated `ScaleBar(height_fraction=)` and `DataFrame.applymap` calls.

## [0.5.0]

See the GitHub release notes.
