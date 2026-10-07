# Changelog

All notable changes to SarcAsM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0b1] — 2026-10-07

A breaking major release. Analyses produced by 0.5.x cannot be read by 1.0 — install
`sarc-asm==0.5.*` to open them, or recompute. Result keys changed throughout; see
[`docs/key_migration.md`](docs/key_migration.md) for the full old → new table.

### Requirements

- Python **3.12 or 3.13** (was ≥ 3.10).
- New dependencies: `zarr ≥ 3.2` (the analysis store) and `opencv-python-headless ≥ 4.8`
  (the optional flow predictor); `scipy` and `tqdm` are now declared.
- `pandas ≥ 2.1`, `napari ≥ 0.5, < 0.9` (0.7 and 0.8 supported), `numpy < 3`,
  `bio-image-unet ≥ 1.2.2`.
- `numba < 1.0` (was `< 0.62`, which capped numpy at 2.2).

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
- `analyze_track_motion(aggregate=)` takes `'mean'` (default, every grouping) or
  `'median'`; the per-kind `None`/'auto' resolution and the `nanmean`/`nanmedian` names are
  gone.
- **ContractionNet retrained** (polarity-invariant, duty-cycle and sampling robust); the
  operating threshold is read from the checkpoint. Pre-1.0 checkpoints are rejected.
- **Sarcomere U-Net checkpoint chosen by pixel size** (`model_path='auto'`): the
  scale-augmented `generalist` (v1) at ≥ 0.08 µm/px, the previous checkpoint (`legacy`)
  below, where v1 fragments Z-/M-band lines. Either can be forced. The two are not
  interchangeable within one study (cell-mask area shifts).
- The app's 3D U-Net step is a collapsed "advanced" sub-menu of *Detect sarcomeres*, with
  a note that the bundled 3D model usually has to be re-trained per recording type.
- `detect_sarcomeres` computes the cell-mask features (`structure.cell.*`) itself;
  `analyze_cell_mask` remains for re-evaluating with another threshold. The app's
  "Analyze cell mask" step and batch checkbox are gone.
- `analyze_sarcomere_vectors`: `peak_algorithm` and `smooth_zbands_sigma` removed;
  new defaults `peak_prominence=0.4`, `interp_factor=4` (was 0), `linewidth=0.3` µm
  (was 0.2). Sarcomere lengths differ slightly from 0.5 with default settings.
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
- **Optional image-flow motion predictor** for coarse frame rates:
  `track_sarcomere_vectors(motion_predictor='flow')` (default `'none'`) predicts each
  sarcomere's step from dense optical flow of the raw image before matching: along the
  sarcomere axis always, sideways only where the flow is coherent and large. At low frame
  rates relative to the contraction it gives fewer fragments and higher coverage; at high
  frame rates it changes nothing. In the app: the *flow predictor* checkbox on the Motion tab.
- `analyze_sarcomere_vectors(smooth_orientation_sigma=)`: optional temporal smoothing of
  the orientation field (off by default).
- Automatic patch and batch sizing for U-Net prediction: `detect_sarcomeres(max_patch_size='auto',
  batch_size='auto', memory_budget_gb=2.0)` (and `'auto'` for the 3D fast-movie model)
  size patches from free device memory; the app's prediction panels have an *Auto*
  checkbox, on by default.
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
  of the group with the clicked sarcomere highlighted, zoom/pan toolbar); the summary figure
  gains a raster of every sarcomere's ΔSL over the averaged cycle, sorted by time to peak.
- Per-group `equ_std` (spread of the members' resting lengths); `slen_std` (within-group
  SL spread per frame) is exported as its time mean.
- `SarcAsM.get_track_kinematics()` (per-track ΔSL / velocity / resting length) and
  `Plots.plot_track_raster` (cycle-averaged sarcomere × time raster sorted by
  time-to-peak or amplitude, or the full recording by group).
- Documentation: `docs/key_migration.md`, the tracking tutorial, a rewritten quickstart;
  a CI test workflow gates PyPI publishing and standalone builds.

### Changed

- `BatchExport` writes `.pkl` pickles (was `.pd`); old `.pd` files still load.
- Image and overlay plots use equal aspect; `plot_slen_mean` defaults to a 1.3–2.0 µm
  y-range.
- The `.ome.zarr` store packs large arrays (image, masks, track blocks) into shard files of
  about 256 MiB (zarr v3 sharding) instead of one file per frame or row chunk: a 500-frame
  store drops from ~4500 files to ~200. Per-frame and per-track reads are unchanged. Stores
  written by earlier 1.0 betas stay readable; an array is re-laid out when an analysis step
  rewrites it.

### Performance

- `analyze_sarcomere_vectors` is about 7× faster (≈ 7 → 1 min on a 500-frame movie):
  orientation is sampled only at M-band skeleton points and peak finding runs in a numba
  kernel. Values within the filter radius of the image border shift slightly (edges are
  replicated instead of zero-padded).
- Long stacks are predicted in memory-budgeted blocks and written straight into the store:
  a 50-frame 2000 × 2000 movie peaks at 3.6 GB instead of ≈ 13 GB.

### Fixed

- `rescale_factor` was applied twice, so masks came back at the wrong size.
- `frames=` accepts any sequence of frame indices (`range`, tuple, numpy integers), not
  only a list.
- A partial detection no longer silently truncates the vector or Z-band analysis; frames
  are clamped to the detected ones with a warning.
- App: the parameter dock stays usable at narrow widths (long rows wrap, horizontal
  scrollbar).
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
- App: the sarcomere-vector arrows carried the frame index as their time component, so
  the viewer's frame range doubled (empty second half) after the vector analysis.
- Deprecated `ScaleBar(height_fraction=)` and `DataFrame.applymap` calls.

## [0.5.0]

See the GitHub release notes.
