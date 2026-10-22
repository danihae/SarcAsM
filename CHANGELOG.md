# Changelog

All notable changes to SarcAsM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Headless command line: `sarcasm info | structure | motion | results | features FILE`. Each command prints one
  JSON document to stdout (logs and progress bars on stderr) and repeats the warnings of the run in its
  `"warnings"` list, which explain null values. An error is a JSON object with a hint on the last line of stderr;
  the exit code is 3 for input to fix (missing file, missing or conflicting calibration, implausible axes, unknown
  result key), 2 for bad usage, 1 otherwise. `sarcasm` without a command still opens the GUI. See `docs/cli.rst`.
- `llms.txt` at the repository root: the commands and the rules that keep results correct, for language-model
  agents.
- Pixel-size check in `analyze_sarcomere_vectors`: the stripe period of the raw image along the sarcomere axis
  (autocorrelation of intensity profiles through the vectors of the first analyzed frame, independent of the
  pixel size) is compared with the measured sarcomere length. When the period is less than 0.75x the length, a
  warning asks whether the pixel size is right (no value is suggested): a pixel size about 2x too small makes the
  detection pair every other Z-band, so the lengths look normal while every trajectory jumps by half a
  sarcomere. New result `structure.sarcomere.stripe_period_ratio`; `sarcomere_vectors.stripe_period`,
  `sarcomere_vectors.pixelsize_plausibility`.

### Changed

- `import sarcasm` resolves its public names on first access, so it no longer imports torch, numba and matplotlib
  up front (`sarcasm --help` 5.8 s → 0.1 s). `from sarcasm import *` is unchanged.
- Picking channel 0 of a multi-channel image by default logs a warning instead of info.

### Fixed

- Reopening an analysed file ignored an explicit `pixelsize=`, `frametime=`, `axes=` or `channel=` without notice:
  the stored metadata replaced it. A value that differs from the store now raises `MetaDataError` pointing to
  `restart=True` (a warning in the GUI); a frame time the store does not have is filled in.
- An image with more than 6 channels and no `channel=` raises `MetaDataError` instead of analysing channel 0.
  Dozens of "channels" are a time or Z axis mislabelled in the file header (tifffile's ImageJ writer labels a 3D
  stack saved without `axes` this way), and channel 0 is then a single frame of the movie. Pass `axes='TYX'`, or
  `channel=` if the channels are real.
- A Git LFS pointer in place of the image fails with a message naming `git lfs pull` instead of a TIFF error.

## [1.0.0b3] — 2026-10-21

### Added

- `track_sarcomere_vectors(refine_lengths=True, orientation_window_s=None)`: after tracking, each track gets one
  sarcomere axis, the median of its detected orientations, and its length is re-measured along that axis in every
  frame, with the Z-band map and length estimator of `analyze_sarcomere_vectors`. Sarcomeres do not turn during a
  contraction, but their Z-bands tilt where neighbouring myofibrils shorten by different amounts (a shear), and a
  detection's orientation, the Z-band normal in that frame, follows that tilt and jitters on top of it; normal to
  tilted Z-bands the spacing is shorter than along the filaments by the cosine of the tilt. On a 61-fps movie the
  Z-bands tilted by 5° (SD) during contractions while the sarcomere axes, from the tracked displacement field,
  turned by less than 1° (the tracker and decomposition recover a synthetic 4° rotation as 4.0° and a 4° shear as
  3.9° tilt / 0.1° turn). Along the axis, the frame-to-frame length noise fell from 19.7 to 15.3 nm, the spread in
  quiet diastolic windows from 20.8 to 17.7 nm, and the cell-average shortening was 10 % smaller (84 vs 93 nm;
  individual per-cycle shortening 176 vs 197 nm) than along the detection orientations; the contraction windows
  detected on these lengths were shorter (57 vs 63 % of the frames). `orientation_window_s`
  gives a running median over that interval instead, for long recordings in which the tissue slowly turns (warns
  and falls back to one axis per track with fewer than 5 frames in it or an unknown frametime). The detection
  values are kept in `motion.tracks.slen_raw` / `motion.tracks.orientations_raw`. Functions:
  `sarcomere_tracking.track_axis_orientations`, `sarcomere_tracking.refine_track_lengths`.
- `analyze_sarcomere_vectors` now stores its `interpolation_method`
  (`params.analyze_sarcomere_vectors.interpolation_method`), so later steps can re-measure with the same
  estimator.

### Fixed

- `Utils.fast_profile_lines` (the Z-band profiles behind every sarcomere length) sampled the image
  nearest-neighbour and spread its samples over `ceil(length + 1)` points, i.e. 1 % closer than a pixel, while
  the length estimator assumes one pixel per sample. Every sarcomere length was therefore ~2 % too long
  (+29 nm on a 61-fps movie of 3.4 M tracked frames, with a spread of 17 nm that depended on the sub-pixel
  position of the profile), and the midpoint-corrected vector positions sat 0.5 px off along the axis. Profiles
  are now sampled bilinearly, one sample per pixel, centred on the line's midpoint. Lengths measured with earlier
  versions are longer by about 1.7 % + 1 %; length *changes* and velocities are affected by the 1 % only.
- `track_sarcomere_vectors(refine_lengths=True)` also moves each track's position to the midpoint between its two
  Z-bands in every frame (`motion.tracks.positions_px` / `positions_um`). The tracker follows integer M-band
  skeleton pixels, whose ±0.5 px quantisation showed as a staircase in the position and Z-band trajectories.
- `structure.domain.members` held every Leiden cluster of a frame, including those that the size filters
  (`area_min`, more than 10 vectors) removed, so it did not line up with `structure.domain.area`, `slen`,
  `oop`, `orientation` and the domain mask (e.g. 135 member lists for 20 domains), and the per-domain
  Z-band registry was averaged over all clusters. It now holds only the domains that passed, in label order.
  The domain masks drawn by the plots and the app and the domain grouping of tracks are unchanged.
- LOI contraction detection (`Motion.predict_contractions`) missed weak contractions: it averaged the ContractionNet
  probability over all Z-band position and sarcomere length traces together, so length traces dominated by noise
  diluted a clear Z-band signal below the threshold (a 1 µM Mavacamten LOI with a 5-nm mean shortening: Z-band
  channel 0.70, length channel 0.18, joint mean 0.40 < 0.45, no contraction detected). The probability is now
  averaged per channel and the larger channel mean is thresholded. On 50 hand-annotated LOIs the median F1 score is
  unchanged within 0.01 (16 LOIs from wells held out in training: 0.926 → 0.918), the Mavacamten LOI goes from 0
  to 0.79, and slightly more frames next to annotated contractions are called contracting. This also applies to
  track chains viewed as LOIs (`get_track_motion`); the group analysis of `analyze_track_motion`, which applies
  the network to the group-average length, is not affected.
- `detect_lois(distance_threshold_lois=...)` was in pixels (default 40), so the LOI clusters depended on the
  pixel size (40 px = 2.4 µm at 0.061 µm/px, 10.4 µm at 0.26 µm/px). The Hausdorff distances between lines are
  now computed in µm and the threshold is in µm, default 2.5 (≈ the old default at 0.06 µm/px).
  `loi_detection.hausdorff_distance_lois` is vectorised (identical values; 2,300 lines in about a minute instead of
  ~17 min).
- `detect_lois(mode='longest_in_cluster')` took the longest line of the clusters ranked by the length of that
  line, not of the largest clusters as documented, and `mode='random_from_cluster'` drew from randomly chosen
  clusters of any size. Lines that grow along the same myofibril but differ in extent fall into different
  clusters (the symmetric Hausdorff distance is set by where they diverge), so a one-line fragment along a long
  myofibril competed with the cluster of that myofibril and the selected LOIs overlapped. Both modes now use the
  `n_lois` largest clusters (most lines).
- `detect_lois` defaults `persistence` 4 → 8 and `threshold_distance` 0.5 → 0.3 µm (also in the app, whose
  cluster threshold is now in µm, default 2.5). Sarcomere vectors are dense M-band midline points, so each growth
  step could snap up to 0.5 µm sideways along an M-band and the LOIs were wavy and drifted across Z-bands; with
  the new defaults the mean change of direction per sarcomere of the selected LOIs on a 61-nm-pixel movie fell
  from 2.5° to 1.4°.

## [1.0.0b2] — 2026-10-19

### Added

- Z-band registry of the sarcomere vectors (`analyze_sarcomere_vectors`). Every sarcomere vector defines a
  Z-disc lattice, its two Z-discs repeated every sarcomere length along its axis; a neighbouring sarcomere's
  Z-disc at offset *x* from the nearest lattice line scores cos(2π *x* / L), averaged over its two discs:
  1 when the Z-discs are level with the vector's own, 0 on average for random positions, −1 when they sit at
  its M-band. The lateral registry is the mean score over the neighbours one myofibril over
  (`registry_lateral_band`, default 1.5–2.5 µm across the axis), the axial registry over the parallel
  neighbours along the myofibril (`registry_axial_range`, default 3 sarcomeres; `registry_max_angle`, default
  20°). Stored per vector (`structure.sarcomere.registry_lateral` / `registry_axial`, NaN for vectors without
  such neighbours) and per frame as `registry_lateral_mean`, `registry_axial_mean` and their average
  `registry_mean`; `analyze_sarcomere_domains` adds the same per domain
  (`structure.domain.registry_lateral` / `registry_axial` and their frame means). It needs only the stored
  vectors (about 0.1 s per 2000² frame). On a chronic plate the lateral registry rose from 0.10 at day 0.5
  to 0.26 at day 8 and the axial registry from 0.55 to 0.68; mavacamten (10 µM) lowered them to 0.04 and 0.47.
  The parameters are in the app's vector settings and in batch processing.
- Ratiometric Z-band contrast (`analyze_z_bands`): for each Z-band, (mean intensity of the Z-band −
  background) / (mean intensity of the non-Z-band cell pixels within 1 µm of it − background), stored as
  `structure.zbands.contrast` with the per-frame median `structure.zbands.contrast_median`. Neighbouring
  Z-bands split the space between them, so the ring holds no Z-band pixels. The background is the median
  intensity outside the cell mask (`structure.zbands.background`); frames almost fully covered by cells take
  the background of the recording (the median over up to five evenly spaced detected frames). Unlike the
  absolute intensities, the ratio does not depend on exposure,
  gain or bleaching. It measures the in-focus Z-band signal relative to the diffuse signal around it
  (unassembled α-actinin and out-of-focus light), which is stored as
  `structure.zbands.surround_intensity_median`. On a chronic plate the contrast was 3.7 under mavacamten
  against 2.2 in controls, because the diffuse signal was weaker, and it fell during the first four days of
  culture, when the Z-band signal dropped while the diffuse signal stayed constant; blur lowers it by about
  6 % per 0.1 µm.
- Percentile twitch kinetics in the track-based motion analysis (`analyze_track_motion`, every grouping
  kind): `time_to_50` (onset to 50 % shortening), `time_contr_10_90` (10 % to 90 % shortening),
  `time_relax_50` and `time_relax_90` (peak to 50 % / 90 % relaxation). They are first crossings of
  fractions of the cycle's peak shortening, interpolated between frames; relaxation is followed past the
  end of the contraction window up to the next cycle. Before, times to custom percentages existed only as
  the opt-in `custom_perc` of the legacy LOI analysis.
- `analyze_sarcomere_vectors` removes false-positive sarcomere vectors. A vector whose
  orientation deviates by more than `max_neighbor_deviation` (default 30°) from the mean
  orientation of the vectors within `neighbor_radius` (default 1 µm) is dropped; these are
  M-band fragments pointing across their neighbours, mostly at crossings of differently
  oriented myofibrils. Vectors without a neighbour inside the radius are always kept, so
  thin or short M-bands are not affected. The number removed per frame is stored as
  `structure.sarcomere.n_vectors_removed`; `max_neighbor_deviation=None` keeps every
  vector. The same settings are in the app's vector parameters and in batch processing.
- `Plots.plot_track_vectors` draws every tracked sarcomere of a frame as its Z-band-to-Z-band
  segment, coloured by length, length change, velocity, track coverage or group. `Animations`
  turns the same drawing into movies: `animate_image` (raw movie and Z-band masks side by
  side), `animate_track_vectors` (one live colormap panel) and `animate_results` (image panels
  above grouped time series with a time cursor); `.mp4` needs ffmpeg (system or
  `imageio-ffmpeg`), `.gif` works without.

### Changed

- Frame selection is the same for every step and stored by frame number. Every `frames` argument
  (detection, the analyses, tracking, `full_analysis_structure`) and every single frame (`reference_frame`,
  `detect_lois`, the plots) accepts a frame number in any spelling: `k`, `np.int64(k)`, `[k]`, `(k,)`,
  `range(k, k + 1)` and `np.array([k])` select the same frame, and frame lists may be unsorted or repeat
  frames. Results are stored under their frame numbers in every case: after `detect_sarcomeres(frames=[5,
  9])`, `analyze_z_bands()` stores frames 5 and 9 under 5 and 9 (before: under positions 0 and 1), and
  `analyze_z_bands(frames=0)` after `detect_sarcomeres(frames=1)` raises instead of analysing frame 1 under
  0. Frames that were not detected, negative or out-of-range frames, floats and empty selections raise an
  error that names the valid frames, instead of failing inside zarr, running on another frame or being stored
  under index −1. A detection of one frame of a movie stores its masks with a frame axis, like any other
  subset; stores written before read correctly. The app's frame field accepts `all`, one frame or frames
  separated by commas and marks anything else as invalid; before, an invalid entry ran frame 0 and `1, 3`
  ran frame 1 only.
- Lateral Z-band connections (`analyze_z_bands`) no longer join staggered Z-bands. A connection
  continues a Z-band across a gap, so a pair of Z-band ends is not linked when the link is offset
  along the sarcomere axis by more than `max_axial_offset` (new parameter, default 0.9 µm, about
  half a sarcomere: such a link reaches the next Z-band of the neighbouring myofibril) or when its
  straight segment runs over a third Z-band. The direction of each Z-band end is now taken from
  the Z-band's own skeleton over its last 0.5 µm instead of from the orientation field at the end
  pixel, where the field is least defined. On ten 8-day control movies, links offset by more than
  0.9 µm drop from 25 % to 5 % and links over a third Z-band from 2 % to 0; lateral groups are
  about 20 % shorter. `max_axial_offset=None` turns the offset limit off. The setting is in the
  app's Z-band parameters and in batch processing.

### Removed

- Z-band straightness (`structure.zbands.straightness`, `_mean`, `_std`) and myofibril straightness
  (`structure.myofibril.straightness`, `_mean`, `_std`). Z-band straightness (area / convex-hull area)
  measured Z-band length rather than curvature: per Z-band it fell with length (Spearman −0.72), and its
  decline over 8 days of culture disappeared when Z-bands of equal length were compared. Myofibril
  straightness was length-dependent in the same way; `structure.myofibril.bending` stays.
- `structure.zbands.lat_size_groups` (`_mean`, `_std`). With at most one lateral link per Z-band end, the
  laterally linked groups are chains, so the mean group size is `2 / (2 − lat_neighbors_mean)` and the size
  of each group is the length of its entry in `structure.zbands.lat_groups`.
- Lateral Z-band alignment A (`structure.zbands.lat_alignment`, `_mean`, `_std`, and per lateral group
  `lat_alignment_groups`, `_mean`, `_std`). A scored the geometry of each lateral connection, so it
  described the connections the linker had already accepted rather than the order of the tissue: on a
  recomputed chronic plate it stayed at 0.79-0.80 under 10 µM mavacamten while sarcomeres fell apart. The
  Z-band registry of the sarcomere vectors replaces it (see Added). The score still chooses the lateral
  connections (`a_min`) but is no longer reported; the compound-screen feature set of `downstream.mfpca`
  uses the lateral and axial registry instead.
- `z_band_analysis.analyze_z_bands` returns a dict instead of an 18-element tuple.
- Stores analysed before keep the removed keys until they are re-analysed with `restart=True`.

### Fixed

- Z-band intensities (`intensity_mean`, `intensity_std`, `mask_intensity`) after a detection of selected
  frames, e.g. a single frame of a time-lapse in the app, were measured on the first raw frames of the movie
  instead of the detected ones. `analyze_cell_mask(frames='all')` after such a detection paired the masks
  with the first raw frames in the same way; it now analyzes the detected frames.
- `detect_sarcomeres(frames=[k])` (any one-element selection) stored masks that `analyze_cell_mask` could not
  read, so detection failed with an `IndexError` after the masks were written but before their frame list
  was. Detection now stores the frame list right after the masks.
- After a detection of one frame k of a movie, the analyses filed the Z-band and vector results of
  `frames=k` under 0 but the cell-mask features under k, so `analyze_myofibrils(frames=k)` failed and the
  area ratios were NaN. After a detection of several frames, `frames='all'` divided the Z-band and sarcomere
  areas by the cell area of another frame (twice the true ratio in a test movie). Vector analysis with the
  fast-movie Z-bands combined the Z-bands of frame 0 with the M-bands of the detected frame, overlay plots
  drew raw frame 0 under the results of the detected frame, and `load_mask_full_stack('zbands_fast_movie')`
  returned the stack repeated once per frame.
- The per-sarcomere LOI analysis (`Motion.analyze_trajectories`) reported the time to peak of contraction
  cycles that began before the recording and the time to relaxation of cycles that ended after it, and the
  extrema of a cycle where they sit on such a cut edge. These are NaN now, with the rules
  `analyze_track_motion` uses since 1.0; complete cycles keep their values (on eLife-era LOIs and track
  chains every value of a complete cycle is unchanged). The eLife analysis dropped edge cycles altogether.
- `Motion.analyze_trajectories(custom_perc=...)` failed with `All-NaN slice encountered` when one sarcomere
  had no value in a cycle, and a NaN in a cycle could return the time of the wrong sample. The times are now
  the first crossings of the percentages of the cycle's peak shortening, interpolated between frames, as for
  the percentile times of `analyze_track_motion` (before: the sample nearest to the level anywhere in the
  shortening or relaxation phase); NaN for a cycle with gaps, without shortening, or whose trace does not
  reach the level.
- `get_trajectories` called a second time on a myofibril chain from tracks (`get_track_motion`) smoothed the
  already smoothed sarcomere lengths again, because `Utils.nan_sav_golay` smoothed its input in place and the
  chain's lengths are stored under the key the smoothed ones replace. The measured lengths are kept as
  `slen_raw`, `nan_sav_golay` no longer changes its input, and a single call gives the same values as
  before.
- `get_track_motion(analyze=True)` did not run `analyze_popping`, so `Plots.plot_popping_events` failed on
  the fibre it returned.
- The animations (`animate_image`, `animate_track_vectors`, `animate_results`) showed movie frame p at
  position p of the tracked frames, i.e. the wrong raw frames and Z-bands, and failed on the tracks, when
  tracking did not start at frame 0.
- The lateral Z-band group plot (`Plots.plot_z_lateral_connections` with groups) and the app's group layer
  left out the first group of every frame.
- `analyze_sarcomere_domains`, the domain plot, the app's domain layer and `group_tracks(by='domain')` failed
  with an `OverflowError` on frames with more than 255 domains, because the domain mask was 8-bit; it is
  16-bit now.
- A frame without M-bands returned its vector positions as a plain empty list. Next to the `(n, 2)`
  arrays of the other frames, that made the results store write the whole per-frame list of
  `structure.sarcomere.pos` / `pos_px` as an inline JSON attribute instead of a ragged array: in a plate
  analysis 233 of 864 stores were affected, with `zarr.json` files of up to 221 MB. Such a frame now
  yields empty arrays of the usual shape.
- Per-frame results whose arrays vary along a trailing axis were read back from the wrong frames when one
  frame was empty along it: the lateral Z-band links of a frame with Z-bands but no links, `(4, 0)`, shifted
  the stored offsets, so the next frame read back the links of another frame (the lateral-connection plots
  then failed with an index error). Such lists are now stored as attributes.
- The two ends of one Z-band are excluded from lateral linking again. The exclusion mask paired
  rows (2k, 2k + 1) of the list of Z-band ends, which is stored in Fortran order, so it excluded
  end pairs of two different Z-bands and let a curved Z-band link to itself.
- Myofibril line growth could run forever: the growth loop had no memory of the points it had
  already passed, so a ring of vectors whose predicted steps lead back onto the line (crossings,
  branch points) was walked round and round until the process was killed. 4 of 864 movies of a
  plate analysis hung this way. A line now stops when its next point is already part of it.
- `structure.sarcomere.orientation_mean` and `orientation_std` are now axial statistics: the
  orientations (in `[0, π)`) are doubled, averaged on the circle and halved, so the mean lies
  in `[0, π)` and no longer depends on which of the two equivalent directions a vector was
  assigned. Before, `scipy.stats.circmean` with its 2π period averaged θ and θ + π to a
  different axis.
- Masks of a subset detection (`detect_sarcomeres(frames=[32, 33])`) are stored only for those
  frames; `analyze_sarcomere_vectors`, the plot backgrounds and the animations passed movie
  frame numbers straight to the store and failed with a bounds error, and the cell-mask
  features of such a run were indexed by position instead of frame. `_read_mask` now maps
  frame numbers to store pages itself, and the cell-mask features are per-frame arrays like
  every other structure result.
- Analyses of a field with nothing detected no longer fail when results are saved. An
  empty sparse result (e.g. a per-frame domain mask with no domains) was written with
  `chunks=None`, which zarr ≥ 3.3 rejects, so the whole flush raised and nothing was
  stored. Empty arrays now let zarr choose the chunking.

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

- `analyze_sarcomere_vectors` removes false-positive sarcomere vectors. A vector whose
  orientation deviates by more than `max_neighbor_deviation` (default 30°) from the mean
  orientation of the vectors within `neighbor_radius` (default 1 µm) is dropped; these are
  M-band fragments pointing across their neighbours, mostly at crossings of differently
  oriented myofibrils. Vectors without a neighbour inside the radius are always kept, so
  thin or short M-bands are not affected. The number removed per frame is stored as
  `structure.sarcomere.n_vectors_removed`; `max_neighbor_deviation=None` keeps every
  vector. The same settings are in the app's vector parameters and in batch processing.
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

- `structure.sarcomere.orientation_mean` and `orientation_std` are now axial statistics: the
  orientations (in `[0, π)`) are doubled, averaged on the circle and halved, so the mean lies
  in `[0, π)` and no longer depends on which of the two equivalent directions a vector was
  assigned. Before, `scipy.stats.circmean` with its 2π period averaged θ and θ + π to a
  different axis.
- Masks of a subset detection (`detect_sarcomeres(frames=[32, 33])`) are stored only for those
  frames; `analyze_sarcomere_vectors`, the plot backgrounds and the animations passed movie
  frame numbers straight to the store and failed with a bounds error, and the cell-mask
  features of such a run were indexed by position instead of frame. `_read_mask` now maps
  frame numbers to store pages itself, and the cell-mask features are per-frame arrays like
  every other structure result.
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
