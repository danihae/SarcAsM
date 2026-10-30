# Sarcomere detection models

| alias | file | validated pixel size | notes |
|---|---|---|---|
| `generalist` | `model_sarcomeres_generalist_v2.pt` | 0.06–0.45 µm | v1 fine-tuned on simulated myofibrils and myofibril-axis orientation labels |
| (not bundled) | `model_sarcomeres_generalist_v1.pt` | 0.08–0.45 µm | scale-augmented (v1.0.0b1–b3); see below |
| `legacy` | `model_sarcomeres_generalist.pt` | 0.06–0.35 µm | pre-v1.0.0; trained on high-magnification data at native resolution |

The default is `generalist` (v2) at every pixel size (`model_path=None` or `'auto'`).
Earlier versions, including the 1.0.0 betas, picked `legacy` below 0.08 µm/px (high magnification, e.g. the high-speed
single-cell movies), because v1 fragmented the Z- and M-band lines there: on the 0.061 µm/px
20 kPa movie it returns 26 % fewer sarcomere vectors than `legacy`. v2 closes most of that gap
(frames 0 and 250: 4074 and 4160 vectors vs 4443 and 4576 for `legacy`, median length
+0.02 µm), so it is the default there too. Set `model_path='legacy'` to reproduce
high-magnification results of earlier versions, including the paper's high-speed movies.

```python
sarc.detect_sarcomeres(frames=0)                          # generalist (v2)
sarc.detect_sarcomeres(frames=0, model_path='generalist')  # force v2
sarc.detect_sarcomeres(frames=0, model_path='model_sarcomeres_generalist_v1.pt')  # the betas' v1, restored (below)
sarc.detect_sarcomeres(frames=0, model_path='legacy')      # the first model (earlier default below 0.08 µm/px)
```

The resolved alias is stored as `params.detect_sarcomeres.model_path`.

## What changed in v2

v2 is v1 fine-tuned for 20,000 steps (learning rate 3e-5, same scale augmentation), with two changes.

**Simulated myofibrils.** Half of every batch is simulated images with exact labels: myofibrils from
very thin to thick, straight or sheared, in rows of at least three Z-bands, plus what does not count as a
sarcomere and must stay unlabelled: premyofibrils (puncta spaced below 1.4 µm), continuous stress fibres,
isolated Z-body pairs. The real labels come from the wavelet analysis and miss most thin myofibrils.

**Orientation along the myofibril.** Where Z-bands are sheared against their myofibril, the stored label
(and v1) points along the Z-band normal. The training labels were re-derived from line continuity: the
direction in which SarcAsM's line growth stays on its M-band chain longest. The orientation loss weight is
4 (v1: 1).

| | v1 | **v2** |
|---|---|---|
| detection benchmark (Supplementary Note 2), F1 median | 0.798 | **0.833** (better on 10 of 10 images) |
| precision / recall | 0.858 / 0.785 | 0.874 / 0.818 |
| length vs manual line ROIs, median abs. error | 68 nm | 64 nm |
| length SD vs manual SD, mean abs. deviation | **30 nm** | 40 nm |
| thin simulated myofibrils found (150 held-out images) | 0.18 | **0.48** |
| simulated stress fibres detected | 0.14 | 0.10 |
| orientation vs known myofibril axis (3 micropatterns + 1 NRVM lines image, held out) | 9.1° | **7.5°** |
| — where Z-bands tilt > 15° | 9.2° | 7.9° |
| sheared Z-bands on 8 real test images, orientation vs pattern axis | 13.2° | **10.4°** (better on 8 of 8) |
| orientation vs line-continuity label, 25 validation images | 9.3° | 8.7° |

Orientation errors are taken only where both models call sarcomere, so mask size does not leak into them.
At high magnification `legacy` still scores higher on the 0.061 µm/px benchmark image (F1 0.89 vs 0.83 for v2
and 0.77 for v1); v2 is the default there anyway, so that one model serves all data.

**Known limitation: sarcomere precursors.** In immature or noisy cells v2 puts more Z-bands, sarcomere vectors
and sarcomere area on precursors (isolated α-actinin puncta, dense speckle) than v1, which marks some of them
too. On a 12-day hiPSC-CM time course (0.107 µm/px, frames at days 1, 7.6 and 12), vectors outside v1's
sarcomere mask rise from 3–5 to 116–226 per frame (at most 2.3 % of all vectors) and `sarcomere_area_ratio`
by 16–36 %, while the mean sarcomere length moves by at most 0.025 µm. Part of the extra area is real (faint
and thin myofibrils), part is precursors and the dark gaps between adjacent myofibrils. The cause is in the
real Z-band training labels, which were never restricted to sarcomeres.

## What changed in v1.0.0

**The orientation labels were mirrored.** `training_data_generation.create_orientation_map`
wrote the sarcomere angle in the convention the wavelet analysis uses — the axis for a
stored angle `o` is `(sin o, -cos o)` in `(row, col)` — while
`analysis/sarcomere_vectors.get_sarcomere_vectors` reads a predicted angle as
`(sin o, cos o)`, the mirror image. Measured against image geometry (structure tensor,
calibrated on synthetic stripes), the stored labels sat a median 39° from the analysis
convention — chance — while the shipped checkpoint sat at 8.6°. Any model trained on the
uncorrected labels produces an orientation field the analysis reads mirrored: sarcomere
length spreads out and the order parameter collapses, with no training metric noticing,
because the model reproduces its labels perfectly. `create_orientation_map` now stores
`-o`; the generator's own sarcomere mask was already drawn in that convention.

**Scale augmentation.** The training pool's native pixel sizes are clustered — 70 of 126
images within 6 % of 0.11 µm/px, 22 at 0.066, none above 0.1625 — so more than half of
the supported range had no training data in it. Patches are now resized so their effective
pixel size lands log-uniformly in 0.08–0.45 µm, downscaling only (never upsampling, which
would invent detail), and for 60 % of patches, so native-resolution data and the sub-0.10
outliers survive.

**The non-sarcomeric fibre head was removed.** Its class was defined by exclusion, so it
absorbed whatever the sarcomere heads failed to explain: 43.6 % of the label lay within
1 µm of a Z-band, 11.6 % inside the sarcomere mask, and on micropatterned cells 78.1 % was
the cell cortex.

## Not interchangeable

The models measure the same quantities, but not identically.

| | `legacy` | v1 |
|---|---|---|
| length drift, 0.11→0.40 µm/px | 0.346 µm | **0.122 µm** |
| vectors at 0.40 µm/px | 183 | 5885 |
| cell-mask area vs label | −2.2 % | +4.0 % |
| sarcomere length spread | **0.185 µm** | 0.23 µm |

The cell-mask difference shifts the denominator of `sarcomere_area_ratio`, so values are
not comparable across the two. Pick one per study. `legacy` still has the tighter length
spread, most likely because it detects less and what it detects is the easy subset.

The same holds for v1 and v2. v2's cell mask matches v1's (area ratio 1.00, 41 images), but its sarcomere
mask is larger — median +10 % (interquartile range +3 to +18 %) — so `sarcomere_area_ratio` comes out
about 10 % higher, and its sarcomere-length spread is wider (table above). v1 is no longer bundled
(it kept the package under PyPI's 100 MB limit). To reproduce results of 1.0.0b1–b3, restore it with
`git show v1.0.0b3:sarcasm/models/model_sarcomeres_generalist_v1.pt > model_sarcomeres_generalist_v1.pt`
and pass that file as `model_path`.

# 3D Z-band model for high-speed movies

`detect_z_bands_fast_movie` predicts Z-bands from the whole movie at once, for high-speed recordings
(about 6 or more frames per contraction). Only the Z-bands come from it; M-bands and orientation still
come from the 2D model.

| alias | file | notes |
|---|---|---|
| `generalist` | `model_z_bands_unet3d_v2.pt` | default |
| `legacy` | `model_z_bands_unet3d.pt` | shipped before v1.0.0 |

```python
sarc.detect_z_bands_fast_movie()                     # 'generalist'
sarc.detect_z_bands_fast_movie(model_path='legacy')  # the pre-v1.0.0 3D model
```

The resolved alias is stored as `params.detect_z_bands_fast_movie.model_path`.

## What changed in v2

`legacy` saw only one kind of movie. On movies from another microscope it recovers only 22–62 % of the
Z-band pixels the 2D model finds, and on the Fig. 5 movie it drops Z-bands at peak contraction. v2 was trained with `scripts/retrain_unet3d/` on
40 % synthetic movies with exact labels (stills moved by simulated contraction plus rigid and non-rigid
jitter, including low photon counts), 30 % of our own movies and 30 % of movies from two other
microscopes, both labelled frame by frame by the 2D model on flow-denoised frames. It was then fine-tuned
with a temporal penalty (L1 between consecutive frames, weight 3; the checkpoint with the best validation
score, after 2 of 12 epochs). The weight trades trajectory noise against sarcomere popping, which is fast and
physiological. On the Fig. 5 movie, weight 0 / 2 / 3 / 5 / 10 gives a per-frame length noise of
20.5 / 11.3 / 9.1 / 5.5 / 3.5 nm and finds popping in 10.2 / 9.5 / 8.9 / 8.3 / 6.2 % of sarcomere-beats.
Weight 3 is the lowest that is no noisier than `legacy`.

| | `legacy` | **v2** |
|---|---|---|
| synthetic movies, exact labels: Dice, all frames / fastest-moving quarter | 0.21 / 0.15 | **0.58 / 0.51** |
| held-out movies from two other microscopes (2D teacher labels): Dice | 0.54 | **0.89** |
| Fig. 5 movie (Series095, held out), tracked sarcomeres: per-frame length noise | 9.6 nm | 9.1 nm |
| — popping: sarcomere-beats lengthening > 0.25 µm above rest | 7.7 % | **8.9 %** |
| — length steps > 0.2 µm between frames, outside peak contraction, per 1000 | **0.39** | 0.63 |
| — track coverage, median / fragmentation | 0.71 / 1.63 | 0.73 / 1.62 |
| — pool contraction amplitude | 0.093 µm | 0.094 µm |

Per-frame noise is 1.4826 × MAD of the second difference of each track's length, divided by √6, so rare
real events such as popping do not enter it. Trajectories differ between the two models (noise, popping
counts), so pick one per study; the Fig. 5 analysis of the paper used `legacy`.

# Contraction detection model (ContractionNet)

`model_ContractionNet.pt` reads one trace at a time -- a sarcomere length, a Z-band position, a
pole distance -- and gives every frame a probability of being in a contraction, plus the
probabilities of a contraction onset and offset. It is the only ContractionNet bundled.

**Network.** A dilated temporal CNN (`ContractionNet`, `norm='group'`, no attention): 64
filters, a stem and 7 residual blocks with dilations 1–64, 513 frames of context and 175,747
parameters. GroupNorm (8 groups) normalises every frame by statistics pooled over the whole
trace.

**Input: one polarity decision per trace.** The network reads traces whose rest is the high
side. A rule stored in the checkpoint (`polarity_rule`, version 1, `contraction_net.polarity`)
decides which side that is: the Bowley skewness (10/50/90) of the trace minus its 2-s running
median, after a 0.1-s boxcar, taken as `(g(x) - g(-x)) / 2` so that it is exactly odd. The trace
is multiplied by the sign of that score and conditioned by the `q90` rule (the 90th percentile
at 0, the 10–90 % spread as the unit, plus its slope). `x` and `-x` therefore reach the network
as the same array and get the same output, bit for bit on the CPU, where inference runs. A
trace whose score lies within ±0.10, a flat trace and one shorter than 2 s or 32 frames get no
contractions, never the reading of the wrong side; `predict_contractions(...,
return_decision=True)` says why. A caller that knows the rest side can declare it instead
(`polarity=+1` for rest high, as for sarcomere length, which shortens in a contraction).

**Decoding.** Stored in the checkpoint: threshold 0.45 and confidence peak 0.85. A contraction is
a run of frames above 0.45 that reaches 0.85 somewhere; no gap between runs is closed, so
overlapping twitches stay separate contractions. `contraction_net.decoding.decode_contractions`
is the same rule in SarcAsM and CardioTrace; only the minimum duration and what happens to a
contraction cut off by the start or end of the recording differ.

```python
from contraction_net.decoding import decode_contractions, recommended_decoding
from contraction_net.prediction import predict_contractions

model = 'sarcasm/models/model_ContractionNet.pt'
probs, decision = predict_contractions(trace, model, frametime=ft, return_decision=True)
events = decode_contractions(probs[0], ft, **recommended_decoding(model),
                             min_duration_s=0.2)  # [(start, stop), ...] in frames
```

**Training.** 8,000 simulated traces, traces distilled from analysed recordings of an
acute-drug panel, and hand-annotated EHT recordings (each drawn 20 times per epoch). Every
beating trace is presented rest-high; quiet traces come in both signs, as the polarity rule
hands them over. A quarter of the simulated traces are quiet and carry the drift, steps,
outliers and quantisation of beating ones. Overlapping twitches are labelled as separate
contractions, cut at the trough between them over 10 % of the shorter twitch. The loss is BCE
without a Dice term, the weights are those of the last of 60 epochs, and the threshold is the
one whose called duty matches the labelled duty on the development data. Of eight runs that
differ only in the seed, this one has the highest mean event F1 on the development data.

On test recordings that played no part in training or selection, against the
`SymmetrizedContractionNet` of the 1.0.0 betas, both read at their own threshold with a
confidence peak of 0.85 and CardioTrace's decoder settings (contractions of 0.1 s or more, none
within 5 frames of either end) unless noted:

| | 1.0.0 betas | **this model** |
|---|---|---|
| simulated plates (96 wells, 30 fps): beat count within ±1 | 0.76 | **0.94** |
| — high-duty wells (B5, B6, row G; 16 wells) | 0.19 | **1.00** |
| — beats found / called contractions that are beats | 0.66 / 0.99 | **0.98 / 1.00** |
| simulated plates at 15 / 30 / 60 fps (2 plates each): beating wells within ±1 | 0.60–0.62 / 0.64–0.71 / 0.67–0.73 | **0.73–0.80 / 0.93 / 0.98–1.00** |
| real pole-distance plate, 36 clear wells: beat-count deviations from a consensus of 1116 beats, summed | 67.5 | **9.5** |
| real non-beating wells of pole-distance plates (76): false contractions | 0 | 0 (7 with rest declared high) |
| synthetic quiet traces (920, read at both signs): false contractions per reading, rest declared high / polarity rule | 0.91 / 0.91 | **0.37 / 0.07** |
| drug-panel cells, quiescent (7): false contractions | 6 (in 2 cells) | **0** |
| — beating (87): contraction IoU / beats found | 0.891 / 1.000 | 0.890 / 0.998 |

The cells were read as `Motion.predict_contractions` reads an LOI, with SarcAsM's decoder
settings (contractions of 0.2 s or more, those at the ends kept): Z-band rows oriented by the
polarity rule at margin 0, sarcomere-length rows declared rest-high, and the mean of the two
channel medians (the median over the rows of each channel at each frame) as the cell's
probability. With this model, the mean of the two channel means would find 1639 beats and no
false contraction; the larger of the two channel means 1658 beats and 18 false contractions in 3
quiescent cells (betas: 1652 and none; 1650 and 15 in 4). The mean misses beats because the rows
whose noise hides a beat (often single sarcomere lengths) pull it below the 0.85 peak. On 50
hand-annotated LOIs, most from wells used in training, the median finds 890 of 1017
contractions, the mean 610 and the larger 906. A contraction that most rows of both channels
miss is not detected, as in some cells under 1 µM Mavacamten.

**Known limitation: quiet stretches next to beats.** Because GroupNorm pools over the whole
trace, a long quiet stretch is read differently next to beats than on its own. On 36 quiet
segments (correlated noise or drift) followed by beats of three times the noise level, the model
calls 148 contractions in frames 300–1200 of the quiet segments, against 7 when the same
segments stand alone (1.0.0 betas: 149 against 26).

**Earlier models.** The model of the 1.0.0 betas is no longer bundled.
`git show v1.0.0b3:sarcasm/models/model_ContractionNet.pt > model_ContractionNet_b3.pt` restores
it, and `predict_contractions(trace, 'model_ContractionNet_b3.pt')` reads it by path with the
`'symmetric'` conditioning it was trained with; the polarity rule applies only to `'q90'`
checkpoints. Checkpoints older than 1.0, which record no architecture, are rejected.
