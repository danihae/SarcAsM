# Sarcomere detection models

| alias | file | validated pixel size | notes |
|---|---|---|---|
| `generalist` | `model_sarcomeres_generalist_v2.pt` | 0.08–0.45 µm | v1 fine-tuned on simulated myofibrils and myofibril-axis orientation labels |
| `generalist_v1` | `model_sarcomeres_generalist_v1.pt` | 0.08–0.45 µm | scale-augmented (v1.0.0 beta releases) |
| `legacy` | `model_sarcomeres_generalist.pt` | 0.06–0.35 µm | pre-v1.0.0; trained on high-magnification data at native resolution |

The default, `model_path='auto'`, picks by pixel size: **`legacy` below 0.08 µm/px**
(high-magnification, e.g. the high-speed single-cell movies), `generalist` otherwise.
The generalists' scale augmentation only ever downscales into 0.08–0.45 µm/px, so it never saw
high-magnification frames at native resolution; on the 0.061 µm/px 20 kPa movie it
fragments the Z- and M-band lines and returns 28 % fewer sarcomere vectors than `legacy`
(3119 vs 4340 on frame 0), and rescaling it into its range (`rescale_factor=0.33`)
recovers only part of that (4036) at the cost of Z-band localisation precision.

```python
sarc.detect_sarcomeres(frames=0)                          # 'auto': by pixel size
sarc.detect_sarcomeres(frames=0, model_path='generalist')  # force v2
sarc.detect_sarcomeres(frames=0, model_path='generalist_v1')  # reproduce results of the v1.0.0 betas
sarc.detect_sarcomeres(frames=0, model_path='legacy')      # force the pre-v1.0.0 model
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
v2 does not replace `legacy` at high magnification: on the 0.061 µm/px benchmark image F1 is 0.89
(`legacy`), 0.83 (v2), 0.77 (v1), so `'auto'` still picks `legacy` below 0.08 µm/px.

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

| | `legacy` | `generalist_v1` |
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
about 10 % higher, and its sarcomere-length spread is wider (table above). Results of the v1.0.0 betas
reproduce with `model_path='generalist_v1'`.
