# Sarcomere detection models

| alias | file | validated pixel size | notes |
|---|---|---|---|
| `generalist` | `model_sarcomeres_generalist_v1.pt` | 0.08–0.45 µm | scale-augmented (v1.0.0) |
| `legacy` | `model_sarcomeres_generalist.pt` | 0.06–0.35 µm | pre-v1.0.0; trained on high-magnification data at native resolution |

The default, `model_path='auto'`, picks by pixel size: **`legacy` below 0.08 µm/px**
(high-magnification, e.g. the high-speed single-cell movies), `generalist` otherwise.
v1's scale augmentation only ever downscales into 0.08–0.45 µm/px, so it never saw
high-magnification frames at native resolution; on the 0.061 µm/px 20 kPa movie it
fragments the Z- and M-band lines and returns 28 % fewer sarcomere vectors than `legacy`
(3119 vs 4340 on frame 0), and rescaling it into its range (`rescale_factor=0.33`)
recovers only part of that (4036) at the cost of Z-band localisation precision.

```python
sarc.detect_sarcomeres(frames=0)                          # 'auto': by pixel size
sarc.detect_sarcomeres(frames=0, model_path='generalist')  # force v1
sarc.detect_sarcomeres(frames=0, model_path='legacy')      # force the pre-v1.0.0 model
```

The resolved alias is stored as `params.detect_sarcomeres.model_path`.

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

Both models measure the same quantities, but not identically.

| | `legacy` | `generalist` (v1) |
|---|---|---|
| length drift, 0.11→0.40 µm/px | 0.346 µm | **0.122 µm** |
| vectors at 0.40 µm/px | 183 | 5885 |
| cell-mask area vs label | −2.2 % | +4.0 % |
| sarcomere length spread | **0.185 µm** | 0.23 µm |

The cell-mask difference shifts the denominator of `sarcomere_area_ratio`, so values are
not comparable across the two. Pick one per study. `legacy` still has the tighter length
spread, most likely because it detects less and what it detects is the easy subset.
