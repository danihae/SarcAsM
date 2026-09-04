# ContractionNet: drug corpus, polarity invariance and architecture comparison

> **Shipped.** `sarcasm/models/model_ContractionNet.pt` is now
> `SymmetrizedContractionNet(norm='group', attention='post')` trained on the simulated set
> plus a corpus distilled from the acute-drug panel, at threshold 0.45.
>
> The checkpoint it replaced is recoverable from git
> (`git show fc05f00:sarcasm/models/model_ContractionNet.pt`), and the corpus builder, the
> comparison harness, the restored pre-1.0 arms and the gallery scripts that produced the
> numbers below live on the `feature/contraction-net-drug-corpus` branch rather than in the
> package.


Branch `feature/contraction-net-drug-corpus`. Every arm trains through the same loop and
is scored by the same probes (`python -m contraction_net.compare`). Raw tables in
`runs_v3/`; the ground-truth study that preceded it is in `runs_v2/gallery/`.

## Ground truth

Individual traces are labelled from their cell, not from themselves, so that per-group
error averages out. Arriving at a usable definition took four iterations, each fixing a
failure the previous one hid:

| rule | IoU vs its own myofibrils | cells collapsing to all-zero |
|---|---|---|
| vote on per-myofibril masks | 0.858\* | 19% |
| **mean of probabilities** | 0.860\* | 16% |
| network on the mean | 0.800 | 3% |
| amplitude on the mean | 0.689 | 0% |
| *sibling myofibril (asynchrony floor)* | *0.731* | |

\* self-fulfilling: the vote is derived from the masks it is scored against.

The final definition averages the per-myofibril probabilities and thresholds at 0.4, with
these filters:

- **myofibrils of more than eight sarcomeres**, at least two per cell
- **beating decided by rhythm, not amplitude**: autocorrelation `r >= 0.4` keeps 97% of
  Control while admitting 6% of fully suppressed cells; an amplitude floor of 0.06 µm
  keeps 89% and still admits 35%
- **no rhythm splits two ways on amplitude**: below 0.06 µm nothing is moving and the cell
  is a genuine quiescent negative; above it the cell contracts erratically and is dropped.
  12% of the panel was previously being labelled flat while visibly contracting.
- **regularity**: at least five contractions, beat-period CV at most 0.20
- **within-myofibril dispersion** as corroborating evidence, added upward at weight 0.30.
  The inter-quartile range across a chain runs 1.3-1.5x its resting value during
  contraction. It is added, never averaged: dispersion is a weak per-frame discriminator
  (IoU 0.24 alone), so averaging it in costs detections while adding it can only support.

Result: 4333 traces from 64 wells; about 45% of cells labelled, 7% kept as quiescent
negatives, 47% dropped.

## Results

Synthetic stress set has exact labels and is the primary metric. `upright` is the
downward-only set; `mixed` also contains mirrored traces, which is what
`Motion.predict_contractions` actually feeds the network.

| arm | seeds | upright | mixed | quiescent FP | polarity disagr. | held-out real |
|---|---|---|---|---|---|---|
| A0 bundled 1.0 model | 1 | 0.709 | 0.524 | **0.038** | 0.466 | 0.787 |
| pre-1.0 recipe, end to end | 1 | 0.570 | 0.595 | 0.855 | 0.149 | 0.801 |
| pre-1.0 backbone, 1.0 recipe | 1 | 0.681 | 0.680 | 0.242 | 0.132 | 0.782 |
| 1.0 architecture (BatchNorm) | 3 | 0.644 | 0.671 | 0.109 | 0.062 | 0.788 |
| GroupNorm | 3 | 0.747 | 0.763 | 0.137 | 0.071 | 0.814 |
| GroupNorm + attention | 3 | 0.740 | 0.765 | 0.123 | 0.078 | 0.821 |
| **symmetrized + GroupNorm + attention** | **3** | **0.753** | **0.774** | 0.102 | **0.000** | **0.828** |
| best trunk, no drug corpus | 1 | 0.716 | 0.759 | 0.071 | 0.067 | 0.468 |

Held-out real = agreement with the cell label on 13 of 64 wells the model never trained
on. Circular by construction, so read the per-type split:

| arm | group slen | track slen | track z-pos |
|---|---|---|---|
| symmetrized + GroupNorm + attention | 0.861 | **0.816** | **0.818** |
| A0 bundled 1.0 model | 0.983 *(identity)* | 0.711 | 0.716 |
| best trunk, no drug corpus | 0.632 | 0.283 | 0.523 |

## Where the attention layer belongs

| position | upright | quiescent FP | high duty | onset F1 |
|---|---|---|---|---|
| none | 0.747 | 0.187 | 0.836 | 0.604 |
| pre | 0.733 | 0.110 | 0.760 | 0.608 |
| mid | 0.739 | **0.088** | 0.770 | **0.733** |
| post | 0.742 | 0.155 | 0.678 | 0.655 |
| both | **0.749** | 0.146 | 0.806 | 0.609 |

Position is close to a null result on IoU: the spread, 0.733-0.749, sits inside the
seed-to-seed noise of 0.010-0.026. `mid` stands out elsewhere -- half the quiescent false
positives of no-attention and the best onset F1, which matters because beating rate and
cycle duration are computed from that head.

**Correcting an earlier claim in this file:** attention was credited with a gain of about
0.09. Comparing across corpora, that gain belongs to **GroupNorm** (BatchNorm 0.644 ->
GroupNorm 0.747); attention adds little to IoU on top of it, though it does help the
boundary heads.

## Four findings

**1. The drug corpus buys real-trace agreement, not synthetic score.** The best trunk
without it scores 0.716 upright, close to the 0.752 with it, but its agreement on held-out
individual tracks collapses from 0.816 to 0.283. The simulator sets synthetic performance;
the corpus is what anchors the model to real traces.

**2. Polarity invariance is nearly free and worth a great deal.** Ablating it scores 0.664
upright against 0.644 for the same architecture with it, while mixed-polarity performance
runs 0.524 (bundled model) against 0.774. Making the invariance architectural rather than
learned costs nothing and takes disagreement from 7% to exactly zero.

**3. The 1.0 rewrite was justified; its architecture was not optimal.** The full pre-1.0
recipe is the worst arm (0.570, and 85% false positives on quiescent traces) and the
pre-1.0 backbone under the modern recipe is still behind the 1.0 one. But replacing
BatchNorm with GroupNorm gains 0.10, which contradicts the reasoning in the
`ContractionNet` docstring: GroupNorm does pool over the time axis, yet it beats BatchNorm
at high duty (0.717 vs 0.652), exactly where that argument predicts it should fail.

**4. The stricter ground truth improved real-trace agreement without changing synthetic
scores**, as expected from (1): held-out `track_slen` went 0.789 -> 0.816 and `track_z_pos`
0.762 -> 0.818 between corpus versions, while upright stress moved 0.758 -> 0.752, inside
noise.

## How long a contraction is called

At its originally tuned threshold the recommended model ran contractions 3.7% short on real
traces. That was a tuning artefact rather than a limit of the model: IoU is nearly flat from
0.35 to 0.55 and barely registers a few per cent of duration bias, so maximising it alone
picks a point in that band close to arbitrarily.

| threshold | upright IoU | quiescent FP | real interval, vs truth |
|---|---|---|---|
| 0.40 | 0.764 | 0.104 | 0.995 |
| **0.45 (refitted, in use)** | 0.763 | 0.080 | 0.979 |
| 0.50 (original) | 0.759 | 0.060 | 0.963 |

`Trainer.tune_threshold` now breaks near-ties in IoU towards the threshold whose predicted
duty matches the target, and every checkpoint was refitted on its own held-out calibration
recordings (`scripts/retune_threshold.py`). Mean interval on held-out real traces went from
0.493 s to 0.516 s against a target of 0.527 s, and seed-to-seed variance tightened -- the
1.0-architecture arm's spread fell from 0.026 to 0.002 -- because thresholds no longer land
arbitrarily inside the flat band. The bundled model shares the same bias, worse, at 0.936.

**The checkpoints keep the tuned 0.45.** It recovers 89% of the duration gap against the
bundled model -- 0.987 against 0.883, target 1.0 -- while holding 26% fewer quiescent false
positives than 0.40. Dropping to 0.40 buys the last 1.3% of duration and pays for it on
non-beating cells, which is the wrong side of that trade when quiescent specificity is
already this model's weak point, and 1.3% sits far below the measurement noise of
`time_contr` and its relatives.

## Is it an improvement on the bundled model?

On five axes clearly, on two not. Measured at each model's own operating point, the
recommended model at 0.40 against the bundled model at 0.50:

| | bundled 1.0 | recommended @0.45 | |
|---|---|---|---|
| synthetic, upright | 0.710 | **0.760** | better |
| synthetic, mixed polarity | 0.517 | **0.791** | better |
| polarity disagreement | 0.447 | **0.000** | better |
| held-out real traces | 0.789 | **0.845** | better |
| interval duration vs truth | 0.883 | **0.987** | better |
| quiescent false positives | **0.042** | 0.091 | **worse** |
| duty called on genuinely flat held-out cells | **0.063** | 0.183 | **worse** |
| onset F1 | **0.804** | 0.627 | **worse** |

The quiescent row is the one that matters operationally: on non-beating held-out cells the
recommended model marks 18.3% of frames as contracting against the bundled model's 6.3%.
Those are exactly the
suppressed Mavacamten and Verapamil cells the panel exists to measure, so spurious
contractions there are not a cosmetic problem.

## Attention halfway up: a rejected arm, and why

Attention in the `mid` position looked promising at one seed -- onset F1 0.733 against 0.623
for the incumbent -- so it was rerun at three seeds together with a symmetrized variant.
Neither claim survived, and the arm was rejected on a failure the aggregate metrics hide.

The onset advantage was noise: at three seeds `mid` gives 0.661 and the symmetrized version
0.622, against the incumbent's 0.623. **Attention position does not fix the boundary heads.**

What did replicate was a gain on realistic traces -- symmetrized `mid` scores 0.781 upright
against 0.753, with half the seed variance -- and a loss on the controlled duty x duration
grid, 0.819 against 0.921 at low duty. The grid explains both:

```
                                     probability across one 127-frame contraction
  incumbent   0.93 1.00 0.99 0.99 0.99 0.99 0.98 0.99 0.96 1.00 0.65   mean 0.98
  mid         0.90 0.99 0.23 0.07 0.02 0.03 0.04 0.07 0.18 1.00 0.97   mean 0.37
```

`mid` holds the onset, collapses to 0.02 across the plateau, and recovers at the offset: it
marks contraction *edges* rather than contraction *state*, so every sustained contraction is
split in two -- 10 predicted intervals where there are 5, duty 0.42 against 0.64, consistent
across seeds.

The aggregate metrics miss this because real drug traces have a median interval of about
0.53 s, some 32 frames, while the failure only appears from roughly 80 frames upward. The
regime simply does not occur in the corpus, so `mid`'s advantage is earned entirely outside
the range where it breaks. It would trigger on exactly the cases worth trusting: Omecamtiv's
prolonged systole, and any tonic cell. Splitting one contraction in two doubles `n_contr`,
and with it the beating rate, and halves `time_contr`.

`evaluate` now reports `fragmentation` -- predicted intervals per true interval on sustained
contractions -- and `plateau_prob`, the mean probability across the middle half of one. The
incumbent scores 1.000 and 0.991, the bundled model 1.000 and 1.000, symmetrized `mid` 1.556
and 0.413. A model that marks edges instead of state cannot pass silently again.

## Remaining regressions against the bundled model

Quiescent false positives, 0.102 against 0.038, and onset F1, 0.623 against 0.804. The
`mid` attention position recovers much of the boundary gap (0.733) and halves the
false-positive rate, and is the first thing to try at more seeds.

## Reproducing

The corpus builder and the arm-comparison harness are on the
`feature/contraction-net-drug-corpus` branch:

```
git checkout feature/contraction-net-drug-corpus
python -m contraction_net.drug_dataset --out training_data/drug_corpus_v3.npz
bash scripts/run_contraction_arms_v5.sh
python scripts/held_out_report.py
```

The package itself keeps only what is needed to run the model, retrain it on a corpus in
the layout `ContractionDataset.load_corpus` reads, and score the result.
