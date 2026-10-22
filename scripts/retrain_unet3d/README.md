# Retraining the 3D U-Net for high-speed movies

`SarcAsM.detect_z_bands_fast_movie` predicts Z-bands with a 3D U-Net that looks at
several frames at once. It gives steadier Z-band masks than the per-frame 2D model,
but only on data like what it was trained on. These four scripts retrain it on your
own movies, or fine-tune it for them.

**The 3D model is meant for high-speed recordings only**: about 30 frames per second
or faster, so that a contraction spans at least ~6 frames, and high magnification,
with a sarcomere about 10-32 px long. For slower or lower-magnification movies, use
`detect_sarcomeres` (2D).

## Where the labels come from

Nobody labels movies by hand. Two sources are combined:

| | what | labels |
|---|---|---|
| **synthetic clips** (step 1) | single images of the 2D training set, animated: contraction along the myofibrils, drift, frame-to-frame jitter, camera noise | exact (the label is warped with the image) |
| **teacher clips** (step 2) | short windows cut from real high-speed movies | the 2D U-Net, run on flow-denoised frames and averaged over neighbouring frames |

The synthetic clips teach the model about motion and noise it never saw in real movies.
The teacher clips teach it what real data from your microscope look like.

## Steps

```bash
cd scripts/retrain_unet3d

# 1. synthetic clips from the 2D training set (CPU, minutes)
python 1_make_synthetic_clips.py /path/to/training_data_2d clips/synthetic --n 4000
python 1_make_synthetic_clips.py /path/to/training_data_2d clips/synthetic_val --n 100 --split val

# 2. teacher clips from real movies (GPU helps; ~1 min per movie)
#    movies.csv has the columns  path,pixelsize,name
python 2_make_teacher_clips.py movies.csv clips/real
python 2_make_teacher_clips.py movies_val.csv clips/real_val     # whole movies held out

# 3. train (GPU; a few hours)
python 3_train.py --out runs/my_model \
    --data synthetic=clips/synthetic:0.5 \
    --data real=clips/real:0.5:real \
    --val synthetic=clips/synthetic_val --val real=clips/real_val

# 4. compare with the bundled model and with per-frame 2D
python 4_evaluate.py clips/synthetic_val clips/real_val \
    --model mine=runs/my_model/best.pt --model bundled --with_2d 0.1
```

Then use the model:

```python
sarc.detect_z_bands_fast_movie(model_path='runs/my_model/best.pt')
```

## Things worth knowing

* **Weights, not clip counts, set the mix.** `NAME=FOLDER:WEIGHT` in step 3 sets how
  often patches come from each folder. A single collaborator movie with weight 0.2 is
  seen as often as 4000 synthetic clips with weight 0.2.
* **Hold out whole movies** for validation, not windows of training movies; neighbouring
  frames are nearly identical, so a split by window measures memory, not generalisation.
* **Teacher labels are only as good as the 2D model.** Look at a few clips from step 2
  (e.g. in napari) before training. If the 2D model is wrong on your data, the 3D model
  learns the same mistake.
* **Keep `--time_weight 0`** unless you really want smoother masks: the temporal penalty
  makes the model lag behind fast motion. `4_evaluate.py` reports Dice on the
  fastest-moving frames (`fast_*`) so you can see this.
* **Fine-tuning** from the bundled or your own model: `3_train.py --init model.pt --epochs 15 --lr 1e-4`.

Code: `sarcasm/training/` (`synthetic_motion.py`, `teacher_labels.py`, `unet3d_trainer.py`).
