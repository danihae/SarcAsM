"""Step 4: compare 3D models (and the per-frame 2D model) on validation clips.

Runs each model through the same prediction path SarcAsM uses
(motion.detect_z_bands_fast_movie) and scores it against the clip labels:

  dice   Dice at 0.5 over all frames
  fast   Dice on the quarter of frames where the label moves most (smoothing shows here)
  tiou   frame-to-frame IoU of the prediction (flicker; the label's own value is 'tiou_label')

    python 4_evaluate.py clips/synthetic_val clips/collab_val \
        --model new=runs/generalist/best.pt --model generalist --model legacy --with_2d 0.108 --out eval.csv

'--model generalist' / '--model legacy' are the 3D models shipped with SarcAsM (default / pre-v1.0.0).
'--with_2d PIXELSIZE' adds the per-frame 2D model (pixel size in µm; synthetic clips take it from
their params.jsonl).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import tifffile
from bio_image_unet.progress import ProgressNotifier

from sarcasm.analysis.detection import detect_z_bands_fast_movie_unet
from sarcasm.training.teacher_labels import MODEL_DIR, zbands_2d
from sarcasm.training.unet3d_trainer import dice_scores

SARCOMERE_LENGTH_UM = 1.85   # turns a synthetic clip's period in px into a pixel size


def frame_iou(pred, threshold=0.5):
    b = pred > threshold
    union = (b[1:] | b[:-1]).sum((1, 2))
    return float(np.nanmedian(np.where(union > 0, (b[1:] & b[:-1]).sum((1, 2)) / np.maximum(union, 1), np.nan)))


def predict_3d(movie, path):
    out = detect_z_bands_fast_movie_unet(movie, model_path=path, model_dir=MODEL_DIR,
                                         progress_notifier=ProgressNotifier.silent_notifier())
    return np.asarray(out['zbands_fast_movie'], np.float32).reshape(movie.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+', help='validation clip folders')
    ap.add_argument('--model', action='append', default=[],
                    help="NAME=CHECKPOINT, or a bundled alias ('generalist', 'legacy')")
    ap.add_argument('--with_2d', type=float, metavar='PIXELSIZE', help='also score the per-frame 2D model')
    ap.add_argument('--max_clips', type=int, default=50, help='per folder')
    ap.add_argument('--out', default='evaluation.csv')
    args = ap.parse_args()
    models = {m.split('=', 1)[0]: (m.split('=', 1)[1] if '=' in m else m) for m in args.model}

    rows = []
    for folder in args.folders:
        params = {}
        if os.path.exists(os.path.join(folder, 'params.jsonl')):
            params = {r['file']: r for r in map(json.loads, open(os.path.join(folder, 'params.jsonl')))}
        for f in sorted(glob.glob(os.path.join(folder, 'volume', '*.tif')))[:args.max_clips]:
            name = os.path.basename(f)
            movie = tifffile.imread(f)
            label = tifffile.imread(os.path.join(folder, 'zbands_fast_movie', name)) / 255
            row = dict(folder=os.path.basename(os.path.normpath(folder)), clip=name, tiou_label=frame_iou(label))
            preds = {n: predict_3d(movie, p) for n, p in models.items()}
            if args.with_2d:
                px = SARCOMERE_LENGTH_UM / params[name]['period_px'] if name in params else args.with_2d
                preds['2d'] = zbands_2d(movie, px)
            for n, pred in preds.items():
                row[f'dice_{n}'], row[f'fast_{n}'] = dice_scores(pred, label)
                row[f'tiou_{n}'] = frame_iou(pred)
            rows.append(row)
            print(name, {k: round(float(v), 3) for k, v in row.items() if isinstance(v, float)}, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby('folder')[[c for c in df.columns if c.split('_')[0] in ('dice', 'fast', 'tiou')]]
          .median().round(3).T.to_string())
