"""Step 1: make synthetic training clips from the 2D training set.

Every clip animates one 2D training image (contraction, drift, jitter, camera noise) and
carries its exact Z-band label in every frame. See sarcasm/training/synthetic_motion.py.

The 2D training set folder must contain same-named TIFFs in image/, zbands/,
orientation/ and sarcomere_mask/. Images are split into train and validation by a hash
of their name, so the two splits never share a source image.

    python 1_make_synthetic_clips.py TRAINING_SET_2D OUT --n 4000
    python 1_make_synthetic_clips.py TRAINING_SET_2D OUT_VAL --n 100 --split val

Writes OUT/volume/*.tif, OUT/zbands_fast_movie/*.tif and OUT/params.jsonl.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tifffile

from sarcasm.training.synthetic_motion import make_clip

PARTS = ('image', 'zbands', 'orientation', 'sarcomere_mask')


def in_validation_split(name):
    """About one image in seven goes to validation, fixed by its name."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 7 == 0


def make_one(job):
    training_set, out, name, index, seed, n_frames, size = job
    arrays = [tifffile.imread(os.path.join(training_set, part, name)) for part in PARTS]
    arrays = [a[0] if a.ndim == 3 else a for a in arrays]          # first plane of a stack
    clip = make_clip(*arrays, n_frames=n_frames, size=size, rng=np.random.default_rng([seed, index]))
    fname = f'synthetic_{index:05d}.tif'
    tifffile.imwrite(os.path.join(out, 'volume', fname), clip.movie, compression='zlib')
    tifffile.imwrite(os.path.join(out, 'zbands_fast_movie', fname), clip.label, compression='zlib')
    params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in clip.params.items()}
    return dict(file=fname, source=name, **params)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('training_set', help='2D training set folder (image/, zbands/, orientation/, sarcomere_mask/)')
    ap.add_argument('out', help='output folder')
    ap.add_argument('--n', type=int, default=4000, help='number of clips')
    ap.add_argument('--split', choices=['train', 'val'], default='train')
    ap.add_argument('--frames', type=int, default=64, help='frames per clip')
    ap.add_argument('--size', type=int, default=192, help='frame size in px (square)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()

    names = sorted(set.intersection(*(set(os.listdir(os.path.join(args.training_set, p))) for p in PARTS)))
    names = [n for n in names if in_validation_split(n) == (args.split == 'val')]
    print(f'{len(names)} source images in the {args.split} split -> {args.n} clips')
    for sub in ('volume', 'zbands_fast_movie'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    jobs = [(args.training_set, args.out, names[i % len(names)], i, args.seed, args.frames,
             (args.size, args.size)) for i in range(args.n)]
    with ProcessPoolExecutor(args.workers) as pool, open(os.path.join(args.out, 'params.jsonl'), 'a') as log:
        for i, record in enumerate(pool.map(make_one, jobs, chunksize=4)):
            log.write(json.dumps(record) + '\n')
            if i % 200 == 0:
                print(f'{i}/{args.n}', flush=True)
    print('done')
