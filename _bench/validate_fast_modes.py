"""Score the opt-in fast prediction modes against annotated ground truth.

Usage:
    python _bench/validate_fast_modes.py                       # prune levels 3 and 2
    python _bench/validate_fast_modes.py --rescale 1.0 0.7     # also vary rescale_factor
    python _bench/validate_fast_modes.py --n 40                # fewer images, faster

``prune_level`` and ``rescale_factor`` both trade accuracy for speed, so neither
should be turned on without knowing what it costs. Comparing a fast mode against
the full model would only say how far it moved, not whether it moved the wrong
way -- so this scores every mode against the manual annotations in
``training_data/`` instead, and reports Dice per output head alongside ms/frame.

Note that the images in ``training_data/`` were used to train the bundled model,
so absolute scores are optimistic. The comparison between modes is what matters.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import tifffile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bio_image_unet.multi_output_unet.multi_output_nested_unet import MultiOutputNestedUNet_3Levels
from bio_image_unet.multi_output_unet.predict import Predict
from bio_image_unet.progress import ProgressNotifier

from sarcasm.utils import Utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'sarcasm', 'models', 'model_sarcomeres_generalist.pt')
GT = os.path.join(ROOT, 'training_data')

# Head -> (ground-truth folder, how to binarise the annotation)
HEADS = {
    'zbands': ('zbands', lambda a: a > 0.5),
    'mbands': ('mbands', lambda a: a > 0.5),
    'sarcomere_mask': ('sarcomere_mask', lambda a: a.astype(bool)),
    'cell_mask': ('cell_mask', lambda a: a.astype(bool)),
}


def load_pairs(limit):
    """Images annotated for every head, as (name, image, {head: mask}).

    The annotation set is not uniform in size, so this keeps only the most common
    shape -- enough images to compare modes, and it lets the whole set be
    predicted as one stack.
    """
    candidates = []
    for image_path in sorted(glob.glob(os.path.join(GT, 'image', '*.tif'))):
        name = os.path.basename(image_path)
        truths = {}
        for head, (folder, binarise) in HEADS.items():
            path = os.path.join(GT, folder, name)
            if not os.path.exists(path):
                break
            truths[head] = binarise(tifffile.imread(path))
        else:
            image = tifffile.imread(image_path)
            if all(m.shape == image.shape for m in truths.values()):
                candidates.append((name, image, truths))

    shapes = {}
    for entry in candidates:
        shapes.setdefault(entry[1].shape, []).append(entry)
    if not shapes:
        return []
    best = max(shapes.values(), key=len)
    dropped = len(candidates) - len(best)
    if dropped:
        print(f'  using the {len(best)} images of shape {best[0][1].shape}; '
              f'{dropped} of other shapes skipped')
    return best[:limit]


def dice(a, b):
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * np.logical_and(a, b).sum() / denom


def run_mode(images, device, prune_level, rescale_factor):
    """Predict a stack under one mode; returns (results, seconds per frame)."""
    from skimage.transform import rescale as sk_rescale

    stack = np.stack(images)
    if rescale_factor != 1.0:
        stack = sk_rescale(stack, (1.0, rescale_factor, rescale_factor), order=0,
                           mode='reflect', preserve_range=True,
                           channel_axis=None).astype(stack.dtype)
    t0 = time.perf_counter()
    result = Predict(stack, model_params=MODEL, result_path=None,
                     network=MultiOutputNestedUNet_3Levels, normalization_mode='single',
                     clip_threshold=(0., 99.98), device=device, prune_level=prune_level,
                     progress_notifier=ProgressNotifier.silent_notifier()).result
    elapsed = time.perf_counter() - t0

    if rescale_factor != 1.0:
        from skimage.transform import resize
        target = images[0].shape
        result = {k: np.stack([resize(f, target, order=0, preserve_range=True, anti_aliasing=False)
                               for f in v]) for k, v in result.items() if k in HEADS}
    return result, elapsed / len(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prune', nargs='*', type=int, default=[3, 2, 1])
    ap.add_argument('--rescale', nargs='*', type=float, default=[1.0])
    ap.add_argument('--n', type=int, default=60, help='number of annotated images to score')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else Utils.get_device()
    pairs = load_pairs(args.n)
    if not pairs:
        raise SystemExit(f'no fully annotated images found under {GT}')
    print(f'device: {device}   scoring {len(pairs)} annotated images from training_data/')

    images = [p[1] for p in pairs]
    truths = {head: np.stack([p[2][head] for p in pairs]) for head in HEADS}

    header = f"{'mode':<22} {'ms/frame':>9} " + ' '.join(f'{h:>15}' for h in HEADS)
    print('\n' + header)
    print('-' * len(header))
    for rescale_factor in args.rescale:
        for prune_level in args.prune:
            try:
                result, per_frame = run_mode(images, device, prune_level, rescale_factor)
            except Exception as exc:
                print(f"prune={prune_level} rescale={rescale_factor}: {type(exc).__name__}: {exc}")
                continue
            scores = ' '.join(f'{dice(result[h] > 0.5, truths[h]):>15.4f}' for h in HEADS)
            label = f'prune={prune_level} rescale={rescale_factor}'
            print(f'{label:<22} {per_frame * 1000:>9.0f} {scores}')
    print('\nDice against manual annotations, per output head (higher is better).')


if __name__ == '__main__':
    main()
