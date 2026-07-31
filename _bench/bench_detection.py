"""Benchmark the U-Net inference paths behind detect_sarcomeres / detect_z_bands_fast_movie.

Usage:
    python _bench/bench_detection.py                      # run the default cases
    python _bench/bench_detection.py --cases 512 2000      # subset
    python _bench/bench_detection.py --save-ref out/base   # store masks for later comparison
    python _bench/bench_detection.py --check-ref out/base  # compare masks against a stored run

Reports, per case: wall clock, per-stage breakdown (preprocess / split / predict /
stitch), the tile grid and how many pixels it processes relative to the image
("redundancy"), and peak RSS. The stage breakdown is obtained by wrapping the
private methods of the ``Predict`` classes, so the script works unchanged
against old and new versions of bio-image-unet.

--save-ref / --check-ref exist to prove that a speed change did not move the
output: tiling changes shift tile boundaries, so the bar is Dice >= 0.999 at
threshold 0.5 plus a small absolute tolerance on the probability maps, not bit
equality.
"""
import argparse
import os
import resource
import sys
import time

import numpy as np
import tifffile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bio_image_unet.multi_output_unet.multi_output_nested_unet import MultiOutputNestedUNet_3Levels
from bio_image_unet.progress import ProgressNotifier
import bio_image_unet.multi_output_unet.predict as predict2d
import bio_image_unet.multi_output_unet3d.predict as predict3d

from sarcasm.utils import Utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_2D = os.path.join(ROOT, 'sarcasm', 'models', 'model_sarcomeres_generalist.pt')
MODEL_3D = os.path.join(ROOT, 'sarcasm', 'models', 'model_z_bands_unet3d.pt')
DATA = os.path.join(ROOT, 'test_data')

STAGES = ('preprocess', 'split', 'predict', 'stitch')


def peak_rss_gb():
    """Peak resident set size of this process, in GB (macOS reports bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e9 if sys.platform == 'darwin' else ru / 1e6


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


# Wrappers accumulate here; run_case clears it before each run. A single shared
# dict keeps the instrumentation idempotent across the several Predict classes.
_TIMINGS = {}


def instrument(cls, device):
    """Wrap the private stage methods of a Predict class so they record into _TIMINGS."""
    for stage in STAGES:
        attr = f'_{cls.__name__}__{stage}'
        original = getattr(cls, attr, None)
        if original is None or getattr(original, '_benched', False):
            continue

        def make(orig, key):
            def wrapper(self, *args, **kwargs):
                sync(device)
                t0 = time.perf_counter()
                out = orig(self, *args, **kwargs)
                sync(device)
                _TIMINGS[key] = _TIMINGS.get(key, 0.0) + time.perf_counter() - t0
                return out
            wrapper._benched = True
            return wrapper

        setattr(cls, attr, make(original, stage))
    return _TIMINGS


# --------------------------------------------------------------------------- cases

def _tile_to(img, shape):
    """Tile/crop a 2D frame up to ``shape`` so synthetic sizes keep realistic content."""
    reps = [int(np.ceil(s / i)) for s, i in zip(shape, img.shape)]
    return np.tile(img, reps)[:shape[0], :shape[1]]


def case_2d_512(n=30):
    return tifffile.imread(os.path.join(DATA, 'CM_sarc_graph_2019', 'real_data_E5_frame0_to29.tif'))[:n]


def case_2d_1200(n=8):
    """1200x1200 -- the worst case for the current tiling (image just over one tile)."""
    src = tifffile.imread(os.path.join(DATA, 'CM_sarc_graph_2019', 'real_data_E5_frame0_to29.tif'))[:n]
    return np.stack([_tile_to(f, (1200, 1200)) for f in src])


def case_2d_2000(n=30):
    path = os.path.join(DATA, 'long_term_2D_ACTN2-citrine_CM', '20211115_ACTN2_CMs_96well_control_12days.tif')
    return tifffile.imread(path, key=range(n))


def case_3d(n=64):
    return tifffile.imread(os.path.join(DATA, '_tracking_validation', '053_crop.tif'))[:n]


def case_3d_small():
    """(30, 200, 512): depth and height are below the patch size and not multiples of 8."""
    return tifffile.imread(os.path.join(DATA, '_tracking_validation', 'fast_crop.tif'))


CASES = {
    '512':      dict(loader=case_2d_512,   kind='2d', label='2D  30 x 512^2'),
    '1200':     dict(loader=case_2d_1200,  kind='2d', label='2D   8 x 1200^2'),
    '2000':     dict(loader=case_2d_2000,  kind='2d', label='2D  30 x 2000^2'),
    '3d':       dict(loader=case_3d,       kind='3d', label='3D  64 x 428x472'),
    '3d_small': dict(loader=case_3d_small, kind='3d', label='3D  30 x 200x512'),
}
DEFAULT_CASES = ['512', '1200', '2000', '3d', '3d_small']


# --------------------------------------------------------------------------- run

def run_case(name, device, max_patch_2d='auto', max_patch_3d='auto'):
    spec = CASES[name]
    imgs = spec['loader']()
    kind = spec['kind']

    if kind == '2d':
        cls = predict2d.Predict
        timings = instrument(cls, device)
        timings.clear()
        t0 = time.perf_counter()
        pred = cls(imgs, model_params=MODEL_2D, result_path=None,
                   max_patch_size=Utils.check_and_round_max_patch_size(max_patch_2d),
                   normalization_mode='all', network=MultiOutputNestedUNet_3Levels,
                   clip_threshold=(0., 99.98), device=device,
                   progress_notifier=ProgressNotifier.silent_notifier())
        total = time.perf_counter() - t0
        n_frames = imgs.shape[0] if imgs.ndim == 3 else 1
        grid = (pred.N_x, pred.N_y)
        processed = pred.N_x * pred.patch_size[0] * pred.N_y * pred.patch_size[1]
        redundancy = processed / float(np.prod(imgs.shape[-2:]))
    else:
        cls = predict3d.Predict
        timings = instrument(cls, device)
        timings.clear()
        t0 = time.perf_counter()
        pred = cls(imgs, model_params=MODEL_3D, result_path=None,
                   max_patch_size=Utils.check_and_round_max_patch_size(max_patch_3d),
                   normalization_mode='all', clip_threshold=(0., 99.8), device=device,
                   progress_notifier=ProgressNotifier.silent_notifier())
        total = time.perf_counter() - t0
        n_frames = imgs.shape[0]
        grid = (pred.N_z, pred.N_y, pred.N_x)
        processed = int(np.prod(grid)) * int(np.prod(pred.patch_size))
        redundancy = processed / float(np.prod(imgs.shape))

    return dict(name=name, label=spec['label'], total=total, per_frame=total / n_frames * 1000,
                timings=dict(timings), patch=tuple(pred.patch_size), grid=grid,
                redundancy=redundancy, result=pred.result, shape=imgs.shape)


def report(rows):
    print()
    header = f"{'case':<20} {'ms/frame':>10} {'total s':>9} {'tiles':>12} {'redund':>7}   stage breakdown"
    print(header)
    print('-' * len(header))
    for r in rows:
        stages = '  '.join(f"{s}={r['timings'].get(s, 0.0):.2f}s"
                           f"({100 * r['timings'].get(s, 0.0) / r['total']:.0f}%)" for s in STAGES)
        grid = 'x'.join(str(g) for g in r['grid'])
        print(f"{r['label']:<20} {r['per_frame']:>10.0f} {r['total']:>9.1f} "
              f"{grid + ' @' + 'x'.join(str(p) for p in r['patch']):>12} "
              f"{r['redundancy']:>6.2f}x   {stages}")
    print(f"\npeak RSS: {peak_rss_gb():.2f} GB")


def save_ref(rows, path):
    os.makedirs(path, exist_ok=True)
    for r in rows:
        for key, arr in r['result'].items():
            np.save(os.path.join(path, f"{r['name']}__{key}.npy"), np.asarray(arr, dtype=np.float32))
    print(f"\nreference masks written to {path}")


def check_ref(rows, path, atol=1e-3, dice_min=0.999):
    print(f"\nchecking against {path}")
    ok = True
    for r in rows:
        for key, arr in r['result'].items():
            f = os.path.join(path, f"{r['name']}__{key}.npy")
            if not os.path.exists(f):
                print(f"  {r['name']}/{key}: MISSING reference")
                continue
            ref = np.load(f)
            new = np.asarray(arr, dtype=np.float32)
            if ref.shape != new.shape:
                print(f"  {r['name']}/{key}: SHAPE {ref.shape} -> {new.shape}")
                ok = False
                continue
            a, b = ref > 0.5, new > 0.5
            denom = a.sum() + b.sum()
            dice = 1.0 if denom == 0 else 2 * (a & b).sum() / denom
            max_abs = float(np.abs(ref - new).max())
            good = dice >= dice_min and max_abs <= atol
            ok &= good
            print(f"  {'ok ' if good else 'FAIL'} {r['name']}/{key}: Dice={dice:.5f} max|d|={max_abs:.2e}")
    print('\nALL PASS' if ok else '\nFAILURES PRESENT')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='*', default=DEFAULT_CASES, choices=list(CASES))
    ap.add_argument('--device', default=None)
    ap.add_argument('--save-ref', default=None)
    ap.add_argument('--check-ref', default=None)
    ap.add_argument('--patch-2d', default=None, help="e.g. '1024,1024' to override auto sizing")
    ap.add_argument('--patch-3d', default=None, help="e.g. '32,256,256' to override auto sizing")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else Utils.get_device()
    print(f"device: {device}   torch {torch.__version__}")

    rows = []
    for name in args.cases:
        try:
            kw = {}
            if args.patch_2d:
                kw['max_patch_2d'] = tuple(int(v) for v in args.patch_2d.split(','))
            if args.patch_3d:
                kw['max_patch_3d'] = tuple(int(v) for v in args.patch_3d.split(','))
            rows.append(run_case(name, device, **kw))
            print(f"  ran {name}: {rows[-1]['per_frame']:.0f} ms/frame")
        except Exception as exc:  # a crashing case is itself a result worth seeing
            print(f"  {name}: {type(exc).__name__}: {exc}")

    report(rows)
    if args.save_ref:
        save_ref(rows, args.save_ref)
    if args.check_ref:
        sys.exit(0 if check_ref(rows, args.check_ref) else 1)


if __name__ == '__main__':
    main()
