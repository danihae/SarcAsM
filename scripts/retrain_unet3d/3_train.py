"""Step 3: train the 3D U-Net on the clips from steps 1 and 2.

Each --data entry is one domain:  NAME=FOLDER:WEIGHT  (add :real for real movies).
The weight is how often patches come from that folder, independent of its size.

    python 3_train.py --out runs/generalist \
        --data synthetic=clips/synthetic:0.4 \
        --data ours=clips/ours:0.3:real \
        --data collaborators=clips/collab:0.3:real \
        --val synthetic=clips/synthetic_val --val collaborators=clips/collab_val

A GPU is strongly recommended (the default 60 epochs take a few hours on a workstation
GPU; Apple-silicon MPS is ~20x slower). The best checkpoint is runs/.../best.pt; use it with

    sarc.motion.detect_z_bands_fast_movie(model_path='runs/generalist/best.pt')
"""
import argparse

from sarcasm.training.unet3d_trainer import Domain, train


def parse_domain(spec):
    name, rest = spec.split('=', 1)
    parts = rest.split(':')
    return Domain(name=name, folder=parts[0], weight=float(parts[1]) if len(parts) > 1 else 1.0,
                  real=len(parts) > 2 and parts[2] == 'real')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', action='append', required=True, help='NAME=FOLDER:WEIGHT[:real]')
    ap.add_argument('--val', action='append', default=[], help='NAME=FOLDER (validation)')
    ap.add_argument('--out', required=True, help='output folder for checkpoints and log')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--iters', type=int, default=1000, help='batches per epoch')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patch', type=int, nargs=3, default=(48, 128, 128), help='frames rows cols')
    ap.add_argument('--n_filter', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--time_weight', type=float, default=0.0,
                    help='temporal-consistency penalty; >0 makes the model smooth fast motion')
    ap.add_argument('--init', help='start from this checkpoint (fine-tuning)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    best = train([parse_domain(s) for s in args.data], args.out,
                 val_sets=dict(s.split('=', 1) for s in args.val),
                 epochs=args.epochs, iters_per_epoch=args.iters, batch_size=args.batch,
                 patch=tuple(args.patch), n_filter=args.n_filter, lr=args.lr,
                 time_weight=args.time_weight, init=args.init, workers=args.workers)
    print('best checkpoint:', best)
