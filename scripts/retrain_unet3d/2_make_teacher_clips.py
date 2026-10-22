"""Step 2: make training clips from real high-speed movies, labelled by the 2D U-Net.

For each movie a few time windows are labelled by the 2D model on flow-denoised frames,
then averaged over neighbouring frames (sarcasm/training/teacher_labels.py), and the
crops with the most Z-bands are saved.

List your movies in a CSV with the columns  path,pixelsize,name  (name: short, unique;
it prefixes the clip files), one movie per line:

    path,pixelsize,name
    /data/our_lab/Series004.tif,0.0659,ours_s004
    /data/collaborator/movie_051.tif,0.108,collab_051

then run

    python 2_make_teacher_clips.py movies.csv OUT [--windows 3 --crops 2]

Keep whole movies out of training for validation: put them in a second CSV and write
them to a separate folder. Writes OUT/volume/, OUT/zbands_fast_movie/, OUT/clips.jsonl.
Movies that already have clips in OUT are skipped, so an interrupted run can be resumed.
"""
import argparse
import csv
import glob
import os

from sarcasm.training.teacher_labels import clips_from_movie

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('movies_csv', help='CSV with columns path,pixelsize,name')
    ap.add_argument('out', help='output folder')
    ap.add_argument('--windows', type=int, default=3, help='time windows per movie')
    ap.add_argument('--crops', type=int, default=2, help='crops per window')
    ap.add_argument('--frames', type=int, default=64, help='frames per clip')
    ap.add_argument('--crop_size', type=int, default=256, help='crop size in px')
    ap.add_argument('--k', type=int, default=2, help='neighbour frames on each side for the teacher')
    args = ap.parse_args()

    with open(args.movies_csv) as f:
        movies = list(csv.DictReader(f))
    for i, m in enumerate(movies):
        if glob.glob(os.path.join(args.out, 'volume', f"{m['name']}_w*")):
            continue
        try:
            files = clips_from_movie(m['path'], args.out, float(m['pixelsize']), m['name'],
                                     n_windows=args.windows, crops_per_window=args.crops,
                                     n_frames=args.frames, crop_size=args.crop_size, k=args.k)
            print(f"[{i + 1}/{len(movies)}] {m['name']}: {len(files)} clips", flush=True)
        except Exception as e:          # one unreadable movie should not stop the batch
            print(f"[{i + 1}/{len(movies)}] {m['name']}: SKIPPED ({e})", flush=True)
