"""Batch motion analysis: track every sarcomere in every movie of a folder.

The v1.0.0 pipeline is

    detect_sarcomeres -> analyze_sarcomere_vectors -> track_sarcomere_vectors
    -> analyze_track_motion(by=...)

Tune the parameters on one or two representative movies first, see
``docs/notebooks/tutorial_sarcomere_tracking.ipynb``. Run this as a script rather
than in a notebook -- ``multiprocessing`` needs the ``__main__`` guard below.
"""
import glob
import os
from multiprocessing import Pool

from sarcasm import SarcAsM, BatchExport

# --- what to analyze ---------------------------------------------------------
folder = 'D:/SarcAsM_drugs/'

# Frames to track. 'all' takes the whole movie. A subset must be contiguous and
# start at frame 0, e.g. range(0, 400) -- masks are stored one page per detected
# frame, so a later window would be indexed against the wrong pages.
frames = 'all'

# High-speed movies: the 3D U-Net gives Z-bands that are far less noisy from
# frame to frame. It replaces the Z-band mask ONLY -- M-bands, orientation and
# the sarcomere mask still come from detect_sarcomeres, which is why that has to
# cover every tracked frame either way.
fast_movie_zbands = False

# --- how much GPU each worker may take ---------------------------------------
# All workers predict on the SAME GPU, and on 'auto' each one sizes its patch and
# batch as if it had the card to itself -- with several workers the device runs
# out of memory and every prediction falls back to a smaller patch, which is
# slower and changes the mask at the tile seams. Pinning both makes the footprint
# per worker predictable: (1024, 1024) with batch_size=1 peaks at about 3 GB,
# (512, 512) at about 0.7 GB. Size n_pools so that n_pools x that fits the card.
# On a CPU-only machine these are speed knobs only.
n_pools = 4
max_patch_size = (1024, 1024)
batch_size = 1


def analyze_motion(file):
    """Analyze one movie. Returns None on success, or (file, error) on failure."""
    try:
        print(file, flush=True)
        sarc = SarcAsM(file)

        # Z-bands, M-bands, orientation and masks for every frame to be tracked:
        # the tracker needs M-bands and orientation per frame, so detecting the
        # first frame only is not enough here.
        sarc.detect_sarcomeres(frames=frames, max_patch_size=max_patch_size,
                               batch_size=batch_size)

        if fast_movie_zbands:
            sarc.detect_z_bands_fast_movie()

        # per-frame sarcomere vectors (position, length, orientation)
        sarc.analyze_sarcomere_vectors(frames=frames)

        # dense (n_tracks, T) trajectories
        sarc.track_sarcomere_vectors(frames=frames)

        # contractions of all tracks pooled into one averaged signal. Group
        # instead by 'mband', 'myofibril', 'loi', 'domain' or 'custom' to analyze
        # each group separately -- see the tracking tutorial.
        sarc.analyze_track_motion(by='pool')

        # remove intermediate masks to save storage, optional
        # sarc.remove_intermediate_masks()

        print(f'{file} successfully analyzed!', flush=True)
    except Exception as error:
        # One unanalyzable movie -- no calibration in the file, no sarcomeres
        # detected -- must not take the whole batch down with it.
        message = f'{type(error).__name__}: {error}'
        print(f'{file} FAILED -- {message}', flush=True)
        return file, message


if __name__ == '__main__':
    files = glob.glob(os.path.join(folder, '*.tif'))
    print(f'{len(files)} tif-files found')

    with Pool(n_pools) as p:
        failed = [result for result in p.map(analyze_motion, files) if result is not None]

    print(f'{len(files) - len(failed)}/{len(files)} movies analyzed')
    for file, message in failed:
        print(f'  failed: {file} -- {message}')

    # collect the per-group motion features of all movies into one table
    batch = BatchExport(files, folder=folder)
    batch.get_motion_data()
    batch.export_data(os.path.join(folder, 'motion_features.xlsx'))
