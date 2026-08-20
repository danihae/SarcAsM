"""Batch structure analysis: all sarcomere structure features for every tif in a folder.

Tune the parameters on one or two representative files first, see
``docs/notebooks/tutorial_structure_analysis.ipynb`` for the step-by-step
equivalent of ``full_analysis_structure``. Run this as a script rather than in a
notebook -- ``multiprocessing`` needs the ``__main__`` guard below.
"""
import glob
import os
from multiprocessing import Pool

from sarcasm import SarcAsM

# --- what to analyze ---------------------------------------------------------
folder = 'D:/2023_SarcAsM_drugs_chronic/'

# Frames to analyze. 'all' takes every frame of a time-lapse; 0 analyzes the
# first frame only, which is what a folder of single images needs. A subset must
# start at frame 0, e.g. list(range(0, 10)) -- pass a list, not a range object.
frames = 'all'

# --- how much GPU each worker may take ---------------------------------------
# All workers predict on the SAME GPU, and on 'auto' each one sizes its patch and
# batch as if it had the card to itself -- with several workers the device runs
# out of memory and every prediction falls back to a smaller patch, which is
# slower and changes the mask at the tile seams. Pinning both makes the footprint
# per worker predictable: (1024, 1024) with batch_size=1 peaks at about 3 GB,
# (512, 512) at about 0.7 GB, while the (2048, 2048) that suits a single process
# takes about 13 GB. Size n_pools so that n_pools x that fits the card.
n_pools = 3
max_patch_size = (1024, 1024)
batch_size = 1


def analyze_tif(file):
    """Analyze one file. Returns None on success, or (file, error) on failure."""
    try:
        print(file, flush=True)
        sarc = SarcAsM(file)

        # detect sarcomere Z-bands, M-bands, orientation and the cell mask
        sarc.detect_sarcomeres(frames=frames, max_patch_size=max_patch_size,
                               batch_size=batch_size)

        # cell mask, Z-bands, sarcomere vectors, myofibrils and domains in one
        # call -- use the step-by-step analysis to tune individual parameters
        sarc.full_analysis_structure(frames=frames)

        # remove intermediate masks to save storage, optional
        # sarc.remove_intermediate_masks()

        print(f'{file} successfully analyzed!', flush=True)
    except Exception as error:
        # One unanalyzable file -- no calibration in the file, no sarcomeres
        # detected -- must not take the whole batch down with it.
        message = f'{type(error).__name__}: {error}'
        print(f'{file} FAILED -- {message}', flush=True)
        return file, message


if __name__ == '__main__':
    tif_files = glob.glob(os.path.join(folder, '*.tif'))
    print(f'{len(tif_files)} tif-files found')

    with Pool(n_pools) as p:
        failed = [result for result in p.map(analyze_tif, tif_files) if result is not None]

    print(f'{len(tif_files) - len(failed)}/{len(tif_files)} files analyzed')
    for file, message in failed:
        print(f'  failed: {file} -- {message}')
