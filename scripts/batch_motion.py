import glob
import os
from multiprocessing import Pool

from sarcasm import SarcAsM, BatchExport

folder = 'D:/SarcAsM_drugs/'

# find files
files = glob.glob(os.path.join(folder, '*.tif'))
print(f'{len(files)} tif-files found')


# analyze sarcomere motion of a single movie
def analyze_motion(file):
    print(file)
    # initialize file
    sarc = SarcAsM(file)

    # detect all sarcomere features for the first frame only
    sarc.detect_sarcomeres(frames=0)

    # detect Z-bands for all frames with the time-consistent 3D-U-Net,
    # alternatively run detect_sarcomeres(frames='all')
    sarc.detect_z_bands_fast_movie()

    # analyze sarcomere vectors in all frames
    sarc.analyze_sarcomere_vectors(frames='all')

    # track individual sarcomere vectors through the movie
    sarc.track_sarcomere_vectors()

    # analyze contractions of all tracks pooled into one averaged signal
    sarc.analyze_track_motion(by='pool')

    # remove intermediate masks to save storage, optional
    # sarc.remove_intermediate_masks()

    print(f'{file} successfully analyzed!')


# set number of pools
n_pools = 4

if __name__ == '__main__':
    with Pool(n_pools) as p:
        p.map(analyze_motion, files)

    # collect the per-group motion features of all movies into one table
    batch = BatchExport(files, folder=folder)
    batch.get_motion_data()
    batch.export_data(os.path.join(folder, 'motion_features.xlsx'))
