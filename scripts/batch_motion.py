import glob
import os
from multiprocessing import Pool
from sarcasm import *

folder = 'D:/SarcAsM_drugs/'

# find files
files = glob.glob(os.path.join(folder, '*.tif'))
print(f'{len(files)} tif-files found')


# detect LOIs
def detect_lois(file):
    # initialize file
    sarc = Structure(file)

    # detect all sarcomere features for first frame only
    sarc.detect_sarcomeres(frames=0)

    # detect Z-bands for all frames with time-consistent 3D-U-Net, alternatively run detect_sarcomeres(frames='all')
    sarc.detect_z_bands_fast_movie()

    # analyze sarcomere vectors in first frame
    sarc.analyze_sarcomere_vectors(frames=0)

    # detect lines of interest (LOIs)
    sarc.detect_lois(n_lois=4)

    # remove intermediate tiff files to save storage, optional
    # sarc.remove_intermediate_tiffs()


# analyze all LOIs of one tif-file
def analyze_lois(file):
    lois = Utils.get_lois_of_file(file)
    for file, loi in lois:
        try:
            # initialize LOI
            mot_obj = Motion(file, loi)

            # analysis of LOI with default parameters
            mot_obj.full_analysis_loi()

        except Exception as e:
            print(file, loi)
            print(repr(e))


if __name__ == '__main__':
    # find LOIs
    with Pool(4) as p:
        p.map(detect_lois, files)

    # analyze LOIs
    with Pool(12) as p:
        p.map(analyze_lois, files)