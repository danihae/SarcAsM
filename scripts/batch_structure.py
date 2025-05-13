import glob
import os
from multiprocessing import Pool
from sarcasm import Structure

# select folder with tif files
folder = 'D:/2023_SarcAsM_drugs_chronic/'

# find all tif files in folder
tif_files = glob.glob(os.path.join(folder, '*.tif'))
print(f'{len(tif_files)} tif-files found')

# function for analysis of single tif-file
def analyze_tif(file):
    print(file)
    # initialize SarcAsM object
    sarc = Structure(file)

    # detect sarcomere z-bands, m-bands, sarcomere orientation and cell masks
    sarc.detect_sarcomeres(max_patch_size=(2048, 2048))

    # analyze sarcomere structures (or use step-by-step analysis, see tutorial structure analysis)
    sarc.full_analysis_structure(frames='all')

    print(f'{file} successfully analyzed!')


# set number of pools
n_pools = 3

if __name__ == '__main__':
    with Pool(n_pools) as p:
        p.map(analyze_tif, tif_files)