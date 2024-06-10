from multiprocessing import Pool
from sarcasm import *

# select folder with tif files
folder = 'D:/2023_SarcAsM_drugs_chronic/'

# find all tif files in folder
tif_files = glob.glob(folder + '*/*.tif')
print(len(tif_files))


def analyze_tif(file):
    print(file)
    # initialize SarcAsM object
    sarc_obj = SarcAsM(file)

    # predict sarcomere z-bands and cell area
    sarc_obj.predict_z_bands(size=(2048, 2048))
    sarc_obj.predict_cell_area(size=(2048, 2048))

    # analyze cell area and sarcomere area
    sarc_obj.analyze_cell_area(frames='all')
    sarc_obj.analyze_sarcomere_area(frames='all')

    # analyze sarcomere structures
    sarc_obj.full_analysis_structure(frames='all')

    print(f'{file} successfully analyzed!')


# set number of pools
n_pools = 3

if __name__ == '__main__':
    with Pool(n_pools) as p:
        p.map(analyze_tif, tif_files)
