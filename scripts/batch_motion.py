from multiprocessing import Pool
import glob
from sarcasm import *


folder = 'D:/SarcAsM_drugs/'

# find files
files = glob.glob(folder + '/*/*.tif')[::-1]
print(f'{len(files)} tif-files found')


def find_rois(file):
    sarc_obj = SarcAsM(file)
    sarc_obj.structure.predict_z_bands(siam_unet=True)
    sarc_obj.structure.analyze_sarcomere_length_orient(frames=0)
    sarc_obj.structure.detect_rois(timepoint=0)


def analyze_motion(file):
    rois = Utils.get_rois_of_cell(file)
    for file, roi in rois:
        try:
            mot_obj = Motion(file, roi)
            mot_obj.full_analysis_loi()
        except Exception as e:
            print(file, roi)
            print(repr(e))


if __name__ == '__main__':
    # find ROIs
    with Pool(4) as p:
        p.map(find_rois, files)

    # analyze ROIs
    with Pool(12) as p:
        p.map(analyze_motion, files)

