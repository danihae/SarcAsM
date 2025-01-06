from multiprocessing import Pool
import glob
from sarcasm import *


folder = 'D:/2024_competition_data/'

# find files
files = glob.glob(folder + '/*/*.tif')[::-1]
print(f'{len(files)} tif-files found')


def find_lois(file):
    sarc_obj = SarcAsM(file)
    sarc_obj.structure.predict_z_bands(siam_unet=True)
    sarc_obj.structure.analyze_sarcomere_vectors(frames=0)
    sarc_obj.structure.detect_rois(timepoint=0)


def analyze_motion(file):
    lois = Utils.get_lois_of_file(file)
    for file, loi in lois:
        try:
            mot_obj = Motion(file, loi)
            mot_obj.full_analysis_loi()
            Plots.plot_loi_summary_motion(mot_obj)
        except Exception as e:
            print(file, loi)
            print(repr(e))


if __name__ == '__main__':
    # analyze ROIs
    with Pool(8) as p:
        p.map(analyze_motion, files)

