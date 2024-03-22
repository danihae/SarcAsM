# this script is used for building the classes/functions/methods for analysis the following:
import sarcasm.nuclei.helper as helper
from sarcasm.nuclei.nuclei import NucleiAndSarcomereCorrelation
import os


def process_directory(directory_path, correct_phase_leica=False, network_model='generalist', siam_unet=False,
                      width=1024, height=1024, clip_thresh_min=0., clip_thresh_max=99.8, pixel_size=None,
                      frame_time=None, do_closing=False, do_fill_holes=False, min_area=100, max_area=100000,
                      limit_organized_sarcomere=1, threshold_method_nuclei='YEN', manual_threshold_nuclei=None,
                      preprocess_sigma_nuclei=1.0, blur_sigma_fibers=4.0, threshold_method_fiber='ISODATA',
                      manual_threshold_fiber=None):
    # scan directory for tif files we assume that all tif files are the 2 channel files
    # put each of the necessary parameters in the parameter list of this function
    # process each of the files and store outputs (use the sarcasm_old generated data directories)
    image_files = [file for file in os.listdir(directory_path) if file.endswith(".tif")]
    data_list = list()
    for file in image_files:
        tmp = NucleiAndSarcomereCorrelation()
        tmp.read_structure_file(file_path=os.path.join(directory_path, file))
        tmp.analyze_sarcomeres(correct_phase_leica=correct_phase_leica, network_model=network_model,
                               siam_unet=siam_unet, width=width,
                               height=height, clip_thresh_min=clip_thresh_min, clip_thresh_max=clip_thresh_max,
                               pixel_size=pixel_size, frame_time=frame_time)
        tmp.analyze_nuclei(do_closing=do_closing, do_fill_holes=do_fill_holes, min_area=min_area, max_area=max_area,
                           threshold_method=threshold_method_nuclei, manual_threshold=manual_threshold_nuclei,
                           preprocess_sigma=preprocess_sigma_nuclei)
        tmp.analyze_fibers(blur_sigma=blur_sigma_fibers, threshold_method=threshold_method_fiber,
                           threshold=manual_threshold_fiber)
        tmp.cross_analyze_sarcomeres(sarcomere_image_path=os.path.join(tmp.sarcasm.folder, 'sarcomeres.tif'),
                                     output_directory=tmp.sarcasm.folder,
                                     limit_organized_sarcomere=limit_organized_sarcomere)
        print(tmp.stats)
        helper.plot_stats(stats=tmp.stats,
                          store_plot_path=os.path.join(tmp.sarcasm.folder, 'nuclei_analysis_plot.png'))
        tmp.unload_memory_intensive_data()
        data_list.append(tmp)
        pass

    stat_dict_list = [k.stats for k in data_list]

    for key in helper.StatKeys:
        if key.value == helper.StatKeys.FILE_NAME.value:
            pass
        else:
            helper.plot_stat(stat_dict_list=stat_dict_list, stat_key=key, title=key.value,
                             store_plot_path=os.path.join(directory_path, key.value + ".png"))

            pass

        pass

    import csv
    with open(os.path.join(directory_path, 'summary.csv'), 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='#', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([s.value for s in helper.StatKeys])
        for stat_dict in stat_dict_list:
            writer.writerow([v for v in stat_dict.values()])
            pass

    pass


# Vorweg: alle Rechtschreibfehler und Grammatikfehler sind geplant und dienen dem künstlerischen Ausdruck :P (es liegt sicher nicht an mangelnden Kenntnissen der Muttersprache)

# Liebe Mitmenschen welche mutig genug sind dieses skript zu benutzen.
# Als directory_path bitte das Verzeichnis angeben welches analysiert werden soll.
# als pixel_size bitte die Pixelgröße angeben mit der die Bilder gemacht wurden (es sollten Bilder des gleichen Mikroskops zusammen analysiert werden, im Optimalfall).
# als frame_time - da es sich um Einzelbilder handelt ? - ist dieser Wert eigentlich egal :D - einfach so lassen
# für min_area und max_area die Werte angeben in Pixel, welche die Größe von Nuclei einschränken soll.
# limit_organized_sarcomere gibt an wieviel Sarcomere-Pixel in einem Nuclei enthalten sein sollen damit es als "enthält Sarcomere" gewertet wird.
# threshold_method_nuclei gibt an welche thresholding methode genutzt werden soll für das nuclei bild. genaueres in ADD 1)
# manual_threshold_nuclei wenn dieser Wert nicht auf None ist, wird ein manuelles thresholding gemacht mit dem angegebenen Wert. Bei Bildern welche unter den gleichen Umständen entstanden sind, könnte dies durchaus zu guten Ergebnissen führen.
# preprocess_sigma_nuclei bestimmt den Sigma Wert für gaussian blur beim Erstellen des Nuclei-Bildes/Clustering
# blur_sigma_fibers bestimmt den Sigma Wert für gaussian blur beim Erstellen des Muskelflächen-Bildes (je höher desto unschärfer)
# threshold_method_fiber gibt an welche thresolding methode genutzt werden soll für das Muskelflächen-Bild. genaueres in ADD 1)
# manual_threshold_fiber wenn dieser Wert nicht auf None ist, wird ein manuelles thresholding gemacht mit dem angegebenen Wert.

# ADD 1)
# zur Auswahl gibt es folgende 4 Möglichkeiten (es können gerne im helper.py noch weitere hinzugefügt werden):
# 'TRIANGLE','YEN','OTSU','ISODATA'
#
if __name__ == "__main__":
    pixel_size = 0.053
    frame_time = 0.1
    process_directory(directory_path='D:\\Test\\1_MalteTib\\TestDir', correct_phase_leica=False, pixel_size=pixel_size,
                      frame_time=frame_time, min_area=100, max_area=100000, limit_organized_sarcomere=10,
                      threshold_method_nuclei='YEN', manual_threshold_nuclei=None,
                      preprocess_sigma_nuclei=1.0, blur_sigma_fibers=4.0, threshold_method_fiber='ISODATA',
                      manual_threshold_fiber=None)
    pass
