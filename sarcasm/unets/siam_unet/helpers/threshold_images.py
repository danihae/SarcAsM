import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

def threshold_images(in_path, out_path):
    """
    Performs the threshold function on all the images in the folder `in_path` and outputs them in `out_path`
    Params:
        in_path: folder of source images
        out_path: folder to place images
    """

    files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
    for f in files:
        img = np.array(cv2.imread(join(in_path, f), cv2.IMREAD_GRAYSCALE))
        out_img = ((img >= 150) * np.ones(img.shape) * 255 ).astype(np.uint8)
        cv2.imwrite(filename=join(out_path, f), img=out_img, )

def invert_images(in_path, out_path):
    """
    Performs the invert function on all the images in the folder `in_path` and outputs them in `out_path`
    Params:
        in_path: folder of source images
        out_path: folder to place images
    """

    files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
    for f in files:
        img = np.array(cv2.imread(join(in_path, f), cv2.IMREAD_GRAYSCALE))
        img = (np.ones(img.shape) * 255 - img).astype(np.uint8)
        cv2.imwrite(filename=join(out_path, f), img=img, )


if __name__ == '__main__':
    # threshold all the images I have drawn 
    pass