import cv2
import os
os.nice(20)
import subprocess
import numpy as np
import tifffile
from skimage import morphology

import platform
if platform.system() != 'Linux':
    raise Exception  # this script is designed to use Linux bash commands. Please use Linux

try:
    subprocess.run(["ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    raise Exception   # ffmpeg not detected. Please install ffmpeg


def convert_to_mp4(tiff_movie, output_file="out.mp4", fps=30, perform_threshold=False, threshold_val=250, invert=False, normalize_to_0_255=True, closing=False, close_thresh=10):
    """
    Converts tiff_movie to an mp4 movie

    params:
        tiff_movie: file name for input
        fps: number of frames per second in the output
        perform_threshold: whether we want the images to be thresholded (i.e. the pixels to be aligned to either white or black), 0 is black, 255 is white
        threshold_val: the value at which thresholding is done. pixel_val = 255 if pixel_val >= threshold_val else 0
        normalize_to_0_255: scales each individual image to the full range of (0, 255).
    """

    imgs = frame_generator(tiff_movie)
    temp_dir = f'temp_{tiff_movie.split("/")[-1]}'
    os.system(f"mkdir -p \'{temp_dir}\'")

    ct = 0
    for img in imgs:
        if invert:
            img = np.ones(img.shape) * 255 - img
        if perform_threshold:
            img = ((img >= threshold_val) * np.ones(img.shape) * 255 ).astype(np.uint8)
        if normalize_to_0_255:
            img = img.astype(np.double)
            img = (img - np.min(img))/(np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
        if closing:
            img = morphology.opening(img, selem=morphology.star(close_thresh))
        name = str(ct).zfill(5)
        cv2.imwrite(f"{temp_dir}/{name}.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        ct += 1

    # convert to mp4
    # -crf 17 is visually "lossless" as mentioned in https://trac.ffmpeg.org/wiki/Encode/H.264
    ffmpeg_command = f'ffmpeg -y -r {fps} -i \'{temp_dir}/%5d.png\' -c:v libx264 -crf 17 -pix_fmt yuv420p \'{output_file}\''

    # execution and cleanup
    print('Executing ' + ffmpeg_command)
    os.system(f'{ffmpeg_command} ; rm -rf \'{temp_dir}\'')

def frame_generator(tiff_movie):
    """Returns a generator for each individual frame of the movie. Should function in the same way as imgs.

    Args:
        tiff_movie (str): path to tiff movie
    """
    tif_key = tifffile.TiffFile(tiff_movie)
    tif_len = len(tif_key.pages)
    for i in range(tif_len):
        yield tifffile.imread(tiff_movie, key=i)


if __name__ == '__main__':
    # convert_to_mp4('/media/longyuxi/H is for HUGE/docmount backup/unet_pytorch/training_data/test_data/new_microscope/21B11-shgGFP-kin-18-bro4.tif', output_file='/media/longyuxi/H is for HUGE/docmount backup/predicted_out/siam_bce_predicted_lowmem_21B11-shgGFP-kin-18-bro4.mp4', normalize_to_0_255=True)
    convert_to_mp4('/media/longyuxi/H is for HUGE/docmount backup/predicted_out/minitest.tif', output_file='/media/longyuxi/H is for HUGE/docmount backup/predicted_out/minitest_closing.mp4', normalize_to_0_255=True, closing=True)
    