import cv2
import os
os.nice(20)
import subprocess
import numpy as np

def generate_plain_image(pixel_value=255, shape=(1000, 500), outfile_name="val_255.png"):
	"""Generates a plain image. Useful for figuring out whether a pixel value of 255 means white or 0 means white. (Hint: 255 is whie in png files)

	Args:
		pixel_value (int, optional): the pixel value to generate. Defaults to 255.
		shape (tuple, optional): size of image. Defaults to (1000, 500).
		outfile_name (str, optional): where to save the file. Defaults to "val_255.png".
	"""
	out = (np.ones(shape) * pixel_value).astype(np.uint8)
	cv2.imwrite(filename=outfile_name, img=out, )

if __name__ == '__main__':
	generate_plain_image()