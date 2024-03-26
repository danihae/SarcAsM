import cv2 # type: ignore
import numpy as np
from util import write_info_file

def average_tifs(inputs : list, output):
	"""Averages a list of single channel tiff movies

	Args:
		inputs (list): A list of paths (str) to tiff movies
		output (str): Location to output the tiff file

	Raises:
		IOError: If one of the input tiff movies is not found
	"""
	tifs = []
	for tiff_movie in inputs:
		retval, imgs = cv2.imreadmulti(tiff_movie)
		if not retval:
			print(f'Error: {tiff_movie} not found.')
			raise IOError   # tiff file not found
		tifs.append(imgs)
	print('Averaging', len(tifs), 'images')
	out_tifs = [] # list (frames) of numpy arrays that are of size e.g. 1160*2300
	for i in range(len(tifs[0])):
		out_tifs.append(np.zeros(tifs[0][0].shape, dtype=np.int16))

	# add each frame from the input to the output
	for tif_idx, tif in enumerate(tifs):
		for frame_idx, frame in enumerate(tif):
			# add each frame to the corresponding frame in out_tif
			out_tifs[frame_idx] += frame

	write_info_file(f'{output}.info.txt', f'Operation: Average\nOutfile: {output}\nInput files:{inputs}')
	final_output_tif = []
	# average the frames in the output
	for frame in out_tifs:
		final_output_tif.append((frame / len(tifs)).astype('uint8'))
	
	cv2.imwritemulti(output, final_output_tif)


if __name__ == '__main__':
	pass