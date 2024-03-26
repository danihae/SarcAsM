import cv2 # type: ignore
import numpy as np

def extract_frame_of_movie(tiff_movie, frame_number, output_file):
	"""Extract a certain frame of tiff_movie, and write it into a separate file. Used for testing find_frame_of_image

	Args:
		tiff_movie (str): path to input
		frame_number (int): which frame to extract

	Raises:
		IOError: raised when tiff file cannot be found
	"""
	retval, imgs = cv2.imreadmulti(tiff_movie)
	if not retval:
		print(f'Error: {tiff_movie} not found.')
		raise IOError   # tiff file not found
	output = imgs[frame_number]
	print(output.shape)
	cv2.imwrite(filename=output_file, img=output, )

def extract_frames_of_movie(tiff_movie, frame_number, output_file):
	"""Extract up to a certain frame of tiff_movie, and write it into a separate file. Used for testing find_frame_of_image

	Args:
		tiff_movie (str): path to input
		frame_number (int): up to which frame to extract

	Raises:
		IOError: raised when tiff file cannot be found
	"""
	retval, imgs = cv2.imreadmulti(tiff_movie)
	if not retval:
		print(f'Error: {tiff_movie} not found.')
		raise IOError   # tiff file not found
	output = imgs[:frame_number]
	cv2.imwritemulti(filename=output_file, img=output, )


if __name__ == '__main__':
	# extract_frames_of_movie(f'{constants.LOCAL_MOVIE_DIR}/{constants.TIFS_NAMES[1]}.tif', 10, f'{constants.LOCAL_OUT_DIR}/{constants.TIFS_NAMES[1]}_up_to_frame_10.tif')
	pass