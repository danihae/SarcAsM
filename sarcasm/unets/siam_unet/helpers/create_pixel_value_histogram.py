import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def create_pixel_value_histogram(input_tifs, frames_per_hist=100, bin_width=8):
	"""Creates a histogram for the pixel values in a tif file

	Args:
		input_tifs (str path to .tif files): input
		frames_per_hist (int, optional): number of frames to plot a histogram. Defaults to 100.
		bin_width (int, optional): bin width on the outpuut tif plot. Should be set to a value divisible by 256. Defaults to 8.

	Raises:
		IOError: An IOError is raised when the input tif file cannot be found
	"""
	retval, imgs = cv2.imreadmulti(input_tifs[0])
	fig, axs = plt.subplots(math.ceil(len(imgs)/frames_per_hist), len(input_tifs), figsize=(5, 20))
	fig.suptitle([input_tif.split('/')[-1] for input_tif in input_tifs])
	for input_tif_idx, input_tif in enumerate(input_tifs):
		if input_tif_idx != 0:
			retval, imgs = cv2.imreadmulti(input_tif)
		if not retval:
			print(f'Error: {input_tif} not found.')
			raise IOError   # tiff file not found		
		axis_idx = 0
		for idx, img in enumerate(imgs):
			if idx % frames_per_hist == 0:
				img_val = img.flatten()
				axs[axis_idx, input_tif_idx].hist(img_val, bins=range(0, 256, bin_width))
				axs[axis_idx, input_tif_idx].set(title=f'frame={idx}')
				axs[axis_idx, input_tif_idx].set_xlim([0, 255])
				axs[axis_idx, input_tif_idx].set_yscale('log')
				axis_idx += 1
	plt.show()


def main():
	# tif_name = constants.TIFS_NAMES[3]
	# create_pixel_value_histogram([f"/home/longyuxi/Documents/mount/predicted_out/old_self_model_predictions/predicted_{tif_name}_with_cosh_dice.tif", f"/home/longyuxi/Documents/mount/predicted_out/predicted_newly_drawn_{tif_name}_epoch_500.tif"], bin_width=2)
	pass



if __name__ == '__main__':
	main()