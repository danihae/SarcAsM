import tifffile
from tifffile import TiffFile

def fetch_frame(tif_file):
    """Returns a generator of frames in a tif file

    Args:
        tif_file (str): path to the tif file of concern

    Yields:
        numpy ndarray: A matrix representing a frame in the image. Should be two dimensional if the input is grayscale. You can also quickly get its dimension with tif_key.pages[0].shape

    """
    tif_key = TiffFile(tif_file)
    tif_len = len(tif_key.pages)
    for i in range(0, tif_len):
        yield tifffile.imread(tif_file, key=i)

def individual_tif_generator(dir, tif_len):
    """
    A generator that returns each frame in a directory. See siam_unet_cosh.py for usage.
    """
    for i in range(tif_len):
        yield tifffile.imread(f'{dir}/{i}.tif')
    
    """
    And now, because tifffile supports a iterable object as its data parameter, you can do

        tifffile.imwrite(data=tqdm(self.individual_tif_generator(dir=temp_dir), total=self.tif_len, unit='frame'), file=self.result_name, dtype=np.uint8, shape=self.imgs_shape)
    
    Again, see siam_unet_cosh.py for usage.
    """


if __name__ == '__main__':
    ## A demonstration of fetch_frame()
    import cv2
    import constants
    ###
    # Now instead of
    #  for frame in movie:
    # You can do
    for frame in fetch_frame(f'{constants.RAZER_LOCAL_MOVIE_DIR}/{constants.TIFS_NAMES[0]}.tif'):
        cv2.imshow('i', frame)
        cv2.waitKey(0)
