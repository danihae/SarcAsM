import cv2 # type: ignore
import numpy as np
import time
import tifffile
from tifffile import TiffFile
import os

def find_frame_of_image(query_image, search_space=[], save_machine_readable_output=True, machine_readable_output_filename='search_result_mr.txt'):
    """Finds the frame number of query_image within search_space. 
    
    This function will move on to the next candidate in search_space if the dimensions of query_image doesn't match a certain candidate. If the dimensions of query_image and candidate match, this function first attempts to find an exact match, but resorts to finding frame with the least mean squared error (MSE) if no exact match is found.

    If save_machine_readable_output is set to True, this function will write a tab-separated file containing the names (not absolute paths) of the query image (label), the candidate image with the best match, and the frame number (0-indexed) that has the best match.

    This function is best used with the function utilize_search_result.generate_siam_unet_input_imgs().

    Args:
        query_image (str): path to query tiff image, must be single frame
        search_space (list of str, optional): The list of images over which to search query_image. Defaults to constants.TIFS_NAMES.
    """

    query = cv2.imread(query_image, cv2.IMREAD_GRAYSCALE)
    for candidate in search_space:
        tif_key = TiffFile(candidate)
        # tif_len = len(tif_key.pages)

        # check if dimensions are the same
        if not tif_key.pages[0].shape == query.shape:
            print(f'Shape of query {query.shape} differs with shape of {candidate} {tif_key.pages[0].shape}')
            continue

        # if yes, progress
        mses = []
        frame_nbr = 0
        imgs = frame_generator(candidate)
        for img in imgs:
            # print(frame_nbr)
            if np.array_equal(img, query):
                print(f"\n\n!!!!Found exact match in frame {frame_nbr} of {candidate}.\n\n")
                continue
            mses.append(mse(query, img))
            frame_nbr += 1
        
        # if there is not an exact match, resort to the best MSE solution
        print(f'No exact match was found in {candidate}. The closest matching frame was {mses.index(min(mses))} with MSE of {min(mses)}')


        if save_machine_readable_output:
            # save information directly if:
            #   1. frame numbers match
            #   2. MSE < 1000
            if mses.index(min(mses)) == int(os.path.basename(query_image).split(".")[0]):
                if min(mses) < 1000:
                    with open(machine_readable_output_filename, 'a') as o:
                        o.write(f'{os.path.basename(query_image)}\t{os.path.basename(candidate)}\t{mses.index(min(mses))}\n')

def frame_generator(tiff_movie):
    """Returns a generator for each individual frame of the movie. Should function in the same way as imgs in the expression "for img in imgs".

    Args:
        tiff_movie (str): path to tiff movie
    """
    tif_key = tifffile.TiffFile(tiff_movie)
    tif_len = len(tif_key.pages)
    for i in range(tif_len):
        yield tifffile.imread(tiff_movie, key=i)

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


if __name__ == '__main__':
    # find_frame_of_image(f'{constants.LOCAL_OUT_DIR}/21B25_shgGFP_kin_1_Pos0_frame_10.tif', [f'{constants.LOCAL_MOVIE_DIR}/{constants.TIFS_NAMES[1]}.tif'])

    # the missing search space was 21D16_shgGFPkin_Pos7.tif


    # # successful at finding the location to the amnioserosa images
    # file_names = ['105.tif','111.tif','120.tif','121.tif','146.tif','165.tif','166.tif','167.tif','1.tif','212.tif','224.tif','231.tif','253.tif','268.tif','2.tif','305.tif','314.tif','328.tif','331.tif','335.tif','369.tif','372.tif','3.tif','413.tif','439.tif','480.tif','496.tif','502.tif','510.tif','522.tif','5.tif','625.tif','646.tif','681.tif','69.tif','700.tif','707.tif','732.tif','778.tif','792.tif','7.tif','822.tif','837.tif','83.tif','840.tif']

    # # file_names = ['105.tif']
    # for file_name in file_names:
    #     print(file_name)
    #     find_frame_of_image(f'{constants.RAZER_LOCAL_BASE_DIR}/training_data/training_data/yokogawa/amnioserosa/image/{file_name}', search_space=constants.RAZER_LARGE_SEARCH_SPACE)
    pass