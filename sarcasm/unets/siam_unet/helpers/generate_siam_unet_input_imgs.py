import cv2
import numpy as np
import tifffile
import os
import glob
from scipy.ndimage import geometric_transform
import tifffile
import glob
import numpy as np
import matplotlib.pyplot as plt


def generate_coupled_image(movie, frame, output):
    """
    Generates an image from the previous frame of the given frame and the given frame of the given movie. Should be used as input to Siam_UNet. The output image is a tiff file with the files like this, with the output image of twice the width of the input.
    [    Previous   |     Frame to be    ]
    [     Frame     |   Inferred From    ]

    Note on arg: 
        frame (int): 0-indexed
    """

    curr_frame = tifffile.imread(movie, key=frame)
    if frame == 0:
        prev_frame = tifffile.imread(movie, key=(frame + 1))
    else:
        prev_frame = tifffile.imread(movie, key=(frame - 1))

    if curr_frame is None:
        print(f'Error: {movie} not found.')
        raise IOError  # tiff file not found

    out = np.concatenate((prev_frame, curr_frame), axis=1).astype(np.uint8)
    cv2.imwrite(filename=output, img=out, )


def generate_coupled_image_from_self(img, out_img, noise_amp=10):
    """
        Generates an input image for siam by concatenating an image with a transformed version of itself 
    """

    def __synthesize_prev_img(in_img, noise_amp=10):
        """Synthesizes previous frame by transforming the input image

        Args:
            in_img (str): input image path
            noise_amp (int, optional): Defaults to 10.

        Returns:
            2-D ndarray: the synthesized previous image
        """
        data = tifffile.imread(in_img)
        image = data
        modes_x, modes_y = 10, 4
        amp = 1
        amps_x, amps_y = np.random.random_sample(modes_x) * amp, np.random.random_sample(modes_y) * amp

        def func(xy):
            return (xy[0] + np.sum(amps_y * np.sin(modes_y * 2 * np.pi * xy[0] / image.shape[0])),
                    xy[1] + np.sum(amps_x * np.sin(modes_x * 2 * np.pi * xy[1] / image.shape[1])))

        out = geometric_transform(image, func)
        noise = np.random.normal(0, noise_amp, size=image.shape)
        out = out + noise
        out[out < 0] = 0
        out[out > 255] = 255

        return out

    curr_frame = tifffile.imread(img)
    synthesized_previous_frame = __synthesize_prev_img(img, noise_amp)

    if curr_frame is None:
        raise IOError  # tiff file not found

    out = np.concatenate((synthesized_previous_frame, curr_frame), axis=1).astype(np.uint8)
    cv2.imwrite(filename=out_img, img=out, )


def utilize_search_result(search_result_mr_txt, movie_path_prefix, labels_path_prefix, output_folder):
    """Parses search results obtained from find_frame_of_image.find_frame_of_image()'s machine readable output and pass them to generate_coupled_image() to create training data for Siam_UNet. 

    This function outputs two folders under output_folder: image and label. All the output will have a suffix of .tif, and thyey will have the same names as the names of input labels. The output in the image directory will be the current frame concatenated with the previous frame, while the label directory will contain the corresponding labels, directly copied from the input labels directory. Visually, the directory structure at output:

    output_folder
    |
    |----image
    |       |-----concatenated images
    |
    |
    |----label
    |       |-----labels copied from the input labels directory

    Args:
        search_result_mr_txt (str): path to the machine readable output
        movie_path_prefix, labels_path_prefix (str): prefixes to name of each movie in the search result file, respectively
        output_folder (str): path to the output folder. See above.
    """

    image_output_folder = output_folder + '/image/'
    label_output_folder = output_folder + '/label/'

    # make output folder if it does not exist
    os.system(f'mkdir -p \'{output_folder}\'')
    # recreate image and label output folders
    os.system(f'rm -rf \'{image_output_folder}\'')
    os.system(f'rm -rf \'{label_output_folder}\'')
    os.system(f'mkdir -p \'{image_output_folder}\'')
    os.system(f'mkdir -p \'{label_output_folder}\'')

    with open(search_result_mr_txt) as sr:
        lines = sr.readlines()
        for line in lines:
            label_path = labels_path_prefix + '/' + line.split('\t')[0]
            image_path = movie_path_prefix + '/' + line.split('\t')[1]
            output_image_path = image_output_folder + line.split('\t')[0]
            frame_number = int(line.split('\t')[2])
            # copy label to label folder
            os.system(f'cp \'{label_path}\' \'{label_output_folder}\'')
            # create concatenated image in image_folder
            generate_coupled_image(image_path, frame=frame_number, output=output_image_path)

    for file in glob.glob(f'{label_output_folder}/*.tif'):
        i = cv2.imread(file)
        fo = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
        os.remove(file)
        cv2.imwrite(file, fo)