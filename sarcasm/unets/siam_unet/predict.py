import os

import numpy as np
import tifffile
import torch
from tifffile.tifffile import TiffFile
from tqdm import tqdm as tqdm
from biu.progress import ProgressNotifier

from .siam_unet import Siam_UNet

# select device
if torch.has_cuda:
    device = torch.device('cuda:0')
elif hasattr(torch, 'has_mps'):  # only for apple m1/m2/...
    if torch.has_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cpu')


class Predict:
    """
    Class for prediction of tif-movies.
    1) Loading file and preprocess (normalization)
    2) Resizing of images into patches with resize_dim
    3) Prediction with U-Net
    4) Stitching of predicted patches and averaging of overlapping regions
    """

    def __init__(self, tif_file, result_name, model_params, resize_dim=(512, 512), invert=False,
                 normalization_mode='single', clip_threshold=(0.0, 99.98), add_tile=0, normalize_result=False,
                 progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """
        Predicts a tif movie

        Parameters
        ----------
        tif_file : str
            Path to input tif stack
        result_name : str
            Path of result file
        tif_file : str
            path of tif file
        result_name : str
            path for result
        model_params : str
            path of u-net parameters (.pth file)
        resize_dim
            Image dimensions for resizing for prediction. If resize_dim=None, the image will not be resized but
            the whole image will be processed by the convolution layers.
        invert : bool
            Invert greyscale of image(s) before prediction
        normalization_mode : str
            Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
            'all': based on histogram of full stack, 'first': based on histogram of first image in stack)
        clip_threshold : Tuple[float, float]
            Clip threshold for image intensity before prediction
        add_tile : int, optional
            Add additional tiles for splitting large images to increase overlap
        normalize_result : bool
            If True, results are normalized to [0, 255]
        progress_notifier:
            Wrapper to show tqdm progress notifier in gui
        """
        self.tif_file = tif_file
        self.add_tile = add_tile
        self.invert = invert
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.result_name = result_name
        self.normalize_result = normalize_result  # todo to be implemented for Siam-U-Net?

        # load model
        self.model_params = torch.load(model_params, map_location=device)
        self.model = Siam_UNet(n_filter=self.model_params['n_filter'], mode=self.model_params['mode']).to(device)
        self.model.load_state_dict(self.model_params['state_dict'])
        self.model.eval()

        # split data into groups of two images
        tif_key = TiffFile(self.tif_file)
        self.tif_len = len(tif_key.pages)
        self.imgs_shape = [self.tif_len, tif_key.pages[0].shape[0], tif_key.pages[0].shape[1]]
        if resize_dim is not None:
            self.resize_dim = resize_dim
        else:
            self.resize_dim = (self.imgs_shape[1], self.imgs_shape[2])

        # create temp folder
        temp_dir = f'temp_{self.tif_file.split("/")[-1]}'

        # taken from split()
        # number of patches in x and y
        self.N_x = int(np.ceil(self.imgs_shape[1] / self.resize_dim[0])) + self.add_tile
        self.N_y = int(np.ceil(self.imgs_shape[2] / self.resize_dim[1])) + self.add_tile
        self.N_per_img = self.N_x * self.N_y
        self.N = self.N_x * self.N_y  # total number of patches

        self.progress_notifier = progress_notifier

        # predict each pair, and save the output of each one as a separate image
        os.makedirs(temp_dir, exist_ok=True)
        print('Predicting data ...')
        with tifffile.TiffWriter(self.result_name, bigtiff=False) as tif:
            for i, _ in enumerate(self.progress_notifier.iterator(range(self.tif_len))):
                if i == 0:
                    if self.tif_len == 1:
                        prev_img = tifffile.imread(self.tif_file, key=0)
                    else:
                        prev_img = tifffile.imread(self.tif_file, key=1)
                else:
                    prev_img = current_img
                current_img = tifffile.imread(self.tif_file, key=i)

                img_stack = np.array([prev_img, current_img])
                img_stack = self.__preprocess(img_stack)
                patches = self.__split(img_stack)
                print(f'Patches shape:{patches.shape}') if i == 0 else None
                result_patches = self.__predict(patches)
                imgs_result = self.__stitch(result_patches)
                tif.write(imgs_result, contiguous=True)

    def __preprocess(self, imgs):
        if len(imgs.shape) == 3:
            if self.normalization_mode == 'single':
                for i, img in enumerate(imgs):
                    img = np.clip(img, a_min=np.nanpercentile(img, self.clip_threshold[0]),
                                  a_max=np.percentile(img, self.clip_threshold[1]))
                    img = img - np.min(img)
                    img = img / np.max(img) * 255
                    if self.invert:
                        img = 255 - img
                    imgs[i] = img
            elif self.normalization_mode == 'first':
                clip_threshold = (np.nanpercentile(imgs[0], self.clip_threshold[0]),
                                  np.percentile(imgs[0], self.clip_threshold[1]))
                imgs = np.clip(imgs, clip_threshold[0], clip_threshold[1])
                imgs = imgs - np.min(imgs)
                imgs = imgs / np.max(imgs) * 255
                if self.invert:
                    imgs = 255 - imgs
            elif self.normalization_mode == 'all':
                clip_threshold = (np.nanpercentile(imgs, self.clip_threshold[0]),
                                  np.percentile(imgs, self.clip_threshold[1]))
                imgs = np.clip(imgs, clip_threshold[0], clip_threshold[1])
                imgs = imgs - np.min(imgs)
                imgs = imgs / np.max(imgs) * 255
                if self.invert:
                    imgs = 255 - imgs
            else:
                raise ValueError(f'normalization_mode {self.normalization_mode} not valid!')
        if len(imgs.shape) == 2:
            imgs = np.clip(imgs, a_min=np.nanpercentile(imgs, self.clip_threshold[0]),
                           a_max=np.percentile(imgs, self.clip_threshold[1]))
            imgs = imgs - np.min(imgs)
            imgs = imgs / np.max(imgs) * 255
            if self.invert:
                imgs = 255 - imgs
        imgs = imgs.astype('uint8')
        return imgs

    def __split(self, imgs):
        # define array for prediction
        patches = np.zeros((self.N, 2, self.resize_dim[0], self.resize_dim[1]), dtype='uint8')

        # zero padding of image if imgs_shape < resize_dim
        if self.imgs_shape[0] > 1:
            if self.imgs_shape[1] < self.resize_dim[0]:  # for x
                imgs = np.pad(imgs, ((0, 0), (0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)),
                              'constant')
            if self.imgs_shape[2] < self.resize_dim[1]:  # for y
                imgs = np.pad(imgs, ((0, 0), (0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])),
                              'constant')
        elif self.imgs_shape[0] == 1:
            if self.imgs_shape[1] < self.resize_dim[0]:  # for x
                imgs = np.pad(imgs, ((0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)), 'constant')
            if self.imgs_shape[2] < self.resize_dim[1]:  # for y
                imgs = np.pad(imgs, ((0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])), 'constant')

        # starting indices of patches
        self.X_start = np.linspace(0, self.imgs_shape[1] - self.resize_dim[0], self.N_x).astype('uint16')
        self.Y_start = np.linspace(0, self.imgs_shape[2] - self.resize_dim[1], self.N_y).astype('uint16')

        # split in resize_dim
        n = 0

        i = 1
        for j in range(self.N_x):
            for k in range(self.N_y):
                patches[n, 0, :, :] = imgs[i][self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                      self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                patches[n, 1, :, :] = imgs[i - 1][self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                      self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                n += 1
        return patches

    def __predict(self, patches):
        result_patches = np.zeros((patches.shape[0], 1, patches.shape[2], patches.shape[3]), dtype='uint8')
        with torch.no_grad():
            for i, patch_i in enumerate(patches):
                image_patch_i = patch_i[0, :, :]
                prev_image_patch_i = patch_i[1, :, :]

                image_patch_i = torch.from_numpy(image_patch_i.astype('float32') / 255).to(device).view(
                    (1, 1, self.resize_dim[0], self.resize_dim[1]))
                prev_image_patch_i = torch.from_numpy(prev_image_patch_i.astype('float32') / 255).to(device).view(
                    (1, 1, self.resize_dim[0], self.resize_dim[1]))

                res_i = self.model(image_patch_i, prev_image_patch_i)[0].view(
                    (1, self.resize_dim[0], self.resize_dim[1])).cpu().numpy() * 255
                result_patches[i] = res_i.astype('uint8')
                del patch_i, res_i
        return result_patches

    def __stitch(self, result_patches):
        dtype = result_patches.dtype
        i = 0
        if self.imgs_shape[0] > 1:  # if stack
            stack_result_i = np.zeros((self.N_per_img, np.max((self.resize_dim[0], self.imgs_shape[1])),
                                       np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype=dtype) * np.nan
        elif self.imgs_shape[0] == 1:  # if only one image
            stack_result_i = np.zeros((self.N_per_img, np.max((self.resize_dim[0], self.imgs_shape[1])),
                                       np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype=dtype) * np.nan
        else:
            raise ValueError('Wrong data format!')
        n = 0
        for j in range(self.N_x):
            for k in range(self.N_y):
                stack_result_i[n, self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]] = result_patches[i * self.N_per_img + n, 0, :, :]
                n += 1
        # average overlapping regions
        imgs_result = np.nanmean(stack_result_i, axis=0).astype(dtype)

        # change to input size (if zero padding)
        imgs_result = imgs_result[:self.imgs_shape[1], :self.imgs_shape[2]]

        return imgs_result
