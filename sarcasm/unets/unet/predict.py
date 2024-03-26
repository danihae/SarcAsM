import os
import re

import numpy as np
import tifffile
import torch

from biu.progress import ProgressNotifier
from .unet import Unet
from .utils import save_as_tif

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
    """Class for prediction of movies and images with U-Net"""
    def __init__(self, tif_file, result_name, model_params, network=Unet, resize_dim=(512, 512),
                 invert=False, frame_lim=None, normalization_mode='single', clip_threshold=(0., 99.8), add_tile=0,
                 normalize_result=False, progress_bar=True,
                 progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        """
        Prediction of tif files with standard 2D U-Net

        1) Loading file and preprocess (normalization)
        2) Resizing of images into patches with resize_dim
        3) Prediction with U-Net
        4) Stitching of predicted patches and averaging of overlapping regions

        Parameters
        ----------
        tif_file : str
            path of tif file
        result_name : str
            path for result
        model_params : str
            path of u-net parameters (.pth file)
        network
            Network class
        resize_dim
            Image dimensions for resizing for prediction
        invert : bool
            Invert greyscale of image(s) before prediction
        frame_lim : Tuple[int, int], optional
            If not None, predict only interval
        normalization_mode : str
            Mode for intensity normalization for 3D stacks prior to prediction ('single': each image individually,
            'all': based on histogram of full stack, 'first': based on histogram of first image in stack)
        clip_threshold : Tuple[float, float]
            Clip threshold for image intensity before prediction
        add_tile : int, optional
            Add additional tiles for splitting large images to increase overlap
        normalize_result : bool
            If true, results are normalized to [0, 255]
        progress_notifier:
            Wrapper to show tqdm progress notifier in gui
        """
        self.tif_file = tif_file
        self.resize_dim = resize_dim
        self.add_tile = add_tile
        self.normalize_result = normalize_result
        self.invert = invert
        self.normalization_mode = normalization_mode
        self.clip_threshold = clip_threshold
        self.frame_lim = frame_lim
        self.result_name = result_name
        self.progress_bar = progress_bar

        # read, preprocess and split data
        imgs = self.__read_data()
        imgs = self.__preprocess(imgs)
        patches = self.__split(imgs)
        del imgs

        # load model and predict data
        self.model_params = torch.load(model_params, map_location=device)
        self.model = network(n_filter=self.model_params['n_filter']).to(device)
        self.model.load_state_dict(self.model_params['state_dict'])
        self.model.eval()
        result_patches = self.__predict(patches, progress_notifier)
        del patches
        # stitch patches (mean of overlapped regions)
        imgs_result = self.__stitch(result_patches)
        del result_patches

        # save as .tif file
        save_as_tif(imgs_result, self.result_name, normalize=normalize_result)

    def __read_data(self):
        imgs = tifffile.imread(self.tif_file)
        if self.frame_lim is not None:
            imgs = imgs[self.frame_lim[0]:self.frame_lim[1]]
        self.imgs_shape = imgs.shape
        if len(self.imgs_shape) == 2:  # if single image
            self.imgs_shape = [1, self.imgs_shape[0], self.imgs_shape[1]]
        return imgs

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
        # number of patches in x and y
        self.N_x = int(np.ceil(self.imgs_shape[1] / self.resize_dim[0])) + self.add_tile
        self.N_y = int(np.ceil(self.imgs_shape[2] / self.resize_dim[1])) + self.add_tile
        self.N_per_img = self.N_x * self.N_y
        self.N = self.N_x * self.N_y * self.imgs_shape[0]  # total number of patches
        print('Resizing into each %s patches ...' % self.N_per_img)

        # define array for prediction
        patches = np.zeros((self.N, 1, self.resize_dim[0], self.resize_dim[1]), dtype='uint8')

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
                imgs = np.pad(imgs, ((0, self.resize_dim[0] - self.imgs_shape[1]), (0, 0)), 'reflect')
            if self.imgs_shape[2] < self.resize_dim[1]:  # for y
                imgs = np.pad(imgs, ((0, 0), (0, self.resize_dim[1] - self.imgs_shape[2])), 'reflect')

        # starting indices of patches
        self.X_start = np.linspace(0, self.imgs_shape[1] - self.resize_dim[0], self.N_x).astype('uint16')
        self.Y_start = np.linspace(0, self.imgs_shape[2] - self.resize_dim[1], self.N_y).astype('uint16')

        # split in resize_dim
        n = 0
        if self.imgs_shape[0] > 1:
            for i, img in enumerate(imgs):
                for j in range(self.N_x):
                    for k in range(self.N_y):
                        patches[n, 0, :, :] = img[self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                              self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                        n += 1
        elif self.imgs_shape[0] == 1:
            for j in range(self.N_x):
                for k in range(self.N_y):
                    patches[n, 0, :, :] = imgs[self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                                          self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]]
                    n += 1
        return patches

    def __predict(self, patches, progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()):
        result_patches = np.zeros(patches.shape, dtype='uint8')
        print('Predicting data ...')
        with torch.no_grad():
            for i, patch_i in enumerate(progress_notifier.iterator(patches)):
                patch_i = torch.from_numpy(patch_i.astype('float32') / 255).to(device).view((1, 1, self.resize_dim[0],
                                                                                             self.resize_dim[1]))
                res_i, logits_i = self.model(patch_i)
                res_i = res_i.view((1, self.resize_dim[0], self.resize_dim[1])).cpu().numpy() * 255
                result_patches[i] = res_i.astype('uint8')
                del patch_i, res_i
        return result_patches

    def __stitch(self, result_patches):
        print('Stitching patches back together ...')
        # create array
        imgs_result = np.zeros((self.imgs_shape[0], np.max((self.resize_dim[0], self.imgs_shape[1])),
                                np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8')
        for i in range(self.imgs_shape[0]):
            if self.imgs_shape[0] > 1:  # if stack
                stack_result_i = np.zeros((self.N_per_img, np.max((self.resize_dim[0], self.imgs_shape[1])),
                                           np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8') * np.nan
            elif self.imgs_shape[0] == 1:  # if only one image
                stack_result_i = np.zeros((self.N_per_img, np.max((self.resize_dim[0], self.imgs_shape[1])),
                                           np.max((self.resize_dim[1], self.imgs_shape[2]))), dtype='uint8') * np.nan
            else:
                raise ValueError('Data structure not known!')
            n = 0
            for j in range(self.N_x):
                for k in range(self.N_y):
                    stack_result_i[n, self.X_start[j]:self.X_start[j] + self.resize_dim[0],
                    self.Y_start[k]:self.Y_start[k] + self.resize_dim[1]] = result_patches[i * self.N_per_img + n, 0, :,
                                                                            :]
                    n += 1

            # average overlapping regions
            if self.imgs_shape[0] > 1:  # if stack
                imgs_result[i] = np.nanmean(stack_result_i, axis=0)
            elif self.imgs_shape[0] == 1:  # if only one image
                imgs_result = np.nanmean(stack_result_i, axis=0)

        # change to input size (if zero padding)
        if self.imgs_shape[0] > 1:  # if stack
            imgs_result = imgs_result[:, :self.imgs_shape[1], :self.imgs_shape[2]]
        elif self.imgs_shape[0] == 1:  # if only one image
            imgs_result = imgs_result[:self.imgs_shape[1], :self.imgs_shape[2]]

        return imgs_result
