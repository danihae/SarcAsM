import glob
import os
import shutil
from skimage import morphology, transform
from albumentations import (
    ShiftScaleRotate, GaussNoise,
    RandomBrightnessContrast, Flip, Compose, RandomRotate90)

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class DataProcess(Dataset):
    """Class for training data generation (processing, splitting, augmentation)"""
    def __init__(self, source_dir, dim_out=(256, 256), aug_factor=10, data_path='../data/',
                 dilate_mask=0, dilate_kernel='disk', val_split=0.2, invert=False, skeletonize=False, create=True,
                 clip_threshold=(0.2, 99.8), shiftscalerotate=(0, 0, 0), noise_amp=10, brightness_contrast=(0.25, 0.25),
                 rescale=None):
        """
        Create training data object for network training

        1) Create folder structure for training data
        2) Move and preprocess training images
        3) Split input images into tiles
        4) Augment training data
        5) Create object of PyTorch Dataset class for training

        Parameters
        ----------
        source_dir : Tuple[str, str]
            Path of training data [images, labels]. Images need to be tif files.
        dim_out : Tuple[int, int]
            Resize dimensions of images for training
        aug_factor : int
            Factor of image augmentation
        data_path : str
            Base path of directories for training data
        dilate_mask
            Radius of binary dilation of masks [-2, -1, 0, 1, 2]
        dilate_kernel : str
            Dilation kernel ('disk' or 'square')
        val_split : float
            Validation split for training
        invert : bool
            If True, greyscale binary labels is inverted
        skeletonize : bool
            If True, binary labels are skeletonized
        create : bool, optional
            If False, existing data set in data_path is used
        clip_threshold : Tuple[float, float]
            Clip thresholds for intensity normalization of images
        shiftscalerotate : [float, float, float]
            Shift, scale and rotate image during augmentation
        noise_amp : float
            Amplitude of Gaussian noise for image augmentation
        brightness_contrast : Tuple[float, float]
            Adapt brightness and contrast of images during augmentation
        rescale : float, optional
            Rescale all images and labels by factor rescale
        """
        self.source_dir = source_dir
        self.create = create
        self.data_path = data_path
        self.dim_out = dim_out
        self.skeletonize = skeletonize
        self.invert = invert
        self.clip_threshold = clip_threshold
        self.aug_factor = aug_factor
        self.shiftscalerotate = shiftscalerotate
        self.brightness_contrast = brightness_contrast
        self.noise_amp = noise_amp
        self.rescale = rescale
        self.dilate_mask = dilate_mask
        self.dilate_kernel = dilate_kernel
        self.val_split = val_split
        self.mode = 'train'

        self.__make_dirs()
        if create:
            self.__move_and_edit()
            self.__merge_images()
            self.__split()
            if self.aug_factor is not None:
                self.__augment()

    def __make_dirs(self):
        self.image_path = self.data_path + '/image/'
        self.mask_path = self.data_path + '/mask/'
        self.merge_path = self.data_path + '/merge/'
        self.split_merge_path = self.data_path + '/split/merge/'
        self.split_image_path = self.data_path + '/split/image/'
        self.split_mask_path = self.data_path + '/split/mask/'
        self.aug_image_path = self.data_path + '/augmentation/aug_image/'
        self.aug_mask_path = self.data_path + '/augmentation/aug_mask/'

        # delete old files
        if self.create:
            try:
                shutil.rmtree(self.data_path)
            except:
                pass
        # make folders
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.merge_path, exist_ok=True)
        os.makedirs(self.split_merge_path, exist_ok=True)
        os.makedirs(self.split_image_path, exist_ok=True)
        os.makedirs(self.split_mask_path, exist_ok=True)
        os.makedirs(self.aug_image_path, exist_ok=True)
        os.makedirs(self.aug_mask_path, exist_ok=True)

    def __move_and_edit(self):
        # create image data
        files_image = glob.glob(self.source_dir[0] + '*')
        for file_i in files_image:
            img_i = tifffile.imread(file_i)
            # clip and normalize (0,255)
            img_i_nan = np.copy(img_i).astype('float32')
            img_i_nan[img_i == 0] = np.nan
            img_i = np.clip(img_i, a_min=np.nanpercentile(img_i_nan, self.clip_threshold[0]),
                            a_max=np.percentile(img_i_nan, self.clip_threshold[1]))
            img_i = img_i - np.min(img_i)
            img_i = img_i / np.max(img_i) * 255
            img_i[np.isnan(img_i_nan)] = 0
            if self.rescale is not None:
                img_i = transform.rescale(img_i, self.rescale)
            img_i = img_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            tifffile.imsave(self.image_path + save_i + '.tif', img_i)
        # create masks
        files_mask = glob.glob(self.source_dir[1] + '*')
        print('%s files found' % len(files_mask))
        for file_i in files_mask:
            mask_i = tifffile.imread(file_i)
            if self.rescale is not None:
                mask_i = transform.rescale(mask_i, self.rescale) * 255
                mask_i[mask_i < 255] = 0
            if self.skeletonize:
                mask_i[mask_i > 1] = 1
                mask_i = 1 - mask_i
                mask_i = morphology.skeletonize(mask_i)
                mask_i = 255 * (1 - mask_i)
            if self.dilate_kernel == 'disk':
                kernel = morphology.disk
            elif self.dilate_kernel == 'square':
                kernel = morphology.square
            else:
                raise ValueError(f'Dilate kernel {self.dilate_kernel} unknown!')
            if self.dilate_mask > 0:
                mask_i = morphology.erosion(mask_i, kernel(self.dilate_mask))
            elif self.dilate_mask < 0:
                mask_i = morphology.dilation(mask_i, kernel(-self.dilate_mask))
            if self.invert:
                mask_i = 255 - mask_i
            mask_i = mask_i.astype('uint8')
            save_i = os.path.splitext(os.path.basename(file_i))[0]
            save_i = save_i.replace(' ', '_')
            tifffile.imsave(self.mask_path + save_i + '.tif', mask_i.astype('int8'))

    def __merge_images(self):
        self.mask_files = glob.glob(self.data_path + '/mask/*.tif')
        self.image_files = glob.glob(self.data_path + '/image/*.tif')

        if len(self.mask_files) != len(self.image_files):
            raise ValueError('Number of ground truth does not match number of image stacks')

        for i, file_i in enumerate(self.mask_files):
            basename_i = os.path.basename(file_i)
            mask_i = tifffile.imread(self.data_path + '/mask/' + basename_i)
            image_i = tifffile.imread(self.data_path + '/image/' + basename_i)
            merge = np.zeros((mask_i.shape[0], mask_i.shape[1], 3))
            merge[:, :, 0] = mask_i
            merge[:, :, 1] = image_i
            merge = merge.astype('uint8')
            tifffile.imsave(self.merge_path + str(i) + '.tif', merge)

    def __split(self):
        self.merges = glob.glob(self.merge_path + '*.tif')
        n = 0
        for i in range(len(self.merges)):
            merge = tifffile.imread(self.merge_path + str(i) + '.tif')
            dim_in = merge.shape
            # padding if dim_in < dim_out
            x_gap = max(0, self.dim_out[0] - dim_in[0])
            y_gap = max(0, self.dim_out[1] - dim_in[1])
            merge = np.pad(merge, ((0, x_gap), (0, y_gap), (0, 0)), 'reflect')
            # number of patches in x and y
            dim_in = merge.shape
            N_x = int(np.ceil(dim_in[0] / self.dim_out[0]))
            N_y = int(np.ceil(dim_in[1] / self.dim_out[1]))
            # starting indices of patches
            X_start = np.linspace(0, dim_in[0] - self.dim_out[0], N_x).astype('int16')
            Y_start = np.linspace(0, dim_in[1] - self.dim_out[1], N_y).astype('int16')
            for j in range(N_x):
                for k in range(N_y):
                    patch_ij = merge[X_start[j]:X_start[j] + self.dim_out[0], Y_start[k]:Y_start[k] + self.dim_out[1],
                               :]
                    mask_ij = patch_ij[:, :, 0]
                    image_ij = patch_ij[:, :, 1]
                    tifffile.imsave(self.split_merge_path + f'{n}.tif', patch_ij)
                    tifffile.imsave(self.split_mask_path + f'{n}.tif', mask_ij)
                    tifffile.imsave(self.split_image_path + f'{n}.tif', image_ij)
                    n += 1

    def __augment(self, p=0.8):
        aug_pipeline = Compose([
            Flip(),
            RandomRotate90(p=1.0),
            ShiftScaleRotate(self.shiftscalerotate[0], self.shiftscalerotate[1], self.shiftscalerotate[2]),
            GaussNoise(var_limit=(self.noise_amp, self.noise_amp), p=0.3),
            RandomBrightnessContrast(brightness_limit=self.brightness_contrast[0],
                                     contrast_limit=self.brightness_contrast[1], p=0.5),
        ],
            p=p)

        patches_image = glob.glob(self.split_image_path + '*.tif')
        patches_mask = glob.glob(self.split_mask_path + '*.tif')
        n_patches = len(patches_image)
        k = 0
        for i in range(n_patches):
            image_i = tifffile.imread(patches_image[i])
            mask_i = tifffile.imread(patches_mask[i])

            data_i = {'image': image_i, 'mask': mask_i}
            data_aug_i = np.asarray([aug_pipeline(**data_i) for _ in range(self.aug_factor)])
            imgs_aug_i = np.asarray([data_aug_i[j]['image'] for j in range(self.aug_factor)])
            masks_aug_i = np.asarray([data_aug_i[j]['mask'] for j in range(self.aug_factor)])

            for j in range(self.aug_factor):
                tifffile.imsave(self.aug_image_path + f'{k}.tif', imgs_aug_i[j])
                tifffile.imsave(self.aug_mask_path + f'{k}.tif', masks_aug_i[j])
                k += 1
        print(f'Number of training images: {k}')

    def __len__(self):
        if self.aug_factor is not None:
            return len(os.listdir(self.aug_image_path))
        else:
            return len(os.listdir(self.split_image_path))

    def __getitem__(self, idx):
        imgname = str(idx) + '.tif'
        midname = os.path.basename(imgname)
        if self.aug_factor is not None:
            image_0 = tifffile.imread(self.aug_image_path + midname).astype('float32') / 255
            mask_0 = tifffile.imread(self.aug_mask_path + midname).astype('float32') / 255
        else:
            image_0 = tifffile.imread(self.split_image_path + midname).astype('float32') / 255
            mask_0 = tifffile.imread(self.split_image_path + midname).astype('float32') / 255
        image = torch.from_numpy(image_0)
        mask = torch.from_numpy(mask_0)
        del image_0, mask_0
        sample = {'image': image, 'mask': mask}
        return sample
