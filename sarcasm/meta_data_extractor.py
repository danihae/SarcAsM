import json

import PIL
from tifffile import tifffile
from PIL import Image

from .exceptions import MetaDataError


class MetaDataExtractor:

    def __init__(self):
        pass

    @staticmethod
    def check_meta_data_exists(tif_file: str) -> bool:
        try:
            frames, size, pixelsize, _, timestamps = MetaDataExtractor.extract_meta_data(tif_file)
            return True
        except MetaDataError:
            return False
            pass

        pass

    @staticmethod
    def extract_meta_data(tif_file: str, channel=None, use_gui=False, info={}):
        # attempt to extract metadata from tif file
        with tifffile.TiffFile(tif_file) as tif:
            if hasattr(tif, 'imagej_metadata'):
                # frame number
                if tif.imagej_metadata is not None:
                    if 'frames' in tif.imagej_metadata.keys():
                        frames = tif.imagej_metadata['frames']
                    elif 'slices' in tif.imagej_metadata.keys():
                        frames = tif.imagej_metadata['slices']
                    else:
                        frames, _ = MetaDataExtractor.__get_shape_from_file(tif_file, channel)
                else:
                    frames, _ = MetaDataExtractor.__get_shape_from_file(tif_file, channel)
                # frame time
                if tif.imagej_metadata is not None:
                    if 'finterval' in tif.imagej_metadata.keys():
                        frametime = tif.imagej_metadata['finterval']
                    else:
                        frametime = None
                else:
                    frametime = None
                # timestamps
                if tif.imagej_metadata is not None:
                    if 'timestamps' in tif.imagej_metadata.keys():
                        try:
                            timestamps = json.loads(tif.imagej_metadata['timestamps'])
                        except:
                            timestamps = tif.imagej_metadata['timestamps']
                    else:
                        timestamps = None
                else:
                    timestamps = None
            else:
                frames, _ = MetaDataExtractor.__get_shape_from_file(tif_file, channel)
        # pixelsize xy
        if 'pixelsize' in info.keys():
            pixelsize = info['pixelsize']
            _, size = MetaDataExtractor.__get_shape_from_file(tif_file, channel)
        else:
            with PIL.Image.open(tif_file) as tif:
                if 'resolution' in tif.info.keys():
                    if float(tif.info['resolution'][0]) != 1:
                        pixelsize = 1 / float(tif.info['resolution'][0])
                    else:
                        pixelsize = None
                else:
                    pixelsize = None
                # size
                if 'size' in tif.info.keys():
                    size = tif.info.size
                else:
                    _, size = MetaDataExtractor.__get_shape_from_file(tif_file, channel)
            if pixelsize is None and not use_gui:
                raise MetaDataError('Pixel size could not be extracted from tif file. '
                                    'Please enter manually by, e.g., SarcAsM(filename, pixelsize=0.1)!')
        if frametime is None and frames > 1:
            print('frametime could not be extracted from tif file. Please enter manually if needed for analysis, '
                  'e.g. SarcAsM(file, frametime=0.1).')
        return frames, size, pixelsize, frametime, timestamps

        pass

    @staticmethod
    def __get_shape_from_file(file, channel=None):
        _data = MetaDataExtractor.__read_image(file, channel)
        if len(_data.shape) == 2:
            frames = 1
            size = _data.shape
        elif len(_data.shape) == 3:
            frames = _data.shape[0]
            size = _data.shape[1:]
        else:
            frames = None
            size = None
        return frames, size

    pass

    @staticmethod
    def __read_image(filename, channel=None, timepoint=None):
        """Load tif file, and optionally select channel"""
        if timepoint is None or timepoint == 'all':
            data = tifffile.imread(filename)
        else:
            data = tifffile.imread(filename, key=timepoint)
        if channel is not None:
            if data.ndim == 3:
                data = data[:, :, channel]
            elif data.ndim == 4:
                data = data[:, :, :, channel]
        return data
