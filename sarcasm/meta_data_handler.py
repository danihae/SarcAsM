# Usage of this software for commercial purposes without a license is strictly prohibited.

import hashlib
import json
import os
import re
from typing import Union, Tuple, List

import PIL
import numpy as np
from PIL import Image
from tifffile import tifffile

from .ioutils import IOUtils
from .exceptions import MetaDataError


class MetaDataHandler:

    def __init__(self, sarc_obj):
        self.sarc_obj = sarc_obj
        self.sarc_obj.metadata = {}

        # due to renaming of meta_cell.json to metadata.json -> change name of meta data file todo remove later
        if os.path.exists(os.path.join(self.sarc_obj.data_folder, "meta_cell.json")):
            os.rename(os.path.join(self.sarc_obj.data_folder, "meta_cell.json"),
                      os.path.join(self.sarc_obj.data_folder, "metadata.json"))

        # check whether metadata file already exists
        if os.path.exists(self.get_meta_data_file()) and not self.sarc_obj.restart:
            self.load_meta_data()
        else:
            # create metadata dict
            self.create_meta_data()

    @staticmethod
    def check_meta_data_exists(tif_file: str, channel: Union[int, None]) -> bool:
        try:
            frames, size, pixelsize, _, timestamps = MetaDataHandler.extract_meta_data(tif_file=tif_file,
                                                                                       channel=channel)
            return True
        except MetaDataError:
            return False
            pass

    @staticmethod
    def extract_meta_data(tif_file: str, channel: Union[int, None], use_gui: bool = False,
                          info: dict = {}) -> Tuple[
        Union[int, None], Union[int, None], Union[float, None], Union[float, None], Union[List[str], None]]:
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
                        frames, _ = MetaDataHandler.__get_shape_from_file(tif_file, channel)
                else:
                    frames, _ = MetaDataHandler.__get_shape_from_file(tif_file, channel)

                # frame time
                if tif.imagej_metadata is not None:
                    frametime = (
                            tif.imagej_metadata.get('finterval') or
                            tif.imagej_metadata.get('Frame interval') or
                            tif.imagej_metadata.get('frame_interval')
                    )
                    if frametime is None:
                        fps = (
                                tif.imagej_metadata.get('fps') or
                                tif.imagej_metadata.get('Frames per second') or
                                tif.imagej_metadata.get('frame_rate')
                        )
                        if fps is not None:
                            try:
                                frametime = 1 / float(fps)
                            except (ValueError, ZeroDivisionError):
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
                frames, _ = MetaDataHandler.__get_shape_from_file(tif_file, channel)
        # pixelsize xy
        if 'pixelsize' in info.keys():
            pixelsize = info['pixelsize']
            _, size = MetaDataHandler.__get_shape_from_file(tif_file, channel)
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
                    _, size = MetaDataHandler.__get_shape_from_file(tif_file, channel)
            if pixelsize is None and not use_gui:
                raise MetaDataError('Pixel size could not be extracted from tif file. '
                                    'Please enter manually by, e.g., SarcAsM(filename, pixelsize=0.1)!')
        if 'frametime' in info.keys():
            frametime = info['frametime']
        elif frametime is None and frames > 1:
            print('frametime could not be extracted from tif file. Please enter manually if needed for analysis, '
                  'e.g. SarcAsM(file, frametime=0.1).')
        return frames, size, pixelsize, frametime, timestamps

    @staticmethod
    def __get_shape_from_file(file, channel: Union[int, None] = None):
        _data = MetaDataHandler.__read_image(file, channel)
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
    def __read_image(filename, channel: Union[int, None, str] = None, frame: Union[int, None] = None):
        """Load tif file, and optionally select channel"""
        if frame is None or frame == 'all':
            data = tifffile.imread(filename)
        else:
            data = tifffile.imread(filename, key=frame)
        if channel is not None:
            if channel == 'RGB':
                # Convert RGB to grayscale
                if data.ndim == 3 and data.shape[2] == 3:  # Single RGB image
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
                elif data.ndim == 4 and data.shape[3] == 3:  # Stack of RGB images
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
            elif isinstance(channel, int):
                if data.ndim == 3:
                    data = data[:, :, channel]
                elif data.ndim == 4:
                    data = data[:, :, :, channel]
            else:
                raise Exception('Parameter "channel" must be either int or "RGB"')

        if channel is not None:
            if data.ndim == 3:
                data = data[:, :, channel]
            elif data.ndim == 4:
                data = data[:, :, :, channel]
        return data

    @staticmethod
    def get_info_from_filename(filename: str):
        file_name = os.path.basename(filename)
        extract = lambda regex: next((m.group(1) for m in re.finditer(regex, filename)), None)

        date = extract('(\d{6,8})')
        days_culture = extract('day(\d+)')
        if days_culture: days_culture = int(days_culture)
        well = extract('[dish|well](\d+)')
        substrate = extract('(\d+)kpa')

        components = [str(date), str(days_culture), str(substrate), str(well)]
        measurement_id = hashlib.md5(''.join(components).encode()).hexdigest() if all(components) else None

        file_components = [file_name] + components
        file_id = hashlib.md5(''.join(file_components).encode()).hexdigest() if all(file_components) else None

        return file_name, date, days_culture, well, substrate, measurement_id, file_id

    def load_meta_data(self):
        if os.path.exists(self.get_meta_data_file()):
            # persistent file exists, try using it
            try:
                self.metadata = IOUtils.json_deserialize(self.get_meta_data_file())
            except:
                if os.path.exists(self.get_meta_data_file(is_temp_file=True)):
                    self.metadata = IOUtils.json_deserialize(self.get_meta_data_file(is_temp_file=True))
        else:
            # no persistent file exists, look if a temp-file exists
            if os.path.exists(self.get_meta_data_file()):
                self.metadata = IOUtils.json_deserialize(self.get_meta_data_file())
        if self.metadata is None:
            raise Exception('loading of metadata failed')
        else:
            # temp for backwards compatibility
            if 'resxy' in self.metadata.keys():
                self.metadata['pixelsize'] = self.metadata['resxy']
            if 'tint' in self.metadata.keys():
                self.metadata['frametime'] = self.metadata['tint']
            if 'resxy' in self.metadata.keys() or 'tint' in self.metadata.keys():
                self.store_meta_data()
            # commit changes
            self.commit()

    def get_meta_data_file(self, is_temp_file: bool = False):
        if is_temp_file:
            return os.path.join(self.sarc_obj.data_folder, "metadata.temp.json")
        else:
            return os.path.join(self.sarc_obj.data_folder, "metadata.json")

    def create_meta_data(self):
        """Create metadata for tif-file"""
        print('Creating metadata...')
        # get metadata from tif file
        frames, size, pixelsize, frametime, timestamps = MetaDataHandler.extract_meta_data(
            tif_file=self.sarc_obj.filename,
            channel=self.sarc_obj.channel,
            use_gui=self.sarc_obj.use_gui,
            info=self.sarc_obj.info)
        # create time array
        if frametime is not None:
            time = np.arange(0, frames * frametime, frametime)
        else:
            time = None
        # get info from filename
        (file_name, date, days_culture, well, substrate,
         measurement_id, file_id) = MetaDataHandler.get_info_from_filename(filename=self.sarc_obj.filename)
        # create dictionary for meta data and info (keyword args)
        self.metadata = {'file_name': file_name, 'size': size, 'pixelsize': pixelsize, 'frametime': frametime,
                         'frames': frames, 'file_path': self.sarc_obj.filename, 'substrate': substrate, 'date': date,
                         'days_culture': days_culture, 'time': time,
                         'well': well, 'measurement_id': measurement_id, 'file_id': file_id,
                         'timestamps': timestamps}
        self.metadata.update(self.sarc_obj.info)
        self.store_meta_data(override=True)

    def store_meta_data(self, override: bool = True):
        # only store if path doesn't exist or override is true
        if override or (not os.path.exists(self.get_meta_data_file())):
            IOUtils.json_serialize(self.metadata, self.get_meta_data_file())
            self.commit()

    def commit(self):
        """
        commit meta data (either rename temp-file to normal data file name or just write it again+remove temp-file)
        """
        if os.path.exists(self.get_meta_data_file(is_temp_file=True)):
            if os.path.exists(self.get_meta_data_file()):
                os.remove(self.get_meta_data_file())
            os.rename(self.get_meta_data_file(is_temp_file=True), self.get_meta_data_file())
