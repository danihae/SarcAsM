# -*- coding: utf-8 -*-
# Copyright (c) 2025 University Medical Center Göttingen, Germany.
# All rights reserved.
#
# Patent Pending: DE 10 2024 112 939.5
# SPDX-License-Identifier: LicenseRef-Proprietary-See-LICENSE
#
# This software is licensed under a custom license. See the LICENSE file
# in the root directory for full details.
#
# **Commercial use is prohibited without a separate license.**
# Contact MBM ScienceBridge GmbH (https://sciencebridge.de/en/) for licensing.


import json
import os
from pathlib import Path
from typing import Union, Tuple, List, Optional, Dict

import PIL.Image
import numpy as np
from tifffile import TiffFile, imread

from sarcasm.exceptions import MetaDataError
from sarcasm.ioutils import IOUtils


class MetaDataHandler:
    def __init__(self, sarc_obj) -> None:
        self.sarc_obj = sarc_obj
        # Store metadata in the sarc_obj
        self.sarc_obj.metadata = {}
        self.data_folder: Path = Path(self.sarc_obj.data_dir)

        # On initialization, load existing metadata if available and not restarting, otherwise create new metadata.
        if self.get_meta_data_file().exists() and not self.sarc_obj.restart:
            self.load_meta_data()
        else:
            self.create_meta_data()

    @staticmethod
    def check_meta_data_exists(tif_file: str, channel: Optional[int]) -> bool:
        try:
            # Attempt to extract required metadata; ignore the returned values.
            MetaDataHandler.extract_meta_data(tif_file=tif_file, channel=channel)
            return True
        except MetaDataError:
            return False

    @staticmethod
    def extract_meta_data(tif_file: str,
                          channel: Optional[int],
                          use_gui: bool = False,
                          info: Dict = {}) -> Tuple[
        Optional[int], Optional[Tuple[int, ...]], Optional[float], Optional[float], Optional[List]]:
        # Open the TIFF file and try to extract ImageJ metadata if available.
        with TiffFile(tif_file) as tif:
            imagej_md = getattr(tif, "imagej_metadata", None)
            if imagej_md:
                # frame number (fallback to shape if not provided)
                if 'frames' in imagej_md:
                    frames = imagej_md['frames']
                elif 'slices' in imagej_md:
                    frames = imagej_md['slices']
                else:
                    frames, _ = MetaDataHandler.__get_shape_from_file(tif_file, channel)

                # Try to obtain frametime via various keys.
                frametime = (
                        imagej_md.get('finterval') or
                        imagej_md.get('Frame interval') or
                        imagej_md.get('frame_interval')
                )
                if frametime is None:
                    fps = (
                            imagej_md.get('fps') or
                            imagej_md.get('Frames per second') or
                            imagej_md.get('frame_rate')
                    )
                    if fps:
                        try:
                            frametime = 1 / float(fps)
                        except (ValueError, ZeroDivisionError):
                            frametime = None

                # Try to load timestamps, attempt JSON-decoding if needed.
                if 'timestamps' in imagej_md:
                    try:
                        timestamps = json.loads(imagej_md['timestamps'])
                    except Exception:
                        timestamps = imagej_md['timestamps']
                else:
                    timestamps = None
            else:
                frames, _ = MetaDataHandler.__get_shape_from_file(tif_file, channel)
                frametime = None
                timestamps = None

        # Determine pixelsize and image size.
        if 'pixelsize' in info:
            pixelsize = info['pixelsize']
            _, size = MetaDataHandler.__get_shape_from_file(tif_file, channel)
        else:
            with PIL.Image.open(tif_file) as img:
                img_info = img.info
                if 'resolution' in img_info:
                    try:
                        res = float(img_info['resolution'][0])
                        pixelsize = 1 / res if res != 1 else None
                    except (ValueError, TypeError):
                        pixelsize = None
                else:
                    pixelsize = None

                size = img.size if 'size' in img_info else MetaDataHandler.__get_shape_from_file(tif_file, channel)[1]

            if pixelsize is None and not use_gui:
                raise MetaDataError(f"Pixel size could not be extracted from {tif_file}. "
                                    f"Please enter manually (e.g., Structure(filename, pixelsize=0.1)).")

        # Allow manual override of frametime.
        if 'frametime' in info:
            frametime = info['frametime']
        elif frametime is None and frames and frames > 1:
            print('Warning: frametime could not be extracted from tif file. '
                  'Please enter manually if needed (e.g., SarcAsM(file, frametime=0.1)).')

        return frames, size, pixelsize, frametime, timestamps

    @staticmethod
    def __get_shape_from_file(file: str, channel: Optional[int] = None) -> Tuple[
        Optional[int], Optional[Tuple[int, ...]]]:
        data = MetaDataHandler.__read_image(file, channel)
        if data.ndim == 2:
            return 1, data.shape
        elif data.ndim == 3:
            return data.shape[0], data.shape[1:]
        else:
            return None, None

    @staticmethod
    def __read_image(filename: str,
                     channel: Optional[Union[int, str]] = None,
                     frame: Optional[int] = None) -> np.ndarray:
        """Load a TIFF file and optionally select a channel.

        If channel is 'RGB', the image is converted to grayscale.
        Otherwise, if channel is an int, the specified channel is selected.
        """
        # Read the image (all frames by default)
        data = imread(filename) if frame is None or frame == 'all' else imread(filename, key=frame)

        if channel is not None:
            if channel == 'RGB':
                # Convert RGB image or stack of RGB images to grayscale
                if data.ndim == 3 and data.shape[-1] == 3:
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
                elif data.ndim == 4 and data.shape[-1] == 3:
                    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
            elif isinstance(channel, int):
                if data.ndim == 3:
                    data = data[:, :, channel]
                elif data.ndim == 4:
                    data = data[:, :, :, channel]
            else:
                raise ValueError('Parameter "channel" must be either an int or "RGB".')
        return data

    def load_meta_data(self) -> None:
        meta_file = Path(self.get_meta_data_file())
        temp_meta_file = Path(self.get_meta_data_file(is_temp_file=True))
        errors = []
        # Try to load persistent file first, then the temporary file.
        for file in (meta_file, temp_meta_file):
            if file.exists():
                try:
                    self.sarc_obj.metadata = IOUtils.json_deserialize(str(file))
                    break
                except Exception as err:
                    errors.append(f"Error loading {file}: {err}")
        else:
            raise Exception(f"Loading of metadata failed. Errors: {'; '.join(errors)}")

        # Backward compatibility updates for metadata keys.
        if 'resxy' in self.sarc_obj.metadata:
            self.sarc_obj.metadata['pixelsize'] = self.sarc_obj.metadata['resxy']
        if 'tint' in self.sarc_obj.metadata:
            self.sarc_obj.metadata['frametime'] = self.sarc_obj.metadata['tint']
        if 'resxy' in self.sarc_obj.metadata or 'tint' in self.sarc_obj.metadata:
            self.store_meta_data()
        self.commit()

    def get_meta_data_file(self, is_temp_file: bool = False) -> Path:
        filename = "metadata.temp.json" if is_temp_file else "metadata.json"
        return Path(self.sarc_obj.data_dir) / filename

    def create_meta_data(self) -> None:
        print("Creating metadata...")
        frames, size, pixelsize, frametime, timestamps = MetaDataHandler.extract_meta_data(
            tif_file=self.sarc_obj.filepath,
            channel=self.sarc_obj.channel,
            use_gui=self.sarc_obj.use_gui,
            info=self.sarc_obj.info
        )
        time_array = np.arange(0, frames * frametime, frametime) if frametime is not None else None

        self.sarc_obj.metadata = {
            "file_name": os.path.basename(self.sarc_obj.filepath),
            "file_path": self.sarc_obj.filepath,
            "size": size,
            "pixelsize": pixelsize,
            "frametime": frametime,
            "frames": frames,
            "time": time_array,
            "timestamps": timestamps
        }
        # Merge any additional info from sarc_obj.
        self.sarc_obj.metadata.update(self.sarc_obj.info)
        self.store_meta_data(override=True)

    def store_meta_data(self, override: bool = True) -> None:
        meta_file = Path(self.get_meta_data_file())
        if override or not meta_file.exists():
            IOUtils.json_serialize(self.sarc_obj.metadata, str(meta_file))
            self.commit()

    def commit(self) -> None:
        """
        Commit metadata by renaming the temporary file to the persistent file,
        ensuring an atomic update.
        """
        meta_file = Path(self.get_meta_data_file())
        temp_meta_file = Path(self.get_meta_data_file(is_temp_file=True))
        if temp_meta_file.exists():
            if meta_file.exists():
                meta_file.unlink()
            temp_meta_file.rename(meta_file)
