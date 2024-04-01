import hashlib
import os
import re
import shutil

import PIL
import tifffile

from .ioutils import *
from .structure import Structure
from .utils import correct_phase_confocal


class SarcAsM(Structure):
    """
    Base class for sarcomere structural and functional analysis
    """

    def __init__(self, filename, restart=False, channel=None, correct_phase_leica=False, auto_save=True, use_gui=False,
                 **info):
        """
        Initialize cell object

        Parameters
        ----------
        filename : str
            Filename of tif-file
        restart : bool
            If True, all previous data is deleted. If False, previous analysis is loaded.
        channel : int
            Choose channel with sarcomeres in multi-color stacks. If None, all channels are loaded.
        correct_phase_leica : bool
            If True, the phase of the confocal image is corrected (only for bidirectional scan function in Leica scopes)
        auto_save : bool
            If True, analysis results are stored in dictionary after each step.
        use_gui : bool
            Set True when using SarcAsM from GUI.
        info : kwargs
            Additional information added to the metadata (**kwargs), e.g. cell_line='wt', drug_conc=10, ...
        """
        if not os.path.exists(filename):
            raise FileExistsError(f'The file {filename} does not exist!')
        self.metadata = None
        self.filename = filename
        self.auto_save = auto_save
        self.channel = channel
        self.correct_phase_leica = correct_phase_leica
        self.use_gui = use_gui
        self.info = info
        # create folders
        self.folder = os.path.splitext(filename)[0] + '/'
        self.data_folder = self.folder + '/data/'
        self.analysis_folder = self.folder + '/analysis/'
        # remove
        if restart and os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.analysis_folder, exist_ok=True)
        # load class for structure
        super().__init__()

        # due to renaming of meta_cell.json to metadata.json -> change name of meta data file todo remove later
        if os.path.exists(self.data_folder + "meta_cell.json"):
            os.rename(self.data_folder + "meta_cell.json", self.data_folder + "metadata.json")

        # path for segmented z-bands and cell mask
        if os.path.exists(self.folder + 'sarcomeres.tif'):
            self.file_sarcomeres = self.folder + 'sarcomeres.tif'
        else:
            self.file_sarcomeres = None
        if os.path.exists(self.folder + 'cell_mask.tif'):
            self.file_cell_mask = self.folder + 'cell_mask.tif'
        else:
            self.file_cell_mask = None

        # check whether metadata file already exists
        if os.path.exists(self.__get_meta_data_file()) and not restart:
            self._load_meta_data()

        else:
            # correct phase for bidirectional confocal scanner (Leica confocal microscopes)
            if self.correct_phase_leica:
                correct_phase_confocal(self.filename)

            # create metadata dict
            self.create_meta_data()

    def __get_meta_data_from_tif(self):
        """ Read metadata from tif (works only with imagej metadata - not generic, optimized for data from
        Leica and Yokogawa microscopes) """

        def __get_shape_from_file(file):
            _data = self.read_imgs()
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

        # attempt to extract metadata from tif file
        with tifffile.TiffFile(self.filename) as tif:
            if hasattr(tif, 'imagej_metadata'):
                # frame number
                if tif.imagej_metadata is not None:
                    if 'frames' in tif.imagej_metadata.keys():
                        frames = tif.imagej_metadata['frames']
                    elif 'slices' in tif.imagej_metadata.keys():
                        frames = tif.imagej_metadata['slices']
                    else:
                        frames, _ = __get_shape_from_file(self.filename)
                else:
                    frames, _ = __get_shape_from_file(self.filename)
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
                frames, _ = __get_shape_from_file(self.filename)
        # pixelsize xy
        if 'pixelsize' in self.info.keys():
            pixelsize = self.info['pixelsize']
            _, size = __get_shape_from_file(self.filename)
        else:
            with PIL.Image.open(self.filename) as tif:
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
                    _, size = __get_shape_from_file(self.filename)
            if pixelsize is None and self.use_gui == False:
                raise MetaDataError('Pixel size could not be extracted from tif file. '
                                    'Please enter manually by, e.g., SarcAsM(filename, pixelsize=0.1)!')
        if frametime is None:
            print('frametime could not be extracted from tif file. Please enter manually if needed for analysis!')
        return frames, size, pixelsize, frametime, timestamps

    def __get_info_from_filename(self):
        """Extract information from filename - specific for format YYYYMMDD_dayXX_VVkPa.tif and create hash as ID"""
        # todo make more general?
        file_name = os.path.basename(self.filename)
        try:
            date = re.findall('(\d{6,8})', self.filename)[0]
        except:
            date = None
        try:
            days_culture = int(re.findall('day(\d+)', self.filename)[0])
        except:
            days_culture = None
        try:
            well = re.findall('[dish|well](\d+)', self.filename)[0]
        except:
            well = None
        try:
            substrate = re.findall('(\d+)kpa', self.filename, re.IGNORECASE)[0]
        except:
            substrate = None
        try:
            string = str(date) + str(days_culture) + str(substrate) + str(well)
            measurement_id = hashlib.md5(string.encode()).hexdigest()
        except:
            measurement_id = None
        try:
            string = str(file_name) + str(date) + str(days_culture) + str(substrate) + str(well)
            file_id = hashlib.md5(string.encode()).hexdigest()
        except:
            file_id = None
        return file_name, date, days_culture, well, substrate, measurement_id, file_id

    def __info__(self):
        pass
        # todo print small summary - that's a great idea!

    def __get_meta_data_file(self, is_temp_file=False):
        # renaming meta_cell.json to metadata.json todo remove when complete
        if is_temp_file:
            return self.data_folder + "metadata.temp.json"
        else:
            return self.data_folder + "metadata.json"

    def create_meta_data(self):
        """Create metadata for tif-file"""
        print('Creating metadata...')
        # get metadata from tif file
        frames, size, pixelsize, frametime, timestamps = self.__get_meta_data_from_tif()
        # create time array
        if frametime is not None:
            time = np.arange(0, frames * frametime, frametime)
        else:
            time = None
        # get info from filename
        file_name, date, days_culture, well, substrate, measurement_id, file_id = self.__get_info_from_filename()
        # create dictionary for meta data and info (keyword args)
        self.metadata = {'file_name': file_name, 'size': size, 'pixelsize': pixelsize, 'frametime': frametime,
                         'frames': frames, 'file_path': self.filename, 'substrate': substrate, 'date': date,
                         'days_culture': days_culture, 'phase_corrected': self.correct_phase_leica, 'time': time,
                         'well': well, 'measurement_id': measurement_id, 'file_id': file_id,
                         'timestamps': timestamps}
        self.metadata.update(self.info)
        self.store_meta_data(override=True)

    def store_meta_data(self, override=True):
        # IOUtils.serialize_pickle(self.metadata,self.data_folder + 'meta_cell.p')
        # only store if path doesn't exist or override is true
        if override or (not os.path.exists(self.__get_meta_data_file())):
            IOUtils.json_serialize(self.metadata, self.__get_meta_data_file())
            self.commit()

    def read_imgs(self, timepoint=None):
        """Load tif file, and optionally select channel"""
        if timepoint is None:
            data = tifffile.imread(self.filename)
        else:
            data = tifffile.imread(self.filename, key=timepoint)
        if self.channel is not None:
            if data.ndim == 3:
                data = data[:, :, self.channel]
            elif data.ndim == 4:
                data = data[:, :, :, self.channel]
        return data

    def commit(self):
        """
        commit data (either rename temp-file to normal data file name or just write it again+remove temp-file)
        """
        super().commit()
        if os.path.exists(self.__get_meta_data_file(is_temp_file=True)):
            if os.path.exists(self.__get_meta_data_file()):
                os.remove(self.__get_meta_data_file())
            os.rename(self.__get_meta_data_file(is_temp_file=True), self.__get_meta_data_file())

    def _load_meta_data(self):
        if os.path.exists(self.__get_meta_data_file()):
            # persistent file exists, try using it
            try:
                self.metadata = IOUtils.json_deserialize(self.__get_meta_data_file())
            except:
                if os.path.exists(self.__get_meta_data_file(is_temp_file=True)):
                    self.metadata = IOUtils.json_deserialize(self.__get_meta_data_file(is_temp_file=True))
        else:
            # no persistent file exists, look if a temp-file exists
            if os.path.exists(self.__get_meta_data_file()):
                self.metadata = IOUtils.json_deserialize(self.__get_meta_data_file())
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


# exception if extraction of metadata from tif-file fails
class MetaDataError(Exception):
    pass
