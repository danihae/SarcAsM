import os.path
import types
import numpy as np

import pandas as pd
from tqdm import tqdm as tqdm

from .core import SarcAsM
from .motion import Motion


class MultiStructureAnalysis:
    """
    Class for multi-cell comparison of structure

    Parameters:
    - list_files (list): List of tif files
    - folder (str): Path to a folder to store data and results
    - experiment (str, optional): Name of the experiment
    - load_data (bool, optional): Whether to load the dataframe from previous analysis from the data folder
    - **conditions: Keyword arguments with regex functions to extract information from the filename

    Attributes:
    - folder (str): Path to the folder with data and results
    - experiment (str): Name of the experiment
    - files (list): List of tif files
    - conditions (dict): Keyword arguments with regex functions to extract information from the filename
    - data (pandas.DataFrame): DataFrame to store the structure data

    Methods:
    - get_data(structure_keys=None, meta_keys=None): Iterate files and get structure data
    - save_data(): Save the DataFrame to the folder
    - load_data(): Load the DataFrame from the folder
    """

    def __init__(self, list_files, folder, experiment=None, load_data=False, **conditions):
        self.folder = folder
        self.experiment = experiment
        self.files = list_files
        self.conditions = conditions
        self.data = None

        if load_data:
            self.load_data()

    def get_data(self, structure_keys=None, meta_keys=None):
        """
        Iterate files and get structure data

        Parameters:
        - structure_keys (list, optional): List of keys to extract structure data
        - meta_keys (list, optional): List of keys to extract metadata

        Returns:
        None
        """

        self.data = []
        for i, tif_file in enumerate(tqdm(self.files)):
            try:
                sarc_obj = SarcAsM(tif_file)
                dict_i = get_structure_dict(sarc_obj, meta_keys, structure_keys, experiment=self.experiment,
                                            **self.conditions)
                self.data.append(dict_i)
            except Exception as e:
                print(f'{tif_file} failed!')
                print(repr(e))

        self.data = pd.DataFrame.from_records(self.data)
        self.save_data()

    def save_data(self):
        """
        Save the DataFrame to the data folder

        Returns:
        None
        """

        self.data.to_pickle(self.folder + f'data_structure.pd')

    def load_data(self):
        """
        Load the DataFrame from the data folder

        Returns:
        None
        """
        if os.path.exists(self.folder + f'data_structure.pd'):
            self.data = pd.read_pickle(self.folder + f'data_structure.pd')
        else:
            raise FileExistsError('Data from previous analysis does not exist and cannot be loaded. '
                                  'Set load_data=False.')


class MultiLOIAnalysis:
    """
    Class for multi-LOI comparison

    Parameters:
    - list_lois (list): List of tuples containing tif file paths and LOI names
    - folder (str): Path to a folder to store data and results
    - load_data (bool, optional): Whether to load the dataframe from previous analysis from the folder
    - **conditions: Keyword arguments with regex functions to extract information from the filename

    Attributes:
    - folder (str): Path to the folder with data and results
    - lois (list): List of tuples containing tif file paths and LOI names
    - conditions (dict): Keyword arguments with regex functions to extract information from the filename
    - data (pandas.DataFrame): DataFrame to store the motion data

    Methods:
    - get_data(loi_keys=None, meta_keys=None): Iterate files and get motion data
    - save_data(): Save the DataFrame to the folder
    - load_data(): Load the DataFrame from the folder
    """

    def __init__(self, list_lois, folder, load_data=False, **conditions):
        self.folder = folder
        self.lois = list_lois
        self.conditions = conditions
        self.data = None

        if load_data:
            self.load_data()

    def get_data(self, loi_keys=None, meta_keys=None):
        """
        Iterate files and get motion data

        Parameters:
        - loi_keys (list, optional): List of keys to extract motion data
        - meta_keys (list, optional): List of keys to extract metadata

        Returns:
        None
        """

        self.data = []
        for tif_file, loi_name in tqdm(self.lois):
            try:
                motion_obj = Motion(tif_file, loi_name)
                dict_i = get_motion_dict(motion_obj, meta_keys, loi_keys, **self.conditions)
                self.data.append(dict_i)
            except Exception as e:
                print(f'{tif_file}, {loi_name} failed!')
                print(repr(e))

        self.data = pd.DataFrame.from_records(self.data)
        self.save_data()

    def save_data(self):
        """
        Save the DataFrame to the data folder

        Returns:
        None
        """

        self.data.to_pickle(self.folder + 'data_motion.pd')

    def load_data(self):
        """
        Load the DataFrame from the data folder

        Returns:
        None
        """
        if os.path.exists(self.folder + 'data_motion.pd'):
            self.data = pd.read_pickle(self.folder + 'data_motion.pd')
        else:
            raise FileExistsError('Data from previous analysis does not exist and cannot be loaded. '
                                  'Set load_data=False.')


def get_structure_dict(sarc_obj: SarcAsM, meta_keys=None, structure_keys=None, **conditions):
    """Create dictionary structure and metadata features from SarcAsM object, additionally accepts keyword args for
    condition

    Parameters
    ----------
    sarc_obj : object
        Object of SarcAsM class or Motion class.
    meta_keys : list
        List of metadata keys
    structure_keys : list
        List of structure keys
    conditions : kwargs
        Keyword arguments to add information to dictionary (e.g., "cell_line"= "wt", "info_xyz"=42)
    """
    # get data from metadata and structure dicts
    if structure_keys is None:
        structure_keys = structure_keys_default
    if meta_keys is None:
        meta_keys = meta_keys_default
    missing_meta_keys = [key for key in meta_keys if key not in sarc_obj.metadata]
    if missing_meta_keys:
        print('Missing metadata keys: ', missing_meta_keys)
    dict_metadata_select = {key: sarc_obj.metadata[key] if key in sarc_obj.metadata else np.nan for key in meta_keys}
    missing_structure_keys = [key for key in structure_keys if key not in sarc_obj.structure]
    if missing_structure_keys:
        print('Missing structure keys: ', missing_structure_keys)
    dict_structure_select = {key: sarc_obj.structure[key] if key in sarc_obj.structure else np.nan for key in structure_keys}
    dict_ = {**dict_metadata_select, **dict_structure_select}
    # add keyword args with experimental conditions, also accepts functions
    for condition, value in conditions.items():
        if isinstance(value, types.FunctionType):
            dict_[condition] = value(sarc_obj.filename)
        else:
            dict_[condition] = value
    return dict_


# function to create structure dataframe for SarcAsM object
def get_motion_dict(motion_obj: Motion, meta_keys=None, loi_keys=None, concat=False, **conditions):
    """Create dictionary of motion features and metadata from Motion object, additionally accepts keyword args for
    condition

    Parameters
    ----------
    motion_obj : Motion
        Object of Motion class for LOI analysis
    meta_keys : list
        List of metadata keys
    loi_keys : list
        List of LOI keys
    concat : bool
        If True, all 2D arrays will be concatenated to 1D arrays
    conditions : kwargs
        Keyword arguments to add to dictionary, can be any information, e.g., drug='ABC'
    """
    # get data from metadata and structure dicts
    if loi_keys is None:
        loi_keys = loi_keys_default
    if meta_keys is None:
        meta_keys = meta_keys_default
    missing_meta_keys = [key for key in meta_keys if key not in motion_obj.metadata]
    if missing_meta_keys:
        print('Missing metadata keys: ', missing_meta_keys)
    dict_metadata_select = {key: motion_obj.metadata[key] if key in motion_obj.metadata else np.nan for key in meta_keys}
    missing_loi_keys = [key for key in loi_keys if key not in motion_obj.loi_data]
    if missing_loi_keys:
        print('Missing loi keys: ', missing_loi_keys)
    dict_loi_select = {key: motion_obj.loi_data[key] if key in motion_obj.loi_data else np.nan for key in loi_keys}
    dict_ = {**dict_metadata_select, **dict_loi_select, 'loi_name': motion_obj.loi_name}
    # add keyword args with experimental conditions, also accepts functions
    for condition, value in conditions.items():
        if isinstance(value, types.FunctionType):
            dict_[condition] = value(motion_obj.filename)
        else:
            dict_[condition] = value
    # convert 2d arrays to 1d
    if concat:
        for key, value in dict_.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    dict_[key] = np.concatenate(value)
    # update filename if path changed
    dict_['tif_name'] = motion_obj.filename
    return dict_

# functions for export to .xlsx, .csv and .xml data files
# def export_structure_data(filepath, sarc_obj, meta_keys=None, structure_keys=None, reduce=True, fileformat='.xlsx'):
#     structure_dict = get_structure_dict(sarc_obj, meta_keys=meta_keys, structure_keys=structure_keys)
#     structure_df = pd.DataFrame(structure_dict)
#     if reduce:
#         structure_df = reduce_dataframe(structure_df)
#     if fileformat == '.xlsx':
#         structure_df.to_excel(filepath)
#     elif fileformat == '.csv':
#         structure_df.to_csv(filepath)
#     elif fileformat == '.xml':
#         structure_df.to_xml(filepath)
#
#
# def reduce_dataframe(df):
#     df_reduced = df.copy()
#     for key in df.keys():
#         if isinstance(df[key][0], np.ndarray):
#             df_reduced.drop(key, axis=1, inplace=True)
#     return df_reduced
#
#
# def export_motion_data(mot_obj, motion_keys, fileformat='.xlsx'):
#     pass


# define dictionary keys
meta_keys_default = ['file_id', 'file_name', 'file_path', 'date', 'frames', 'size', 'pixelsize', 'timestamps',
                     'measurement_id', 'time', 'frametime']
structure_keys_default = ['avg_intensity', 'cell_area', 'cell_area_ratio', 'domain_area_mean',
                          'domain_area_median', 'domain_area_std', 'domain_oop_mean',
                          'domain_oop_median', 'domain_oop_std', 'domain_slen_mean',
                          'domain_slen_median', 'domain_slen_std', 'myof_length_max',
                          'myof_length_mean', 'myof_length_median', 'myof_length_std',
                          'myof_msc_mean', 'myof_msc_median', 'myof_msc_std', 'n_domains',
                          'sarcomere_area', 'sarcomere_area_ratio', 'sarcomere_length_mean',
                          'sarcomere_length_median', 'sarcomere_length_std', 'sarcomere_oop',
                          'z_intensity_mean', 'z_intensity_std', 'z_lat_alignment_mean',
                          'z_lat_alignment_std', 'z_lat_dist_mean', 'z_lat_dist_std',
                          'z_lat_neighbors_mean', 'z_lat_neighbors_std', 'z_length_max',
                          'z_length_mean', 'z_length_std', 'z_oop', 'z_ratio_intensity',
                          'z_straightness_mean', 'z_straightness_std']


loi_keys_default = ['beating_rate', 'beating_rate_variability', 'contr_max', 'contr_max_avg', 'elong_max',
                    'elong_max_avg', 'equ', 'time', 'vel_contr_max', 'vel_contr_max_avg', 'vel_elong_max',
                    'vel_elong_max_avg', 'n_sarcomeres', 'n_contr', 'ratio_nans', 'popping_freq_time',
                    'popping_freq_sarcomeres', 'popping_freq', 'popping_events', 'popping_dist', 'popping_tau',
                    'popping_ks_dist_pvalue', 'popping_ks_dist_statistic',  'popping_p_dist', 'popping_p_tau',
                    'popping_ks_tau_pvalue', 'popping_ks_tau_statistic', 'time_to_peak', 'time_to_peak_avg',
                    'pearson_d_t_cycle_avg_delta_slen', 'pearson_d_t_cycle_avg_vel', 'pearson_d_t_delta_slen',
                    'pearson_d_t_vel', 'delta_slen_avg_tps', 'delta_slen_avg_tps_frame', 'corr_cycles_auto_cross',
                    'time_contr', 'time_quiet',
                    'corr_delta_slen', 'corr_vel',
                    'corr_delta_slen_serial', 'corr_delta_slen_mutual', 'corr_vel_serial', 'corr_vel_mutual',
                    'ratio_delta_slen_mutual_serial', 'ratio_vel_mutual_serial']
