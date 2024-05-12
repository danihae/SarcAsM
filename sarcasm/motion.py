import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pywt import cwt
from scipy.ndimage import binary_closing, binary_opening, label, binary_dilation
from scipy.stats import kstest, geom
from skimage.segmentation import clear_border
from trackpy import link_iter

from .contraction_net.prediction import predict_contractions as contraction_net
from .core import SarcAsM
from .ioutils import IOUtils
from .utils import Utils


class Motion(SarcAsM):
    """Class for tracking and analysis of sarcomere motion at region of interest"""

    def __init__(self, filename, loi_name, restart=False, auto_save=True):
        """
        Initialization of SarcomereAnalysis object for single LOI (Line of Interest) analysis

        Parameters
        ----------
        filename : str
            Filename of cardiomyocyte tif-movie
        loi_name : str
            Filename of LOI (only basename). All LOI files can be found by loi_files = glob.glob(cell.folder + '*.json')
        restart : bool
            If True, analysis is started from beginning, and empty LOI dictionary is initialized
        auto_save : bool
            If True, LOI dictionary is saved at end of processing steps.
        """
        super().__init__(filename)  # init super SarcAsM object

        self.loi_data = {}  # init empty dictionary
        self.loi_file = os.path.join(os.path.splitext(filename)[0], loi_name)  # folder for loi data
        self.loi_name = Motion.get_loi_name_from_file_name(loi_name)  # loi_name is the file name of the json self file

        # create folder for LOI (sub-folder in cell folder) for analysis
        self.loi_folder = os.path.join(self.folder, self.loi_name)
        os.makedirs(self.loi_folder, exist_ok=True)

        # flag to automatically save dict after processing
        self.auto_save = auto_save

        # load data dict or create new dict (of not yet analyzed or restart=True)
        self.__load_analysed_data_or_create(restart)

    def __load_analysed_data_or_create(self, restart: bool):
        # load data if already analyzed
        if os.path.exists(self.__get_loi_data_file_name(is_temp_file=False)) and not restart:
            self.load_loi_data()
        else:
            print('LOI not yet analyzed.')
            # return if the file does not exist
            if not os.path.exists(self.loi_file):
                return
            self.__create_loi_data()

    def __create_loi_data(self):
        # read file with profiles and get time array
        x_pos, y_int, y_int_raw, line, time, line_width = self.read_profile_data()
        # initialize and save dictionary
        self.loi_data = {'x_pos': x_pos, 'y_int': y_int, 'y_int_raw': y_int_raw,
                         'time': time, 'line': line, 'line_width': line_width}
        if self.auto_save:
            self.store_loi_data()

    @staticmethod
    def get_loi_name_from_file_name(filename) -> str:
        return filename.replace(".temp", "").replace("_loi", "").replace(".json", "").replace(".csv", "")

    def __get_loi_data_file_name(self, is_temp_file=False) -> str:
        if is_temp_file:
            return os.path.join(self.data_folder, self.loi_name + "_loi_data.temp.json")
        else:
            return os.path.join(self.data_folder, self.loi_name + "_loi_data.json")

    def load_loi_data(self):
        if os.path.exists(self.__get_loi_data_file_name(is_temp_file=False)):
            # persistent file exists, try using it
            try:
                self.loi_data = IOUtils.json_deserialize(self.__get_loi_data_file_name())
            except:
                if os.path.exists(self.__get_loi_data_file_name(is_temp_file=True)):
                    self.loi_data = IOUtils.json_deserialize(self.__get_loi_data_file_name(is_temp_file=True))
        else:
            # no persistent file exists, look if a temp-file exists
            if os.path.exists(self.__get_loi_data_file_name(is_temp_file=True)):
                self.loi_data = IOUtils.json_deserialize(self.__get_loi_data_file_name(is_temp_file=True))
        if self.loi_data is None or not self.loi_data:
            # self data is empty, reload from self file
            if os.path.exists(self.__get_loi_data_file_name()):
                os.remove(self.__get_loi_data_file_name())
            if not os.path.exists(self.loi_file):
                return
            self.__create_loi_data()
            if not self.auto_save:
                self.store_loi_data()
        self.commit()

    def store_loi_data(self):
        """Save LOI data"""
        IOUtils.json_serialize(self.loi_data, self.__get_loi_data_file_name())
        print('LOI data saved!')

    def commit(self):
        if os.path.exists(self.__get_loi_data_file_name(is_temp_file=True)):
            if os.path.exists(self.__get_loi_data_file_name(is_temp_file=False)):
                os.remove(self.__get_loi_data_file_name(is_temp_file=False))
            os.rename(self.__get_loi_data_file_name(is_temp_file=True), self.__get_loi_data_file_name())
        pass

    def read_profile_data(self):
        """
        Read z-band profile data
        """
        # return if the file does not exist
        if not os.path.exists(self.loi_file):
            return

        elif ".json" in self.loi_file:
            data = IOUtils.json_deserialize(self.loi_file)
            if 'length' not in data.keys():
                data['length'] = np.sqrt((data['line_start_x'] - data['line_end_x']) ** 2 +
                                         (data['line_start_y'] - data['line_end_y']) ** 2) * self.metadata['pixelsize']
                data['line'] = np.asarray(
                    [[data['line_start_x'], data['line_start_y']], [data['line_end_x'], data['line_end_y']]])
            # x_pos is 0 until line length(included)
            x_pos = np.linspace(0, data['length'], data['profiles'].shape[1])
            no_frames = len(data['profiles'])
            time = np.arange(0, no_frames * self.metadata['frametime'], self.metadata['frametime'])
            if 'profiles_raw' not in data.keys():
                data['profiles_raw'] = None
            return (x_pos, data['profiles'], data['profiles_raw'], data['line'], time,
                    np.int8(data['linewidth']))
        else:
            raise ValueError('LOI-File is not .json')

    def detekt_peaks(self, thres=0.05, min_dist=1., width=7, plot=False):
        """
        Detect peaks of z-band intensity profiles

        Parameters
        ----------
        thres : float
            Threshold for peak finder
        min_dist : float
            Minimal distance of z-band peaks in µm
        width : int
            Width of interval around peak for precise determination of peak center, in pixels
        plot : bool
            If True, peak detection result for each time point are plotted (for parameter optimization)

        """
        peaks = []
        if not self.loi_data:
            raise ValueError('loi_data is not initialized, create intensity profiles first')
        self.loi_data['parameters.detect_peaks'] = {'thresh': thres, 'min_dist': min_dist, 'width': width}
        min_dist_frames = int(min_dist / self.metadata['pixelsize'])
        for i, y in enumerate(self.loi_data['y_int']):

            peaks_i = Utils.peakdetekt(self.loi_data['x_pos'], y, thres, min_dist_frames, width)
            peaks.append(peaks_i[~np.isnan(peaks_i)])

            if plot:
                plt.figure(figsize=(8, 3))
                plt.plot(self.loi_data['x_pos'], self.loi_data['y_int'][i], c='b')
                for peak_pos in peaks_i:
                    plt.axvline(peak_pos, linestyle=':', color='r')
                plt.ylabel('Intensity')
                plt.xlabel('x in µm')
                plt.tight_layout()
                plt.show()

        # save peaks
        self.loi_data['peaks'] = peaks
        if self.auto_save:
            self.store_loi_data()

    def track_z_bands(self, search_range=1, memory_tracking=10, memory_interpol=6, t_range=None, z_range=None,
                      min_length=5, filter_params=(13, 5)):
        """
        Track peaks of intensity profile over time with Crocker-Grier algorithm from TrackPy package

        Parameters
        ----------
        search_range : float
            Search range for tracking algorithm (see documentation of trackpy package)
        memory_tracking : int
            Memory for tracking algorithm, in frames (see documentation of trackpy package)
        memory_interpol : int
            Memory (max interval) to interpolate gaps in trajectories, in frames (see documentation of trackpy package)
        t_range : float[int, int]
            If not None, select time-interval of data, in frames
        z_range : float[int, int]
            If not None, select range of z-bands
        min_length : float
            Minimal length of z-band trajectory in seconds. Shorter trajectories will not be deleted but set to np.nan.
        filter_params : tuple(float, float)
            Parameters window length and poly order of Savitzky-Golay filter to smooth z position
        """
        self.loi_data['parameters.track_peaks'] = {'search_range': search_range, 'memory_tracking': memory_tracking,
                                                   'memory_interpol': memory_interpol, 't_range': t_range,
                                                   'z_range': z_range}
        peaks = self.loi_data['peaks'].copy()
        # make x,y array
        peaks = [np.asarray([p, np.zeros_like(p)]).T for p in peaks]
        # make iterator of peaks
        peaks_iter = iter(peaks)

        # Crocker-Grier linking algorithm
        trajs_idx = pd.DataFrame(
            link_iter(peaks_iter, search_range=search_range, memory=memory_tracking, link_strategy='auto'))[1]
        trajs_idx = trajs_idx.to_numpy()

        # sort array into z-band trajectories
        z_pos = np.zeros((len(trajs_idx[0]), len(self.loi_data['time']))) * np.nan
        for t, idx in enumerate(trajs_idx):
            for n, j in enumerate(idx):
                if j < len(trajs_idx[0]):
                    z_pos[j][t] = self.loi_data['peaks'][t][n]

        # interpolate gaps in trajectories (interpolate with pandas)
        z_pos = pd.DataFrame(z_pos)
        z_pos = z_pos.interpolate(limit=memory_interpol, axis=1)
        z_pos = z_pos.to_numpy()

        # set short trajectories (len<min_length) to np.nan
        len_z_pos = np.count_nonzero(~np.isnan(z_pos), axis=1)
        z_pos = z_pos[len_z_pos > int(min_length / self.metadata['frametime'])]

        # set t range and z range
        if t_range is not None:
            z_pos = z_pos[:, t_range[0]:t_range[1]]
            self.loi_data['time'] = self.loi_data['time'][:t_range[1] - t_range[0]]
        if z_range is not None:
            z_pos = z_pos[z_range[0]:z_range[1], :]

        # filter z positions
        z_pos_filt = z_pos.copy()
        z_pos_filt = Utils.nan_sav_golay(z_pos_filt, window_length=filter_params[0], polyorder=filter_params[1])

        # calculate sarcomere lengths
        slen = np.diff(z_pos_filt, axis=0)

        # save data
        dict_temp = {'z_pos_raw': z_pos, 'z_pos': z_pos_filt, 'slen': slen,
                     'parameters.track_z_bands': {'search_range': search_range, 'memory_tracking': memory_tracking,
                                                  'memory_interpol': memory_interpol, 't_range': t_range,
                                                  'z_range': z_range, 'min_length': min_length,
                                                  'filter_params': filter_params}}

        self.loi_data.update(dict_temp)
        if self.auto_save:
            self.store_loi_data()

    def detect_analyze_contractions(self, model=None, threshold=0.6, slen_lims=(1.2, 3),
                                    n_sarcomeres_min=4,
                                    buffer_frames=3, contr_time_min=0.2, merge_time_max=0.05):
        """
        Detect contractions from contraction time-series using convolutional neural network and analyze beating

        1. Predict contractions / contraction state (0 = quiescent, 1 = contracting) from sarcomere lengths (average or percentile)
        2. Optimize state by morphological closing and opening (minimal time of contraction cycle=contr_time_min,
            merge contractions closer than merge_time_max). Remove cycles at very beginning or end (buffer_frames).
        3. Characterize state: obtain start times of contr. cycles (start_contractions_frame in frames, start_contr in s),
            number of cycles (n_contr), label contraction cycles (1 to n_contr), duration of contr. cycles (time_contractions)

        Parameters
        ----------
        model : str
            Neural network parameters (.pth file)
        threshold : float
            Binary threshold for contraction state (0, 1) after prediction
        slen_lims : tuple(float, float)
            Minimal and maximal sarcomere lengths, sarcomere outside interval are set to NaN
        n_sarcomeres_min : int
            Minimal number of sarcomeres, if lower, contraction state is set to 0.
        buffer_frames : int
            Remove contraction cycles / contractions within "buffer_frames" frames to start and end of time-series
        contr_time_min : float
            Minimal time of contraction in seconds. If smaller, contraction is removed.
        merge_time_max : float
            Maximal time between two contractions. If smaller, two contractions are merged to one.
        """

        # select weights for convolutional neural network
        if model is None or model is 'default':
            model = self.model_dir + 'model_ContractionNet.pth'
        # detect contractions with convolutional neural network (0 = quiescence, 1 = contraction)
        contr = self.predict_contractions(self.loi_data['z_pos'], self.loi_data['slen'], model,
                                          threshold=threshold)
        # edit contractions
        # filter sarcomeres by sarcomere lengths and set to 0 if less sarcomeres than n_sarcomere_min
        slen = np.diff(self.loi_data['z_pos'], axis=0)
        slen[(slen < slen_lims[0]) | (slen > slen_lims[1])] = np.nan
        n_sarcomeres_time = np.count_nonzero(~np.isnan(slen), axis=0)
        contr[n_sarcomeres_time < n_sarcomeres_min] = 0
        # merge very close contractions and remove short contractions
        contr = binary_opening(
            binary_closing(contr, structure=np.ones(int(merge_time_max / self.metadata['frametime']))),
            structure=np.ones(int(contr_time_min / self.metadata['frametime'])))
        # remove incomplete contractions at the beginning and end of time series
        contr = clear_border(contr, buffer_size=buffer_frames)

        # analyze contractions
        start_contr_frame = np.where(np.diff(contr.astype('float32')) > 0.5)[0]
        start_contr = start_contr_frame * self.metadata['frametime']
        labels_contr, n_contr = label(contr)
        time_contr = np.asarray(
            [np.count_nonzero(labels_contr == i) for i in np.unique(labels_contr)[1:]]) * \
                     self.metadata['frametime']
        beating_rate = 1 / np.mean(np.diff(start_contr))
        beating_rate_variability = np.std(np.diff(start_contr))

        # analyze quiescent period
        quiet = 1 - contr.copy()
        # remove incomplete quiescent periods at the beginning and end of time series
        quiet = clear_border(quiet, buffer_size=buffer_frames)
        start_quiet_frame = np.where(np.diff(quiet.astype('float32')) > 0.5)[0]
        start_quiet = start_quiet_frame * self.metadata['frametime']
        labels_quiet, n_quiet = label(quiet)
        time_quiet = np.asarray(
            [np.count_nonzero(labels_quiet == i) for i in np.unique(labels_quiet)[1:]]) * \
                     self.metadata['frametime']
        # time of full contraction cycles (equivalent to 1/beating_rate)
        time_cycle = time_contr[:-1] + time_quiet

        # store in LOI dict
        dict_temp = {'parameters.detect_analyze_contractions': {'model': model, 'slen_lims': slen_lims,
                                                                'n_sarcomeres_min': n_sarcomeres_min,
                                                                'buffer_frames': buffer_frames,
                                                                'contr_time_min': contr_time_min,
                                                                'merge_time_max': merge_time_max},
                     'contr': contr, 'start_contr_frame': start_contr_frame, 'start_contr': start_contr,
                     'quiet': quiet, 'start_quiet_frame': start_quiet_frame, 'start_quiet': start_quiet,
                     'labels_contr': labels_contr, 'labels_quiet': labels_quiet,
                     'time_contr': time_contr, 'time_quiet': time_quiet, 'time_cycle': time_cycle,
                     'n_contr': n_contr, 'n_quiet': n_quiet,
                     'beating_rate_variability': beating_rate_variability, 'beating_rate': beating_rate, }
        self.loi_data.update(dict_temp)

        if self.auto_save:
            self.store_loi_data()

    def get_trajectories(self, slen_lims=(1.2, 3.), filter_params_vel=(13, 5), dilate_contr=0, equ_lims=(1.5, 2.2)):
        """
        1. Calculate sarcomere lengths (single and avg) and filter too large and too small values (slen_lims).
        2. Calculate sarcomere velocities (single and avg), prior smoothing of s'lengths with Savitzky-Golay filter
            (filter_params_vel)
        3. Calculate sarcomere equilibrium lengths (equ) and delta_slen

        Parameters
        ----------
        slen_lims : tuple(float, float)
            Lower and upper limits of sarcomere lengths, values outside are set to nan
        filter_params_vel : tuple(int, int)
            Window length and poly order for Savitky-Golay filter for smoothing of delta_slen prior to differentiation
            to obtain sarcomere velocities
        dilate_contr : float
            Dilation time (in seconds) of contraction time-series to shorten time-interval during diastole at which the sarcomere
            equilibrium lengths are determined
        equ_lims : tuple(float, float)
            Lower and upper limits of sarcomere equilibrium lengths, values outside are set to nan
        """
        # calculate sarcomere lengths
        slen = np.diff(self.loi_data['z_pos'], axis=0)
        slen[(slen < slen_lims[0]) | (slen > slen_lims[1])] = np.nan
        slen_avg = np.nanmean(slen, axis=0)
        n_sarcomeres = slen.shape[0]
        n_sarcomeres_time = np.count_nonzero(~np.isnan(slen), axis=0)
        frametime = self.metadata['frametime']

        # smooth slen with sav. golay filter and calculate velocity
        vel = Utils.custom_diff(Utils.nan_sav_golay(slen, filter_params_vel[0], filter_params_vel[1]), frametime)
        vel_avg = np.nanmean(vel, axis=0)

        # calculate sarcomere equ length and delta sarcomere length
        dilate_contr = int(dilate_contr * 2 / self.metadata['frametime'])
        if dilate_contr == 0:
            contr_dilated = self.loi_data['contr']
        elif dilate_contr > 0:
            contr_dilated = binary_dilation(self.loi_data['contr'],
                                            structure=np.ones(dilate_contr))
        else:
            raise ValueError(f'Parameter dilate_contr={dilate_contr} not valid!')

        equ = np.asarray([np.nanmedian(s[contr_dilated == 0]) for s in slen])
        equ[(equ < equ_lims[0]) | (equ > equ_lims[1])] = np.nan
        delta_slen = np.asarray([slen[i] - equ[i] for i in range(len(equ))])
        delta_slen_avg = np.nanmean(delta_slen, axis=0)
        if np.count_nonzero(delta_slen) > 0:
            ratio_nans = np.count_nonzero(np.isnan(delta_slen)) / np.count_nonzero(delta_slen)
        else:
            ratio_nans = np.nan

        # store data in LOI dictionary
        dict_temp = {
            'parameters.get_sarcomere_trajectories': {'slen_lims': slen_lims,
                                                      'filter_params_vel': filter_params_vel},
            'slen': slen, 'slen_avg': slen_avg, 'vel': vel, 'vel_avg': vel_avg, 'n_sarcomeres': n_sarcomeres,
            'n_sarcomeres_time': n_sarcomeres_time, 'equ': equ, 'delta_slen': delta_slen,
            'delta_slen_avg': delta_slen_avg, 'ratio_nans': ratio_nans}
        self.loi_data.update(dict_temp)
        if self.auto_save:
            self.store_loi_data()

    def analyze_trajectories(self):
        """ Analyze sarcomere single and average trajectories (extrema of sarcomeres contraction and velocity) """

        # initialize arrays
        # maximal contraction
        contr_max = np.zeros((len(self.loi_data['delta_slen']), self.loi_data['n_contr'])) * np.nan
        contr_max_avg = np.zeros(self.loi_data['n_contr']) * np.nan
        # maximal elongation
        elong_max = np.zeros_like(contr_max) * np.nan
        elong_max_avg = np.zeros_like(contr_max_avg) * np.nan
        # maximal velocity in both directions
        vel_contr_max = np.zeros_like(contr_max) * np.nan
        vel_elong_max = np.zeros_like(contr_max) * np.nan
        vel_contr_max_avg = np.zeros_like(contr_max_avg) * np.nan
        vel_elong_max_avg = np.zeros_like(contr_max_avg) * np.nan
        # time to peak (0% to 100%)
        time_to_peak = np.zeros_like(contr_max) * np.nan
        time_to_peak_avg = np.zeros_like(contr_max_avg) * np.nan
        # relaxation time (100% to 0%)
        time_relax = np.zeros_like(contr_max) * np.nan
        time_relax_avg = np.zeros_like(contr_max_avg) * np.nan

        # iterate single sarcomeres
        labels_contr = self.loi_data['labels_contr']
        for j, delta_j in enumerate(self.loi_data['delta_slen']):
            vel_j = self.loi_data['vel'][j]
            for i in range(self.loi_data['n_contr']):
                # get time-series of one contraction cycle (start to start)
                delta_i = delta_j[labels_contr == i + 1]
                vel_i = vel_j[labels_contr == i + 1]
                # find extrema
                contr_max[j][i] = np.nanmin(delta_i)
                elong_max[j][i] = np.nanmax(delta_i)
                vel_contr_max[j][i] = np.nanmin(vel_i)
                vel_elong_max[j][i] = np.nanmax(vel_i)
                # time to peak
                if np.count_nonzero(np.isnan(delta_i)) == 0:
                    time_to_peak[j][i] = np.nanargmin(delta_i) * self.metadata['frametime']
                    time_relax[j][i] = (len(delta_i) - np.nanargmin(delta_i)) * self.metadata['frametime']

        # average contraction
        for i in range(self.loi_data['n_contr']):
            # get time-series of one contraction cycle (start to start)
            delta_i = self.loi_data['delta_slen_avg'][labels_contr == i + 1]
            vel_i = self.loi_data['vel_avg'][labels_contr == i + 1]
            # find extrema
            contr_max_avg[i] = np.nanmin(delta_i)
            elong_max_avg[i] = np.nanmax(delta_i)
            vel_contr_max_avg[i] = np.nanmin(vel_i)
            vel_elong_max_avg[i] = np.nanmax(vel_i)
            # time to peak
            if np.count_nonzero(np.isnan(delta_i)) == 0:
                time_to_peak_avg[i] = np.nanargmin(delta_i) * self.metadata['frametime']
                time_relax_avg[i] = (len(delta_i) - np.nanargmin(delta_i)) * self.metadata['frametime']

        # calculate surplus motion index
        self.surplus_motion_index()

        # save data in LOI dict
        self.loi_data.update({'contr_max': contr_max, 'elong_max': elong_max, 'vel_contr_max': vel_contr_max,
                              'vel_elong_max': vel_elong_max, 'contr_max_avg': contr_max_avg,
                              'elong_max_avg': elong_max_avg, 'vel_contr_max_avg': vel_contr_max_avg,
                              'vel_elong_max_avg': vel_elong_max_avg, 'time_to_peak': time_to_peak,
                              'time_to_peak_avg': time_to_peak_avg, 'time_relax': time_relax,
                              'time_relax_avg': time_relax_avg})
        if self.auto_save:
            self.store_loi_data()

    def surplus_motion_index(self):
        """Calculate surplus motion index (SMI) for sarcomere motion: average distance traveled by
        individual sarcomeres contractions divided by distance traveled by sarcomere average"""

        vel = self.loi_data['vel']
        vel_avg = self.loi_data['vel_avg']
        contr = self.loi_data['contr']

        # label contractions
        contraction_labels, n_contr = label(contr)

        # define arrays
        abs_motion_single = np.zeros((n_contr, vel.shape[0])) * np.nan
        abs_motion_avg = np.zeros(n_contr) * np.nan

        # iterate contractions
        for i, contraction_i in enumerate(np.arange(1, n_contr + 1)):
            vel_i = vel[:, contraction_labels == contraction_i]
            vel_avg_i = vel_avg[contraction_labels == contraction_i]
            abs_motion_single_i = np.sum(np.abs(vel_i), axis=1) * self.metadata['frametime']
            abs_motion_avg_i = np.sum(np.abs(vel_avg_i)) * self.metadata['frametime']
            abs_motion_single[i] = abs_motion_single_i
            abs_motion_avg[i] = abs_motion_avg_i

        # calculate surplus motion index per contraction cycle and store in dict
        smi = np.nanmean(abs_motion_single) / np.nanmean(abs_motion_avg)
        self.loi_data['smi'] = smi
        if self.auto_save:
            self.store_loi_data()

    def analyze_popping(self, thres_popping=0.25):
        """Analyze sarcomere popping - popping if elongation larger than thres_popping

        Parameters
        ----------
        thres_popping : float
            Threshold above which sarcomere is identified as popping, in µm beyond equilibrium length
        """
        # identify popping events
        elong_max = self.loi_data['elong_max']
        popping = np.zeros_like(elong_max, dtype='bool')
        popping[elong_max > thres_popping] = 1

        # calculate popping frequencies
        freq_time = np.mean(popping, axis=0)
        freq_sarcomeres = np.mean(popping, axis=1)
        freq = np.mean(popping)

        # dictionary
        dict_popping = {'popping_freq_time': freq_time, 'popping_freq_sarcomeres': freq_sarcomeres,
                        'popping_freq': freq, 'popping_events': popping}

        popping_events = dict_popping['popping_events']
        idxs_popping_s, idxs_popping_c = np.where(popping_events == 1)

        # inter sarcomere distance of popping events in each contraction cycle
        cycles = np.unique(idxs_popping_c)
        dist = [np.diff(idxs_popping_s[idxs_popping_c == t]) for t in cycles]
        dist = np.concatenate(dist) if dist else []

        # time gap between popping events of the same sarcomere
        sarcomeres = np.unique(idxs_popping_s)
        tau = [np.diff(idxs_popping_c[idxs_popping_s == s]) for s in sarcomeres]
        tau = np.concatenate(tau) if tau else []

        dist = np.array(dist)  # convert dist to numpy array
        tau = np.array(tau)  # convert tau to numpy array

        p_dist = 1 / np.mean(dist) if dist.size != 0 else 0
        p_tau = 1 / np.mean(tau) if tau.size != 0 else 0

        if dist.size != 0:
            kstest_result_dist = kstest(dist, geom(p_dist).cdf)
        else:
            kstest_result_dist = (np.nan, np.nan)

        if tau.size != 0:
            kstest_result_tau = kstest(tau, geom(p_tau).cdf)
        else:
            kstest_result_tau = (np.nan, np.nan)

        dict_popping.update({'popping_p_dist': p_dist, 'popping_p_tau': p_tau,
                             'popping_ks_dist_statistic': kstest_result_dist[0],
                             'popping_ks_dist_pvalue': kstest_result_dist[1],
                             'popping_ks_tau_statistic': kstest_result_tau[0],
                             'popping_ks_tau_pvalue': kstest_result_tau[1],
                             'popping_tau': tau, 'popping_dist': dist})

        self.loi_data.update(dict_popping)
        if self.auto_save:
            self.store_loi_data()

    def correlation_mutual_serial(self):
        """
        Computes the Pearson correlation coefficients for sarcomere motion patterns (∆SL and V) across different contraction
        cycles and between sarcomeres within the same cycle to analyze static and stochastic heterogeneity in sarcomere dynamics.
        It calculates the average serial (r_s) and mutual (r_m) correlation coefficients, and introduces the ratio R of serial
        to mutual correlations to distinguish between static and stochastic heterogeneity. The function updates the instance's
        loi_data with correlation data, including the calculated R values, and stores the data if auto_save is enabled.
        """
        if self.loi_data['n_contr'] > 0:
            time_contr_median = int(np.median(self.loi_data['time_contr']) / self.metadata['frametime'])

            corr_delta_slen = np.zeros((self.loi_data['n_sarcomeres'], self.loi_data['n_sarcomeres'],
                                        self.loi_data['n_contr'], self.loi_data['n_contr'])) * np.nan
            corr_vel = np.zeros((self.loi_data['n_sarcomeres'], self.loi_data['n_sarcomeres'],
                                 self.loi_data['n_contr'], self.loi_data['n_contr'])) * np.nan

            for i in range(self.loi_data['n_sarcomeres']):
                for j in range(self.loi_data['n_sarcomeres']):
                    if i >= j:
                        delta_slen_i = self.loi_data['delta_slen'][i]
                        vel_i = self.loi_data['vel'][i]
                        delta_slen_j = self.loi_data['delta_slen'][j]
                        vel_j = self.loi_data['vel'][j]
                        for k, contr_k in enumerate(self.loi_data['start_contr_frame'][:-1]):
                            for l, contr_l in enumerate(self.loi_data['start_contr_frame'][:-1]):
                                if k >= l:
                                    if i != j or k != l:
                                        corr_delta_slen[i, j, k, l] = \
                                            np.corrcoef(delta_slen_i[contr_k:contr_k + time_contr_median],
                                                        delta_slen_j[contr_l:contr_l + time_contr_median])[1, 0]
                                        corr_vel[i, j, k, l] = np.corrcoef(vel_i[contr_k:contr_k + time_contr_median],
                                                                           vel_j[
                                                                           contr_l:contr_l + time_contr_median])[1, 0]

            # serial correlation
            corr_delta_slen_serial = np.nanmean(np.diagonal(corr_delta_slen))
            corr_vel_serial = np.nanmean(np.diagonal(corr_vel))

            # mutual correlation
            corr_delta_slen_mutual = np.nanmean(np.diagonal(corr_delta_slen, axis1=1, axis2=2))
            corr_vel_mutual = np.nanmean(np.diagonal(corr_vel, axis1=1, axis2=2))

            # ratio R of mutual and serial correlation
            ratio_delta_slen_mutual_serial = corr_delta_slen_mutual / corr_delta_slen_serial
            ratio_vel_mutual_serial = corr_vel_mutual / corr_vel_serial

        else:
            corr_delta_slen = None
            corr_vel = None
            corr_delta_slen_serial = np.nan
            corr_vel_serial = np.nan
            corr_delta_slen_mutual = np.nan
            corr_vel_mutual = np.nan
            ratio_delta_slen_mutual_serial = np.nan
            ratio_vel_mutual_serial = np.nan

        corr_dict = {'corr_delta_slen': corr_delta_slen, 'corr_vel': corr_vel,
                     'corr_delta_slen_serial': corr_delta_slen_serial, 'corr_delta_slen_mutual': corr_delta_slen_mutual,
                     'corr_vel_serial': corr_vel_serial, 'corr_vel_mutual': corr_vel_mutual,
                     'ratio_delta_slen_mutual_serial': ratio_delta_slen_mutual_serial,
                     'ratio_vel_mutual_serial': ratio_vel_mutual_serial}
        self.loi_data.update(corr_dict)

        if self.auto_save:
            self.store_loi_data()

    def analyze_oscillations(self, min_scale=6, max_scale=180, num_scales=60, wavelet='morl', freq_thres=2, plot=False):
        """
        Analyze the oscillation frequencies of average and individual sarcomere length changes.

        Parameters
        ----------
        min_scale : float, optional
            Minimum scale to use for the wavelet transform (default is 6).
        max_scale : float, optional
            Maximum scale to use for the wavelet transform (default is 150).
        num_scales : int, optional
            Number of scales to use for the wavelet transform (default is 100).
        wavelet : str, optional
            Type of wavelet to use for the wavelet transform (default is 'morl' = Morlet wavelet).
        freq_thres : float, optional
            Frequency threshold in Hz for distinguishing low-freq. oscillations at beating rate, and high-freq.
            oscillations.
        plot : bool, optional
            If True, a plot illustrating the analysis is shown.

        Returns
        -------
        None
        """

        # Analyze oscillation frequencies of average sarcomere length change
        cfs_avg, frequencies = self.wavelet_analysis_oscillations(self.loi_data['delta_slen_avg'],
                                                                  self.metadata['frametime'],
                                                                  min_scale=min_scale,
                                                                  max_scale=max_scale,
                                                                  num_scales=num_scales,
                                                                  wavelet=wavelet)

        mask = self.loi_data['contr'] != 0
        mag_avg = np.nanmean(np.abs(cfs_avg[:, mask]), axis=1)

        # Analyze individual sarcomere oscillation frequencies
        cfs = []
        mags = []
        for d_i in self.loi_data['delta_slen']:
            cfs_i, _ = self.wavelet_analysis_oscillations(d_i,
                                                          self.metadata['frametime'],
                                                          min_scale=min_scale,
                                                          max_scale=max_scale,
                                                          num_scales=num_scales,
                                                          wavelet=wavelet)
            mag_i = np.nanmean(np.abs(cfs_i[:, mask]), axis=1)
            cfs.append(cfs_i)
            mags.append(mag_i)

        mag_all_mean, mag_all_std = np.nanmean(mags, axis=0), np.nanstd(mags, axis=0)

        freq_thres = max(freq_thres, self.loi_data['beating_rate'] * 2.1)

        # find first peak corresponding to beating rate
        peak_avg = frequencies[np.argmax(mag_avg)]
        amp_avg = np.max(mag_avg)
        mag_all_mean_1 = mag_all_mean.copy()
        mag_all_mean_1[frequencies > freq_thres] = np.nan
        peak_1_single = frequencies[np.nanargmax(mag_all_mean_1)]
        amp_1_single = np.max(mag_all_mean_1)

        # find second peak corresponding to high-frequency oscillations of individual sarcomeres
        mag_all_mean_2 = mag_all_mean.copy()
        mag_all_mean_2[frequencies < freq_thres] = np.nan
        min_freq = np.min(frequencies[frequencies >= freq_thres])
        peak_2_single = frequencies[np.nanargmax(mag_all_mean_2)]
        amp_2_single = np.max(mag_all_mean_2)
        if peak_2_single == min_freq:
            peak_2_single = np.nan
            amp_2_single = np.nan

        dict_oscill = {'parameters.analyze_oscillations': {'min_scale': 6, 'max_scale': 180, 'num_scales': 60,
                                                           'wavelet': 'morl', 'freq_thres': 2},
                       'oscill_frequencies': frequencies,
                       'oscill_cfs_avg': cfs_avg,
                       'oscill_cfs': np.asarray(cfs),
                       'oscill_magnitudes_avg': mag_avg,
                       'oscill_magnitudes': np.asarray(mags),
                       'oscill_peak_avg': peak_avg,
                       'oscill_peak_1_single': peak_1_single,
                       'oscill_peak_2_single': peak_2_single,
                       'oscill_amp_avg': amp_avg,
                       'oscill_amp_1_single': amp_1_single,
                       'oscill_amp_2_single': amp_2_single}

        self.loi_data.update(dict_oscill)

        if self.auto_save:
            self.store_loi_data()

        if plot:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(frequencies, mag_avg, c='r', label='Average')
            ax.plot(frequencies, np.asarray(mags).T, c='k', alpha=0.1)
            ax.fill_between(frequencies, mag_all_mean - mag_all_std,
                            mag_all_mean + mag_all_std, color='k', alpha=0.25)
            ax.plot(frequencies, mag_all_mean, c='k', label='Single')
            ax.axvline(self.loi_data['beating_rate'], c='k', linestyle='--', label='Beating rate')
            ax.axvspan(0, freq_thres, zorder=-5, color='silver', alpha=0.5)
            ax.axvline(peak_avg, c='b', linestyle=':', label='Peak avg 1')
            ax.axvline(peak_2_single, c='g', linestyle=':', color='gold', label='Peak 2')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Average magnitude')
            ax.legend()
            plt.tight_layout()
            plt.show()

    def full_analysis_loi(self):
        """Full analysis of LOI with default parameters"""
        auto_save_ = self.auto_save
        self.auto_save = False
        self.detekt_peaks()
        self.track_z_bands()
        self.detect_analyze_contractions()
        self.get_trajectories()
        self.analyze_trajectories()
        self.analyze_popping()
        self.auto_save = auto_save_
        self.store_loi_data()

    @staticmethod
    def predict_contractions(z_pos, slen, weights, threshold=0.33):
        """Predict contractions from motion of z-bands and sarcomere lengths, then calculate mean state and threshold to
        get more accurate estimation of contractions

        Parameters
        ----------
        z_pos : ndarray
            Time-series of Z-band positions
        slen : ndarray
            Time-series of sarcomere lengths
        weights : str
            Neural network parameters (.pth file)
        threshold : float
            Binary threshold for contraction state (0, 1)
        """
        data = np.concatenate([z_pos, slen])
        contr_all = np.asarray([contraction_net(d, weights)[0] for d in data])
        contr_mean = np.nanmean(contr_all, axis=0)
        return contr_mean > threshold

    @staticmethod
    def wavelet_analysis_oscillations(data, frametime, min_scale=6, max_scale=150, num_scales=100, wavelet='morl'):
        """
        Perform a wavelet transform of the data.

        Parameters
        ----------
        data : array_like
            1-D input signal.
        frametime : float
            Sampling period of the signal.
        min_scale : float, optional
            Minimum scale to use for the wavelet transform (default is 6).
        max_scale : float, optional
            Maximum scale to use for the wavelet transform (default is 150).
        num_scales : int, optional
            Number of scales to use for the wavelet transform (default is 200).
        wavelet : str, optional
            Type of wavelet to use for the wavelet transform (default is 'morl').

        Returns
        -------
        cfs : ndarray
            Continuous wavelet transform coefficients.
        frequencies : ndarray
            Corresponding frequencies for each scale.

        """
        # Generate a range of scales that are logarithmically spaced
        scales = np.geomspace(min_scale, max_scale, num=num_scales)

        # Perform the wavelet transform
        cfs, frequencies = cwt(data, scales, wavelet, sampling_period=frametime)

        return cfs, frequencies
