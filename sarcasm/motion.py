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

"""Tracking and analysis of sarcomere motion along a line of interest (LOI)."""

import os
import logging
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pywt import cwt
from scipy.ndimage import binary_closing, binary_opening, label, binary_dilation
from scipy.stats import kstest, geom
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import clear_border

from contraction_net.prediction import predict_contractions
from sarcasm.core import SarcAsMBase
from sarcasm.io.ioutils import IOUtils
from sarcasm.utils import Utils

logger = logging.getLogger(__name__)


class Motion(SarcAsMBase):
    """
    Tracking and analysis of sarcomere motion along a line of interest (LOI).

    Subclasses :class:`~sarcasm.core.SarcAsMBase` and accepts the same
    constructor arguments.
    """

    @classmethod
    def from_loi_data(cls, file_path: str, loi_name: str, loi_data: dict,
                      auto_save: bool = False, frametime: Union[float, None] = None) -> "Motion":
        """
        Build a :class:`Motion` from pre-computed ``loi_data``.

        Wraps a synthesized myofibril chain (an ordered ``z_pos`` / ``slen``
        built from 2D tracks, see
        :meth:`sarcasm.structure.SarcAsM.get_track_motion`) so the full LOI
        analysis and every existing LOI plot work unchanged. ``loi_data`` is
        marked ``synthetic=True``. Nothing is written to disk unless
        ``auto_save=True``.

        Parameters
        ----------
        file_path : str
            Path of the cardiomyocyte tif-movie (for metadata).
        loi_name : str
            Logical name for this (virtual) LOI.
        loi_data : dict
            Must contain at least ``z_pos`` ``(n_z, T)``, ``slen`` ``(n_z-1, T)``
            and ``time`` ``(T,)``.
        auto_save : bool, optional
            If True, create the LOI folder and persist
            ``{loi_name}_loi_data.json``. Default is False (purely in-memory view).
        frametime : float or None, optional
            Frame time (s) override, forwarded to the metadata loader. Needed
            when the movie has no embedded frametime (e.g. high-speed
            single-cell tifs). Default is None.

        Returns
        -------
        Motion
            A :class:`Motion` instance wrapping ``loi_data``.
        """
        obj = cls.__new__(cls)
        SarcAsMBase.__init__(obj, file_path, frametime=frametime)
        assert obj.metadata.frametime is not None, (
            "frametime is not defined in metadata; pass frametime=... to from_loi_data")
        obj.loi_data = dict(loi_data)
        obj.loi_data.setdefault('synthetic', True)
        obj.loi_name = Motion.get_loi_name_from_file_name(loi_name)
        obj.loi_file = os.path.join(os.path.splitext(file_path)[0], loi_name)
        obj.loi_folder = os.path.join(obj.base_dir, obj.loi_name)
        obj.auto_save = auto_save
        if auto_save:
            os.makedirs(obj.loi_folder, exist_ok=True)
            obj.store_loi_data()
        return obj

    @staticmethod
    def get_loi_name_from_file_name(file_name) -> str:
        """
        Strip temp/LOI/extension suffixes to derive the LOI name from a file name.

        Parameters
        ----------
        file_name : str
            File name to normalize.

        Returns
        -------
        str
            The bare LOI name.
        """
        return file_name.replace(".temp", "").replace("_loi", "").replace(".json", "").replace(".csv", "")

    def __get_loi_data_file_name(self, is_temp_file=False) -> str:
        """
        Build the path of the LOI data JSON file.

        Parameters
        ----------
        is_temp_file : bool, optional
            If True, return the temporary file path. Default is False.

        Returns
        -------
        str
            Absolute path of the LOI data file.
        """
        if is_temp_file:
            return os.path.join(self.data_dir, self.loi_name + "_loi_data.temp.json")
        else:
            return os.path.join(self.data_dir, self.loi_name + "_loi_data.json")

    def store_loi_data(self):
        """Save LOI data"""
        IOUtils.json_serialize(self.loi_data, self.__get_loi_data_file_name())

    def detect_analyze_contractions(self, model: Union[str, None] = None, threshold: float = 0.3,
                                    slen_lims: Tuple[float, float] = (1.2, 3), n_sarcomeres_min: int = 4,
                                    buffer_frames: int = 3, contr_time_min: float = 0.2, merge_time_max: float = 0.05):
        """
        Detect contractions from contraction time-series using convolutional neural network and analyze beating

        1. Predict contractions / contraction state (0 = quiescent, 1 = contracting) from sarcomere lengths (average or percentile)
        2. Optimize state by morphological closing and opening (minimal time of contraction cycle=contr_time_min,
            merge contractions closer than merge_time_max). Remove cycles at very beginning or end (buffer_frames).
        3. Characterize state: obtain start times of contr. cycles (start_contractions_frame in frames, start_contr in s),
            number of cycles (n_contr), label contraction cycles (1 to n_contr), duration of contr. cycles (time_contractions)

        Parameters
        ----------
        model : str or None, optional
            Path to the neural network weights (.pt file). None or 'default'
            uses the bundled ContractionNet model. Default is None.
        threshold : float, optional
            Binary threshold for contraction state (0, 1) after prediction.
            Default is 0.3.
        slen_lims : tuple of float, optional
            Minimal and maximal sarcomere lengths (µm); values outside the
            interval are set to NaN. Default is (1.2, 3).
        n_sarcomeres_min : int, optional
            Minimal number of sarcomeres; if fewer, contraction state is set
            to 0. Default is 4.
        buffer_frames : int, optional
            Remove contraction cycles within this many frames of the start and
            end of the time-series. Default is 3.
        contr_time_min : float, optional
            Minimal contraction time in seconds; shorter contractions are
            removed. Default is 0.2.
        merge_time_max : float, optional
            Maximal time between two contractions in seconds; closer
            contractions are merged. Default is 0.05.
        """

        # select weights for convolutional neural network
        if model == None or model == 'default':
            model = os.path.join(self.model_dir, 'model_ContractionNet.pt')
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
        structure_closing = np.ones(max(1, int(merge_time_max / self.metadata.frametime)))
        structure_opening = np.ones(max(1, int(contr_time_min / self.metadata.frametime)))
        contr = binary_opening(binary_closing(contr, structure=structure_closing), structure=structure_opening)
        # remove incomplete contractions at the beginning and end of time series
        contr = clear_border(contr, buffer_size=buffer_frames)

        # analyze contractions
        start_contr_frame = np.where(np.diff(contr.astype('float32')) > 0.5)[0]
        start_contr = start_contr_frame * self.metadata.frametime
        labels_contr, n_contr = label(contr)
        time_contr = np.asarray(
            [np.count_nonzero(labels_contr == i) for i in np.unique(labels_contr)[1:]]) * \
                     self.metadata.frametime
        beating_rate = 1 / np.mean(np.diff(start_contr))
        beating_rate_variability = np.std(np.diff(start_contr))

        # analyze quiescent period
        quiet = 1 - contr.copy()
        # remove incomplete quiescent periods at the beginning and end of time series
        quiet = clear_border(quiet, buffer_size=buffer_frames)
        start_quiet_frame = np.where(np.diff(quiet.astype('float32')) > 0.5)[0]
        start_quiet = start_quiet_frame * self.metadata.frametime
        labels_quiet, n_quiet = label(quiet)
        time_quiet = np.asarray(
            [np.count_nonzero(labels_quiet == i) for i in np.unique(labels_quiet)[1:]]) * \
                     self.metadata.frametime
        time_quiet_avg = np.mean(time_quiet)
        time_contr_avg = np.mean(time_contr)
        # time of full contraction cycles (equivalent to 1/beating_rate)
        time_cycle = time_contr[:-1] + time_quiet

        # store in LOI dict
        dict_temp = {'params.detect_analyze_contractions.model': model,
                     'params.detect_analyze_contractions.slen_lims': slen_lims,
                     'params.detect_analyze_contractions.n_sarcomeres_min': n_sarcomeres_min,
                     'params.detect_analyze_contractions.buffer_frames': buffer_frames,
                     'params.detect_analyze_contractions.contr_time_min': contr_time_min,
                     'params.detect_analyze_contractions.merge_time_max': merge_time_max,
                     'contr': contr, 'start_contr_frame': start_contr_frame, 'start_contr': start_contr,
                     'quiet': quiet, 'start_quiet_frame': start_quiet_frame, 'start_quiet': start_quiet,
                     'labels_contr': labels_contr, 'labels_quiet': labels_quiet,
                     'time_contr': time_contr, 'time_quiet': time_quiet, 'time_quiet_avg': time_quiet_avg,
                     'time_contr_avg': time_contr_avg, 'time_cycle': time_cycle,
                     'n_contr': n_contr, 'n_quiet': n_quiet,
                     'beating_rate_variability': beating_rate_variability, 'beating_rate': beating_rate, }
        self.loi_data.update(dict_temp)

        if self.auto_save:
            self.store_loi_data()

    def get_trajectories(self, slen_lims: Tuple[float, float] = (1.2, 3.), filter_params_vel: Tuple[int, int] = (13, 5),
                         dilate_contr: float = 0, equ_lims: Tuple[float, float] = (1.5, 2.3)):
        """
        1. Calculate sarcomere lengths (single and avg) and filter too large and too small values (slen_lims).
        2. Calculate sarcomere velocities (single and avg), prior smoothing of s'lengths with Savitzky-Golay filter
            (filter_params_vel)
        3. Calculate sarcomere equilibrium lengths (equ) and delta_slen

        Parameters
        ----------
        slen_lims : tuple of float, optional
            Lower and upper limits of sarcomere lengths (µm); values outside
            are set to NaN. Default is (1.2, 3.).
        filter_params_vel : tuple of int, optional
            Window length and polynomial order for the Savitzky-Golay filter
            used to smooth lengths prior to velocity differentiation.
            Default is (13, 5).
        dilate_contr : float, optional
            Dilation time (s) of the contraction time-series, shortening the
            diastolic interval used to determine equilibrium lengths.
            Default is 0.
        equ_lims : tuple of float, optional
            Lower and upper limits of sarcomere equilibrium lengths (µm);
            values outside are set to NaN. Default is (1.5, 2.3).
        """
        # calculate sarcomere lengths
        slen = np.diff(self.loi_data['z_pos'], axis=0)
        slen[(slen < slen_lims[0]) | (slen > slen_lims[1])] = np.nan
        slen_avg = np.nanmean(slen, axis=0)
        n_sarcomeres = slen.shape[0]
        n_sarcomeres_time = np.count_nonzero(~np.isnan(slen), axis=0)
        frametime = self.metadata.frametime

        # smooth slen with sav. golay filter and calculate velocity
        vel = Utils.custom_diff(Utils.nan_sav_golay(slen, filter_params_vel[0], filter_params_vel[1]), frametime)
        vel_avg = np.nanmean(vel, axis=0)

        # calculate sarcomere equ length and delta sarcomere length
        dilate_contr = int(dilate_contr * 2 / self.metadata.frametime)
        if dilate_contr == 0:
            contr_dilated = self.loi_data['contr']
        elif dilate_contr > 0:
            contr_dilated = binary_dilation(self.loi_data['contr'],
                                            structure=np.ones(dilate_contr))
        else:
            raise ValueError(f'Parameter dilate_contr={dilate_contr} not valid!')

        equ = np.asarray([
            np.nanmedian(s[contr_dilated == 0]) if np.any(~np.isnan(s[contr_dilated == 0])) else np.nan
            for s in slen
        ])
        delta_slen = np.asarray([slen[i] - equ[i] for i in range(len(equ))])
        delta_slen_avg = np.nanmean(delta_slen, axis=0)
        if np.count_nonzero(delta_slen) > 0:
            ratio_nans = np.count_nonzero(np.isnan(delta_slen)) / np.count_nonzero(delta_slen)
        else:
            ratio_nans = np.nan

        # store data in LOI dictionary
        dict_temp = {
            'params.get_sarcomere_trajectories.slen_lims': slen_lims,
            'params.get_sarcomere_trajectories.filter_params_vel': filter_params_vel,
            'slen': slen, 'slen_avg': slen_avg, 'vel': vel, 'vel_avg': vel_avg, 'n_sarcomeres': n_sarcomeres,
            'n_sarcomeres_time': n_sarcomeres_time, 'equ': equ, 'delta_slen': delta_slen,
            'delta_slen_avg': delta_slen_avg, 'ratio_nans': ratio_nans}
        self.loi_data.update(dict_temp)
        if self.auto_save:
            self.store_loi_data()

    def analyze_trajectories(self, custom_perc: Union[List[Tuple[float, float]], None] = None):
        """
        Analyze sarcomere single and average trajectories (extrema of sarcomeres contraction and velocity)
        and sarcomere popping

        Parameters
        ----------
        custom_perc : list of tuple of float, or None, optional
            List of ``(p0, p1)`` percentage pairs defining custom intervals to
            analyze, e.g. the time from p0% to p1% of individual and average
            sarcomere length change. If ``p0 < p1`` shortening is analyzed; if
            ``p1 < p0`` elongation is analyzed. Default is None.
        """
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
        time_to_relax = np.zeros_like(contr_max) * np.nan
        time_to_relax_avg = np.zeros_like(contr_max_avg) * np.nan
        # custom percentages (p0% to p1%)
        if custom_perc and not isinstance(custom_perc, list):
            custom_perc = [custom_perc]
        elif not custom_perc:
            custom_perc = []
        custom_perc_time = [np.zeros_like(contr_max) for _ in range(len(custom_perc))]
        custom_perc_time_avg = [np.zeros_like(contr_max_avg) for _ in range(len(custom_perc))]

        # iterate individual sarcomeres
        labels_contr = self.loi_data['labels_contr']
        for j, delta_j in enumerate(self.loi_data['delta_slen']):
            vel_j = self.loi_data['vel'][j]
            for i in range(self.loi_data['n_contr']):
                # get time-series of one contraction cycle (start to start)
                delta_i = delta_j[labels_contr == i + 1]
                vel_i = vel_j[labels_contr == i + 1]
                # find extrema
                contr_max[j][i] = np.nanmin(delta_i) if np.any(~np.isnan(delta_i)) else np.nan
                elong_max[j][i] = np.nanmax(delta_i) if np.any(~np.isnan(delta_i)) else np.nan
                vel_contr_max[j][i] = np.nanmin(vel_i) if np.any(~np.isnan(vel_i)) else np.nan
                vel_elong_max[j][i] = np.nanmax(vel_i) if np.any(~np.isnan(vel_i)) else np.nan
                # time to peak
                if np.count_nonzero(np.isnan(delta_i)) == 0:
                    time_to_peak[j][i] = np.nanargmin(delta_i) * self.metadata.frametime
                    time_to_relax[j][i] = (len(delta_i) - np.nanargmin(delta_i)) * self.metadata.frametime
                if custom_perc:
                    for k, (p0, p1) in enumerate(custom_perc):
                        if p0 < p1:  # shortening
                            delta_shortening_i = delta_i[: np.nanargmin(delta_i)+1]
                            if len(delta_shortening_i) > 0:
                                if p0 == 0:
                                    t0, contr0 = 0, 0
                                else:
                                    t0, contr0 = Utils.find_closest(delta_shortening_i, contr_max[j][i] * p0 * 0.01)
                                t1, contr1 = Utils.find_closest(delta_shortening_i, contr_max[j][i] * p1 * 0.01)
                            else:
                                t0, contr0 = np.nan, np.nan
                                t1, contr1 = np.nan, np.nan
                        elif p0 > p1:  # elongation
                            delta_elongation_i = delta_i[np.nanargmin(delta_i):]
                            if len(delta_elongation_i) > 0:
                                if p1 == 0:
                                    t1, contr1 = len(delta_elongation_i), 0
                                else:
                                    t1, contr1 = Utils.find_closest(delta_elongation_i, contr_max[j][i] * p1 * 0.01)
                                t0, contr0 = Utils.find_closest(delta_elongation_i, contr_max[j][i] * p0 * 0.01)
                            else:
                                t0, contr0 = np.nan, np.nan
                                t1, contr1 = np.nan, np.nan
                        else:
                            raise ValueError('p0 and p1 must be different.')
                        custom_perc_time[k][j, i] = (t1 - t0) * self.metadata.frametime

        # average contraction
        for i in range(self.loi_data['n_contr']):
            # get time-series of one contraction cycle (start to start)
            delta_i = self.loi_data['delta_slen_avg'][labels_contr == i + 1]
            vel_i = self.loi_data['vel_avg'][labels_contr == i + 1]
            # find extrema
            contr_max_avg[i] = np.nanmin(delta_i) if np.any(~np.isnan(delta_i)) else np.nan
            elong_max_avg[i] = np.nanmax(delta_i) if np.any(~np.isnan(delta_i)) else np.nan
            vel_contr_max_avg[i] = np.nanmin(vel_i) if np.any(~np.isnan(vel_i)) else np.nan
            vel_elong_max_avg[i] = np.nanmax(vel_i) if np.any(~np.isnan(vel_i)) else np.nan
            # time to peak
            if np.count_nonzero(np.isnan(delta_i)) == 0:
                time_to_peak_avg[i] = np.nanargmin(delta_i) * self.metadata.frametime
                time_to_relax_avg[i] = (len(delta_i) - np.nanargmin(delta_i)) * self.metadata.frametime
            if custom_perc:
                for k, (p0, p1) in enumerate(custom_perc):
                    if p0 < p1:  # shortening
                        delta_shortening_i = delta_i[: np.nanargmin(delta_i)+1]
                        if len(delta_shortening_i) > 0:
                            if p0 == 0:
                                t0_avg, contr0_avg = 0, 0
                            else:
                                t0_avg, contr0_avg = Utils.find_closest(delta_shortening_i, contr_max_avg[i] * p0 * 0.01)
                            t1_avg, contr1_avg = Utils.find_closest(delta_shortening_i, contr_max_avg[i] * p1 * 0.01)
                        else:
                            t0_avg, contr0_avg = np.nan, np.nan
                            t1_avg, contr1_avg = np.nan, np.nan
                    elif p0 > p1:  # elongation
                        delta_elongation_i = delta_i[np.nanargmin(delta_i):]
                        if len(delta_elongation_i) > 0:
                            if p1 == 0:
                                t1_avg, contr1_avg = len(delta_elongation_i), 0
                            else:
                                t1_avg, contr1_avg = Utils.find_closest(delta_elongation_i, contr_max_avg[i] * p1 * 0.01)
                            t0_avg, contr0_avg = Utils.find_closest(delta_elongation_i, contr_max_avg[i] * p0 * 0.01)
                        else:
                            t0_avg, contr0_avg = np.nan, np.nan
                            t1_avg, contr1_avg = np.nan, np.nan
                    else:
                        raise ValueError('p0 and p1 must be different.')
                    custom_perc_time_avg[k][i] = (t1_avg - t0_avg) * self.metadata.frametime

        # calculate surplus motion index
        self.calculate_surplus_motion_index()

        # save data in LOI dict
        self.loi_data.update({'contr_max': contr_max, 'elong_max': elong_max, 'vel_contr_max': vel_contr_max,
                              'vel_elong_max': vel_elong_max, 'contr_max_avg': contr_max_avg,
                              'elong_max_avg': elong_max_avg, 'vel_contr_max_avg': vel_contr_max_avg,
                              'vel_elong_max_avg': vel_elong_max_avg, 'time_to_peak': time_to_peak,
                              'time_to_peak_avg': time_to_peak_avg, 'time_to_relax': time_to_relax,
                              'time_to_relax_avg': time_to_relax_avg, 'custom_perc_time': np.asarray(custom_perc_time),
                              'custom_perc_time_avg': np.asarray(custom_perc_time_avg)})
        if self.auto_save:
            self.store_loi_data()

    def calculate_surplus_motion_index(self):
        """
        Calculate surplus motion index (SMI) for sarcomere motion: average distance traveled by
        individual sarcomeres contractions divided by distance traveled by sarcomere average
        """
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
            abs_motion_single_i = np.sum(np.abs(vel_i), axis=1) * self.metadata.frametime
            abs_motion_avg_i = np.sum(np.abs(vel_avg_i)) * self.metadata.frametime
            abs_motion_single[i] = abs_motion_single_i
            abs_motion_avg[i] = abs_motion_avg_i

        # calculate surplus motion index per contraction cycle and store in dict
        smi = np.nanmean(abs_motion_single) / np.nanmean(abs_motion_avg)
        self.loi_data['smi'] = smi
        if self.auto_save:
            self.store_loi_data()

    def analyze_popping(self, thres_popping: float = 0.25):
        """
        Analyze sarcomere popping - popping if elongation of individual sarcomere is larger than thres_popping

        Parameters
        ----------
        thres_popping : float, optional
            Threshold above which a sarcomere is identified as popping, in µm
            beyond equilibrium length. Default is 0.25.
        """
        # identify popping events
        elong_max = self.loi_data['elong_max']
        popping = np.zeros_like(elong_max, dtype='bool')
        popping[elong_max > thres_popping] = 1

        # calculate popping rates
        rate_contr = np.mean(popping, axis=0)
        rate_sarcomeres = np.mean(popping, axis=1)
        rate = np.mean(popping)

        # dictionary
        dict_popping = {'popping_rate_contr': rate_contr, 'popping_rate_sarcomeres': rate_sarcomeres,
                        'popping_rate': rate, 'popping_events': popping,
                        'params.analyze_popping.thres_popping': thres_popping}

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

    def analyze_correlations(self):
        """
        Compute Pearson correlation coefficients for sarcomere motion patterns
        (∆SL and V) across contraction cycles and between sarcomeres within the
        same cycle to analyze static and stochastic heterogeneity.

        Calculates the average serial (r_s) and mutual (r_m) correlation
        coefficients and the ratio R of serial to mutual correlations, then
        updates ``self.loi_data`` and stores it if ``auto_save`` is enabled.

        Notes
        -----
        Updates ``self.loi_data`` with the following keys:

        - ``'corr_delta_slen'`` (np.ndarray or None): Correlation matrix for length changes.
        - ``'corr_vel'`` (np.ndarray or None): Correlation matrix for velocities.
        - ``'corr_delta_slen_serial'`` (float): Average serial correlation for length changes.
        - ``'corr_delta_slen_mutual'`` (float): Average mutual correlation for length changes.
        - ``'corr_vel_serial'`` (float): Average serial correlation for velocities.
        - ``'corr_vel_mutual'`` (float): Average mutual correlation for velocities.
        - ``'ratio_delta_slen_mutual_serial'`` (float): Mutual/serial ratio for length changes.
        - ``'ratio_vel_mutual_serial'`` (float): Mutual/serial ratio for velocities.
        """
        if self.loi_data['n_contr'] > 0:
            time_contr_median = int(np.median(self.loi_data['time_contr']) / self.metadata.frametime)

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

    def analyze_oscillations(self, min_scale: float = 6, max_scale: float = 180, num_scales: int = 60,
                             wavelet: str = 'morl', freq_thres: float = 2, plot: bool = False):
        """
        Analyze the oscillation frequencies of average and individual sarcomere length changes.

        Parameters
        ----------
        min_scale : float, optional
            Minimum scale for the wavelet transform. Default is 6.
        max_scale : float, optional
            Maximum scale for the wavelet transform. Default is 180.
        num_scales : int, optional
            Number of scales for the wavelet transform. Default is 60.
        wavelet : str, optional
            Wavelet type for the transform ('morl' = Morlet). Default is 'morl'.
        freq_thres : float, optional
            Frequency threshold (Hz) separating low-frequency oscillations at
            the beating rate from high-frequency oscillations. Default is 2.
        plot : bool, optional
            If True, show a plot illustrating the analysis. Default is False.
        """

        # Analyze oscillation frequencies of average sarcomere length change
        cfs_avg, frequencies = self.wavelet_analysis_oscillations(self.loi_data['delta_slen_avg'],
                                                                  self.metadata.frametime,
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
                                                          self.metadata.frametime,
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

        dict_oscill = {'params.analyze_oscillations.min_scale': min_scale,
                       'params.analyze_oscillations.max_scale': max_scale,
                       'params.analyze_oscillations.num_scales': num_scales,
                       'params.analyze_oscillations.wavelet': wavelet,
                       'params.analyze_oscillations.freq_thres': freq_thres,
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

    @staticmethod
    def predict_contractions(z_pos: np.ndarray, slen: np.ndarray, weights: str, threshold: float = 0.33):
        """
        Predict contractions from Z-band motion and sarcomere lengths.

        Predicts per-trace contraction state, then averages and thresholds it
        for a more accurate contraction estimate.

        Parameters
        ----------
        z_pos : np.ndarray
            Time-series of Z-band positions.
        slen : np.ndarray
            Time-series of sarcomere lengths.
        weights : str
            Path to the neural network weights (.pt file).
        threshold : float, optional
            Binary threshold for contraction state (0, 1). Default is 0.33.

        Returns
        -------
        np.ndarray
            Boolean contraction state per frame (True where contracting).
        """
        data = np.concatenate([z_pos, slen])
        # ContractionNet needs a finite trace. z_pos/slen legitimately carry
        # NaN (e.g. synthesized track chains keep gap/edge frames as NaN rather
        # than holding a constant length), so gap-fill each trace transiently for
        # the prediction only — the stored z_pos/slen keep their NaNs.
        contr_all = np.asarray([predict_contractions(Motion._fill_trace_nans(d), weights)[0] for d in data])
        contr_mean = np.nanmean(contr_all, axis=0)
        return contr_mean > threshold

    @staticmethod
    def _fill_trace_nans(trace: np.ndarray) -> np.ndarray:
        """Make a 1D trace finite for the contraction network.

        Interior NaNs are linearly interpolated and leading/trailing NaNs are
        held at the nearest finite sample (acceptable here because the result is
        transient model input, never stored). An all-NaN trace becomes zeros."""
        trace = np.asarray(trace, dtype=float)
        mask = np.isnan(trace)
        if not mask.any():
            return trace
        if mask.all():
            return np.zeros_like(trace)
        idx = np.arange(trace.shape[0])
        out = trace.copy()
        out[mask] = np.interp(idx[mask], idx[~mask], trace[~mask])
        return out

    @staticmethod
    def wavelet_analysis_oscillations(data: np.ndarray, frametime: float, min_scale: float = 6, max_scale: float = 150,
                                      num_scales: int = 100, wavelet: str = 'morl'):
        """
        Perform a wavelet transform of the data.

        Parameters
        ----------
        data : array_like
            1-D input signal.
        frametime : float
            Sampling period of the signal in seconds.
        min_scale : float, optional
            Minimum scale for the wavelet transform. Default is 6.
        max_scale : float, optional
            Maximum scale for the wavelet transform. Default is 150.
        num_scales : int, optional
            Number of scales for the wavelet transform. Default is 100.
        wavelet : str, optional
            Wavelet type for the transform. Default is 'morl'.

        Returns
        -------
        cfs : np.ndarray
            Continuous wavelet transform coefficients.
        frequencies : np.ndarray
            Corresponding frequencies for each scale.
        """
        # Generate a range of scales that are logarithmically spaced
        scales = np.geomspace(min_scale, max_scale, num=num_scales)

        # Perform the wavelet transform
        cfs, frequencies = cwt(data, scales, wavelet, sampling_period=frametime)

        return cfs, frequencies
