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


import napari

from .parameters import Parameters
from .parameter import Parameter
from sarcasm import SarcAsM, Motion, TypeUtils, Structure
from typing import Optional


class ApplicationModel:
    """
    The ApplicationModel concentrates all necessary parameters for calling the sarcasm_old backend methods
    and provides via Parameters and Parameter class methods to bind those to the UI.
    """

    def __init__(self):
        self._cell: Optional[SarcAsM] = None
        self.__cell_file_name: Optional[str] = None
        self.currentlyProcessing = Parameter("currentlyProcessing", False)
        self.__file_extension = ".json"
        self.__line_dictionary = {}  # todo: remove the line dictionary
        self.__sarcomere: Optional[Motion] = None
        self.__scheme = f'%d_%d_%d_%d_%.2f'
        self.__parameters = Parameters()
        self.__create_parameters()
        self.set_to_default()

    @property
    def scheme(self):
        return self.__scheme

    def reset_model(self):
        self._cell = None
        self.__cell_file_name = None
        self.__line_dictionary = {}
        self.__sarcomere = None

    @property
    def line_dictionary(self):
        return self.__line_dictionary

    # todo: to prevent annoying warnings of optional on usage the return type could be left without optional and
    #   there could be a null check and exception in case that its null?

    @property
    def parameters(self):
        return self.__parameters

    @property
    def sarcomere(self) -> Optional[Motion]:
        return self.__sarcomere

    @property
    def cell(self) -> Optional[Structure]:
        return self._cell

    @property
    def file_extension(self):
        return self.__file_extension

    def init_cell(self, cell_file):
        self.__cell_file_name = cell_file
        # this is no longer of type SarcAsM but of Type Structure
        self._cell = Structure(cell_file, use_gui=True)

    def init_sarcomere(self, loi_name):
        cell_file_name = TypeUtils.unbox(self.__cell_file_name)
        self.__sarcomere = Motion(cell_file_name, loi_name=loi_name)

    def is_initialized(self):
        # check if file is loaded, check if viewer is active(not closed)
        result = True
        if self._cell is None:
            result = False
        if self.__cell_file_name == '' or self.__cell_file_name is None:
            result = False
        if napari.current_viewer() is None:
            result = False
        return result

    def set_to_default(self):
        # region file-load parameters
        self.__parameters.get_parameter(name='file.load.correct_phase').set_value(False)
        # endregion
        # region structure parameters
        self.__parameters.get_parameter(name='structure.predict.network_path').set_value('generalist')

        self.__parameters.get_parameter(name='structure.predict.size_width').set_value(
            1024)  # is the predict_size_min from ui
        self.__parameters.get_parameter(name='structure.predict.size_height').set_value(
            1024)  # is the predict_size_max from ui
        self.__parameters.get_parameter(name='structure.predict.clip_thresh_min').set_value(0.)
        self.__parameters.get_parameter(name='structure.predict.clip_thresh_max').set_value(99.8)

        self.__parameters.get_parameter(name='structure.predict_fast_movie.network_path').set_value('generalist')
        self.__parameters.get_parameter(name='structure.predict_fast_movie.n_frames').set_value(32)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.size_width').set_value(256)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.size_height').set_value(256)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.clip_thresh_min').set_value(0.)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.clip_thresh_max').set_value(99.8)

        self.__parameters.get_parameter(name='structure.cell_mask.threshold').set_value(0.1)


        self.__parameters.get_parameter(name='structure.frames').set_value('all')
        self.__parameters.get_parameter(name='structure.plot').set_value(False)


        self.__parameters.get_parameter(name='structure.z_band_analysis.threshold').set_value(0.5)
        self.__parameters.get_parameter(name='structure.z_band_analysis.min_length').set_value(0.2)
        self.__parameters.get_parameter(name='structure.z_band_analysis.median_filter_radius').set_value(0.2)
        self.__parameters.get_parameter(name='structure.z_band_analysis.theta_phi_min').set_value(0.4)
        self.__parameters.get_parameter(name='structure.z_band_analysis.a_min').set_value(0.3)
        self.__parameters.get_parameter(name='structure.z_band_analysis.d_max').set_value(4.0)
        self.__parameters.get_parameter(name='structure.z_band_analysis.d_min').set_value(0.00)


        self.__parameters.get_parameter(name='structure.vectors.radius').set_value(0.25)
        self.__parameters.get_parameter(name='structure.vectors.line_width').set_value(0.2)
        self.__parameters.get_parameter(name='structure.vectors.interpolation_factor').set_value(0)
        self.__parameters.get_parameter(name='structure.vectors.length_limit_lower').set_value(1.0)
        self.__parameters.get_parameter(name='structure.vectors.length_limit_upper').set_value(3.0)


        self.__parameters.get_parameter(name='structure.myofibril.ratio_seeds').set_value(0.1)
        self.__parameters.get_parameter(name='structure.myofibril.persistence').set_value(3)
        self.__parameters.get_parameter(name='structure.myofibril.threshold_distance').set_value(0.5)
        self.__parameters.get_parameter(name='structure.myofibril.n_min').set_value(4)
        self.__parameters.get_parameter(name='structure.myofibril.median_filter_radius').set_value(0.5)



        self.__parameters.get_parameter(name='structure.domain.analysis.d_max').set_value(3.0)
        self.__parameters.get_parameter(name='structure.domain.analysis.cosine_min').set_value(0.65)
        self.__parameters.get_parameter(name='structure.domain.analysis.leiden_resolution').set_value(0.06)
        self.__parameters.get_parameter(name='structure.domain.analysis.random_seed').set_value(42)
        self.__parameters.get_parameter(name='structure.domain.analysis.area_min').set_value(20.0)
        self.__parameters.get_parameter(name='structure.domain.analysis.dilation_radius').set_value(0.3)
        # endregion

        # region loi parameters
        """
                 frame: int = 0, n_lois: int = 4, n_seeds: int = 200, persistence: int = 2,
                            threshold_distance: float = 0.3, score_threshold: Optional[float] = None,
                            mode: str = 'longest_in_cluster', random_seed: Optional[int] = None,
                            number_lims: Tuple[int, int] = (10, 50), length_lims: Tuple[float, float] = (0, 200),
                            sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
                            sarcomere_std_length_lims: Tuple[float, float] = (0, 1),
                            msc_lims: Tuple[float, float] = (0, 1), max_orient_change: float = 30,
                            midline_mean_length_lims: Tuple[float, float] = (0, 20),
                            midline_std_length_lims: Tuple[float, float] = (0, 5),
                            midline_min_length_lims: Tuple[float, float] = (0, 20), distance_threshold_lois: float = 40,
                            linkage: str = 'single', linewidth: float = 0.65, order: int = 0, export_raw: bool = False
                """
        self.__parameters.get_parameter(name='loi.detect.frame').set_value(0)
        self.__parameters.get_parameter(name='loi.detect.n_lois').set_value(4)
        self.__parameters.get_parameter(name='loi.detect.ratio_seeds').set_value(0.1)
        self.__parameters.get_parameter(name='loi.detect.persistence').set_value(4)
        self.__parameters.get_parameter(name='loi.detect.threshold_distance').set_value(0.5)
        self.__parameters.get_parameter(name='loi.detect.mode').set_value('longest_in_cluster')
        self.__parameters.get_parameter(name='loi.detect.number_limits_lower').set_value(10)
        self.__parameters.get_parameter(name='loi.detect.number_limits_upper').set_value(50)
        self.__parameters.get_parameter(name='loi.detect.length_limits_lower').set_value(0.0)
        self.__parameters.get_parameter(name='loi.detect.length_limits_upper').set_value(200.0)
        self.__parameters.get_parameter(name='loi.detect.sarcomere_mean_length_limits_lower').set_value(1.0)
        self.__parameters.get_parameter(name='loi.detect.sarcomere_mean_length_limits_upper').set_value(3.0)
        self.__parameters.get_parameter(name='loi.detect.sarcomere_std_length_limits_lower').set_value(0.0)
        self.__parameters.get_parameter(name='loi.detect.sarcomere_std_length_limits_upper').set_value(1.0)
        self.__parameters.get_parameter(name='loi.detect.midline_mean_length_limits_lower').set_value(0.0)
        self.__parameters.get_parameter(name='loi.detect.midline_mean_length_limits_upper').set_value(50.0)
        self.__parameters.get_parameter(name='loi.detect.midline_std_length_limits_lower').set_value(0.0)
        self.__parameters.get_parameter(name='loi.detect.midline_std_length_limits_upper').set_value(50.0)
        self.__parameters.get_parameter(name='loi.detect.midline_min_length_limits_lower').set_value(0.0)
        self.__parameters.get_parameter(name='loi.detect.midline_min_length_limits_upper').set_value(50.0)
        self.__parameters.get_parameter(name='loi.detect.cluster_threshold_lois').set_value(40.0)
        self.__parameters.get_parameter(name='loi.detect.linkage').set_value('single')
        self.__parameters.get_parameter(name='loi.detect.line_width').set_value(0.65)
        self.__parameters.get_parameter(name='loi.detect.order').set_value(0)
        self.__parameters.get_parameter(name='loi.detect.plot').set_value(False)
        # endregion

        # region motion parameters
        self.__parameters.get_parameter(name='motion.detect_peaks.threshold').set_value(0.2)
        self.__parameters.get_parameter(name='motion.detect_peaks.min_distance').set_value(1.4)
        self.__parameters.get_parameter(name='motion.detect_peaks.width').set_value(0.5)

        self.__parameters.get_parameter(name='motion.track_z_bands.search_range').set_value(2.0)
        self.__parameters.get_parameter(name='motion.track_z_bands.memory').set_value(10)
        self.__parameters.get_parameter(name='motion.track_z_bands.memory_interpolation').set_value(3)

        self.__parameters.get_parameter(name='motion.systoles.weights').set_value('default')  # weights is a network file
        self.__parameters.get_parameter(name='motion.systoles.threshold').set_value(0.03)
        self.__parameters.get_parameter(name='motion.systoles.slen_limits.lower').set_value(1.2)
        self.__parameters.get_parameter(name='motion.systoles.slen_limits.upper').set_value(3.0)
        self.__parameters.get_parameter(name='motion.systoles.n_sarcomeres_min').set_value(4)
        self.__parameters.get_parameter(name='motion.systoles.buffer_frames').set_value(3)
        self.__parameters.get_parameter(name='motion.systoles.contr_time_min').set_value(0.2)
        self.__parameters.get_parameter(name='motion.systoles.merge_time_max').set_value(0.05)


        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_lower').set_value(1.2)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_upper').set_value(3.0)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.dilate_systoles').set_value(0.0)
        self.__parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.filter_params_vel.window_length').set_value(13)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.filter_params_vel.polyorder').set_value(
            5)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.equ_limits_lower').set_value(1.5)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.equ_limits_upper').set_value(2.3)
        # endregion

        # region batch processing parameters
        # todo: currently set values here for testing (default values, matching the test data i'm using for batch processing ui tests)
        # todo: remove those values when finished with testing
        self.__parameters.get_parameter(name='batch.pixel.size').set_value(0.13)
        self.__parameters.get_parameter(name='batch.frame.time').set_value(0.1)
        self.__parameters.get_parameter(name='batch.force.override').set_value(False)
        self.__parameters.get_parameter(name='batch.thread_pool_size').set_value(3)
        self.__parameters.get_parameter(name='batch.root').set_value('D:\\Test\\SarcasmTestBatch')
        self.__parameters.get_parameter(name='batch.recalculate.for.motion').set_value(True)
        self.__parameters.get_parameter(name='batch.do_cellmask').set_value(True)
        self.__parameters.get_parameter(name='batch.do_zbands').set_value(True)
        self.__parameters.get_parameter(name='batch.do_vectors').set_value(True)
        self.__parameters.get_parameter(name='batch.do_myofibrils').set_value(True)
        self.__parameters.get_parameter(name='batch.do_domains').set_value(True)
        # endregion

        pass

    def __create_parameters(self):
        # region file-load parameters
        self.__parameters.set_parameter(name='file.load.correct_phase')
        # endregion
        # region structure parameters
        self.__parameters.set_parameter(name='structure.predict.network_path')

        # was removed
        #self.__parameters.set_parameter(name='structure.predict.time_consistent')
        #self.__parameters.set_parameter(name='structure.predict.time_consistent.frame')
        self.__parameters.set_parameter(name='structure.predict.size_width')  # is the predict_size_min from ui
        self.__parameters.set_parameter(name='structure.predict.size_height')  # is the predict_size_max from ui
        self.__parameters.set_parameter(name='structure.predict.clip_thresh_min')
        self.__parameters.set_parameter(name='structure.predict.clip_thresh_max')

        self.__parameters.set_parameter(name='structure.predict_fast_movie.network_path')
        self.__parameters.set_parameter(name='structure.predict_fast_movie.n_frames')
        self.__parameters.set_parameter(name='structure.predict_fast_movie.size_width')
        self.__parameters.set_parameter(name='structure.predict_fast_movie.size_height')
        self.__parameters.set_parameter(name='structure.predict_fast_movie.clip_thresh_min')
        self.__parameters.set_parameter(name='structure.predict_fast_movie.clip_thresh_max')

        self.__parameters.set_parameter(name='structure.cell_mask.threshold')

        self.__parameters.set_parameter(name='structure.frames')
        self.__parameters.set_parameter(name='structure.plot')

        self.__parameters.set_parameter(name='structure.z_band_analysis.threshold')
        self.__parameters.set_parameter(name='structure.z_band_analysis.min_length')
        self.__parameters.set_parameter(name='structure.z_band_analysis.median_filter_radius')
        self.__parameters.set_parameter(name='structure.z_band_analysis.theta_phi_min')
        self.__parameters.set_parameter(name='structure.z_band_analysis.a_min')
        self.__parameters.set_parameter(name='structure.z_band_analysis.d_max')
        self.__parameters.set_parameter(name='structure.z_band_analysis.d_min')


        self.__parameters.set_parameter(name='structure.vectors.radius')
        self.__parameters.set_parameter(name='structure.vectors.line_width')
        self.__parameters.set_parameter(name='structure.vectors.interpolation_factor')
        self.__parameters.set_parameter(name='structure.vectors.length_limit_lower')
        self.__parameters.set_parameter(name='structure.vectors.length_limit_upper')


        self.__parameters.set_parameter(name='structure.myofibril.ratio_seeds')
        self.__parameters.set_parameter(name='structure.myofibril.persistence')
        self.__parameters.set_parameter(name='structure.myofibril.threshold_distance')
        self.__parameters.set_parameter(name='structure.myofibril.n_min')
        self.__parameters.set_parameter(name='structure.myofibril.median_filter_radius')


        self.__parameters.set_parameter(name='structure.domain.analysis.d_max')
        self.__parameters.set_parameter(name='structure.domain.analysis.cosine_min')
        self.__parameters.set_parameter(name='structure.domain.analysis.leiden_resolution')
        self.__parameters.set_parameter(name='structure.domain.analysis.random_seed')
        self.__parameters.set_parameter(name='structure.domain.analysis.area_min')
        self.__parameters.set_parameter(name='structure.domain.analysis.dilation_radius')
        # endregion

        # region loi parameters
        self.__parameters.set_parameter(name='loi.detect.frame')
        self.__parameters.set_parameter(name='loi.detect.n_lois')
        self.__parameters.set_parameter(name='loi.detect.ratio_seeds')
        self.__parameters.set_parameter(name='loi.detect.persistence')
        self.__parameters.set_parameter(name='loi.detect.threshold_distance')
        self.__parameters.set_parameter(name='loi.detect.mode')
        self.__parameters.set_parameter(name='loi.detect.number_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.number_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.length_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.length_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.sarcomere_mean_length_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.sarcomere_mean_length_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.sarcomere_std_length_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.sarcomere_std_length_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.midline_mean_length_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.midline_mean_length_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.midline_std_length_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.midline_std_length_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.midline_min_length_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.midline_min_length_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.cluster_threshold_lois')
        self.__parameters.set_parameter(name='loi.detect.linkage')
        self.__parameters.set_parameter(name='loi.detect.line_width')
        self.__parameters.set_parameter(name='loi.detect.order')
        self.__parameters.set_parameter(name='loi.detect.plot')
        # endregion

        # region motion parameters
        self.__parameters.set_parameter(name='motion.detect_peaks.threshold')
        self.__parameters.set_parameter(name='motion.detect_peaks.min_distance')
        self.__parameters.set_parameter(name='motion.detect_peaks.width')

        self.__parameters.set_parameter(name='motion.track_z_bands.search_range')
        self.__parameters.set_parameter(name='motion.track_z_bands.memory')
        self.__parameters.set_parameter(name='motion.track_z_bands.memory_interpolation')

        self.__parameters.set_parameter(name='motion.systoles.weights')  # weights is a network file
        self.__parameters.set_parameter(name='motion.systoles.threshold')
        self.__parameters.set_parameter(name='motion.systoles.slen_limits.lower')
        self.__parameters.set_parameter(name='motion.systoles.slen_limits.upper')
        self.__parameters.set_parameter(name='motion.systoles.n_sarcomeres_min')
        self.__parameters.set_parameter(name='motion.systoles.buffer_frames')
        self.__parameters.set_parameter(name='motion.systoles.contr_time_min')
        self.__parameters.set_parameter(name='motion.systoles.merge_time_max')


        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_lower')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_upper')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.dilate_systoles')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.filter_params_vel.window_length')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.filter_params_vel.polyorder')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.equ_limits_lower')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.equ_limits_upper')
        # endregion

        # region batch processing parameters
        self.__parameters.set_parameter(name='batch.pixel.size')
        self.__parameters.set_parameter(name='batch.frame.time')
        self.__parameters.set_parameter(name='batch.force.override')
        self.__parameters.set_parameter(name='batch.thread_pool_size')
        self.__parameters.set_parameter(name='batch.root')
        self.__parameters.set_parameter(name='batch.recalculate.for.motion')
        self.__parameters.set_parameter(name='batch.do_cellmask')
        self.__parameters.set_parameter(name='batch.do_zbands')
        self.__parameters.set_parameter(name='batch.do_vectors')
        self.__parameters.set_parameter(name='batch.do_myofibrils')
        self.__parameters.set_parameter(name='batch.do_domains')
        # endregion
