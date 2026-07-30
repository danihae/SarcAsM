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
from sarcasm import SarcAsM, TypeUtils
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
        self.__scheme = '%d_%d_%d_%d_%.2f'
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

    @property
    def line_dictionary(self):
        return self.__line_dictionary

    # todo: to prevent annoying warnings of optional on usage the return type could be left without optional and
    #   there could be a null check and exception in case that its null?

    @property
    def parameters(self):
        return self.__parameters

    @property
    def cell(self) -> Optional[SarcAsM]:
        return self._cell

    @property
    def file_extension(self):
        return self.__file_extension

    def init_cell(self, cell_file):
        self.__cell_file_name = cell_file
        self._cell = SarcAsM(cell_file, use_gui=True)

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
        self._set_defaults_file_load()
        self._set_defaults_structure()
        self._set_defaults_motion()
        self._set_defaults_batch()

    def _set_defaults_file_load(self):
        self.__parameters.get_parameter(name='file.load.correct_phase').set_value(False)

    def _set_defaults_structure(self):
        self.__parameters.get_parameter(name='structure.predict.network_path').set_value('generalist')
        self.__parameters.get_parameter(name='structure.predict.rescale_factor').set_value(1.0)
        self.__parameters.get_parameter(name='structure.predict.size_width').set_value(
            1024)  # is the predict_size_min from ui
        self.__parameters.get_parameter(name='structure.predict.size_height').set_value(
            1024)  # is the predict_size_max from ui
        self.__parameters.get_parameter(name='structure.predict.clip_thresh_min').set_value(0.)
        self.__parameters.get_parameter(name='structure.predict.clip_thresh_max').set_value(99.98)

        self.__parameters.get_parameter(name='structure.predict_fast_movie.network_path').set_value('generalist')
        self.__parameters.get_parameter(name='structure.predict_fast_movie.n_frames').set_value(32)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.size_width').set_value(256)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.size_height').set_value(256)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.clip_thresh_min').set_value(0.)
        self.__parameters.get_parameter(name='structure.predict_fast_movie.clip_thresh_max').set_value(99.98)

        self.__parameters.get_parameter(name='structure.cell_mask.threshold').set_value(0.1)

        self.__parameters.get_parameter(name='structure.frames').set_value('all')
        self.__parameters.get_parameter(name='structure.plot').set_value(False)

        self.__parameters.get_parameter(name='structure.z_band_analysis.threshold').set_value(0.5)
        self.__parameters.get_parameter(name='structure.z_band_analysis.min_length').set_value(0.2)
        self.__parameters.get_parameter(name='structure.z_band_analysis.median_filter_radius').set_value(0.2)
        self.__parameters.get_parameter(name='structure.z_band_analysis.theta_phi_min').set_value(0.4)
        self.__parameters.get_parameter(name='structure.z_band_analysis.a_min').set_value(0.3)
        self.__parameters.get_parameter(name='structure.z_band_analysis.d_max').set_value(3.0)
        self.__parameters.get_parameter(name='structure.z_band_analysis.d_min').set_value(0.00)


        self.__parameters.get_parameter(name='structure.vectors.radius').set_value(0.25)
        self.__parameters.get_parameter(name='structure.vectors.line_width').set_value(0.2)
        self.__parameters.get_parameter(name='structure.vectors.interpolation_factor').set_value(0)
        self.__parameters.get_parameter(name='structure.vectors.length_limit_lower').set_value(1.0)
        self.__parameters.get_parameter(name='structure.vectors.length_limit_upper').set_value(3.0)
        self.__parameters.get_parameter(name='structure.vectors.smooth_orientation_sigma').set_value(0.0)


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

    def _set_defaults_motion(self):
        # Track sarcomere vectors
        self.__parameters.get_parameter(name='motion.track.max_disp_along').set_value(1.0)
        self.__parameters.get_parameter(name='motion.track.max_disp_perp').set_value(0.2)
        self.__parameters.get_parameter(name='motion.track.ori_tol').set_value(45.0)
        self.__parameters.get_parameter(name='motion.track.min_duration_s').set_value(0.08)
        self.__parameters.get_parameter(name='motion.track.max_gap_interp').set_value(3)

        # Group tracks
        self.__parameters.get_parameter(name='motion.group.by').set_value('myofibril')
        self.__parameters.get_parameter(name='motion.group.reference_frame').set_value(0)
        self.__parameters.get_parameter(name='motion.group.min_coverage').set_value(0.5)

        # Analyze track motion (ContractionNet)
        self.__parameters.get_parameter(name='motion.analyze.aggregate').set_value('auto')
        self.__parameters.get_parameter(name='motion.analyze.threshold').set_value(0.3)
        self.__parameters.get_parameter(name='motion.analyze.contr_time_min').set_value(0.2)
        self.__parameters.get_parameter(name='motion.analyze.merge_time_max').set_value(0.05)
        self.__parameters.get_parameter(name='motion.analyze.buffer_frames').set_value(3)
        self.__parameters.get_parameter(name='motion.analyze.min_valid_frames').set_value(0.5)
        self.__parameters.get_parameter(name='motion.analyze.filter_wl').set_value(13)
        self.__parameters.get_parameter(name='motion.analyze.filter_po').set_value(5)
        self.__parameters.get_parameter(name='motion.analyze.slen_lower').set_value(1.0)
        self.__parameters.get_parameter(name='motion.analyze.slen_upper').set_value(3.0)

        # LOI auto-detection (used when grouping by 'loi' without drawn lines)
        self.__parameters.get_parameter(name='loi.detect.n_lois').set_value(4)
        self.__parameters.get_parameter(name='loi.detect.ratio_seeds').set_value(0.1)
        self.__parameters.get_parameter(name='loi.detect.persistence').set_value(4)
        self.__parameters.get_parameter(name='loi.detect.threshold_distance').set_value(0.5)
        self.__parameters.get_parameter(name='loi.detect.mode').set_value('longest_in_cluster')
        self.__parameters.get_parameter(name='loi.detect.number_limits_lower').set_value(10)
        self.__parameters.get_parameter(name='loi.detect.number_limits_upper').set_value(50)
        self.__parameters.get_parameter(name='loi.detect.length_limits_lower').set_value(0.0)
        self.__parameters.get_parameter(name='loi.detect.length_limits_upper').set_value(200.0)
        self.__parameters.get_parameter(name='loi.detect.cluster_threshold_lois').set_value(40.0)
        self.__parameters.get_parameter(name='loi.detect.linkage').set_value('single')

    def _set_defaults_batch(self):
        self.__parameters.get_parameter(name='batch.pixel.size').set_value(0.1)
        self.__parameters.get_parameter(name='batch.frame.time').set_value(0.1)
        self.__parameters.get_parameter(name='batch.channel').set_value(0)
        self.__parameters.get_parameter(name='batch.axes').set_value("")
        self.__parameters.get_parameter(name='batch.force.override').set_value(False)
        self.__parameters.get_parameter(name='batch.thread_pool_size').set_value(3)
        self.__parameters.get_parameter(name='batch.recalculate.for.motion').set_value(False)
        self.__parameters.get_parameter(name='batch.delete_intermediary_tiffs').set_value(True)
        self.__parameters.get_parameter(name='batch.do_cellmask').set_value(True)
        self.__parameters.get_parameter(name='batch.do_zbands').set_value(True)
        self.__parameters.get_parameter(name='batch.do_vectors').set_value(True)
        self.__parameters.get_parameter(name='batch.do_myofibrils').set_value(True)
        self.__parameters.get_parameter(name='batch.do_domains').set_value(True)

    def __create_parameters(self):
        # region file-load parameters
        self.__parameters.set_parameter(name='file.load.correct_phase')
        # endregion
        # region structure parameters
        self.__parameters.set_parameter(name='structure.predict.network_path')
        self.__parameters.set_parameter(name='structure.predict.rescale_factor')
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
        self.__parameters.set_parameter(name='structure.vectors.smooth_orientation_sigma')


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

        # region motion parameters (track-based)
        self.__parameters.set_parameter(name='motion.track.max_disp_along')
        self.__parameters.set_parameter(name='motion.track.max_disp_perp')
        self.__parameters.set_parameter(name='motion.track.ori_tol')
        self.__parameters.set_parameter(name='motion.track.min_duration_s')
        self.__parameters.set_parameter(name='motion.track.max_gap_interp')

        self.__parameters.set_parameter(name='motion.group.by')
        self.__parameters.set_parameter(name='motion.group.reference_frame')
        self.__parameters.set_parameter(name='motion.group.min_coverage')

        self.__parameters.set_parameter(name='motion.analyze.aggregate')
        self.__parameters.set_parameter(name='motion.analyze.threshold')
        self.__parameters.set_parameter(name='motion.analyze.contr_time_min')
        self.__parameters.set_parameter(name='motion.analyze.merge_time_max')
        self.__parameters.set_parameter(name='motion.analyze.buffer_frames')
        self.__parameters.set_parameter(name='motion.analyze.min_valid_frames')
        self.__parameters.set_parameter(name='motion.analyze.filter_wl')
        self.__parameters.set_parameter(name='motion.analyze.filter_po')
        self.__parameters.set_parameter(name='motion.analyze.slen_lower')
        self.__parameters.set_parameter(name='motion.analyze.slen_upper')

        for _n in ('n_lois', 'ratio_seeds', 'persistence', 'threshold_distance', 'mode',
                   'number_limits_lower', 'number_limits_upper', 'length_limits_lower',
                   'length_limits_upper', 'cluster_threshold_lois', 'linkage'):
            self.__parameters.set_parameter(name=f'loi.detect.{_n}')
        # endregion

        # region batch processing parameters
        self.__parameters.set_parameter(name='batch.pixel.size')
        self.__parameters.set_parameter(name='batch.frame.time')
        self.__parameters.set_parameter(name='batch.channel')
        self.__parameters.set_parameter(name='batch.axes')
        self.__parameters.set_parameter(name='batch.force.override')
        self.__parameters.set_parameter(name='batch.thread_pool_size')
        self.__parameters.set_parameter(name='batch.delete_intermediary_tiffs')
        self.__parameters.set_parameter(name='batch.root')
        self.__parameters.set_parameter(name='batch.recalculate.for.motion')
        self.__parameters.set_parameter(name='batch.do_cellmask')
        self.__parameters.set_parameter(name='batch.do_zbands')
        self.__parameters.set_parameter(name='batch.do_vectors')
        self.__parameters.set_parameter(name='batch.do_myofibrils')
        self.__parameters.set_parameter(name='batch.do_domains')
        # endregion
