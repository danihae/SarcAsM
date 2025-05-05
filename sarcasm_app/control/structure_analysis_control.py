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


from typing import Any, Union, Tuple

import qtutils
from PyQt5.QtWidgets import QFileDialog
from bio_image_unet.progress import ProgressNotifier

from sarcasm import SarcAsM, Structure
from .chain_execution import ChainExecution
from .application_control import ApplicationControl
from .popup_export import ExportPopup
from ..view.parameter_structure_analysis import Ui_Form as StructureAnalysisWidget
from ..model import ApplicationModel
from sarcasm.type_utils import TypeUtils


class StructureAnalysisControl:
    """
    Handles button calls, parameter changes etc. from structure view
    """

    def __init__(self, structure_parameters_widget: StructureAnalysisWidget, main_control: ApplicationControl):
        self.__structure_parameters_widget = structure_parameters_widget
        self.__main_control = main_control
        self.__thread = None
        self.__worker = None
        self.__popup = None

    def __get_progress_notifier(self, worker) -> ProgressNotifier:
        progress_notifier = ProgressNotifier()

        def __internal_function(p):
            qtutils.inmain(lambda: self.__main_control.update_progress(int(p * 100)))  # wrap with qt main thread
            pass

        progress_notifier.set_progress_report(__internal_function)
        progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: worker.progress_details.emit(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (
                    hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))
        return progress_notifier

    def __predict_call(self, worker, model: ApplicationModel):

        progress_notifier = self.__get_progress_notifier(worker)

        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None
        cell: Structure = TypeUtils.unbox(model.cell)
        size: Union[Tuple[int, int]] = (model.parameters.get_parameter('structure.predict.size_width').get_value(),
                                        model.parameters.get_parameter('structure.predict.size_height').get_value())

        cell.detect_sarcomeres(frames=model.parameters.get_parameter('structure.frames').get_value(),
                               model_path=network_model,
                               max_patch_size=size,
                               clip_thres=(
                                   model.parameters.get_parameter('structure.predict.clip_thresh_min').get_value(),
                                   model.parameters.get_parameter('structure.predict.clip_thresh_max').get_value()),
                               progress_notifier=progress_notifier)

        pass

    def __predict_call_fast_movie(self, worker, model: ApplicationModel):
        progress_notifier = self.__get_progress_notifier(worker)

        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None
        cell: Structure = TypeUtils.unbox(model.cell)
        size: Union[Tuple[int, int, int]] = (model.parameters.get_parameter(
            'structure.predict_fast_movie.n_frames').get_value(),
                                             model.parameters.get_parameter(
                                                 'structure.predict_fast_movie.size_width').get_value(),
                                             model.parameters.get_parameter(
                                                 'structure.predict_fast_movie.size_height').get_value())
        cell.detect_z_bands_fast_movie(model_path=network_model,
                                       max_patch_size=size,
                                       clip_thres=(
                                           model.parameters.get_parameter(
                                               'structure.predict_fast_movie.clip_thresh_min').get_value(),
                                           model.parameters.get_parameter(
                                               'structure.predict_fast_movie.clip_thresh_max').get_value()),
                                       progress_notifier=progress_notifier)
        pass

    def __chk_prediction_network_fast_movie(self):
        if self.__main_control.model.parameters.get_parameter(
                'structure.predict_fast_movie.network_path').get_value() == '':
            self.__main_control.debug('no network file was chosen for fast movie prediction')
            return False
        return True

    def __chk_prediction_network(self):  # todo rename to zband_prediction or similar
        if self.__main_control.model.parameters.get_parameter('structure.predict.network_path').get_value() == '':
            self.__main_control.debug('no network file was chosen for prediction')
            return False
        return True

    def __chk_frames(self):
        frames = self.__main_control.model.parameters.get_parameter('structure.frames').get_value()
        if frames is None or frames == '':
            self.__check_frame_syntax()
            self.__main_control.debug(
                'no frames selected, please select the frame(s) in the specified format')
            return False
        return True

    def __chk_initialized(self):
        if not self.__main_control.model.is_initialized():
            self.__main_control.debug('file is not correctly initialized (or viewer was closed)')
            return False
        return True

    def on_btn_z_bands_predict(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_prediction_network():
            return
        cell: Structure = TypeUtils.unbox(self.__main_control.model.cell)
        message_finished = f'Z-bands detected and saved in {cell.base_dir}'
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__predict_call,
                                                   start_message='Start prediction of sarcomere z-bands',
                                                   finished_message=message_finished,
                                                   finished_action=self.__detect_sarcomeres_finished,
                                                   finished_successful_action=cell.commit)
        self.__worker = worker
        return worker

    def on_btn_z_bands_predict_fast_movies(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_prediction_network_fast_movie():
            return
        cell: Structure = TypeUtils.unbox(self.__main_control.model.cell)
        message_finished = f'Z-bands fast movies detected and saved in {cell.base_dir}'
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__predict_call_fast_movie,
                                                   start_message='Start prediction of sarcomere z-bands fast movies',
                                                   finished_message=message_finished,
                                                   finished_action=self.__detect_z_bands_fast_movie_finished,
                                                   finished_successful_action=cell.commit)
        self.__worker = worker
        return worker

    def __on_btn_analyze_cell_mask(self):
        if not self.__chk_initialized():
            return
        cell: Structure = TypeUtils.unbox(self.__main_control.model.cell)
        message_finished = f'Cell mask analysis completed.'

        def __internal_call(w, m: ApplicationModel):
            cell.analyze_cell_mask(frames=m.parameters.get_parameter('structure.frames').get_value(),
                                   threshold=m.parameters.get_parameter('structure.cell_mask.threshold').get_value())
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   start_message='Starting analysis of cell mask.',
                                                   finished_message=message_finished,
                                                   finished_successful_action=cell.commit)
        self.__worker = worker
        return worker

    def __parse_frames(self, frames_str: str):
        if frames_str == '':
            return None
        if frames_str.lower().__eq__('all'):
            return frames_str.lower()
        if frames_str.isnumeric():
            return int(frames_str)
        if frames_str.__contains__(','):
            list_str = frames_str.split(',')
            parsed_list = []
            for x in list_str:
                if x.isnumeric():
                    parsed_list.append(int(x))
            return parsed_list
        return 0  # if it's a wrong value just process first image (processing all could take long for a wrong input)

    def on_btn_z_band(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        def __internal_call(w, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: Structure = TypeUtils.unbox(m.cell)
            cell.analyze_z_bands(frames=m.parameters.get_parameter('structure.frames').get_value(),
                                 threshold=m.parameters.get_parameter(
                                     'structure.z_band_analysis.threshold').get_value(),
                                 min_length=m.parameters.get_parameter(
                                     'structure.z_band_analysis.min_length').get_value(),
                                 median_filter_radius=m.parameters.get_parameter(
                                     name='structure.z_band_analysis.median_filter_radius').get_value(),
                                 theta_phi_min=m.parameters.get_parameter(
                                     name='structure.z_band_analysis.theta_phi_min').get_value(),
                                 a_min=m.parameters.get_parameter(
                                     name='structure.z_band_analysis.a_min').get_value(),
                                 d_max=m.parameters.get_parameter(
                                     name='structure.z_band_analysis.d_max').get_value(),
                                 d_min=m.parameters.get_parameter(
                                     name='structure.z_band_analysis.d_min').get_value(),
                                 progress_notifier=progress_notifier)
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   start_message='Start Z-band Analysis',
                                                   finished_message='Finished Z-band Analysis',
                                                   finished_action=self.__z_band_analysis_finished,
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.commit()))
        self.__worker = worker
        return worker

    def on_btn_vectors(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        def __internal_call(w: Any, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: Structure = TypeUtils.unbox(m.cell)
            cell.analyze_sarcomere_vectors(
                frames=m.parameters.get_parameter('structure.frames').get_value(),
                slen_lims=(
                    m.parameters.get_parameter('structure.vectors.length_limit_lower').get_value(),
                    m.parameters.get_parameter('structure.vectors.length_limit_upper').get_value()
                ),
                median_filter_radius=m.parameters.get_parameter('structure.vectors.radius').get_value(),
                linewidth=m.parameters.get_parameter('structure.vectors.line_width').get_value(),
                interp_factor=m.parameters.get_parameter('structure.vectors.interpolation_factor').get_value(),
                backend='threading', progress_notifier=progress_notifier
            )
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   finished_message='Finished sarcomere vectors analysis',
                                                   start_message='Start sarcomere vectors analysis',
                                                   finished_action=self.__sarcomere_analysis_finished,
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.commit()))
        self.__worker = worker
        return worker
        # AND-gated double wavelet analysis of sarcomere structure to locally obtain length and angle

        pass

    def on_btn_myofibril(self):
        """
        analyze_myofibrils(self, frames=None, n_seeds=200, score_threshold=None, persistence=3,
                           threshold_distance=0.3,
                           save_all=False, plot=False)
        """
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        # estimate myofibril lengths using line-growth algorithm
        def __internal_call(w: Any, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: Structure = TypeUtils.unbox(m.cell)
            cell.analyze_myofibrils(
                frames=m.parameters.get_parameter('structure.frames').get_value(),
                ratio_seeds=m.parameters.get_parameter('structure.myofibril.ratio_seeds').get_value(),
                persistence=m.parameters.get_parameter('structure.myofibril.persistence').get_value(),
                threshold_distance=m.parameters.get_parameter('structure.myofibril.threshold_distance').get_value(),
                n_min=m.parameters.get_parameter('structure.myofibril.n_min').get_value(),
                progress_notifier=progress_notifier
            )
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   start_message='Start myofibril analysis',
                                                   finished_message='Finished myofibril analysis',
                                                   finished_action=self.__myofibril_analysis_finished,
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.commit()))
        self.__worker = worker
        return worker
        pass

    def on_btn_domain_analysis(self):
        """
        call domain analysis in backend
        """
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        def __internal_call(w: Any, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: Structure = TypeUtils.unbox(m.cell)
            cell.analyze_sarcomere_domains(
                frames=m.parameters.get_parameter('structure.frames').get_value(),
                d_max=m.parameters.get_parameter('structure.domain.analysis.d_max').get_value(),
                cosine_min=m.parameters.get_parameter('structure.domain.analysis.cosine_min').get_value(),
                leiden_resolution=m.parameters.get_parameter('structure.domain.analysis.leiden_resolution').get_value(),
                random_seed=m.parameters.get_parameter('structure.domain.analysis.random_seed').get_value(),
                area_min=m.parameters.get_parameter('structure.domain.analysis.area_min').get_value(),
                dilation_radius=m.parameters.get_parameter('structure.domain.analysis.dilation_radius').get_value(),
                progress_notifier=progress_notifier
            )
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   start_message='Start sarcomere domains analysis',
                                                   finished_message='Finished sarcomere domains analysis',
                                                   finished_action=self.__domain_analysis_finished,
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.commit()))
        self.__worker = worker
        return worker
        pass

    def on_btn_search_network(self):
        # f_name is a tuple
        f_name = QFileDialog.getOpenFileName(caption='Open Network File', filter="Network Files (*.pt)")
        if f_name is not None:
            self.__structure_parameters_widget.le_network.setText(f_name[0])

    def on_btn_fast_movie_search_network(self):
        f_name = QFileDialog.getOpenFileName(caption='Open Network File', filter="Network Files (*.pt)")
        if f_name is not None:
            self.__structure_parameters_widget.le_fast_movie_network_model.setText(f_name[0])
        pass

    def __filter_input_prediction_size(self, element):
        if not (hasattr(element, 'value') and hasattr(element, 'setValue')):
            return

        value = element.value()
        factor = value // 16
        if factor * 16 != value:
            element.setValue(factor * 16)
            pass
        pass

    def __check_frame_syntax(self):
        text = self.__structure_parameters_widget.le_general_frames.text()
        value = self.__parse_frames(text)
        if not text.isnumeric() and (value == 0 or value is None):
            # this is an error
            self.__structure_parameters_widget.le_general_frames.setStyleSheet("QLineEdit{background : red;}")
            pass
        else:
            self.__structure_parameters_widget.le_general_frames.setStyleSheet(
                "QLineEdit{background : lightgreen;}")
        pass

    def on_analyze_structure(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_prediction_network():
            return
        if not self.__chk_frames():
            return
        # if not self.__chk_cell_mask_prediction_network():
        #    return
        # predict, z band analysis, wavelet analysis, myofibril length
        chain = ChainExecution(self.__main_control.model.currentlyProcessing, self.__main_control.debug)
        chain.add_step(self.on_btn_z_bands_predict)
        # chain.add_step(self.__on_btn_analyze_cell_mask) # excluded from analyze structure
        chain.add_step(self.on_btn_z_band)
        chain.add_step(self.on_btn_vectors)
        chain.add_step(self.on_btn_myofibril)
        chain.add_step(self.on_btn_domain_analysis)
        chain.execute()
        pass

    def __open_export_popup(self):
        if not self.__chk_initialized():
            return

        from pathlib import Path
        name=Path(self.__main_control.model.cell.filepath).stem
        self.__popup = ExportPopup(self.__main_control.model, self.__main_control, popup_type='structure',filename_pattern=f'%_{name}.csv')
        self.__popup.show_popup()

    def bind_events(self):
        """
        Binds ui events to backend methods/functions
        also binds ui fields to model parameters
        """
        self.__structure_parameters_widget.btn_analyze_structure.clicked.connect(self.on_analyze_structure)
        self.__structure_parameters_widget.btn_export_structure_data.clicked.connect(self.__open_export_popup)

        # monitor the value of predict_size and keep it dividable by 16
        self.__structure_parameters_widget.sb_predict_size_width.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_predict_size_width))
        self.__structure_parameters_widget.sb_predict_size_height.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_predict_size_height))

        self.__structure_parameters_widget.le_general_frames.editingFinished.connect(self.__check_frame_syntax)

        self.__structure_parameters_widget.btn_structure_predict.clicked.connect(self.on_btn_z_bands_predict)
        self.__structure_parameters_widget.btn_structure_z_band.clicked.connect(self.on_btn_z_band)
        self.__structure_parameters_widget.btn_structure_vectors.clicked.connect(self.on_btn_vectors)
        self.__structure_parameters_widget.btn_structure_myofibril.clicked.connect(self.on_btn_myofibril)
        self.__structure_parameters_widget.btn_search_network.clicked.connect(self.on_btn_search_network)
        self.__structure_parameters_widget.btn_structure_domain_analysis.clicked.connect(self.on_btn_domain_analysis)
        self.__structure_parameters_widget.btn_fast_movie_search.clicked.connect(self.on_btn_fast_movie_search_network)
        self.__structure_parameters_widget.btn_fast_movie.clicked.connect(self.on_btn_z_bands_predict_fast_movies)
        self.__structure_parameters_widget.btn_structure_analyze_cell_mask.clicked.connect(self.__on_btn_analyze_cell_mask)

        # todo: bind parameters to ui elements
        parameters = self.__main_control.model.parameters
        widget = self.__structure_parameters_widget

        parameters.get_parameter(name='structure.predict.network_path').connect(widget.le_network)
        parameters.get_parameter(name='structure.predict.size_width').connect(widget.sb_predict_size_width)
        parameters.get_parameter(name='structure.predict.size_height').connect(widget.sb_predict_size_height)
        parameters.get_parameter(name='structure.predict.clip_thresh_min').connect(widget.dsb_predict_clip_thresh_min)
        parameters.get_parameter(name='structure.predict.clip_thresh_max').connect(widget.dsb_predict_clip_thresh_max)

        parameters.get_parameter(name='structure.predict_fast_movie.network_path').connect(
            widget.le_fast_movie_network_model)
        parameters.get_parameter(name='structure.predict_fast_movie.n_frames').connect(widget.sb_fast_movie_n_frames)
        parameters.get_parameter(name='structure.predict_fast_movie.size_width').connect(widget.sb_fast_movie_width)
        parameters.get_parameter(name='structure.predict_fast_movie.size_height').connect(widget.sb_fast_movie_height)
        parameters.get_parameter(name='structure.predict_fast_movie.clip_thresh_min').connect(
            widget.dsb_fast_movie_clip_thresh_min)
        parameters.get_parameter(name='structure.predict_fast_movie.clip_thresh_max').connect(
            widget.dsb_fast_movie_clip_thresh_max)

        parameters.get_parameter(name='structure.cell_mask.threshold').connect(widget.dsb_cell_mask_threshold)

        parameters.get_parameter(name='structure.frames').set_value_parser(self.__parse_frames)
        parameters.get_parameter(name='structure.frames').connect(widget.le_general_frames)

        parameters.get_parameter(name='structure.z_band_analysis.threshold').connect(widget.dsb_z_band_threshold)
        parameters.get_parameter(name='structure.z_band_analysis.min_length').connect(widget.dsb_z_band_min_length)
        parameters.get_parameter(name='structure.z_band_analysis.median_filter_radius').connect(
            widget.dsb_z_band_median_filter_radius)
        parameters.get_parameter(name='structure.z_band_analysis.theta_phi_min').connect(
            widget.dsb_z_band_theta_phi_min)
        parameters.get_parameter(name='structure.z_band_analysis.a_min').connect(widget.dsb_z_band_a_min)
        parameters.get_parameter(name='structure.z_band_analysis.d_max').connect(widget.dsb_z_band_d_max)
        parameters.get_parameter(name='structure.z_band_analysis.d_min').connect(widget.dsb_z_band_d_min)

        parameters.get_parameter(name='structure.vectors.radius').connect(widget.dsb_vectors_radius)
        parameters.get_parameter(name='structure.vectors.line_width').connect(widget.dsb_vectors_line_width)
        parameters.get_parameter(name='structure.vectors.interpolation_factor').connect(
            widget.sb_vectors_interpolation_factor)
        parameters.get_parameter(name='structure.vectors.length_limit_lower').connect(widget.dsb_vectors_len_lims_min)
        parameters.get_parameter(name='structure.vectors.length_limit_upper').connect(widget.dsb_vectors_len_lims_max)

        parameters.get_parameter(name='structure.myofibril.ratio_seeds').connect(widget.dsb_myofibril_ratio_seeds)
        parameters.get_parameter(name='structure.myofibril.persistence').connect(widget.sb_myofibril_persistence)
        parameters.get_parameter(name='structure.myofibril.threshold_distance').connect(
            widget.dsb_myofibril_thresh_dist)
        parameters.get_parameter(name='structure.myofibril.n_min').connect(widget.sb_myofibril_n_min)
        parameters.get_parameter(name='structure.myofibril.median_filter_radius').connect(
            widget.dsb_myofibril_median_filter_radius)

        parameters.get_parameter(name='structure.domain.analysis.d_max').connect(widget.dsb_domains_d_max)
        parameters.get_parameter(name='structure.domain.analysis.cosine_min').connect(widget.dsb_domains_cosine_min)
        parameters.get_parameter(name='structure.domain.analysis.leiden_resolution').connect(
            widget.dsb_domains_leiden_resolution)
        parameters.get_parameter(name='structure.domain.analysis.random_seed').connect(widget.sb_domains_random_seed)
        parameters.get_parameter(name='structure.domain.analysis.area_min').connect(widget.dsb_domains_area_min)
        parameters.get_parameter(name='structure.domain.analysis.dilation_radius').connect(
            widget.dsb_domains_dilation_radius)

        pass

    def __detect_sarcomeres_finished(self):
        self.__main_control.init_z_band_stack()
        self.__main_control.init_m_band_stack()
        self.__main_control.init_cell_mask_stack()
        self.__main_control.init_sarcomere_mask_stack()

    def __detect_z_bands_fast_movie_finished(self):
        self.__main_control.init_z_band_stack(fastmovie=True)

    def __z_band_analysis_finished(self):
        self.__main_control.init_z_lateral_connections()

    def __sarcomere_analysis_finished(self):
        self.__main_control.init_sarcomere_vector_stack()

    def __myofibril_analysis_finished(self):
        self.__main_control.init_myofibril_lines_stack()

    def __domain_analysis_finished(self):
        self.__main_control.init_sarcomere_domain_stack()
