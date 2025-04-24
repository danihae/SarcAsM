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


import numpy as np
from PyQt5.QtWidgets import QFileDialog

from sarcasm import Plots
from sarcasm_app.control.chain_execution import ChainExecution
from sarcasm_app import ApplicationControl
from sarcasm_app.control.popup_export import ExportPopup
from sarcasm_app.view.parameter_motion_analysis import Ui_Form as MotionAnalysisWidget
from sarcasm.motion import Motion
from sarcasm.type_utils import TypeUtils


class MotionAnalysisControl:
    """
    The MotionAnalysisControl handles all the motion analysis functions on the ui.
    """

    def __init__(self, motion_analysis_widget: MotionAnalysisWidget, main_control: ApplicationControl):
        self.__motion_analysis_widget = motion_analysis_widget
        self.__main_control = main_control
        self.__thread = None
        self.__worker = None
        self.__popup = None
        self.__export_popup = None
        pass

    def __chk_loi_file_selected(self):
        if self.__main_control.model.sarcomere is None:
            self.__main_control.debug('no loi is selected')
            return False
        return True
        pass

    def __chk_initialized(self):
        if not self.__main_control.model.is_initialized():
            self.__main_control.debug('file is not correctly initialized (or viewer was closed)')
            return False
        return True

    def __chk_contraction_weights(self):
        if self.__main_control.model.parameters.get_parameter('motion.systoles.weights').get_value() == '':
            self.__main_control.debug('no file was chosen for systoles weights')
            return False
        return True

    def __on_loi_selection_changed(self, txt):
        print(txt)
        if txt is None or txt == '':  # exit method on empty selection
            return
        # linedict [filename][line_as_txt] = line_object
        # line objects müssten also alle im line_dict drin sein
        cell = TypeUtils.unbox(self.__main_control.model.cell)

        line = self.__main_control.model.line_dictionary[cell.filepath][txt]
        file_name, scan_line = self.__main_control.get_file_name_from_scheme(cell.filepath, txt)
        if self.__main_control.model.sarcomere is None or \
                self.__main_control.model.sarcomere.loi_name != Motion.get_loi_name_from_file_name(file_name):
            self.__main_control.model.init_sarcomere(file_name)
            print('sarcomere reloaded:' + txt)
            pass
        # get selection and change color of selected sarcomere-loi

        loi_layer = self.__main_control.layer_loi
        lines = loi_layer.data
        widths = loi_layer.edge_width

        loi_layer.data.clear()
        loi_layer.edge_width.clear()
        loi_layer.edge_color = []

        for index, line_data in enumerate(lines):
            if line_data[0][0] == line[0][0] and line_data[0][-1] == line[0][1] and line_data[-1][0] == line[1][0] and \
                    line_data[-1][-1] == line[1][1] and line[2] == loi_layer.edge_width[index]:
                loi_layer.add_paths(data=line_data, edge_width=widths[index], edge_color='yellow')
                pass
            else:
                loi_layer.add_paths(data=line_data, edge_width=widths[index], edge_color='red')
                pass
            pass
        pass

    def __update_loi_combo_box(self, lines):
        self.__motion_analysis_widget.cb_loi_file.clear()
        keys = lines.keys()
        self.__motion_analysis_widget.cb_loi_file.addItems(list(keys))
        pass

    def on_btn_detect_peaks(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_loi_file_selected():
            return

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=lambda w, m: m.sarcomere.detekt_peaks(
                                                       thres=m.parameters.get_parameter(
                                                           'motion.detect_peaks.threshold').get_value(),
                                                       min_dist=m.parameters.get_parameter(
                                                           'motion.detect_peaks.min_distance').get_value(),
                                                       width=m.parameters.get_parameter(
                                                           'motion.detect_peaks.width').get_value()),
                                                   start_message='Starting Detect Peaks',
                                                   finished_message='Finished Detect Peaks',
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.sarcomere, lambda s: s.commit()))
        self.__worker = worker
        return worker
        pass

    def on_btn_track_z_bands(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_loi_file_selected():
            return
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=lambda w, m: m.sarcomere.track_z_bands(
                                                       search_range=m.parameters.get_parameter(
                                                           'motion.track_z_bands.search_range').get_value(),
                                                       memory_tracking=m.parameters.get_parameter(
                                                           'motion.track_z_bands.memory').get_value(),
                                                       memory_interpol=m.parameters.get_parameter(
                                                           'motion.track_z_bands.memory_interpolation').get_value()),
                                                   start_message='Starting track z bands',
                                                   finished_message='Finished track z bands',
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.sarcomere, lambda s: s.commit()))
        self.__worker = worker
        return worker
        pass

    def on_btn_predict_analyze_contractions(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_loi_file_selected():
            return
        if not self.__chk_contraction_weights():
            return
        worker = self.__main_control.run_async_new(
            parameters=self.__main_control.model,
            call_lambda=lambda w, m: m.sarcomere.detect_analyze_contractions(model=m.parameters.get_parameter(
                'motion.systoles.weights').get_value(),
                                                                             threshold=m.parameters.get_parameter(
                                                                                 'motion.systoles.threshold').get_value(),
                                                                             slen_lims=(m.parameters.get_parameter(
                                                                                 'motion.systoles.slen_limits.lower').get_value(),
                                                                                        m.parameters.get_parameter(
                                                                                            'motion.systoles.slen_limits.upper').get_value()),
                                                                             n_sarcomeres_min=m.parameters.get_parameter(
                                                                                 'motion.systoles.n_sarcomeres_min').get_value(),
                                                                             buffer_frames=m.parameters.get_parameter(
                                                                                 'motion.systoles.buffer_frames').get_value(),
                                                                             contr_time_min=m.parameters.get_parameter(
                                                                                 'motion.systoles.contr_time_min').get_value(),
                                                                             merge_time_max=m.parameters.get_parameter(
                                                                                 'motion.systoles.merge_time_max').get_value()),
            start_message='Starting Detect and Analyze Contractions',
            finished_message='Finished Detect and Analyze Contractions',
            finished_successful_action=TypeUtils.if_present(self.__main_control.model.sarcomere, lambda s: s.commit()))
        self.__worker = worker
        return worker
        pass

    def on_btn_get_sarcomere_trajs(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_loi_file_selected():
            return
        worker = self.__main_control.run_async_new(
            parameters=self.__main_control.model,
            call_lambda=self.__on_btn_get_and_analyze_sarcomere_trajectories,
            start_message='Starting get sarcomere trajectories',
            finished_message='Finished get sarcomere trajectories',
            finished_successful_action=TypeUtils.if_present(self.__main_control.model.sarcomere, lambda s: s.commit()))
        self.__worker = worker
        return worker
        pass

    @staticmethod
    def __on_btn_get_and_analyze_sarcomere_trajectories(w, m):
        m.sarcomere.get_trajectories(
            slen_lims=(m.parameters.get_parameter(
                'motion.get_sarcomere_trajectories.s_length_limits_lower').get_value(),
                       m.parameters.get_parameter(
                           'motion.get_sarcomere_trajectories.s_length_limits_upper').get_value()),
            dilate_contr=m.parameters.get_parameter(
                'motion.get_sarcomere_trajectories.dilate_systoles').get_value(),
            filter_params_vel=(m.parameters.get_parameter(
                'motion.get_sarcomere_trajectories.filter_params_vel.window_length').get_value(),
                               m.parameters.get_parameter(
                                   'motion.get_sarcomere_trajectories.filter_params_vel.polyorder').get_value()),
            equ_lims=(m.parameters.get_parameter(
                'motion.get_sarcomere_trajectories.equ_limits_lower').get_value(),
                      m.parameters.get_parameter(
                          'motion.get_sarcomere_trajectories.equ_limits_upper').get_value()))

        m.sarcomere.analyze_trajectories()
        pass

    def on_btn_systoles_search_weights(self):
        # f_name is a tuple
        f_name = QFileDialog.getOpenFileName(caption='Open Weights File', filter="Network Files (*.pt)")
        if f_name is not None:
            self.__motion_analysis_widget.le_systoles_weights.setText(f_name[0])
        pass

    def on_analyze_motion(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_loi_file_selected():
            return
        if not self.__chk_contraction_weights():
            return
        chain = ChainExecution(self.__main_control.model.currentlyProcessing, self.__main_control.debug)
        chain.add_step(self.on_btn_detect_peaks)
        chain.add_step(self.on_btn_track_z_bands)
        chain.add_step(self.on_btn_predict_analyze_contractions)
        chain.add_step(self.on_btn_get_sarcomere_trajs)
        chain.execute()

        pass

    def __on_btn_plot_summary(self):
        if not self.__chk_initialized():
            return
        Plots.plot_loi_summary_motion(self.__main_control.model.sarcomere)

        pass

    def __on_btn_export_motion_data(self):
        if not self.__chk_initialized():
            return

        from pathlib import Path
        name = Path(self.__main_control.model.cell.filepath).stem
        self.__export_popup = ExportPopup(self.__main_control.model, self.__main_control, popup_type='motion',filename_pattern=f'%_{name}_{self.__main_control.model.sarcomere.loi_name}.csv')
        self.__export_popup.show_popup()


    def bind_events(self):
        self.__motion_analysis_widget.btn_export_motion_data.clicked.connect(self.__on_btn_export_motion_data)
        self.__motion_analysis_widget.btn_plot_summary.clicked.connect(self.__on_btn_plot_summary)
        self.__motion_analysis_widget.btn_analyze_motion.clicked.connect(self.on_analyze_motion)
        self.__motion_analysis_widget.btn_motion_detect_peaks.clicked.connect(self.on_btn_detect_peaks)
        self.__motion_analysis_widget.btn_motion_track_z_bands.clicked.connect(self.on_btn_track_z_bands)
        self.__motion_analysis_widget.btn_motion_get_sarcomere_trajs.clicked.connect(self.on_btn_get_sarcomere_trajs)

        self.__motion_analysis_widget.btn_motion_systoles.clicked.connect(self.on_btn_predict_analyze_contractions)
        self.__motion_analysis_widget.btn_systoles_search_weights.clicked.connect(self.on_btn_systoles_search_weights)

        self.__main_control.set_callback_loi_list_updated(self.__update_loi_combo_box)
        self.__motion_analysis_widget.cb_loi_file.currentTextChanged.connect(self.__on_loi_selection_changed)

        self.__main_control.model.parameters.get_parameter(name='motion.detect_peaks.threshold').connect(
            self.__motion_analysis_widget.dsb_detect_peaks_threshold)
        self.__main_control.model.parameters.get_parameter(name='motion.detect_peaks.min_distance').connect(
            self.__motion_analysis_widget.dsb_detect_peaks_min_distance)
        self.__main_control.model.parameters.get_parameter(name='motion.detect_peaks.width').connect(
            self.__motion_analysis_widget.dsb_detect_peaks_width)

        self.__main_control.model.parameters.get_parameter(name='motion.track_z_bands.search_range').connect(
            self.__motion_analysis_widget.dsb_track_z_bands_search_range)
        self.__main_control.model.parameters.get_parameter(name='motion.track_z_bands.memory').connect(
            self.__motion_analysis_widget.sb_track_z_bands_memory)
        self.__main_control.model.parameters.get_parameter(name='motion.track_z_bands.memory_interpolation').connect(
            self.__motion_analysis_widget.sb_track_z_band_memory_interp)

        self.__main_control.model.parameters.get_parameter(name='motion.systoles.weights').connect(
            self.__motion_analysis_widget.le_systoles_weights)
        self.__main_control.model.parameters.get_parameter(name='motion.systoles.threshold').connect(
            self.__motion_analysis_widget.dsb_systoles_thresh)
        self.__main_control.model.parameters.get_parameter(name='motion.systoles.slen_limits.lower').connect(
            self.__motion_analysis_widget.dsb_systoles_slen_limits_lower)
        self.__main_control.model.parameters.get_parameter(name='motion.systoles.slen_limits.upper').connect(
            self.__motion_analysis_widget.dsb_systoles_slen_limits_upper)
        self.__main_control.model.parameters.get_parameter(name='motion.systoles.n_sarcomeres_min').connect(
            self.__motion_analysis_widget.sb_systoles_n_sarcomere_min)
        self.__main_control.model.parameters.get_parameter(name='motion.systoles.buffer_frames').connect(
            self.__motion_analysis_widget.sb_systoles_buffer_frames)
        self.__main_control.model.parameters.get_parameter(name='motion.systoles.contr_time_min').connect(
            self.__motion_analysis_widget.dsb_systoles_contr_time_min)
        self.__main_control.model.parameters.get_parameter(name='motion.systoles.merge_time_max').connect(
            self.__motion_analysis_widget.dsb_systoles_merge_time_max)

        self.__main_control.model.parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.s_length_limits_lower').connect(
            self.__motion_analysis_widget.dsb_get_sarc_traj_slen_lower)
        self.__main_control.model.parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.s_length_limits_upper').connect(
            self.__motion_analysis_widget.dsb_get_sarc_traj_slen_upper)
        self.__main_control.model.parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.dilate_systoles').connect(
            self.__motion_analysis_widget.dsb_get_sarc_traj_dilate_systoles)
        self.__main_control.model.parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.filter_params_vel.window_length').connect(
            self.__motion_analysis_widget.dsb_get_sarc_traj_filter_vel_wl)
        self.__main_control.model.parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.filter_params_vel.polyorder').connect(
            self.__motion_analysis_widget.dsb_get_sarc_traj_filter_vel_po)
        self.__main_control.model.parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.equ_limits_lower').connect(
            self.__motion_analysis_widget.dsb_get_sarc_traj_equ_lims_lower)
        self.__main_control.model.parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.equ_limits_upper').connect(
            self.__motion_analysis_widget.dsb_get_sarc_traj_equ_lims_upper)

        pass
