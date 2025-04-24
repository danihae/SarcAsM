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
import os

from sarcasm import TypeUtils, SarcAsM, Structure
from .application_control import ApplicationControl
from ..view.parameter_loi_analysis import Ui_Form as LoiAnalysisWidget
from ..model import ApplicationModel, Parameters


class LOIAnalysisControl:
    """
    The LOIAnalysisControl handles the connection between loi-analysis-backend and the view.
    """

    def __init__(self, loi_analysis_widget: LoiAnalysisWidget, main_control: ApplicationControl):
        self.__loi_analysis_widget = loi_analysis_widget
        self.__main_control = main_control
        self.__thread = None
        self.__worker = None

    def __chk_initialized(self):
        if not self.__main_control.model.is_initialized():
            self.__main_control.debug('file is not correctly initialized (or viewer was closed)')
            return False
        return True

    @staticmethod
    def __call_detect_lois(w, m: ApplicationModel):
        print('start detect lois')
        cell: Structure = TypeUtils.unbox(m.cell)

        cell.detect_lois(frame=m.parameters.get_parameter(name='loi.detect.frame').get_value(),
                         n_lois=m.parameters.get_parameter(name='loi.detect.n_lois').get_value(),
                         ratio_seeds=m.parameters.get_parameter(name='loi.detect.ratio_seeds').get_value(),
                         persistence=m.parameters.get_parameter(name='loi.detect.persistence').get_value(),
                         threshold_distance=m.parameters.get_parameter(
                             name='loi.detect.threshold_distance').get_value(),
                         mode=m.parameters.get_parameter(name='loi.detect.mode').get_value(),
                         number_lims=(
                             m.parameters.get_parameter(name='loi.detect.number_limits_lower').get_value(),
                             m.parameters.get_parameter(name='loi.detect.number_limits_upper').get_value()),
                         length_lims=(
                             m.parameters.get_parameter(name='loi.detect.length_limits_lower').get_value(),
                             m.parameters.get_parameter(name='loi.detect.length_limits_upper').get_value()),
                         sarcomere_mean_length_lims=(m.parameters.get_parameter(
                             name='loi.detect.sarcomere_mean_length_limits_lower').get_value(),
                                                     m.parameters.get_parameter(
                                                         name='loi.detect.sarcomere_mean_length_limits_upper').get_value()),
                         sarcomere_std_length_lims=(m.parameters.get_parameter(
                             name='loi.detect.sarcomere_std_length_limits_lower').get_value(),
                                                    m.parameters.get_parameter(
                                                        name='loi.detect.sarcomere_std_length_limits_upper').get_value()),
                         midline_mean_length_lims=(m.parameters.get_parameter(
                             name='loi.detect.midline_mean_length_limits_lower').get_value(),
                                                   m.parameters.get_parameter(
                                                       name='loi.detect.midline_mean_length_limits_upper').get_value()),
                         midline_std_length_lims=(m.parameters.get_parameter(
                             name='loi.detect.midline_std_length_limits_lower').get_value(),
                                                  m.parameters.get_parameter(
                                                      name='loi.detect.midline_std_length_limits_upper').get_value()),
                         midline_min_length_lims=(m.parameters.get_parameter(
                             name='loi.detect.midline_min_length_limits_lower').get_value(),
                                                  m.parameters.get_parameter(
                                                      name='loi.detect.midline_min_length_limits_upper').get_value()),
                         distance_threshold_lois=m.parameters.get_parameter(
                             name='loi.detect.cluster_threshold_lois').get_value(),
                         linkage=m.parameters.get_parameter(name='loi.detect.linkage').get_value(),
                         linewidth=m.parameters.get_parameter(name='loi.detect.line_width').get_value(),
                         order=m.parameters.get_parameter(name='loi.detect.order').get_value())

    def _finished_detect_lois(self):
        # get loi's from cell and add them to napari
        print('finished loi detection...')
        # before adding line to napari, check if the line is already in napari's 'loi' layer
        if self.__main_control.model.cell is None:  # exit method
            return

        self.__main_control.init_lois()
        pass


    def on_btn_detect_lois(self):
        if not self.__chk_initialized() or self.__main_control.model.cell is None:
            return
        self.__worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                          call_lambda=self.__call_detect_lois,
                                                          start_message='Starting loi detection',
                                                          finished_message='Finished loi detection',
                                                          finished_action=self._finished_detect_lois,
                                                          finished_successful_action=self.__main_control.
                                                          model.cell.commit)

    @staticmethod
    def __store_lois(w, p):
        # note that first coordinate in the point tuples is Y and second is X
        # pos_vectors = np.array([[[100, 100], [200, 200]],[[300,300],[400,300]]])
        # self.__main_control.layer_loi.add_lines(pos_vectors,edge_width=[10,5],edge_color='red')
        # self.__main_control.layer_loi.add_lines(np.array([[100,200],[100,400]]),edge_color='red',edge_width=15)
        # data=self.__main_control.layer_loi.data
        # widths=self.__main_control.layer_loi.edge_width
        # print(data)
        # print(widths)
        # [array([[100., 100.],[200., 200.]]), array([[300., 300.],[400., 300.]]), array([[100., 200.],[100., 400.]])]
        w.progress.emit(10)
        max_count = len(p['loi_layer'].data)
        step = 90 / max_count
        for index, line in enumerate(p['loi_layer'].data):
            width = float(p['loi_layer'].edge_width[index])
            start = (int(line[0][0]), int(line[0][1]))
            end = (int(line[-1][0]), int(line[-1][1]))
            loi_file = p['cell'].base_dir + f'{start[0]}_{start[1]}_{end[0]}_{end[1]}_{width}_loi.json'
            if not os.path.exists(loi_file):
                p['main_control'].on_update_loi_list(line_start=start, line_end=end, line_thickness=width,line=line)
                # extract intensity profiles and save LOI files
                p['cell'].create_loi_data(line, linewidth=width) # todo: has to be tested
                pass
            w.progress.emit(10 + index * step)
            pass
        # [10, 5, 15]
        w.progress.emit(100)
        pass

    def on_btn_store_lois(self):
        # check loi's in napari
        # extract loi coordinates
        # todo: add button cleanup loi's (remove all loi files from loi-lines not existing in napari currently)
        if not self.__chk_initialized():
            return
        parameters = {
            'loi_layer': self.__main_control.layer_loi,
            'cell': self.__main_control.model.cell,
            'main_control': self.__main_control
        }

        self.__worker = self.__main_control.run_async_new(parameters=parameters, call_lambda=self.__store_lois,
                                                          start_message='Start Store LOI\'s',
                                                          finished_message='Finished Store LOI\'s')
        pass

    def bind_events(self):
        """if just detect_lois is called,
         then we need a second button for storing and calculating loi data from manually drawn lois

         """
        self.__loi_analysis_widget.btn_detect_lois.clicked.connect(self.on_btn_detect_lois)
        self.__loi_analysis_widget.btn_store_lois.clicked.connect(self.on_btn_store_lois)

        parameters: Parameters = self.__main_control.model.parameters
        widget = self.__loi_analysis_widget
        widget.cb_mode.addItems(['fit_straight_line', 'longest_in_cluster', 'random_from_cluster', 'random_line'])

        parameters.get_parameter(name='loi.detect.frame').connect(widget.sb_detect_loi_frame)
        parameters.get_parameter(name='loi.detect.n_lois').connect(widget.sb_detect_loi_n_lois)
        parameters.get_parameter(name='loi.detect.ratio_seeds').connect(widget.dsb_ratio_seeds)
        parameters.get_parameter(name='loi.detect.persistence').connect(widget.sb_detect_loi_persistence)
        parameters.get_parameter(name='loi.detect.threshold_distance').connect(widget.dsb_detect_loi_threshold_distance)
        parameters.get_parameter(name='loi.detect.mode').connect(widget.cb_mode)
        parameters.get_parameter(name='loi.detect.number_limits_lower').connect(widget.sb_detect_loi_num_lims_min)
        parameters.get_parameter(name='loi.detect.number_limits_upper').connect(widget.sb_detect_loi_num_lims_max)
        parameters.get_parameter(name='loi.detect.length_limits_lower').connect(widget.dsb_limit_length_min)
        parameters.get_parameter(name='loi.detect.length_limits_upper').connect(widget.dsb_limit_length_max)
        parameters.get_parameter(name='loi.detect.sarcomere_mean_length_limits_lower').connect(
            widget.dsb_mean_length_limit_min)
        parameters.get_parameter(name='loi.detect.sarcomere_mean_length_limits_upper').connect(
            widget.dsb_mean_length_limit_max)
        parameters.get_parameter(name='loi.detect.sarcomere_std_length_limits_lower').connect(
            widget.dsb_std_length_lims_min)
        parameters.get_parameter(name='loi.detect.sarcomere_std_length_limits_upper').connect(
            widget.dsb_std_length_lims_max)
        parameters.get_parameter(name='loi.detect.midline_mean_length_limits_lower').connect(
            widget.dsb_midline_mean_length_lims_min)
        parameters.get_parameter(name='loi.detect.midline_mean_length_limits_upper').connect(
            widget.dsb_midline_mean_length_lims_max)
        parameters.get_parameter(name='loi.detect.midline_std_length_limits_lower').connect(
            widget.dsb_limit_midline_std_dev_length_min)
        parameters.get_parameter(name='loi.detect.midline_std_length_limits_upper').connect(
            widget.dsb_limit_midline_std_dev_length_max)
        parameters.get_parameter(name='loi.detect.midline_min_length_limits_lower').connect(
            widget.dsb_limit_midline_min_length_min)
        parameters.get_parameter(name='loi.detect.midline_min_length_limits_upper').connect(
            widget.dsb_limit_midline_min_length_max)
        parameters.get_parameter(name='loi.detect.cluster_threshold_lois').connect(
            widget.dsb_detect_loi_clustering_threshold)
        parameters.get_parameter(name='loi.detect.linkage').connect(widget.le_linkage)
        parameters.get_parameter(name='loi.detect.line_width').connect(widget.dsb_detect_loi_line_width)
        parameters.get_parameter(name='loi.detect.order').connect(widget.sb_order)

        pass
