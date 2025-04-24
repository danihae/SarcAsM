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


import glob

import numpy as np
from PyQt5.QtWidgets import QFileDialog
from tifffile import tifffile

from sarcasm import SarcAsM, Structure
from .popup_export import ExportPopup
from .application_control import ApplicationControl
from ..view.file_selection import Ui_Form as FileSelectionWidget
from sarcasm.type_utils import TypeUtils
import platform
import os
import subprocess

from sarcasm.ioutils import IOUtils


class FileSelectionControl:
    """
    The file selection control handles the file selection gui module and its functionality.
    """

    def __init__(self, file_selection_widget: FileSelectionWidget, main_control: ApplicationControl):
        self.__file_selection_widget = file_selection_widget
        self.__main_control = main_control

    @property
    def __cell(self):
        return self.__main_control.model.cell

    def bind_events(self):
        self.__file_selection_widget.btn_search.clicked.connect(self.on_search)
        self.__file_selection_widget.btn_set_to_default.clicked.connect(self.on_set_to_default)
        self.__file_selection_widget.btn_open_folder.clicked.connect(self.on_open_cell_folder)
        self.__file_selection_widget.btn_store_metadata.clicked.connect(self.on_store_meta_data)
        self.__file_selection_widget.le_cell_file.returnPressed.connect(self.on_return_pressed_cell_file)

        # call the method on editFinished and returnPressed
        self.__file_selection_widget.le_pixel_size.editingFinished.connect(self.on_return_pressed_framerate_pixelsize)
        self.__file_selection_widget.le_frame_time.editingFinished.connect(self.on_return_pressed_framerate_pixelsize)
        self.__file_selection_widget.le_frame_time.returnPressed.connect(self.on_return_pressed_framerate_pixelsize)
        self.__file_selection_widget.le_pixel_size.returnPressed.connect(self.on_return_pressed_framerate_pixelsize)

        self.__file_selection_widget.btn_search_parameters_file.clicked.connect(self.on_search_parameters_file)
        self.__file_selection_widget.btn_import_parameters.clicked.connect(self.on_btn_import_parameters)
        self.__file_selection_widget.btn_export_parameters.clicked.connect(self.on_btn_export_parameters)
        pass

    def on_set_to_default(self):
        # set all parameters back to default values
        self.__main_control.model.set_to_default()
        pass

    def on_search_parameters_file(self):
        f_name = QFileDialog.getOpenFileName(caption='Open Parameters File', filter="Json File (*.json *.JSON)")
        if f_name is not None:
            self.__file_selection_widget.le_parameters_file.setText(f_name[0])
        pass

    def on_btn_import_parameters(self):
        # todo: import parameters from file, re-bind parameters to ui
        if self.__file_selection_widget.le_parameters_file.text() != '':
            self.__main_control.model.parameters.load(self.__file_selection_widget.le_parameters_file.text())
            self.__main_control.debug('Parameters imported')
        pass

    def on_btn_export_parameters(self):
        if self.__file_selection_widget.le_parameters_file.text() != '':
            self.__main_control.model.parameters.store(self.__file_selection_widget.le_parameters_file.text())
            self.__main_control.debug(
                'Parameters exported to:' + self.__file_selection_widget.le_parameters_file.text())
            pass
        pass

    def on_search(self):
        # f_name is a tuple
        f_name = QFileDialog.getOpenFileName(caption='Open Cell file', filter="Tiff Images (*.tif *.tiff)")
        if f_name is not None:
            self.__file_selection_widget.le_cell_file.setText(f_name[0])
            self.on_return_pressed_cell_file()

    def on_return_pressed_cell_file(self, event=None):
        if len(self.__file_selection_widget.le_cell_file.text()) == 0:
            self.__main_control.debug('Empty File-Path')
            return
        if not os.path.exists(self.__file_selection_widget.le_cell_file.text()):
            self.__main_control.debug("The file doesn't exist")
            return
        self._init_file(self.__file_selection_widget.le_cell_file.text())

    def on_return_pressed_framerate_pixelsize(self, event=None):
        if self.__main_control.model.cell is None:
            return
        pixel_size = self.__file_selection_widget.le_pixel_size.text()
        if pixel_size is not None and pixel_size != '':
            try:
                d_pixel_size = float(pixel_size)
                if d_pixel_size != 0 and d_pixel_size is not None:
                    self.__main_control.model.cell.metadata['resxy'] = d_pixel_size
                    self.__file_selection_widget.le_pixel_size.setStyleSheet("")  # reset style (red background)
            except ValueError:
                self.__main_control.debug('the value in pixel size is not a number')

        frame_rate = self.__file_selection_widget.le_frame_time.text()
        if frame_rate is not None and frame_rate != '':
            try:
                d_frame_rate = float(frame_rate)
                if d_frame_rate != 0 and d_frame_rate is not None:
                    self.__main_control.model.cell.metadata['frametime'] = d_frame_rate
                    self.__file_selection_widget.le_frame_time.setStyleSheet("")  # QLineEdit{background : lightgreen;}
            except ValueError:
                self.__main_control.debug('the value in frame rate is not a number')

    def _init_file(self, file):
        # on file changed, clean up old files, napari viewer, models, etc.
        # todo: maybe switch to threaded execution (run_async_new)

        if self.__main_control.model.currentlyProcessing.get_value():
            self.__main_control.debug('still processing something')
            return

        self.__main_control.clean_up_on_new_image()
        self.__main_control.model.currentlyProcessing.set_value(True)
        self.__main_control.update_progress(10)

        image = tifffile.imread(file)
        lower_perc, upper_perc = np.percentile(image, q=[0.1, 99.9])
        self.__main_control.viewer.add_image(image, name='ImageData', contrast_limits=[lower_perc, upper_perc])

        self.__main_control.model.init_cell(file)
        self.__main_control.init_z_band_stack(fastmovie=True)
        self.__main_control.init_m_band_stack(visible=False)
        self.__main_control.init_z_lateral_connections(visible=False)
        self.__main_control.init_cell_mask_stack()
        self.__main_control.init_sarcomere_mask_stack()
        self.__main_control.init_sarcomere_vector_stack()
        self.__main_control.init_myofibril_lines_stack(visible=False)
        self.__main_control.init_sarcomere_domain_stack(visible=False)
        self.__main_control.viewer.dims.set_current_step(0, 0)

        self.init_line_layer()  # initializes the layer for drawing loi's

        # init or update dictionary
        cell: SarcAsM = TypeUtils.unbox(self.__main_control.model.cell)

        if cell.filepath not in self.__main_control.model.line_dictionary:
            self.__main_control.model.line_dictionary[cell.filepath] = {}
            pass

        self._init_meta_data()
        self._init_loi_from_file()
        self.__main_control.debug('Initialized: ' + file)
        self.__main_control.update_progress(100)
        self.__main_control.model.currentlyProcessing.set_value(False)

    def init_line_layer(self):
        if self.__main_control.viewer.layers.__contains__('LOIs'):
            layer = self.__main_control.viewer.layers.__getitem__('LOIs')
            self.__main_control.viewer.layers.remove(layer)
        # set the pre-selected color to red
        self.__main_control.init_loi_layer(self.__main_control.viewer.add_shapes(name='LOIs', edge_color='#FF0000'))

        pass

    def on_open_cell_folder(self):
        if len(self.__file_selection_widget.le_cell_file.text()) == 0:
            self.__main_control.debug('Empty File-Path')
            return
        if not os.path.exists(self.__file_selection_widget.le_cell_file.text()):
            self.__main_control.debug("The path doesn't exist")
            return

        cell = TypeUtils.unbox(self.__main_control.model.cell)
        str_path = cell.base_dir

        if platform.system() == 'Windows':
            os.startfile(str_path)
        elif platform.system() == 'Linux':
            subprocess.Popen(["xdg-open", str_path])
        elif platform.system() == 'Darwin':  # mac device
            subprocess.Popen(["open", str_path])

    def on_store_meta_data(self):
        # get values from entries and check if float
        def isfloat(num):
            try:
                float(num)
                return True
            except ValueError:
                return False

        cell = TypeUtils.unbox(self.__main_control.model.cell)
        if isfloat(self.__file_selection_widget.le_pixel_size.text()):
            cell.metadata['pixelsize'] = float(
                self.__file_selection_widget.le_pixel_size.text())
        if isfloat(self.__file_selection_widget.le_frame_time.text()):
            cell.metadata['frametime'] = float(
                self.__file_selection_widget.le_frame_time.text())
        cell.meta_data_handler.store_meta_data(True)  # store meta-data and override if necessary
        cell.meta_data_handler.commit()

    def _init_meta_data(self):
        # set metadata with cut off comma's
        cell = TypeUtils.unbox(self.__main_control.model.cell)

        if 'pixelsize' in cell.metadata and cell.metadata['pixelsize'] is not None:
            pixel_size = cell.metadata['pixelsize']
            pixel_size *= 10000
            pixel_size = int(pixel_size)
            pixel_size = float(pixel_size) / 10000
            self.__file_selection_widget.le_pixel_size.setText(str(pixel_size))
        else:
            self.__file_selection_widget.le_pixel_size.setPlaceholderText('- enter metadata manually -')
            self.__file_selection_widget.le_pixel_size.setStyleSheet("QLineEdit{background : red;}")

        if 'frametime' in cell.metadata and cell.metadata['frametime'] is not None:
            frame_rate = cell.metadata['frametime']
            frame_rate *= 10000
            frame_rate = int(frame_rate)
            frame_rate = float(frame_rate) / 10000
            self.__file_selection_widget.le_frame_time.setText(str(frame_rate))
        else:
            # no need for marking frame time
            self.__file_selection_widget.le_frame_time.setPlaceholderText('- enter metadata manually -')
            # self.__file_selection_widget.le_frame_time.setStyleSheet("QLineEdit{background : red;}")
        pass

    def _init_loi_from_file(self):
        # read loi files, store the line data in dictionary and in ui loi list
        self.__main_control.init_lois()
        pass
