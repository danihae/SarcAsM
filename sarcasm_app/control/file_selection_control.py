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


import json
import os
import platform
import subprocess

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from sarcasm import SarcAsM
from sarcasm.type_utils import TypeUtils
from .application_control import ApplicationControl
from ..view.file_selection import Ui_Form as FileSelectionWidget


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
        self.__file_selection_widget.le_pixel_size.editingFinished.connect(self.on_return_pressed_pixel_size_frame_rate)
        self.__file_selection_widget.le_frame_time.editingFinished.connect(self.on_return_pressed_pixel_size_frame_rate)
        self.__file_selection_widget.le_frame_time.returnPressed.connect(self.on_return_pressed_pixel_size_frame_rate)
        self.__file_selection_widget.le_pixel_size.returnPressed.connect(self.on_return_pressed_pixel_size_frame_rate)

        self.__file_selection_widget.spinbox_channel.valueChanged.connect(self.on_changed_channel)

        self.__file_selection_widget.btn_search_parameters_file.clicked.connect(self.on_search_parameters_file)
        self.__file_selection_widget.btn_import_parameters.clicked.connect(self.on_btn_import_parameters)
        self.__file_selection_widget.btn_export_parameters.clicked.connect(self.on_btn_export_parameters)
        pass

    def on_changed_channel(self):
        self.__main_control.model.cell.metadata.channel = int(self.__file_selection_widget.spinbox_channel.value())
        self.__main_control.init_image_stack()


    def on_set_to_default(self):
        # set all parameters back to default values
        self.__main_control.model.set_to_default()
        pass

    def on_search_parameters_file(self):
        """Handle parameter file selection with JSON validation"""
        dialog = QFileDialog()
        dialog.setWindowTitle("Select Parameter File")
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)  # Shows "Open" button
        dialog.setNameFilter("JSON Files (*.json)")
        dialog.setDefaultSuffix("json")
        dialog.setOption(QFileDialog.DontUseNativeDialog)

        # Configure for both existing and new files
        if dialog.exec_():
            selected_files = dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]

                # Ensure .json extension
                if not file_path.lower().endswith('.json'):
                    file_path += '.json'

                # Create file if it doesn't exist
                if not os.path.exists(file_path):
                    try:
                        with open(file_path, 'w') as f:
                            f.write('{}')  # Create valid empty JSON
                    except Exception as e:
                        QMessageBox.critical(None, "Error",
                                             f"Could not create file:\n{str(e)}")
                        return

                # Validate JSON structure
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)  # Verify JSON is parseable
                    self.__file_selection_widget.le_parameters_path.setText(file_path)
                except json.JSONDecodeError:
                    QMessageBox.warning(None, "Invalid JSON",
                                        "The selected file contains invalid JSON format")
                except Exception as e:
                    QMessageBox.critical(None, "Error",
                                         f"Failed to read file:\n{str(e)}")

    def on_btn_import_parameters(self):
        if self.__file_selection_widget.le_parameters_path.text() != '':
            file_path = self.__file_selection_widget.le_parameters_path.text()

            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.__main_control.model.parameters.load(file_path)
                self.__main_control.debug('Parameters imported')
            else:
                self.__main_control.debug('Parameters not imported, file does not exist')
        pass

    def on_btn_export_parameters(self):
        if self.__file_selection_widget.le_parameters_path.text() != '':
            file_path = self.__file_selection_widget.le_parameters_path.text()
            if not os.path.isdir(file_path):
                self.__main_control.model.parameters.store(file_path)
                self.__main_control.debug('Parameters exported to:' + file_path)
            else:
                self.__main_control.debug('Parameters NOT exported.')
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

    def on_return_pressed_pixel_size_frame_rate(self, event=None):
        if self.__main_control.model.cell is None:
            return
        pixel_size = self.__file_selection_widget.le_pixel_size.text()
        if pixel_size is not None and pixel_size != '':
            try:
                d_pixel_size = float(pixel_size)
                if d_pixel_size != 0 and d_pixel_size is not None:
                    self.__main_control.model.cell.metadata.pixelsize = d_pixel_size
                    self.__file_selection_widget.le_pixel_size.setStyleSheet("")  # reset style (red background)
            except ValueError:
                self.__main_control.debug('the value in pixel size is not a number')

        frame_rate = self.__file_selection_widget.le_frame_time.text()
        if frame_rate is not None and frame_rate != '':
            try:
                d_frame_rate = float(frame_rate)
                if d_frame_rate != 0 and d_frame_rate is not None:
                    self.__main_control.model.cell.metadata.frametime = d_frame_rate
                    self.__file_selection_widget.le_frame_time.setStyleSheet("")  # QLineEdit{background : lightgreen;}
            except ValueError:
                self.__main_control.debug('the value in frame rate is not a number')

        self.__main_control.init_scale_bar()

    def _init_file(self, file):
        # on file changed, clean up old files, napari viewer, models, etc.
        # todo: maybe switch to threaded execution (run_async_new)

        if self.__main_control.model.currentlyProcessing.get_value():
            self.__main_control.debug('still processing something')
            return

        self.__main_control.clean_up_on_new_image()
        self.__main_control.model.currentlyProcessing.set_value(True)
        self.__main_control.update_progress(10)

        self.__main_control.model.init_cell(file)
        self.__main_control.init_scale_bar()
        self.__main_control.init_image_stack()
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
        _scale = self.__main_control.model.cell.scale[-2:]
        self.__main_control.init_loi_layer(self.__main_control.viewer.add_shapes(name='LOIs', edge_color='#FF0000', scale=_scale))

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
            cell.metadata.pixelsize = float(
                self.__file_selection_widget.le_pixel_size.text())
        if isfloat(self.__file_selection_widget.le_frame_time.text()):
            cell.metadata.frametime = float(
                self.__file_selection_widget.le_frame_time.text())

        axes = self.__file_selection_widget.le_axes.text().upper()
        # check if all letters are either X, Y, T, C or Z and that not one letter appears more than once
        def validate_letters_warn(seq: str) -> bool:
            """Return True if `seq` is a unique subset of {X,Y,T,C,Z}; else print warnings."""
            allowed = set("XYTCZ")
            invalid = set(seq) - allowed
            dup = {c for c in seq if seq.count(c) > 1}
            if invalid:
                self.__main_control.debug(f"Warning: invalid character(s): {''.join(sorted(invalid))}")
            if dup:
                self.__main_control.debug(f"Warning: duplicate character(s): {''.join(sorted(dup))}")
            return not (invalid or dup)

        axes_valid = validate_letters_warn(axes)
        if axes_valid:
            self.__main_control.model.cell.metadata.axes = axes
            self.__main_control.init_image_stack()
        else:
            self.__file_selection_widget.le_axes.setStyleSheet("QLineEdit{background : red;}")

        cell.save_metadata()

    def _init_meta_data(self):
        # set metadata
        cell = TypeUtils.unbox(self.__main_control.model.cell)

        # pixel size
        if cell.metadata.pixelsize is not None:
            pixel_size = cell.metadata.pixelsize
            self.__file_selection_widget.le_pixel_size.setText(str(round(pixel_size, 5)))
            if not 0.5 >= pixel_size >= 0.01:
                self.__main_control.debug(f"Warning: Pixel size of {round(pixel_size, 5)} µm not in reasonable range "
                                          f"between 0.01–0.5 µm. Please enter correct pixel size. ")
                self.__file_selection_widget.le_pixel_size.setStyleSheet("QLineEdit{background : red;}")

        else:
            self.__file_selection_widget.le_pixel_size.setPlaceholderText('- enter metadata manually -')
            self.__file_selection_widget.le_pixel_size.setStyleSheet("QLineEdit{background : red;}")

        # frame time
        if cell.metadata.frametime is not None:
            frame_rate = cell.metadata.frametime
            self.__file_selection_widget.le_frame_time.setText(str(round(frame_rate, 5)))
        else:
            # no need for marking frame time
            self.__file_selection_widget.le_frame_time.setPlaceholderText('- enter metadata manually -')

        # axes
        axes = cell.metadata.axes
        self.__file_selection_widget.le_axes.setText(str(axes))

        # channel
        if cell.metadata.channel is not None:
            self.__file_selection_widget.spinbox_channel.setDisabled(False)
            self.__file_selection_widget.spinbox_channel.setValue(cell.metadata.channel)
            self.__file_selection_widget.spinbox_channel.setMinimum(0)
            self.__file_selection_widget.spinbox_channel.setMaximum(cell.metadata.shape_orig[-1] - 1)

    def _init_loi_from_file(self):
        # read loi files, store the line data in dictionary and in ui loi list
        self.__main_control.init_lois()
        pass
