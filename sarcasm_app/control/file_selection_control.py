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
import logging
import os
import platform
import subprocess

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from sarcasm.type_utils import TypeUtils
from .application_control import ApplicationControl
from ..view.file_selection import Ui_Form as FileSelectionWidget

logger = logging.getLogger(__name__)

# Input validation error styling — soft tint + border so text stays legible in
# both light and dark palettes (old `background: red` made text unreadable).
_INVALID_INPUT_QSS = (
    "QLineEdit { border: 1px solid #e06060; "
    "background-color: rgba(224,96,96,0.18); }"
)


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
        # Let files dropped on the embedded napari viewer go through the same
        # import flow as the buttons (full analysis layers + metadata panel).
        self.__main_control.set_open_file_handler(self.open_file)

        self.__file_selection_widget.btn_search.clicked.connect(self.on_search)
        self.__file_selection_widget.btn_open_zarr.clicked.connect(self.on_open_zarr)
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
                        QMessageBox.critical(self.__file_selection_widget.btn_search_parameters_file, "Error",
                                             f"Could not create file:\n{str(e)}")
                        return

                # Validate JSON structure
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)  # Verify JSON is parseable
                    self.__file_selection_widget.le_parameters_path.setText(file_path)
                except json.JSONDecodeError:
                    QMessageBox.warning(self.__file_selection_widget.btn_search_parameters_file,
                                        "Invalid JSON",
                                        "The selected file contains invalid JSON format")
                except Exception as e:
                    QMessageBox.critical(self.__file_selection_widget.btn_search_parameters_file,
                                         "Error",
                                         f"Failed to read file:\n{str(e)}")

    def on_btn_import_parameters(self):
        if self.__file_selection_widget.le_parameters_path.text() != '':
            file_path = self.__file_selection_widget.le_parameters_path.text()

            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.__main_control.model.parameters.load(file_path)
                logger.info('Parameters imported')
            else:
                logger.warning('Parameters not imported, file does not exist')
        pass

    def on_btn_export_parameters(self):
        if self.__file_selection_widget.le_parameters_path.text() != '':
            file_path = self.__file_selection_widget.le_parameters_path.text()
            if not os.path.isdir(file_path):
                self.__main_control.model.parameters.store(file_path)
                logger.info(f'Parameters exported to: {file_path}')
            else:
                logger.warning('Parameters NOT exported.')
            pass
        pass

    def on_search(self):
        # f_name is a tuple
        f_name = QFileDialog.getOpenFileName(caption='Open .tif movie', filter="Tiff Images (*.tif *.tiff)")
        if f_name is not None and f_name[0]:
            self.__file_selection_widget.le_cell_file.setText(f_name[0])
            self.on_return_pressed_cell_file()

    def on_open_zarr(self):
        # .ome.zarr is a directory, so select it with a directory picker
        path = QFileDialog.getExistingDirectory(caption='Open .ome.zarr store')
        if not path:
            return
        if not path.endswith('.zarr') and not os.path.exists(os.path.join(path, 'zarr.json')):
            logger.warning(f"'{path}' does not look like a .ome.zarr store; trying anyway.")
        self.open_file(path)

    def open_file(self, path):
        """Load *path* via the standard import flow, updating the file field.

        Shared entry point for the buttons and for files dropped on the embedded
        napari viewer. A trailing separator (Qt hands directory drops back with
        one) is stripped so the path round-trips through the importer cleanly.
        """
        path = path.rstrip('/\\') if path else path
        self.__file_selection_widget.le_cell_file.setText(path)
        self.on_return_pressed_cell_file()

    def on_return_pressed_cell_file(self, event=None):
        if len(self.__file_selection_widget.le_cell_file.text()) == 0:
            logger.warning('Empty File-Path')
            return
        if not os.path.exists(self.__file_selection_widget.le_cell_file.text()):
            logger.warning("The file doesn't exist")
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
                logger.warning('The value in pixel size is not a number')

        frame_rate = self.__file_selection_widget.le_frame_time.text()
        if frame_rate is not None and frame_rate != '':
            try:
                d_frame_rate = float(frame_rate)
                if d_frame_rate != 0 and d_frame_rate is not None:
                    self.__main_control.model.cell.metadata.frametime = d_frame_rate
                    self.__file_selection_widget.le_frame_time.setStyleSheet("")  # QLineEdit{background : lightgreen;}
            except ValueError:
                logger.warning('The value in frame rate is not a number')

        self.__main_control.init_scale_bar()

    def _init_file(self, file):
        # on file changed, clean up old files, napari viewer, models, etc.
        # todo: maybe switch to threaded execution (run_async_new)

        if self.__main_control.model.currentlyProcessing.get_value():
            logger.warning('Still processing something')
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

        self.__main_control.init_tracks_stack(visible=False)
        self.__main_control.init_track_groups_stack(visible=False)

        self._init_meta_data()
        self.__main_control.set_viewer_title(file)
        self.__main_control.raise_viewer()
        logger.info(f'Initialized: {file}')
        self.__main_control.update_progress(100)
        self.__main_control.model.currentlyProcessing.set_value(False)


    def on_open_cell_folder(self):
        if len(self.__file_selection_widget.le_cell_file.text()) == 0:
            logger.warning('Empty File-Path')
            return
        if not os.path.exists(self.__file_selection_widget.le_cell_file.text()):
            logger.warning("The path doesn't exist")
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
                logger.warning(f"Invalid character(s): {''.join(sorted(invalid))}")
            if dup:
                logger.warning(f"Duplicate character(s): {''.join(sorted(dup))}")
            return not (invalid or dup)

        axes_valid = validate_letters_warn(axes)
        if axes_valid:
            self.__main_control.model.cell.metadata.axes = axes
            self.__main_control.init_image_stack()
        else:
            self.__file_selection_widget.le_axes.setStyleSheet(_INVALID_INPUT_QSS)

        cell.save_metadata()

    def _init_meta_data(self):
        # set metadata
        cell = TypeUtils.unbox(self.__main_control.model.cell)

        # pixel size
        if cell.metadata.pixelsize is not None:
            pixel_size = cell.metadata.pixelsize
            self.__file_selection_widget.le_pixel_size.setText(str(round(pixel_size, 5)))
            if not 0.5 >= pixel_size >= 0.01:
                logger.warning(f"Pixel size of {round(pixel_size, 5)} µm not in reasonable range "
                               f"between 0.01–0.5 µm. Please enter correct pixel size.")
                self.__file_selection_widget.le_pixel_size.setStyleSheet(_INVALID_INPUT_QSS)

        else:
            self.__file_selection_widget.le_pixel_size.setPlaceholderText('- enter metadata manually -')
            self.__file_selection_widget.le_pixel_size.setStyleSheet(_INVALID_INPUT_QSS)

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

