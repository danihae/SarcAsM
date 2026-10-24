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

import numpy as np

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from sarcasm.type_utils import TypeUtils
from .application_control import ApplicationControl
from ..view.file_selection import Ui_Form as FileSelectionWidget

logger = logging.getLogger(__name__)

# Input validation error styling — soft tint + border so text stays legible in
# both light and dark palettes.
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
        if self.__main_control.model.cell is None:
            return
        self.apply_metadata(channel=int(self.__file_selection_widget.spinbox_channel.value()))


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
        values = {}
        for name, field, label in (('pixelsize', self.__file_selection_widget.le_pixel_size, 'pixel size'),
                                   ('frametime', self.__file_selection_widget.le_frame_time, 'frame time')):
            text = field.text().strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                logger.warning(f'The {label} "{text}" is not a number.')
                field.setStyleSheet(_INVALID_INPUT_QSS)
                continue
            if value <= 0:
                logger.warning(f'The {label} must be positive.')
                field.setStyleSheet(_INVALID_INPUT_QSS)
                continue
            field.setStyleSheet("")
            values[name] = value
        if values:
            self.apply_metadata(**values)

    #: decimals of pixel size / frame time in the Metadata fields (see _init_meta_data)
    _FIELD_DECIMALS = 5

    @classmethod
    def _differs(cls, new, old) -> bool:
        """Whether a typed value changes the stored one. Numbers are compared as the fields show
        them, so pressing Enter on an unchanged (rounded) field is not a change."""
        if new is None:
            return False
        if isinstance(new, float) and old is not None:
            return round(new, cls._FIELD_DECIMALS) != round(float(old), cls._FIELD_DECIMALS)
        return new != old

    def apply_metadata(self, pixelsize=None, frametime=None, axes=None, channel=None) -> bool:
        """Apply calibration / layout typed in the Metadata row to the open file.

        Axes or channel change how the pixels are read, so the file is opened again from
        scratch. A pixel size (or, with motion results, a frame time) that differs from
        the one the results were computed with needs a re-analysis: the user confirms that
        the results are deleted, or the fields go back to the stored values. A frame time
        the file did not have is simply filled in. Returns True if anything changed.
        """
        cell = self.__main_control.model.cell
        if cell is None:
            return False
        md = cell.metadata
        if axes is not None and 'C' not in axes:
            channel = None                       # a channel only exists with a C axis
        layout = self._differs(axes, md.axes) or (channel is not None and self._differs(channel, md.channel))
        px_changed = self._differs(pixelsize, md.pixelsize)
        ft_changed = self._differs(frametime, md.frametime)
        if not (layout or px_changed or ft_changed):
            return False

        keys = list(cell.data.keys())
        has_results = bool(keys) or bool(cell.store.exists and cell.store.mask_names())
        has_motion = any(k.startswith('motion.') for k in keys)
        needs_restart = layout or (px_changed and has_results) or (ft_changed and md.frametime is not None
                                                                    and has_motion)
        if not needs_restart:
            if px_changed:
                md.pixelsize = pixelsize
            if ft_changed:
                md.frametime = frametime
                if md.n_stack and md.n_stack > 1:
                    md.time = np.arange(md.n_stack) * frametime
            cell.save_metadata()
            self.__main_control.init_scale_bar()
            logger.info('Metadata saved.')
            return True

        what = ', '.join(n for n, c in (('axes/channel', layout), ('pixel size', px_changed),
                                         ('frame time', ft_changed)) if c)
        if has_results:
            answer = QMessageBox.question(
                self.__file_selection_widget.btn_store_metadata, 'Re-analyse with new metadata?',
                f'Changing the {what} of an analysed file invalidates its results, which were computed '
                f'with the old values.\n\nDelete the results of this file and open it again with the new '
                f'values? (Then run the analysis again.)',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                self._init_meta_data()           # put the stored values back into the fields
                return False
        new_axes = axes if axes is not None else md.axes
        # unchanged values keep their full precision (the fields show them rounded)
        kwargs = dict(restart=True,
                      pixelsize=pixelsize if px_changed else md.pixelsize,
                      frametime=frametime if ft_changed else md.frametime,
                      axes=new_axes)
        if 'C' in (new_axes or ''):
            kwargs['channel'] = channel if channel is not None else md.channel
        self._init_file(cell.file_path, **kwargs)
        return True

    def _init_file(self, file, **cell_kwargs):
        # on file changed, clean up old files, napari viewer, models, etc.
        # cell_kwargs (restart, calibration, axes, channel) are passed to SarcAsM

        if self.__main_control.model.currentlyProcessing.get_value():
            logger.warning('Please wait: another step is still running.')
            return

        self.__main_control.clean_up_on_new_image()
        self.__main_control.model.currentlyProcessing.set_value(True)
        self.__main_control.update_progress(10)

        try:
            self.__main_control.model.init_cell(file, **cell_kwargs)
        except Exception as e:
            logger.error(f'Could not open {file}: {e}')
            QMessageBox.warning(self.__file_selection_widget.btn_search, 'Could not open the file',
                                f'{e}\n\nCheck the axes order (e.g. TYX for a movie, YX for an image) and '
                                f'the channel in the Metadata row.')
            self.__main_control.update_progress(0)
            self.__main_control.model.currentlyProcessing.set_value(False)
            return
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

        self.__main_control.init_sarcomere_dots(visible=False)
        self.__main_control.init_track_groups_layer(visible=False)
        # like after an analysis step: the image and the most advanced result, the rest hidden
        # (with motion results the track layers, toggled in the Motion tab, are that result)
        layers = self.__main_control.viewer.layers
        if 'Sarcomeres' in layers or 'Groups' in layers:
            self.__main_control.show_only_results()
        else:
            latest = next((name for name in ('SarcomereDomains', 'MyofibrilLines', 'SarcomereVectors', 'ZbandMask')
                           if name in layers), None)
            self.__main_control.show_only_results(*((latest, 'MidlinePoints') if latest == 'SarcomereVectors'
                                                     else (latest,) if latest else ()))
        for callback in self.__main_control.file_opened_callbacks:
            callback()

        self._init_meta_data()
        self.__main_control.set_viewer_title(file)
        self.__main_control.raise_viewer()
        logger.info(f'Initialized: {file}')
        self.__main_control.update_progress(100)
        self.__main_control.model.currentlyProcessing.set_value(False)


    def on_store_meta_data(self):
        # get values from entries and check if float
        def isfloat(num):
            try:
                float(num)
                return True
            except ValueError:
                return False

        if self.__main_control.model.cell is None:
            return
        pixelsize = (float(self.__file_selection_widget.le_pixel_size.text())
                     if isfloat(self.__file_selection_widget.le_pixel_size.text()) else None)
        frametime = (float(self.__file_selection_widget.le_frame_time.text())
                     if isfloat(self.__file_selection_widget.le_frame_time.text()) else None)

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

        if not validate_letters_warn(axes) or not axes:
            self.__file_selection_widget.le_axes.setStyleSheet(_INVALID_INPUT_QSS)
            return
        self.__file_selection_widget.le_axes.setStyleSheet("")
        channel = (int(self.__file_selection_widget.spinbox_channel.value())
                   if self.__file_selection_widget.spinbox_channel.isEnabled() else None)
        self.apply_metadata(pixelsize=pixelsize, frametime=frametime, axes=axes, channel=channel)

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
        spin = self.__file_selection_widget.spinbox_channel
        spin.blockSignals(True)                  # filling the field is not a user change
        try:
            if cell.metadata.channel is not None and axes and 'C' in axes:
                spin.setDisabled(False)
                spin.setMinimum(0)
                spin.setMaximum(int(cell.metadata.shape_orig[axes.index('C')]) - 1)
                spin.setValue(cell.metadata.channel)
            else:
                spin.setValue(0)
                spin.setDisabled(True)
        finally:
            spin.blockSignals(False)

