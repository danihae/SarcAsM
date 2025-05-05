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


import sys
from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QPalette, QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QStyleFactory, QAbstractSpinBox
from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QVBoxLayout, QToolBox, QScrollArea, QProgressBar, QTextEdit

from .control.application_control import ApplicationControl
from .control.file_selection_control import FileSelectionControl
from .control.motion_analysis_control import MotionAnalysisControl
from .control.loi_analysis_control import LOIAnalysisControl
from .control.structure_analysis_control import StructureAnalysisControl
from .control.batch_processing_control import BatchProcessingControl
from .model import ApplicationModel
from .view.file_selection import Ui_Form as FileSelectionWidget
from .view.parameter_structure_analysis import Ui_Form as StructureAnalysisWidget
from .view.parameter_loi_analysis import Ui_Form as LoiAnalysisWidget
from .view.parameter_motion_analysis import Ui_Form as MotionAnalysisWidget
from .view.parameters_batch_processing import Ui_Form as BatchProcessingWidget

from sarcasm import __version__ as version


class Application:

    def __init__(self):
        self.__app = QApplication([])
        QLocale.setDefault(QLocale(QLocale.English,QLocale.UnitedStates))
        self.__app.setStyle(QStyleFactory.create("Fusion"))  # try to fix the layout issue on macOs
        self.__app.setWindowIcon(QIcon("../docs/images/sarcasm.ico"))

        # one of the "fast solutions" without dependencies on stackoverflow page above
        # Now use a palette to switch to dark colors:
        self.__palette = QPalette()
        self.__palette.setColor(QPalette.Window, QColor(53, 53, 53))
        self.__palette.setColor(QPalette.WindowText, Qt.white)
        self.__palette.setColor(QPalette.Base, QColor(25, 25, 25))
        self.__palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.__palette.setColor(QPalette.ToolTipBase, Qt.black)
        self.__palette.setColor(QPalette.ToolTipText, Qt.white)
        self.__palette.setColor(QPalette.Text, Qt.white)
        self.__palette.setColor(QPalette.Button, QColor(53, 53, 53))
        self.__palette.setColor(QPalette.ButtonText, Qt.white)
        self.__palette.setColor(QPalette.BrightText, Qt.red)
        self.__palette.setColor(QPalette.Link, QColor(42, 130, 218))
        self.__palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.__palette.setColor(QPalette.HighlightedText, Qt.black)
        self.__app.setPalette(self.__palette)

        self.__window = QWidget()

        self.__file_selection = FileSelectionWidget()
        self.__structure_analysis_parameters = StructureAnalysisWidget()
        self.__loi_analysis = LoiAnalysisWidget()
        self.__motion_analysis = MotionAnalysisWidget()
        self.__batch_processing = BatchProcessingWidget()
        self.__progress_bar = QProgressBar()
        self.__text_debug = QTextEdit()
        self.__text_debug.setObjectName("messageArea")
        self.__label_gpu = QLabel("GPU")
        self.__label_busy = QLabel("IDLE")
        self.__status_bar = QWidget()
        self.__control = ApplicationControl(self.__window, ApplicationModel())
        self.__file_selection_control = FileSelectionControl(self.__file_selection, self.__control)
        self.__structure_analysis_control = StructureAnalysisControl(self.__structure_analysis_parameters,
                                                                     self.__control)
        self.__loi_analysis_control = LOIAnalysisControl(self.__loi_analysis, self.__control)
        self.__motion_analysis_control = MotionAnalysisControl(self.__motion_analysis, self.__control)
        self.__batch_processing_control = BatchProcessingControl(self.__batch_processing, self.__control)

    def __disable_scroll_on_spinbox(self):
        opts = Qt.FindChildrenRecursively
        spinboxes = self.__window.findChildren(QAbstractSpinBox, options=opts)
        for box in spinboxes:
            box.wheelEvent = lambda *event: None

    def __center_ui(self):
        qt_rectangle = self.__window.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.__window.move(qt_rectangle.topLeft())

    def __get_parameter_scroll_box(self):
        widget_parameter_scrollbox = QScrollArea()
        scroll_area_layout = QVBoxLayout()

        widget_parameter_toolbox = QToolBox()

        widget_structure_parameters = QWidget()
        self.__structure_analysis_parameters.setupUi(widget_structure_parameters)

        widget_loi_analysis = QWidget()
        self.__loi_analysis.setupUi(widget_loi_analysis)

        widget_motion_analysis = QWidget()
        self.__motion_analysis.setupUi(widget_motion_analysis)

        widget_batch_processing = QWidget()
        self.__batch_processing.setupUi(widget_batch_processing)

        widget_parameter_toolbox.addItem(widget_structure_parameters, 'Structure Analysis')
        widget_parameter_toolbox.addItem(widget_loi_analysis, 'LOI Finder')
        widget_parameter_toolbox.addItem(widget_motion_analysis, 'Motion Analysis')
        widget_parameter_toolbox.addItem(widget_batch_processing, 'Batch Processing')

        scroll_area_layout.addWidget(widget_parameter_toolbox)
        widget_parameter_scrollbox.setLayout(scroll_area_layout)
        return widget_parameter_scrollbox

    def __bind_events(self):
        self.__control.set_debug_action(self.debug)
        self.__file_selection_control.bind_events()
        self.__structure_analysis_control.bind_events()
        self.__loi_analysis_control.bind_events()
        self.__motion_analysis_control.bind_events()
        self.__batch_processing_control.bind_events()

        print()  # in this method the binding to gui buttons should be handled

    def debug(self, message):
        self.__text_debug.append(message)

    def __update_busy_label(self, new_value):
        if new_value:
            # currently processing is true -> busy
            self.__label_busy.setText('BUSY')
            self.__label_busy.setStyleSheet('background-color:rgba(255,0,0,0.5);')
            pass
        else:
            self.__label_busy.setText('IDLE')
            self.__label_busy.setStyleSheet('background-color:rgba(0,255,0,0.3);')
            pass

        pass

    def __init_status_bar(self):
        h_box = QHBoxLayout()
        h_box.addWidget(self.__label_gpu, 0)
        h_box.addWidget(self.__label_busy, 0)
        h_box.addWidget(self.__progress_bar, 1)
        self.__status_bar.setLayout(h_box)

        # init busy label with IDLE&green bg
        self.__label_busy.setStyleSheet('color:rgba(255,255,255,0.9);')
        self.__label_busy.setStyleSheet('background-color:rgba(0,255,0,0.3);')

        self.__label_gpu.setStyleSheet('color:rgba(255,255,255,0.9);')
        if self.__control.is_gpu_available():
            self.__label_gpu.setStyleSheet('background-color:rgba(0,255,0,0.3);')
            pass
        else:
            self.__label_gpu.setStyleSheet('background-color:rgba(255,0,0,0.5);')
            pass

        # in case of idle -> the label should display IDLE with green background
        # in case of busy -> the label should display BUSY with red background
        self.__control.model.currentlyProcessing.connect(
            ui_element=lambda new_value: self.__update_busy_label(new_value))
        pass

    def init_gui(self):
        self.__window.setWindowTitle(f'SarcAsM - v{version}')
        self.__window.setGeometry(0, 0, 800, 1000)
        self.__center_ui()

        main_layout = QVBoxLayout()

        self.__progress_bar.setObjectName("progressBarMain")

        widget_file_selection = QWidget()
        self.__file_selection.setupUi(widget_file_selection)

        main_layout.addWidget(widget_file_selection, 1)

        widget_center = QWidget()
        center_layout = QHBoxLayout()

        center_layout.addWidget(self.__get_parameter_scroll_box(), 3)
        # center_layout.addWidget(widget_main_content,7)
        widget_center.setLayout(center_layout)

        main_layout.addWidget(widget_center, 9)

        self.__text_debug.setReadOnly(True)

        main_layout.addWidget(self.__text_debug, 2)
        self.__init_status_bar()
        main_layout.addWidget(self.__status_bar, 0)
        self.__bind_events()

        self.__window.setLayout(main_layout)
        self.__disable_scroll_on_spinbox()
        self.__window.show()

        sys.exit(self.__app.exec_())
