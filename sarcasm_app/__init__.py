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


import logging
import sys
from pathlib import Path

import requests
from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QStyleFactory, QAbstractSpinBox,
                             QAction, QMenuBar, QMessageBox, QSplitter, QTabWidget)
from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QProgressBar, QTextEdit

_ICON_DIR = Path(__file__).parent / "icons"
# Prefer platform-native icon format at runtime (Windows reads .ico, macOS .icns,
# both fall back gracefully on Linux since QIcon loads whichever file exists).
if sys.platform == "darwin" and (_ICON_DIR / "sarcasm.icns").exists():
    _APP_ICON_PATH = str(_ICON_DIR / "sarcasm.icns")
else:
    _APP_ICON_PATH = str(_ICON_DIR / "sarcasm.ico")

# App-wide stylesheet. The accent rules highlight primary "run full analysis"
# buttons so users can spot the main actions at a glance. Buttons opt in via
# `setProperty("accent", True)` — see _ACCENT_BUTTONS below.
_APP_STYLESHEET = """
QPushButton[accent="true"] {
    background-color: #2a82da;
    color: white;
    border: 1px solid #1f5f9f;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton[accent="true"]:hover {
    background-color: #3d93e8;
    border-color: #2a82da;
}
QPushButton[accent="true"]:pressed {
    background-color: #1f6fbd;
}
QPushButton[accent="true"]:disabled {
    background-color: #3a4a5a;
    color: rgba(255,255,255,0.45);
    border-color: #2d3846;
}
"""


def _mark_accent(button):
    """Tag a button so the app stylesheet renders it with the accent color.
    Must be called after the button is added to a widget hierarchy, because
    Qt only re-evaluates the stylesheet on the next style polish."""
    button.setProperty("accent", True)
    button.style().unpolish(button)
    button.style().polish(button)

from .control.application_control import ApplicationControl
from .control.logging_handler import setup_gui_logging
from .control.file_selection_control import FileSelectionControl
from .control.motion_analysis_control import MotionAnalysisControl
from .control.loi_analysis_control import LOIAnalysisControl
from .control.structure_analysis_control import StructureAnalysisControl
from .control.batch_processing_control import BatchProcessingControl
from .model import ApplicationModel
from .view.file_selection import Ui_Form as FileSelectionWidget
from .view.parameters_structure_analysis import Ui_Form as StructureAnalysisWidget
from .view.parameters_loi_analysis import Ui_Form as LoiAnalysisWidget
from .view.parameters_motion_analysis import Ui_Form as MotionAnalysisWidget
from .view.parameters_batch_processing import Ui_Form as BatchProcessingWidget

from sarcasm import __version__ as version

# IMPORTANT: Qt attributes must be set BEFORE QApplication is created
# This fixes high-DPI scaling issues on Windows (Qt 5.6+)
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
# PassThrough keeps widget sizes consistent under fractional Windows scaling (125%, 150%)
try:
    from PyQt5.QtGui import QGuiApplication
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
except (AttributeError, ImportError):
    pass

class Application:

    def __init__(self):
        self.__app = QApplication([])
        QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))

        # Fusion + dark palette on every platform so the app always matches
        # napari's native dark look regardless of the user's OS theme.
        self.__app.setStyle(QStyleFactory.create("Fusion"))
        self.__app.setWindowIcon(QIcon(_APP_ICON_PATH))

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
        self.__app.setStyleSheet(_APP_STYLESHEET)

        self.__window = QWidget()

        self.__file_selection = FileSelectionWidget()
        self.__structure_analysis_parameters = StructureAnalysisWidget()
        self.__loi_analysis = LoiAnalysisWidget()
        self.__motion_analysis = MotionAnalysisWidget()
        self.__batch_processing = BatchProcessingWidget()
        self.__progress_bar = QProgressBar()
        self.__text_debug = QTextEdit()
        self.__text_debug.setObjectName("messageArea")
        
        # Setup GUI logging to display sarcasm package log messages in the message area
        self.__log_handler = setup_gui_logging(self.__text_debug, level=logging.INFO)
        
        # Unicode markers (● filled / ○ hollow / ⏳ hourglass) make status readable
        # without relying solely on red/green, which colorblind users can't distinguish.
        self.__label_gpu = QLabel("● GPU")
        self.__label_busy = QLabel("● IDLE")
        self.__status_bar = QWidget()
        self.__wait_cursor_active = False
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
        # QTabWidget (horizontal tabs) instead of QToolBox — reclaims ~100px of
        # vertical space that QToolBox's stacked section headers consumed.
        widget_parameter_tabs = QTabWidget()
        widget_parameter_tabs.setDocumentMode(True)
        widget_parameter_tabs.setUsesScrollButtons(True)

        widget_structure_parameters = QWidget()
        self.__structure_analysis_parameters.setupUi(widget_structure_parameters)
        _mark_accent(self.__structure_analysis_parameters.btn_analyze_structure)

        widget_loi_analysis = QWidget()
        self.__loi_analysis.setupUi(widget_loi_analysis)
        _mark_accent(self.__loi_analysis.btn_detect_lois)

        widget_motion_analysis = QWidget()
        self.__motion_analysis.setupUi(widget_motion_analysis)
        _mark_accent(self.__motion_analysis.btn_analyze_motion)

        widget_batch_processing = QWidget()
        self.__batch_processing.setupUi(widget_batch_processing)
        _mark_accent(self.__batch_processing.btn_batch_processing_structure)

        widget_parameter_tabs.addTab(widget_structure_parameters, 'Structure Analysis')
        widget_parameter_tabs.addTab(widget_loi_analysis, 'LOI Detection')
        widget_parameter_tabs.addTab(widget_motion_analysis, 'Motion Analysis')
        widget_parameter_tabs.addTab(widget_batch_processing, 'Batch Processing')

        # QScrollArea must own the content via setWidget (not setLayout), otherwise
        # it never scrolls and tall parameter forms get clipped on small screens.
        widget_parameter_scrollbox = QScrollArea()
        widget_parameter_scrollbox.setWidget(widget_parameter_tabs)
        widget_parameter_scrollbox.setWidgetResizable(True)
        widget_parameter_scrollbox.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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

    def __build_menu_bar(self):
        # QMenuBar on a QWidget layout works cross-platform; on macOS Qt routes
        # it automatically to the native menu bar at the top of the screen.
        menu_bar = QMenuBar(self.__window)

        file_menu = menu_bar.addMenu("&File")
        action_quit = QAction("&Quit", self.__window)
        action_quit.setShortcut("Ctrl+Q")  # Qt translates to Cmd+Q on macOS
        action_quit.setMenuRole(QAction.QuitRole)
        action_quit.triggered.connect(self.__app.quit)
        file_menu.addAction(action_quit)

        view_menu = menu_bar.addMenu("&View")
        self.__action_toggle_log = QAction("Show &Log Panel", self.__window)
        self.__action_toggle_log.setCheckable(True)
        self.__action_toggle_log.setChecked(True)
        self.__action_toggle_log.setShortcut("Ctrl+L")
        self.__action_toggle_log.triggered.connect(self.__toggle_log_panel)
        view_menu.addAction(self.__action_toggle_log)

        help_menu = menu_bar.addMenu("&Help")
        action_about = QAction("&About SarcAsM", self.__window)
        action_about.setMenuRole(QAction.AboutRole)
        action_about.triggered.connect(self.__show_about)
        help_menu.addAction(action_about)

        return menu_bar

    def __toggle_log_panel(self, checked):
        self.__text_debug.setVisible(checked)

    def __show_about(self):
        QMessageBox.about(
            self.__window,
            "About SarcAsM",
            f"<h3>SarcAsM v{version}</h3>"
            "<p>Sarcomere Analysis Multi-tool.</p>"
            "<p>University Medical Center Göttingen</p>"
            "<p>Patent Pending: DE 10 2024 112 939.5</p>",
        )

    # Stylesheets for status indicators (color + background in one rule — setStyleSheet
    # replaces the whole stylesheet, so splitting across calls drops earlier properties)
    __STATUS_OK_QSS = (
        'padding: 2px 6px; border-radius: 3px; '
        'color: rgba(255,255,255,0.9); background-color: rgba(0,160,60,0.6);'
    )
    __STATUS_BAD_QSS = (
        'padding: 2px 6px; border-radius: 3px; '
        'color: rgba(255,255,255,0.9); background-color: rgba(200,60,60,0.7);'
    )

    def __update_busy_label(self, new_value):
        if new_value:
            self.__label_busy.setText('⏳ BUSY')
            self.__label_busy.setStyleSheet(self.__STATUS_BAD_QSS)
        else:
            self.__label_busy.setText('● IDLE')
            self.__label_busy.setStyleSheet(self.__STATUS_OK_QSS)
        self.__sync_wait_cursor(bool(new_value))

    def __sync_wait_cursor(self, busy):
        # Mirror the override cursor to the busy state so long-running ops show
        # a spinner instead of silence. Track our own push count so we don't
        # accidentally restore cursors pushed by other code paths.
        if busy and not self.__wait_cursor_active:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.__wait_cursor_active = True
        elif not busy and self.__wait_cursor_active:
            QApplication.restoreOverrideCursor()
            self.__wait_cursor_active = False

    def __init_status_bar(self):
        h_box = QHBoxLayout()
        h_box.setContentsMargins(4, 2, 4, 2)
        h_box.addWidget(self.__label_gpu, 0)
        h_box.addWidget(self.__label_busy, 0)
        h_box.addWidget(self.__progress_bar, 1)
        self.__status_bar.setLayout(h_box)

        self.__label_busy.setStyleSheet(self.__STATUS_OK_QSS)
        if self.__control.is_gpu_available():
            self.__label_gpu.setText('● GPU')
            self.__label_gpu.setStyleSheet(self.__STATUS_OK_QSS)
        else:
            self.__label_gpu.setText('○ GPU')
            self.__label_gpu.setStyleSheet(self.__STATUS_BAD_QSS)

        # in case of idle -> the label should display IDLE with green background
        # in case of busy -> the label should display BUSY with red background
        self.__control.model.currentlyProcessing.connect(
            ui_element=lambda new_value: self.__update_busy_label(new_value))
        pass

    @staticmethod
    def check_github_release(owner, repo, current_version):
        """Check GitHub for the latest release and print if a new version is available.
           Only runs when packaged with PyInstaller.
        """
        # Only check if running in a PyInstaller bundle
        if not (getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')):
            return None  # Or "" if you prefer

        url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                latest = response.json()
                latest_version = latest.get("tag_name") or latest.get("name")
                release_url = latest.get("html_url", f"https://github.com/{owner}/{repo}/releases/latest")
                # Try to get the first asset download link, if it exists
                assets = latest.get("assets", [])
                if assets:
                    asset = assets[0]
                    download_url = asset.get("browser_download_url")
                else:
                    download_url = release_url  # Fallback to release page

                if latest_version and latest_version != current_version:
                    msg = (
                        f"New release available: {latest_version} (You have: {current_version})\n"
                        f"Download: {download_url}\n"
                    )
                    return msg
                else:
                    return "You have the latest version."
            else:
                return f"Failed to fetch release info: {response.status_code}"
        except Exception as e:
            return f"Error checking for updates: {e}"

    def init_gui(self):
        self.__window.setWindowTitle(f'SarcAsM - v{version}')
        # Clamp initial size to available screen so the window never opens off-screen
        # (e.g. Windows 1080p at 125% scaling where 1000px height overflows)
        available = QDesktopWidget().availableGeometry()
        default_w, default_h = 800, 1000
        w = min(default_w, available.width() - 40)
        h = min(default_h, available.height() - 80)
        self.__window.setMinimumSize(600, 500)
        self.__window.resize(w, h)
        self.__center_ui()

        main_layout = QVBoxLayout()
        main_layout.setMenuBar(self.__build_menu_bar())

        self.__progress_bar.setObjectName("progressBarMain")

        widget_file_selection = QWidget()
        self.__file_selection.setupUi(widget_file_selection)

        main_layout.addWidget(widget_file_selection, 0)

        widget_center = QWidget()
        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.__get_parameter_scroll_box(), 3)
        widget_center.setLayout(center_layout)

        self.__text_debug.setReadOnly(True)

        # Vertical splitter lets users drag the log panel down to collapse it
        # when they don't need it. View → Toggle Log also hides/shows it.
        self.__main_splitter = QSplitter(Qt.Vertical)
        self.__main_splitter.addWidget(widget_center)
        self.__main_splitter.addWidget(self.__text_debug)
        self.__main_splitter.setStretchFactor(0, 4)
        self.__main_splitter.setStretchFactor(1, 1)
        self.__main_splitter.setCollapsible(0, False)
        self.__main_splitter.setCollapsible(1, True)
        main_layout.addWidget(self.__main_splitter, 1)

        self.__init_status_bar()
        main_layout.addWidget(self.__status_bar, 0)
        self.__bind_events()

        self.__window.setLayout(main_layout)
        self.__disable_scroll_on_spinbox()

        self.__window.show()

        # # Check release and notify when there's an update
        # owner = "danihae"
        # repo = "SarcAsM"
        # msg = self.check_github_release(owner, repo, version)
        # self.debug(msg)
        # if msg and "New release available" in msg:
        #     QMessageBox.information(self.__window, "Update Available", msg)

        sys.exit(self.__app.exec_())
