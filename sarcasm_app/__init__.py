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
import re
import sys
from pathlib import Path

# Import torch before PyQt5: on Windows, importing PyQt5 first leaves the
# DLL loader in a state where torch's bundled CUDA libs fail with WinError 1114
# ("DLL initialization routine failed") when c10.dll is loaded.
import torch  # noqa: F401

import requests
from PyQt5.QtCore import Qt, QLocale, QUrl
from PyQt5.QtGui import QPalette, QColor, QIcon, QDesktopServices
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QStyleFactory, QAbstractSpinBox,
                             QAction, QMenuBar, QMessageBox, QSplitter, QTabWidget,
                             QPushButton, QRadioButton, QButtonGroup, QFormLayout, QFrame)
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
QLabel[role="tabBanner"] {
    color: rgba(255,255,255,0.75);
    background-color: rgba(42,130,218,0.12);
    border-left: 3px solid #2a82da;
    padding: 6px 10px;
    font-size: 11px;
}
QFrame[role="scopeRow"] {
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px;
}
"""


def _mark_accent(button):
    """Tag a button so the app stylesheet renders it with the accent color.
    Must be called after the button is added to a widget hierarchy, because
    Qt only re-evaluates the stylesheet on the next style polish."""
    button.setProperty("accent", True)
    button.style().unpolish(button)
    button.style().polish(button)


# Matches a unit expressed like "[µm]", "[s]", "[frames]" inside a label.
_UNIT_PATTERN = re.compile(r'\s*\[([^\]]+)\]')


def _apply_spinbox_unit_suffixes(root: QWidget) -> None:
    """Walk every QFormLayout under `root` and, where the label text contains
    a unit like "[µm]", move that unit onto the paired spinbox via setSuffix.
    Keeps the label shorter and displays the unit next to the actual value."""
    for form in root.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, QFormLayout.LabelRole)
            field_item = form.itemAt(row, QFormLayout.FieldRole)
            if label_item is None or field_item is None:
                continue
            label = label_item.widget()
            if not isinstance(label, QLabel):
                continue
            match = _UNIT_PATTERN.search(label.text() or '')
            if not match:
                continue
            unit = match.group(1)

            spinboxes = []
            field_widget = field_item.widget()
            if isinstance(field_widget, QAbstractSpinBox):
                spinboxes.append(field_widget)
            elif field_item.layout() is not None:
                layout = field_item.layout()
                for i in range(layout.count()):
                    w = layout.itemAt(i).widget()
                    if isinstance(w, QAbstractSpinBox):
                        spinboxes.append(w)
            if not spinboxes:
                continue
            for sb in spinboxes:
                sb.setSuffix(f' {unit}')
            trimmed = _UNIT_PATTERN.sub('', label.text()).strip()
            label.setText(trimmed)

from .control.application_control import ApplicationControl
from .control.logging_handler import setup_gui_logging
from .control.file_selection_control import FileSelectionControl
from .control.motion_analysis_control import MotionAnalysisControl
from .control.structure_analysis_control import StructureAnalysisControl
from .control.batch_processing_control import BatchProcessingControl
from .model import ApplicationModel
from .view.file_selection import Ui_Form as FileSelectionWidget
from .view.parameters_structure_analysis import Ui_Form as StructureAnalysisWidget
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

    def __wrap_tab(self, inner: QWidget, banner_text: str, reset_callback) -> QWidget:
        """Wrap a parameter panel in a container with a help banner and a
        "Reset tab defaults" button so each tab carries its own scope reset
        instead of sharing a single global reset."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        banner = QLabel(banner_text)
        banner.setWordWrap(True)
        banner.setProperty("role", "tabBanner")
        header.addWidget(banner, 1)
        btn_reset = QPushButton("Reset tab defaults")
        btn_reset.setToolTip("Restore default values for parameters on this tab")
        btn_reset.clicked.connect(reset_callback)
        header.addWidget(btn_reset, 0, Qt.AlignTop)
        layout.addLayout(header)
        layout.addWidget(inner, 1)
        return container

    def __build_analysis_scope_row(self) -> QWidget:
        """Build a persistent "Frames to analyze" row above the parameter tabs.

        Provides a two-mode toggle:
          [All frames]  vs  [Selected frames: ___________]
        When "All frames" is active the text entry is hidden and 'all' is written
        to le_general_frames automatically.  When "Selected frames" is active the
        entry is shown so the user can type a single number or comma-separated list.

        le_general_frames stays reparented here so the existing parameter binding
        in structure_analysis_control continues to read/write it without changes.
        """
        le = self.__structure_analysis_parameters.le_general_frames
        # label_4 ("Frames") is no longer needed as a free-floating label;
        # just hide it so it does not consume space in the old form layout.
        self.__structure_analysis_parameters.label_4.setVisible(False)

        scope = QFrame()
        scope.setProperty("role", "scopeRow")
        row = QHBoxLayout(scope)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(10)

        lbl = QLabel("Frames to analyze:")
        lbl.setStyleSheet("font-weight: 600;")
        row.addWidget(lbl)

        radio_all = QRadioButton("All frames")
        radio_sel = QRadioButton("Selected:")
        radio_all.setToolTip("Run analysis on every frame in the file")
        radio_sel.setToolTip(
            "Run analysis on specific frames only. "
            "Enter a single frame number or comma-separated list (frames start at 0)."
        )
        grp = QButtonGroup(scope)
        grp.addButton(radio_all)
        grp.addButton(radio_sel)
        row.addWidget(radio_all)
        row.addWidget(radio_sel)

        # Re-parent the existing QLineEdit into this row.
        le.setParent(scope)
        le.setPlaceholderText("e.g. 0, 1, 5")
        le.setMaximumWidth(220)
        row.addWidget(le)
        row.addStretch(1)

        def _sync_from_toggle():
            if radio_all.isChecked():
                le.setText("all")
                le.setVisible(False)
            else:
                if le.text().strip().lower() == "all":
                    le.setText("")
                le.setVisible(True)
                le.setFocus()

        def _sync_from_text():
            """Keep the radio buttons in sync when the model writes 'all' back."""
            val = le.text().strip().lower()
            if val == "all":
                radio_all.setChecked(True)
                le.setVisible(False)
            else:
                radio_sel.setChecked(True)
                le.setVisible(True)

        radio_all.toggled.connect(lambda checked: _sync_from_toggle() if checked else None)
        radio_sel.toggled.connect(lambda checked: _sync_from_toggle() if checked else None)
        le.textChanged.connect(_sync_from_text)

        # Set initial state from the current parameter value.
        _sync_from_text()

        # Keep references so they are not garbage-collected.
        self.__scope_radio_all = radio_all
        self.__scope_radio_sel = radio_sel
        self.__scope_button_group = grp
        return scope

    def __install_override_radio(self) -> None:
        """B3: Replace the plain "Force overwrite metadata" checkbox with a
        two-option segmented control in the batch tab. The checkbox stays in
        place (hidden) so existing parameter bindings keep working."""
        chk = self.__batch_processing.chk_force_override
        host = chk.parentWidget()
        if host is None:
            return
        host_layout = host.layout()
        if host_layout is None:
            return

        radio_use_meta = QRadioButton("Use file metadata")
        radio_use_meta.setToolTip("Read pixel size and frame time from each image file")
        radio_override = QRadioButton("Override for all files")
        radio_override.setToolTip("Apply the pixel size and frame time entered above to every image in the batch")

        group = QButtonGroup(host)
        group.addButton(radio_use_meta)
        group.addButton(radio_override)
        radio_use_meta.setChecked(not chk.isChecked())
        radio_override.setChecked(chk.isChecked())

        radio_override.toggled.connect(chk.setChecked)
        # Keep radio state in sync if something else flips the hidden checkbox
        chk.toggled.connect(lambda checked: (radio_override.setChecked(checked)
                                             if checked else radio_use_meta.setChecked(True)))

        # Insert the radio group in place of the hidden checkbox.
        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(12)
        hl.addWidget(radio_use_meta)
        hl.addWidget(radio_override)
        hl.addStretch(1)

        idx = host_layout.indexOf(chk)
        if idx >= 0:
            host_layout.insertWidget(idx, container)
        else:
            host_layout.addWidget(container)
        chk.setVisible(False)
        # Keep a reference so it isn't garbage-collected.
        self.__override_button_group = group
        self.__override_radio_container = container

    def __get_parameter_scroll_box(self):
        # QTabWidget (horizontal tabs) instead of QToolBox — reclaims ~100px of
        # vertical space that QToolBox's stacked section headers consumed.
        widget_parameter_tabs = QTabWidget()
        widget_parameter_tabs.setDocumentMode(True)
        widget_parameter_tabs.setUsesScrollButtons(True)

        widget_structure_parameters = QWidget()
        self.__structure_analysis_parameters.setupUi(widget_structure_parameters)
        _mark_accent(self.__structure_analysis_parameters.btn_analyze_structure)

        widget_motion_analysis = QWidget()
        self.__motion_analysis.setupUi(widget_motion_analysis)
        _mark_accent(self.__motion_analysis.btn_analyze_motion)

        widget_batch_processing = QWidget()
        self.__batch_processing.setupUi(widget_batch_processing)
        _mark_accent(self.__batch_processing.btn_batch_processing_structure)
        self.__install_override_radio()

        # B5: unit suffixes (after setupUi so the labels exist).
        for w in (widget_structure_parameters,
                  widget_motion_analysis, widget_batch_processing):
            _apply_spinbox_unit_suffixes(w)

        model = self.__control.model
        tabs = [
            (widget_structure_parameters,
             'Structure Analysis',
             'Analyze sarcomere morphology: Z-bands, vectors, myofibrils, and domains.',
             model._set_defaults_structure),
            (widget_motion_analysis,
             'Motion',
             'Track sarcomeres across frames, group into fibres/domains, and analyze contraction per group.',
             model._set_defaults_motion),
            (widget_batch_processing,
             'Batch Processing',
             'Run the configured structure analysis on every file in a directory tree.',
             model._set_defaults_batch),
        ]
        for inner, label, banner, reset_cb in tabs:
            widget_parameter_tabs.addTab(self.__wrap_tab(inner, banner, reset_cb), label)

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
        action_docs = QAction("&Documentation", self.__window)
        action_docs.setShortcut("F1")
        action_docs.triggered.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://sarcasm.readthedocs.io/en/latest/index.html")))
        help_menu.addAction(action_docs)

        action_github = QAction("&GitHub Repository", self.__window)
        action_github.triggered.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/danihae/SarcAsM")))
        help_menu.addAction(action_github)

        help_menu.addSeparator()

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
            "<p>Patent Pending: DE 10 2024 112 939.5</p>"
            "<p>"
            "<a href='https://sarcasm.readthedocs.io/en/latest/index.html'>Documentation</a> &nbsp;·&nbsp; "
            "<a href='https://github.com/danihae/SarcAsM'>GitHub</a>"
            "</p>",
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

        # The scroll box runs setupUi on the structure tab, so it must be
        # built before __build_analysis_scope_row can reparent the Frames widgets.
        parameter_scroll_box = self.__get_parameter_scroll_box()
        main_layout.addWidget(self.__build_analysis_scope_row(), 0)

        widget_center = QWidget()
        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(parameter_scroll_box, 3)
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

        # Closing the main window must also tear down the napari viewer; otherwise
        # napari stays open as an orphan and the Qt event loop keeps running.
        # Monkey-patch is the lightest hook here — subclassing QWidget just to
        # override closeEvent would require restructuring the constructor.
        _original_close = self.__window.closeEvent
        _control = self.__control
        def _close_event(event):
            _control.shutdown()
            _original_close(event)
        self.__window.closeEvent = _close_event

        self.__window.show()

        # # Check release and notify when there's an update
        # owner = "danihae"
        # repo = "SarcAsM"
        # msg = self.check_github_release(owner, repo, version)
        # self.debug(msg)
        # if msg and "New release available" in msg:
        #     QMessageBox.information(self.__window, "Update Available", msg)

        sys.exit(self.__app.exec_())
