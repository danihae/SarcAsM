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

"""Matplotlib figures in their own Qt windows, independent of the matplotlib backend.

``plt.show()`` does nothing under the Agg backend the packaged app uses and blocks the
button under the macOS backend, so the GUI never calls it: figures drawn with pyplot are
handed to :func:`show_new_figures`, which moves each into a non-blocking window with the
usual zoom / pan / save toolbar.
"""

from typing import Iterable, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

# open windows; a window without a Python reference would be garbage-collected
_WINDOWS: List[QWidget] = []


class FigureWindow(QWidget):
    """A top-level window holding one matplotlib figure and its toolbar."""

    def __init__(self, figure: Figure, title: str = ''):
        super().__init__()
        self.setWindowTitle(title or 'SarcAsM figure')
        self.canvas = FigureCanvasQTAgg(figure)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        w, h = figure.get_size_inches() * figure.dpi
        self.resize(int(min(w, 1400)), int(min(h, 1000)) + 40)

    def closeEvent(self, event):
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        super().closeEvent(event)


def show_figure(figure: Figure, title: str = '') -> FigureWindow:
    """Show ``figure`` in its own window (non-blocking) and return the window."""
    window = FigureWindow(figure, title)
    _WINDOWS.append(window)
    window.show()
    window.raise_()
    return window


def show_new_figures(before: Iterable[int], title: str = '') -> List[FigureWindow]:
    """Move every pyplot figure not in ``before`` (figure numbers) into its own window.

    The figures are released from pyplot first, so its backend never shows or keeps them.
    """
    before = set(before)
    windows = []
    for num in plt.get_fignums():
        if num in before:
            continue
        figure = plt.figure(num)
        suptitle = figure._suptitle.get_text() if figure._suptitle is not None else ''
        plt.close(figure)
        windows.append(show_figure(figure, suptitle or title))
    return windows
