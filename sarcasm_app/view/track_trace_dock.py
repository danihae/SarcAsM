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

"""Dock panel with the time series of one tracked sarcomere (SL / ΔSL / velocity)."""

from typing import Optional

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

_SERIES = {'ΔSL': ('delta_slen', 'ΔSL [µm]'), 'SL': ('slen', 'SL [µm]'), 'Velocity': ('vel', 'V [µm/s]')}


class TrackTraceDock(QWidget):
    """Overlay of the sarcomere time series (SL / ΔSL / velocity) of one group, the group
    mean, and the clicked sarcomere highlighted, with the contraction cycles shaded.

    ``set_source(kinematics, group_id, frametime, kind)`` supplies the per-track
    arrays from :meth:`sarcasm.SarcAsM.get_track_kinematics`; ``show_track(track_id)``
    draws one; ``set_frame(t)`` moves the cursor line with the viewer.
    """

    #: individual traces drawn at most (a random subset beyond that; the clicked one always)
    MAX_LINES = 300

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._kin = None
        self._gid = None
        self._frametime = None
        self._kind = ''
        self._track_id: Optional[int] = None
        self._group: Optional[int] = None
        self._frame = 0

        self.cb_series = QComboBox()
        self.cb_series.addItems(list(_SERIES))
        self.chk_group_mean = QCheckBox('Group mean')
        self.chk_group_mean.setChecked(True)
        self.chk_cycles = QCheckBox('Contraction cycles')
        self.chk_cycles.setChecked(True)
        self.lbl_info = QLabel('Track sarcomeres first.')
        self.lbl_info.setWordWrap(True)

        bar = QHBoxLayout()
        bar.addWidget(QLabel('Series'))
        bar.addWidget(self.cb_series)
        bar.addWidget(self.chk_group_mean)
        bar.addWidget(self.chk_cycles)
        bar.addStretch(1)

        self.figure = Figure(figsize=(6, 2.6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)   # zoom / pan / save
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.cb_series.currentTextChanged.connect(lambda _: self.redraw())
        self.chk_group_mean.toggled.connect(lambda _: self.redraw())
        self.chk_cycles.toggled.connect(lambda _: self.redraw())

        # the cursor follows the frame slider: blitted (only the line is redrawn) and
        # throttled, so playing the movie does not saturate the event loop
        self._background = None
        self._cursor = None
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(50)
        self._cursor_timer.timeout.connect(self._blit_cursor)
        self.canvas.mpl_connect('draw_event', self._on_draw)

    # ------------------------------------------------------------------ #
    def set_source(self, kinematics: dict, group_id: Optional[np.ndarray], frametime: float, kind: str = ''):
        gid = None if group_id is None else np.asarray(group_id).reshape(-1)
        same = (kinematics is self._kin and self._frametime == float(frametime) and self._kind == (kind or '')
                and ((gid is None and self._gid is None)
                     or (gid is not None and self._gid is not None and gid.shape == self._gid.shape
                         and np.array_equal(gid, self._gid))))
        if same:
            return
        self._kin = kinematics
        self._gid = None if group_id is None else np.asarray(group_id).reshape(-1)
        self._frametime = float(frametime)
        self._kind = kind or ''
        if self._kin is not None and self._track_id is not None and self._track_id >= self._kin['slen'].shape[0]:
            self._track_id = None
        if self._gid is None or (self._group is not None and not np.any(self._gid == self._group)):
            self._group = None
        self.redraw()

    def show_track(self, track_id: int):
        """Highlight one sarcomere within the overlay of its group."""
        self._track_id = int(track_id)
        self._group = None
        if self._gid is not None and self._track_id < self._gid.size and self._gid[self._track_id] >= 0:
            self._group = int(self._gid[self._track_id])
        self.redraw()

    def show_group(self, group: Optional[int]):
        """Overlay one group (None: every track) without a highlighted sarcomere."""
        self._group = None if group is None else int(group)
        self._track_id = None
        self.redraw()

    def set_frame(self, frame: int):
        self._frame = int(frame)
        if self.isVisible() and not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _on_draw(self, _event):
        """After a full draw (which skips the animated cursor), remember the
        background and paint the cursor on top."""
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        if self._cursor is not None:
            self.ax.draw_artist(self._cursor)
            self.canvas.blit(self.ax.bbox)

    def _blit_cursor(self):
        if self._cursor is None or self._background is None:
            return
        x = self._frame * (self._frametime or 1.0)
        self._cursor.set_xdata([x, x])
        self.canvas.restore_region(self._background)
        self.ax.draw_artist(self._cursor)
        self.canvas.blit(self.ax.bbox)

    # ------------------------------------------------------------------ #
    def redraw(self):
        ax = self.ax
        ax.clear()
        self._cursor = None
        if self._kin is None:
            self.lbl_info.setText('Track sarcomeres first.')
            self.canvas.draw_idle()
            return
        key, ylabel = _SERIES[self.cb_series.currentText()]
        series = np.asarray(self._kin[key], dtype=float)
        T = series.shape[1]
        time = np.arange(T) * self._frametime

        if self.chk_cycles.isChecked():
            contr = np.asarray(self._kin.get('contr', np.zeros(T, bool)), dtype=bool)
            if contr.any():
                edges = np.flatnonzero(np.diff(np.r_[0, contr.astype(int), 0]))
                for a, b in zip(edges[::2], edges[1::2]):
                    ax.axvspan(a * self._frametime, b * self._frametime, color='lavender', alpha=0.7, lw=0)

        # the set of traces to overlay: the selected group, else every track
        i = self._track_id
        members = None
        label = 'all tracks'
        if self._group is not None and self._gid is not None:
            members = np.flatnonzero(self._gid == self._group)
            label = f'{self._kind or "group"} {self._group}'
        if members is None or members.size == 0:
            members = np.flatnonzero(np.isfinite(series).any(axis=1))
            label = 'all tracks'
        shown = members
        if shown.size > self.MAX_LINES:
            shown = np.sort(np.random.default_rng(0).choice(members, self.MAX_LINES, replace=False))
            if i is not None and i not in shown:
                shown = np.sort(np.append(shown, i))
        if shown.size:
            alpha = float(max(0.04, min(0.5, 30.0 / shown.size)))
            ax.plot(time, series[shown].T, c='0.6', lw=0.4, alpha=alpha)
        if self.chk_group_mean.isChecked() and members.size > 1:
            with np.errstate(all='ignore'):
                ax.plot(time, np.nanmean(series[members], axis=0), c='k', lw=1.6, zorder=3,
                        label=f'{label} mean ({members.size})')
        if i is not None:
            ax.plot(time, series[i], c='C3', lw=1.2, zorder=4, label=f'track {i}')
            cov = self._kin['coverage'][i] if 'coverage' in self._kin else np.nan
            equ = self._kin['equ'][i] if 'equ' in self._kin else np.nan
            self.lbl_info.setText(f'Track {i}  ·  {label} ({members.size} sarcomeres)  ·  '
                                  f'coverage {cov:.0%}  ·  resting SL {equ:.2f} µm')
        else:
            self.lbl_info.setText(f'Overlay of {label} ({members.size} sarcomeres'
                                  f'{", " + str(shown.size) + " drawn" if shown.size < members.size else ""}); '
                                  'click a sarcomere in the viewer to highlight it and its group.')
        if key != 'slen':
            ax.axhline(0, c='k', lw=0.6, ls=':')
        ax.set_ylim(*self._ylim(series, shown, members, i))
        ax.set_xlim(0, T * self._frametime)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize='small', frameon=False)
        self._draw_cursor()
        self.canvas.draw_idle()

    @staticmethod
    def _ylim(series, shown, members, track):
        """Limits from the 1st-99th percentile of the drawn traces (so a single
        stray value cannot blow up the axis), widened to hold the mean and the
        highlighted track completely, with 15 % padding."""
        drawn = series[shown][np.isfinite(series[shown])] if shown.size else np.zeros(0)
        if drawn.size == 0:
            return (-0.5, 0.5)
        lo, hi = np.percentile(drawn, [1, 99])
        with np.errstate(all='ignore'):
            mean = np.nanmean(series[members], axis=0)
        for extra in ([mean] + ([series[track]] if track is not None else [])):
            finite = extra[np.isfinite(extra)]
            if finite.size:
                lo, hi = min(lo, finite.min()), max(hi, finite.max())
        span = hi - lo if hi > lo else 0.1
        return (lo - 0.15 * span, hi + 0.15 * span)

    def _draw_cursor(self):
        if self._frametime is None:
            return
        x = self._frame * self._frametime
        self._cursor = self.ax.axvline(x, c='tab:blue', lw=0.8, alpha=0.8, animated=True)
        self._background = None
