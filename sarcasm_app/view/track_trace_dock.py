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
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QLabel, QPushButton,
                             QFileDialog, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

_SERIES = {'ΔSL': ('delta_slen', 'ΔSL [µm]'), 'SL': ('slen', 'SL [µm]'), 'Velocity': ('vel', 'V [µm/s]')}


class TrackTraceDock(QWidget):
    """Overlay of the sarcomere time series (SL / ΔSL / velocity) of one group, the group
    mean, and the clicked sarcomere highlighted, with the contraction cycles shaded.

    ``set_source(kinematics, group_id, frametime, kind, group_contr)`` supplies the per-track
    arrays from :meth:`sarcasm.SarcAsM.get_track_kinematics` and, if the grouping was
    analysed, the ContractionNet intervals of every group; ``show_track(track_id)``
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
        self._group_contr: Optional[np.ndarray] = None
        self._track_id: Optional[int] = None
        self._group: Optional[int] = None
        self._frame = 0
        #: where "Export data" proposes to save (set by the motion control)
        self.export_dir: Optional[str] = None
        self.export_stem = 'traces'

        self.cb_series = QComboBox()
        self.cb_series.addItems(list(_SERIES))
        self.cb_series.setToolTip("<p>Which time series to plot. SL: sarcomere length [µm]. ΔSL: length minus the sarcomere's "
                                  'resting length [µm] (negative = shortened). Velocity: rate of length change [µm/s] (negative = '
                                  'shortening), from the length smoothed over 13 frames (Savitzky–Golay, order 5). The resting '
                                  "length is the median outside contractions if the motion analysis was run with 'Group by' pool, "
                                  'otherwise over all frames. Lengths outside 1.2–3.0 µm are left out.</p>')
        self.chk_group_mean = QCheckBox('Group mean')
        self.chk_group_mean.setChecked(True)
        self.chk_group_mean.setToolTip('<p>Draws the frame-by-frame mean of all sarcomeres of the selected group as a thick black line '
                                       '(the legend gives their number). With no group selected, the mean of all tracked sarcomeres. '
                                       'Untick to see only the single traces (grey) and the clicked sarcomere (red). The .csv export '
                                       'always contains the mean. Default on.</p>')
        self.chk_cycles = QCheckBox('Contraction cycles')
        self.chk_cycles.setChecked(True)
        self.chk_cycles.setToolTip('<p>Shades the contraction intervals that ContractionNet detected (lavender). With a group '
                                   "selected, the group's own intervals, from '3. Analyze track motion' on the current grouping; "
                                   "otherwise, or before that analysis, the whole cell's intervals (from 'Group by' pool). The "
                                   'legend says which. Without either analysis nothing is shaded. When ticked, the .csv export '
                                   'gets a column marking these frames. Default on.</p>')
        self.lbl_info = QLabel('Track sarcomeres first.')
        self.lbl_info.setWordWrap(True)

        bar = QHBoxLayout()
        lbl_series = QLabel('Series')
        lbl_series.setToolTip(self.cb_series.toolTip())
        bar.addWidget(lbl_series)
        bar.addWidget(self.cb_series)
        bar.addWidget(self.chk_group_mean)
        bar.addWidget(self.chk_cycles)
        bar.addStretch(1)
        self.btn_export = QPushButton('Export data (.csv)…')
        self.btn_export.setToolTip('<p>Saves the chosen series (SL, ΔSL or velocity) as a table: time [s], the group mean, and one '
                                   'column per sarcomere of the shown group, including those not drawn (the plot shows at most 300).'
                                   " With 'Contraction cycles' ticked, a column marks the contracting frames. Suggested name: "
                                   '&lt;image&gt;_&lt;group&gt;_&lt;series&gt;.csv in the image folder. Save the plot itself with '
                                   'the disk icon of the plot toolbar.</p>')
        self.btn_export.clicked.connect(self._on_export)
        bar.addWidget(self.btn_export)

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
    def set_source(self, kinematics: dict, group_id: Optional[np.ndarray], frametime: float, kind: str = '',
                   group_contr: Optional[np.ndarray] = None):
        gid = None if group_id is None else np.asarray(group_id).reshape(-1)
        same = (kinematics is self._kin and self._frametime == float(frametime) and self._kind == (kind or '')
                and group_contr is self._group_contr
                and ((gid is None and self._gid is None)
                     or (gid is not None and self._gid is not None and gid.shape == self._gid.shape
                         and np.array_equal(gid, self._gid))))
        if same:
            return
        self._kin = kinematics
        self._gid = None if group_id is None else np.asarray(group_id).reshape(-1)
        self._frametime = float(frametime)
        self._kind = kind or ''
        self._group_contr = group_contr
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
            contr, source = self._contraction_mask(T)
            if contr is not None and contr.any():
                edges = np.flatnonzero(np.diff(np.r_[0, contr.astype(int), 0]))
                for n, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
                    ax.axvspan(a * self._frametime, b * self._frametime, color='lavender', alpha=0.7, lw=0,
                               label=f'contracting ({source})' if n == 0 else None)

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

    def _contraction_mask(self, T):
        """The contraction intervals to shade, ``(n_frames,)`` bool, and where they come from:
        the selected group's own ContractionNet intervals if the grouping was analysed,
        else the whole cell's (pool) intervals. ``(None, '')`` if there are neither."""
        gc = self._group_contr
        if (self._group is not None and gc is not None and gc.ndim == 2 and gc.shape[1] == T
                and 0 <= self._group < gc.shape[0]):
            return gc[self._group].astype(bool), f'{self._kind or "group"} {self._group}'
        contr = self._kin.get('contr')
        if contr is not None and np.asarray(contr).size == T and np.asarray(contr).any():
            return np.asarray(contr, dtype=bool).reshape(-1), 'whole cell'
        return None, ''

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

    def traces_table(self):
        """The plotted series as a table (pandas DataFrame): ``time [s]``, the group mean, the
        highlighted track and every member of the shown group, one column each."""
        import pandas as pd
        if self._kin is None:
            return None
        series_label = self.cb_series.currentText()
        key, ylabel = _SERIES[series_label]
        unit = ylabel[ylabel.find('['):] if '[' in ylabel else ''
        series = np.asarray(self._kin[key], dtype=float)
        T = series.shape[1]
        cols = {'time [s]': np.arange(T) * self._frametime}
        members = None
        label = 'all tracks'
        if self._group is not None and self._gid is not None:
            members = np.flatnonzero(self._gid == self._group)
            label = f'{self._kind or "group"} {self._group}'
        if members is None or members.size == 0:
            members = np.flatnonzero(np.isfinite(series).any(axis=1))
        with np.errstate(all='ignore'):
            cols[f'{label} mean {series_label} {unit}'.strip()] = np.nanmean(series[members], axis=0)
        if self.chk_cycles.isChecked():
            contr, source = self._contraction_mask(T)
            if contr is not None:
                cols[f'contracting ({source})'] = contr[:T]
        ids = self._kin.get('track_ids')
        for m in members:
            name = int(ids[m]) if ids is not None else int(m)
            cols[f'track {name} {series_label} {unit}'.strip()] = series[m]
        return pd.DataFrame(cols)

    def _on_export(self):
        import os
        table = self.traces_table()
        if table is None:
            QMessageBox.information(self, 'Nothing to export', 'Track sarcomeres first, then click a sarcomere '
                                    'or a fibre path in the viewer.')
            return
        series = {'ΔSL': 'delta_slen', 'SL': 'slen', 'Velocity': 'velocity'}[self.cb_series.currentText()]
        part = f'{self._kind or "group"}{self._group}' if self._group is not None else 'all'
        default = os.path.join(self.export_dir or os.path.expanduser('~'),
                               f'{self.export_stem}_{part}_{series}.csv')
        path, _ = QFileDialog.getSaveFileName(self, 'Save time series', default, 'CSV (*.csv)')
        if not path:
            return
        try:
            table.to_csv(path, index=False)
        except OSError as e:
            QMessageBox.warning(self, 'Not saved', f'Could not write\n{path}\n\n{e.strerror or e}. Choose a '
                                f'folder you can write to, and close the file if it is open elsewhere.')
            return
        QMessageBox.information(self, 'Export complete',
                                f'Saved {table.shape[1] - 1} time series ({table.shape[0]} frames) to\n{path}')

    def _draw_cursor(self):
        if self._frametime is None:
            return
        x = self._frame * self._frametime
        self._cursor = self.ax.axvline(x, c='tab:blue', lw=0.8, alpha=0.8, animated=True)
        self._background = None
