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
import os
import traceback
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sarcasm import Utils
from sarcasm.type_utils import TypeUtils
from sarcasm.analysis.domain_clustering import analyze_domains

import napari
import numpy as np
import torch
from PyQt5.QtCore import QObject, QEvent, QTimer, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QProgressBar, QTextEdit, QApplication
from bio_image_unet.progress import ProgressNotifier

from ..model import ApplicationModel

logger = logging.getLogger(__name__)


class _ViewerFileDropFilter(QObject):
    """Route file drops on the app's embedded napari viewer to the SarcAsM
    importer (full analysis layers) instead of napari's generic reader plugin.

    Installed as a Qt event filter on the viewer's QtViewer widget, so it sees
    the drop *before* napari's own ``dropEvent`` and can consume it. Non-image
    drops (or anything we don't recognise) fall through to napari's default
    handling.
    """

    # Suffixes the SarcAsM importer can open (movies + analysed stores).
    _EXTS = (".ome.zarr", ".zarr", ".ome.tif", ".ome.tiff", ".tif", ".tiff")

    def __init__(self, open_handler):
        super().__init__()
        self._open_handler = open_handler

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Drop:
            md = event.mimeData()
            if md is not None and md.hasUrls():
                paths = [u.toLocalFile() for u in md.urls() if u.toLocalFile()]
                target = next((p for p in paths
                               if p.lower().rstrip("/").endswith(self._EXTS)), None)
                if target is not None:
                    event.acceptProposedAction()
                    # Defer the (heavy, synchronous) import out of the event
                    # handler so the drop returns immediately and Qt isn't
                    # blocked mid-event.
                    QTimer.singleShot(0, lambda p=target: self._open_handler(p))
                    return True  # consume — don't fall through to napari
        return False


class ApplicationControl:
    """
    Main application control.
    It contains some public utility methods and handles parts of the general application flow.
    """

    def __init__(self, window: QWidget, model):
        """
        window: QWidget
        model: ApplicationModel (has to be of that type), due to removing possible circular dependencies
        -> removed the import statement and type specifier

        """
        self._window = window
        self._model: ApplicationModel = model
        self._viewer = None  # napari.Viewer(title='Image Window(napari)')  # the napari viewer object
        self.__debug_action = None
        self.__worker_thread: Optional[QThread] = None
        # Drop-to-import on the embedded viewer: a handler registered by the
        # file-selection control + the Qt event filter that calls it.
        self.__open_file_handler = None
        self.__drop_filter: Optional[_ViewerFileDropFilter] = None
        # Called after a file (tif or existing .ome.zarr) has been opened and its
        # layers built, so the tab controls can refresh state read from the store.
        self.file_opened_callbacks: list = []
        # Called with a track id when a sarcomere is clicked in the viewer, and with a
        # group id when a fibre path of the Groups layer is clicked.
        self.track_selected_callbacks: list = []
        self.group_selected_callbacks: list = []
        self._track_click_installed = False

        self.progress_notifier = ProgressNotifier()
        self.progress_notifier.set_progress_report(lambda p: self.update_progress(p * 100))
        self.debug("dummy line")  # to prevent the actual first line to get replaced by testing progress detail
        self.progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: self.debug_replace(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))

    def debug(self, message):
        # for now, just print
        if self.__debug_action is not None:
            self.__debug_action(message)

    def debug_replace(self, message):
        te = self._window.findChild(QTextEdit, name="messageArea")
        if te is not None:
            text = te.toPlainText()
            te.setText(text[:text.rfind('\n')])
            te.append(message)
            # te.verticalScrollBar().setValue(te.verticalScrollBar().maximum())  # scroll messageArea to last line!
            TypeUtils.if_present(te.verticalScrollBar(), lambda sc: sc.setValue(sc.maximum()))
            # remove last line
            # append message as last line

    def set_debug_action(self, debug_action):
        self.__debug_action = debug_action

    @property
    def model(self) -> ApplicationModel:
        return self._model

    @property
    def viewer(self) -> napari.Viewer:
        return self._viewer

    def set_open_file_handler(self, handler):
        """Register the callback used to import a file dropped on the viewer.

        ``handler`` is called with a single path string (on the Qt main thread)
        and should run the same import flow as the file-selection buttons.
        """
        self.__open_file_handler = handler

    @staticmethod
    def is_gpu_available():
        gpu_flag = False
        if torch.cuda.is_available():
            gpu_flag = True
        elif hasattr(torch, 'mps'):  # only for apple m1/m2/...
            if torch.backends.mps.is_available():
                gpu_flag = True
                pass
            pass
        return gpu_flag
        pass

    def clean_up_on_new_image(self):
        """Reset model to default state. Reuse the existing napari viewer when
        possible — clearing its layers preserves window position, zoom, and any
        dock widgets the user arranged. Only spawn a new viewer if the previous
        one was closed by the user."""
        if self._viewer is not None and napari.current_viewer() is self._viewer:
            self._viewer.layers.clear()
        else:
            self._viewer = napari.Viewer(title='SarcAsM')
            self.__configure_viewer_window(self._viewer)
        self.model.reset_model()

    def __viewer_qt_window(self):
        """Return the napari viewer's underlying QMainWindow, or None if napari
        changed its private API. Guarded so a napari upgrade can't crash the app."""
        if self._viewer is None:
            return None
        try:
            return self._viewer.window._qt_window
        except AttributeError:
            return None

    def __viewer_qt_viewer(self):
        """Return the napari QtViewer widget (the drop target), or None if napari
        changed its private API. Guarded like ``__viewer_qt_window``."""
        if self._viewer is None:
            return None
        win = getattr(self._viewer, 'window', None)
        for attr in ('_qt_viewer', 'qt_viewer'):
            obj = getattr(win, attr, None)
            if obj is not None:
                return obj
        return None

    def __configure_viewer_window(self, viewer):
        """Apply icon, paired placement, and close-coupling to a freshly created
        napari viewer. Uses napari's private `_qt_window` attribute, so each access
        is guarded — if napari renames it, we degrade silently rather than crash."""
        qt_window = self.__viewer_qt_window()
        if qt_window is None:
            return

        app = QApplication.instance()
        if app is not None and not app.windowIcon().isNull():
            qt_window.setWindowIcon(app.windowIcon())

        # Place the viewer to the right of the main window so they appear paired
        # rather than stacked. Clamp into the available screen area.
        if self._window is not None and self._window.isVisible():
            main_geom = self._window.frameGeometry()
            screen = app.primaryScreen().availableGeometry() if app is not None else None
            x = main_geom.right() + 10
            y = main_geom.top()
            if screen is not None:
                vw = qt_window.width() or 800
                if x + vw > screen.right():
                    x = max(screen.left(), screen.right() - vw)
                y = max(screen.top(), min(y, screen.bottom() - 100))
            qt_window.move(x, y)

        # When the user closes the napari window, also close the main window so
        # the app exits cleanly instead of stranding a controller with no viewer.
        qt_window.destroyed.connect(self.__on_viewer_destroyed)

        # Route file drops on this viewer to the SarcAsM importer (full analysis
        # layers) instead of napari's generic reader. A fresh filter per viewer
        # (each new viewer is a new QtViewer); kept referenced so it isn't GC'd.
        qt_viewer = self.__viewer_qt_viewer()
        if qt_viewer is not None and self.__open_file_handler is not None:
            self.__drop_filter = _ViewerFileDropFilter(self.__open_file_handler)
            qt_viewer.setAcceptDrops(True)
            qt_viewer.installEventFilter(self.__drop_filter)

    def __on_viewer_destroyed(self, *_):
        self._viewer = None
        if self._window is not None and self._window.isVisible():
            self._window.close()

    def shutdown(self):
        """Called from the main window's closeEvent so closing the SarcAsM window
        also tears down the napari viewer instead of leaving an orphan window."""
        if self._viewer is not None:
            try:
                if napari.current_viewer() is self._viewer:
                    self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def set_viewer_title(self, file_path):
        """Reflect the currently-loaded file in both the main window and the
        napari viewer titles, so the two windows are obviously paired."""
        name = os.path.basename(file_path) if file_path else ''
        suffix = f' — {name}' if name else ''
        if self._viewer is not None:
            self._viewer.title = f'SarcAsM{suffix}'
        if self._window is not None:
            from sarcasm import __version__ as version
            self._window.setWindowTitle(f'SarcAsM - v{version}{suffix}')

    def raise_viewer(self):
        """Bring the napari viewer to the front so analysis results aren't hidden
        behind the main window."""
        qt_window = self.__viewer_qt_window()
        if qt_window is None:
            return
        qt_window.raise_()
        qt_window.activateWindow()

    # def init_viewer(self, viewer):
    #    self._viewer = viewer

    def update_progress(self, value):
        progress_bar = self._window.findChild(QProgressBar, name='progressBarMain')
        if progress_bar is not None:
            progress_bar.setValue(int(value))

    # ------------------------------------------------------------------ #
    # track layers: trajectories (napari Tracks) + per-frame sarcomere state (Points)
    # ------------------------------------------------------------------ #
    #: Feature shown by the track layers -> (label, colormap, symmetric about 0?)
    TRACK_COLOR_FEATURES = {
        'delta_slen': ('ΔSL', 'RdBu_r', True),
        'slen': ('SL', 'viridis', False),
        'vel': ('Velocity', 'PuOr_r', True),
        'group_id': ('Group', 'gist_rainbow', False),
        'coverage': ('Coverage', 'viridis', False),
        'track_id': ('Track id', 'turbo', False),
    }
    #: Combo-box label -> feature key (the parameter stores the label)
    TRACK_COLOR_LABELS = {label: key for key, (label, _, _) in TRACK_COLOR_FEATURES.items()}
    #: Vertices the Trajectories layer is thinned to (every k-th frame per track): napari's
    #: Tracks layer builds its graph in O(rows) and takes seconds beyond a few million.
    TRACK_MAX_VERTICES = 600_000

    @classmethod
    def _ensure_track_colormaps(cls):
        """Register the matplotlib colormaps the track layers use with napari
        (its registry has no diverging map)."""
        from napari.utils.colormaps import ensure_colormap
        for _, cmap, _ in cls.TRACK_COLOR_FEATURES.values():
            try:
                ensure_colormap(cmap)
            except Exception as e:
                logger.debug(f'colormap {cmap} unavailable in napari: {e}')

    def track_kinematics(self):
        """``SarcAsM.get_track_kinematics()`` of the current cell, cached until the
        tracks, the grouping or the pooled contraction state change. None without tracks."""
        cell = self.model.cell
        if cell is None or 'motion.tracks.slen' not in cell.data:
            return None
        key = (id(cell.data.get('motion.tracks.slen')), cell.data.get('motion.groups.hash'),
               id(cell.data.get('motion.pool.contr')))
        cached = getattr(cell, '_track_kinematics_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        kin = cell.get_track_kinematics()
        cell._track_kinematics_cache = (key, kin)
        return kin

    def track_layer_table(self):
        """Per-observation table of the current tracks for the napari layers.

        Returns None without tracks. Otherwise a dict of equal-length arrays over
        every observed (track, frame) pair, sorted by track then frame:
        ``track_id, t, y, x`` (px) and the features ``slen, delta_slen, vel,
        group_id, coverage`` — plus ``contr`` (the pooled contraction mask) and
        ``n_tracks``. Cached on the cell until the tracks or the grouping change.
        """
        cell = self.model.cell
        if cell is None or 'motion.tracks.positions_px' not in cell.data:
            return None
        pos = np.asarray(cell.data['motion.tracks.positions_px'], dtype=float)
        if pos.ndim != 3 or pos.shape[0] == 0:
            return None
        n_tracks, T = pos.shape[:2]
        gid = cell.data.get('motion.tracks.group_id')
        gid = np.asarray(gid).reshape(-1) if gid is not None else np.full(n_tracks, -1)
        if gid.shape[0] != n_tracks:
            gid = np.full(n_tracks, -1)                      # stale grouping
        key = (id(cell.data.get('motion.tracks.slen')), cell.data.get('motion.groups.hash'),
               id(cell.data.get('motion.pool.contr')))
        cached = getattr(cell, '_track_layer_table', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        kin = self.track_kinematics()
        mask = np.isfinite(pos[..., 0]) & np.isfinite(pos[..., 1])
        observed = cell.data.get('motion.tracks.observed')
        if observed is not None and np.shape(observed) == mask.shape:
            mask &= np.asarray(observed).astype(bool)
        ii, tt = np.nonzero(mask)
        table = {
            'n_tracks': n_tracks, 'n_frames': T, 'contr': kin['contr'],
            'track_id': ii.astype(np.int64), 't': tt.astype(np.int64),
            'y': pos[ii, tt, 0], 'x': pos[ii, tt, 1],
            'slen': kin['slen'][ii, tt], 'delta_slen': kin['delta_slen'][ii, tt],
            'vel': kin['vel'][ii, tt], 'group_id': gid[ii].astype(np.int64),
            'coverage': kin['coverage'][ii],
        }
        cell._track_layer_table = (key, table)
        return table

    def _track_features(self, table, color_by, dsl_limit):
        """Feature columns for the layers, with the coloured one clipped so a
        diverging colormap is centred on zero. The base columns are built once per
        table; only the coloured column is recomputed."""
        base = table.get('_features_base')
        if base is None:
            base = {k: np.nan_to_num(np.asarray(table[k], dtype=float), nan=0.0)
                    for k in self.TRACK_COLOR_FEATURES}
            base['group_id'] = np.where(np.asarray(table['group_id']) < 0, 0.0, base['group_id'])
            table['_features_base'] = base
        feats = dict(base)
        _, _, symmetric = self.TRACK_COLOR_FEATURES[color_by]
        if symmetric:
            raw = np.asarray(table[color_by], dtype=float)
            lim = self._symmetric_limit(raw, dsl_limit if color_by == 'delta_slen' else None)
            col = np.nan_to_num(np.clip(raw, -lim, lim), nan=0.0)
            # anchor the range so the midpoint of the colormap is exactly 0
            if col.size:
                col[np.argmin(col)] = -lim
                col[np.argmax(col)] = lim
            feats[color_by] = col
        return feats

    @staticmethod
    def _symmetric_limit(values, explicit):
        if explicit is not None and explicit > 0:
            return float(explicit)
        finite = values[np.isfinite(values)]
        return float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0

    def init_tracks_stack(self, visible=True):
        """napari Tracks layer of the sarcomere trajectories with fading tails.

        Coloured by the feature chosen in the Motion tab (``motion.display.color_by``);
        every feature is attached, so the colouring can also be switched in the
        layer controls."""
        self._ensure_track_colormaps()
        cell = self.model.cell
        if 'Trajectories' in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers['Trajectories'])
        table = self.track_layer_table()
        if table is None or cell.metadata.n_stack is None or cell.metadata.n_stack <= 1:
            return
        color_by, dsl_limit, tail = self._track_display_settings()
        feats = self._track_features(table, color_by, dsl_limit)
        n_rows = table['track_id'].size
        stride = max(1, int(np.ceil(n_rows / self.TRACK_MAX_VERTICES)))
        keep = table['t'] % stride == 0
        data = np.column_stack([table['track_id'][keep], table['t'][keep],
                                table['y'][keep], table['x'][keep]]).astype(float)
        layer = self.viewer.add_tracks(data, name='Trajectories',
                                       features={k: v[keep] for k, v in feats.items()},
                                       color_by=color_by, colormap=self.TRACK_COLOR_FEATURES[color_by][1],
                                       tail_length=tail, tail_width=3, head_length=0,
                                       visible=visible, scale=cell.scale)
        layer.blending = 'translucent'
        layer.metadata['stride'] = stride
        if 'Selected group' in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers['Selected group'])
        if not self._track_click_installed:
            self.viewer.mouse_drag_callbacks.append(self._on_canvas_click)
            self._track_click_installed = True

    def init_track_groups_layer(self, visible=True):
        """Shapes layer 'Groups': one path per fibre for the chain groupings (myofibril /
        LOI) through its members at the reference frame, coloured by group and labelled
        with the group id. Other groupings show their groups through the Trajectories
        layer coloured by group."""
        if 'Groups' in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers['Groups'])
        cell = self.model.cell
        if cell is None or 'motion.tracks.group_id' not in cell.data:
            return
        kind = cell.data.get('motion.groups.kind')
        if kind not in ('myofibril', 'loi'):
            return
        pos = np.asarray(cell.data['motion.tracks.positions_px'], dtype=float)
        n_tracks, T = pos.shape[:2]
        gid = np.asarray(cell.data['motion.tracks.group_id']).reshape(-1)
        order = np.asarray(cell.data.get('motion.tracks.group_order', np.zeros(n_tracks))).reshape(-1)
        if gid.shape[0] != n_tracks:
            return
        ref = int(cell.data.get('params.group_tracks.reference_frame', 0))
        try:
            ref_idx = cell._tracked_frame_index(ref)
        except Exception:
            ref_idx = 0
        n_groups = max(1, int(cell.data.get('motion.groups.n', int(gid.max()) + 1)))
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('gist_rainbow')
        paths, colors, ids = [], [], []
        for g in range(n_groups):
            members = np.flatnonzero(gid == g)
            if members.size < 2:
                continue
            members = members[np.argsort(order[members])]
            yx = pos[members, ref_idx]
            ok = np.isfinite(yx).all(axis=1)
            if ok.sum() < 2:
                yx = np.array([np.nanmean(pos[m], axis=0) for m in members])   # fall back to track means
                ok = np.isfinite(yx).all(axis=1)
                if ok.sum() < 2:
                    continue
            paths.append(yx[ok])
            colors.append(cmap((g % n_groups) / n_groups))
            ids.append(g)
        if not paths:
            return
        layer = self.viewer.add_shapes(paths, name='Groups', shape_type='path',
                                       edge_color=np.asarray(colors), edge_width=1.5, opacity=0.8,
                                       features={'group': np.asarray(ids)},
                                       text={'string': '{group}', 'size': 8, 'color': 'white',
                                             'anchor': 'upper_left', 'translation': (-2, 0)},
                                       visible=visible, scale=cell.scale[-2:], ndim=2)
        layer.mouse_drag_callbacks.append(self._on_group_click)

    def highlight_group(self, group):
        """Ring the members of ``group`` in the viewer (all frames); None clears.

        The 'Selected group' layer is created once and only its data is swapped, so
        a selection never adds or removes a layer (which would refresh every layer)."""
        table = self.track_layer_table()
        cell = self.model.cell
        if table is None:
            return
        multi_frame = cell.metadata.n_stack is not None and cell.metadata.n_stack > 1
        ndim = 3 if multi_frame else 2
        if group is None or int(group) < 0:                  # -1 = unassigned, not a group
            pts = np.zeros((0, ndim))
        else:
            sel = table['group_id'] == int(group)
            pts = (np.column_stack([table['t'][sel], table['y'][sel], table['x'][sel]]) if multi_frame
                   else np.column_stack([table['y'][sel], table['x'][sel]])).astype(float)
        if 'Selected group' in self.viewer.layers:
            layer = self.viewer.layers['Selected group']
            layer.data = pts
            layer.visible = pts.shape[0] > 0
            return
        if pts.shape[0] == 0:
            return
        size = max(2.0, 0.3 / cell.metadata.pixelsize) * 1.8
        self.viewer.add_points(pts, name='Selected group', ndim=ndim, face_color='transparent',
                               border_color='yellow', border_width=0.25, size=size,
                               opacity=1.0, scale=cell.scale)

    def _on_group_click(self, layer, event):
        """Mouse callback on the Groups layer: a click on a fibre path selects its group."""
        start = np.asarray(event.position)
        yield
        while event.type == 'mouse_move':
            yield
        if np.linalg.norm(np.asarray(event.position) - start) > 1e-6:
            return
        idx = layer.get_value(event.position, view_direction=event.view_direction,
                              dims_displayed=event.dims_displayed, world=True)
        # never leave a shape selected: napari's highlight code cannot cope with a
        # selected shape on an invisible layer when the visibility is toggled
        layer.selected_data = set()
        shape_idx = idx[0] if isinstance(idx, tuple) else idx
        if shape_idx is None:
            return
        group = int(layer.features['group'].iloc[int(shape_idx)])
        for cb in self.group_selected_callbacks:
            cb(group)

    def _track_display_settings(self):
        p = self.model.parameters
        try:
            color_by = str(p.get_parameter('motion.display.color_by').get_value())
            dsl_limit = float(p.get_parameter('motion.display.dsl_limit').get_value())
            tail = int(p.get_parameter('motion.display.tail_frames').get_value())
        except Exception:
            color_by, dsl_limit, tail = 'delta_slen', 0.0, 30
        color_by = self.TRACK_COLOR_LABELS.get(color_by, color_by)
        if color_by not in self.TRACK_COLOR_FEATURES:
            color_by = 'delta_slen'
        return color_by, dsl_limit, max(1, tail)

    def apply_track_display(self, show_trajectories=None):
        """Re-colour the Trajectories layer from the Motion tab's Display settings
        without rebuilding it; optionally toggle its visibility."""
        self._ensure_track_colormaps()
        table = self.track_layer_table()
        if table is None:
            return
        color_by, dsl_limit, tail = self._track_display_settings()
        feats = self._track_features(table, color_by, dsl_limit)
        cmap = self.TRACK_COLOR_FEATURES[color_by][1]
        if 'Trajectories' in self.viewer.layers:
            layer = self.viewer.layers['Trajectories']
            keep = table['t'] % int(layer.metadata.get('stride', 1)) == 0
            if int(keep.sum()) != layer.data.shape[0]:       # tracks changed: rebuild
                self.init_tracks_stack(visible=layer.visible)
                layer = self.viewer.layers['Trajectories']
            else:
                layer.features = {k: v[keep] for k, v in feats.items()}
                layer.colormap = cmap
                layer.color_by = color_by
            layer.tail_length = tail
            if show_trajectories is not None:
                layer.visible = bool(show_trajectories)

    #: Click-to-select radius around a sarcomere, in µm
    TRACK_CLICK_RADIUS_UM = 0.6

    def nearest_track(self, world_position):
        """Track id of the sarcomere nearest to a world position ``(t, y, x)`` in the
        current frame, or None when nothing is within ``TRACK_CLICK_RADIUS_UM``.

        A numpy search over the current frame's observed positions, so selecting a
        sarcomere needs no napari layer holding every observation."""
        table = self.track_layer_table()
        cell = self.model.cell
        if table is None or cell is None:
            return None
        pos = np.asarray(world_position, dtype=float)
        if pos.size < 2:
            return None
        scale = np.asarray(cell.scale, dtype=float)
        if pos.size == 3:
            frame = int(round(pos[0] / scale[0]))
            yx = pos[1:] / scale[1:]
        else:
            frame = 0
            yx = pos[-2:] / scale[-2:]
        rows = np.flatnonzero(table['t'] == frame)
        if rows.size == 0:
            return None
        d2 = (table['y'][rows] - yx[0]) ** 2 + (table['x'][rows] - yx[1]) ** 2
        i = int(np.argmin(d2))
        radius_px = self.TRACK_CLICK_RADIUS_UM / (cell.metadata.pixelsize or 1.0)
        if d2[i] > radius_px ** 2:
            return None
        return int(table['track_id'][rows[i]])

    def _on_canvas_click(self, viewer, event):
        """Viewer mouse callback: a click (no drag) on a sarcomere selects it."""
        if self.model.cell is None or 'Trajectories' not in self.viewer.layers:
            return
        start = np.asarray(event.position)
        yield
        while event.type == 'mouse_move':
            yield
        if np.linalg.norm(np.asarray(event.position) - start) > 1e-6:
            return                                             # it was a drag
        track_id = self.nearest_track(event.position)
        if track_id is None:
            return
        for cb in self.track_selected_callbacks:
            cb(track_id)

    def init_draw_loi_layer(self):
        """Add (or focus) a Shapes layer for drawing LOI lines on the movie.

        With 'Group by = loi', any drawn polylines are grouped + ordered along the
        line exactly like auto-detected LOIs (see SarcAsM.group_tracks by='loi')."""
        cell = self.model.cell
        if cell is None:
            return
        if 'DrawnLOIs' in self.viewer.layers:
            layer = self.viewer.layers['DrawnLOIs']
        else:
            layer = self.viewer.add_shapes(name='DrawnLOIs', edge_color='#FF0000',
                                           scale=cell.scale[-2:])
        layer.mode = 'add_path'
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass

    def get_drawn_lois(self):
        """Drawn LOI polylines as a list of (M, 2) yx-pixel arrays, or None if none.

        Must be called on the Qt main thread (reads the napari Shapes layer)."""
        if 'DrawnLOIs' not in self.viewer.layers:
            return None
        layer = self.viewer.layers['DrawnLOIs']
        lines = [np.asarray(p, dtype=float).reshape(-1, 2) for p in layer.data
                 if len(p) >= 2]
        return lines or None

    def init_scale_bar(self):
        # Extract metadata
        frames = self.model.cell.metadata.n_stack
        px = self.model.cell.metadata.pixelsize

        # Unit and base voxel size
        unit = 'µm' if px is not None else 'pixel'
        size = px or 1

        # Build scale tuple: 2D for single frame, 3D otherwise
        if frames == 1:
            scale = (size, size)
        else:
            scale = (1, size, size)
        self.model.cell.scale = scale

        # Apply to all layers, matching scale dimensionality to each layer's ndim
        for layer in self.viewer.layers:
            layer_ndim = layer.ndim
            if layer_ndim == 2:
                layer.scale = (size, size)
            elif layer_ndim == 3:
                layer.scale = (1, size, size)
            else:
                # For higher dimensions, prepend 1s as needed
                layer.scale = (1,) * (layer_ndim - 2) + (size, size)

        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = unit
        self.viewer.reset_view()

    def init_image_stack(self):
        if self.viewer.layers.__contains__('ImageData'):
            layer = self.viewer.layers.__getitem__('ImageData')
            self.viewer.layers.remove(layer)
        tmp = self.model.cell.image
        lower_perc, upper_perc = np.percentile(tmp, q=[0.1, 99.9])
        self.viewer.add_image(tmp, name='ImageData', contrast_limits=[lower_perc, upper_perc],
                              scale=self.model.cell.scale)
        current_index = list(self.viewer.layers).index(self.viewer.layers['ImageData'])
        self.viewer.layers.move(current_index, 0)

    def init_z_band_stack(self, visible=True):
        if self.model.cell is not None:
            tmp = self.model.cell.load_mask_full_stack('zbands')
            if tmp is None:
                return
            if self.viewer.layers.__contains__('ZbandMask'):
                layer = self.viewer.layers.__getitem__('ZbandMask')
                self.viewer.layers.remove(layer)
            tmp[tmp < 0.3] = np.nan  # Higher threshold for crisper edges
            if self.model.cell.metadata.n_stack > 1 and tmp.ndim==2:
                tmp = np.expand_dims(tmp, axis=0)
            self.viewer.add_image(tmp, name='ZbandMask', opacity=0.7, colormap='yellow', blending='additive',
                                  visible=visible, scale=self.model.cell.scale)

    def init_m_band_stack(self, visible=True):
        if self.model.cell is not None:
            tmp = self.model.cell.load_mask_full_stack('mbands')
            if tmp is None:
                return
            if self.viewer.layers.__contains__('MbandMask'):
                layer = self.viewer.layers.__getitem__('MbandMask')
                self.viewer.layers.remove(layer)
            tmp[tmp < 0.1] = np.nan
            if self.model.cell.metadata.n_stack > 1 and tmp.ndim==2:
                tmp = np.expand_dims(tmp, axis=0)
            self.viewer.add_image(tmp, name='MbandMask', opacity=0.5, colormap='blue', blending='additive',
                                  visible=visible, scale=self.model.cell.scale)

    def init_cell_mask_stack(self, visible=True):
        if self.model.cell is not None:
            tmp = self.model.cell.load_mask_full_stack('cell_mask')
            if tmp is None:
                return
            if self.viewer.layers.__contains__('CellMask'):
                layer = self.viewer.layers.__getitem__('CellMask')
                self.viewer.layers.remove(layer)
            tmp[tmp < 0.5] = np.nan
            if self.model.cell.metadata.n_stack > 1 and tmp.ndim==2:
                tmp = np.expand_dims(tmp, axis=0)
            self.viewer.add_image(tmp, name='CellMask', opacity=0.2, colormap='yellow', blending='additive',
                                  visible=visible, scale=self.model.cell.scale)

    def init_z_lateral_connections(self, visible=True):
        cell = self.model.cell
        if cell is None or 'structure.zbands.labels' not in cell.data:
            return
        for name in ('ZbandLatGroups', 'ZbandLatConnections', 'ZbandEnds'):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])

        n_stack = cell.metadata.n_stack
        pixelsize = cell.metadata.pixelsize
        multi_frame = n_stack > 1
        analyzed_frames = cell.data.get('params.analyze_z_bands.frames', [])

        labels_groups = np.zeros((n_stack, *cell.metadata.size), dtype='uint16')
        connections = []
        for frame in range(n_stack):
            if frame not in analyzed_frames or cell.data['structure.zbands.labels'][frame] is None:
                continue
            labels_frame = cell.data['structure.zbands.labels'][frame].toarray()
            groups_frame = cell.data['structure.zbands.lat_groups'][frame]
            labels_groups_frame = np.zeros_like(labels_frame)
            for i, group in enumerate(groups_frame[1:]):
                mask = np.zeros_like(labels_frame, dtype=bool)
                for label in group:
                    mask |= (labels_frame == label + 1)
                labels_groups_frame[mask] = i + 1
            labels_groups[frame] = Utils.shuffle_labels(labels_groups_frame)

            z_ends_frame = np.asarray(cell.data['structure.zbands.ends'][frame], dtype=float) / pixelsize
            z_links_frame = cell.data['structure.zbands.lat_links'][frame]
            if z_links_frame is None or z_links_frame.size == 0:
                continue
            for (i, k, j, l) in z_links_frame.T:
                p0 = z_ends_frame[i, k]
                p1 = z_ends_frame[j, l]
                if multi_frame:
                    connections.append([[frame, p0[0], p0[1]], [frame, p1[0], p1[1]]])
                else:
                    connections.append([[p0[0], p0[1]], [p1[0], p1[1]]])

        labels_data = labels_groups if multi_frame else labels_groups[0]
        labels_scale = (1, pixelsize, pixelsize) if multi_frame else (pixelsize, pixelsize)
        self.viewer.add_labels(labels_data, name='ZbandLatGroups', opacity=0.5,
                               visible=visible, scale=labels_scale)
        ndim = 3 if multi_frame else 2
        self.viewer.add_shapes(connections, name='ZbandLatConnections', shape_type='path',
                               edge_color='white', edge_width=1, opacity=0.5,
                               visible=visible, scale=labels_scale, ndim=ndim)

    def init_myofibril_lines_stack(self, visible=True):
        cell = self.model.cell
        if cell is None or 'structure.myofibril.lines' not in cell.data:
            return
        if 'MyofibrilLines' in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers['MyofibrilLines'])

        n_stack = cell.metadata.n_stack
        pixelsize = cell.metadata.pixelsize
        multi_frame = n_stack > 1

        myof_lines = cell.data['structure.myofibril.lines']
        pos_vectors_px = cell.data['structure.sarcomere.pos_px']
        paths = []
        for frame, (lines_i, pv_i) in enumerate(zip(myof_lines, pos_vectors_px)):
            if lines_i is None or pv_i is None:
                continue
            pv_arr = np.asarray(pv_i, dtype=float)
            if pv_arr.ndim != 2 or pv_arr.shape[1] < 2:
                continue
            for line_j in lines_i:
                idx = np.asarray(line_j, dtype=int).ravel()
                if idx.size == 0:
                    continue
                yx = pv_arr[idx, :2]
                if multi_frame:
                    path = np.column_stack((np.full(idx.size, frame, dtype=float), yx))
                else:
                    path = yx
                paths.append(path)

        scale = (1, pixelsize, pixelsize) if multi_frame else (pixelsize, pixelsize)
        ndim = 3 if multi_frame else 2
        self.viewer.add_shapes(name='MyofibrilLines', data=paths, shape_type='path',
                               edge_color='red', edge_width=2, opacity=0.5,
                               visible=visible, scale=scale, ndim=ndim)

    def init_sarcomere_vector_stack(self, visible=True):
        if self.model.cell is not None and 'structure.sarcomere.pos' in self.model.cell.data.keys():
            if self.viewer.layers.__contains__('SarcomereVectors'):
                layer = self.viewer.layers.__getitem__('SarcomereVectors')
                self.viewer.layers.remove(layer)
                pass
            if self.viewer.layers.__contains__('MidlinePoints'):
                layer = self.viewer.layers.__getitem__('MidlinePoints')
                self.viewer.layers.remove(layer)
                pass
            # create sarcomere vectors for all frames and add as vector layer
            vectors = []
            pos_vectors = []
            for frame in range(self.model.cell.metadata.n_stack):
                if 'params.analyze_sarcomere_vectors.frames' in self.model.cell.data and frame in \
                        self.model.cell.data['params.analyze_sarcomere_vectors.frames'] and self.model.cell.data['structure.sarcomere.pos'][frame] is not None:
                    pos_vectors_frame = self.model.cell.data['structure.sarcomere.pos'][frame] / self.model.cell.metadata.pixelsize
                    if len(pos_vectors_frame) > 0:
                        sarc_orientation_vectors = self.model.cell.data['structure.sarcomere.orientation'][
                            frame]
                        sarc_length_vectors = self.model.cell.data['structure.sarcomere.slen'][frame] / \
                                              self.model.cell.metadata.pixelsize
                        orientation_vectors = np.asarray(
                            [np.sin(sarc_orientation_vectors), np.cos(sarc_orientation_vectors)])
                        for i in range(len(pos_vectors_frame)):
                            start_point = [frame, pos_vectors_frame[i][0], pos_vectors_frame[i][1]]
                            vector_1 = [frame, orientation_vectors[0][i] * sarc_length_vectors[i] * 0.5,
                                        orientation_vectors[1][i] * sarc_length_vectors[i] * 0.5]
                            vector_2 = [frame, -orientation_vectors[0][i] * sarc_length_vectors[i] * 0.5,
                                        -orientation_vectors[1][i] * sarc_length_vectors[i] * 0.5]
                            pos_vectors.append(start_point)
                            vectors.append([start_point, vector_1])
                            vectors.append([start_point, vector_2])
            self.viewer.add_vectors(vectors, edge_width=0.5, edge_color='lightgray', name='SarcomereVectors', opacity=0.8,
                                    vector_style='arrow', visible=visible)
            self.viewer.add_points(name='MidlinePoints', data=pos_vectors, face_color='darkgreen', size=0.2 / self.model.cell.metadata.pixelsize,
                                   visible=visible)
            self.viewer.layers['SarcomereVectors'].scale = self.model.cell.scale
            self.viewer.layers['MidlinePoints'].scale = self.model.cell.scale

    def init_sarcomere_mask_stack(self, visible=True):
        if self.model.cell is not None:
            tmp = self.model.cell.load_mask_full_stack('sarcomere_mask')
            if tmp is None:
                return
            if 'SarcomereMask' in self.viewer.layers:
                layer = self.viewer.layers['SarcomereMask']
                self.viewer.layers.remove(layer)

            if self.model.cell.metadata.n_stack > 1 and tmp.ndim==2:
                tmp = np.expand_dims(tmp, axis=0)

            if tmp.ndim == 2:  # Single image
                rgba_image = np.zeros((tmp.shape[0], tmp.shape[1], 4), dtype='uint8')
                rgba_image[..., 0] = 255  # Red channel
                rgba_image[..., 1] = 255  # Green channel
                rgba_image[..., 2] = 0  # Blue channel
                rgba_image[..., 3] = np.where(tmp > 0.5, 102, 0)  # Alpha channel (40% opacity)
            elif tmp.ndim == 3:  # Stack of images
                rgba_image = np.zeros((tmp.shape[0], tmp.shape[1], tmp.shape[2], 4), dtype='uint8')
                rgba_image[..., 0] = 255  # Red channel
                rgba_image[..., 1] = 255  # Green channel
                rgba_image[..., 2] = 0  # Blue channel
                rgba_image[..., 3] = np.where(tmp > 0.5, 102, 0)  # Alpha channel (40% opacity)

            self.viewer.add_image(rgba_image, name='SarcomereMask', opacity=0.5, blending='translucent',
                                  visible=visible, scale=self.model.cell.scale)

    def init_sarcomere_domain_stack(self, visible=True):
        if self.model.cell is None or 'structure.domain.members' not in self.model.cell.data:
            return
        if 'SarcomereDomains' in self.viewer.layers:
            self.viewer.layers.remove('SarcomereDomains')

        cell = self.model.cell
        total_frames = cell.metadata.n_stack
        size = cell.metadata.size

        _domain_masks = np.zeros((total_frames, *size), dtype='uint16')

        if 'params.analyze_sarcomere_domains.frames' in cell.data:
            frames_to_analyze = [f for f in range(total_frames)
                                 if f in cell.data['params.analyze_sarcomere_domains.frames']]
        else:
            frames_to_analyze = range(total_frames)

        area_min = cell.data.get('params.analyze_sarcomere_domains.area_min')
        dilation_radius = cell.data.get('params.analyze_sarcomere_domains.dilation_radius')
        pixelsize = cell.metadata.pixelsize

        def process_frame(frame, cell, area_min, dilation_radius, pixelsize, size):
            domains = cell.data['structure.domain.members'][frame]
            pos_vectors = cell.data['structure.sarcomere.pos'][frame]
            sarcomere_orientation_vectors = cell.data['structure.sarcomere.orientation'][frame]
            sarcomere_length_vectors = cell.data['structure.sarcomere.slen'][frame]

            domain_mask = analyze_domains(
                domains,
                pos_vectors=pos_vectors,
                sarcomere_length_vectors=sarcomere_length_vectors,
                sarcomere_orientation_vectors=sarcomere_orientation_vectors,
                size=size,
                pixelsize=pixelsize,
                dilation_radius=dilation_radius,
                area_min=area_min
            )[0]

            return frame, domain_mask

        process_frame_partial = partial(
            process_frame,
            cell=cell,
            area_min=area_min,
            dilation_radius=dilation_radius,
            pixelsize=pixelsize,
            size=size
        )

        with ThreadPoolExecutor(max_workers=min(8, len(frames_to_analyze))) as executor:
            for frame, mask in executor.map(process_frame_partial, frames_to_analyze):
                _domain_masks[frame] = mask
        self.viewer.add_labels(_domain_masks, name='SarcomereDomains', opacity=0.35, visible=visible,
                               scale=self.model.cell.scale)

    def run_async_new(self, parameters, call_lambda, start_message, finished_message, finished_action=None,
                      finished_successful_action=None):
        """
        parameters is a dictionary which contains all necessary variables
        call lambda can be a lambda or function (needs to be callable)
        requirement: it needs two function parameters, first is the worker and second is the parameter-dictionary

        and should use the parameters dictionary

        this method should work (tested roughly :D)
        """
        if self.model.currentlyProcessing.get_value():
            logger.warning('Still processing something')
            return

        self.model.currentlyProcessing.set_value(True)
        logger.info(start_message)

        class Worker(QObject):
            finished = pyqtSignal()
            finished_successful = pyqtSignal()
            progress = pyqtSignal(int)
            progress_details = pyqtSignal(str)
            exception = pyqtSignal(str)

            def __init__(self, parameters, call_lambda):
                super().__init__()
                self.succeeded = None
                self.parameters = parameters
                self.call_lambda = call_lambda

            def run(self):
                self.progress.emit(0)
                try:
                    self.call_lambda(self, parameters)
                    self.finished_successful.emit()
                    self.succeeded = True
                except Exception:
                    tb = traceback.format_exc()
                    logger.error(f"Worker exception:\n{tb}")
                    self.succeeded = False
                    self.exception.emit(tb)
                self.progress.emit(100)
                self.finished.emit()

        # Step 2: Create a QThread object
        self.__worker_thread = QThread()
        # Step 3: Create a worker object
        worker = Worker(parameters=parameters, call_lambda=call_lambda)
        # Step 4: Move worker to the thread
        worker.moveToThread(self.__worker_thread)
        # Step 5: Connect signals and slots
        self.__worker_thread.started.connect(worker.run)
        worker.finished.connect(self.__worker_thread.quit)
        if finished_action is not None:
            worker.finished.connect(finished_action)
        if finished_successful_action is not None:
            worker.finished_successful.connect(finished_successful_action)

        worker.finished.connect(worker.deleteLater)
        self.__worker_thread.finished.connect(self.__worker_thread.deleteLater)

        worker.exception.connect(self.debug)
        worker.progress.connect(self.update_progress)
        worker.progress_details.connect(self.debug_replace)
        # Step 6: Start the thread
        self.__worker_thread.start()
        # Final resets

        self.__worker_thread.finished.connect(lambda: self.__finished_task(finished_message))

        return worker

    def worker_thread(self, on_finished):
        if self.__worker_thread is not None:
            self.__worker_thread.finished.connect(on_finished)

    def __finished_task(self, finished_message=None):
        logger.info(finished_message)
        self.model.currentlyProcessing.set_value(False)
