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

    def init_tracks_stack(self, visible=True):
        """Draw sarcomere trajectories as lines plus a dot at each timepoint.

        Each track becomes one polyline (its full trajectory) in a Shapes
        layer, and a sibling Points layer marks the sampled position at every
        timepoint. Coloured per track id, or by group once
        :meth:`SarcAsM.group_tracks` has run (matching the ``TrackGroups``
        point colours). Both are static 2D overlays, so each trajectory reads
        as a continuous line across the whole movie rather than a fading tail."""
        cell = self.model.cell
        if cell is None or 'tracks_positions_px' not in cell.data:
            return
        if cell.metadata.n_stack is None or cell.metadata.n_stack <= 1:
            return  # tracking requires a time series
        for name in ('Tracks', 'TrackPoints'):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
        pos = np.asarray(cell.data['tracks_positions_px'], dtype=float)  # (N, T, 2) yx, px
        if pos.ndim != 3 or pos.shape[0] == 0:
            return
        n_tracks = pos.shape[0]
        mask = np.isfinite(pos[..., 0]) & np.isfinite(pos[..., 1])
        snapped = cell.data.get('tracks_snapped')
        if snapped is not None:
            snapped = np.asarray(snapped)
            if snapped.shape == mask.shape:
                mask &= snapped.astype(bool)
        # colour by group when a grouping exists; drop unassigned (gid < 0) tracks
        gid = cell.data.get('track_group_id')
        gid = None if gid is None else np.asarray(gid).reshape(-1)
        if gid is not None and gid.shape[0] == n_tracks:
            mask &= (gid >= 0)[:, None]
        else:
            gid = None

        import matplotlib.pyplot as plt
        if gid is not None:
            # match TrackGroups colouring so lines and group dots agree
            n_groups = max(1, int(cell.data.get('n_groups', int(gid.max()) + 1)))
            track_colors = plt.get_cmap('gist_rainbow')((gid % n_groups) / n_groups)
        else:
            track_colors = plt.get_cmap('turbo')(np.arange(n_tracks) / max(1.0, n_tracks - 1))

        paths, path_colors, dot_pts, dot_colors = [], [], [], []
        for i in range(n_tracks):
            ts = np.nonzero(mask[i])[0]  # timepoints present, ascending
            if ts.size == 0:
                continue
            yx = pos[i, ts, :2]
            if ts.size >= 2:  # a path needs at least two vertices
                paths.append(yx)
                path_colors.append(track_colors[i])
            dot_pts.append(yx)
            dot_colors.append(np.broadcast_to(track_colors[i], (ts.size, 4)))
        if not dot_pts:
            return

        scale2d = cell.scale[-2:]  # static y/x overlay, broadcast across all frames
        if paths:
            self.viewer.add_shapes(paths, name='Tracks', shape_type='path',
                                   edge_color=np.asarray(path_colors), edge_width=1.5,
                                   opacity=0.8, visible=visible, scale=scale2d, ndim=2)
        self.viewer.add_points(np.concatenate(dot_pts), name='TrackPoints',
                               face_color=np.concatenate(dot_colors), border_width=0,
                               size=max(1.5, 0.2 / cell.metadata.pixelsize),
                               opacity=0.9, visible=visible, scale=scale2d)

    def init_track_groups_stack(self, visible=True):
        """napari Points layer of tracked sarcomeres colored by their group."""
        cell = self.model.cell
        if cell is None or 'tracks_positions_px' not in cell.data or 'track_group_id' not in cell.data:
            return
        if 'TrackGroups' in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers['TrackGroups'])
        pos = np.asarray(cell.data['tracks_positions_px'], dtype=float)  # (N, T, 2) yx, px
        gid = np.asarray(cell.data['track_group_id'])                    # (N,)
        if pos.ndim != 3 or pos.shape[0] == 0:
            return
        if gid.shape[0] != pos.shape[0]:
            return  # stale grouping (track count changed); skip until re-grouped
        multi_frame = cell.metadata.n_stack is not None and cell.metadata.n_stack > 1
        mask = np.isfinite(pos[..., 0]) & np.isfinite(pos[..., 1])       # (N, T)
        snapped = cell.data.get('tracks_snapped')
        if snapped is not None:
            snapped = np.asarray(snapped)
            if snapped.shape == mask.shape:
                mask &= snapped.astype(bool)
        mask &= (gid >= 0)[:, None]
        ii, tt = np.nonzero(mask)
        if ii.size == 0:
            return
        ys, xs = pos[ii, tt, 0], pos[ii, tt, 1]
        points = np.column_stack([tt.astype(float), ys, xs]) if multi_frame else np.column_stack([ys, xs])
        n_groups = max(1, int(cell.data.get('n_groups', int(gid.max()) + 1)))
        import matplotlib.pyplot as plt
        face = plt.get_cmap('gist_rainbow')((gid[ii] % n_groups) / n_groups)
        self.viewer.add_points(points, name='TrackGroups', face_color=face,
                               size=max(2.0, 0.3 / cell.metadata.pixelsize), visible=visible)
        self.viewer.layers['TrackGroups'].scale = cell.scale

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

    def init_z_band_stack(self, visible=True, fastmovie=False):
        if fastmovie and not self.model.cell._mask_exists('zbands_fast_movie'):
            fastmovie = False
        if self.model.cell is not None:
            tmp = self.model.cell.load_mask_full_stack(
                'zbands_fast_movie' if fastmovie else 'zbands')
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
        if cell is None or 'z_labels' not in cell.data:
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
            if frame not in analyzed_frames or cell.data['z_labels'][frame] is None:
                continue
            labels_frame = cell.data['z_labels'][frame].toarray()
            groups_frame = cell.data['z_lat_groups'][frame]
            labels_groups_frame = np.zeros_like(labels_frame)
            for i, group in enumerate(groups_frame[1:]):
                mask = np.zeros_like(labels_frame, dtype=bool)
                for label in group:
                    mask |= (labels_frame == label + 1)
                labels_groups_frame[mask] = i + 1
            labels_groups[frame] = Utils.shuffle_labels(labels_groups_frame)

            z_ends_frame = np.asarray(cell.data['z_ends'][frame], dtype=float) / pixelsize
            z_links_frame = cell.data['z_lat_links'][frame]
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
        if cell is None or 'myof_lines' not in cell.data:
            return
        if 'MyofibrilLines' in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers['MyofibrilLines'])

        n_stack = cell.metadata.n_stack
        pixelsize = cell.metadata.pixelsize
        multi_frame = n_stack > 1

        myof_lines = cell.data['myof_lines']
        pos_vectors_px = cell.data['pos_vectors_px']
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
        if self.model.cell is not None and 'pos_vectors' in self.model.cell.data.keys():
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
                        self.model.cell.data['params.analyze_sarcomere_vectors.frames'] and self.model.cell.data['pos_vectors'][frame] is not None:
                    pos_vectors_frame = self.model.cell.data['pos_vectors'][frame] / self.model.cell.metadata.pixelsize
                    if len(pos_vectors_frame) > 0:
                        sarc_orientation_vectors = self.model.cell.data['sarcomere_orientation_vectors'][
                            frame]
                        sarc_length_vectors = self.model.cell.data['sarcomere_length_vectors'][frame] / \
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
        if self.model.cell is None or 'domains' not in self.model.cell.data:
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
            domains = cell.data['domains'][frame]
            pos_vectors = cell.data['pos_vectors'][frame]
            sarcomere_orientation_vectors = cell.data['sarcomere_orientation_vectors'][frame]
            sarcomere_length_vectors = cell.data['sarcomere_length_vectors'][frame]

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

        # todo: this message gets "eaten" by the last progress_details (depends which is called first)
        self.__worker_thread.finished.connect(lambda: self.__finished_task(finished_message))

        return worker

    def worker_thread(self, on_finished):
        if self.__worker_thread is not None:
            self.__worker_thread.finished.connect(on_finished)

    def __finished_task(self, finished_message=None):
        logger.info(finished_message)
        self.model.currentlyProcessing.set_value(False)
