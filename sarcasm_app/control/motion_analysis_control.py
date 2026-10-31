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

"""Track-based motion analysis control.

Workflow: Track sarcomere vectors -> group tracks (pool / m-band / myofibril /
domain / loi) -> analyze track motion (ContractionNet per group). Visualization:
napari track overlays, matplotlib grouped summaries, and a per-fibre LOI-style
detail panel via :meth:`MotionAnalysis.get_track_motion`.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import qtutils
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox
from bio_image_unet.progress import ProgressNotifier

from sarcasm import Plots, SarcAsM
from sarcasm.type_utils import TypeUtils
from .application_control import ApplicationControl
from .chain_execution import ChainExecution
from .popup_export import ExportPopup
from ..model import ApplicationModel, default_min_coverage
from ..view.collapsible import make_collapsible
from ..view.parameters_motion_analysis import Ui_Form as MotionAnalysisWidget
from ..view.track_trace_dock import TrackTraceDock

logger = logging.getLogger(__name__)

_FIBRE_KINDS = ('myofibril', 'loi')


#: width of the motion summary figure (inches)
SUMMARY_WIDTH_IN = 12.0


def summary_layout(aspect: float, is_fibre: bool):
    """Panel layout and row heights (inches) of the motion summary for an image of width /
    height ``aspect``, so the maps keep their proportions for wide, square and tall movies.

    Wide images get one full-width row per map, square ones two maps side by side, tall
    ones every map side by side in one row; a map row is as high as the image is at that
    width, within limits that keep extreme shapes readable.
    """
    def height(width_in, lo, hi):
        return float(np.clip(width_in / aspect, lo, hi))

    traces = [['slen', 'dslen'], ['raster', 'raster']]
    trace_heights = [4.2, 3.6]
    if aspect > 2:                                   # wide: full-width maps
        maps = [['coverage', 'coverage'], ['groups', 'groups']] + ([['fibrils', 'fibrils']] if is_fibre else [])
        layout = maps[:2] + traces + maps[2:]
        h = height(SUMMARY_WIDTH_IN, 1.5, 5.0)
        heights = [h, h] + trace_heights + ([h] if is_fibre else [])
    elif aspect >= 0.8 or not is_fibre:              # square-ish: maps side by side
        layout = [['coverage', 'groups']] + traces + ([['fibrils', 'fibrils']] if is_fibre else [])
        h = height(SUMMARY_WIDTH_IN / 2, 2.5, 8.0)
        heights = [h] + trace_heights + ([height(SUMMARY_WIDTH_IN, 2.5, 8.0)] if is_fibre else [])
    else:                                            # tall: the three maps in one row (6-column grid)
        layout = [['coverage'] * 2 + ['groups'] * 2 + ['fibrils'] * 2,
                  ['slen'] * 3 + ['dslen'] * 3,
                  ['raster'] * 6]
        heights = [height(SUMMARY_WIDTH_IN / 3, 3.0, 9.0)] + trace_heights
    return layout, heights


def _pv(model: ApplicationModel, name: str):
    """Shortcut: read a bound parameter value off the model."""
    return model.parameters.get_parameter(name).get_value()


def detect_lois_kwargs(get) -> dict:
    """``MotionAnalysis.detect_lois`` keyword arguments from the ``loi.detect.*`` parameters.

    ``get(name)`` returns a parameter value. Shared by the interactive Motion tab and
    the batch run so both use the same tuned LOI detection.
    """
    return dict(
        n_lois=int(get('loi.detect.n_lois')),
        ratio_seeds=get('loi.detect.ratio_seeds'),
        persistence=int(get('loi.detect.persistence')),
        threshold_distance=get('loi.detect.threshold_distance'),
        mode=get('loi.detect.mode'),
        number_lims=(int(get('loi.detect.number_limits_lower')),
                     int(get('loi.detect.number_limits_upper'))),
        length_lims=(get('loi.detect.length_limits_lower'),
                     get('loi.detect.length_limits_upper')),
        distance_threshold_lois=get('loi.detect.cluster_threshold_lois'),
        linkage=get('loi.detect.linkage'))


class MotionAnalysisControl:
    """Handles the track-based Motion tab."""

    def __init__(self, motion_analysis_widget: MotionAnalysisWidget, main_control: ApplicationControl):
        self.__trace_dock: Optional[TrackTraceDock] = None
        self.__motion_analysis_widget = motion_analysis_widget
        self.__main_control = main_control
        self.__worker = None
        self.__export_popup = None
        self.__fibre_motion = None
        self.__fibre_group = None
        self.__drawn_lois = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def __chk_initialized(self) -> bool:
        if not self.__main_control.model.is_initialized():
            logger.warning('File is not correctly initialized (or viewer was closed)')
            return False
        return True

    def __chk_movie(self) -> bool:
        """Motion analysis needs a movie; say so instead of failing deep in tracking."""
        if self.__main_control.model.cell.metadata.n_stack < 2:
            QMessageBox.warning(self.__motion_analysis_widget.btn_track_vectors, 'Motion analysis needs a movie',
                                'This file is a single image, so there is no motion to analyse. '
                                'If it is a movie, check the Axes field (e.g. TYX) and press '
                                '"Save metadata".')
            return False
        return True

    def __chk_all_frames(self) -> bool:
        """Tracking follows only the frames with sarcomere vectors: ask before tracking a part of the movie."""
        cell = self.__main_control.model.cell
        frames = cell.data.get('params.structure.analyze_vectors.frames')
        n_stack = int(cell.metadata.n_stack or 1)
        if frames is None or len(frames) >= n_stack:
            return True
        answer = QMessageBox.question(
            self.__motion_analysis_widget.btn_track_vectors, 'Track only part of the movie?',
            f'The structure analysis covered {len(frames)} of {n_stack} frames, so the tracking can follow the '
            f'sarcomeres only in those frames.\n\nTo analyse the whole movie, click No, select "All frames" in '
            f'the Structure tab and run the structure analysis again.\n\nTrack the {len(frames)} frames anyway?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def __get_progress_notifier(self, worker) -> ProgressNotifier:
        progress_notifier = ProgressNotifier()

        def __internal_function(p):
            qtutils.inmain(lambda: self.__main_control.update_progress(int(p * 100)))

        progress_notifier.set_progress_report(__internal_function)
        progress_notifier.set_progress_detail(
            lambda hh_c, mm_c, ss_c, hh_e, mm_e, ss_e: worker.progress_details.emit(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (hh_c, mm_c, ss_c, hh_e, mm_e, ss_e)))
        return progress_notifier

    # ------------------------------------------------------------------ #
    # step 1: track sarcomere vectors
    # ------------------------------------------------------------------ #
    def __track_call(self, worker, m: ApplicationModel):
        pn = self.__get_progress_notifier(worker)
        cell: SarcAsM = TypeUtils.unbox(m.cell)
        cell.motion.track(
            frames='all',
            max_disp_along_um=_pv(m, 'motion.track.max_disp_along'),
            max_disp_perp_um=_pv(m, 'motion.track.max_disp_perp'),
            ori_tol_deg=_pv(m, 'motion.track.ori_tol'),
            min_track_duration_s=_pv(m, 'motion.track.min_duration_s'),
            max_gap_interpolation_s=float(_pv(m, 'motion.track.max_gap_interp_s')),
            motion_predictor='flow' if bool(_pv(m, 'motion.track.flow_predictor')) else 'none',
            refine_lengths=bool(_pv(m, 'motion.track.refine_lengths')),
            progress_notifier=pn)

    def on_btn_track_vectors(self):
        if not self.__chk_initialized() or not self.__chk_movie() or not self.__chk_all_frames():
            return
        cell: SarcAsM = TypeUtils.unbox(self.__main_control.model.cell)
        worker = self.__main_control.run_async_new(
            parameters=self.__main_control.model,
            call_lambda=self.__track_call,
            start_message='Tracking sarcomere vectors…',
            finished_message='Finished tracking sarcomere vectors',
            finished_action=self.__tracks_finished,
            finished_successful_action=cell.commit)
        self.__worker = worker
        return worker

    # ------------------------------------------------------------------ #
    # step 2: group tracks
    # ------------------------------------------------------------------ #
    def __group_call(self, worker, m: ApplicationModel):
        cell: SarcAsM = TypeUtils.unbox(m.cell)
        by = _pv(m, 'motion.group.by')
        ref = int(_pv(m, 'motion.group.reference_frame'))
        cov = _pv(m, 'motion.group.min_coverage')
        min_size = int(_pv(m, 'motion.group.min_group_size'))
        if by == 'loi':
            if self.__drawn_lois:
                # user-drawn lines (yx-px polylines) feed the same loi grouping
                cell.data['motion.loi.data'] = {'loi_lines': self.__drawn_lois}
            else:
                cell.motion.detect_lois(frame=ref, **detect_lois_kwargs(lambda name: _pv(m, name)))
        cell.motion.group_tracks(by=by, reference_frame=ref, min_coverage=cov, min_group_size=min_size)

    def on_btn_draw_lois(self):
        if not self.__chk_initialized():
            return
        self.__main_control.init_draw_loi_layer()

    def __update_loi_controls_visibility(self, *_):
        """Show the LOI draw button + auto-detect settings only for 'loi' grouping."""
        is_loi = self.__motion_analysis_widget.cb_group_by.currentText() == 'loi'
        self.__motion_analysis_widget.btn_draw_lois.setVisible(is_loi)
        self.__motion_analysis_widget.groupBox_loi_detect.setVisible(is_loi)

    def on_btn_group_tracks(self):
        if not self.__chk_initialized():
            return
        m = self.__main_control.model
        by = m.parameters.get_parameter('motion.group.by').get_value()
        # read any drawn LOI polylines on the main thread before the worker starts
        self.__drawn_lois = self.__main_control.get_drawn_lois() if by == 'loi' else None
        cell: SarcAsM = TypeUtils.unbox(m.cell)
        worker = self.__main_control.run_async_new(
            parameters=self.__main_control.model,
            call_lambda=self.__group_call,
            start_message='Grouping tracks…',
            finished_message='Finished grouping tracks',
            finished_action=self.__grouping_finished,
            finished_successful_action=cell.commit)
        self.__worker = worker
        return worker

    # ------------------------------------------------------------------ #
    # step 3: analyze track motion
    # ------------------------------------------------------------------ #
    def __analyze_call(self, worker, m: ApplicationModel):
        cell: SarcAsM = TypeUtils.unbox(m.cell)
        cell.motion.analyze(
            aggregate=str(_pv(m, 'motion.analyze.aggregate')),
            slen_lims=(_pv(m, 'motion.analyze.slen_lower'), _pv(m, 'motion.analyze.slen_upper')),
            threshold=_pv(m, 'motion.analyze.threshold'),
            contr_time_min=_pv(m, 'motion.analyze.contr_time_min'),
            buffer_frames=int(_pv(m, 'motion.analyze.buffer_frames')),
            min_valid_frames=_pv(m, 'motion.analyze.min_valid_frames'),
            filter_params=(int(_pv(m, 'motion.analyze.filter_wl')),
                           int(_pv(m, 'motion.analyze.filter_po'))))

    def on_btn_analyze_track_motion(self):
        if not self.__chk_initialized():
            return
        cell: SarcAsM = TypeUtils.unbox(self.__main_control.model.cell)
        worker = self.__main_control.run_async_new(
            parameters=self.__main_control.model,
            call_lambda=self.__analyze_call,
            start_message='Analyzing track motion…',
            finished_message='Finished analyzing track motion',
            finished_action=self.__refresh_trace_source,
            finished_successful_action=cell.commit)
        self.__worker = worker
        return worker

    # ------------------------------------------------------------------ #
    # master: track -> group -> analyze
    # ------------------------------------------------------------------ #
    def on_analyze_motion(self):
        if not self.__chk_initialized() or not self.__chk_movie() or not self.__chk_all_frames():
            return
        self.__main_control.raise_viewer()
        chain = ChainExecution(self.__main_control.model.currentlyProcessing, self.__main_control.debug)
        chain.add_step(self.on_btn_track_vectors)
        chain.add_step(self.on_btn_group_tracks)
        chain.add_step(self.on_btn_analyze_track_motion)
        chain.execute()

    # ------------------------------------------------------------------ #
    # finished-actions (run on the Qt main thread): refresh napari overlays
    # ------------------------------------------------------------------ #
    def __tracks_finished(self):
        self.__main_control.init_sarcomere_dots()
        self.__sync_display_visibility()
        self.__refresh_trace_source()

    def __grouping_finished(self):
        # the tracks are unchanged: refresh the features (group ids) in place instead of
        # rebuilding the layers, which takes seconds for millions of vertices
        self.__main_control.highlight_group(None)
        self.__main_control.init_track_groups_layer()
        self.__sync_display_visibility()
        self.__refresh_trace_source()
        self.refresh_fibre_combo()

    def refresh_fibre_combo(self):
        """Fill the per-fibre combo from the grouping in the store (also on file open)."""
        w = self.__motion_analysis_widget
        cell = self.__main_control.model.cell
        w.cb_fibre_group.clear()
        kind = cell.data.get('motion.groups.kind') if cell is not None else None
        n_groups = int(cell.data.get('motion.groups.n', 0)) if cell is not None else 0
        is_fibre = kind in _FIBRE_KINDS and n_groups > 0
        if is_fibre:
            w.cb_fibre_group.addItems([str(g) for g in range(n_groups)])
        w.cb_fibre_group.setEnabled(is_fibre)
        w.btn_show_fibre_detail.setEnabled(is_fibre)

    # ------------------------------------------------------------------ #
    # per-fibre detail panel (LOI-style plots via sarc.motion.get_track_motion)
    # ------------------------------------------------------------------ #
    def __fibre_call(self, worker, m: ApplicationModel):
        cell: SarcAsM = TypeUtils.unbox(m.cell)
        group = int(self.__motion_analysis_widget.cb_fibre_group.currentText())
        motion = cell.motion.get_track_motion(group, analyze=True)
        try:
            motion.analyze_popping()
        except Exception as e:
            logger.debug(f'analyze_popping skipped: {e}')
        self.__fibre_motion = motion
        self.__fibre_group = group

    def on_btn_show_fibre_detail(self):
        if not self.__chk_initialized():
            return
        cell = self.__main_control.model.cell
        kind = cell.data.get('motion.groups.kind') if cell is not None else None
        if kind not in _FIBRE_KINDS:
            logger.warning("Per-fibre detail requires a 'myofibril' or 'loi' grouping.")
            return
        if self.__motion_analysis_widget.cb_fibre_group.currentText() == '':
            logger.warning('No fibre group selected.')
            return
        worker = self.__main_control.run_async_new(
            parameters=self.__main_control.model,
            call_lambda=self.__fibre_call,
            start_message='Building per-fibre motion…',
            finished_message='Finished per-fibre motion',
            finished_action=self.__render_fibre_detail)
        self.__worker = worker
        return worker

    def __render_fibre_detail(self):
        import matplotlib.pyplot as plt
        from sarcasm_app.view.figure_window import show_new_figures
        motion = self.__fibre_motion
        if motion is None:
            return
        before = plt.get_fignums()
        cell = self.__main_control.model.cell
        kind = cell.data.get('motion.groups.kind', '') if cell is not None else ''
        fig, axd = plt.subplot_mosaic([['zpos', 'phase'], ['dslen', 'dslen']],
                                      figsize=(12, 7), constrained_layout=True)
        for key, fn in (('zpos', lambda ax: Plots.plot_z_pos(ax, motion)),
                        ('phase', lambda ax: Plots.plot_phase_space(ax, motion)),
                        ('dslen', lambda ax: Plots.plot_delta_slen(ax, motion))):
            try:
                fn(axd[key])
            except Exception as e:
                logger.warning(f'fibre-detail {key} plot failed: {e}')
        fig.suptitle(f'Fibre {self.__fibre_group} ({kind})')
        try:
            Plots.plot_popping_events(motion)  # opens its own figure
        except Exception as e:
            logger.debug(f'plot_popping_events skipped: {e}')
        show_new_figures(before, f'Fibre {self.__fibre_group}')

    # ------------------------------------------------------------------ #
    # grouped-motion summary
    # ------------------------------------------------------------------ #
    def __on_btn_plot_summary(self):
        if not self.__chk_initialized():
            return
        import matplotlib.pyplot as plt
        from sarcasm_app.view.figure_window import show_new_figures
        before = plt.get_fignums()
        cell = self.__main_control.model.cell
        kind = cell.data.get('motion.groups.analyzed_kind') if cell is not None else None
        if kind is None:
            logger.warning('No track motion analyzed yet — run "Analyze track motion" first.')
            return
        current = cell.data.get('motion.groups.kind')
        if current != kind or (cell.data.get('params.motion.analyze.grouping_hash')
                               != cell.data.get('motion.groups.hash')):
            # the tracks were grouped again after the analysis: the summary would mix both groupings
            QMessageBox.information(self.__motion_analysis_widget.btn_analyze_track_motion, 'Analyze the new grouping first',
                                    f'The tracks were grouped again ({current}) after the last motion analysis '
                                    f'({kind}). Click "3. Analyze track motion" (or "Track → Group → Analyze"), '
                                    f'then plot the summary.')
            return
        is_fibre = kind in _FIBRE_KINDS
        # raster of every sarcomere over one averaged cycle (needs the pooled cycles),
        # else over the full recording
        cycle = 'motion.pool.labels_contr' in cell.data
        H, W = cell.metadata.shape[-2:]
        layout, heights = summary_layout(W / H, is_fibre)
        fig, axd = plt.subplot_mosaic(layout, figsize=(SUMMARY_WIDTH_IN, sum(heights)),
                                      height_ratios=heights, constrained_layout=True)
        def raster(ax):
            if cycle:
                try:
                    Plots.plot_track_raster(ax, cell, cycle_average=True, sort_by='time_to_peak',
                                            title='ΔSL of every sarcomere over the averaged cycle, sorted by time to peak')
                    return
                except ValueError as e:                      # e.g. fewer than two complete cycles
                    logger.debug(f'cycle raster skipped: {e}')
                    ax.clear()
            Plots.plot_track_raster(ax, cell, cycle_average=False,
                                    title='ΔSL of every sarcomere over the recording, by group')

        panels = (('coverage', lambda ax: Plots.plot_track_coverage_map(ax, cell)),
                  ('groups', lambda ax: Plots.plot_track_groups(ax, cell)),
                  ('slen', lambda ax: Plots.plot_slen_mean(ax, cell, n_rows=8, readable_labels=True)),
                  ('dslen', lambda ax: Plots.plot_delta_slen_mean(ax, cell, n_rows=8, readable_labels=True)),
                  ('raster', raster))
        for key, fn in panels:
            try:
                fn(axd[key])
            except Exception as e:
                logger.warning(f'summary {key} plot failed: {e}')
        if is_fibre:
            try:
                Plots.plot_track_myofibrils(axd['fibrils'], cell, color_by='beating_rate', colorbar=True)
            except Exception as e:
                logger.warning(f'summary fibrils plot failed: {e}')
        fig.suptitle(f'{Path(cell.file_path).name} — {kind}')
        show_new_figures(before, 'Motion summary')

    # ------------------------------------------------------------------ #
    # display of the tracks in the viewer + time-series panel + raster
    # ------------------------------------------------------------------ #
    def __display_visibility(self):
        m = self.__main_control.model
        return (bool(_pv(m, 'motion.display.show_sarcomeres')),
                bool(_pv(m, 'motion.display.show_groups')))

    def __sync_display_visibility(self):
        show_sarc, show_groups = self.__display_visibility()
        self.__main_control.apply_track_display(show_sarcomeres=show_sarc)
        layers = self.__main_control.viewer.layers
        if 'Groups' in layers:
            groups = layers['Groups']
            groups.selected_data = set()      # see ApplicationControl._on_group_click
            groups.visible = show_groups

    def __on_display_changed(self):
        if self.__main_control.model.cell is None:
            return
        try:
            self.__sync_display_visibility()
        except Exception as e:
            logger.warning(f'Updating the track display failed: {e}')

    def __open_trace_dock(self):
        if not self.__chk_initialized():
            return None
        if self.__trace_dock is None:
            self.__trace_dock = TrackTraceDock()
            viewer = self.__main_control.viewer
            viewer.window.add_dock_widget(self.__trace_dock, name='Sarcomere time series', area='bottom')
            viewer.dims.events.current_step.connect(
                lambda event: self.__trace_dock.set_frame(int(viewer.dims.current_step[0])))
        self.__refresh_trace_source()
        return self.__trace_dock

    def __refresh_trace_source(self):
        if self.__trace_dock is None:
            return
        cell = self.__main_control.model.cell
        kin = self.__main_control.track_kinematics()
        if cell is None or kin is None:
            self.__trace_dock.set_source(None, None, 1.0)
            return
        gid = cell.data.get('motion.tracks.group_id')
        kind = cell.data.get('motion.groups.kind', '')
        # each group's own ContractionNet intervals, if the current grouping was analysed
        group_contr = None
        if kind and cell.data.get(f'params.motion.analyze.grouping_hash_{kind}') == cell.data.get('motion.groups.hash'):
            group_contr = cell.data.get(f'motion.{kind}.contr')
            group_contr = None if group_contr is None else np.asarray(group_contr)
        self.__trace_dock.export_dir = os.path.dirname(cell.file_path)
        self.__trace_dock.export_stem = Path(cell.file_path).stem
        self.__trace_dock.set_source(kin, gid, cell.metadata.frametime, kind, group_contr)

    def __on_track_selected(self, track_id: int):
        """A sarcomere was clicked: select its group, highlight it, show its overlay."""
        cell = self.__main_control.model.cell
        gid = cell.data.get('motion.tracks.group_id') if cell is not None else None
        group = None
        if gid is not None:
            gid = np.asarray(gid).reshape(-1)
            if track_id < gid.size and gid[track_id] >= 0:
                group = int(gid[track_id])
        self.__select_group(group)
        # the ring is painted first; the panel follows on the next event-loop turn
        QTimer.singleShot(0, lambda: self.__show_in_dock(track=track_id))

    def __on_group_selected(self, group: int):
        """A fibre path of the Groups layer was clicked."""
        self.__select_group(group)
        QTimer.singleShot(0, lambda: self.__show_in_dock(group=group))

    def __show_in_dock(self, track=None, group=None):
        dock = self.__open_trace_dock()
        if dock is None:
            return
        if track is not None:
            dock.show_track(track)
        else:
            dock.show_group(group)

    def __select_group(self, group):
        """Ring the group's members in the viewer and point the per-fibre combo at it."""
        try:
            self.__main_control.highlight_group(group)
        except Exception as e:
            logger.debug(f'highlight_group skipped: {e}')
        w = self.__motion_analysis_widget
        if group is not None and w.cb_fibre_group.isEnabled():
            idx = w.cb_fibre_group.findText(str(group))
            if idx >= 0:
                w.cb_fibre_group.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    def __on_btn_export_motion_data(self):
        if not self.__chk_initialized():
            return
        cell = self.__main_control.model.cell
        stem = Path(cell.file_path).stem         # the popup appends the chosen grouping
        self.__export_popup = ExportPopup(self.__main_control.model, self.__main_control,
                                          popup_type='motion', filename_stem=stem)
        self.__export_popup.show_popup()

    # ------------------------------------------------------------------ #
    # event binding
    # ------------------------------------------------------------------ #
    def bind_events(self):
        w = self.__motion_analysis_widget
        # combobox items must exist before the parameter default is pushed via setCurrentText
        w.cb_group_by.addItems(['pool', 'mband', 'myofibril', 'domain', 'loi'])
        w.cb_motion_aggregate.addItems(['mean', 'median'])
        w.cb_loi_mode.addItems(['longest_in_cluster', 'fit_straight_line',
                                'random_from_cluster', 'random_line'])
        w.cb_loi_linkage.addItems(['single', 'complete', 'average'])

        w.btn_track_vectors.clicked.connect(self.on_btn_track_vectors)
        w.btn_group_tracks.clicked.connect(self.on_btn_group_tracks)
        w.btn_draw_lois.clicked.connect(self.on_btn_draw_lois)
        w.btn_analyze_track_motion.clicked.connect(self.on_btn_analyze_track_motion)
        w.btn_analyze_motion.clicked.connect(self.on_analyze_motion)
        w.btn_show_fibre_detail.clicked.connect(self.on_btn_show_fibre_detail)
        w.btn_plot_summary.clicked.connect(self.__on_btn_plot_summary)
        w.btn_export_motion_data.clicked.connect(self.__on_btn_export_motion_data)
        w.btn_show_fibre_detail.setEnabled(False)
        self.__main_control.file_opened_callbacks.append(self.refresh_fibre_combo)
        self.__main_control.file_opened_callbacks.append(self.__sync_display_visibility)
        self.__main_control.track_selected_callbacks.append(self.__on_track_selected)
        self.__main_control.group_selected_callbacks.append(self.__on_group_selected)

        # display of the tracks in the viewer
        w.cb_display_color_by.addItems(list(ApplicationControl.TRACK_COLOR_LABELS))

        p = self.__main_control.model.parameters.get_parameter
        # track
        p('motion.track.max_disp_along').connect(w.dsb_track_max_disp_along)
        p('motion.track.max_disp_perp').connect(w.dsb_track_max_disp_perp)
        p('motion.track.ori_tol').connect(w.dsb_track_ori_tol)
        p('motion.track.min_duration_s').connect(w.dsb_track_min_duration_s)
        p('motion.track.max_gap_interp_s').connect(w.dsb_track_max_gap_interp_s)
        p('motion.track.flow_predictor').connect(w.chk_track_flow_predictor)
        p('motion.track.refine_lengths').connect(w.chk_track_refine_lengths)
        # group
        p('motion.group.by').connect(w.cb_group_by)
        p('motion.group.reference_frame').connect(w.sb_group_reference_frame)
        p('motion.group.min_coverage').connect(w.dsb_group_min_coverage)
        p('motion.group.min_group_size').connect(w.sb_group_min_group_size)
        # the coverage floor follows the grouping, like MotionAnalysis.group_tracks(min_coverage=None)
        w.cb_group_by.currentTextChanged.connect(
            lambda by: w.dsb_group_min_coverage.setValue(default_min_coverage(by)) if by else None)
        # display
        p('motion.display.color_by').connect(w.cb_display_color_by)
        p('motion.display.dsl_limit').connect(w.dsb_display_dsl_limit)
        p('motion.display.show_sarcomeres').connect(w.chk_display_sarcomeres)
        p('motion.display.show_groups').connect(w.chk_display_groups)
        # the parameter bindings above record the values; these apply them to the layers
        w.cb_display_color_by.currentTextChanged.connect(lambda _: self.__on_display_changed())
        w.dsb_display_dsl_limit.valueChanged.connect(lambda _: self.__on_display_changed())
        w.chk_display_sarcomeres.toggled.connect(lambda _: self.__on_display_changed())
        w.chk_display_groups.toggled.connect(lambda _: self.__on_display_changed())
        # LOI auto-detection (advanced, collapsed by default)
        make_collapsible(w.groupBox_loi_detect)
        p('loi.detect.n_lois').connect(w.sb_loi_n_lois)
        p('loi.detect.ratio_seeds').connect(w.dsb_loi_ratio_seeds)
        p('loi.detect.persistence').connect(w.sb_loi_persistence)
        p('loi.detect.threshold_distance').connect(w.dsb_loi_threshold_distance)
        p('loi.detect.mode').connect(w.cb_loi_mode)
        p('loi.detect.number_limits_lower').connect(w.sb_loi_number_lower)
        p('loi.detect.number_limits_upper').connect(w.sb_loi_number_upper)
        p('loi.detect.length_limits_lower').connect(w.dsb_loi_length_lower)
        p('loi.detect.length_limits_upper').connect(w.dsb_loi_length_upper)
        p('loi.detect.cluster_threshold_lois').connect(w.dsb_loi_cluster_threshold)
        p('loi.detect.linkage').connect(w.cb_loi_linkage)

        # LOI controls (draw button + auto-detect group) only show for 'loi' grouping
        w.cb_group_by.currentTextChanged.connect(self.__update_loi_controls_visibility)
        self.__update_loi_controls_visibility()
        # analyze
        p('motion.analyze.aggregate').connect(w.cb_motion_aggregate)
        p('motion.analyze.threshold').connect(w.dsb_motion_threshold)
        p('motion.analyze.contr_time_min').connect(w.dsb_motion_contr_time_min)
        p('motion.analyze.merge_time_max').connect(w.dsb_motion_merge_time_max)
        p('motion.analyze.buffer_frames').connect(w.sb_motion_buffer_frames)
        p('motion.analyze.min_valid_frames').connect(w.dsb_motion_min_valid_frames)
        p('motion.analyze.filter_wl').connect(w.sb_motion_filter_wl)
        p('motion.analyze.filter_po').connect(w.sb_motion_filter_po)
        # the Savitzky-Golay polyorder must stay below the window length
        w.sb_motion_filter_wl.valueChanged.connect(lambda wl: w.sb_motion_filter_po.setMaximum(min(9, wl - 1)))
        w.sb_motion_filter_po.setMaximum(min(9, w.sb_motion_filter_wl.value() - 1))
        p('motion.analyze.slen_lower').connect(w.dsb_motion_slen_lower)
        p('motion.analyze.slen_upper').connect(w.dsb_motion_slen_upper)
