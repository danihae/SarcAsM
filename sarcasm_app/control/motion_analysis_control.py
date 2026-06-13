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

"""Track-based motion analysis control (replaces the removed LOI workflow).

Workflow: Track sarcomere vectors -> group tracks (pool / m-band / myofibril /
domain / loi) -> analyze track motion (ContractionNet per group). Visualization:
napari track overlays, matplotlib grouped summaries, and a per-fibre LOI-style
detail panel via :meth:`SarcAsM.get_track_motion`.
"""

import logging
from pathlib import Path

import qtutils
from bio_image_unet.progress import ProgressNotifier

from sarcasm import Plots, SarcAsM
from sarcasm.type_utils import TypeUtils
from .application_control import ApplicationControl
from .chain_execution import ChainExecution
from .popup_export import ExportPopup
from ..model import ApplicationModel
from ..view.parameters_motion_analysis import Ui_Form as MotionAnalysisWidget

logger = logging.getLogger(__name__)

_FIBRE_KINDS = ('myofibril', 'loi')


def _pv(model: ApplicationModel, name: str):
    """Shortcut: read a bound parameter value off the model."""
    return model.parameters.get_parameter(name).get_value()


class MotionAnalysisControl:
    """Handles the track-based Motion tab."""

    def __init__(self, motion_analysis_widget: MotionAnalysisWidget, main_control: ApplicationControl):
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
        cell.track_sarcomere_vectors(
            frames='all',
            threshold_mbands=_pv(m, 'motion.track.threshold_mbands'),
            threshold_zbands=_pv(m, 'motion.track.threshold_zbands'),
            max_disp_along_um=_pv(m, 'motion.track.max_disp_along'),
            max_disp_perp_um=_pv(m, 'motion.track.max_disp_perp'),
            ori_tol_deg=_pv(m, 'motion.track.ori_tol'),
            memory=int(_pv(m, 'motion.track.memory')),
            min_track_length=int(_pv(m, 'motion.track.min_length')),
            max_gap_interpolation=int(_pv(m, 'motion.track.max_gap_interp')),
            merge_tracks=bool(_pv(m, 'motion.track.merge')),
            slen_lims=(_pv(m, 'motion.track.slen_lower'), _pv(m, 'motion.track.slen_upper')),
            progress_notifier=pn)

    def on_btn_track_vectors(self):
        if not self.__chk_initialized():
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
        if by == 'loi':
            if self.__drawn_lois:
                # user-drawn lines (yx-px polylines) feed the same loi grouping
                cell.data['loi_data'] = {'loi_lines': self.__drawn_lois}
            else:
                cell.detect_lois(
                    frame=ref,
                    n_lois=int(_pv(m, 'loi.detect.n_lois')),
                    ratio_seeds=_pv(m, 'loi.detect.ratio_seeds'),
                    persistence=int(_pv(m, 'loi.detect.persistence')),
                    threshold_distance=_pv(m, 'loi.detect.threshold_distance'),
                    mode=_pv(m, 'loi.detect.mode'),
                    number_lims=(int(_pv(m, 'loi.detect.number_limits_lower')),
                                 int(_pv(m, 'loi.detect.number_limits_upper'))),
                    length_lims=(_pv(m, 'loi.detect.length_limits_lower'),
                                 _pv(m, 'loi.detect.length_limits_upper')),
                    distance_threshold_lois=_pv(m, 'loi.detect.cluster_threshold_lois'),
                    linkage=_pv(m, 'loi.detect.linkage'))
        cell.group_tracks(by=by, reference_frame=ref, min_coverage=cov)

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
        agg = _pv(m, 'motion.analyze.aggregate')
        agg = None if agg in (None, '', 'auto') else agg
        cell.analyze_track_motion(
            aggregate=agg,
            slen_lims=(_pv(m, 'motion.analyze.slen_lower'), _pv(m, 'motion.analyze.slen_upper')),
            threshold=_pv(m, 'motion.analyze.threshold'),
            contr_time_min=_pv(m, 'motion.analyze.contr_time_min'),
            merge_time_max=_pv(m, 'motion.analyze.merge_time_max'),
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
            finished_successful_action=cell.commit)
        self.__worker = worker
        return worker

    # ------------------------------------------------------------------ #
    # master: track -> group -> analyze
    # ------------------------------------------------------------------ #
    def on_analyze_motion(self):
        if not self.__chk_initialized():
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
        self.__main_control.init_tracks_stack()

    def __grouping_finished(self):
        self.__main_control.init_tracks_stack()        # recolour trajectory lines by group
        self.__main_control.init_track_groups_stack()
        self.__refresh_fibre_combo()

    def __refresh_fibre_combo(self):
        w = self.__motion_analysis_widget
        cell = self.__main_control.model.cell
        w.cb_fibre_group.clear()
        kind = cell.data.get('group_kind') if cell is not None else None
        n_groups = int(cell.data.get('n_groups', 0)) if cell is not None else 0
        is_fibre = kind in _FIBRE_KINDS and n_groups > 0
        if is_fibre:
            w.cb_fibre_group.addItems([str(g) for g in range(n_groups)])
        w.cb_fibre_group.setEnabled(is_fibre)
        w.btn_show_fibre_detail.setEnabled(is_fibre)

    # ------------------------------------------------------------------ #
    # per-fibre detail panel (LOI-style plots via get_track_motion)
    # ------------------------------------------------------------------ #
    def __fibre_call(self, worker, m: ApplicationModel):
        cell: SarcAsM = TypeUtils.unbox(m.cell)
        group = int(self.__motion_analysis_widget.cb_fibre_group.currentText())
        motion = cell.get_track_motion(group, analyze=True)
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
        kind = cell.data.get('group_kind') if cell is not None else None
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
        motion = self.__fibre_motion
        if motion is None:
            return
        cell = self.__main_control.model.cell
        kind = cell.data.get('group_kind', '') if cell is not None else ''
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
        plt.show()

    # ------------------------------------------------------------------ #
    # grouped-motion summary
    # ------------------------------------------------------------------ #
    def __on_btn_plot_summary(self):
        if not self.__chk_initialized():
            return
        import matplotlib.pyplot as plt
        cell = self.__main_control.model.cell
        kind = cell.data.get('track_motion_kind') if cell is not None else None
        if kind is None:
            logger.warning('No track motion analyzed yet — run "Analyze track motion" first.')
            return
        is_fibre = kind in _FIBRE_KINDS
        layout = [['tracks', 'groups'], ['slen', 'dslen']]
        if is_fibre:
            layout.append(['fibrils', 'fibrils'])
        fig, axd = plt.subplot_mosaic(layout, figsize=(12, 10 if is_fibre else 8),
                                      constrained_layout=True)
        # 'tracks' = trajectory lines coloured by coverage (track quality);
        # 'groups' = partition scatter coloured by group — complementary views.
        panels = (('tracks', lambda ax: Plots.plot_tracks(ax, cell, color_by='coverage', colorbar=True)),
                  ('groups', lambda ax: Plots.plot_track_groups(ax, cell)),
                  ('slen', lambda ax: Plots.plot_slen_mean(ax, cell)),
                  ('dslen', lambda ax: Plots.plot_delta_slen_mean(ax, cell)))
        for key, fn in panels:
            try:
                fn(axd[key])
            except Exception as e:
                logger.warning(f'summary {key} plot failed: {e}')
        if is_fibre:
            try:
                Plots.plot_track_myofibrils(axd['fibrils'], cell, color_by='beating_rate')
            except Exception as e:
                logger.warning(f'summary fibrils plot failed: {e}')
        fig.suptitle(f'{Path(cell.file_path).name} — {kind}')
        plt.show()

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    def __on_btn_export_motion_data(self):
        if not self.__chk_initialized():
            return
        cell = self.__main_control.model.cell
        kind = cell.data.get('track_motion_kind', '') if cell is not None else ''
        stem = f'{Path(cell.file_path).stem}_{kind}'
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
        w.cb_motion_aggregate.addItems(['auto', 'nanmedian', 'nanmean'])
        w.cb_loi_mode.addItems(['longest_in_cluster', 'fit_straight_line',
                                'random_from_cluster', 'random_line'])
        w.cb_loi_linkage.addItems(['single', 'complete', 'average', 'ward'])

        w.btn_track_vectors.clicked.connect(self.on_btn_track_vectors)
        w.btn_group_tracks.clicked.connect(self.on_btn_group_tracks)
        w.btn_draw_lois.clicked.connect(self.on_btn_draw_lois)
        w.btn_analyze_track_motion.clicked.connect(self.on_btn_analyze_track_motion)
        w.btn_analyze_motion.clicked.connect(self.on_analyze_motion)
        w.btn_show_fibre_detail.clicked.connect(self.on_btn_show_fibre_detail)
        w.btn_plot_summary.clicked.connect(self.__on_btn_plot_summary)
        w.btn_export_motion_data.clicked.connect(self.__on_btn_export_motion_data)
        w.btn_show_fibre_detail.setEnabled(False)

        p = self.__main_control.model.parameters.get_parameter
        # track
        p('motion.track.threshold_mbands').connect(w.dsb_track_threshold_mbands)
        p('motion.track.threshold_zbands').connect(w.dsb_track_threshold_zbands)
        p('motion.track.max_disp_along').connect(w.dsb_track_max_disp_along)
        p('motion.track.max_disp_perp').connect(w.dsb_track_max_disp_perp)
        p('motion.track.ori_tol').connect(w.dsb_track_ori_tol)
        p('motion.track.memory').connect(w.sb_track_memory)
        p('motion.track.min_length').connect(w.sb_track_min_length)
        p('motion.track.max_gap_interp').connect(w.sb_track_max_gap_interp)
        p('motion.track.merge').connect(w.chk_track_merge)
        p('motion.track.slen_lower').connect(w.dsb_track_slen_lower)
        p('motion.track.slen_upper').connect(w.dsb_track_slen_upper)
        # group
        p('motion.group.by').connect(w.cb_group_by)
        p('motion.group.reference_frame').connect(w.sb_group_reference_frame)
        p('motion.group.min_coverage').connect(w.dsb_group_min_coverage)
        # LOI auto-detection (advanced, collapsed by default)
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
        p('motion.analyze.slen_lower').connect(w.dsb_motion_slen_lower)
        p('motion.analyze.slen_upper').connect(w.dsb_motion_slen_upper)
