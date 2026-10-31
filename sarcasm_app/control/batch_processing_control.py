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
from pathlib import Path
import concurrent.futures
import importlib

import qtutils
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from bio_image_unet.progress import ProgressNotifier

from sarcasm import SarcAsM, BatchExport, Export
from .application_control import ApplicationControl
from .motion_analysis_control import detect_lois_kwargs
from ..view.parameters_batch_processing import Ui_Form as BatchProcessingWidget
from ..model import patch_size_from_parameters

logger = logging.getLogger(__name__)


def _parallel_files_safe() -> bool:
    """Whether several files may be analysed at once in threads.

    Not with GPU inference (MPS/CUDA kernels from several threads corrupt or crash), and not
    with numba's default 'workqueue' threading layer, which aborts the process when parallel
    functions run from several threads (the only layer on macOS without TBB/OpenMP).
    """
    from sarcasm import Utils
    if Utils.get_device().type != 'cpu':
        return False
    for module in ('numba.np.ufunc.tbbpool', 'numba.np.ufunc.omppool'):
        try:
            importlib.import_module(module)
            return True
        except Exception:
            continue
    return False


def _raise_batch_failures(failures, n_files: int):
    """End a batch that had failing files with an error naming them (the worker shows it in a dialog);
    the other files were analysed and saved."""
    if not failures:
        return
    listed = '\n'.join(f'{os.path.basename(str(f))}: {e}' for f, e in failures[:10])
    more = f'\n... and {len(failures) - 10} more (see the log)' if len(failures) > 10 else ''
    raise RuntimeError(f'{len(failures)} of {n_files} files failed; the other {n_files - len(failures)} were '
                       f'analysed and saved.\n\n{listed}{more}')


def _safe_pool_size(requested: int) -> int:
    """The requested number of parallel files, or 1 when parallel files are unsafe here."""
    if requested > 1 and not _parallel_files_safe():
        logger.info(f'Analysing the files one after another (thread pool size {requested} requested): '
                    f'files in parallel are not safe with GPU inference or without a thread-safe '
                    f'numba threading layer on this computer.')
        return 1
    return requested


# The structure steps with the settings of the Structure tab, shared by the structure batch (the selected
# frames) and the motion batch (every frame, or the reference frame); `p(name)` reads a parameter.
def _quiet(progress_notifier):
    return progress_notifier if progress_notifier is not None else ProgressNotifier()


def _detect_sarcomeres(sarc_obj: SarcAsM, parameters, frames, progress_notifier=None):
    def p(name):
        return parameters.get_parameter(name).get_value()
    sarc_obj.detect_sarcomeres(frames=frames,
                               model_path=p('structure.predict.network_path'),
                               max_patch_size=patch_size_from_parameters(parameters, 'structure.predict'),
                               rescale_factor=p('structure.predict.rescale_factor'),
                               clip_thres=(p('structure.predict.clip_thresh_min'),
                                           p('structure.predict.clip_thresh_max')),
                               progress_notifier=_quiet(progress_notifier))


def _detect_z_bands_fast_movie(sarc_obj: SarcAsM, parameters, progress_notifier=None):
    network_model = parameters.get_parameter('structure.predict_fast_movie.network_path').get_value()
    sarc_obj.motion.detect_z_bands_fast_movie(
        model_path=None if network_model in ('default', 'generalist') else network_model,
        max_patch_size=patch_size_from_parameters(parameters, 'structure.predict_fast_movie'),
        clip_thres=(parameters.get_parameter('structure.predict_fast_movie.clip_thresh_min').get_value(),
                    parameters.get_parameter('structure.predict_fast_movie.clip_thresh_max').get_value()),
        progress_notifier=_quiet(progress_notifier))


def _analyze_z_bands(sarc_obj: SarcAsM, p, frames):
    sarc_obj.structure.analyze_z_bands(frames=frames,
                             threshold=p('structure.z_band_analysis.threshold'),
                             min_length=p('structure.z_band_analysis.min_length'),
                             median_filter_radius=p('structure.z_band_analysis.median_filter_radius'),
                             theta_phi_min=p('structure.z_band_analysis.theta_phi_min'),
                             a_min=p('structure.z_band_analysis.a_min'),
                             d_max=p('structure.z_band_analysis.d_max'),
                             d_min=p('structure.z_band_analysis.d_min'),
                             max_axial_offset=p('structure.z_band_analysis.max_axial_offset'))


def _analyze_sarcomere_vectors(sarc_obj: SarcAsM, p, frames, progress_notifier=None, use_fast_movie_zbands=True):
    sarc_obj.structure.analyze_vectors(
        frames=frames,
        use_fast_movie_zbands=use_fast_movie_zbands,
        slen_lims=(p('structure.vectors.length_limit_lower'), p('structure.vectors.length_limit_upper')),
        median_filter_radius=p('structure.vectors.radius'),
        linewidth=p('structure.vectors.line_width'),
        interp_factor=p('structure.vectors.interpolation_factor'),
        smooth_orientation_sigma=p('structure.vectors.smooth_orientation_sigma'),
        neighbor_radius=p('structure.vectors.neighbor_radius'),
        max_neighbor_deviation=(p('structure.vectors.max_neighbor_deviation')
                                if p('structure.vectors.filter_neighbors') else None),
        progress_notifier=_quiet(progress_notifier))


def _analyze_myofibrils(sarc_obj: SarcAsM, p, frames, progress_notifier=None):
    sarc_obj.structure.analyze_myofibrils(frames=frames,
                                ratio_seeds=p('structure.myofibril.ratio_seeds'),
                                persistence=p('structure.myofibril.persistence'),
                                threshold_distance=p('structure.myofibril.threshold_distance'),
                                n_min=p('structure.myofibril.n_min'),
                                median_filter_radius=p('structure.myofibril.median_filter_radius'),
                                progress_notifier=_quiet(progress_notifier))


def _analyze_sarcomere_domains(sarc_obj: SarcAsM, p, frames, progress_notifier=None):
    sarc_obj.structure.analyze_domains(frames=frames,
                                       d_max=p('structure.domain.analysis.d_max'),
                                       cosine_min=p('structure.domain.analysis.cosine_min'),
                                       leiden_resolution=p('structure.domain.analysis.leiden_resolution'),
                                       random_seed=p('structure.domain.analysis.random_seed'),
                                       area_min=p('structure.domain.analysis.area_min'),
                                       dilation_radius=p('structure.domain.analysis.dilation_radius'),
                                       progress_notifier=_quiet(progress_notifier))


#: Groupings the motion batch can analyse, in the order they run (the batch tab has one checkbox each).
_BATCH_GROUPINGS = ('pool', 'mband', 'myofibril', 'domain', 'loi')


class BatchProcessingControl:

    def __init__(self, batch_processing_widget: BatchProcessingWidget, main_control: ApplicationControl):
        self.__batch_processing_widget = batch_processing_widget
        self.__main_control = main_control
        self.__worker = None
        pass

    def bind_events(self):
        parameters = self.__main_control.model.parameters
        widget = self.__batch_processing_widget

        self.__batch_processing_widget.btn_batch_processing_structure.clicked.connect(
            self.on_btn_batch_processing_structure)
        self.__batch_processing_widget.btn_batch_processing_motion.clicked.connect(
            self.on_btn_batch_processing_motion)
        self.__batch_processing_widget.btn_search.clicked.connect(self.on_search)
        self.__batch_processing_widget.btn_batch_export_structure.clicked.connect(self.on_btn_batch_export_structure)
        self.__batch_processing_widget.btn_batch_export_motion.clicked.connect(self.on_btn_batch_export_motion)

        parameters.get_parameter(name='batch.pixel.size').connect(widget.dsb_pixel_size)
        parameters.get_parameter(name='batch.frame.time').connect(widget.dsb_frame_time)
        parameters.get_parameter(name='batch.channel').connect(widget.sb_batch_channel)
        parameters.get_parameter(name='batch.axes').connect(widget.le_batch_axes)
        parameters.get_parameter(name='batch.force.override').connect(widget.chk_force_override)
        parameters.get_parameter(name='batch.thread_pool_size').connect(widget.sb_thread_pool_size)
        parameters.get_parameter(name='batch.delete_intermediate_masks').connect(widget.chk_delete_intermediate_masks)
        parameters.get_parameter(name='batch.root').connect(widget.le_root_directory)
        parameters.get_parameter(name='batch.recalculate.for.motion').connect(widget.chk_recalculate_for_motion)
        parameters.get_parameter(name='batch.do_zbands').connect(widget.chk_do_zbands)
        parameters.get_parameter(name='batch.do_vectors').connect(widget.chk_do_vectors)
        parameters.get_parameter(name='batch.do_myofibrils').connect(widget.chk_do_myofibrils)
        parameters.get_parameter(name='batch.do_domains').connect(widget.chk_do_domains)
        parameters.get_parameter(name='batch.motion.fast_movie').connect(widget.chk_batch_fast_movie)
        for kind in _BATCH_GROUPINGS:
            parameters.get_parameter(name=f'batch.motion.group_{kind}').connect(
                getattr(widget, f'chk_batch_group_{kind}'))

        pass

    def __get_progress_notifier(self, worker) -> ProgressNotifier:
        progress_notifier = ProgressNotifier()

        def __internal_function(p):
            qtutils.inmain(lambda: self.__main_control.update_progress(int(p * 100)))  # wrap with qt main thread
            pass

        progress_notifier.set_progress_report(__internal_function)
        progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: worker.progress_details.emit(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (
                    hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))
        return progress_notifier

    def __batch_files_or_explain(self) -> bool:
        """Whether the batch folder holds .tif files; otherwise say so in a dialog (the log is easy to miss)."""
        root = self.__main_control.model.parameters.get_parameter(name='batch.root').get_value()
        if not root or str(root) == 'None':
            QMessageBox.information(self.__batch_processing_widget.btn_batch_processing_structure,
                                    'No folder chosen', 'Choose the folder with your .tif files first.')
            return False
        if not self.__get_tiff_files(self.__main_control.model):
            QMessageBox.information(self.__batch_processing_widget.btn_batch_processing_structure,
                                    'No .tif files', f'No .tif / .tiff files were found in\n{root}\n'
                                    f'or its subfolders.')
            return False
        return True

    def on_btn_batch_processing_structure(self):
        if not self.__batch_files_or_explain():
            return
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__batch_process_structure_async,
                                                   start_message='Start batch processing structure ',
                                                   finished_message='Finished batch processing structure ')
        self.__worker = worker

        pass

    def __get_tiff_files(self, model):
        root = model.parameters.get_parameter(name='batch.root').get_value()
        if not root or str(root) == 'None' or not Path(root).is_dir():
            logger.warning(f'The batch folder {root!r} does not exist.' if root and str(root) != 'None'
                           else 'Choose the batch folder first.')
            return []
        root = Path(root)

        excluded_names = {
            'zbands.tif',
            'mbands.tif',
            'cell_mask.tif',
            'orientation.tif',
            'sarcomere_mask.tif'
        }

        tif_files = [
            p for p in root.rglob('*')
            if p.suffix.lower() in ('.tif', '.tiff') and p.name.lower() not in excluded_names
        ]

        n_tif_files = len(tif_files)
        if n_tif_files == 0:
            logger.warning(f"No TIFF files found in {root}")
        else:
            logger.info(f"Found {n_tif_files} TIFF files to process")

        return tif_files

    def __batch_process_structure_async(self, worker, model):
        """
        Analyse every TIFF in `tif_files` concurrently.

        Each file is completely independent, so we can run the costly
        `__single_structure_analysis` in a thread pool.

        Errors are tunneled back to the main thread so that we can log them
        through Qt-safe helpers (`qtutils.inmain`).
        """
        tif_files: list[Path] = self.__get_tiff_files(model)

        n_pools = _safe_pool_size(int(model.parameters
                                          .get_parameter(name='batch.thread_pool_size')
                                          .get_value()) or 1)

        frame_time = model.parameters.get_parameter(name='batch.frame.time').get_value()
        pixel_size = model.parameters.get_parameter(name='batch.pixel.size').get_value()
        channel = model.parameters.get_parameter(name='batch.channel').get_value()
        axes = model.parameters.get_parameter(name='batch.axes').get_value()
        force_override = model.parameters.get_parameter(name='batch.force.override').get_value()

        # Helper that is executed inside every worker thread
        def _run_single(file_):
            try:
                self.__single_structure_analysis(
                    file_, frame_time, pixel_size,
                    channel, axes, force_override, model
                )
                return (file_, None)          # success
            except Exception as exc:          # capture exception -> propagate
                return (file_, exc)

        failures = []
        # Kick off the pool and iterate over completed futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_pools) as pool:
            future_map = {pool.submit(_run_single, f): f for f in tif_files}

            total = len(tif_files)
            finished = 0

            for fut in concurrent.futures.as_completed(future_map):
                finished += 1

                qtutils.inmain(
                    self.__main_control.update_progress,
                    finished / total * 100
                )

                file_, err = fut.result()     # unpack tuple returned above
                if err is not None:
                    failures.append((file_, err))
                    # Marshal logging back to the Qt main thread
                    qtutils.inmain(
                        self.__main_control.debug,
                        message=f'Exception while processing {file_}: {err!r}'
                    )
                    traceback.print_exception(err)
        _raise_batch_failures(failures, len(tif_files))


    def on_btn_batch_export_structure(self):
        """UI callback: *Batch → Export Structure* button."""
        model = self.__main_control.model
        tif_files = self.__get_tiff_files(model)
        folder = self.__main_control.model.parameters.get_parameter(name='batch.root').get_value()

        if not self.__batch_files_or_explain():                        # nothing to do
            return

        # let the user pick / create the Excel workbook
        default = Path(folder)
        file_export, _ = QFileDialog.getSaveFileName(
            caption="Save structure measurements",
            directory=str(default),
            filter="Excel Workbook (*.xlsx)"
        )
        if not file_export:                                            # user pressed Cancel
            return
        if not file_export.lower().endswith('.xlsx'):                  # the content is always a workbook
            file_export = str(Path(file_export).with_suffix('.xlsx'))

        # store arguments for the worker and launch it asynchronously
        self.__export_args = (tif_files, Path(file_export))

        worker = self.__main_control.run_async_new(
            parameters=model,
            call_lambda=self.__batch_export_structure_async,
            start_message="Start batch export structure ",
            finished_message="Finished batch export structure "
        )
        self.__worker = worker

    def __batch_export_structure_async(self, worker, model):
        progress_notifier = self.__get_progress_notifier(worker)       # reuse existing helper
        tif_files, file_export = self.__export_args
        folder = self.__main_control.model.parameters.get_parameter(name='batch.root').get_value()

        try:
            msa = BatchExport(tif_files, folder=folder)
            # progress bar while gathering data
            for _ in progress_notifier.iterator([0]):
                msa.get_data()

            msa.export_data(file_export)
            if str(file_export).lower().endswith('.xlsx'):
                Export.append_info_sheets(str(file_export), None,
                                          [c for c in msa.data.columns if str(c).startswith('structure.')])
            logger.info(f'Exported {len(msa.data)} file(s) to {file_export}')
        except PermissionError as e:
            # re-raised: the worker reports it in a dialog and no "Finished" follows
            raise PermissionError(f'Could not write {file_export}: {e.strerror or e}. Choose a folder you can '
                                  f'write to, and close the file if it is open in Excel.') from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not write {file_export}: the folder does not exist.') from e

    def on_btn_batch_export_motion(self):
        """UI callback: *Batch → Export motion data* button."""
        if not self.__batch_files_or_explain():
            return
        model = self.__main_control.model
        folder = model.parameters.get_parameter(name='batch.root').get_value()
        file_export, _ = QFileDialog.getSaveFileName(caption='Save motion measurements', directory=str(folder),
                                                     filter='Excel Workbook (*.xlsx)')
        if not file_export:
            return
        if not file_export.lower().endswith('.xlsx'):
            file_export = str(Path(file_export).with_suffix('.xlsx'))
        self.__export_args = (self.__get_tiff_files(model), Path(file_export))
        self.__worker = self.__main_control.run_async_new(
            parameters=model, call_lambda=self.__batch_export_motion_async,
            start_message='Start batch export motion ', finished_message='Finished batch export motion ')

    def __batch_export_motion_async(self, worker, model):
        tif_files, file_export = self.__export_args
        folder = model.parameters.get_parameter(name='batch.root').get_value()
        msa = BatchExport(tif_files, folder=folder)
        msa.get_motion_data()
        if msa.data.empty:
            raise ValueError('None of the movies has motion results yet: run the batch motion analysis first.')
        try:
            msa.export_data(file_export)
        except PermissionError as e:
            raise PermissionError(f'Could not write {file_export}: {e.strerror or e}. Choose a folder you can '
                                  f'write to, and close the file if it is open in Excel.') from e
        Export.append_info_sheets(str(file_export), None,
                                  [c for c in msa.data.columns if c in Export.motion_keys_default], registry='motion')
        logger.info(f'Exported {len(msa.data)} group(s) of {msa.data["tif_name"].nunique()} movie(s) to {file_export}')

    def on_btn_batch_processing_motion(self):
        if not self.__batch_files_or_explain():
            return
        if not self.__batch_groupings(self.__main_control.model):
            QMessageBox.information(self.__batch_processing_widget.btn_batch_processing_motion, 'No grouping chosen',
                                    'Tick at least one grouping under "Group by" (e.g. Whole cell).')
            return
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__batch_process_motion_async,
                                                   start_message='Start batch processing motion ',
                                                   finished_message='Finished batch processing motion ')
        self.__worker = worker
        pass

    def __batch_process_motion_async(self, worker, model):
        progress_notifier = self.__get_progress_notifier(worker)

        # the same files as the structure batch: every TIFF below the root, in any subfolder
        tif_files = [str(p) for p in self.__get_tiff_files(model)]
        frame_time = model.parameters.get_parameter(name='batch.frame.time').get_value()
        pixel_size = model.parameters.get_parameter(name='batch.pixel.size').get_value()
        channel = model.parameters.get_parameter(name='batch.channel').get_value()
        axes = model.parameters.get_parameter(name='batch.axes').get_value()
        force_override = model.parameters.get_parameter(name='batch.force.override').get_value()

        # one file after another: each step reports its own progress, the log says which file runs
        failures = []
        for i, file in enumerate(tif_files):
            qtutils.inmain(self.__main_control.debug,
                           message=f'Batch motion: file {i + 1} of {len(tif_files)}, {os.path.basename(file)}')
            try:
                self.__single_motion_analysis(file, frame_time, pixel_size, channel, axes, force_override, model,
                                              progress_notifier)
            except Exception as e:
                failures.append((file, e))
                qtutils.inmain(self.__main_control.debug,
                               message=f'Exception while processing {file}: {e!r}')
                traceback.print_exception(e)
        _raise_batch_failures(failures, len(tif_files))

    @staticmethod
    def __get_sarc_object(file: str, frame_time: float, pixel_size: float, channel: int, axes: str,
                          force_override: bool) -> SarcAsM:

        sarc_obj = SarcAsM(file, use_gui=True)
        if force_override:
            sarc_obj.metadata.pixelsize = pixel_size
            sarc_obj.metadata.frametime = frame_time
            sarc_obj.metadata.channel = channel
            sarc_obj.metadata.axes = axes
            sarc_obj.save_metadata()
            pass
        return sarc_obj

    @staticmethod
    def __batch_groupings(model) -> list:
        return [kind for kind in _BATCH_GROUPINGS
                if model.parameters.get_parameter(f'batch.motion.group_{kind}').get_value()]

    def __single_motion_analysis(self, file: str, frame_time: float, pixel_size: float, channel: int, axes: str,
                                 force_override: bool, model, progress_notifier=None):
        """Track one movie, then group and analyse it once per grouping ticked in the batch tab. The structure
        steps it needs run first when the movie lacks them (or always with 'Redo the structure steps first'),
        with the settings of the Structure tab; tracking, grouping and analysis use those of the Motion tab."""
        sarc_obj = BatchProcessingControl.__get_sarc_object(file=file, frame_time=frame_time, pixel_size=pixel_size,
                                                            channel=channel, axes=axes, force_override=force_override)

        def _p(name):
            return model.parameters.get_parameter(name).get_value()

        n_stack = int(sarc_obj.metadata.n_stack or 1)
        if n_stack < 2:
            raise ValueError('a single image, so there is no motion to analyse (if it is a movie, check its axes)')
        if not sarc_obj.metadata.frametime:
            raise ValueError('no frame time in the file; enter it with Metadata: "Override for all files"')
        ref = int(_p('motion.group.reference_frame'))
        if not 0 <= ref < n_stack:
            raise ValueError(f'the reference frame {ref} is outside the movie (frames 0 to {n_stack - 1})')

        data = sarc_obj.data
        redo = bool(_p('batch.recalculate.for.motion'))
        fast = bool(_p('batch.motion.fast_movie'))

        def _all_frames(step):
            done = data.get(f'params.{step}.frames')
            return done is not None and len(done) >= n_stack

        source = data.get('params.structure.analyze_vectors.zbands_source')
        run_fast = fast and (redo or not sarc_obj._mask_exists('zbands_fast_movie'))
        run_vectors = (redo or run_fast or not _all_frames('structure.analyze_vectors')
                       or fast != (source == 'fast_movie_3d'))
        # the vectors read the detection masks, which 'Delete intermediate masks' may have removed
        run_detect = redo or not _all_frames('detect_sarcomeres') or (run_vectors and not all(
            sarc_obj._mask_exists(m) for m in ('zbands', 'mbands', 'orientation', 'sarcomere_mask')))
        run_vectors = run_vectors or run_detect
        if run_detect:
            _detect_sarcomeres(sarc_obj, model.parameters, 'all', progress_notifier)
        if run_fast:
            _detect_z_bands_fast_movie(sarc_obj, model.parameters, progress_notifier)
        if run_vectors:
            _analyze_sarcomere_vectors(sarc_obj, _p, 'all', progress_notifier, use_fast_movie_zbands=fast)

        sarc_obj.motion.track(
            frames='all',
            max_disp_along_um=_p('motion.track.max_disp_along'),
            max_disp_perp_um=_p('motion.track.max_disp_perp'),
            ori_tol_deg=_p('motion.track.ori_tol'),
            min_track_duration_s=_p('motion.track.min_duration_s'),
            max_gap_interpolation_s=float(_p('motion.track.max_gap_interp_s')),
            motion_predictor='flow' if bool(_p('motion.track.flow_predictor')) else 'none',
            refine_lengths=bool(_p('motion.track.refine_lengths')),
            progress_notifier=progress_notifier)

        for by in self.__batch_groupings(model):
            if by == 'myofibril':
                lines = data.get('structure.myofibril.lines')
                if run_vectors or lines is None or lines[ref] is None:
                    _analyze_myofibrils(sarc_obj, _p, [ref], progress_notifier)
            elif by == 'domain':
                if run_vectors or ref not in list(data.get('params.structure.analyze_domains.frames', [])):
                    _analyze_sarcomere_domains(sarc_obj, _p, [ref], progress_notifier)
            elif by == 'loi':
                sarc_obj.motion.detect_lois(frame=ref, **detect_lois_kwargs(_p))
            sarc_obj.motion.group_tracks(by=by, reference_frame=ref, min_coverage=_p('motion.group.min_coverage'),
                                  min_group_size=int(_p('motion.group.min_group_size')))
            sarc_obj.motion.analyze(
                aggregate=str(_p('motion.analyze.aggregate')),
                slen_lims=(_p('motion.analyze.slen_lower'), _p('motion.analyze.slen_upper')),
                threshold=_p('motion.analyze.threshold'),
                contr_time_min=_p('motion.analyze.contr_time_min'),
                buffer_frames=int(_p('motion.analyze.buffer_frames')),
                min_valid_frames=_p('motion.analyze.min_valid_frames'),
                filter_params=(int(_p('motion.analyze.filter_wl')), int(_p('motion.analyze.filter_po'))))
        sarc_obj.commit()

        if _p('batch.delete_intermediate_masks'):
            # the Z-band masks stay: tracking the movie again re-measures the lengths on them
            sarc_obj.remove_intermediate_masks([m for m in sarc_obj._STORE_MASKS
                                                if m not in ('zbands', 'zbands_fast_movie')])

    def __single_structure_analysis(self, file: str, frame_time: float, pixel_size: float, channel: int, axes: str,
                                    force_override: bool, model):
        # attention: this method is not executed in qt thread! --> every information to ui needs to be done either
        # on another place or within a wrapper for QT Main thread (like the package qtutils.inmain does)
        sarc_obj = BatchProcessingControl.__get_sarc_object(file=file, frame_time=frame_time, pixel_size=pixel_size,
                                                            channel=channel, axes=axes, force_override=force_override)

        def _p(name):
            return model.parameters.get_parameter(name).get_value()

        frames = _p('structure.frames')
        _detect_sarcomeres(sarc_obj, model.parameters, frames)
        # the cell-mask features come with the detection
        if _p('batch.do_zbands'):
            _analyze_z_bands(sarc_obj, _p, frames)
        # careful this method highly depends on pixel size setting
        if _p('batch.do_vectors'):
            _analyze_sarcomere_vectors(sarc_obj, _p, frames)
        if _p('batch.do_myofibrils'):
            _analyze_myofibrils(sarc_obj, _p, frames)
        if _p('batch.do_domains'):
            _analyze_sarcomere_domains(sarc_obj, _p, frames)

        sarc_obj.store_structure_data()
        if _p('batch.delete_intermediate_masks'):
            # tracking re-measures lengths on the Z-band masks: a movie keeps them for the motion analysis
            keep = ('zbands', 'zbands_fast_movie') if (sarc_obj.metadata.n_stack or 1) > 1 else ()
            sarc_obj.remove_intermediate_masks([m for m in sarc_obj._STORE_MASKS if m not in keep])

    def on_search(self):
        # f_name is a tuple
        file = str(QFileDialog.getExistingDirectory(caption="Select Root Directory"))
        if file is not None and file != '':
            self.__batch_processing_widget.le_root_directory.setText(file)
        pass

    pass