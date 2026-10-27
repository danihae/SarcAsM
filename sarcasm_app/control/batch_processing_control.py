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

    def on_btn_batch_processing_motion(self):
        if not self.__batch_files_or_explain():
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

        failures = []
        for i, file in enumerate(progress_notifier.iterator(tif_files)):
            try:
                self.__single_motion_analysis(file, frame_time, pixel_size, channel, axes, force_override, model)
            except Exception as e:
                failures.append((file, e))
                # this part has to be added to qt thread
                qtutils.inmain(self.__main_control.debug,
                               message='Exception happened during processing of file:' + file)
                qtutils.inmain(self.__main_control.debug, message=f'Error: {repr(e)}')
                qtutils.inmain(self.__main_control.debug, message='')
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
    def __calculate_requirements_of_motion(sarc_obj: SarcAsM, model):
        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        size = patch_size_from_parameters(model.parameters, 'structure.predict')

        sarc_obj.detect_sarcomeres(frames=model.parameters.get_parameter('structure.frames').get_value(),
                                   model_path=network_model,
                                   max_patch_size=size,
                                   clip_thres=(
                                       model.parameters.get_parameter('structure.predict.clip_thresh_min').get_value(),
                                       model.parameters.get_parameter('structure.predict.clip_thresh_max').get_value())
                                   )
        sarc_obj.analyze_sarcomere_vectors(
            frames=model.parameters.get_parameter('structure.frames').get_value(),
            slen_lims=(
                model.parameters.get_parameter('structure.vectors.length_limit_lower').get_value(),
                model.parameters.get_parameter('structure.vectors.length_limit_upper').get_value()
            ),
            median_filter_radius=model.parameters.get_parameter('structure.vectors.radius').get_value(),
            linewidth=model.parameters.get_parameter('structure.vectors.line_width').get_value(),
            interp_factor=model.parameters.get_parameter('structure.vectors.interpolation_factor').get_value(),
            neighbor_radius=model.parameters.get_parameter('structure.vectors.neighbor_radius').get_value(),
            max_neighbor_deviation=(
                model.parameters.get_parameter('structure.vectors.max_neighbor_deviation').get_value()
                if model.parameters.get_parameter('structure.vectors.filter_neighbors').get_value() else None),
        )
        pass

    def __single_motion_analysis(self, file: str, frame_time: float, pixel_size: float, channel: int, axes: str,
                                 force_override: bool, model):
        sarc_obj = BatchProcessingControl.__get_sarc_object(file=file, frame_time=frame_time, pixel_size=pixel_size,
                                                            channel=channel, axes=axes, force_override=force_override)

        def _p(name):
            return model.parameters.get_parameter(name).get_value()

        by = _p('motion.group.by')
        if _p('batch.recalculate.for.motion'):
            self.__calculate_requirements_of_motion(sarc_obj, model)
            if by == 'myofibril':
                sarc_obj.analyze_myofibrils()
            elif by == 'domain':
                sarc_obj.analyze_sarcomere_domains()

        sarc_obj.track_sarcomere_vectors(
            frames='all',
            max_disp_along_um=_p('motion.track.max_disp_along'),
            max_disp_perp_um=_p('motion.track.max_disp_perp'),
            ori_tol_deg=_p('motion.track.ori_tol'),
            min_track_duration_s=_p('motion.track.min_duration_s'),
            max_gap_interpolation_s=float(_p('motion.track.max_gap_interp_s')),
            motion_predictor='flow' if bool(_p('motion.track.flow_predictor')) else 'none')

        ref = int(_p('motion.group.reference_frame'))
        if by == 'loi' and 'motion.loi.data' not in sarc_obj.data:
            sarc_obj.detect_lois(frame=ref, **detect_lois_kwargs(_p))
        sarc_obj.group_tracks(by=by, reference_frame=ref, min_coverage=_p('motion.group.min_coverage'),
                              min_group_size=int(_p('motion.group.min_group_size')))

        sarc_obj.analyze_track_motion(
            aggregate=str(_p('motion.analyze.aggregate')),
            slen_lims=(_p('motion.analyze.slen_lower'), _p('motion.analyze.slen_upper')),
            threshold=_p('motion.analyze.threshold'),
            contr_time_min=_p('motion.analyze.contr_time_min'),
            merge_time_max=_p('motion.analyze.merge_time_max'),
            buffer_frames=int(_p('motion.analyze.buffer_frames')),
            min_valid_frames=_p('motion.analyze.min_valid_frames'),
            filter_params=(int(_p('motion.analyze.filter_wl')), int(_p('motion.analyze.filter_po'))))

        if _p('batch.delete_intermediate_masks'):
            sarc_obj.remove_intermediate_masks()

    def __single_structure_analysis(self, file: str, frame_time: float, pixel_size: float, channel: int, axes: str,
                                    force_override: bool, model):
        # attention: this method is not executed in qt thread! --> every information to ui needs to be done either
        # on another place or within a wrapper for QT Main thread (like the package qtutils.inmain does)

        # initialize SarcAsM object
        # check for metadata
        sarc_obj = BatchProcessingControl.__get_sarc_object(file=file, frame_time=frame_time, pixel_size=pixel_size,
                                                            channel=channel, axes=axes, force_override=force_override)
        # predict sarcomere z-bands and cell mask
        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        size = patch_size_from_parameters(model.parameters, 'structure.predict')

        sarc_obj.detect_sarcomeres(frames=model.parameters.get_parameter('structure.frames').get_value(),
                                   model_path=network_model,
                                   max_patch_size=size,
                                   rescale_factor=model.parameters.get_parameter('structure.predict.rescale_factor').get_value(),
                                   clip_thres=(
                                       model.parameters.get_parameter('structure.predict.clip_thresh_min').get_value(),
                                       model.parameters.get_parameter('structure.predict.clip_thresh_max').get_value()),
                                   )

        # analyze sarcomere structures (the cell-mask features come with the detection)
        if model.parameters.get_parameter('batch.do_zbands').get_value():
            sarc_obj.analyze_z_bands(frames=model.parameters.get_parameter('structure.frames').get_value(),
                                     threshold=model.parameters.get_parameter(
                                         'structure.z_band_analysis.threshold').get_value(),
                                     min_length=model.parameters.get_parameter(
                                         'structure.z_band_analysis.min_length').get_value(),
                                     median_filter_radius=model.parameters.get_parameter(
                                         'structure.z_band_analysis.median_filter_radius').get_value(),
                                     theta_phi_min=model.parameters.get_parameter(
                                         'structure.z_band_analysis.theta_phi_min').get_value(),
                                     a_min=model.parameters.get_parameter(
                                         'structure.z_band_analysis.a_min').get_value(),
                                     d_max=model.parameters.get_parameter(
                                         'structure.z_band_analysis.d_max').get_value(),
                                     d_min=model.parameters.get_parameter(
                                         'structure.z_band_analysis.d_min').get_value(),
                                     max_axial_offset=model.parameters.get_parameter(
                                         'structure.z_band_analysis.max_axial_offset').get_value())

        # careful this method highly depends on pixel size setting
        if model.parameters.get_parameter('batch.do_vectors').get_value():
            sarc_obj.analyze_sarcomere_vectors(
                frames=model.parameters.get_parameter('structure.frames').get_value(),
                slen_lims=(
                    model.parameters.get_parameter('structure.vectors.length_limit_lower').get_value(),
                    model.parameters.get_parameter('structure.vectors.length_limit_upper').get_value()
                ),
                median_filter_radius=model.parameters.get_parameter('structure.vectors.radius').get_value(),
                linewidth=model.parameters.get_parameter('structure.vectors.line_width').get_value(),
                interp_factor=model.parameters.get_parameter('structure.vectors.interpolation_factor').get_value(),
                smooth_orientation_sigma=model.parameters.get_parameter(
                    'structure.vectors.smooth_orientation_sigma').get_value(),
                neighbor_radius=model.parameters.get_parameter('structure.vectors.neighbor_radius').get_value(),
                max_neighbor_deviation=(
                    model.parameters.get_parameter('structure.vectors.max_neighbor_deviation').get_value()
                    if model.parameters.get_parameter('structure.vectors.filter_neighbors').get_value() else None),
            )

        if model.parameters.get_parameter('batch.do_myofibrils').get_value():
            sarc_obj.analyze_myofibrils(
                frames=model.parameters.get_parameter('structure.frames').get_value(),
                ratio_seeds=model.parameters.get_parameter('structure.myofibril.ratio_seeds').get_value(),
                persistence=model.parameters.get_parameter('structure.myofibril.persistence').get_value(),
                threshold_distance=model.parameters.get_parameter('structure.myofibril.threshold_distance').get_value(),
                n_min=model.parameters.get_parameter('structure.myofibril.n_min').get_value(),
                median_filter_radius=model.parameters.get_parameter('structure.myofibril.median_filter_radius').get_value()
            )

        if model.parameters.get_parameter('batch.do_domains').get_value():
            sarc_obj.analyze_sarcomere_domains(
                frames=model.parameters.get_parameter('structure.frames').get_value(),
                d_max=model.parameters.get_parameter('structure.domain.analysis.d_max').get_value(),
                cosine_min=model.parameters.get_parameter('structure.domain.analysis.cosine_min').get_value(),
                leiden_resolution=model.parameters.get_parameter(
                    'structure.domain.analysis.leiden_resolution').get_value(),
                random_seed=model.parameters.get_parameter('structure.domain.analysis.random_seed').get_value(),
                area_min=model.parameters.get_parameter('structure.domain.analysis.area_min').get_value(),
                dilation_radius=model.parameters.get_parameter('structure.domain.analysis.dilation_radius').get_value()
            )

        sarc_obj.store_structure_data()
        if model.parameters.get_parameter('batch.delete_intermediate_masks').get_value():
            # tracking re-measures lengths on the Z-band masks: a movie keeps them for the motion analysis
            keep = ('zbands', 'zbands_fast_movie') if (sarc_obj.metadata.n_stack or 1) > 1 else ()
            sarc_obj.remove_intermediate_masks([m for m in sarc_obj._STORE_MASKS if m not in keep])

        pass

    def on_search(self):
        # f_name is a tuple
        file = str(QFileDialog.getExistingDirectory(caption="Select Root Directory"))
        if file is not None and file != '':
            self.__batch_processing_widget.le_root_directory.setText(file)
        pass

    pass