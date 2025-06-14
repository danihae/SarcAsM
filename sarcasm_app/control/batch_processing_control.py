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


import glob
import traceback
from pathlib import Path
from typing import Union, Tuple
import concurrent.futures

import qtutils
from PyQt5.QtWidgets import QFileDialog
from bio_image_unet.progress import ProgressNotifier

from sarcasm import Utils, Motion, Structure, MultiStructureAnalysis
from .application_control import ApplicationControl
from ..view.parameters_batch_processing import Ui_Form as BatchProcessingWidget


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
        parameters.get_parameter(name='batch.delete_intermediary_tiffs').connect(widget.chk_delete_intermediary_tiffs)
        parameters.get_parameter(name='batch.root').connect(widget.le_root_directory)
        parameters.get_parameter(name='batch.recalculate.for.motion').connect(widget.chk_calc_lois)
        parameters.get_parameter(name='batch.do_cellmask').connect(widget.chk_do_cellmask)
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

    def on_btn_batch_processing_structure(self):

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__batch_process_structure_async,
                                                   start_message='Start batch processing structure ',
                                                   finished_message='Finished batch processing structure ')
        self.__worker = worker

        pass

    def __get_tiff_files(self, model):
        root = Path(model.parameters.get_parameter(name='batch.root').get_value())

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
            self.__main_control.debug(f"No TIFF files found in {root}")
        else:
            self.__main_control.debug(f"Found {n_tif_files} TIFF files to process")

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

        n_pools = int(model.parameters
                           .get_parameter(name='batch.thread_pool_size')
                           .get_value()) or 1

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
                    # Marshal logging back to the Qt main thread
                    qtutils.inmain(
                        self.__main_control.debug,
                        message=f'Exception while processing {file_}: {err!r}'
                    )
                    traceback.print_exception(err)


    def on_btn_batch_export_structure(self):
        """UI callback: *Batch → Export Structure* button."""
        model = self.__main_control.model
        tif_files = self.__get_tiff_files(model)
        folder = self.__main_control.model.parameters.get_parameter(name='batch.root').get_value()

        if not tif_files:                                              # nothing to do
            self.__main_control.debug("No processable TIFF files were found in the selected directory.")
            return

        # let the user pick / create the Excel workbook
        default = Path(folder)
        file_export, _ = QFileDialog.getSaveFileName(
            caption="Save structure measurements",
            directory=str(default),
            filter="Excel Workbook (*.xlsx);;All Files (*)"
        )
        if not file_export:                                            # user pressed Cancel
            return

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
            msa = MultiStructureAnalysis(tif_files, folder=folder)
            # progress bar while gathering data
            for _ in progress_notifier.iterator([0]):
                msa.get_data()

            msa.export_data(file_export)
        except Exception as e:
            # marshal the error back to the GUI
            qtutils.inmain(self.__main_control.debug,
                           message=f"Export failed: {repr(e)}")
            traceback.print_exception(e)

    def on_btn_batch_processing_motion(self):
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__batch_process_motion_async,
                                                   start_message='Start batch processing motion ',
                                                   finished_message='Finished batch processing motion ')
        self.__worker = worker
        pass

    def __batch_process_motion_async(self, worker, model):
        progress_notifier = self.__get_progress_notifier(worker)

        tif_files = glob.glob(model.parameters.get_parameter(name='batch.root').get_value() + '*/*.tif')
        n_pools = model.parameters.get_parameter(name='batch.thread_pool_size').get_value()
        frame_time = model.parameters.get_parameter(name='batch.frame.time').get_value()
        pixel_size = model.parameters.get_parameter(name='batch.pixel.size').get_value()
        channel = model.parameters.get_parameter(name='batch.channel').get_value()
        axes = model.parameters.get_parameter(name='batch.axes').get_value()
        force_override = model.parameters.get_parameter(name='batch.force.override').get_value()

        for i, file in enumerate(progress_notifier.iterator(tif_files)):
            try:
                self.__single_motion_analysis(file, frame_time, pixel_size, channel, axes, force_override, model)
            except Exception as e:
                # this part has to be added to qt thread
                qtutils.inmain(self.__main_control.debug,
                               message='Exception happened during processing of file:' + file)
                qtutils.inmain(self.__main_control.debug, message=f'Error: {repr(e)}')
                qtutils.inmain(self.__main_control.debug, message='')
                # todo: add log file to batch processing
                pass
            pass
        pass

    @staticmethod
    def __get_sarc_object(file: str, frame_time: float, pixel_size: float, channel: int, axes: str,
                          force_override: bool) -> Structure:

        sarc_obj = Structure(file, use_gui=True)
        if force_override:
            sarc_obj.metadata.pixelsize = pixel_size
            sarc_obj.metadata.frametime = frame_time
            sarc_obj.metadata.channel = channel
            sarc_obj.metadata.axes = axes
            sarc_obj.save_metadata()
            pass
        return sarc_obj

    @staticmethod
    def __calculate_requirements_of_motion(sarc_obj: Structure, model):
        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None
            pass

        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None

        size: Union[Tuple[int, int]] = (model.parameters.get_parameter('structure.predict.size_width').get_value(),
                                        model.parameters.get_parameter('structure.predict.size_height').get_value())

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
            interp_factor=model.parameters.get_parameter('structure.vectors.interpolation_factor').get_value()
        )
        pass

    def __single_motion_analysis(self, file: str, frame_time: float, pixel_size: float, channel: int, axes: str,
                                 force_override: bool, model):
        sarc_obj = BatchProcessingControl.__get_sarc_object(file=file, frame_time=frame_time, pixel_size=pixel_size,
                                                            channel=channel, axes=axes, force_override=force_override)
        # add some flag if those calculations should be done or not
        if model.parameters.get_parameter('batch.recalculate.for.motion').get_value():
            self.__calculate_requirements_of_motion(sarc_obj, model)
            pass

        sarc_obj.detect_lois(frame=model.parameters.get_parameter(name='loi.detect.frame').get_value(),
                             n_lois=model.parameters.get_parameter(name='loi.detect.n_lois').get_value(),
                             ratio_seeds=model.parameters.get_parameter(name='loi.detect.ratio_seeds').get_value(),
                             persistence=model.parameters.get_parameter(name='loi.detect.persistence').get_value(),
                             threshold_distance=model.parameters.get_parameter(
                                 name='loi.detect.threshold_distance').get_value(),
                             mode=model.parameters.get_parameter(name='loi.detect.mode').get_value(),
                             number_lims=(
                                 model.parameters.get_parameter(name='loi.detect.number_limits_lower').get_value(),
                                 model.parameters.get_parameter(name='loi.detect.number_limits_upper').get_value()),
                             length_lims=(
                                 model.parameters.get_parameter(name='loi.detect.length_limits_lower').get_value(),
                                 model.parameters.get_parameter(name='loi.detect.length_limits_upper').get_value()),
                             sarcomere_mean_length_lims=(model.parameters.get_parameter(
                                 name='loi.detect.sarcomere_mean_length_limits_lower').get_value(),
                                                         model.parameters.get_parameter(
                                                             name='loi.detect.sarcomere_mean_length_limits_upper').get_value()),
                             sarcomere_std_length_lims=(model.parameters.get_parameter(
                                 name='loi.detect.sarcomere_std_length_limits_lower').get_value(),
                                                        model.parameters.get_parameter(
                                                            name='loi.detect.sarcomere_std_length_limits_upper').get_value()),
                             midline_mean_length_lims=(model.parameters.get_parameter(
                                 name='loi.detect.midline_mean_length_limits_lower').get_value(),
                                                       model.parameters.get_parameter(
                                                           name='loi.detect.midline_mean_length_limits_upper').get_value()),
                             midline_std_length_lims=(model.parameters.get_parameter(
                                 name='loi.detect.midline_std_length_limits_lower').get_value(),
                                                      model.parameters.get_parameter(
                                                          name='loi.detect.midline_std_length_limits_upper').get_value()),
                             midline_min_length_lims=(model.parameters.get_parameter(
                                 name='loi.detect.midline_min_length_limits_lower').get_value(),
                                                      model.parameters.get_parameter(
                                                          name='loi.detect.midline_min_length_limits_upper').get_value()),
                             distance_threshold_lois=model.parameters.get_parameter(
                                 name='loi.detect.cluster_threshold_lois').get_value(),
                             linkage=model.parameters.get_parameter(name='loi.detect.linkage').get_value(),
                             linewidth=model.parameters.get_parameter(name='loi.detect.line_width').get_value(),
                             order=model.parameters.get_parameter(name='loi.detect.order').get_value())

        lois = Utils.get_lois_of_file(file)
        for file, loi in lois:
            try:
                motion_obj = Motion(file, loi)
                self.__single_motion_loi_analysis(motion_obj, model)
                pass
            except Exception as e:
                # this part has to be added to qt thread
                qtutils.inmain(self.__main_control.debug,
                               message='Exception happened during processing of file:' + file)
                qtutils.inmain(self.__main_control.debug, message='message:' + repr(e))
                qtutils.inmain(self.__main_control.debug, message='')
                # todo: add log file to batch processing
                pass
            pass

        if model.parameters.get_parameter('batch.delete_intermediary_tiffs').get_value():
            sarc_obj.remove_intermediate_tiffs()

        pass

    def __single_motion_loi_analysis(self, motion_obj: Motion, model):
        auto_save_ = motion_obj.auto_save
        motion_obj.auto_save = False
        motion_obj.detekt_peaks(thres=model.parameters.get_parameter('motion.detect_peaks.threshold').get_value(),
                                min_dist=model.parameters.get_parameter('motion.detect_peaks.min_distance').get_value(),
                                width=model.parameters.get_parameter('motion.detect_peaks.width').get_value())

        motion_obj.track_z_bands(
            search_range=model.parameters.get_parameter('motion.track_z_bands.search_range').get_value(),
            memory_tracking=model.parameters.get_parameter('motion.track_z_bands.memory').get_value(),
            memory_interpol=model.parameters.get_parameter('motion.track_z_bands.memory_interpolation').get_value())

        motion_obj.detect_analyze_contractions(
            model=model.parameters.get_parameter('motion.systoles.weights').get_value(),
            threshold=model.parameters.get_parameter('motion.systoles.threshold').get_value(),
            slen_lims=(model.parameters.get_parameter('motion.systoles.slen_limits.lower').get_value(),
                       model.parameters.get_parameter('motion.systoles.slen_limits.upper').get_value()),
            n_sarcomeres_min=model.parameters.get_parameter('motion.systoles.n_sarcomeres_min').get_value(),
            buffer_frames=model.parameters.get_parameter('motion.systoles.buffer_frames').get_value(),
            contr_time_min=model.parameters.get_parameter('motion.systoles.contr_time_min').get_value(),
            merge_time_max=model.parameters.get_parameter('motion.systoles.merge_time_max').get_value())

        motion_obj.get_trajectories(
            slen_lims=(
                model.parameters.get_parameter('motion.get_sarcomere_trajectories.s_length_limits_lower').get_value(),
                model.parameters.get_parameter('motion.get_sarcomere_trajectories.s_length_limits_upper').get_value()),
            dilate_contr=model.parameters.get_parameter(
                'motion.get_sarcomere_trajectories.dilate_systoles').get_value(),
            filter_params_vel=(model.parameters.get_parameter(
                'motion.get_sarcomere_trajectories.filter_params_vel.window_length').get_value(),
                               model.parameters.get_parameter(
                                   'motion.get_sarcomere_trajectories.filter_params_vel.polyorder').get_value()),
            equ_lims=(model.parameters.get_parameter('motion.get_sarcomere_trajectories.equ_limits_lower').get_value(),
                      model.parameters.get_parameter('motion.get_sarcomere_trajectories.equ_limits_upper').get_value()))
        motion_obj.analyze_trajectories()
        motion_obj.analyze_popping()  # todo: implement on ui?
        motion_obj.auto_save = auto_save_
        motion_obj.store_loi_data()
        pass

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
        if network_model == 'generalist':
            network_model = None
            pass

        size: Union[Tuple[int, int]] = (model.parameters.get_parameter('structure.predict.size_width').get_value(),
                                        model.parameters.get_parameter('structure.predict.size_height').get_value())

        sarc_obj.detect_sarcomeres(frames=model.parameters.get_parameter('structure.frames').get_value(),
                                   model_path=network_model,
                                   max_patch_size=size,
                                   rescale_factor=model.parameters.get_parameter('structure.predict.rescale_factor').get_value(),
                                   clip_thres=(
                                       model.parameters.get_parameter('structure.predict.clip_thresh_min').get_value(),
                                       model.parameters.get_parameter('structure.predict.clip_thresh_max').get_value()),
                                   )

        # analyze cell mask and sarcomere area
        if model.parameters.get_parameter('batch.do_cellmask').get_value():
            sarc_obj.analyze_cell_mask()
        # analyze sarcomere structures
        if model.parameters.get_parameter('batch.do_zbands').get_value():
            sarc_obj.analyze_z_bands(frames=model.parameters.get_parameter('structure.frames').get_value(),
                                     threshold=model.parameters.get_parameter(
                                         'structure.z_band_analysis.threshold').get_value(),
                                     min_length=model.parameters.get_parameter(
                                         'structure.z_band_analysis.min_length').get_value())

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
                interp_factor=model.parameters.get_parameter('structure.vectors.interpolation_factor').get_value()
            )

        if model.parameters.get_parameter('batch.do_myofibrils').get_value():
            sarc_obj.analyze_myofibrils(
                frames=model.parameters.get_parameter('structure.frames').get_value(),
                ratio_seeds=model.parameters.get_parameter('structure.myofibril.ratio_seeds').get_value(),
                persistence=model.parameters.get_parameter('structure.myofibril.persistence').get_value(),
                threshold_distance=model.parameters.get_parameter('structure.myofibril.threshold_distance').get_value(),
                n_min=model.parameters.get_parameter('structure.myofibril.n_min').get_value()
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
        if model.parameters.get_parameter('batch.delete_intermediary_tiffs').get_value():
            sarc_obj.remove_intermediate_tiffs()

        pass

    def on_search(self):
        # f_name is a tuple
        file = str(QFileDialog.getExistingDirectory(caption="Select Root Directory"))
        if file is not None and file is not '':
            self.__batch_processing_widget.le_root_directory.setText(file)
        pass

    pass