import glob

import qtutils
import traceback
from PyQt5.QtWidgets import QFileDialog
from multiprocessing import Pool

from ..view.parameters_batch_processing import Ui_Form as BatchProcessingWidget
from .application_control import ApplicationControl
from ... import SarcAsM, Utils, Motion
from ...meta_data_handler import MetaDataHandler
from biu.progress import ProgressNotifier


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

        parameters.get_parameter(name='batch.pixel.size').connect(widget.dsb_pixel_size)
        parameters.get_parameter(name='batch.frame.time').connect(widget.dsb_frame_time)
        parameters.get_parameter(name='batch.force.override').connect(widget.chk_force_override)
        parameters.get_parameter(name='batch.thread_pool_size').connect(widget.sb_thread_pool_size)
        parameters.get_parameter(name='batch.root').connect(widget.le_root_directory)
        parameters.get_parameter(name='batch.recalculate.for.motion').connect(widget.chk_calc_rois)

        pass

    def on_btn_batch_processing_structure(self):

        tif_files = glob.glob(self.__batch_processing_widget.le_root_directory.text() + '*/*.tif')
        print(len(tif_files))

        # todo: call synchronous (with freezing ui) for testing if there is some issue with threading
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__batch_process_structure_async,
                                                   start_message='Start batch processing structure ',
                                                   finished_message='Finished batch processing structure ')
        self.__worker = worker

        pass

    def __batch_process_structure_async(self, worker, model):
        progress_notifier = ProgressNotifier()
        progress_notifier.set_progress_report(lambda p: worker.progress.emit(p * 100))
        progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: worker.progress_details.emit(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (
                    hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))

        tif_files = glob.glob(model.parameters.get_parameter(name='batch.root').get_value() + '*/*.tif')

        n_pools = model.parameters.get_parameter(name='batch.thread_pool_size').get_value()
        frame_time = model.parameters.get_parameter(name='batch.frame.time').get_value()
        pixel_size = model.parameters.get_parameter(name='batch.pixel.size').get_value()
        force_override = model.parameters.get_parameter(name='batch.force.override').get_value()

        for i, file in enumerate(progress_notifier.iterator(tif_files)):
            try:
                self.__single_structure_analysis(file, frame_time, pixel_size, force_override, model)
            except Exception as e:
                # this part has to be added to qt thread
                qtutils.inmain(self.__main_control.debug,
                               message='Exception happened during processing of file:' + file)
                qtutils.inmain(self.__main_control.debug, message='message:' + repr(e))
                qtutils.inmain(self.__main_control.debug, message='')
                traceback.print_exception(e)
                # todo: add log file to batch processing

                pass
            pass

        # todo: parallel method 1
        # with Pool(n_pools) as p:
        #    p.map(lambda f: self.__single_structure_analysis(f, frame_time, pixel_size, force_override,worker), tif_files)

        # todo: parallel method 2
        # from joblib import Parallel, delayed
        # def yourfunction(k):
        #    s = 3.14 * k * k
        #    print
        #    "Area of a circle with a radius ", k, " is:", s
        # element_run = Parallel(n_jobs=-1)(delayed(yourfunction)(k) for k in range(1, 10))

        # todo: parallel method 3  ---> this one seems programming wise the most appropriate?
        # from dask.distributed import Client
        # client = Client(n_workers=8) # In this example I have 8 cores and processes (can also use threads if desired)
        # def my_function(i):
        #    output = <code to execute in the for loop here>
        #    return output
        # futures = []
        # for i in <whatever you want to loop across here>:
        #    future = client.submit(my_function, i)
        #    futures.append(future)
        # results = client.gather(futures)
        # client.close()

    pass

    def on_btn_batch_processing_motion(self):
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__batch_process_motion_async,
                                                   start_message='Start batch processing motion ',
                                                   finished_message='Finished batch processing motion ')
        self.__worker = worker
        pass

    def __batch_process_motion_async(self, worker, model):
        progress_notifier = ProgressNotifier()
        progress_notifier.set_progress_report(lambda p: worker.progress.emit(p * 100))
        progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: worker.progress_details.emit(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (
                    hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))

        tif_files = glob.glob(model.parameters.get_parameter(name='batch.root').get_value() + '*/*.tif')
        n_pools = model.parameters.get_parameter(name='batch.thread_pool_size').get_value()
        frame_time = model.parameters.get_parameter(name='batch.frame.time').get_value()
        pixel_size = model.parameters.get_parameter(name='batch.pixel.size').get_value()
        force_override = model.parameters.get_parameter(name='batch.force.override').get_value()

        for i, file in enumerate(progress_notifier.iterator(tif_files)):
            try:
                self.__single_motion_analysis(file, frame_time, pixel_size, force_override, model)
            except Exception as e:
                # this part has to be added to qt thread
                qtutils.inmain(self.__main_control.debug,
                               message='Exception happened during processing of file:' + file)
                qtutils.inmain(self.__main_control.debug, message='message:' + repr(e))
                qtutils.inmain(self.__main_control.debug, message='')
                # todo: add log file to batch processing
                pass
            pass
        pass

    @staticmethod
    def __get_sarc_object(file: str, frame_time: float, pixel_size: float, force_override: bool) -> SarcAsM:
        has_metadata = MetaDataHandler.check_meta_data_exists(file)
        sarc_obj = SarcAsM(file, use_gui=True)
        if not has_metadata or force_override:
            sarc_obj.metadata['pixelsize'] = pixel_size
            sarc_obj.metadata['frametime'] = frame_time
            sarc_obj.meta_data_handler.store_meta_data(True)  # store meta-data and override if necessary
            sarc_obj.meta_data_handler.commit()
            pass
        return sarc_obj
        pass

    def __calculate_requirements_of_motion(self, sarc_obj: SarcAsM, model):
        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None
        sarc_obj.structure.predict_z_bands(model_path=network_model,
                                           time_consistent=model.parameters.get_parameter(
                                               'structure.predict.time_consistent').get_value(),
                                           size=(
                                               model.parameters.get_parameter(
                                                   'structure.predict.size_width').get_value(),
                                               model.parameters.get_parameter(
                                                   'structure.predict.size_height').get_value()
                                           ),
                                           clip_thres=(
                                               model.parameters.get_parameter(
                                                   'structure.predict.clip_thresh_min').get_value(),
                                               model.parameters.get_parameter(
                                                   'structure.predict.clip_thresh_max').get_value()
                                           ))
        sarc_obj.structure.analyze_sarcomere_length_orient(
            frames=model.parameters.get_parameter('loi.detect.frame').get_value(),
            size=model.parameters.get_parameter('structure.wavelet.filter_size').get_value(),
            sigma=model.parameters.get_parameter('structure.wavelet.sigma').get_value(),
            width=model.parameters.get_parameter('structure.wavelet.width').get_value(),
            len_lims=(
                model.parameters.get_parameter('structure.wavelet.length_limit_lower').get_value(),
                model.parameters.get_parameter('structure.wavelet.length_limit_upper').get_value()
            ),
            len_step=model.parameters.get_parameter('structure.wavelet.length_step').get_value(),
            orient_lims=(
                model.parameters.get_parameter('structure.wavelet.orientation_limit_lower').get_value(),
                model.parameters.get_parameter('structure.wavelet.orientation_limit_upper').get_value()
            ),
            orient_step=model.parameters.get_parameter('structure.wavelet.orientation_step').get_value(),
            score_threshold=model.parameters.get_parameter('structure.wavelet.score_threshold').get_value(),
            abs_threshold=model.parameters.get_parameter('structure.wavelet.absolute_threshold').get_value(),
            save_all=model.parameters.get_parameter('structure.wavelet.save_all').get_value()
        )
        pass

    def __single_motion_analysis(self, file: str, frame_time: float, pixel_size: float, force_override: bool,
                                 model):
        sarc_obj = BatchProcessingControl.__get_sarc_object(file=file, frame_time=frame_time, pixel_size=pixel_size,
                                                            force_override=force_override)
        # add some flag if those calculations should be done or not
        if model.parameters.get_parameter('batch.recalculate.for.motion').get_value():
            self.__calculate_requirements_of_motion(sarc_obj, model)
            pass

        sarc_obj.structure.detect_lois(frame=model.parameters.get_parameter('loi.detect.frame').get_value(),
                                       persistence=model.parameters.get_parameter('loi.detect.persistence').get_value(),
                                       threshold_distance=model.parameters.get_parameter(
                                           'loi.detect.threshold_distance').get_value(),
                                       score_threshold=None if model.parameters.get_parameter(
                                           'loi.detect.score_threshold_automatic').get_value() else model.parameters.get_parameter(
                                           'loi.detect.score_threshold').get_value(),
                                       number_lims=(
                                           model.parameters.get_parameter('loi.detect.number_limits_lower').get_value(),
                                           model.parameters.get_parameter('loi.detect.number_limits_upper').get_value()
                                       ),
                                       msc_lims=(
                                           model.parameters.get_parameter('loi.detect.msc_limits_lower').get_value(),
                                           model.parameters.get_parameter('loi.detect.msc_limits_upper').get_value()),
                                       distance_threshold_lois=model.parameters.get_parameter(
                                           'loi.detect.distance_threshold_lois').get_value(),
                                       n_longest=model.parameters.get_parameter('loi.detect.n_longest').get_value(),
                                       linewidth=model.parameters.get_parameter('loi.detect.line_width').get_value())
        rois = Utils.get_lois_of_cell(file)
        for file, roi in rois:
            try:
                motion_obj = Motion(file, roi)
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

    def __single_structure_analysis(self, file: str, frame_time: float, pixel_size: float, force_override: bool,
                                    model):
        # attention: this method is not executed in qt thread! --> every information to ui needs to be done either
        # on another place or within a wrapper for QT Main thread (like the package qtutils.inmain does)

        # initialize SarcAsM object
        # check for metadata
        sarc_obj = BatchProcessingControl.__get_sarc_object(file=file, frame_time=frame_time, pixel_size=pixel_size,
                                                            force_override=force_override)

        frames = model.parameters.get_parameter('structure.frames').get_value()

        # predict sarcomere z-bands and cell area
        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None
        sarc_obj.structure.predict_z_bands(model_path=network_model,
                                           time_consistent=model.parameters.get_parameter(
                                               'structure.predict.time_consistent').get_value(),
                                           size=(
                                               model.parameters.get_parameter(
                                                   'structure.predict.size_width').get_value(),
                                               model.parameters.get_parameter(
                                                   'structure.predict.size_height').get_value()
                                           ),
                                           clip_thres=(
                                               model.parameters.get_parameter(
                                                   'structure.predict.clip_thresh_min').get_value(),
                                               model.parameters.get_parameter(
                                                   'structure.predict.clip_thresh_max').get_value()
                                           ))
        sarc_obj.structure.predict_cell_area(model_path=network_model,
                                             size=(
                                                 model.parameters.get_parameter(
                                                     'structure.predict.cell_area.size_width').get_value(),
                                                 model.parameters.get_parameter(
                                                     'structure.predict.cell_area.size_height').get_value()
                                             ),
                                             clip_thres=(
                                                 model.parameters.get_parameter(
                                                     'structure.predict.cell_area.clip_thresh_min').get_value(),
                                                 model.parameters.get_parameter(
                                                     'structure.predict.cell_area.clip_thresh_max').get_value()
                                             ))
        # analyze cell area and sarcomere area
        sarc_obj.structure.analyze_cell_area(frames=frames)
        # analyze sarcomere structures
        sarc_obj.structure.analyze_z_bands(
            frames=frames,
            threshold=model.parameters.get_parameter('structure.z_band_analysis.threshold').get_value(),
            min_length=model.parameters.get_parameter('structure.z_band_analysis.min_length').get_value())

        # careful this method highly depends on pixel size setting
        sarc_obj.structure.analyze_sarcomere_length_orient(
            frames=frames,
            size=model.parameters.get_parameter('structure.wavelet.filter_size').get_value(),
            sigma=model.parameters.get_parameter('structure.wavelet.sigma').get_value(),
            width=model.parameters.get_parameter('structure.wavelet.width').get_value(),
            len_lims=(
                model.parameters.get_parameter('structure.wavelet.length_limit_lower').get_value(),
                model.parameters.get_parameter('structure.wavelet.length_limit_upper').get_value()
            ),
            len_step=model.parameters.get_parameter('structure.wavelet.length_step').get_value(),
            orient_lims=(
                model.parameters.get_parameter('structure.wavelet.orientation_limit_lower').get_value(),
                model.parameters.get_parameter('structure.wavelet.orientation_limit_upper').get_value()
            ),
            orient_step=model.parameters.get_parameter('structure.wavelet.orientation_step').get_value(),
            score_threshold=model.parameters.get_parameter('structure.wavelet.score_threshold').get_value(),
            abs_threshold=model.parameters.get_parameter('structure.wavelet.absolute_threshold').get_value(),
            save_all=model.parameters.get_parameter('structure.wavelet.save_all').get_value()
        )

        sarc_obj.structure.analyze_myofibrils(
            frames=frames,
            n_seeds=model.parameters.get_parameter('structure.myofibril.n_seeds').get_value(),
            score_threshold=None if model.parameters.get_parameter(
                'structure.myofibril.score_threshold_empty').get_value() else model.parameters.get_parameter(
                'structure.myofibril.score_threshold').get_value(),
            persistence=model.parameters.get_parameter('structure.myofibril.persistence').get_value(),
            threshold_distance=model.parameters.get_parameter('structure.myofibril.threshold_distance').get_value()
        )

        sarc_obj.structure.analyze_sarcomere_domains(
            frames=frames,
            score_threshold=model.parameters.get_parameter('structure.domain.analysis.score_threshold').get_value(),
            reduce=model.parameters.get_parameter('structure.domain.analysis.reduce').get_value(),
            weight_length=model.parameters.get_parameter('structure.domain.analysis.weight_length').get_value(),
            distance_threshold=model.parameters.get_parameter(
                'structure.domain.analysis.distance_threshold').get_value(),
            area_min=model.parameters.get_parameter('structure.domain.analysis.area_min').get_value()
        )
        sarc_obj.structure.store_structure_data()
        pass

    def on_search(self):
        # f_name is a tuple
        file = str(QFileDialog.getExistingDirectory(caption="Select Root Directory"))
        if file is not None and file is not '':
            self.__batch_processing_widget.le_root_directory.setText(file)
        pass

    pass
