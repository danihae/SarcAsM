import glob

import qtutils
from PyQt5.QtWidgets import QFileDialog
from multiprocessing import Pool
from ..view.parameters_batch_processing import Ui_Form as BatchProcessingWidget
from .application_control import ApplicationControl
from ... import SarcAsM, MetaDataHandler, ProgressNotifier


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

        pass

    # todo: implement batch processing functionality
    def on_btn_batch_processing_structure(self):

        tif_files = glob.glob(self.__batch_processing_widget.le_root_directory.text() + '*/*.tif')
        print(len(tif_files))

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
                self.__single_structure_analysis(file, frame_time, pixel_size, force_override)
            except Exception as e:
                # this part has to be added to qt thread
                qtutils.inmain(self.__main_control.debug, message='Exception happened during processing of file:'+file)
                qtutils.inmain(self.__main_control.debug, message='message:' + repr(e))
                qtutils.inmain(self.__main_control.debug, message='')

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
        pass

    def __single_structure_analysis(self, file: str, frame_time: float, pixel_size: float, force_override: bool):
        # attention: this method is not executed in qt thread! --> every information to ui needs to be done either
        # on another place or within a wrapper for QT Main thread (like the package inmain does)

        # initialize SarcAsM object
        # todo check for metadata
        has_metadata = MetaDataHandler.check_meta_data_exists(file)
        sarc_obj = SarcAsM(file, use_gui=True)

        if not has_metadata or force_override:
            sarc_obj.metadata['pixelsize'] = pixel_size
            sarc_obj.metadata['frametime'] = frame_time
            sarc_obj.meta_data_handler.store_meta_data(True)  # store meta-data and override if necessary
            sarc_obj.meta_data_handler.commit()
            pass

        timepoints = 'all'

        # predict sarcomere z-bands and cell area
        sarc_obj.structure.predict_z_bands(size=(2048, 2048))
        sarc_obj.structure.predict_cell_area(size=(2048, 2048))

        # analyze cell area and sarcomere area
        sarc_obj.structure.analyze_cell_area(timepoints=timepoints)
        # analyze sarcomere structures
        sarc_obj.structure.analyze_z_bands(timepoints=timepoints)

        # this method here takes very big amount of memory ~25GB,
        # at least with the current images used for testing (maybe get some other images)
        # on test image real_data_E5_frame0_to29.tif it seems to freeze on sarcomere_length_orient,
        # at least tqdm hasn't shown any progress during a long time period
        sarc_obj.structure.analyze_sarcomere_length_orient(timepoints=timepoints)

        sarc_obj.structure.analyze_myofibrils(timepoints=timepoints)
        sarc_obj.structure.analyze_sarcomere_domains(timepoints=timepoints)
        sarc_obj.structure.store_structure_data()

    def on_search(self):
        # f_name is a tuple
        file = str(QFileDialog.getExistingDirectory(caption="Select Root Directory"))
        if file is not None and file is not '':
            self.__batch_processing_widget.le_root_directory.setText(file)
        pass

    pass
