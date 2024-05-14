from PyQt5.QtWidgets import QFileDialog
from biu.progress import ProgressNotifier

from .chain_execution import ChainExecution
from .application_control import ApplicationControl
from ..view.parameter_structure_analysis import Ui_Form as StructureAnalysisWidget
from ..model import ApplicationModel


class StructureAnalysisControl:
    """
    Handles button calls, parameter changes etc. from structure view
    """

    def __init__(self, structure_parameters_widget: StructureAnalysisWidget, main_control: ApplicationControl):
        self.__structure_parameters_widget = structure_parameters_widget
        self.__main_control = main_control
        self.__thread = None
        self.__worker = None

    @staticmethod
    def __predict_call(worker, model: ApplicationModel):

        progress_notifier = ProgressNotifier()
        progress_notifier.set_progress_report(lambda p: worker.progress.emit(p * 100))
        progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: worker.progress_details.emit(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (
                    hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))

        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None

        model.cell.structure.predict_z_bands(progress_notifier=progress_notifier,
                                             model_path=network_model,
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
        pass

    def __cell_area_predict_call(self, worker, model):
        progress_notifier = ProgressNotifier()
        progress_notifier.set_progress_report(lambda p: worker.progress.emit(p * 100))
        progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: worker.progress_details.emit(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (
                    hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))
        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None

        model.cell.structure.predict_cell_area(progress_notifier=progress_notifier,
                                               model_path=network_model,
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
        model.cell.structure.analyze_cell_area()
        pass

    def __chk_prediction_network(self):  # todo rename to zband_prediction or similar
        if self.__main_control.model.parameters.get_parameter('structure.predict.network_path').get_value() == '':
            self.__main_control.debug('no network file was chosen for prediction')
            return False
        return True

    def __chk_cell_area_prediction_network(self):
        if self.__main_control.model.parameters.get_parameter(
                'structure.predict.cell_area.network_path').get_value() == '':
            self.__main_control.debug('no network file was chosen for cell area prediction')
            return False
        return True

    def __chk_timepoints(self):
        timepoints = self.__main_control.model.parameters.get_parameter('structure.timepoints').get_value()
        if timepoints is None or timepoints == '':
            self.__check_timepoint_syntax()
            self.__main_control.debug(
                'no time points selected, please select the time point(s) in the specified format')
            return False
        return True

    def __chk_initialized(self):
        if not self.__main_control.model.is_initialized():
            self.__main_control.debug('file is not correctly initialized (or viewer was closed)')
            return False
        return True

    def on_btn_cell_level_analysis(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_prediction_network():
            return
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__predict_call,
                                                   start_message='Start prediction of sarcomere z-bands',
                                                   finished_message=f'Z-bands detected and saved in {self.__main_control.model.cell.folder}',
                                                   finished_action=self.__predict_finished,
                                                   finished_successful_action=self.__main_control.model.cell.structure.commit)
        self.__worker = worker
        return worker

    def on_btn_cell_area_predict(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_cell_area_prediction_network():
            return

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__cell_area_predict_call,
                                                   start_message='Start prediction of cell area',
                                                   finished_message='Finished prediction of cell area',
                                                   finished_action=self.__predict_cell_area_finished,
                                                   finished_successful_action=self.__main_control.model.cell.structure.commit)
        self.__worker = worker
        return worker

    def __parse_timepoints(self, timepoints_str: str):
        if timepoints_str == '':
            return None
        if timepoints_str.lower().__eq__('all'):
            return timepoints_str.lower()
        if timepoints_str.isnumeric():
            return int(timepoints_str)
        if timepoints_str.__contains__(','):
            list_str = timepoints_str.split(',')
            parsed_list = []
            for x in list_str:
                if x.isnumeric():
                    parsed_list.append(int(x))
            return parsed_list
        return 0  # if it's a wrong value just process first image (processing all could take long for a wrong input)

    def on_btn_z_band(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_timepoints():
            return
        call_lambda = lambda w, m: m.cell.structure.analyze_z_bands(
            timepoints=m.parameters.get_parameter('structure.timepoints').get_value(),
            threshold=m.parameters.get_parameter('structure.z_band_analysis.threshold').get_value(),
            min_length=m.parameters.get_parameter('structure.z_band_analysis.min_length').get_value())

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=call_lambda,
                                                   start_message='Start Z-band Analysis',
                                                   finished_message='Finished Z-band Analysis',
                                                   finished_successful_action=self.__main_control.model.cell.structure.commit)
        self.__worker = worker
        return worker

    def on_btn_wavelet(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_timepoints():
            return
        call_lambda = lambda w, m: m.cell.structure.analyze_sarcomere_length_orient(
            timepoints=m.parameters.get_parameter('structure.timepoints').get_value(),
            size=m.parameters.get_parameter('structure.wavelet.filter_size').get_value(),
            sigma=m.parameters.get_parameter('structure.wavelet.sigma').get_value(),
            width=m.parameters.get_parameter('structure.wavelet.width').get_value(),
            len_lims=(
                m.parameters.get_parameter('structure.wavelet.length_limit_lower').get_value(),
                m.parameters.get_parameter('structure.wavelet.length_limit_upper').get_value()
            ),
            len_step=m.parameters.get_parameter('structure.wavelet.length_step').get_value(),
            orient_lims=(
                m.parameters.get_parameter('structure.wavelet.orientation_limit_lower').get_value(),
                m.parameters.get_parameter('structure.wavelet.orientation_limit_upper').get_value()
            ),
            orient_step=m.parameters.get_parameter('structure.wavelet.orientation_step').get_value(),
            score_threshold=m.parameters.get_parameter('structure.wavelet.score_threshold').get_value(),
            abs_threshold=m.parameters.get_parameter('structure.wavelet.absolute_threshold').get_value(),
            save_all=m.parameters.get_parameter('structure.wavelet.save_all').get_value()
        )
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=call_lambda,
                                                   finished_message='Finished wavelet analysis',
                                                   start_message='Start wavelet analysis',
                                                   finished_successful_action=self.__main_control.model.cell.structure.commit)
        self.__worker = worker
        return worker
        # AND-gated double wavelet analysis of sarcomere structure to locally obtain length and angle

        pass

    def on_btn_myofibril(self):
        """
        analyze_myofibrils(self, timepoints=None, n_seeds=200, score_threshold=None, persistence=3,
                           threshold_distance=0.3,
                           save_all=False, plot=False)
        """
        if not self.__chk_initialized():
            return
        if not self.__chk_timepoints():
            return
        # estimate myofibril lengths using line-growth algorithm
        # cell.structure.get_myofibril_lengths(plot=True)
        call_lambda = lambda w, m: m.cell.structure.analyze_myofibrils(
            timepoints=m.parameters.get_parameter('structure.timepoints').get_value(),
            n_seeds=m.parameters.get_parameter('structure.myofibril.n_seeds').get_value(),
            score_threshold=None if m.parameters.get_parameter(
                'structure.myofibril.score_threshold_empty').get_value() else m.parameters.get_parameter(
                'structure.myofibril.score_threshold').get_value(),
            persistence=m.parameters.get_parameter('structure.myofibril.persistence').get_value(),
            threshold_distance=m.parameters.get_parameter('structure.myofibril.threshold_distance').get_value()
        )

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=call_lambda,
                                                   start_message='Start myofibril analysis',
                                                   finished_message='Finished myofibril analysis',
                                                   finished_successful_action=self.__main_control.model.cell.structure.commit)
        self.__worker = worker
        return worker
        pass

    def on_btn_domain_analysis(self):
        """
        call domain analysis in backend
        """
        if not self.__chk_initialized():
            return
        if not self.__chk_timepoints():
            return

        call_lambda = lambda w, m: m.cell.structure.analyze_sarcomere_domains(
            timepoints=m.parameters.get_parameter('structure.timepoints').get_value(),
            score_threshold=m.parameters.get_parameter('structure.domain.analysis.score_threshold').get_value(),
            reduce=m.parameters.get_parameter('structure.domain.analysis.reduce').get_value(),
            weight_length=m.parameters.get_parameter('structure.domain.analysis.weight_length').get_value(),
            distance_threshold=m.parameters.get_parameter('structure.domain.analysis.distance_threshold').get_value(),
            area_min=m.parameters.get_parameter('structure.domain.analysis.area_min').get_value()
        )
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=call_lambda,
                                                   start_message='Start sarcomere domains analysis',
                                                   finished_message='Finished sarcomere domains analysis',
                                                   finished_successful_action=self.__main_control.model.cell.structure.commit)
        self.__worker = worker
        return worker

        pass

    def on_btn_search_network(self):
        # f_name is a tuple
        f_name = QFileDialog.getOpenFileName(caption='Open Network File', filter="Network Files (*.pth)")
        if f_name is not None:
            self.__structure_parameters_widget.le_network.setText(f_name[0])

    def on_btn_cell_area_search_network(self):
        f_name = QFileDialog.getOpenFileName(caption='Open Network File', filter="Network Files (*.pth)")
        if f_name is not None:
            self.__structure_parameters_widget.le_cell_area_network.setText(f_name[0])

    def __filter_input_prediction_size(self, element):
        if not (hasattr(element, 'value') and hasattr(element, 'setValue')):
            return

        value = element.value()
        factor = value // 16
        if factor * 16 != value:
            element.setValue(factor * 16)
            pass
        pass

    def __check_timepoint_syntax(self):
        text = self.__structure_parameters_widget.le_general_timepoints.text()
        value = self.__parse_timepoints(text)
        if not text.isnumeric() and (value == 0 or value is None):
            # this is an error
            self.__structure_parameters_widget.le_general_timepoints.setStyleSheet("QLineEdit{background : red;}")
            pass
        else:
            self.__structure_parameters_widget.le_general_timepoints.setStyleSheet(
                "QLineEdit{background : lightgreen;}")
        pass

    def on_analyze_structure(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_prediction_network():
            return
        if not self.__chk_timepoints():
            return
        if not self.__chk_cell_area_prediction_network():
            return
        # predict, z band analysis, wavelet analysis, myofibril length
        chain = ChainExecution(self.__main_control.model.currentlyProcessing, self.__main_control.debug)
        chain.add_step(self.on_btn_cell_level_analysis)
        # chain.add_step(self.on_btn_cell_area_predict) # excluded from analyze structure
        chain.add_step(self.on_btn_z_band)
        chain.add_step(self.on_btn_wavelet)
        chain.add_step(self.on_btn_myofibril)
        chain.add_step(self.on_btn_domain_analysis)
        chain.execute()
        pass

    def bind_events(self):
        """
        Binds ui events to backend methods/functions
        also binds ui fields to model parameters
        """
        self.__structure_parameters_widget.btn_analyze_structure.clicked.connect(self.on_analyze_structure)

        # monitor the value of predict_size and keep it dividable by 16
        self.__structure_parameters_widget.sb_predict_size_min.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_predict_size_min))
        self.__structure_parameters_widget.sb_predict_size_max.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_predict_size_max))

        self.__structure_parameters_widget.sb_cell_area_size_width.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_cell_area_size_width))
        self.__structure_parameters_widget.sb_cell_area_size_height.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_cell_area_size_height))

        self.__structure_parameters_widget.le_general_timepoints.editingFinished.connect(self.__check_timepoint_syntax)

        self.__structure_parameters_widget.btn_structure_cell_area_predict.clicked.connect(
            self.on_btn_cell_area_predict)
        self.__structure_parameters_widget.btn_structure_predict.clicked.connect(self.on_btn_cell_level_analysis)
        self.__structure_parameters_widget.btn_structure_z_band.clicked.connect(self.on_btn_z_band)
        self.__structure_parameters_widget.btn_structure_wavelet.clicked.connect(self.on_btn_wavelet)
        self.__structure_parameters_widget.btn_structure_myofibril.clicked.connect(self.on_btn_myofibril)
        self.__structure_parameters_widget.btn_search_network.clicked.connect(self.on_btn_search_network)
        self.__structure_parameters_widget.btn_cell_area_network_search.clicked.connect(
            self.on_btn_cell_area_search_network)
        self.__structure_parameters_widget.btn_structure_domain_analysis.clicked.connect(self.on_btn_domain_analysis)

        # todo: bind parameters to ui elements
        parameters = self.__main_control.model.parameters
        widget = self.__structure_parameters_widget

        parameters.get_parameter(name='structure.predict.network_path').connect(widget.le_network)
        parameters.get_parameter(name='structure.predict.time_consistent').connect(widget.chk_time_consistent)
        parameters.get_parameter(name='structure.predict.size_width').connect(widget.sb_predict_size_min)
        parameters.get_parameter(name='structure.predict.size_height').connect(widget.sb_predict_size_max)
        parameters.get_parameter(name='structure.predict.clip_thresh_min').connect(widget.dsb_clip_thresh_min)
        parameters.get_parameter(name='structure.predict.clip_thresh_max').connect(widget.dsb_clip_thresh_max)

        parameters.get_parameter(name='structure.predict.cell_area.network_path').connect(widget.le_cell_area_network)
        parameters.get_parameter(name='structure.predict.cell_area.size_width').connect(widget.sb_cell_area_size_width)
        parameters.get_parameter(name='structure.predict.cell_area.size_height').connect(
            widget.sb_cell_area_size_height)
        parameters.get_parameter(name='structure.predict.cell_area.clip_thresh_min').connect(
            widget.dsb_cell_area_clip_threshold_min)
        parameters.get_parameter(name='structure.predict.cell_area.clip_thresh_max').connect(
            widget.dsb_cell_area_clip_threshold_max)

        parameters.get_parameter(name='structure.timepoints').set_value_parser(self.__parse_timepoints)
        parameters.get_parameter(name='structure.timepoints').connect(widget.le_general_timepoints)

        parameters.get_parameter(name='structure.z_band_analysis.threshold').connect(widget.dsb_z_band_threshold)
        parameters.get_parameter(name='structure.z_band_analysis.min_length').connect(widget.dsb_z_band_min_length)
        parameters.get_parameter(name='structure.wavelet.filter_size').connect(widget.dsb_wavelet_filter_size)
        parameters.get_parameter(name='structure.wavelet.sigma').connect(widget.dsb_wavelet_sigma)
        parameters.get_parameter(name='structure.wavelet.width').connect(widget.dsb_wavelet_width)
        parameters.get_parameter(name='structure.wavelet.length_limit_lower').connect(widget.dsb_wavelet_len_lims_min)
        parameters.get_parameter(name='structure.wavelet.length_limit_upper').connect(widget.dsb_wavelet_len_lims_max)
        parameters.get_parameter(name='structure.wavelet.length_step').connect(widget.dsb_wavelet_len_step)
        parameters.get_parameter(name='structure.wavelet.orientation_limit_lower').connect(
            widget.sb_wavelet_theta_lims_min)
        parameters.get_parameter(name='structure.wavelet.orientation_limit_upper').connect(
            widget.sb_wavelet_theta_lims_max)
        parameters.get_parameter(name='structure.wavelet.orientation_step').connect(widget.sb_wavelet_theta_step)
        parameters.get_parameter(name='structure.wavelet.absolute_threshold').connect(widget.chk_wavelet_abs_thresh)
        parameters.get_parameter(name='structure.wavelet.score_threshold').connect(widget.sb_wavelet_score_threshold)
        parameters.get_parameter(name='structure.myofibril.n_seeds').connect(widget.sb_myofibril_n_seeds)
        parameters.get_parameter(name='structure.myofibril.score_threshold_empty').connect(
            widget.chk_myofibril_score_threshold_empty)
        parameters.get_parameter(name='structure.myofibril.score_threshold').connect(
            widget.sb_myofibril_score_threshold)
        parameters.get_parameter(name='structure.myofibril.persistence').connect(widget.sb_myofibril_persistence)
        parameters.get_parameter(name='structure.myofibril.threshold_distance').connect(
            widget.dsb_myofibril_thresh_dist)

        parameters.get_parameter(name='structure.domain.analysis.score_threshold').connect(
            widget.dsb_domain_analysis_score_threshold)
        parameters.get_parameter(name='structure.domain.analysis.reduce').connect(widget.sb_domain_analysis_reduce)
        parameters.get_parameter(name='structure.domain.analysis.distance_threshold').connect(
            widget.dsb_domain_analysis_distance_threshold)
        parameters.get_parameter(name='structure.domain.analysis.area_min').connect(widget.dsb_domain_analysis_area_min)

    def __predict_finished(self):
        self.__main_control.init_zband_stack()

    def __predict_cell_area_finished(self):
        self.__main_control.init_cell_area_stack()
        pass

    pass
