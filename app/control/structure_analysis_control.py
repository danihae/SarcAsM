from typing import Any

import qtutils
from PyQt5.QtWidgets import QFileDialog
from biu.progress import ProgressNotifier

from sarcasm import SarcAsM
from .chain_execution import ChainExecution
from .application_control import ApplicationControl
from ..view.parameter_structure_analysis import Ui_Form as StructureAnalysisWidget
from ..model import ApplicationModel
from sarcasm.type_utils import TypeUtils


class StructureAnalysisControl:
    """
    Handles button calls, parameter changes etc. from structure view
    """

    def __init__(self, structure_parameters_widget: StructureAnalysisWidget, main_control: ApplicationControl):
        self.__structure_parameters_widget = structure_parameters_widget
        self.__main_control = main_control
        self.__thread = None
        self.__worker = None

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

    def __predict_call(self, worker, model: ApplicationModel):

        progress_notifier = self.__get_progress_notifier(worker)

        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None
        cell: SarcAsM = TypeUtils.unbox(model.cell)
        cell.structure.predict_z_bands(progress_notifier=progress_notifier,
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

    def __cell_mask_predict_call(self, worker, model: ApplicationModel):
        progress_notifier = self.__get_progress_notifier(worker)

        network_model = model.parameters.get_parameter('structure.predict.network_path').get_value()
        if network_model == 'generalist':
            network_model = None

        cell: SarcAsM = TypeUtils.unbox(model.cell)
        cell.structure.predict_cell_mask(progress_notifier=progress_notifier,
                                         model_path=network_model,
                                         size=(
                                             model.parameters.get_parameter(
                                                 'structure.predict.cell_mask.size_width').get_value(),
                                             model.parameters.get_parameter(
                                                 'structure.predict.cell_mask.size_height').get_value()
                                         ),
                                         clip_thres=(
                                             model.parameters.get_parameter(
                                                 'structure.predict.cell_mask.clip_thresh_min').get_value(),
                                             model.parameters.get_parameter(
                                                 'structure.predict.cell_mask.clip_thresh_max').get_value()
                                         ))
        cell.structure.analyze_cell_mask()
        pass

    def __chk_prediction_network(self):  # todo rename to zband_prediction or similar
        if self.__main_control.model.parameters.get_parameter('structure.predict.network_path').get_value() == '':
            self.__main_control.debug('no network file was chosen for prediction')
            return False
        return True

    def __chk_cell_mask_prediction_network(self):
        if self.__main_control.model.parameters.get_parameter(
                'structure.predict.cell_mask.network_path').get_value() == '':
            self.__main_control.debug('no network file was chosen for cell mask prediction')
            return False
        return True

    def __chk_frames(self):
        frames = self.__main_control.model.parameters.get_parameter('structure.frames').get_value()
        if frames is None or frames == '':
            self.__check_frame_syntax()
            self.__main_control.debug(
                'no frames selected, please select the frame(s) in the specified format')
            return False
        return True

    def __chk_initialized(self):
        if not self.__main_control.model.is_initialized():
            self.__main_control.debug('file is not correctly initialized (or viewer was closed)')
            return False
        return True

    def on_btn_z_bands_predict(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_prediction_network():
            return
        cell: SarcAsM = TypeUtils.unbox(self.__main_control.model.cell)
        message_finished = f'Z-bands detected and saved in {cell.folder}'
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__predict_call,
                                                   start_message='Start prediction of sarcomere z-bands',
                                                   finished_message=message_finished,
                                                   finished_action=self.__predict_z_bands_finished,
                                                   finished_successful_action=cell.structure.commit)
        self.__worker = worker
        return worker

    def on_btn_cell_mask_predict(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_cell_mask_prediction_network():
            return
        cell: SarcAsM = TypeUtils.unbox(self.__main_control.model.cell)
        worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                   call_lambda=self.__cell_mask_predict_call,
                                                   start_message='Start prediction of cell mask',
                                                   finished_message='Finished prediction of cell mask',
                                                   finished_action=self.__predict_cell_mask_finished,
                                                   finished_successful_action=cell.structure.commit)
        self.__worker = worker
        return worker

    def __parse_frames(self, frames_str: str):
        if frames_str == '':
            return None
        if frames_str.lower().__eq__('all'):
            return frames_str.lower()
        if frames_str.isnumeric():
            return int(frames_str)
        if frames_str.__contains__(','):
            list_str = frames_str.split(',')
            parsed_list = []
            for x in list_str:
                if x.isnumeric():
                    parsed_list.append(int(x))
            return parsed_list
        return 0  # if it's a wrong value just process first image (processing all could take long for a wrong input)

    def on_btn_z_band(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        def __internal_call(w, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: SarcAsM = TypeUtils.unbox(m.cell)
            cell.structure.analyze_z_bands(frames=m.parameters.get_parameter('structure.frames').get_value(),
                                           threshold=m.parameters.get_parameter(
                                               'structure.z_band_analysis.threshold').get_value(),
                                           min_length=m.parameters.get_parameter(
                                               'structure.z_band_analysis.min_length').get_value(),
                                           progress_notifier=progress_notifier)
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   start_message='Start Z-band Analysis',
                                                   finished_message='Finished Z-band Analysis',
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.structure.commit()))
        self.__worker = worker
        return worker

    def on_btn_wavelet(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        def __internal_call(w: Any, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: SarcAsM = TypeUtils.unbox(m.cell)
            cell.structure.analyze_sarcomere_vectors(
                frames=m.parameters.get_parameter('structure.frames').get_value(),
                size=m.parameters.get_parameter('structure.wavelet.filter_size').get_value(),
                minor=m.parameters.get_parameter('structure.wavelet.minor').get_value(),
                major=m.parameters.get_parameter('structure.wavelet.major').get_value(),
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
                save_all=m.parameters.get_parameter('structure.wavelet.save_all').get_value(),
                progress_notifier=progress_notifier
            )
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   finished_message='Finished wavelet analysis',
                                                   start_message='Start wavelet analysis',
                                                   finished_action=self.__sarcomere_analysis_finished,
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.structure.commit()))
        self.__worker = worker
        return worker
        # AND-gated double wavelet analysis of sarcomere structure to locally obtain length and angle

        pass

    def on_btn_myofibril(self):
        """
        analyze_myofibrils(self, frames=None, n_seeds=200, score_threshold=None, persistence=3,
                           threshold_distance=0.3,
                           save_all=False, plot=False)
        """
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        # estimate myofibril lengths using line-growth algorithm
        def __internal_call(w: Any, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: SarcAsM = TypeUtils.unbox(m.cell)
            cell.structure.analyze_myofibrils(
                frames=m.parameters.get_parameter('structure.frames').get_value(),
                n_seeds=m.parameters.get_parameter('structure.myofibril.n_seeds').get_value(),
                persistence=m.parameters.get_parameter('structure.myofibril.persistence').get_value(),
                threshold_distance=m.parameters.get_parameter('structure.myofibril.threshold_distance').get_value(),
                progress_notifier=progress_notifier
            )
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   start_message='Start myofibril analysis',
                                                   finished_message='Finished myofibril analysis',
                                                   finished_action=self.__myofibril_analysis_finished,
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.structure.commit()))
        self.__worker = worker
        return worker
        pass

    def on_btn_domain_analysis(self):
        """
        call domain analysis in backend
        """
        if not self.__chk_initialized():
            return
        if not self.__chk_frames():
            return

        def __internal_call(w: Any, m: ApplicationModel):
            progress_notifier = self.__get_progress_notifier(w)
            cell: SarcAsM = TypeUtils.unbox(m.cell)
            cell.structure.analyze_sarcomere_domains(
                frames=m.parameters.get_parameter('structure.frames').get_value(),
                dist_threshold_ends=m.parameters.get_parameter(
                    'structure.domain.analysis.dist_thresh_ends').get_value(),
                dist_threshold_pos_vectors=m.parameters.get_parameter(
                    'structure.domain.analysis.dist_thresh_pos_vectors').get_value(),
                louvain_resolution=m.parameters.get_parameter(
                    'structure.domain.analysis.louvain_resolution').get_value(),
                louvain_seed=m.parameters.get_parameter('structure.domain.analysis.louvain_seed').get_value(),
                area_min=m.parameters.get_parameter('structure.domain.analysis.area_min').get_value(),
                dilation_radius=m.parameters.get_parameter('structure.domain.analysis.dilation_radius').get_value(),
                progress_notifier=progress_notifier
            )
            pass

        worker = self.__main_control.run_async_new(parameters=self.__main_control.model, call_lambda=__internal_call,
                                                   start_message='Start sarcomere domains analysis',
                                                   finished_message='Finished sarcomere domains analysis',
                                                   finished_action=self.__domain_analysis_finished,
                                                   finished_successful_action=TypeUtils.if_present(
                                                       self.__main_control.model.cell, lambda c: c.structure.commit()))
        self.__worker = worker
        return worker
        pass

    def on_btn_search_network(self):
        # f_name is a tuple
        f_name = QFileDialog.getOpenFileName(caption='Open Network File', filter="Network Files (*.pth)")
        if f_name is not None:
            self.__structure_parameters_widget.le_network.setText(f_name[0])

    def on_btn_cell_mask_search_network(self):
        f_name = QFileDialog.getOpenFileName(caption='Open Network File', filter="Network Files (*.pth)")
        if f_name is not None:
            self.__structure_parameters_widget.le_cell_mask_network.setText(f_name[0])

    def __filter_input_prediction_size(self, element):
        if not (hasattr(element, 'value') and hasattr(element, 'setValue')):
            return

        value = element.value()
        factor = value // 16
        if factor * 16 != value:
            element.setValue(factor * 16)
            pass
        pass

    def __check_frame_syntax(self):
        text = self.__structure_parameters_widget.le_general_frames.text()
        value = self.__parse_frames(text)
        if not text.isnumeric() and (value == 0 or value is None):
            # this is an error
            self.__structure_parameters_widget.le_general_frames.setStyleSheet("QLineEdit{background : red;}")
            pass
        else:
            self.__structure_parameters_widget.le_general_frames.setStyleSheet(
                "QLineEdit{background : lightgreen;}")
        pass

    def on_analyze_structure(self):
        if not self.__chk_initialized():
            return
        if not self.__chk_prediction_network():
            return
        if not self.__chk_frames():
            return
        if not self.__chk_cell_mask_prediction_network():
            return
        # predict, z band analysis, wavelet analysis, myofibril length
        chain = ChainExecution(self.__main_control.model.currentlyProcessing, self.__main_control.debug)
        chain.add_step(self.on_btn_z_bands_predict)
        # chain.add_step(self.on_btn_cell_mask_predict) # excluded from analyze structure
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

        self.__structure_parameters_widget.sb_cell_mask_size_width.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_cell_mask_size_width))
        self.__structure_parameters_widget.sb_cell_mask_size_height.editingFinished.connect(
            lambda: self.__filter_input_prediction_size(self.__structure_parameters_widget.sb_cell_mask_size_height))

        self.__structure_parameters_widget.le_general_frames.editingFinished.connect(self.__check_frame_syntax)

        self.__structure_parameters_widget.btn_structure_cell_mask_predict.clicked.connect(
            self.on_btn_cell_mask_predict)
        self.__structure_parameters_widget.btn_structure_predict.clicked.connect(self.on_btn_z_bands_predict)
        self.__structure_parameters_widget.btn_structure_z_band.clicked.connect(self.on_btn_z_band)
        self.__structure_parameters_widget.btn_structure_wavelet.clicked.connect(self.on_btn_wavelet)
        self.__structure_parameters_widget.btn_structure_myofibril.clicked.connect(self.on_btn_myofibril)
        self.__structure_parameters_widget.btn_search_network.clicked.connect(self.on_btn_search_network)
        self.__structure_parameters_widget.btn_cell_mask_network_search.clicked.connect(
            self.on_btn_cell_mask_search_network)
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

        parameters.get_parameter(name='structure.predict.cell_mask.network_path').connect(widget.le_cell_mask_network)
        parameters.get_parameter(name='structure.predict.cell_mask.size_width').connect(widget.sb_cell_mask_size_width)
        parameters.get_parameter(name='structure.predict.cell_mask.size_height').connect(
            widget.sb_cell_mask_size_height)
        parameters.get_parameter(name='structure.predict.cell_mask.clip_thresh_min').connect(
            widget.dsb_cell_mask_clip_threshold_min)
        parameters.get_parameter(name='structure.predict.cell_mask.clip_thresh_max').connect(
            widget.dsb_cell_mask_clip_threshold_max)

        parameters.get_parameter(name='structure.frames').set_value_parser(self.__parse_frames)
        parameters.get_parameter(name='structure.frames').connect(widget.le_general_frames)

        parameters.get_parameter(name='structure.z_band_analysis.threshold').connect(widget.dsb_z_band_threshold)
        parameters.get_parameter(name='structure.z_band_analysis.min_length').connect(widget.dsb_z_band_min_length)
        parameters.get_parameter(name='structure.wavelet.filter_size').connect(widget.dsb_wavelet_filter_size)
        parameters.get_parameter(name='structure.wavelet.minor').connect(widget.dsb_wavelet_minor)
        parameters.get_parameter(name='structure.wavelet.major').connect(widget.dsb_wavelet_width)
        parameters.get_parameter(name='structure.wavelet.length_limit_lower').connect(widget.dsb_wavelet_len_lims_min)
        parameters.get_parameter(name='structure.wavelet.length_limit_upper').connect(widget.dsb_wavelet_len_lims_max)
        parameters.get_parameter(name='structure.wavelet.length_step').connect(widget.dsb_wavelet_len_step)
        parameters.get_parameter(name='structure.wavelet.orientation_limit_lower').connect(
            widget.sb_wavelet_theta_lims_min)
        parameters.get_parameter(name='structure.wavelet.orientation_limit_upper').connect(
            widget.sb_wavelet_theta_lims_max)
        parameters.get_parameter(name='structure.wavelet.orientation_step').connect(widget.sb_wavelet_theta_step)
        parameters.get_parameter(name='structure.wavelet.absolute_threshold').connect(widget.chk_wavelet_abs_thresh)
        parameters.get_parameter(name='structure.wavelet.score_threshold').connect(widget.dsb_wavelet_score_threshold)
        parameters.get_parameter(name='structure.myofibril.n_seeds').connect(widget.sb_myofibril_n_seeds)
        parameters.get_parameter(name='structure.myofibril.score_threshold_empty').connect(
            widget.chk_myofibril_score_threshold_empty)
        parameters.get_parameter(name='structure.myofibril.score_threshold').connect(
            widget.sb_myofibril_score_threshold)
        parameters.get_parameter(name='structure.myofibril.persistence').connect(widget.sb_myofibril_persistence)
        parameters.get_parameter(name='structure.myofibril.threshold_distance').connect(
            widget.dsb_myofibril_thresh_dist)

        parameters.get_parameter(name='structure.domain.analysis.dist_thresh_ends').connect(widget.dsb_dist_thresh_ends)
        parameters.get_parameter(name='structure.domain.analysis.dist_thresh_pos_vectors').connect(
            widget.dsb_dist_thresh_pos_vectors)
        parameters.get_parameter(name='structure.domain.analysis.louvain_resolution').connect(
            widget.dsb_louvain_resolution)
        parameters.get_parameter(name='structure.domain.analysis.louvain_seed').connect(widget.sb_louvain_seed)
        parameters.get_parameter(name='structure.domain.analysis.area_min').connect(widget.dsb_domain_analysis_area_min)
        parameters.get_parameter(name='structure.domain.analysis.dilation_radius').connect(widget.sb_dilation_radius)

        pass

    def __predict_z_bands_finished(self):
        self.__main_control.init_z_band_stack()

    def __predict_cell_mask_finished(self):
        self.__main_control.init_cell_mask_stack()

    def __z_band_analysis_finished(self):
        pass

    def __sarcomere_analysis_finished(self):
        self.__main_control.init_sarcomere_mask_stack()
        self.__main_control.init_sarcomere_vector_stack()

    def __myofibril_analysis_finished(self):
        self.__main_control.init_myofibril_lines_stack()

    def __domain_analysis_finished(self):
        self.__main_control.init_sarcomere_domain_stack()
