import numpy as np
import os
from .application_control import ApplicationControl
from ..view.parameter_loi_analysis import Ui_Form as LoiAnalysisWidget


class LOIAnalysisControl:
    """
    The LOIAnalysisControl handles the connection between loi-analysis-backend and the view.
    """

    def __init__(self, loi_analysis_widget: LoiAnalysisWidget, main_control: ApplicationControl):
        self.__loi_analysis_widget = loi_analysis_widget
        self.__main_control = main_control
        self.__thread = None
        self.__worker = None

    def __chk_initialized(self):
        if not self.__main_control.model.is_initialized():
            self.__main_control.debug('file is not correctly initialized (or viewer was closed)')
            return False
        return True

    @staticmethod
    def __call_detect_lois(w, m):
        print('start detect lois')

        m.cell.structure.detect_lois(frame=m.parameters.get_parameter('loi.detect.frame').get_value(),
                                     persistence=m.parameters.get_parameter('loi.detect.persistence').get_value(),
                                     threshold_distance=m.parameters.get_parameter(
                                         'loi.detect.threshold_distance').get_value(),
                                     score_threshold=None if m.parameters.get_parameter(
                                         'loi.detect.score_threshold_automatic').get_value() else m.parameters.get_parameter(
                                         'loi.detect.score_threshold').get_value(),
                                     number_lims=(
                                         m.parameters.get_parameter('loi.detect.number_limits_lower').get_value(),
                                         m.parameters.get_parameter('loi.detect.number_limits_upper').get_value()
                                     ),
                                     msc_lims=(m.parameters.get_parameter('loi.detect.msc_limits_lower').get_value(),
                                               m.parameters.get_parameter('loi.detect.msc_limits_upper').get_value()),
                                     distance_threshold_lois=m.parameters.get_parameter(
                                         'loi.detect.distance_threshold_lois').get_value(),
                                     n_longest=m.parameters.get_parameter('loi.detect.n_longest').get_value(),
                                     linewidth=m.parameters.get_parameter('loi.detect.line_width').get_value())

    def _finished_detect_lois(self):
        # get loi's from cell and add them to napari
        print('finished loi detection...')
        # before adding line to napari, check if the line is already in napari's 'loi' layer
        if self.__main_control.model.cell is None:  # exit method
            return

        line_width = self.__main_control.model.parameters.get_parameter('loi.detect.line_width').get_value()
        # todo: the key loi_lines is not found in cell.structure.data
        for line in self.__main_control.model.cell.structure.data['loi_lines']:
            start, end = np.round(line).astype('int')
            self.__main_control.on_update_loi_list(line_start=start,
                                                   line_end=end,
                                                   line_thickness=line_width)
            print(start, end)
            # todo: add loi's to napari
            pass
        pass

    def on_btn_detect_lois(self):
        if not self.__chk_initialized() or self.__main_control.model.cell is None:
            return
        self.__worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                          call_lambda=self.__call_detect_lois,
                                                          start_message='Starting loi detection',
                                                          finished_message='Finished loi detection',
                                                          finished_action=self._finished_detect_lois,
                                                          finished_successful_action=self.__main_control.
                                                          model.cell.structure.commit)

    @staticmethod
    def __store_lois(w, p):
        # note that first coordinate in the point tuples is Y and second is X
        # pos_vectors = np.array([[[100, 100], [200, 200]],[[300,300],[400,300]]])
        # self.__main_control.layer_loi.add_lines(pos_vectors,edge_width=[10,5],edge_color='red')
        # self.__main_control.layer_loi.add_lines(np.array([[100,200],[100,400]]),edge_color='red',edge_width=15)
        # data=self.__main_control.layer_loi.data
        # widths=self.__main_control.layer_loi.edge_width
        # print(data)
        # print(widths)
        # [array([[100., 100.],[200., 200.]]), array([[300., 300.],[400., 300.]]), array([[100., 200.],[100., 400.]])]
        w.progress.emit(10)
        max_count = len(p['loi_layer'].data)
        step = 90 / max_count
        for index, line in enumerate(p['loi_layer'].data):
            width = int(p['loi_layer'].edge_width[index])
            line2 = ((int(line[0][1]), int(line[0][0])), (int(line[1][1]), int(line[1][0])))
            loi_file = p['cell'].folder + f'{line2[0][0]}_{line2[0][1]}_{line2[1][0]}_{line2[1][1]}_{width}_loi.json'
            if not os.path.exists(loi_file):
                p['main_control'].on_update_loi_list(line_start=(line2[0][0], line2[0][1]),
                                                     line_end=(line2[1][0], line2[1][1]),
                                                     line_thickness=width)
                # extract intensity profiles and save LOI files
                p['cell'].create_loi_data(line2, linewidth=width)
                # todo: add the lines to self.__main_control.model.cell.structure.data['loi_lines']?
                pass
            w.progress.emit(10 + index * step)
            pass
        # [10, 5, 15]
        w.progress.emit(100)
        pass

    def on_btn_store_lois(self):
        # check loi's in napari
        # extract loi coordinates
        # todo: add button cleanup loi's (remove all loi files from loi-lines not existing in napari currently)
        if not self.__chk_initialized():
            return
        parameters = {
            'loi_layer': self.__main_control.layer_loi,
            'cell': self.__main_control.model.cell,
            'main_control': self.__main_control
        }

        self.__worker = self.__main_control.run_async_new(parameters=parameters, call_lambda=self.__store_lois,
                                                          start_message='Start Store LOI\'s',
                                                          finished_message='Finished Store LOI\'s')
        pass

    def bind_events(self):
        """if just detect_lois is called,
         then we need a second button for storing and calculating loi data from manually drawn lois

         """
        self.__loi_analysis_widget.btn_detect_lois.clicked.connect(self.on_btn_detect_lois)
        self.__loi_analysis_widget.btn_store_lois.clicked.connect(self.on_btn_store_lois)

        self.__main_control.model.parameters.get_parameter(name='loi.detect.frame').connect(
            self.__loi_analysis_widget.sb_detect_loi_frame)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.persistence').connect(
            self.__loi_analysis_widget.sb_detect_loi_persistence)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.threshold_distance').connect(
            self.__loi_analysis_widget.dsb_detect_loi_threshold_distance)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.score_threshold').connect(
            self.__loi_analysis_widget.dsb_detect_loi_score_threshold)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.score_threshold_automatic').connect(
            self.__loi_analysis_widget.chk_loi_automatic_score_threshold)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.number_limits_lower').connect(
            self.__loi_analysis_widget.sb_detect_loi_num_lims_min)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.number_limits_upper').connect(
            self.__loi_analysis_widget.sb_detect_loi_num_lims_max)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.msc_limits_lower').connect(
            self.__loi_analysis_widget.sb_detect_loi_msc_limits_min)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.msc_limits_upper').connect(
            self.__loi_analysis_widget.sb_detect_loi_msc_limits_max)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.distance_threshold_lois').connect(
            self.__loi_analysis_widget.sb_detect_loi_distance_threshold)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.n_longest').connect(
            self.__loi_analysis_widget.sb_detect_loi_n_longest)
        self.__main_control.model.parameters.get_parameter(name='loi.detect.line_width').connect(
            self.__loi_analysis_widget.sb_detect_loi_line_width)

        pass
