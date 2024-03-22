import numpy as np
import os
from .application_control import ApplicationControl
from ..view.parameter_roi_analysis import Ui_Form as RoiAnalysisWidget


class RoiAnalysisControl:
    """
    The RoiAnalysisControl handles the connection between roi-analysis-backend and the view.
    """
    def __init__(self, roi_analysis_widget: RoiAnalysisWidget, main_control: ApplicationControl):
        self.__roi_analysis_widget = roi_analysis_widget
        self.__main_control = main_control
        self.__thread = None
        self.__worker = None

    def __chk_initialized(self):
        if not self.__main_control.model.is_initialized():
            self.__main_control.debug('file is not correctly initialized (or viewer was closed)')
            return False
        return True

    @staticmethod
    def __call_detect_rois(w, m):
        print('start detect rois')

        m.cell.detect_rois(timepoint=m.parameters.get_parameter('roi.detect.timepoint').get_value(),
                           persistence=m.parameters.get_parameter('roi.detect.persistence').get_value(),
                           threshold_distance=m.parameters.get_parameter('roi.detect.threshold_distance').get_value(),
                           score_threshold=None if m.parameters.get_parameter(
                               'roi.detect.score_threshold_automatic').get_value() else m.parameters.get_parameter(
                               'roi.detect.score_threshold').get_value(),
                           number_lims=(
                               m.parameters.get_parameter('roi.detect.number_limits_lower').get_value(),
                               m.parameters.get_parameter('roi.detect.number_limits_upper').get_value()
                           ),
                           msc_lims=(m.parameters.get_parameter('roi.detect.msc_limits_lower').get_value(),
                                     m.parameters.get_parameter('roi.detect.msc_limits_upper').get_value()),
                           distance_threshold_rois=m.parameters.get_parameter(
                               'roi.detect.distance_threshold_rois').get_value(),
                           n_longest=m.parameters.get_parameter('roi.detect.n_longest').get_value(),
                           linewidth=m.parameters.get_parameter('roi.detect.line_width').get_value())

    def _finished_detect_rois(self):
        # todo: get roi's from cell and add them to napari
        print('finished roi detection...')
        # before adding line to napari, check if the line is already in napari's 'roi' layer

        line_width = self.__main_control.model.parameters.get_parameter('roi.detect.line_width').get_value()

        #  WARNING: Traceback (most recent call last):
        #  File "D:\Development\PycharmProjects\sarcomere_analysis\dist\distribution\sarcasm_old\app\control\roi_analysis_control.py", line 52, in _finished_detect_rois
        #  for line in self.__main_control.model.cell.structure['roi_lines']:
        #        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
        #  KeyError: 'roi_lines'

        for line in self.__main_control.model.cell.structure['roi_lines']:
            start, end = np.round(line).astype('int')
            self.__main_control.on_update_roi_list(line_start=start,
                                                   line_end=end,
                                                   line_thickness=line_width)
            print(start, end)
            # todo: add roi's to napari
            pass
        pass

    def on_btn_detect_rois(self):
        if not self.__chk_initialized():
            return
        self.__worker = self.__main_control.run_async_new(parameters=self.__main_control.model,
                                                          call_lambda=self.__call_detect_rois,
                                                          start_message='Starting Roi detection',
                                                          finished_message='Finished Roi detection',
                                                          finished_action=self._finished_detect_rois,
                                                          finished_successful_action=self.__main_control.model.cell.commit)

    @staticmethod
    def __store_rois(w, p):
        # note that first coordinate in the point tuples is Y and second is X
        # points = np.array([[[100, 100], [200, 200]],[[300,300],[400,300]]])
        # self.__main_control.layer_roi.add_lines(points,edge_width=[10,5],edge_color='red')
        # self.__main_control.layer_roi.add_lines(np.array([[100,200],[100,400]]),edge_color='red',edge_width=15)
        # data=self.__main_control.layer_roi.data
        # widths=self.__main_control.layer_roi.edge_width
        # print(data)
        # print(widths)
        # [array([[100., 100.],[200., 200.]]), array([[300., 300.],[400., 300.]]), array([[100., 200.],[100., 400.]])]
        w.progress.emit(10)
        max_count = len(p['roi_layer'].data)
        step = 90 / max_count
        for index, line in enumerate(p['roi_layer'].data):
            width = int(p['roi_layer'].edge_width[index])
            line2 = ((int(line[0][1]), int(line[0][0])), (int(line[1][1]), int(line[1][0])))
            roi_file = p['cell'].folder + f'{line2[0][0]}_{line2[0][1]}_{line2[1][0]}_{line2[1][1]}_{width}_roi.json'
            if not os.path.exists(roi_file):
                p['main_control'].on_update_roi_list(line_start=(line2[0][0], line2[0][1]),
                                                     line_end=(line2[1][0], line2[1][1]),
                                                     line_thickness=width)
                # extract intensity profiles and save ROI files
                p['cell'].create_roi_data(line2, linewidth=width)
                # todo: add the lines to self.__main_control.model.cell.structure['roi_lines']?
                pass
            w.progress.emit(10 + index * step)
            pass
        # [10, 5, 15]
        w.progress.emit(100)
        pass

    def on_btn_store_rois(self):
        # check roi's in napari
        # extract roi coordinates
        #todo: add button cleanup roi's (remove all roi files from roi-lines not existing in napari currently)
        if not self.__chk_initialized():
            return
        parameters = {
            'roi_layer': self.__main_control.layer_roi,
            'cell': self.__main_control.model.cell,
            'main_control': self.__main_control
        }

        self.__worker = self.__main_control.run_async_new(parameters=parameters, call_lambda=self.__store_rois,
                                                          start_message='Start Store Roi\'s',
                                                          finished_message='Finished Store Roi\'s')
        pass

    def bind_events(self):
        """if just detect_rois is called,
         then we need a second button for storing and calculating roi data from manually drawn rois

         """
        self.__roi_analysis_widget.btn_detect_rois.clicked.connect(self.on_btn_detect_rois)
        self.__roi_analysis_widget.btn_store_rois.clicked.connect(self.on_btn_store_rois)

        self.__main_control.model.parameters.get_parameter(name='roi.detect.timepoint').connect(
            self.__roi_analysis_widget.sb_detect_roi_timepoint)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.persistence').connect(
            self.__roi_analysis_widget.sb_detect_roi_persistence)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.threshold_distance').connect(
            self.__roi_analysis_widget.dsb_detect_roi_threshold_distance)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.score_threshold').connect(
            self.__roi_analysis_widget.dsb_detect_roi_score_threshold)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.score_threshold_automatic').connect(
            self.__roi_analysis_widget.chk_roi_automatic_score_threshold)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.number_limits_lower').connect(
            self.__roi_analysis_widget.sb_detect_roi_num_lims_min)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.number_limits_upper').connect(
            self.__roi_analysis_widget.sb_detect_roi_num_lims_max)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.msc_limits_lower').connect(
            self.__roi_analysis_widget.sb_detect_roi_msc_limits_min)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.msc_limits_upper').connect(
            self.__roi_analysis_widget.sb_detect_roi_msc_limits_max)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.distance_threshold_rois').connect(
            self.__roi_analysis_widget.sb_detect_roi_distance_threshold)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.n_longest').connect(
            self.__roi_analysis_widget.sb_detect_roi_n_longest)
        self.__main_control.model.parameters.get_parameter(name='roi.detect.line_width').connect(
            self.__roi_analysis_widget.sb_detect_roi_line_width)

        pass
