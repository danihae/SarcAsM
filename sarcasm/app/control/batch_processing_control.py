from PyQt5.QtWidgets import QFileDialog

from ..view.parameters_batch_processing import Ui_Form as BatchProcessingWidget
from .application_control import ApplicationControl
from ... import SarcAsM


class BatchProcessingControl:

    def __init__(self, batch_processing_widget: BatchProcessingWidget, main_control: ApplicationControl):
        self.__batch_processing_widget = batch_processing_widget
        self.__main_control = main_control
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

        pass

    # todo: implement batch processing functionality
    def on_btn_batch_processing_structure(self):
        pass

    def on_btn_batch_processing_motion(self):
        pass

    def __single_structure_analysis(self, file: str, frame_time: float, pixel_size: float, force_override: bool):
        # attention: this method is not executed in qt thread! --> every information to ui needs to be done either
        # on another place or within a wrapper for QT Main thread (like the package inmain does)

        print(file)
        # initialize SarcAsM object
        # todo check for metadata

        sarc_obj = SarcAsM(file)
        # predict sarcomere z-bands and cell area
        sarc_obj.predict_z_bands(size=(2048, 2048))
        sarc_obj.predict_cell_area(size=(2048, 2048))
        # analyze cell area and sarcomere area
        sarc_obj.analyze_cell_area(timepoints='all')
        sarc_obj.analyze_sarcomere_domains(timepoints='all')
        # analyze sarcomere structures
        sarc_obj.full_analysis_structure(timepoints='all')
        print(f'{file} successfully analyzed!')

    def on_search(self):
        # f_name is a tuple
        file = str(QFileDialog.getExistingDirectory(caption="Select Root Directory"))
        if file is not None and file is not '':
            self.__batch_processing_widget.le_root_directory.setText(file)
        pass

    pass
