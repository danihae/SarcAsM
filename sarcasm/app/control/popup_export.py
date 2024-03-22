import traceback

from PyQt5.QtWidgets import QDialog, QGroupBox, QVBoxLayout, QGridLayout, QCheckBox, QPushButton, QHBoxLayout, \
    QLineEdit, QWidget, QFileDialog, QMessageBox

import sarcasm.export as export

import numpy as np


class ExportPopup(QDialog):
    """
    Class that handles displaying export options for CSV export
    For every option in export.py create a checkbox with corresponding label/or checkbox text set.
    Create another checkbox "all" with an event when clicked, setting all other checkboxes to "true".


    """
    __max_entries_per_row = 5

    def __init__(self, model, control):
        super().__init__()
        self.setWindowTitle('Export Popup')
        self.__group_structure = None
        self.__group_metadata = None
        self.__group_motion = None
        self.__group_metadata_old = None
        self.__group_structure_old = None
        self.__btn_export_as_csv = None
        self.__model = model
        self.__control = control
        self.__h_box = None
        self.__le_file_path = None
        self.__le_file_name = None
        self.init_ui()

        pass

    def init_ui(self):
        self.setLayout(QVBoxLayout())
        self.__group_structure = QGroupBox(title='Structure Columns')
        self.__group_metadata = QGroupBox(title='Metadata Columns')
        self.__group_motion = QGroupBox(title='Motion Columns')

        self.__group_structure.setLayout(QGridLayout())
        self.__group_metadata.setLayout(QGridLayout())
        self.__group_motion.setLayout(QGridLayout())

        self.layout().addWidget(self.__group_structure)
        self.layout().addWidget(self.__group_metadata)
        self.layout().addWidget(self.__group_motion)

        self.__h_box = QWidget()
        self.__h_box.setLayout(QHBoxLayout())
        self.__le_file_path = QLineEdit()
        self.__le_file_path.setToolTip('directory path where the files should be exported to')
        self.__le_file_name = QLineEdit()
        self.__le_file_name.setText('export_%.csv')
        self.__le_file_name.setToolTip('File name pattern, the % is a must have, \n'
                                       'it is a placeholder for "structure" or "motion"')

        btn_search = QPushButton(text='...')
        btn_search.clicked.connect(self.__on_clicked_btn_search)

        self.__h_box.layout().addWidget(self.__le_file_path, 5)
        self.__h_box.layout().addWidget(self.__le_file_name, 4)
        self.__h_box.layout().addWidget(btn_search, 1)

        self.layout().addWidget(self.__h_box)

        self.__btn_export_as_csv = QPushButton(text='Export as Csv')
        self.layout().addWidget(self.__btn_export_as_csv)

        self.__btn_export_as_csv.clicked.connect(self.__on_clicked_btn_export_as_csv)

        self.__create_checkbox_from_list(export.meta_keys_default, self.__group_metadata)
        self.__create_checkbox_from_list(export.structure_keys_default, self.__group_structure)
        self.__create_checkbox_from_list(export.roi_keys_default, self.__group_motion)
        pass

    def __on_clicked_btn_search(self):
        folderpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.__le_file_path.setText(folderpath)
        pass

    def __on_clicked_btn_export_as_csv(self):
        if self.__le_file_path.text() is None or self.__le_file_path.text() == '' or self.__le_file_name.text() is None or self.__le_file_name.text() == '':
            self.__control.debug('Please select a directory for exporting the data.')
            return

        if self.__le_file_name.text().find('%') == -1:
            QMessageBox.about(self, "Error", "The File Name has to contain a % as placeholder")
            return

        try:
            to_export_structure = None
            to_export_motion = None

            if len(self.__from_checkboxes_to_str_list(self.__group_structure, self.__group_structure_old)) != 0 or len(
                    self.__from_checkboxes_to_str_list(self.__group_metadata, self.__group_metadata_old)) != 0:
                to_export_structure = export.get_structure_dict(sarc_obj=self.__model.cell,
                                                                structure_keys=self.__from_checkboxes_to_str_list(
                                                                    self.__group_structure, self.__group_structure_old),
                                                                meta_keys=self.__from_checkboxes_to_str_list(
                                                                    self.__group_metadata,
                                                                    self.__group_metadata_old))
                print(to_export_structure)
                self.__control.debug('exported following structure keys')
                self.__control.debug(to_export_structure.__str__())

            if len(self.__from_checkboxes_to_str_list(self.__group_metadata, self.__group_metadata_old)) != 0 or len(
                    self.__from_checkboxes_to_str_list(self.__group_motion)) != 0:
                to_export_motion = export.get_motion_dict(motion_obj=self.__model.sarcomere,
                                                          meta_keys=self.__from_checkboxes_to_str_list(
                                                              self.__group_metadata,
                                                              self.__group_metadata_old),
                                                          roi_keys=self.__from_checkboxes_to_str_list(
                                                              self.__group_motion))
                print(to_export_motion)
                self.__control.debug('exported following motion keys')
                self.__control.debug(to_export_motion.__str__())

            filepath_structure = self.__le_file_path.text() + '/' + self.__le_file_name.text().replace('%', 'structure')
            filepath_motion = self.__le_file_path.text() + '/' + self.__le_file_name.text().replace('%', 'motion')

            if to_export_structure is not None:
                self.__export_to_file(filepath_structure, to_export_structure)
            if to_export_motion is not None:
                self.__export_to_file(filepath_motion, to_export_motion)

        except Exception as e:
            tb = traceback.format_exc()
            self.__control.debug(tb)
            print('Exception occurred on export')
            print(tb)
            pass
        pass

    def __export_to_file(self, file, dictionary):
        with open(file, 'w') as f:
            for key in dictionary:
                value = dictionary[key]
                if isinstance(value, np.ndarray):
                    f.write(key + ',')
                    value.tofile(f, ',')
                    f.write('\n')
                    pass
                else:
                    f.write(key + ',' + str(value) + '\n')
                    pass
                pass
            f.flush()
            f.close()
            pass
        pass

    def show_popup(self):
        self.setGeometry(100, 200, 100, 100)
        self.show()

    def __from_checkboxes_to_str_list(self, container1, container2=None, container3=None) -> list:
        result = self.__from_checkboxes_to_str_list_single(container1)
        if container2 is not None:
            result = result + self.__from_checkboxes_to_str_list_single(container2)
            pass
        if container3 is not None:
            result = result + self.__from_checkboxes_to_str_list_single(container3)
            pass
        return result

    @staticmethod
    def __from_checkboxes_to_str_list_single(container) -> list:
        result = []
        index = container.layout().count()
        while index >= 0:
            obj = container.layout().itemAt(index)
            if obj is not None:
                checkbox = obj.widget()
                if checkbox.text() != 'all' and checkbox.isChecked():
                    result.append(checkbox.text())
                    pass
                pass
            index -= 1
        return result

    @staticmethod
    def __on_all_checkbox(container, value):
        index = container.layout().count()
        while index >= 0:
            obj = container.layout().itemAt(index)
            if obj is not None:
                checkbox = obj.widget()
                if checkbox.text() != 'all':
                    checkbox.setChecked(value)
                    pass
                pass
            index -= 1
        pass

    def __create_checkbox_from_list(self, str_list, container):
        row = 0
        col = 0
        all_checkbox = QCheckBox('all')
        # todo: add event to all checkbox: which checks all other checkboxes in that group
        all_checkbox.clicked.connect(lambda value: self.__on_all_checkbox(container, value))

        container.layout().addWidget(all_checkbox, col, row)
        col = col + 1

        for str_name in str_list:
            checkbox = QCheckBox(str_name)
            container.layout().addWidget(checkbox, col, row)
            col = col + 1
            if col >= self.__max_entries_per_row:
                col = 0
                row = row + 1
                pass

            # todo: create checkbox in container (only a fixed amount of checkboxes per line)
            pass
        pass
