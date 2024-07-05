import os
import traceback
from typing import Tuple, Optional

from sarcasm import Utils
from sarcasm.type_utils import TypeUtils

import napari
import numpy as np
import tifffile
import torch
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QProgressBar, QTextEdit
from bio_image_unet.progress import ProgressNotifier
from napari.layers import Shapes

from ..model import ApplicationModel


class ApplicationControl:
    """
    Main application control.
    It contains some public utility methods and handles parts of the general application flow.
    """

    def __init__(self, window: QWidget, model):
        """
        window: QWidget
        model: ApplicationModel (has to be of that type), due to removing possible circular dependencies
        -> removed the import statement and type specifier

        """
        self._window = window
        self._model: ApplicationModel = model
        self._viewer = None  # napari.Viewer(title='Image Window(napari)')  # the napari viewer object
        self.__layer_loi: Optional[Shapes] = None
        self.__debug_action = None
        self.__worker_thread: Optional[QThread] = None
        self.__callback_loi_list_updated = None

        self.progress_notifier = ProgressNotifier()
        self.progress_notifier.set_progress_report(lambda p: self.update_progress(p * 100))
        self.debug("dummy line")  # to prevent the actual first line to get replaced by testing progress detail
        self.progress_notifier.set_progress_detail(
            lambda hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta: self.debug_replace(
                "%02d:%02d:%02d / %02d:%02d:%02d" % (hh_current, mm_current, ss_current, hh_eta, mm_eta, ss_eta)))

    def debug(self, message):
        # for now, just print
        if self.__debug_action is not None:
            self.__debug_action(message)

    def debug_replace(self, message):
        te = self._window.findChild(QTextEdit, name="messageArea")
        if te is not None:
            text = te.toPlainText()
            te.setText(text[:text.rfind('\n')])
            te.append(message)
            # te.verticalScrollBar().setValue(te.verticalScrollBar().maximum())  # scroll messageArea to last line!
            TypeUtils.if_present(te.verticalScrollBar(), lambda sc: sc.setValue(sc.maximum()))
            # remove last line
            # append message as last line

    def set_debug_action(self, debug_action):
        self.__debug_action = debug_action

    def set_callback_loi_list_updated(self, callback):
        self.__callback_loi_list_updated = callback

    @property
    def layer_loi(self):
        return self.__layer_loi

    def init_loi_layer(self, layer):
        self.__layer_loi = layer

    @property
    def model(self) -> ApplicationModel:
        return self._model

    @property
    def viewer(self) -> napari.Viewer:
        return self._viewer

    @staticmethod
    def is_gpu_available():
        gpu_flag = False
        if torch.cuda.is_available():
            gpu_flag = True
        elif hasattr(torch, 'mps'):  # only for apple m1/m2/...
            if torch.backends.mps.is_available():
                gpu_flag = True
                pass
            pass
        return gpu_flag
        pass

    def clean_up_on_new_image(self):
        """Reset model to default state (when loading new image, data of old image should be removed)"""
        if self._viewer is not None:
            if napari.current_viewer() is not None:  # check if viewer was closed
                self._viewer.close()
            self._viewer = napari.Viewer(title='SarcAsM')  # the napari viewer object
        else:
            self._viewer = napari.Viewer(title='SarcAsM')  # the napari viewer object
        self.model.reset_model()
        pass

    # def init_viewer(self, viewer):
    #    self._viewer = viewer

    def update_progress(self, value):
        progress_bar = self._window.findChild(QProgressBar, name='progressBarMain')
        if progress_bar is not None:
            progress_bar.setValue(value)

    def __add_line_to_napari(self, line_to_draw):
        # note that first coordinate in the point tuples is Y and second is X
        # np.array([[[100, 100], [200, 200]], [[300, 300], [400, 300]]])

        pos_vectors = np.array([[line_to_draw[0][1], line_to_draw[0][0]], [line_to_draw[1][1], line_to_draw[1][0]]])
        self.layer_loi.add_lines(pos_vectors, edge_width=line_to_draw[2], edge_color='red')
        # self.__main_control.layer_loi.add_lines(np.array([[100,200],[100,400]]),edge_color='red',edge_width=15)
        # data=self.__main_control.layer_loi.data
        # widths=self.__main_control.layer_loi.edge_width
        # print(data)
        # print(widths)
        # [array([[100., 100.],[200., 200.]]), array([[300., 300.],[400., 300.]]), array([[100., 200.],[100., 400.]])]
        # [10, 5, 15]

    def on_update_loi_list(self, line_start, line_end=None, line_thickness=None, drawn_by_user=True):
        # add handling for bad input

        if self.model.cell is None or line_start is None or line_end is None or len(line_start) != 2 or len(
                line_end) != 2 or line_thickness is None:
            print('info: line updated but wrong data-type')
            return
        line = (line_start, line_end, line_thickness)
        list_entry = self.get_entry_key_for_line(line)
        if list_entry in self.model.line_dictionary[self.model.cell.filename]:  # if element already contained, ignore
            # if its inside and its currently selected, reload the sarcomere (for up to date loi info)
            if 'last' in self.model.line_dictionary[self.model.cell.filename] and line == \
                    self.model.line_dictionary[self.model.cell.filename]['last']:
                file_name, scan_line = self.get_file_name_from_scheme(self.model.cell.filename, 'last')
                self.model.init_sarcomere(file_name)
            return

        # add line and line_ux to dictionary for later usage
        self.model.line_dictionary[self.model.cell.filename][list_entry] = line
        # add line to napari
        self.__add_line_to_napari(line)

        # todo: update combo box on motion analysis parameters page
        # todo: should be done via callback method
        if self.__callback_loi_list_updated is not None:
            self.__callback_loi_list_updated(self.model.line_dictionary[self.model.cell.filename])

        # select the element if it was drawn by user
        if drawn_by_user:
            # set selection to last item
            # entries = len(self.model.line_dictionary[self._cell.filename]) - 1  # remove the last entry from count
            # self.gui.listBoxLoi.select_clear(0, "end")  # clear selection
            # self.gui.listBoxLoi.selection_set(first=entries - 1, last=None)  # counting starts at 0
            # trigger event manually
            # self.on_selection_changed_loi_list(None)
            pass

    @staticmethod
    def get_entry_key_for_line(line) -> str:
        return '(%d,%d)->(%d,%d):%d' % (line[0][0],
                                        line[0][1],
                                        line[1][0],
                                        line[1][1],
                                        line[2])

    def get_file_name_from_scheme(self, cell_file, line) -> Tuple[str, object]:
        scheme = self.model.scheme
        scan_line = self.model.line_dictionary[cell_file][line]
        file_name = scheme % (scan_line[0][0],
                              scan_line[0][1],
                              scan_line[1][0],
                              scan_line[1][1],
                              scan_line[2])
        file_name += "_loi" + self.model.file_extension
        return file_name, scan_line

    def init_z_band_stack(self):
        if self.model.cell is not None and os.path.exists(self.model.cell.file_sarcomeres):
            if self.viewer.layers.__contains__('ZbandMask'):
                layer = self.viewer.layers.__getitem__('ZbandMask')
                self.viewer.layers.remove(layer)
            # load sarcomere Z-band file into unet stack
            tmp = tifffile.imread(self.model.cell.file_sarcomeres).astype('uint8')
            self.viewer.add_image(tmp, name='ZbandMask', opacity=0.4, colormap='viridis')

    def init_cell_mask_stack(self):
        if self.model.cell is not None and os.path.exists(self.model.cell.file_cell_mask):
            if self.viewer.layers.__contains__('CellMask'):
                layer = self.viewer.layers.__getitem__('CellMask')
                self.viewer.layers.remove(layer)
            # load cell mask file into unet stack
            tmp = tifffile.imread(self.model.cell.file_cell_mask).astype('uint8')
            self.viewer.add_image(tmp, name='CellMask', opacity=0.1, )

    def init_z_lateral_connections(self):
        if self.model.cell is not None and 'z_labels' in self.model.cell.structure.data.keys():
            if self.viewer.layers.__contains__('ZbandLatGroups'):
                layer = self.viewer.layers.__getitem__('ZbandLatGroups')
                self.viewer.layers.remove(layer)
                pass
            if self.viewer.layers.__contains__('ZbandLatConnections'):
                layer = self.viewer.layers.__getitem__('ZbandLatConnections')
                self.viewer.layers.remove(layer)
                pass
            if self.viewer.layers.__contains__('ZbandEnds'):
                layer = self.viewer.layers.__getitem__('ZbandEnds')
                self.viewer.layers.remove(layer)
                pass
            # create labels and connections for all frames and add as label and line layers
            labels_groups = np.zeros((self.model.cell.metadata['frames'], *self.model.cell.metadata['size']), dtype='uint16')
            ends = []
            connections = []
            for frame in range(self.model.cell.metadata['frames']):
                if 'params.z_frames' in self.model.cell.structure.data and frame in \
                        self.model.cell.structure.data['params.z_frames']:
                    labels_frame = self.model.cell.structure.data['z_labels'][frame].toarray()
                    groups_frame = self.model.cell.structure.data['z_lat_groups'][frame]
                    labels_groups_frame = np.zeros_like(labels_frame)
                    for i, group in enumerate(groups_frame[1:]):
                        mask = np.zeros_like(labels_frame, dtype=bool)
                        for label in group:
                            mask += (labels_frame == label + 1)
                        labels_groups_frame[mask] = i + 1
                    labels_groups_frame = Utils.shuffle_labels(labels_groups_frame)
                    labels_groups[frame] = labels_groups_frame

                    z_ends_frame = self.model.cell.structure.data['z_ends'][frame] / self.model.cell.metadata['pixelsize']
                    z_links_frame = self.model.cell.structure.data['z_lat_links'][frame]

                    # ends
                    for z_ends_i in z_ends_frame:
                        ends.append([frame, z_ends_i[0, 0], z_ends_i[0, 1]])
                        ends.append([frame, z_ends_i[1, 0], z_ends_i[1, 1]])

                    # connections
                    for (i, k, j, l) in z_links_frame.T:
                        connections.append([[frame, z_ends_frame[i, k, 0], z_ends_frame[i, k, 1]],
                                            [frame, z_ends_frame[j, l, 0], z_ends_frame[j, l, 1]]])

            labels_groups = np.asarray(labels_groups)
            self.viewer.add_labels(labels_groups, name='ZbandLatGroups', opacity=0.5)
            self.viewer.add_shapes(connections, name='ZbandLatConnections', shape_type='path', edge_color='white',
                              edge_width=1, opacity=0.15)
            self.viewer.add_points(name='ZbandEnds', data=ends, face_color='w', size=3)

    def init_myofibril_lines_stack(self):
        if self.model.cell is not None and 'myof_lines' in self.model.cell.structure.data.keys():
            if self.viewer.layers.__contains__('MyofibrilLines'):
                layer = self.viewer.layers.__getitem__('MyofibrilLines')
                self.viewer.layers.remove(layer)
            # load myofibril lines and as multi-segment paths
            myof_lines = self.model.cell.structure.data['myof_lines']
            pos_vectors = self.model.cell.structure.data['pos_vectors']
            myof_lines_pos_vectors = [
                [np.column_stack((np.full((len(line_j), 1), i), pos_vectors_i[:, line_j].T)) for line_j in lines_i]
                if pos_vectors_i is not None and lines_i is not None else None
                for i, (lines_i, pos_vectors_i) in enumerate(zip(myof_lines, pos_vectors))]
            _myof_lines_vector_pos = [line for lines in myof_lines_pos_vectors if lines is not None for line in lines]
            self.viewer.add_shapes(name='MyofibrilLines', data=_myof_lines_vector_pos, shape_type='path',
                                   edge_color='red', edge_width=2, opacity=0.5)

    def init_sarcomere_vector_stack(self):
        if self.model.cell is not None and 'pos_vectors' in self.model.cell.structure.data.keys():
            if self.viewer.layers.__contains__('SarcomereVectors'):
                layer = self.viewer.layers.__getitem__('SarcomereVectors')
                self.viewer.layers.remove(layer)
                pass
            if self.viewer.layers.__contains__('MidlinePoints'):
                layer = self.viewer.layers.__getitem__('MidlinePoints')
                self.viewer.layers.remove(layer)
                pass
            # create sarcomere vectors for all frames and add as vector layer
            vectors = []
            pos_vectors = []
            for frame in range(self.model.cell.metadata['frames']):
                if 'params.wavelet_frames' in self.model.cell.structure.data and frame in \
                        self.model.cell.structure.data['params.wavelet_frames']:
                    pos_vectors_frame = self.model.cell.structure.data['pos_vectors'][frame]
                    if len(pos_vectors_frame) > 0:
                        sarc_orientation_vectors = self.model.cell.structure.data['sarcomere_orientation_vectors'][
                            frame]
                        sarc_length_vectors = self.model.cell.structure.data['sarcomere_length_vectors'][frame] / \
                                              self.model.cell.metadata[
                                                  'pixelsize']
                        orientation_vectors = np.asarray(
                            [-np.sin(sarc_orientation_vectors), np.cos(sarc_orientation_vectors)])
                        for i in range(len(pos_vectors_frame[0])):
                            start_point = [frame, pos_vectors_frame[0][i], pos_vectors_frame[1][i]]
                            vector_1 = [frame, orientation_vectors[0][i] * sarc_length_vectors[i] * 0.5,
                                        orientation_vectors[1][i] * sarc_length_vectors[i] * 0.5]
                            vector_2 = [frame, -orientation_vectors[0][i] * sarc_length_vectors[i] * 0.5,
                                        -orientation_vectors[1][i] * sarc_length_vectors[i] * 0.5]
                            pos_vectors.append(start_point)
                            vectors.append([start_point, vector_1])
                            vectors.append([start_point, vector_2])
            self.viewer.add_vectors(vectors, edge_width=0.5, edge_color='purple', name='SarcomereVectors', opacity=0.8,
                                    vector_style='arrow')
            self.viewer.add_points(name='MidlinePoints', data=pos_vectors, face_color='darkgreen', size=2)

    def init_sarcomere_mask_stack(self):
        if self.model.cell is not None and os.path.exists(self.model.cell.file_sarcomere_mask):
            if 'SarcomereMask' in self.viewer.layers:
                layer = self.viewer.layers['SarcomereMask']
                self.viewer.layers.remove(layer)

            tmp = tifffile.imread(self.model.cell.file_sarcomere_mask).astype('uint8')

            if tmp.ndim == 2:  # Single image
                rgba_image = np.zeros((tmp.shape[0], tmp.shape[1], 4), dtype='uint8')
                rgba_image[..., 0] = 255  # Red channel
                rgba_image[..., 1] = 255  # Green channel
                rgba_image[..., 2] = 0  # Blue channel
                rgba_image[..., 3] = np.where(tmp > 0, 102, 0)  # Alpha channel (40% opacity)
            elif tmp.ndim == 3:  # Stack of images
                rgba_image = np.zeros((tmp.shape[0], tmp.shape[1], tmp.shape[2], 4), dtype='uint8')
                rgba_image[..., 0] = 255  # Red channel
                rgba_image[..., 1] = 255  # Green channel
                rgba_image[..., 2] = 0  # Blue channel
                rgba_image[..., 3] = np.where(tmp > 0, 102, 0)  # Alpha channel (40% opacity)

            self.viewer.add_image(rgba_image, name='SarcomereMask', opacity=0.7)

    def init_sarcomere_domain_stack(self):
        if self.model.cell is not None and 'domain_mask' in self.model.cell.structure.data.keys():
            if self.viewer.layers.__contains__('SarcomereDomains'):
                layer = self.viewer.layers.__getitem__('SarcomereDomains')
                self.viewer.layers.remove(layer)

            domain_masks = self.model.cell.structure.data['domain_mask']

            _domain_masks = np.zeros((self.model.cell.metadata['frames'], *self.model.cell.metadata['size']),
                                     dtype='uint16')
            for frame in range(self.model.cell.metadata['frames']):
                if frame in self.model.cell.structure.data['params.domain_frames']:
                    _domain_masks[frame] = domain_masks[frame].toarray()

            self.viewer.add_labels(_domain_masks, name='SarcomereDomains', opacity=0.35)

    def run_async_new(self, parameters, call_lambda, start_message, finished_message, finished_action=None,
                      finished_successful_action=None):
        """
        parameters is a dictionary which contains all necessary variables
        call lambda can be a lambda or function (needs to be callable)
        requirement: it needs two function parameters, first is the worker and second is the parameter-dictionary

        and should use the parameters dictionary

        this method should work (tested roughly :D)
        """
        # todo: add exception handling and print exception with print() and also print it to text area
        if self.model.currentlyProcessing.get_value():
            self.debug('still processing something')
            return

        self.model.currentlyProcessing.set_value(True)
        self.debug(start_message)

        class Worker(QObject):
            finished = pyqtSignal()
            finished_successful = pyqtSignal()
            progress = pyqtSignal(int)
            progress_details = pyqtSignal(str)
            exception = pyqtSignal(str)

            def __init__(self, parameters, call_lambda):
                super().__init__()
                self.succeeded = None
                self.parameters = parameters
                self.call_lambda = call_lambda

            def run(self):
                self.progress.emit(0)
                try:
                    self.call_lambda(self, parameters)
                    self.finished_successful.emit()
                    self.succeeded = True
                except Exception as e:
                    # todo: improve exception display, type of exception, message etc.
                    tb = traceback.format_exc()
                    print(tb)
                    self.succeeded = False
                    self.exception.emit(tb)  # todo: this does not work?
                self.progress.emit(100)
                self.finished.emit()

        # Step 2: Create a QThread object
        self.__worker_thread = QThread()
        # Step 3: Create a worker object
        worker = Worker(parameters=parameters, call_lambda=call_lambda)
        # Step 4: Move worker to the thread
        worker.moveToThread(self.__worker_thread)
        # Step 5: Connect signals and slots
        self.__worker_thread.started.connect(worker.run)
        worker.finished.connect(self.__worker_thread.quit)
        if finished_action is not None:
            worker.finished.connect(finished_action)
        if finished_successful_action is not None:
            worker.finished_successful.connect(finished_successful_action)

        worker.finished.connect(worker.deleteLater)
        self.__worker_thread.finished.connect(self.__worker_thread.deleteLater)

        worker.exception.connect(self.debug)
        worker.progress.connect(self.update_progress)
        worker.progress_details.connect(self.debug_replace)
        # Step 6: Start the thread
        self.__worker_thread.start()
        # Final resets

        # todo: this message gets "eaten" by the last progress_details (depends which is called first)
        self.__worker_thread.finished.connect(lambda: self.__finished_task(finished_message))

        return worker

    def worker_thread(self, on_finished):
        if self.__worker_thread is not None:
            self.__worker_thread.finished.connect(on_finished)

    def __finished_task(self, finished_message=None):
        self.debug(finished_message)
        self.model.currentlyProcessing.set_value(False)
