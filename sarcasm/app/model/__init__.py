import napari

from .parameters import Parameters
from .parameter import Parameter
from sarcasm import SarcAsM, Motion


class ApplicationModel:
    """
    The ApplicationModel concentrates all necessary parameters for calling the sarcasm_old backend methods
    and provides via Parameters and Parameter class methods to bind those to the UI.
    """
    def __init__(self):
        self._cell = None
        self.__cell_file_name = None
        self.currentlyProcessing = Parameter("currentlyProcessing", False)
        self.__file_extension = ".json"
        self.__line_dictionary = {}  # todo: remove the line dictionary
        self.__sarcomere = None
        self.__scheme = f'%d_%d_%d_%d_%d'
        self.__parameters = Parameters()
        self.__create_parameters()
        self.set_to_default()

    @property
    def scheme(self):
        return self.__scheme

    def reset_model(self):
        self._cell = None
        self.__cell_file_name = None
        self.__line_dictionary = {}
        self.__sarcomere = None

    @property
    def line_dictionary(self):
        return self.__line_dictionary

    @property
    def parameters(self):
        return self.__parameters

    @property
    def sarcomere(self):
        return self.__sarcomere

    @property
    def cell(self):
        return self._cell

    @property
    def file_extension(self):
        return self.__file_extension

    def init_cell(self, cell_file, correct_phase_leica):
        self.__cell_file_name = cell_file
        self._cell = SarcAsM(cell_file, correct_phase_leica=correct_phase_leica, use_gui=True)

    def init_sarcomere(self, loi_name):
        self.__sarcomere = Motion(self.__cell_file_name, loi_name=loi_name)

    def is_initialized(self):
        # check if file is loaded, check if viewer is active(not closed)
        result = True
        if self._cell is None:
            result = False
        if self.__cell_file_name == '' or self.__cell_file_name is None:
            result = False
        if napari.current_viewer() is None:
            result = False
        return result

    def set_to_default(self):
        # region file-load parameters
        self.__parameters.get_parameter(name='file.load.correct_phase').set_value(False)
        # endregion
        # region structure parameters
        self.__parameters.get_parameter(name='structure.predict.network_path').set_value('generalist')
        self.__parameters.get_parameter(name='structure.predict.siam_unet').set_value(False)
        self.__parameters.get_parameter(name='structure.predict.size_width').set_value(
            1024)  # is the predict_size_min from ui
        self.__parameters.get_parameter(name='structure.predict.size_height').set_value(
            1024)  # is the predict_size_max from ui
        self.__parameters.get_parameter(name='structure.predict.clip_thresh_min').set_value(0.)
        self.__parameters.get_parameter(name='structure.predict.clip_thresh_max').set_value(99.8)

        self.__parameters.get_parameter(name='structure.predict.cell_area.network_path').set_value('generalist')
        self.__parameters.get_parameter(name='structure.predict.cell_area.size_width').set_value(1024)
        self.__parameters.get_parameter(name='structure.predict.cell_area.size_height').set_value(1024)
        self.__parameters.get_parameter(name='structure.predict.cell_area.clip_thresh_min').set_value(0.)
        self.__parameters.get_parameter(name='structure.predict.cell_area.clip_thresh_max').set_value(99.8)



        self.__parameters.get_parameter(name='structure.timepoints').set_value('')
        self.__parameters.get_parameter(name='structure.plot').set_value(False)

        self.__parameters.get_parameter(name='structure.z_band_analysis.threshold').set_value(0.1)
        self.__parameters.get_parameter(name='structure.z_band_analysis.min_length').set_value(1.0)

        self.__parameters.get_parameter(name='structure.wavelet.filter_size').set_value(3.0)
        self.__parameters.get_parameter(name='structure.wavelet.sigma').set_value(0.150)
        self.__parameters.get_parameter(name='structure.wavelet.width').set_value(0.5)
        self.__parameters.get_parameter(name='structure.wavelet.length_limit_lower').set_value(1.4)
        self.__parameters.get_parameter(name='structure.wavelet.length_limit_upper').set_value(2.4)
        self.__parameters.get_parameter(name='structure.wavelet.length_step').set_value(0.05)
        self.__parameters.get_parameter(name='structure.wavelet.orientation_limit_lower').set_value(-90)
        self.__parameters.get_parameter(name='structure.wavelet.orientation_limit_upper').set_value(90)
        self.__parameters.get_parameter(name='structure.wavelet.orientation_step').set_value(15)
        self.__parameters.get_parameter(name='structure.wavelet.absolute_threshold').set_value(False)
        self.__parameters.get_parameter(name='structure.wavelet.score_threshold').set_value(90)
        self.__parameters.get_parameter(name='structure.wavelet.save_all').set_value(False)

        self.__parameters.get_parameter(name='structure.myofibril.n_seeds').set_value(500)
        self.__parameters.get_parameter(name='structure.myofibril.score_threshold_empty').set_value(True)
        self.__parameters.get_parameter(name='structure.myofibril.score_threshold').set_value(0)
        self.__parameters.get_parameter(name='structure.myofibril.persistence').set_value(3)
        self.__parameters.get_parameter(name='structure.myofibril.threshold_distance').set_value(0.3)

        self.__parameters.get_parameter(name='structure.domain.analysis.score_threshold').set_value(0.)
        self.__parameters.get_parameter(name='structure.domain.analysis.reduce').set_value(3)
        self.__parameters.get_parameter(name='structure.domain.analysis.weight_length').set_value(0)
        self.__parameters.get_parameter(name='structure.domain.analysis.distance_threshold').set_value(3)
        self.__parameters.get_parameter(name='structure.domain.analysis.area_min').set_value(200)


        # endregion

        # region roi parameters
        self.__parameters.get_parameter(name='loi.detect.timepoint').set_value(0)
        self.__parameters.get_parameter(name='loi.detect.persistence').set_value(8)
        self.__parameters.get_parameter(name='loi.detect.threshold_distance').set_value(0.3)
        self.__parameters.get_parameter(name='loi.detect.score_threshold').set_value(10000.0)
        self.__parameters.get_parameter(name='loi.detect.score_threshold_automatic').set_value(True)
        self.__parameters.get_parameter(name='loi.detect.number_limits_lower').set_value(10)
        self.__parameters.get_parameter(name='loi.detect.number_limits_upper').set_value(50)
        self.__parameters.get_parameter(name='loi.detect.msc_limits_lower').set_value(0)
        self.__parameters.get_parameter(name='loi.detect.msc_limits_upper').set_value(1000)
        self.__parameters.get_parameter(name='loi.detect.distance_threshold_lois').set_value(40)
        self.__parameters.get_parameter(name='loi.detect.n_longest').set_value(4)
        self.__parameters.get_parameter(name='loi.detect.line_width').set_value(12)
        self.__parameters.get_parameter(name='loi.detect.plot').set_value(False)
        self.__parameters.get_parameter(name='loi.detect.export_raw').set_value(False)
        # endregion

        # region motion parameters
        self.__parameters.get_parameter(name='motion.detect_peaks.threshold').set_value(0.05)
        self.__parameters.get_parameter(name='motion.detect_peaks.min_distance').set_value(1)
        self.__parameters.get_parameter(name='motion.detect_peaks.width').set_value(7)

        self.__parameters.get_parameter(name='motion.track_z_bands.search_range').set_value(1.0)
        self.__parameters.get_parameter(name='motion.track_z_bands.memory').set_value(10)
        self.__parameters.get_parameter(name='motion.track_z_bands.memory_interpolation').set_value(5)

        self.__parameters.get_parameter(name='motion.systoles.weights').set_value('')  # weights is a network file
        self.__parameters.get_parameter(name='motion.systoles.threshold').set_value(0.01)
        self.__parameters.get_parameter(name='motion.systoles.slen_limits.lower').set_value(1.2)
        self.__parameters.get_parameter(name='motion.systoles.slen_limits.upper').set_value(3.0)
        self.__parameters.get_parameter(name='motion.systoles.n_sarcomeres_min').set_value(8)
        self.__parameters.get_parameter(name='motion.systoles.buffer_frames').set_value(10)
        self.__parameters.get_parameter(name='motion.systoles.contr_time_min').set_value(0.2)
        self.__parameters.get_parameter(name='motion.systoles.merge_time_max').set_value(0.1)

        self.__parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.filter_params_z_pos.window_length').set_value(13)
        self.__parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.filter_params_z_pos.polyorder').set_value(7)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_lower').set_value(1.2)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_upper').set_value(3.0)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.dilate_systoles').set_value(0.0)
        self.__parameters.get_parameter(
            name='motion.get_sarcomere_trajectories.filter_params_vel.window_length').set_value(13)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.filter_params_vel.polyorder').set_value(
            1)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.equ_limits_lower').set_value(1.5)
        self.__parameters.get_parameter(name='motion.get_sarcomere_trajectories.equ_limits_upper').set_value(2.2)
        # endregion

        # region batch processing parameters
        self.__parameters.get_parameter(name='batch.pixel.size').set_value(0)
        self.__parameters.get_parameter(name='batch.frame.time').set_value(0)
        self.__parameters.get_parameter(name='batch.force.override').set_value(False)
        # endregion

        pass

    def __create_parameters(self):
        # region file-load parameters
        self.__parameters.set_parameter(name='file.load.correct_phase')
        # endregion
        # region structure parameters
        self.__parameters.set_parameter(name='structure.predict.network_path')
        self.__parameters.set_parameter(name='structure.predict.siam_unet')
        self.__parameters.set_parameter(name='structure.predict.size_width')  # is the predict_size_min from ui
        self.__parameters.set_parameter(name='structure.predict.size_height')  # is the predict_size_max from ui
        self.__parameters.set_parameter(name='structure.predict.clip_thresh_min')
        self.__parameters.set_parameter(name='structure.predict.clip_thresh_max')

        self.__parameters.set_parameter(name='structure.predict.cell_area.network_path')
        self.__parameters.set_parameter(name='structure.predict.cell_area.size_width')  # is the predict_size_min from ui
        self.__parameters.set_parameter(name='structure.predict.cell_area.size_height')  # is the predict_size_max from ui
        self.__parameters.set_parameter(name='structure.predict.cell_area.clip_thresh_min')
        self.__parameters.set_parameter(name='structure.predict.cell_area.clip_thresh_max')

        self.__parameters.set_parameter(name='structure.timepoints')
        self.__parameters.set_parameter(name='structure.plot')

        self.__parameters.set_parameter(name='structure.z_band_analysis.threshold')
        self.__parameters.set_parameter(name='structure.z_band_analysis.min_length')

        self.__parameters.set_parameter(name='structure.wavelet.filter_size')
        self.__parameters.set_parameter(name='structure.wavelet.sigma')
        self.__parameters.set_parameter(name='structure.wavelet.width')
        self.__parameters.set_parameter(name='structure.wavelet.length_limit_lower')
        self.__parameters.set_parameter(name='structure.wavelet.length_limit_upper')
        self.__parameters.set_parameter(name='structure.wavelet.length_step')
        self.__parameters.set_parameter(name='structure.wavelet.orientation_limit_lower')
        self.__parameters.set_parameter(name='structure.wavelet.orientation_limit_upper')
        self.__parameters.set_parameter(name='structure.wavelet.orientation_step')
        self.__parameters.set_parameter(name='structure.wavelet.absolute_threshold')
        self.__parameters.set_parameter(name='structure.wavelet.score_threshold')
        self.__parameters.set_parameter(name='structure.wavelet.save_all')

        self.__parameters.set_parameter(name='structure.myofibril.n_seeds')
        self.__parameters.set_parameter(name='structure.myofibril.score_threshold_empty')
        self.__parameters.set_parameter(name='structure.myofibril.score_threshold')
        self.__parameters.set_parameter(name='structure.myofibril.persistence')
        self.__parameters.set_parameter(name='structure.myofibril.threshold_distance')

        self.__parameters.set_parameter(name='structure.domain.analysis.score_threshold')
        self.__parameters.set_parameter(name='structure.domain.analysis.reduce')
        self.__parameters.set_parameter(name='structure.domain.analysis.weight_length')
        self.__parameters.set_parameter(name='structure.domain.analysis.distance_threshold')
        self.__parameters.set_parameter(name='structure.domain.analysis.area_min')

        # endregion

        # region roi parameters
        self.__parameters.set_parameter(name='loi.detect.timepoint')
        self.__parameters.set_parameter(name='loi.detect.persistence')
        self.__parameters.set_parameter(name='loi.detect.threshold_distance')
        self.__parameters.set_parameter(name='loi.detect.score_threshold')
        self.__parameters.set_parameter(name='loi.detect.score_threshold_automatic')
        self.__parameters.set_parameter(name='loi.detect.number_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.number_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.msc_limits_lower')
        self.__parameters.set_parameter(name='loi.detect.msc_limits_upper')
        self.__parameters.set_parameter(name='loi.detect.distance_threshold_lois')
        self.__parameters.set_parameter(name='loi.detect.n_longest')
        self.__parameters.set_parameter(name='loi.detect.line_width')
        self.__parameters.set_parameter(name='loi.detect.plot')
        self.__parameters.set_parameter(name='loi.detect.export_raw')
        # endregion

        # region motion parameters
        self.__parameters.set_parameter(name='motion.detect_peaks.threshold')
        self.__parameters.set_parameter(name='motion.detect_peaks.min_distance')
        self.__parameters.set_parameter(name='motion.detect_peaks.width')

        self.__parameters.set_parameter(name='motion.track_z_bands.search_range')
        self.__parameters.set_parameter(name='motion.track_z_bands.memory')
        self.__parameters.set_parameter(name='motion.track_z_bands.memory_interpolation')

        self.__parameters.set_parameter(name='motion.systoles.weights')  # weights is a network file
        self.__parameters.set_parameter(name='motion.systoles.threshold')
        self.__parameters.set_parameter(name='motion.systoles.slen_limits.lower')
        self.__parameters.set_parameter(name='motion.systoles.slen_limits.upper')
        self.__parameters.set_parameter(name='motion.systoles.n_sarcomeres_min')
        self.__parameters.set_parameter(name='motion.systoles.buffer_frames')
        self.__parameters.set_parameter(name='motion.systoles.contr_time_min')
        self.__parameters.set_parameter(name='motion.systoles.merge_time_max')

        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.filter_params_z_pos.window_length')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.filter_params_z_pos.polyorder')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_lower')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.s_length_limits_upper')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.dilate_systoles')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.filter_params_vel.window_length')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.filter_params_vel.polyorder')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.equ_limits_lower')
        self.__parameters.set_parameter(name='motion.get_sarcomere_trajectories.equ_limits_upper')
        # endregion

        # region batch processing parameters
        self.__parameters.set_parameter(name='batch.pixel.size')
        self.__parameters.set_parameter(name='batch.frame.time')
        self.__parameters.set_parameter(name='batch.force.override')
        # endregion
