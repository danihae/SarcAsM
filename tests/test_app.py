# -*- coding: utf-8 -*-
# Copyright (c) 2025 University Medical Center Göttingen, Germany.
# All rights reserved.
#
# Patent Pending: DE 10 2024 112 939.5
# SPDX-License-Identifier: LicenseRef-Proprietary-See-LICENSE
#
# This software is licensed under a custom license. See the LICENSE file
# in the root directory for full details.
#
# **Commercial use is prohibited without a separate license.**
# Contact MBM ScienceBridge GmbH (https://sciencebridge.de/en/) for licensing.

"""
Tests for the SarcAsM GUI application.

These tests focus on non-GUI components that can be tested without a display:
- ApplicationModel and Parameters
- Error handling and logging
- Utility functions

For full GUI tests requiring a display, use pytest-qt with xvfb.
"""

import json
import logging
import os

import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestApplicationModel:
    """Test the ApplicationModel class."""

    @pytest.fixture
    def model(self):
        """Create an ApplicationModel instance."""
        # We need to mock napari since it requires a display
        with patch('napari.current_viewer', return_value=None):
            from sarcasm_app.model import ApplicationModel
            return ApplicationModel()

    def test_model_initialization(self, model):
        """Test that the model initializes with correct defaults."""
        assert model.cell is None
        assert model.currentlyProcessing.get_value() is False
        assert model.parameters is not None

    def test_model_reset(self, model):
        """Test model reset functionality."""
        model._cell = "dummy_cell"
        model._ApplicationModel__cell_file_name = "test.tif"
        model.reset_model()

        assert model.cell is None

    def test_is_initialized_false_when_no_cell(self, model):
        """Test is_initialized returns False when no cell loaded."""
        with patch('napari.current_viewer', return_value=None):
            assert model.is_initialized() is False


class TestParameters:
    """Test the Parameters system."""

    @pytest.fixture
    def parameters(self):
        """Create a Parameters instance."""
        from sarcasm_app.model.parameters import Parameters
        return Parameters()

    @pytest.fixture
    def model(self):
        """Create an ApplicationModel instance with parameters."""
        with patch('napari.current_viewer', return_value=None):
            from sarcasm_app.model import ApplicationModel
            return ApplicationModel()

    def test_parameter_get_set(self, model):
        """Test getting and setting parameter values."""
        params = model.parameters
        
        # Test structure parameters
        params.get_parameter('structure.predict.rescale_factor').set_value(2.0)
        assert params.get_parameter('structure.predict.rescale_factor').get_value() == 2.0

    def test_parameter_default_values(self, model):
        """Test that default values are set correctly."""
        params = model.parameters
        
        # Check some default values from set_to_default
        assert params.get_parameter('structure.predict.network_path').get_value() == 'auto'
        assert params.get_parameter('structure.predict.rescale_factor').get_value() == 1.0
        assert params.get_parameter('structure.predict.size_width').get_value() == 1024
        assert params.get_parameter('structure.predict.size_height').get_value() == 1024

    def test_set_to_default(self, model):
        """Test resetting parameters to defaults."""
        params = model.parameters
        
        # Modify some values
        params.get_parameter('structure.predict.rescale_factor').set_value(5.0)
        params.get_parameter('batch.pixel.size').set_value(0.5)
        
        # Reset to defaults
        model.set_to_default()
        
        # Verify defaults are restored
        assert params.get_parameter('structure.predict.rescale_factor').get_value() == 1.0
        assert params.get_parameter('batch.pixel.size').get_value() == 0.1

    @pytest.mark.parametrize('param, method, kwarg', [
        ('structure.vectors.line_width', 'analyze_sarcomere_vectors', 'linewidth'),
        ('structure.vectors.interpolation_factor', 'analyze_sarcomere_vectors', 'interp_factor'),
        ('structure.vectors.radius', 'analyze_sarcomere_vectors', 'median_filter_radius'),
        ('structure.z_band_analysis.threshold', 'analyze_z_bands', 'threshold'),
        ('structure.z_band_analysis.min_length', 'analyze_z_bands', 'min_length'),
        ('structure.z_band_analysis.max_axial_offset', 'analyze_z_bands', 'max_axial_offset'),
    ])
    def test_gui_defaults_match_the_python_defaults(self, model, param, method, kwarg):
        """The same image must give the same numbers in the GUI as with SarcAsM() defaults
        (linewidth 0.2 / interp_factor 0 in the GUI once gave 3398 instead of 3447 vectors)."""
        import inspect
        from sarcasm import SarcAsM
        api = inspect.signature(getattr(SarcAsM, method)).parameters[kwarg].default
        assert model.parameters.get_parameter(param).get_value() == pytest.approx(api)

    def test_auto_patch_size_is_the_default(self, model):
        """Both prediction groups default to letting the device decide the patch size."""
        params = model.parameters
        assert params.get_parameter('structure.predict.auto_patch_size').get_value() is True
        assert params.get_parameter('structure.predict_fast_movie.auto_patch_size').get_value() is True

    def test_patch_size_resolves_to_auto_or_manual(self, model):
        """The helper the controls use must yield 'auto' or the entered dimensions."""
        from sarcasm_app.model import patch_size_from_parameters
        params = model.parameters

        assert patch_size_from_parameters(params, 'structure.predict') == 'auto'
        assert patch_size_from_parameters(params, 'structure.predict_fast_movie') == 'auto'

        params.get_parameter('structure.predict.auto_patch_size').set_value(False)
        params.get_parameter('structure.predict_fast_movie.auto_patch_size').set_value(False)
        assert patch_size_from_parameters(params, 'structure.predict') == (1024, 1024)
        # the 3D model's patch carries the frame count first
        assert patch_size_from_parameters(params, 'structure.predict_fast_movie') == (32, 256, 256)

        params.get_parameter('structure.predict.size_width').set_value(768)
        params.get_parameter('structure.predict.size_height').set_value(640)
        assert patch_size_from_parameters(params, 'structure.predict') == (768, 640)

    def test_manual_patch_size_is_accepted_by_the_backend(self, model):
        """Whatever the helper returns must be a valid max_patch_size for detection."""
        from sarcasm import Utils
        from sarcasm_app.model import patch_size_from_parameters
        params = model.parameters

        assert Utils.check_and_round_max_patch_size(
            patch_size_from_parameters(params, 'structure.predict')) == 'auto'
        params.get_parameter('structure.predict.auto_patch_size').set_value(False)
        params.get_parameter('structure.predict.size_width').set_value(700)
        rounded = Utils.check_and_round_max_patch_size(
            patch_size_from_parameters(params, 'structure.predict'))
        assert rounded == (704, 1024), 'non-multiples of 16 should be rounded up'

    def test_parameters_export_import(self, model):
        """Test parameter export and import functionality."""
        params = model.parameters

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Modify a value
            params.get_parameter('structure.predict.rescale_factor').set_value(3.5)
            
            # Export
            params.store(temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Reset and import
            model.set_to_default()
            params.load(temp_path)
            
            # Verify imported value
            assert params.get_parameter('structure.predict.rescale_factor').get_value() == 3.5
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestParameter:
    """Test the Parameter class."""

    def test_parameter_creation(self):
        """Test Parameter class creation."""
        from sarcasm_app.model.parameter import Parameter
        
        param = Parameter("test_param", 42)
        assert param.get_value() == 42

    def test_parameter_value_change(self):
        """Test Parameter value changes."""
        from sarcasm_app.model.parameter import Parameter
        
        param = Parameter("test_param", "initial")
        param.set_value("modified")
        assert param.get_value() == "modified"


class TestLoggingHandler:
    """Test the custom logging handler."""

    def test_log_signal_emitter_creation(self):
        """Test LogSignalEmitter can be created."""
        from sarcasm_app.control.logging_handler import LogSignalEmitter
        
        emitter = LogSignalEmitter()
        assert hasattr(emitter, 'log_message')

    def test_level_colors_defined(self):
        """Test that level colors are defined for all standard levels."""
        from sarcasm_app.control.logging_handler import QTextEditHandler
        from PyQt5.QtWidgets import QTextEdit, QApplication
        
        # Need QApplication for QTextEdit
        if QApplication.instance() is None:
            app = QApplication([])
        
        text_edit = QTextEdit()
        handler = QTextEditHandler(text_edit)
        
        # Check colors are defined
        assert logging.DEBUG in handler.LEVEL_COLORS
        assert logging.INFO in handler.LEVEL_COLORS
        assert logging.WARNING in handler.LEVEL_COLORS
        assert logging.ERROR in handler.LEVEL_COLORS
        assert logging.CRITICAL in handler.LEVEL_COLORS


class TestErrorHandling:
    """Test error handling in the application."""

    def test_worker_exception_handling(self):
        """Test that worker exceptions are properly caught and logged."""
        # This tests the Worker class exception handling in ApplicationControl
        from PyQt5.QtCore import QObject, pyqtSignal
        
        class MockWorker(QObject):
            finished = pyqtSignal()
            finished_successful = pyqtSignal()
            exception = pyqtSignal(str)
            
            def __init__(self):
                super().__init__()
                self.exception_caught = None
                self.exception.connect(self._on_exception)
            
            def _on_exception(self, msg):
                self.exception_caught = msg
        
        worker = MockWorker()
        # Emit an exception
        worker.exception.emit("Test exception message")
        assert worker.exception_caught == "Test exception message"

    def test_metadata_error_exception(self):
        """Test MetaDataError exception."""
        from sarcasm.exceptions import MetaDataError
        
        with pytest.raises(MetaDataError):
            raise MetaDataError("Invalid metadata")


class TestGPUDetection:
    """Test GPU detection functionality."""

    def test_gpu_detection_no_crash(self):
        """Test that GPU detection doesn't crash."""
        from sarcasm_app.control.application_control import ApplicationControl
        
        # Should not raise any exception
        result = ApplicationControl.is_gpu_available()
        assert isinstance(result, bool)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_type_utils_unbox(self):
        """Test TypeUtils unbox function."""
        from sarcasm.type_utils import TypeUtils
        
        # Test with actual value
        assert TypeUtils.unbox(42) == 42
        assert TypeUtils.unbox("test") == "test"

    def test_type_utils_if_present(self):
        """Test TypeUtils if_present function."""
        from sarcasm.type_utils import TypeUtils
        
        # With value
        result = []
        TypeUtils.if_present(42, lambda x: result.append(x))
        assert result == [42]
        
        # Without value
        result = []
        TypeUtils.if_present(None, lambda x: result.append(x))
        assert result == []


class TestFrameParsing:
    """The frame field of the structure panel, parsed by the shared parser (a copy of the old parser lived
    here and pinned its bugs: invalid text -> frame 0, '1, 3' -> [1])."""

    def test_parse_frames_empty(self):
        from sarcasm._internal.frames import parse_frames_text
        assert parse_frames_text('') is None
        assert parse_frames_text('   ') is None

    def test_parse_frames_all(self):
        from sarcasm._internal.frames import parse_frames_text
        for text in ('all', 'ALL', 'All', ' all '):
            assert parse_frames_text(text) == 'all'

    def test_parse_frames_single(self):
        from sarcasm._internal.frames import parse_frames_text
        assert parse_frames_text('0') == [0]
        assert parse_frames_text(' 5') == [5]
        assert parse_frames_text('100,') == [100]

    def test_parse_frames_list(self):
        from sarcasm._internal.frames import parse_frames_text
        assert parse_frames_text('0,1,2') == [0, 1, 2]
        assert parse_frames_text('10, 5,1') == [1, 5, 10]

    @pytest.mark.parametrize('text', ['invalid', '-1', '3-1', '1.5', '0,2,x'])
    def test_parse_frames_invalid(self, text):
        from sarcasm._internal.frames import parse_frames_text
        with pytest.raises(ValueError):
            parse_frames_text(text)


class TestConfigurationFiles:
    """Test configuration file handling."""

    def test_json_parameter_file_creation(self):
        """Test creating a valid JSON parameter file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump({'test': 'value'}, f)
        
        try:
            # Verify file is valid JSON
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert data == {'test': 'value'}
        finally:
            os.unlink(temp_path)

    def test_invalid_json_detection(self):
        """Test that invalid JSON is properly detected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            f.write('invalid json {{{')
        
        try:
            with pytest.raises(json.JSONDecodeError):
                with open(temp_path, 'r') as f:
                    json.load(f)
        finally:
            os.unlink(temp_path)


class TestApplicationStartup:
    """Test application startup without GUI."""

    def test_imports_succeed(self):
        """Test that all major imports succeed."""
        # These should not raise ImportError
        from sarcasm_app.model import ApplicationModel
        from sarcasm_app.model.parameter import Parameter
        from sarcasm_app.model.parameters import Parameters
        from sarcasm_app.control.logging_handler import QTextEditHandler, LogSignalEmitter
        
        # Verify classes are importable
        assert ApplicationModel is not None
        assert Parameter is not None
        assert Parameters is not None

    def test_version_accessible(self):
        """Test that version is accessible."""
        from sarcasm import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestBatchProcessingParameters:
    """Test batch processing parameter defaults."""

    @pytest.fixture
    def model(self):
        """Create model with batch parameters."""
        with patch('napari.current_viewer', return_value=None):
            from sarcasm_app.model import ApplicationModel
            return ApplicationModel()

    def test_batch_defaults(self, model):
        """Test batch processing default values."""
        params = model.parameters
        
        assert params.get_parameter('batch.pixel.size').get_value() == 0.1
        assert params.get_parameter('batch.frame.time').get_value() == 0.1
        assert params.get_parameter('batch.channel').get_value() == 0
        # 1: files in parallel crash on macOS (numba workqueue) and corrupt GPU inference
        assert params.get_parameter('batch.thread_pool_size').get_value() == 1
        assert params.get_parameter('batch.force.override').get_value() is False


class TestMotionParameters:
    """Test motion analysis parameter defaults."""

    @pytest.fixture
    def model(self):
        """Create model with motion parameters."""
        with patch('napari.current_viewer', return_value=None):
            from sarcasm_app.model import ApplicationModel
            return ApplicationModel()

    def test_motion_defaults(self, model):
        """Test motion analysis default values.

        These follow the track-based pipeline; the LOI peak-detection parameters
        this once checked went with the LOI motion path.
        """
        params = model.parameters

        assert params.get_parameter('motion.track.max_disp_along').get_value() == 1.0
        assert params.get_parameter('motion.track.max_disp_perp').get_value() == 0.2
        assert params.get_parameter('motion.track.ori_tol').get_value() == 45.0
        assert params.get_parameter('motion.track.flow_predictor').get_value() is False   # opt-in, matches the package default
        assert params.get_parameter('motion.group.by').get_value() == 'myofibril'
        assert params.get_parameter('motion.analyze.aggregate').get_value() == 'mean'
        # the operating point is read from the shipped checkpoint, not hard-coded
        from sarcasm_app.model import _default_contraction_threshold
        assert params.get_parameter('motion.analyze.threshold').get_value() == _default_contraction_threshold()
        assert params.get_parameter('motion.analyze.contr_time_min').get_value() == 0.2

    def test_every_parameter_has_a_default(self, model):
        """Registering a parameter without defaulting it leaves the GUI reading None.

        Checking named defaults one by one lets a renamed parameter rot unnoticed --
        which is how the LOI-era assertions above survived the migration that
        deleted them. This covers the whole set instead.
        """
        registered = model.parameters._Parameters__parameters_dict
        assert registered, "no parameters registered"
        # batch.root is a folder the user chooses; it has no meaningful default.
        undefaulted = {name for name, param in registered.items()
                       if param.get_raw_value() is None} - {'batch.root'}
        assert not undefaulted, f"parameters registered without a default: {sorted(undefaulted)}"


_QAPP = None


class TestTrackLayers:
    """The napari track layers of the Motion tab, built against a viewer stand-in.

    napari layer objects work without a window, so a fake viewer that forwards
    ``add_*`` to the layer constructors exercises the real data preparation,
    colouring and selection plumbing headless. Needs the 20 kPa store (skipped in CI).
    """

    class _Layers(dict):
        def remove(self, layer):
            for k, v in list(self.items()):
                if v is layer:
                    del self[k]

    class _FakeDims:
        def __init__(self):
            from napari.utils.events import EmitterGroup
            self.current_step = (0, 0, 0)
            self.events = EmitterGroup(source=self, current_step=None)

    class _FakeViewer:
        def __init__(self):
            import napari
            self.layers = TestTrackLayers._Layers()
            self.mouse_drag_callbacks = []
            self.dims = TestTrackLayers._FakeDims()
            self._napari = napari

        def _add(self, cls, data, **kw):
            layer = cls(data, **kw)
            self.layers[kw.get('name', cls.__name__)] = layer
            return layer

        def add_tracks(self, data, **kw):
            return self._add(self._napari.layers.Tracks, data, **kw)

        def add_points(self, data, **kw):
            return self._add(self._napari.layers.Points, data, **kw)

        def add_shapes(self, data, **kw):
            return self._add(self._napari.layers.Shapes, data, **kw)

    @pytest.fixture(scope='class')
    def control(self, motion_file_path_class):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PyQt5.QtWidgets import QApplication
        global _QAPP                                   # keep the application alive for the widgets
        _QAPP = QApplication.instance() or QApplication([])
        from sarcasm import SarcAsM
        from sarcasm_app.control.application_control import ApplicationControl
        from sarcasm_app.model import ApplicationModel
        # in memory only: the 20 kPa store is the shared reference and keeps its results
        cell = SarcAsM(motion_file_path_class, auto_save=False)
        if 'motion.tracks.slen' not in cell.data:
            pytest.skip('20 kPa store has no tracks')
        if 0 not in list(cell.data.get('params.analyze_myofibrils.frames', [])):
            cell.analyze_myofibrils(frames=0)              # the store holds no myofibril lines
        cell.group_tracks(by='myofibril', reference_frame=0, min_group_size=6)
        cell.scale = (1, cell.metadata.pixelsize, cell.metadata.pixelsize)
        model = ApplicationModel()
        model.set_to_default()
        model._cell = cell
        ctl = ApplicationControl.__new__(ApplicationControl)
        ctl._model = model
        ctl._viewer = self._FakeViewer()
        ctl.track_selected_callbacks, ctl.group_selected_callbacks = [], []
        ctl._track_click_installed = False
        return ctl

    def test_layer_table_is_per_observation(self, control):
        table = control.track_layer_table()
        n_obs = int(np.asarray(control.model.cell.data['motion.tracks.observed']).sum())
        assert table['track_id'].size == n_obs
        assert set(table) >= {'track_id', 't', 'y', 'x', 'slen', 'delta_slen', 'vel', 'group_id', 'coverage'}
        assert np.all(np.diff(table['track_id']) >= 0)                    # sorted by track, then frame
        assert control.track_layer_table() is table                       # cached

    def test_dots_follow_the_frame_and_recolour(self, control):
        control.init_sarcomere_dots()
        control.init_track_groups_layer()
        layers = control.viewer.layers
        table = control.track_layer_table()
        dots = layers['Sarcomeres']
        n0 = int((table['t'] == 0).sum())
        assert dots.data.shape == (n0, 2) and dots.face_color.shape == (n0, 4)   # current frame only
        assert len(layers['Groups'].data) > 5 and layers['Groups'].features['group'].size == len(layers['Groups'].data)
        assert control.viewer.mouse_drag_callbacks == [control._on_canvas_click]
        control.init_sarcomere_dots()                                         # rebuild registers the click once
        assert len(control.viewer.mouse_drag_callbacks) == 1
        # frame change swaps the points
        control.viewer.dims.current_step = (200, 0, 0)
        control.viewer.dims.events.current_step()
        dots = layers['Sarcomeres']
        assert dots.data.shape[0] == int((table['t'] == 200).sum()) != n0
        rows = control._frame_rows(table, 200)
        assert np.allclose(dots.data[:, 0], table['y'][rows])
        for label in ('Group', 'Coverage', 'Velocity', 'SL', 'Track id', 'ΔSL'):
            control.model.parameters.get_parameter('motion.display.color_by').set_value(label)
            control.apply_track_display(show_sarcomeres=True)
            assert np.isfinite(layers['Sarcomeres'].face_color).all()
            assert layers['Sarcomeres'].face_color.shape[0] == dots.data.shape[0]

    def test_nearest_track_hit_test(self, control):
        table = control.track_layer_table()
        cell = control.model.cell
        sy, sx = cell.scale[1], cell.scale[2]
        i = 12345                                                             # some observation
        t, y, x = int(table['t'][i]), table['y'][i], table['x'][i]
        assert control.nearest_track((t, y * sy, x * sx)) == int(table['track_id'][i])
        assert control.nearest_track((t, -50.0, -50.0)) is None               # off the cell: nothing
        assert control.nearest_track((t + 0.4, (y + 0.5) * sy, x * sx)) == int(table['track_id'][i])

    def test_group_highlight_and_selection_callbacks(self, control):
        control.init_sarcomere_dots()
        seen = []
        control.track_selected_callbacks.append(seen.append)
        control.highlight_group(0)
        members = control.track_layer_table()['group_id'] == 0
        ring = control.viewer.layers['Selected group']
        assert len(ring.data) == int(members.sum())
        control.highlight_group(1)                                       # same layer, new data
        assert control.viewer.layers['Selected group'] is ring
        assert len(ring.data) == int((control.track_layer_table()['group_id'] == 1).sum())
        control.highlight_group(None)
        assert len(ring.data) == 0 and not ring.visible

    def test_trace_dock_overlays_group_and_track(self, control):
        from sarcasm_app.view.track_trace_dock import TrackTraceDock
        cell = control.model.cell
        dock = TrackTraceDock()
        dock.set_source(control.track_kinematics(), cell.data['motion.tracks.group_id'],
                        cell.metadata.frametime, 'myofibril')
        assert 'all tracks' in dock.lbl_info.text()
        gid = np.asarray(cell.data['motion.tracks.group_id'])
        track = int(np.flatnonzero(gid == 0)[0])
        dock.show_track(track)
        assert f'Track {track}' in dock.lbl_info.text() and 'myofibril 0' in dock.lbl_info.text()
        for series in ('SL', 'Velocity', 'ΔSL'):
            dock.cb_series.setCurrentText(series)
            assert dock.ax.get_ylabel().startswith(series.replace('Velocity', 'V'))
        dock.show_group(1)
        assert 'myofibril 1' in dock.lbl_info.text()
        dock.set_frame(10)
        lo, hi = dock.ax.get_ylim()
        assert hi - lo < 1.0                                               # robust limits, no outlier blow-up


def test_sarcomere_vector_layer_geometry(motion_file_path):
    """Arrows start at the midline point, point both ways along the sarcomere axis with
    half its length, and carry no time component (which used to double the frame range)."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import napari
    from sarcasm import SarcAsM
    from sarcasm_app.control.application_control import ApplicationControl
    from sarcasm_app.model import ApplicationModel
    cell = SarcAsM(motion_file_path)
    if 'structure.sarcomere.pos' not in cell.data:
        pytest.skip('20 kPa store has no sarcomere vectors')
    cell.scale = (1, cell.metadata.pixelsize, cell.metadata.pixelsize)
    model = ApplicationModel()
    model.set_to_default()
    model._cell = cell
    ctl = ApplicationControl.__new__(ApplicationControl)
    ctl._model = model
    ctl._viewer = TestTrackLayers._FakeViewer()
    ctl._viewer.add_vectors = lambda d, **kw: ctl._viewer._add(napari.layers.Vectors, d, **kw)
    ctl.init_sarcomere_vector_stack()
    vec = np.asarray(ctl.viewer.layers['SarcomereVectors'].data)
    mid = np.asarray(ctl.viewer.layers['MidlinePoints'].data)
    assert vec.shape[0] == 2 * mid.shape[0] and vec.shape[1:] == (2, 3)
    assert np.all(vec[:, 1, 0] == 0)                                          # no time component
    assert vec[:, 0, 0].max() <= cell.metadata.n_stack - 1
    frame = int(vec[0, 0, 0])
    pos = np.asarray(cell.data['structure.sarcomere.pos'][frame]) / cell.metadata.pixelsize
    slen = np.asarray(cell.data['structure.sarcomere.slen'][frame]) / cell.metadata.pixelsize
    ori = np.asarray(cell.data['structure.sarcomere.orientation'][frame])
    n = len(pos)
    block = vec[:2 * n]                                                       # this frame's arrows
    assert np.allclose(np.nan_to_num(block[:n, 0, 1:]), np.nan_to_num(pos))   # start = midline point
    assert np.allclose(np.nan_to_num(block[n:, 0, 1:]), np.nan_to_num(pos))
    axis = np.column_stack([np.sin(ori), np.cos(ori)]) * (0.5 * slen)[:, None]
    assert np.allclose(np.nan_to_num(block[:n, 1, 1:]), np.nan_to_num(axis))  # +axis, half a sarcomere
    assert np.allclose(np.nan_to_num(block[n:, 1, 1:]), np.nan_to_num(-axis))  # -axis


class TestGuiFixes2026_10:
    """Regression tests for the 2026-10-22 GUI review."""

    def test_batch_runs_files_one_by_one_when_parallel_is_unsafe(self):
        from sarcasm_app.control import batch_processing_control as b
        with patch.object(b, '_parallel_files_safe', return_value=False):
            assert b._safe_pool_size(3) == 1
        with patch.object(b, '_parallel_files_safe', return_value=True):
            assert b._safe_pool_size(3) == 3
        assert b._safe_pool_size(1) == 1

    def test_batch_pool_default_is_one(self):
        from sarcasm_app.model import ApplicationModel
        assert ApplicationModel().parameters.get_parameter('batch.thread_pool_size').get_value() == 1

    @pytest.mark.parametrize('aspect', [10.0, 5.0, 2.5, 1.0, 0.8, 0.5, 0.2, 0.05])
    @pytest.mark.parametrize('is_fibre', [True, False])
    def test_summary_layout_keeps_map_proportions(self, aspect, is_fibre):
        """The coverage map's panel has about the image's aspect ratio (within the height limits)."""
        from sarcasm_app.control.motion_analysis_control import summary_layout, SUMMARY_WIDTH_IN
        layout, heights = summary_layout(aspect, is_fibre)
        assert len(layout) == len(heights) and len({len(r) for r in layout}) == 1   # rectangular grid
        names = {n for row in layout for n in row}
        assert {'coverage', 'groups', 'slen', 'dslen', 'raster'} <= names
        assert ('fibrils' in names) == is_fibre
        row = next(i for i, r in enumerate(layout) if 'coverage' in r)
        width = SUMMARY_WIDTH_IN * layout[row].count('coverage') / len(layout[row])
        panel_aspect = width / heights[row]
        assert 1.5 <= heights[row] <= 9.0
        # the panel is never more than ~3x off the image's shape, so imshow does not shrink it to a sliver
        # the panel matches the image's shape unless the row hit its height limit (the image is then
        # drawn undistorted, only smaller, with some empty space beside or below it)
        at_limit = heights[row] <= 1.5 or heights[row] >= 8.0
        assert at_limit or (panel_aspect / aspect < 3.5 and aspect / panel_aspect < 3.5)

    @pytest.mark.parametrize('n_stack', [1, 3])
    def test_sarcomere_vectors_match_viewer_dims(self, n_stack):
        """A single image has no time axis: arrows and points must be 2D there (napari raised
        'Vectors dimensions (3) must be equal to ndim (2)' after the structure analysis)."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from sarcasm_app.control.application_control import ApplicationControl
        px = 0.1
        per_frame = lambda v: [np.asarray(v, float)] * n_stack
        data = {'structure.sarcomere.pos': per_frame([[1.0, 2.0], [3.0, 4.0]]),
                'structure.sarcomere.orientation': per_frame([0.0, 1.0]),
                'structure.sarcomere.slen': per_frame([1.8, 1.9]),
                'params.analyze_sarcomere_vectors.frames': list(range(n_stack))}
        scale = (px, px) if n_stack == 1 else (1, px, px)
        cell = SimpleNamespace(data=data, scale=scale,
                               metadata=SimpleNamespace(pixelsize=px, n_stack=n_stack))
        ctl = SimpleNamespace(model=SimpleNamespace(cell=cell), viewer=MagicMock())
        ApplicationControl.init_sarcomere_vector_stack(ctl)
        vectors = ctl.viewer.add_vectors.call_args.args[0]
        points = ctl.viewer.add_points.call_args.args[0]
        assert vectors.shape == (4 * n_stack, 2, len(scale))
        assert points.shape == (2 * n_stack, len(scale))

    def test_trace_table_exports_every_member_with_units(self):
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])          # keep a reference, or Qt aborts
        from sarcasm_app.view.track_trace_dock import TrackTraceDock
        T = 5
        kin = {'slen': np.arange(3 * T, dtype=float).reshape(3, T), 'delta_slen': np.zeros((3, T)),
               'vel': np.zeros((3, T)), 'contr': np.array([0, 1, 1, 0, 0], bool)}
        dock = TrackTraceDock()
        dock.set_source(kin, np.array([0, 0, 1]), 0.01, 'myofibril')
        dock.cb_series.setCurrentText('SL')
        dock.show_group(0)
        table = dock.traces_table()
        assert list(table.columns) == ['time [s]', 'myofibril 0 mean SL [µm]', 'contracting (group signal)',
                                       'track 0 SL [µm]', 'track 1 SL [µm]']
        assert table['time [s]'].iloc[-1] == pytest.approx(0.04)
        assert list(table['track 1 SL [µm]']) == [5, 6, 7, 8, 9]
        dock.deleteLater()
        assert app is not None


class TestGuiInputFixes2026_10_24:
    """Regression tests for the 2026-10-24 review of what careless input does to the GUI."""

    @pytest.mark.parametrize('text, expected', [('0.1', 0.1), ('0,1', 0.1), (' 0.065 ', 0.065), ('', None)])
    def test_calibration_numbers_are_parsed(self, text, expected):
        from sarcasm_app.control.file_selection_control import _parse_positive
        value, error = _parse_positive(text)
        assert error is None and value == expected

    @pytest.mark.parametrize('text', ['abc', '0', '-0.1', 'nan', 'inf', '0.1 µm', '1,000.5'])
    def test_calibration_nonsense_is_refused(self, text):
        from sarcasm_app.control.file_selection_control import _parse_positive
        value, error = _parse_positive(text)
        assert value is None and error

    def test_failed_display_releases_the_busy_flag(self):
        """An error after the file was opened (e.g. wrong axes) left the app refusing every later step."""
        from types import SimpleNamespace
        from sarcasm_app.model.parameter import Parameter
        from sarcasm_app.control import file_selection_control as fsc
        model = SimpleNamespace(currentlyProcessing=Parameter('currentlyProcessing', False),
                                init_cell=lambda file, **kwargs: None, cell=None)
        main = MagicMock()
        main.model = model
        main.init_image_stack.side_effect = ValueError('Permutation mismatch')
        ctl = fsc.FileSelectionControl(MagicMock(), main)
        with patch.object(fsc.QMessageBox, 'warning') as warning:
            ctl._init_file('movie.tif')
        assert warning.called
        assert model.currentlyProcessing.get_value() is False
        assert main.clean_up_on_new_image.call_count == 2          # before opening, and after the failure

    def test_metadata_is_not_changed_while_a_step_runs(self):
        from types import SimpleNamespace
        from sarcasm_app.model.parameter import Parameter
        from sarcasm_app.control import file_selection_control as fsc
        cell = MagicMock()
        main = MagicMock()
        main.model = SimpleNamespace(currentlyProcessing=Parameter('currentlyProcessing', True), cell=cell)
        ctl = fsc.FileSelectionControl(MagicMock(), main)
        with patch.object(fsc.FileSelectionControl, '_init_meta_data') as refill:
            assert ctl.apply_metadata(pixelsize=0.2) is False
        assert refill.called
        cell.save_metadata.assert_not_called()

    def test_parameter_with_a_wrong_type_keeps_its_value(self):
        from PyQt5.QtWidgets import QApplication, QDoubleSpinBox
        app = QApplication.instance() or QApplication([])
        from sarcasm_app.model.parameters import Parameters
        params = Parameters()
        params.set_parameter('structure.predict.rescale_factor', 1.0)
        spin = QDoubleSpinBox()
        params.get_parameter('structure.predict.rescale_factor').connect(spin)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'p.json')
            with open(path, 'w') as f:
                json.dump({'structure.predict.rescale_factor': 'abc'}, f)
            with pytest.raises(ValueError, match='rescale_factor'):
                params.load(path)
            with open(path, 'w') as f:
                json.dump([1, 2], f)
            with pytest.raises(ValueError, match='object'):
                params.load(path)
        assert params.get_parameter('structure.predict.rescale_factor').get_value() == 1.0
        assert app is not None

    def test_out_of_range_frames_message_stays_short(self):
        from sarcasm._internal.frames import canonical_frames
        with pytest.raises(ValueError) as err:
            canonical_frames(list(range(1_000_000)), 10)
        assert len(str(err.value)) < 300
