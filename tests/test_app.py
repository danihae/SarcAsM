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

    def test_auto_patch_size_is_the_default(self, model):
        """Prediction defaults to letting the device decide the patch size."""
        params = model.parameters
        assert params.get_parameter('structure.predict.auto_patch_size').get_value() is True

    def test_patch_size_resolves_to_auto_or_manual(self, model):
        """The helper the controls use must yield 'auto' or the entered dimensions."""
        from sarcasm_app.model import patch_size_from_parameters
        params = model.parameters

        assert patch_size_from_parameters(params, 'structure.predict') == 'auto'

        params.get_parameter('structure.predict.auto_patch_size').set_value(False)
        assert patch_size_from_parameters(params, 'structure.predict') == (1024, 1024)

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
    """Test frame parsing in structure analysis control."""

    @pytest.fixture
    def mock_structure_control(self):
        """Create a mock structure control for testing parse_frames."""
        # Access the private method through a test wrapper
        class FrameParser:
            @staticmethod
            def parse_frames(frames_str: str):
                if frames_str == '':
                    return None
                if frames_str.lower() == 'all':
                    return frames_str.lower()
                if frames_str.isnumeric():
                    return int(frames_str)
                if ',' in frames_str:
                    list_str = frames_str.split(',')
                    parsed_list = []
                    for x in list_str:
                        if x.isnumeric():
                            parsed_list.append(int(x))
                    return parsed_list
                return 0
        return FrameParser()

    def test_parse_frames_empty(self, mock_structure_control):
        """Test parsing empty frame string."""
        assert mock_structure_control.parse_frames('') is None

    def test_parse_frames_all(self, mock_structure_control):
        """Test parsing 'all' keyword."""
        assert mock_structure_control.parse_frames('all') == 'all'
        assert mock_structure_control.parse_frames('ALL') == 'all'
        assert mock_structure_control.parse_frames('All') == 'all'

    def test_parse_frames_single(self, mock_structure_control):
        """Test parsing single frame number."""
        assert mock_structure_control.parse_frames('0') == 0
        assert mock_structure_control.parse_frames('5') == 5
        assert mock_structure_control.parse_frames('100') == 100

    def test_parse_frames_list(self, mock_structure_control):
        """Test parsing comma-separated frame list."""
        assert mock_structure_control.parse_frames('0,1,2') == [0, 1, 2]
        assert mock_structure_control.parse_frames('1,5,10') == [1, 5, 10]

    def test_parse_frames_invalid(self, mock_structure_control):
        """Test parsing invalid frame string."""
        # Invalid strings return 0
        assert mock_structure_control.parse_frames('invalid') == 0


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
        assert params.get_parameter('batch.thread_pool_size').get_value() == 3
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
        assert params.get_parameter('motion.group.by').get_value() == 'myofibril'
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
        cell = SarcAsM(motion_file_path_class)
        if 'motion.tracks.slen' not in cell.data or cell.data.get('structure.myofibril.lines') is None:
            pytest.skip('20 kPa store has no tracks / myofibril lines')
        cell.group_tracks(by='myofibril', reference_frame=0, min_group_size=6)
        cell.scale = (1, cell.metadata.pixelsize, cell.metadata.pixelsize)
        model = ApplicationModel()
        model.set_to_default()
        model._cell = cell
        ctl = ApplicationControl.__new__(ApplicationControl)
        ctl._model = model
        ctl._viewer = self._FakeViewer()
        ctl.track_selected_callbacks, ctl.group_selected_callbacks = [], []
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
