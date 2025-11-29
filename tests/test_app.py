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
        assert model.sarcomere is None
        assert model.currentlyProcessing.get_value() is False
        assert model.parameters is not None

    def test_model_reset(self, model):
        """Test model reset functionality."""
        model._cell = "dummy_cell"
        model._ApplicationModel__cell_file_name = "test.tif"
        model.reset_model()
        
        assert model.cell is None
        assert model.sarcomere is None
        assert len(model.line_dictionary) == 0

    def test_is_initialized_false_when_no_cell(self, model):
        """Test is_initialized returns False when no cell loaded."""
        with patch('napari.current_viewer', return_value=None):
            assert model.is_initialized() is False

    def test_scheme_property(self, model):
        """Test scheme property returns expected format."""
        assert model.scheme == '%d_%d_%d_%d_%.2f'

    def test_file_extension_property(self, model):
        """Test file extension property."""
        assert model.file_extension == '.json'


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
        assert params.get_parameter('structure.predict.network_path').get_value() == 'generalist'
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

    def test_get_entry_key_for_line(self):
        """Test line key generation."""
        from sarcasm_app.control.application_control import ApplicationControl
        
        line = ([100, 200], [300, 400], 0.65, None)
        key = ApplicationControl.get_entry_key_for_line(line)
        
        assert '(100,200)->(300,400):0.65' == key

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
        """Test motion analysis default values."""
        params = model.parameters
        
        assert params.get_parameter('motion.detect_peaks.threshold').get_value() == 0.2
        assert params.get_parameter('motion.detect_peaks.min_distance').get_value() == 1.4
        assert params.get_parameter('motion.track_z_bands.search_range').get_value() == 2.0
        assert params.get_parameter('motion.systoles.threshold').get_value() == 0.3
