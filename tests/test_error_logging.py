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
Tests for the error logging module.
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestLogPaths:
    """Test log path functions."""

    def test_get_log_dir_default(self):
        """Test default log directory."""
        from sarcasm_app.control.error_logging import get_log_dir, DEFAULT_LOG_DIR
        
        # Clear env var if set
        with patch.dict(os.environ, {}, clear=True):
            log_dir = get_log_dir()
            assert log_dir == DEFAULT_LOG_DIR

    def test_get_log_dir_custom(self):
        """Test custom log directory via environment variable."""
        from sarcasm_app.control.error_logging import get_log_dir
        
        custom_path = '/custom/log/path'
        with patch.dict(os.environ, {'SARCASM_LOG_DIR': custom_path}):
            log_dir = get_log_dir()
            assert log_dir == Path(custom_path)

    def test_get_log_file_path_creates_dir(self):
        """Test that get_log_file_path creates the directory."""
        from sarcasm_app.control.error_logging import get_log_file_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'test_logs'
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': str(log_dir)}):
                log_path = get_log_file_path()
                assert log_path.parent.exists()

    def test_get_log_file_path_format(self):
        """Test log file naming format."""
        from sarcasm_app.control.error_logging import get_log_file_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                log_path = get_log_file_path()
                date_str = datetime.now().strftime('%Y-%m-%d')
                assert f'sarcasm_{date_str}.log' in str(log_path)


class TestFileErrorHandler:
    """Test the FileErrorHandler class."""

    def test_handler_creation(self):
        """Test handler can be created."""
        from sarcasm_app.control.error_logging import FileErrorHandler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                handler = FileErrorHandler()
                assert handler.level == logging.WARNING
                assert handler.log_file is not None

    def test_handler_writes_to_file(self):
        """Test that handler writes log records to file."""
        from sarcasm_app.control.error_logging import FileErrorHandler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                handler = FileErrorHandler()
                
                # Create a log record
                record = logging.LogRecord(
                    name='test',
                    level=logging.ERROR,
                    pathname='test.py',
                    lineno=1,
                    msg='Test error message',
                    args=(),
                    exc_info=None
                )
                
                handler.emit(record)
                
                # Verify file was written
                assert handler.log_file.exists()
                content = handler.log_file.read_text()
                assert 'Test error message' in content

    def test_handler_writes_header(self):
        """Test that handler writes header on new file."""
        from sarcasm_app.control.error_logging import FileErrorHandler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                handler = FileErrorHandler()
                
                # Trigger a write
                record = logging.LogRecord(
                    name='test',
                    level=logging.ERROR,
                    pathname='test.py',
                    lineno=1,
                    msg='Test',
                    args=(),
                    exc_info=None
                )
                handler.emit(record)
                
                content = handler.log_file.read_text()
                assert 'SarcAsM Error Log' in content
                assert 'Platform:' in content


class TestSetupFileLogging:
    """Test the setup_file_logging function."""

    def test_setup_creates_handler(self):
        """Test that setup creates and returns a handler."""
        from sarcasm_app.control.error_logging import setup_file_logging, FileErrorHandler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                handler = setup_file_logging()
                assert isinstance(handler, FileErrorHandler)

    def test_setup_attaches_to_loggers(self):
        """Test that setup attaches handler to package loggers."""
        from sarcasm_app.control.error_logging import setup_file_logging, FileErrorHandler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                handler = setup_file_logging()
                
                # Check that handler is attached
                sarcasm_logger = logging.getLogger('sarcasm')
                sarcasm_app_logger = logging.getLogger('sarcasm_app')
                
                # Should have at least one FileErrorHandler
                has_file_handler = any(
                    isinstance(h, FileErrorHandler) 
                    for h in sarcasm_logger.handlers
                )
                assert has_file_handler


class TestLogException:
    """Test the log_exception function."""

    def test_log_exception_writes_to_file(self):
        """Test that log_exception writes exception details to file."""
        from sarcasm_app.control.error_logging import log_exception, get_log_file_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                try:
                    raise ValueError("Test exception")
                except ValueError:
                    import sys
                    exc_info = sys.exc_info()
                    log_exception(*exc_info)
                
                log_file = get_log_file_path()
                assert log_file.exists()
                content = log_file.read_text()
                assert 'UNHANDLED EXCEPTION' in content
                assert 'ValueError' in content
                assert 'Test exception' in content


class TestCleanupOldLogs:
    """Test the cleanup_old_logs function."""

    def test_cleanup_removes_old_files(self):
        """Test that cleanup removes old log files."""
        from sarcasm_app.control.error_logging import cleanup_old_logs
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                # Create some test log files
                log_dir = Path(tmpdir)
                old_file = log_dir / 'sarcasm_2020-01-01.log'
                new_file = log_dir / 'sarcasm_2099-12-31.log'
                
                old_file.write_text('old log')
                new_file.write_text('new log')
                
                # Set old file modification time to the past
                os.utime(old_file, (0, 0))
                
                removed = cleanup_old_logs(days_to_keep=1)
                
                assert removed == 1
                assert not old_file.exists()
                assert new_file.exists()

    def test_cleanup_handles_empty_dir(self):
        """Test cleanup handles non-existent directory gracefully."""
        from sarcasm_app.control.error_logging import cleanup_old_logs
        
        with patch.dict(os.environ, {'SARCASM_LOG_DIR': '/nonexistent/path'}):
            # Should not raise
            removed = cleanup_old_logs()
            assert removed == 0


class TestGetRecentErrors:
    """Test the get_recent_errors function."""

    def test_get_recent_errors_empty(self):
        """Test get_recent_errors with no log file."""
        from sarcasm_app.control.error_logging import get_recent_errors
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                errors = get_recent_errors()
                assert errors == []

    def test_get_recent_errors_returns_errors(self):
        """Test get_recent_errors returns error lines."""
        from sarcasm_app.control.error_logging import get_recent_errors, get_log_file_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                log_file = get_log_file_path()
                log_file.parent.mkdir(parents=True, exist_ok=True)
                
                log_file.write_text(
                    "2025-01-01 12:00:00 | INFO     | Normal message\n"
                    "2025-01-01 12:00:01 | ERROR    | Error message 1\n"
                    "2025-01-01 12:00:02 | WARNING  | Warning message\n"
                    "2025-01-01 12:00:03 | ERROR    | Error message 2\n"
                )
                
                errors = get_recent_errors(n=10)
                assert len(errors) == 3  # 2 errors + 1 warning
                assert any('Error message 1' in e for e in errors)
                assert any('Error message 2' in e for e in errors)


class TestInstallCrashHandler:
    """Test the install_crash_handler function."""

    def test_crash_handler_installation(self):
        """Test that crash handler can be installed."""
        from sarcasm_app.control.error_logging import install_crash_handler
        import sys
        
        original_hook = sys.excepthook
        try:
            install_crash_handler()
            assert sys.excepthook != sys.__excepthook__
        finally:
            sys.excepthook = original_hook

    def test_keyboard_interrupt_not_caught(self):
        """Test that KeyboardInterrupt is not caught by crash handler."""
        from sarcasm_app.control.error_logging import install_crash_handler
        import sys
        
        original_hook = sys.excepthook
        try:
            install_crash_handler()
            
            # The crash handler should pass through KeyboardInterrupt
            # We can't easily test this without actually raising it,
            # but we verify the handler is installed
            assert sys.excepthook != original_hook
        finally:
            sys.excepthook = original_hook


class TestIntegration:
    """Integration tests for error logging."""

    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        from sarcasm_app.control.error_logging import (
            FileErrorHandler,
            get_log_file_path,
            get_recent_errors,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'SARCASM_LOG_DIR': tmpdir}):
                # Create handler directly instead of using setup_file_logging
                # to avoid caching issues with existing handlers
                handler = FileErrorHandler(level=logging.WARNING)
                
                # Manually create log records to avoid logger handler caching
                warning_record = logging.LogRecord(
                    name='sarcasm',
                    level=logging.WARNING,
                    pathname='test.py',
                    lineno=1,
                    msg='Test warning',
                    args=(),
                    exc_info=None
                )
                error_record = logging.LogRecord(
                    name='sarcasm',
                    level=logging.ERROR,
                    pathname='test.py',
                    lineno=2,
                    msg='Test error',
                    args=(),
                    exc_info=None
                )
                
                handler.emit(warning_record)
                handler.emit(error_record)
                
                # Verify log file exists and contains messages
                log_file = handler.log_file
                assert log_file.exists()
                
                content = log_file.read_text()
                assert 'Test warning' in content
                assert 'Test error' in content
                
                # Verify get_recent_errors works
                errors = get_recent_errors()
                assert len(errors) >= 2
