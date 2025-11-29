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
Error logging and crash handling for SarcAsM application.

This module provides:
- File-based error logging to ~/.sarcasm/logs/
- Global exception handler for graceful crash handling
- Error reporting utilities

The log file location can be customized via the SARCASM_LOG_DIR environment variable.
"""

import atexit
import logging
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QMessageBox, QApplication

# Default log directory
DEFAULT_LOG_DIR = Path.home() / '.sarcasm' / 'logs'


def get_log_dir() -> Path:
    """
    Get the log directory path.
    
    Returns
    -------
    Path
        The path to the log directory. Uses SARCASM_LOG_DIR environment
        variable if set, otherwise defaults to ~/.sarcasm/logs/
    """
    log_dir = os.environ.get('SARCASM_LOG_DIR')
    if log_dir:
        return Path(log_dir)
    return DEFAULT_LOG_DIR


def get_log_file_path() -> Path:
    """
    Get the path to the current log file.
    
    Creates log directory if it doesn't exist. Log files are named with
    the current date: sarcasm_YYYY-MM-DD.log
    
    Returns
    -------
    Path
        The full path to today's log file.
    """
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    return log_dir / f'sarcasm_{date_str}.log'


class FileErrorHandler(logging.Handler):
    """
    Logging handler that writes error and above messages to a log file.
    
    This handler creates a rotating log file per day and only logs
    WARNING level and above by default. It includes detailed timestamps,
    module names, and full tracebacks for errors.
    
    Parameters
    ----------
    level : int, optional
        Minimum logging level (default: logging.WARNING)
    
    Attributes
    ----------
    log_file : Path
        Path to the current log file
    """
    
    def __init__(self, level: int = logging.WARNING):
        super().__init__(level)
        self.log_file = get_log_file_path()
        
        # Detailed format for file logging
        self.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Write header on first creation
        self._write_header_if_new()
    
    def _write_header_if_new(self):
        """Write header information if this is a new log file."""
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write('=' * 80 + '\n')
                    f.write(f'SarcAsM Error Log - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                    f.write(f'Platform: {platform.system()} {platform.release()}\n')
                    f.write(f'Python: {sys.version}\n')
                    try:
                        from sarcasm import __version__
                        f.write(f'SarcAsM Version: {__version__}\n')
                    except ImportError:
                        f.write('SarcAsM Version: Unknown\n')
                    f.write('=' * 80 + '\n\n')
            except Exception:
                pass  # Don't crash if we can't write header
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Write a log record to the file.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to write
        """
        try:
            msg = self.format(record)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
                
                # If there's exception info, write the full traceback
                if record.exc_info:
                    f.write('Traceback:\n')
                    f.write(''.join(traceback.format_exception(*record.exc_info)))
                    f.write('\n')
        except Exception:
            self.handleError(record)


def setup_file_logging(level: int = logging.WARNING) -> FileErrorHandler:
    """
    Set up file-based error logging for the application.
    
    This function creates a FileErrorHandler and attaches it to the
    'sarcasm' and 'sarcasm_app' loggers. All WARNING, ERROR, and CRITICAL
    level messages will be written to the log file.
    
    Parameters
    ----------
    level : int, optional
        Minimum logging level to log to file (default: logging.WARNING)
    
    Returns
    -------
    FileErrorHandler
        The created handler
    
    Examples
    --------
    >>> handler = setup_file_logging()
    >>> print(f"Logs will be written to: {handler.log_file}")
    """
    handler = FileErrorHandler(level)
    
    for logger_name in ('sarcasm', 'sarcasm_app'):
        pkg_logger = logging.getLogger(logger_name)
        
        # Set logger level if not already set lower
        if pkg_logger.level == logging.NOTSET or pkg_logger.level > level:
            pkg_logger.setLevel(level)
        
        # Avoid duplicate handlers
        for existing_handler in pkg_logger.handlers:
            if isinstance(existing_handler, FileErrorHandler):
                return existing_handler
        
        pkg_logger.addHandler(handler)
    
    return handler


def log_exception(exc_type, exc_value, exc_tb) -> None:
    """
    Log an exception with full traceback to the error log.
    
    This function can be used as sys.excepthook to catch unhandled exceptions.
    
    Parameters
    ----------
    exc_type : type
        The exception type
    exc_value : BaseException
        The exception instance
    exc_tb : traceback
        The traceback object
    """
    logger = logging.getLogger('sarcasm_app')
    
    # Format the exception
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = ''.join(tb_lines)
    
    logger.critical(f"Unhandled exception: {exc_type.__name__}: {exc_value}")
    
    # Write detailed traceback to file
    try:
        log_file = get_log_file_path()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n' + '!' * 80 + '\n')
            f.write(f'UNHANDLED EXCEPTION at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('!' * 80 + '\n')
            f.write(tb_text)
            f.write('\n')
    except Exception:
        pass  # Don't crash while logging a crash


def show_crash_dialog(exc_type, exc_value, exc_tb) -> None:
    """
    Show a user-friendly crash dialog with option to view log.
    
    Parameters
    ----------
    exc_type : type
        The exception type
    exc_value : BaseException
        The exception instance
    exc_tb : traceback
        The traceback object
    """
    log_file = get_log_file_path()
    
    # Try to show a dialog if Qt is available
    app = QApplication.instance()
    if app is not None:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("SarcAsM - Unexpected Error")
        msg_box.setText(
            f"An unexpected error occurred:\n\n"
            f"{exc_type.__name__}: {exc_value}\n\n"
            f"The error has been logged to:\n{log_file}"
        )
        msg_box.setDetailedText(''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


def install_crash_handler() -> None:
    """
    Install global exception handler for graceful crash handling.
    
    This replaces sys.excepthook with a custom handler that:
    1. Logs the exception to the error log file
    2. Shows a user-friendly error dialog
    3. Allows the application to exit gracefully
    
    Should be called early in application startup.
    
    Examples
    --------
    >>> install_crash_handler()
    >>> # Now unhandled exceptions will be logged and show a dialog
    """
    def exception_hook(exc_type, exc_value, exc_tb):
        # Don't handle KeyboardInterrupt (Ctrl+C)
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        
        # Log the exception
        log_exception(exc_type, exc_value, exc_tb)
        
        # Show dialog to user
        show_crash_dialog(exc_type, exc_value, exc_tb)
        
        # Call the default handler
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    
    sys.excepthook = exception_hook


def cleanup_old_logs(days_to_keep: int = 30) -> int:
    """
    Remove log files older than specified days.
    
    Parameters
    ----------
    days_to_keep : int, optional
        Number of days of logs to keep (default: 30)
    
    Returns
    -------
    int
        Number of log files removed
    """
    log_dir = get_log_dir()
    if not log_dir.exists():
        return 0
    
    removed = 0
    cutoff = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    
    for log_file in log_dir.glob('sarcasm_*.log'):
        try:
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                removed += 1
        except Exception:
            pass  # Ignore files we can't delete
    
    return removed


def get_recent_errors(n: int = 10) -> list[str]:
    """
    Get the most recent error messages from the log file.
    
    Parameters
    ----------
    n : int, optional
        Number of recent errors to return (default: 10)
    
    Returns
    -------
    list[str]
        List of recent error messages
    """
    log_file = get_log_file_path()
    if not log_file.exists():
        return []
    
    errors = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '| ERROR' in line or '| CRITICAL' in line or '| WARNING' in line:
                    errors.append(line.strip())
    except Exception:
        pass
    
    return errors[-n:]
