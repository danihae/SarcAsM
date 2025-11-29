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
Custom logging handler for routing log messages to Qt GUI widgets.

This module provides a thread-safe logging handler that emits log messages
to a QTextEdit widget via Qt signals, with color coding based on log level.
"""

import logging
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import QTextEdit


class LogSignalEmitter(QObject):
    """
    Qt signal emitter for thread-safe logging to GUI.
    
    This class provides a signal that can be emitted from any thread
    and safely received by the main GUI thread.
    """
    log_message = pyqtSignal(str, int)  # message, log level


class QTextEditHandler(logging.Handler):
    """
    Custom logging handler that routes log messages to a QTextEdit widget.
    
    This handler uses Qt signals to ensure thread-safe updates to the GUI,
    which is essential since analysis runs in background threads.
    
    Parameters
    ----------
    text_edit : QTextEdit
        The QTextEdit widget to display log messages.
    
    Attributes
    ----------
    LEVEL_COLORS : dict
        Mapping of log levels to QColor for visual distinction.
    
    Examples
    --------
    >>> text_edit = QTextEdit()
    >>> handler = QTextEditHandler(text_edit)
    >>> handler.setLevel(logging.INFO)
    >>> logging.getLogger('sarcasm').addHandler(handler)
    """
    
    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: QColor(128, 128, 128),      # Gray
        logging.INFO: QColor(255, 255, 255),       # White
        logging.WARNING: QColor(255, 200, 50),     # Orange/Yellow
        logging.ERROR: QColor(255, 80, 80),        # Red
        logging.CRITICAL: QColor(255, 50, 50),     # Bright Red
    }
    
    # Level name prefixes for messages
    LEVEL_PREFIXES = {
        logging.DEBUG: '[DEBUG] ',
        logging.INFO: '',  # No prefix for info - cleaner look
        logging.WARNING: '[WARNING] ',
        logging.ERROR: '[ERROR] ',
        logging.CRITICAL: '[CRITICAL] ',
    }
    
    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit
        self.signal_emitter = LogSignalEmitter()
        
        # Connect signal to slot for thread-safe GUI updates
        self.signal_emitter.log_message.connect(self._append_message)
        
        # Set a simple format without timestamps (as requested)
        self.setFormatter(logging.Formatter('%(message)s'))
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by sending it to the GUI via Qt signal.
        
        Parameters
        ----------
        record : logging.LogRecord
            The log record to emit.
        """
        try:
            msg = self.format(record)
            self.signal_emitter.log_message.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)
    
    def _append_message(self, message: str, level: int) -> None:
        """
        Append a formatted message to the QTextEdit widget.
        
        This method is called in the main GUI thread via Qt signal/slot.
        
        Parameters
        ----------
        message : str
            The formatted log message.
        level : int
            The logging level (e.g., logging.INFO, logging.WARNING).
        """
        # Get color for this log level
        color = self.LEVEL_COLORS.get(level, QColor(255, 255, 255))
        prefix = self.LEVEL_PREFIXES.get(level, '')
        
        # Create text format with the appropriate color
        text_format = QTextCharFormat()
        text_format.setForeground(color)
        
        # Move cursor to end and insert colored text
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Insert prefix (if any) and message with color
        cursor.setCharFormat(text_format)
        cursor.insertText(prefix + message + '\n')
        
        # Scroll to bottom to show latest message
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()


def setup_gui_logging(text_edit: QTextEdit, level: int = logging.INFO) -> QTextEditHandler:
    """
    Set up GUI logging for the sarcasm and sarcasm_app packages.
    
    This function creates a QTextEditHandler and attaches it to both the
    'sarcasm' and 'sarcasm_app' root loggers, enabling all log messages 
    from both packages to appear in the GUI.
    
    Parameters
    ----------
    text_edit : QTextEdit
        The QTextEdit widget to display log messages.
    level : int, optional
        Minimum logging level to display (default: logging.INFO).
    
    Returns
    -------
    QTextEditHandler
        The created handler, which can be used to adjust settings later.
    
    Examples
    --------
    >>> text_edit = QTextEdit()
    >>> handler = setup_gui_logging(text_edit)
    >>> # Later, to change log level:
    >>> handler.setLevel(logging.DEBUG)
    """
    # Create the handler
    handler = QTextEditHandler(text_edit)
    handler.setLevel(level)
    
    # Setup logging for both sarcasm and sarcasm_app packages
    for logger_name in ('sarcasm', 'sarcasm_app'):
        pkg_logger = logging.getLogger(logger_name)
        
        # Set logger level if not already set lower
        if pkg_logger.level == logging.NOTSET or pkg_logger.level > level:
            pkg_logger.setLevel(level)
        
        # Add handler to logger
        pkg_logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        pkg_logger.propagate = False
    
    return handler
