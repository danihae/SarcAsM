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

# Configure matplotlib before any imports to speed up startup
import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser('~'), '.sarcasm_mpl')

import logging

from sarcasm_app import Application
from sarcasm_app.control.error_logging import (
    setup_file_logging,
    install_crash_handler,
    cleanup_old_logs,
    get_log_file_path,
)


def main():
    """
    Main entry point for the SarcAsM application.
    
    Sets up error logging, installs crash handler, and launches the GUI.
    Error logs are written to ~/.sarcasm/logs/sarcasm_YYYY-MM-DD.log
    """
    # Set up file-based error logging (WARNING and above)
    file_handler = setup_file_logging(level=logging.WARNING)
    
    # Install global crash handler for unhandled exceptions
    install_crash_handler()
    
    # Clean up old log files (keep last 30 days)
    cleanup_old_logs(days_to_keep=30)
    
    # Log application startup
    logger = logging.getLogger('sarcasm_app')
    logger.info(f'SarcAsM starting. Log file: {get_log_file_path()}')
    
    try:
        application = Application()
        application.init_gui()
    except Exception as e:
        # Log any startup errors
        logger.critical(f'Failed to start application: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    main()
