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


from sarcasm_app import Application

import sys
import platform

def check_vcredist():
    """Check for required Microsoft Visual C++ Redistributable DLLs on Windows."""
    if platform.system() == "Windows":
        import os
        import ctypes

        required_dlls = ["vcruntime140.dll", "msvcp140.dll"]
        system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        missing = [dll for dll in required_dlls if not os.path.exists(os.path.join(system32, dll))]

        if missing:
            msg = (
                "This application requires the Microsoft Visual C++ Redistributable to run.\n\n"
                "Missing files:\n" + "\n".join(missing) +
                "\n\nPlease install it from:\nhttps://aka.ms/vs/17/release/vc_redist.x64.exe"
            )
            ctypes.windll.user32.MessageBoxW(0, msg, "Missing Dependency", 0x10)
            sys.exit(1)


def main():
    application = Application()
    application.init_gui()


if __name__ == '__main__':
    check_vcredist()
    main()
