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


"""SarcAsM: sarcomere analysis of cardiomyocytes (public API re-exports)."""

from importlib import import_module as _import_module

from ._version import __version__

# Public names resolve on first access (PEP 562), so ``import sarcasm`` and the
# CLI do not pay for torch, numba and matplotlib until they are needed.
_EXPORTS = {
    'SarcAsMBase': '.core', 'SarcAsM': '.structure', 'Motion': '.motion', 'Utils': '.utils',
    'structure_feature_dict': '.features', 'motion_feature_dict': '.features',
    'describe_key': '.features', 'pretty_name': '.features', 'MOTION_KINDS': '.features',
    'IOUtils': '.io.ioutils', 'Plots': '.plotting.plots', 'PlotUtils': '.plotting.plot_utils',
    'Animations': '.plotting.animation', 'Export': '.export', 'BatchExport': '.export',
    'TypeUtils': '.type_utils', 'TrainingDataGenerator': '.training_data_generation',
}
__all__ = ['__version__', *_EXPORTS]


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(_import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sarcasm' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
