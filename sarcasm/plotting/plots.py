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

"""All SarcAsM plots in one place: ``Plots.plot_<name>(ax, sarc, ...)``.

The plots are written in three modules by what they show and combined here:
:mod:`~sarcasm.plotting.structure`, :mod:`~sarcasm.plotting.tracks` and :mod:`~sarcasm.plotting.motion`.
"""

from sarcasm.plotting.motion import MotionPlots
from sarcasm.plotting.structure import StructurePlots
from sarcasm.plotting.tracks import TrackPlots


class Plots(StructurePlots, TrackPlots, MotionPlots):
    """All SarcAsM plots: structure, tracks and the motion of the groups."""
