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


"""``sarc.motion``: sarcomere motion of a recording, from tracked sarcomere vectors."""

from sarcasm._internal.namespace import _Namespace
from sarcasm.motion.analysis import AnalysisMixin
from sarcasm.motion.grouping import GroupingMixin
from sarcasm.motion.tracking import TrackingMixin
from sarcasm.motion.view import Motion  # noqa: F401  (per-group view; replaced by Group in phase 2)


class MotionAnalysis(TrackingMixin, GroupingMixin, AnalysisMixin, _Namespace):
    """Sarcomere motion of one recording, reached as ``sarc.motion``.

    The pipeline is track -> group -> analyze: :meth:`track` follows every sarcomere
    vector through the movie, :meth:`group_tracks` combines the tracks into groups of one
    kind (pool, M-band, myofibril, LOI, domain or custom labels), and :meth:`analyze`
    reads each group's contractions. Results are written to ``sarc.data`` under
    ``motion.*`` and parameters under ``params.motion.*``.
    """
