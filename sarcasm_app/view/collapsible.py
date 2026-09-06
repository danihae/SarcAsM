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

"""Collapse a checkable QGroupBox to its title bar while it is unchecked."""

from PyQt5.QtWidgets import QGroupBox, QWidget


def make_collapsible(group: QGroupBox) -> None:
    """Hide the group's content while it is unchecked, so an unchecked checkable
    group takes only the height of its title (Qt would merely grey the content out).
    Sub-menus that most users never need start unchecked in the ``.ui``."""
    def apply(checked: bool):
        for child in group.findChildren(QWidget):
            if child.parent() is group:
                child.setVisible(checked)
    group.toggled.connect(apply)
    apply(group.isChecked())
