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
        # hidden rows still leave the frame and the layout's row spacing standing:
        # cap the box at its title line while collapsed
        if checked:
            group.setMaximumHeight(16777215)
            if group.layout() is not None:
                group.layout().setContentsMargins(*_MARGINS.get(id(group), (11, 11, 11, 11)))
        else:
            if group.layout() is not None:
                _MARGINS.setdefault(id(group), group.layout().getContentsMargins())
                group.layout().setContentsMargins(0, 0, 0, 0)
            group.setMaximumHeight(group.fontMetrics().height() + 8)
    group.toggled.connect(apply)
    apply(group.isChecked())


_MARGINS: dict = {}
