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


import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QRadioButton, QScrollArea, QVBoxLayout,
    QWidget,
)

from sarcasm.export import Export
from sarcasm_app.control.feature_tiers import (
    MOTION_TIERS, STRUCTURE_TIERS, TierSections, describe, pretty_name,
)

logger = logging.getLogger(__name__)


class ExportPopup(QDialog):
    """Export popup with tiered, grouped feature selection.

    Features are organised into two tiers (Primary / Advanced), each split
    into analysis sections (Z-bands, Myofibrils, ...). Each checkbox carries
    a tooltip with the full description from ``sarcasm.feature_dict``.

    A detail-level toggle controls whether per-frame arrays and raw
    distributions are exported in full or collapsed to temporal mean + std.
    """

    _COLS_PER_SECTION = 2

    def __init__(self, model, control, popup_type: str = 'structure',
                 filename_stem: str = 'export'):
        super().__init__()
        if popup_type not in ('structure', 'motion'):
            raise ValueError(f"popup_type must be 'structure' or 'motion', got {popup_type!r}")

        self._model = model
        self._control = control
        self._popup_type = popup_type
        self._filename_stem = filename_stem
        self._checkboxes: Dict[str, QCheckBox] = {}
        self._advanced_group: Optional[QGroupBox] = None

        self.setWindowTitle(f'Export {popup_type} data')
        self.setMinimumWidth(680)
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        intro = QLabel(
            'Select metrics to export. Primary metrics are pre-checked and '
            'cover the most common use cases. Advanced metrics expose the '
            'full set, including raw per-object distributions.'
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Detail-level toggle
        detail_box = QGroupBox('Detail level')
        detail_layout = QHBoxLayout(detail_box)
        self._rb_summary = QRadioButton('Summary — per-frame stats')
        self._rb_summary.setToolTip(
            'One value per metric per frame. For multi-frame data, one column '
            'per frame; for single-frame data, a single value column. Ragged '
            'per-object distributions are collapsed to per-frame mean. '
            'Works with csv, xlsx, and json.'
        )
        self._rb_full = QRadioButton('Full — raw per-object distributions (JSON only)')
        self._rb_full.setToolTip(
            'Full nested structure including per-object arrays (often thousands '
            'of values per frame). JSON only — does not fit a single table.'
        )
        self._rb_summary.setChecked(True)
        detail_layout.addWidget(self._rb_summary)
        detail_layout.addWidget(self._rb_full)
        detail_layout.addStretch(1)
        root.addWidget(detail_box)

        # Tiered feature groups inside a scroll area
        tiers = STRUCTURE_TIERS if self._popup_type == 'structure' else MOTION_TIERS
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)

        primary_group = self._build_tier_group('Primary metrics', tiers['Primary'],
                                               pre_checked=True, start_collapsed=False)
        inner_layout.addWidget(primary_group)

        self._advanced_group = self._build_tier_group(
            'Advanced metrics (click to expand)', tiers['Advanced'],
            pre_checked=False, start_collapsed=True,
        )
        inner_layout.addWidget(self._advanced_group)
        inner_layout.addStretch(1)

        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Bulk actions
        bulk = QHBoxLayout()
        btn_check_primary = QPushButton('Check all Primary')
        btn_check_primary.clicked.connect(lambda: self._set_all_checked(tiers['Primary'], True))
        btn_check_all = QPushButton('Check all')
        btn_check_all.clicked.connect(lambda: self._set_checked(list(self._checkboxes), True))
        btn_uncheck_all = QPushButton('Uncheck all')
        btn_uncheck_all.clicked.connect(lambda: self._set_checked(list(self._checkboxes), False))
        bulk.addWidget(btn_check_primary)
        bulk.addWidget(btn_check_all)
        bulk.addWidget(btn_uncheck_all)
        bulk.addStretch(1)
        root.addLayout(bulk)

        # Format buttons — each opens a Save-As dialog. Full mode restricts
        # to JSON (tabular formats cannot hold per-object distributions).
        fmt_row = QHBoxLayout()
        self._btn_csv = QPushButton('Export as .csv')
        self._btn_csv.clicked.connect(lambda: self._on_export('csv'))
        self._btn_xlsx = QPushButton('Export as .xlsx')
        self._btn_xlsx.clicked.connect(lambda: self._on_export('xlsx'))
        self._btn_json = QPushButton('Export as .json')
        self._btn_json.clicked.connect(lambda: self._on_export('json'))
        fmt_row.addWidget(self._btn_csv)
        fmt_row.addWidget(self._btn_xlsx)
        fmt_row.addWidget(self._btn_json)
        root.addLayout(fmt_row)

        self._rb_full.toggled.connect(self._on_mode_toggled)
        self._on_mode_toggled(self._rb_full.isChecked())

    def _build_tier_group(self, title: str, sections: Dict[str, List[str]],
                          pre_checked: bool, start_collapsed: bool) -> QGroupBox:
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(not start_collapsed)
        outer = QVBoxLayout(group)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        for section_title, keys in sections.items():
            if not keys:
                continue
            section_box = QGroupBox(section_title)
            grid = QGridLayout(section_box)
            grid.setHorizontalSpacing(16)
            for i, key in enumerate(keys):
                cb = QCheckBox(pretty_name(key, self._popup_type))
                cb.setToolTip(describe(key, self._popup_type))
                cb.setChecked(pre_checked)
                self._checkboxes[key] = cb
                grid.addWidget(cb, i // self._COLS_PER_SECTION, i % self._COLS_PER_SECTION)
            content_layout.addWidget(section_box)

        outer.addWidget(content)
        # Collapse/expand by hiding content when the group is unchecked
        group.toggled.connect(content.setVisible)
        content.setVisible(not start_collapsed)
        return group

    # ------------------------------------------------------------- helpers

    def _selected_keys(self) -> List[str]:
        return [k for k, cb in self._checkboxes.items() if cb.isChecked()]

    def _set_checked(self, keys: List[str], value: bool) -> None:
        for k in keys:
            cb = self._checkboxes.get(k)
            if cb is not None:
                cb.setChecked(value)

    def _set_all_checked(self, sections: Dict[str, List[str]], value: bool) -> None:
        keys: List[str] = []
        for section_keys in sections.values():
            keys.extend(section_keys)
        self._set_checked(keys, value)

    def _on_mode_toggled(self, full_checked: bool) -> None:
        tabular_enabled = not full_checked
        self._btn_csv.setEnabled(tabular_enabled)
        self._btn_xlsx.setEnabled(tabular_enabled)
        tooltip = ('Disabled in Full mode — per-object distributions do not '
                   'fit a table. Use JSON.') if full_checked else ''
        self._btn_csv.setToolTip(tooltip)
        self._btn_xlsx.setToolTip(tooltip)

    def _default_save_dir(self) -> str:
        try:
            if self._popup_type == 'structure':
                return self._model.cell.data_dir
            return self._model.sarcomere.data_dir
        except Exception:
            return str(Path.home())

    # ---------------------------------------------------------- export op

    def _on_export(self, fmt: str) -> None:
        keys = self._selected_keys()
        if not keys:
            QMessageBox.warning(self, 'No metrics selected',
                                'Please select at least one metric to export.')
            return

        default_name = f'{self._popup_type}_{self._filename_stem}.{fmt}'
        default_path = os.path.join(self._default_save_dir(), default_name)
        filter_map = {'csv': 'CSV (*.csv)', 'xlsx': 'Excel (*.xlsx)', 'json': 'JSON (*.json)'}
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save export', default_path, filter_map[fmt]
        )
        if not path:
            return

        try:
            data = self._build_export_dict(keys)
        except Exception as exc:
            logger.exception('Failed to assemble export dictionary')
            QMessageBox.critical(self, 'Export failed',
                                 f'Could not assemble data for export:\n{exc}')
            return

        try:
            Export.write_dict(path, data, fmt, raw=self._rb_full.isChecked())
        except Exception as exc:
            logger.exception('Failed to write export file')
            QMessageBox.critical(self, 'Export failed',
                                 f'Could not write {path}:\n{exc}')
            return

        QMessageBox.information(self, 'Export complete',
                                f'Wrote {len(data)} entries to\n{path}')

    def _build_export_dict(self, keys: List[str]) -> dict:
        if self._popup_type == 'structure':
            return Export.get_structure_dict(sarc_obj=self._model.cell,
                                             structure_keys=keys)
        return Export.get_motion_dict(motion_obj=self._model.sarcomere,
                                      loi_keys=keys)

    # -------------------------------------------------------- public API

    def show_popup(self) -> None:
        self.show()
