# -*- coding: utf-8 -*-
# Copyright (c) 2025 University Medical Center Göttingen, Germany.
# All rights reserved.
#
# Patent Pending: DE 10 2024 112 939.5
# SPDX-License-Identifier: LicenseRef-Proprietary-See-LICENSE

"""Feature tier / grouping metadata for the GUI export popup.

Keys are organised into two tiers (Primary / Advanced) and, within each tier,
into analysis sections that mirror the pipeline (Cell, Sarcomeres, Z-bands,
Myofibrils, Domains, etc.). The popup uses this to present a discoverable,
pre-curated list of metrics, and falls back to ``sarcasm.features`` for
per-key descriptions (shown as tooltips).
"""

from typing import Dict, List

from sarcasm.features import describe_key


TierSections = Dict[str, Dict[str, List[str]]]


STRUCTURE_TIERS: TierSections = {
    'Primary': {
        'Cell & coverage': [
            'structure.cell.mask_area', 'structure.cell.mask_area_ratio',
            'structure.sarcomere.area', 'structure.sarcomere.area_ratio',
        ],
        'Sarcomeres': [
            'structure.sarcomere.slen_mean', 'structure.sarcomere.slen_std',
            'structure.sarcomere.oop',
            'structure.sarcomere.n_vectors',
        ],
        'Z-bands': [
            'structure.zbands.n', 'structure.sarcomere.n_mbands',
            'structure.zbands.length_mean', 'structure.zbands.length_std',
            'structure.zbands.intensity_mean',
            'structure.zbands.lat_alignment_mean', 'structure.zbands.lat_alignment_std',
            'structure.zbands.lat_dist_mean', 'structure.zbands.lat_dist_std',
        ],
        'Myofibrils': [
            'structure.myofibril.length_mean', 'structure.myofibril.length_max',
            'structure.myofibril.bending_mean',
            'structure.myofibril.straightness_mean',
        ],
        'Domains': [
            'structure.domain.n',
            'structure.domain.area_mean', 'structure.domain.oop_mean', 'structure.domain.slen_mean',
        ],
    },
    'Advanced': {
        'Cell & coverage': [
            'structure.cell.mask_intensity',
        ],
        'Sarcomeres': [
            'structure.sarcomere.orientation_mean', 'structure.sarcomere.orientation_std',
        ],
        'Z-bands': [
            'structure.zbands.intensity_std',
            'structure.zbands.length_max',
            'structure.zbands.oop',
            'structure.zbands.mask_area', 'structure.zbands.mask_area_ratio', 'structure.zbands.mask_intensity',
            'structure.zbands.straightness_mean', 'structure.zbands.straightness_std',
            'structure.zbands.lat_neighbors_mean', 'structure.zbands.lat_neighbors_std',
            'structure.zbands.lat_length_groups_mean', 'structure.zbands.lat_length_groups_std',
            'structure.zbands.lat_size_groups_mean', 'structure.zbands.lat_size_groups_std',
            'structure.zbands.lat_alignment_groups_mean', 'structure.zbands.lat_alignment_groups_std',
        ],
        'Myofibrils': [
            'structure.myofibril.length_std',
            'structure.myofibril.bending_std',
            'structure.myofibril.straightness_std',
        ],
        'Domains': [
            'structure.domain.area_std', 'structure.domain.oop_std',
        ],
        'Raw distributions (full detail only)': [
            'structure.sarcomere.slen', 'structure.sarcomere.orientation',
            'structure.sarcomere.midline_length',
            'structure.zbands.length', 'structure.zbands.intensity', 'structure.zbands.straightness', 'structure.zbands.orientation',
            'structure.zbands.lat_alignment', 'structure.zbands.lat_dist',
            'structure.zbands.lat_neighbors', 'structure.zbands.lat_length_groups', 'structure.zbands.lat_size_groups',
            'structure.myofibril.length', 'structure.myofibril.bending', 'structure.myofibril.straightness',
            'structure.domain.area', 'structure.domain.oop', 'structure.domain.slen', 'structure.domain.orientation',
        ],
    },
}


# Track-based grouped motion: one value per group. Keys are the feature
# suffixes resolved to ``<kind>_<suffix>`` per grouping kind by Export.
# (group_id / group_member_count / metadata are always exported.)
MOTION_TIERS: TierSections = {
    'Primary': {
        'Beating': [
            'beating_rate', 'beating_rate_variability', 'n_contr', 'n_contr_complete',
        ],
        'Contractile (per group, mean over cycles)': [
            'contr_max', 'elong_max',
            'vel_contr_max', 'vel_elong_max',
            'time_to_peak', 'equ',
        ],
    },
    'Advanced': {
        'Timing': [
            'time_to_relax', 'time_contr',
        ],
        'Heterogeneity (serial / mutual correlation across cycles)': [
            'corr_delta_slen_serial', 'corr_delta_slen_mutual', 'ratio_delta_slen_mutual_serial',
            'corr_vel_serial', 'corr_vel_mutual', 'ratio_vel_mutual_serial',
        ],
        'Oscillations': [
            'oscill_peak_avg', 'oscill_peak_1_single', 'oscill_peak_2_single',
        ],
    },
}


def describe(key: str, kind: str) -> str:
    """Return a human-readable description for ``key`` from the feature dict.

    ``kind`` is ``'structure'`` or ``'motion'``.
    """
    entry = describe_key(key, registry=kind)
    if entry is None:
        return key
    name = entry.get('name', key)
    desc = entry.get('description', '').strip()
    return f'{name}\n\n{desc}\n\n(key: {key})' if desc else f'{name}\n(key: {key})'


def pretty_name(key: str, kind: str) -> str:
    entry = describe_key(key, registry=kind)
    if entry is None:
        return key
    return entry.get('name', key)


def all_keys(tiers: TierSections) -> List[str]:
    out: List[str] = []
    for tier in tiers.values():
        for section in tier.values():
            out.extend(section)
    return out
