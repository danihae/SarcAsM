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
pre-curated list of metrics, and falls back to ``sarcasm.feature_dict`` for
per-key descriptions (shown as tooltips).
"""

from typing import Dict, List

from sarcasm._internal.feature_dict import structure_feature_dict, motion_feature_dict


TierSections = Dict[str, Dict[str, List[str]]]


STRUCTURE_TIERS: TierSections = {
    'Primary': {
        'Cell & coverage': [
            'cell_mask_area', 'cell_mask_area_ratio',
            'sarcomere_area', 'sarcomere_area_ratio',
        ],
        'Sarcomeres': [
            'sarcomere_length_mean', 'sarcomere_length_std',
            'sarcomere_oop',
            'n_vectors',
        ],
        'Z-bands': [
            'n_zbands', 'n_mbands',
            'z_length_mean', 'z_length_std',
            'z_intensity_mean',
            'z_lat_alignment_mean', 'z_lat_alignment_std',
            'z_lat_dist_mean', 'z_lat_dist_std',
        ],
        'Myofibrils': [
            'myof_length_mean', 'myof_length_max',
            'myof_bending_mean',
            'myof_straightness_mean',
        ],
        'Domains': [
            'n_domains',
            'domain_area_mean', 'domain_oop_mean', 'domain_slen_mean',
        ],
    },
    'Advanced': {
        'Cell & coverage': [
            'cell_mask_intensity',
        ],
        'Sarcomeres': [
            'sarcomere_orientation_mean', 'sarcomere_orientation_std',
        ],
        'Z-bands': [
            'z_intensity_std',
            'z_length_max',
            'z_oop',
            'z_mask_area', 'z_mask_area_ratio', 'z_mask_intensity',
            'z_straightness_mean', 'z_straightness_std',
            'z_lat_neighbors_mean', 'z_lat_neighbors_std',
            'z_lat_length_groups_mean', 'z_lat_length_groups_std',
            'z_lat_size_groups_mean', 'z_lat_size_groups_std',
            'z_lat_alignment_groups_mean', 'z_lat_alignment_groups_std',
        ],
        'Myofibrils': [
            'myof_length_std',
            'myof_bending_std',
            'myof_straightness_std',
        ],
        'Domains': [
            'domain_area_std', 'domain_oop_std',
        ],
        'Raw distributions (full detail only)': [
            'sarcomere_length_vectors', 'sarcomere_orientation_vectors',
            'midline_length_vectors',
            'z_length', 'z_intensity', 'z_straightness', 'z_orientation',
            'z_lat_alignment', 'z_lat_dist',
            'z_lat_neighbors', 'z_lat_length_groups', 'z_lat_size_groups',
            'myof_length', 'myof_bending', 'myof_straightness',
            'domain_area', 'domain_oop', 'domain_slen', 'domain_orientation',
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
    },
}


def describe(key: str, kind: str) -> str:
    """Return a human-readable description for ``key`` from the feature dict.

    ``kind`` is ``'structure'`` or ``'motion'``.
    """
    source = structure_feature_dict if kind == 'structure' else motion_feature_dict
    entry = source.get(key)
    if entry is None:
        return key
    name = entry.get('name', key)
    desc = entry.get('description', '').strip()
    return f'{name}\n\n{desc}\n\n(key: {key})' if desc else f'{name}\n(key: {key})'


def pretty_name(key: str, kind: str) -> str:
    source = structure_feature_dict if kind == 'structure' else motion_feature_dict
    entry = source.get(key)
    if entry is None:
        return key
    return entry.get('name', key)


def all_keys(tiers: TierSections) -> List[str]:
    out: List[str] = []
    for tier in tiers.values():
        for section in tier.values():
            out.extend(section)
    return out
