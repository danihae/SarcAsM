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

from sarcasm.feature_dict import structure_feature_dict, motion_feature_dict


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


MOTION_TIERS: TierSections = {
    'Primary': {
        'Beating kinematics': [
            'beating_rate', 'beating_rate_variability',
            'time_contr', 'time_quiet',
        ],
        'Contractile features': [
            'contr_max_avg', 'elong_max_avg',
            'vel_contr_max_avg', 'vel_elong_max_avg',
            'time_to_peak_avg',
            'equ',
        ],
        'Counts & QC': [
            'n_sarcomeres', 'n_contr', 'ratio_nans',
        ],
    },
    'Advanced': {
        'Per-cycle (non-averaged)': [
            'contr_max', 'elong_max',
            'vel_contr_max', 'vel_elong_max',
            'time_to_peak', 'time_to_relax',
            'time_contr_avg', 'time_quiet_avg', 'time_to_relax_avg',
        ],
        'Popping': [
            'popping_rate', 'popping_rate_contr', 'popping_rate_sarcomeres',
            'popping_events', 'popping_dist', 'popping_tau',
            'popping_ks_dist_pvalue', 'popping_ks_dist_statistic',
            'popping_p_dist', 'popping_p_tau',
            'popping_ks_tau_pvalue', 'popping_ks_tau_statistic',
        ],
        'Correlations': [
            'corr_delta_slen', 'corr_vel',
            'corr_delta_slen_serial', 'corr_delta_slen_mutual',
            'corr_vel_serial', 'corr_vel_mutual',
            'ratio_delta_slen_mutual_serial', 'ratio_vel_mutual_serial',
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
