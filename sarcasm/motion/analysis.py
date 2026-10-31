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


"""Contraction analysis of each group of tracks (``sarc.motion.analyze``)."""

"""Main :class:`SarcAsM` object: sarcomere morphology, full-field 2D tracking, and grouped motion analysis."""

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np



if TYPE_CHECKING:
    from sarcasm.motion.view import Motion

logger = logging.getLogger(__name__)

# Import structure modules
from sarcasm.analysis import (
    contraction_analysis,
    grouped_motion,
    heterogeneity,
)


class AnalysisMixin:
    """Part of :class:`~sarcasm.motion.MotionAnalysis`; see there."""

    def analyze(
        self,
        *,
        by: Optional[str] = None,
        reading: str = 'tracks',
        aggregate: str = 'mean',
        slen_lims: Tuple[float, float] = (1.0, 3.0),
        max_tracks: int = 100,
        max_step_slen: float = 0.3,
        model: Optional[str] = None,
        threshold: Optional[float] = None,
        contr_time_min: float = 0.2,
        merge_time_max: Optional[float] = None,
        buffer_frames: int = 3,
        min_valid_frames: float = 0.5,
        filter_params: Tuple[int, int] = (13, 5),
        reference_frame: int = 0,
        min_coverage: Optional[float] = None,
        min_group_size: int = 1,
        max_drift_slen: Optional[float] = 1.0,
    ) -> None:
        """Per-group sarcomere-contraction analysis over the current track grouping.

        Reads the contraction state of each group with ContractionNet and measures the
        contraction cycles on the member tracks' aggregated ``slen(t)``. By default
        (``reading='tracks'``) the network reads each member track and the group's state is
        decoded from the per-frame median of their probabilities
        (:func:`grouped_motion.track_contraction_prob`); ``reading='aggregate'`` reads the
        aggregated length once, as up to 1.0.0b4. Grouping-blind: it analyzes
        whatever :meth:`group_tracks` produced. Pass ``by=`` to run grouping
        inline — the one-call front door for every level but ``'custom'``.

        Outputs are written under ``motion.<kind>.*`` (e.g. ``motion.pool.beating_rate``,
        ``motion.mband.slen``, ``motion.mband.contr``) with the same members for
        every grouping, so one set of plotting/export code serves them all:
        the aggregated ``slen`` statistics, the ContractionNet cycle metrics
        (including the percentile times ``time_to_50``, ``time_contr_10_90``,
        ``time_relax_50`` and ``time_relax_90``) with the polarity score of each
        group's sarcomere length (``polarity_score``), the per-group heterogeneity of
        the member tracks — serial/mutual correlation of ΔSL and velocity across
        cycles (``corr_*`` / ``ratio_*_mutual_serial``) and the wavelet oscillation
        spectra (``oscill_*``), see :mod:`sarcasm.analysis.heterogeneity`.

        Parameters
        ----------
        by : {'pool', 'mband', 'myofibril', 'loi', 'domain'} or None, optional
            If given, run ``group_tracks(by=..., reference_frame=..., min_coverage=...,
            min_group_size=..., max_drift_slen=...)`` first — the one-call front door
            for every grouping level except ``'custom'``, which needs its own
            :meth:`group_tracks` call to pass ``labels``. If None, use the existing
            grouping (whatever :meth:`group_tracks` produced last). Default is None.
        reading : {'tracks', 'aggregate'}, optional
            What ContractionNet reads. ``'tracks'``: every member track whose length stays in
            ``slen_lims``, is measured in at least ``min_valid_frames`` of the frames and never
            jumps by more than ``max_step_slen`` (the ``max_tracks`` least noisy of them); the
            median of their probabilities is decoded. A beat shortens most sarcomeres at once,
            so the median is high when most tracks read one, and a few tracks with large
            excursions cannot make a contraction of a quiet cell. ``'aggregate'``: the group's
            aggregated length (``aggregate``), read once. Default is 'tracks'.
        aggregate : {'mean', 'median'}, optional
            How the member sarcomeres' lengths are combined into the group's
            ``slen(t)`` for the contraction analysis, over the members present in
            each frame. ``'median'`` is robust to a few mis-measured tracks;
            ``'mean'`` matches the 0.5 domain analysis. The per-frame spread
            (``slen_std``, quartiles) is stored either way. Default is 'mean'.
        slen_lims : tuple of float, optional
            Member lengths outside this µm range are ignored in the aggregate, and a track
            with such a length is not read with ``reading='tracks'``. Default is (1.0, 3.0).
        max_tracks : int, optional
            With ``reading='tracks'``, at most this many tracks per group are read.
            Default is 100.
        max_step_slen : float, optional
            With ``reading='tracks'``, a track whose length jumps by more than this (µm)
            between frames is not read. Default is 0.3.
        model, threshold, contr_time_min, merge_time_max, buffer_frames, min_valid_frames, filter_params
            ContractionNet engine knobs (see :func:`grouped_motion.run_cycle_engine`).
            ``merge_time_max`` is deprecated and ignored: gaps between contractions are
            not closed.
        reference_frame, min_coverage, min_group_size, max_drift_slen
            Forwarded to :meth:`group_tracks` when ``by`` is given. Defaults are
            0, None (0.1 for chain kinds / 0.5 otherwise), 1 and 1.0.
            ``min_group_size`` discards groups with too few
            member tracks (e.g. ``min_group_size=6`` to skip 1–5 sarcomere
            myofibril fragments whose group mean is too noisy to analyse);
            ``max_drift_slen`` drops identity-drifting tracks from the chain
            groupings (myofibril/loi) only.
        """
        contraction_analysis._warn_merge_time_max(merge_time_max)
        if reading not in ('tracks', 'aggregate'):
            raise ValueError(f"reading must be 'tracks' or 'aggregate', got {reading!r}")
        if by is not None:
            if by == 'custom':
                raise ValueError(
                    "by='custom' cannot be used as a front door; call "
                    "group_tracks(by='custom', labels=...) then analyze().")
            self.group_tracks(by=by, reference_frame=reference_frame,
                              min_coverage=min_coverage, min_group_size=min_group_size,
                              max_drift_slen=max_drift_slen)
        elif min_group_size > 1:
            logger.warning(
                'min_group_size is only applied when by=... is given (it is a '
                'grouping parameter); the existing grouping is used unchanged. '
                'Call group_tracks(..., min_group_size=%d) to apply it.', min_group_size)

        if 'motion.tracks.group_id' not in self.data:
            raise ValueError('No track grouping found. Run group_tracks(...) first '
                             '(or pass by=... to analyze).')
        if self.metadata.frametime is None:
            raise ValueError('Frame time not defined in metadata. Required for motion analysis.')

        # Refuse to analyze a grouping that no longer matches the current tracks.
        snapshot_ids = np.asarray(self.data.get('motion.groups.track_ids', []))
        cur_ids = np.asarray(self.data.get('motion.tracks.ids', []))
        if not np.array_equal(snapshot_ids, cur_ids):
            raise ValueError('Tracks changed since group_tracks was run. Re-run '
                             'group_tracks(...) before analyze().')

        kind = self.data['motion.groups.kind']
        n_groups = int(self.data['motion.groups.n'])
        gid = np.asarray(self.data['motion.tracks.group_id'])
        tracks_slen = np.asarray(self.data['motion.tracks.slen'], dtype=float)

        agg = grouped_motion.aggregate_group_slen(
            tracks_slen, gid, n_groups, aggregate=aggregate, slen_lims=slen_lims)

        if model is None or model == 'default':
            model = os.path.join(self.model_dir, 'model_ContractionNet.pt')

        logger.info(f"analyze: {n_groups} '{kind}' groups...")
        group_prob = None
        if reading == 'tracks':
            group_prob = grouped_motion.track_contraction_prob(
                tracks_slen, gid, n_groups, frametime=self.metadata.frametime, model_path=model,
                slen_lims=slen_lims, max_tracks=max_tracks, max_step_slen=max_step_slen,
                min_valid_frames=min_valid_frames)
        engine = grouped_motion.run_cycle_engine(
            agg['slen'], frametime=self.metadata.frametime, model_path=model,
            threshold=threshold, contr_time_min=contr_time_min,
            buffer_frames=buffer_frames, min_valid_frames=min_valid_frames,
            filter_params=filter_params,
            group_label={'loi': 'LOI', 'mband': 'M-band'}.get(kind, str(kind).capitalize()),
            # domain group id = mask label - 1, so log 1-based to match the mask;
            # all other kinds log the 0-based group id (matches track_group_id / the API).
            id_offset=1 if kind == 'domain' else 0,
            group_prob=group_prob)
        # tracks ContractionNet read per group (0 for every group with reading='aggregate')
        engine['n_tracks_read'] = (group_prob['n_tracks_read'] if group_prob is not None
                                   else np.zeros(n_groups, dtype=np.int32))

        # Per-group heterogeneity of the member tracks: serial/mutual correlation of
        # dSL and velocity across cycles, and the oscillation spectra.
        het = heterogeneity.analyze_groups(
            tracks_slen, gid, n_groups, engine['contr'], engine['labels_contr'],
            engine['beating_rate'], frametime=self.metadata.frametime,
            filter_params=filter_params, slen_lims=slen_lims)
        # The aggregate, engine and heterogeneity dicts use bare feature names, so
        # the store key is just the path: every kind lands under motion/<kind>/
        # with identical members, with no per-kind special cases.
        result: dict = {f'motion.{kind}.{name}': value
                        for name, value in {**agg, **engine, **het}.items()}

        result.update({
            'motion.groups.analyzed_kind': kind,
            'params.motion.analyze.group_kind': kind,
            'params.motion.analyze.grouping_hash': self.data['motion.groups.hash'],
            'params.motion.analyze.reading': reading,
            'params.motion.analyze.max_tracks': max_tracks,
            'params.motion.analyze.max_step_slen': max_step_slen,
            'params.motion.analyze.aggregate': aggregate,
            'params.motion.analyze.slen_lims': list(slen_lims),
            'params.motion.analyze.model': model,
            'params.motion.analyze.threshold': threshold,
            'params.motion.analyze.contr_time_min': contr_time_min,
            'params.motion.analyze.merge_time_max': None,   # no gap is closed
            'params.motion.analyze.buffer_frames': buffer_frames,
            'params.motion.analyze.min_valid_frames': min_valid_frames,
            'params.motion.analyze.filter_params': filter_params,
            'params.motion.analyze.n_groups': n_groups,
            # Per-kind provenance, so an earlier kind stays validatable after a later
            # grouping overwrote the keys of the grouping currently in effect.
            f'params.motion.analyze.grouping_hash_{kind}': self.data['motion.groups.hash'],
            f'params.motion.analyze.n_groups_{kind}': n_groups,
        })
        kinds = list(self.data.get('motion.groups.analyzed_kinds') or [])
        if kind not in kinds:
            kinds.append(kind)
        result['motion.groups.analyzed_kinds'] = kinds
        self.data.update(result)
        logger.info(f"analyze complete ('{kind}', {n_groups} groups).")
        if self.auto_save:
            self.store_structure_data()

    def _assert_track_motion_fresh(self, kind: Optional[str] = None) -> None:
        """Hard-raise if grouped track-motion results are missing or stale.

        Guards getters/plots so a grouping changed after analysis can never
        silently return the previous grouping's numbers.

        Parameters
        ----------
        kind : str or None, optional
            Grouping kind to validate. ``None`` (default) checks the grouping
            currently in effect. Naming a kind checks that kind's own recorded
            grouping against the current tracks, so a kind analysed before a later
            ``group_tracks`` call stays readable. Default is None.
        """
        suffix = f'_{kind}' if kind else ''
        used = self.data.get(f'params.motion.analyze.grouping_hash{suffix}')
        if used is None:
            which = f" for kind '{kind}'" if kind else ''
            raise ValueError(f'No grouped track-motion analysis found{which}. '
                             'Run analyze() first.')
        cur = self.data.get(self._grouping_key(kind, 'motion.groups.hash'))
        if cur is None or cur != used:
            raise ValueError('Track grouping changed since analyze() was run. '
                             'Re-run analyze() before reading grouped results.')
        snapshot_ids = np.asarray(
            self.data.get(self._grouping_key(kind, 'motion.groups.track_ids'), []))
        cur_ids = np.asarray(self.data.get('motion.tracks.ids', []))
        if not np.array_equal(snapshot_ids, cur_ids):
            raise ValueError('Tracks changed since grouping. Re-run track '
                             '-> group_tracks -> analyze.')

    def get_track_motion(self, group: int, *, analyze: bool = False,
                         persist_loi: bool = False) -> "Motion":
        """Return a :class:`~sarcasm.motion.Motion` view of one myofibril (fibre).

        The fibre's member tracks (ordered head-to-tail by ``track_group_order``)
        are turned into a synthesized LOI (``z_pos`` = the edges of the member
        sarcomeres, from their tracked positions along the fibre) and wrapped in a
        ``Motion`` object, so the full LOI analysis and **every LOI plot**
        (``plot_z_pos``, ``plot_delta_slen``, ``plot_phase_space``,
        ``plot_popping_events``, …) apply to tracker-derived fibres. Requires a ``'myofibril'`` or
        ``'loi'`` grouping (:meth:`group_tracks` / :meth:`analyze`
        with ``by='myofibril'`` or ``by='loi'``).

        Parameters
        ----------
        group : int
            Fibre group id (``0 .. n_groups-1``).
        analyze : bool, optional
            If True, run the standard LOI chain on the view
            (``detect_analyze_contractions`` -> ``get_trajectories`` ->
            ``analyze_trajectories`` -> ``analyze_popping`` ->
            ``analyze_correlations`` -> ``analyze_oscillations``) so it is
            immediately plot-ready. Default is False.
        persist_loi : bool, optional
            If True, persist the synthesized LOI to ``{base}/track_myofibril_{group}/``.
            Default is False (purely in-memory, keeps the dataset directory clean).
        """
        from sarcasm.motion import Motion

        kind = self.data.get('motion.groups.kind')
        if kind not in ('myofibril', 'loi'):
            raise ValueError("get_track_motion requires a 'myofibril' or 'loi' grouping "
                             "(both order tracks along a fibre). "
                             "Run group_tracks(by='myofibril') or group_tracks(by='loi') first.")
        n_tracks = int(self.data['motion.tracks.n'])
        gid = np.asarray(self.data['motion.tracks.group_id']).reshape(-1)
        order = np.asarray(self.data['motion.tracks.group_order']).reshape(-1)
        members = np.flatnonzero(gid == int(group))
        if members.size == 0:
            raise ValueError(f"No tracks in {kind} group {group} "
                             f'(n_groups={self.data.get("motion.groups.n")}).')
        members = members[np.argsort(order[members])]  # head -> tail along the fibre

        slen = np.asarray(self.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)[members]
        # Pass the members' measured positions so each Z-band boundary is placed from
        # its own sarcomere. Accumulating the lengths instead would let a single
        # undefined member blank every boundary below it.
        pos_um = np.asarray(self.data['motion.tracks.positions_um'],
                            dtype=float).reshape(n_tracks, -1, 2)[members]
        # Anchor the chain geometry on the frame the grouping (and hence the
        # head-to-tail order) was built from.
        ref_frame = int(self.data.get('params.motion.group_tracks.reference_frame', 0))
        try:
            ref_idx = self._tracked_frame_index(ref_frame)
        except (ValueError, KeyError):
            ref_idx = 0
        z_pos, slen_f, time = grouped_motion.synthesize_loi_chain(
            slen, self.metadata.frametime, member_pos=pos_um, ref_idx=ref_idx)
        loi_data = {
            'z_pos': z_pos, 'z_pos_raw': z_pos.copy(), 'slen': slen_f, 'time': time,
            'n_sarcomeres': int(members.size),
            'motion.tracks.ids': np.asarray(self.data['motion.tracks.ids']).reshape(-1)[members],
            'synthetic': True,
        }
        m = Motion.from_loi_data(self.file_path, f'track_{kind}_{int(group)}',
                                 loi_data, auto_save=persist_loi,
                                 frametime=self.metadata.frametime)
        if analyze:
            m.detect_analyze_contractions()
            m.get_trajectories()
            m.analyze_trajectories()
            m.analyze_popping()
            m.analyze_correlations()
            m.analyze_oscillations()
        return m
