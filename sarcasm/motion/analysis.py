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

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np



if TYPE_CHECKING:
    from sarcasm.motion.view import Motion

logger = logging.getLogger(__name__)

# Import structure modules
from sarcasm.analysis import (
    grouped_motion,
    heterogeneity,
)


class _Stored:
    """Default of a grouping argument of :meth:`AnalysisMixin.analyze`: use the stored grouping."""

    def __repr__(self) -> str:
        return 'stored'


_STORED = _Stored()


class AnalysisMixin:
    """Part of :class:`~sarcasm.motion.MotionAnalysis`; see there."""

    def analyze(
        self,
        by: str,
        *,
        reading: str = 'tracks',
        aggregate: str = 'mean',
        slen_lims: Tuple[float, float] = (1.0, 3.0),
        max_tracks: int = 100,
        max_step_slen: float = 0.3,
        model: Optional[str] = None,
        threshold: Optional[float] = None,
        contr_time_min: float = 0.2,
        buffer_frames: int = 3,
        min_valid_frames: float = 0.5,
        filter_params: Tuple[int, int] = (13, 5),
        reference_frame: Any = _STORED,
        min_coverage: Any = _STORED,
        min_group_size: Any = _STORED,
        max_drift_slen: Any = _STORED,
    ) -> None:
        """Contraction analysis of each group of one grouping kind.

        Reads the contraction state of each group with ContractionNet and measures the
        contraction cycles on the member tracks' aggregated ``slen(t)``. By default
        (``reading='tracks'``) the network reads each member track and the group's state is
        decoded from the per-frame median of their probabilities
        (:func:`grouped_motion.track_contraction_prob`); ``reading='aggregate'`` reads the
        aggregated length once, as up to 1.0.0b4. Each kind is analysed on its own
        and stored beside the others.

        Outputs are written under ``motion.<kind>.*`` (e.g. ``motion.pool.beating_rate``,
        ``motion.mband.slen_avg``, ``motion.mband.contr``) with the same members for
        every grouping, so one set of plotting/export code serves them all:
        the group average ``slen_avg`` and its spread, the ContractionNet cycle metrics
        (including the percentile times ``time_to_50``, ``time_contr_10_90``,
        ``time_relax_50`` and ``time_relax_90``) with the polarity score of each
        group's sarcomere length (``polarity_score``), the per-group heterogeneity of
        the member tracks — serial/mutual correlation of ΔSL and velocity across
        cycles (``corr_*`` / ``ratio_*_mutual_serial``) and the wavelet oscillation
        spectra (``oscill_*``), see :mod:`sarcasm.analysis.heterogeneity`.

        Parameters
        ----------
        by : {'pool', 'mband', 'myofibril', 'loi', 'domain', 'custom'}
            The grouping kind to analyse. Its stored grouping is used; if there is none,
            it is made first with :meth:`group_tracks`' defaults, except ``'custom'``,
            which needs its own ``group_tracks(by='custom', labels=...)`` call.
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
        model, threshold, contr_time_min, buffer_frames, min_valid_frames, filter_params
            ContractionNet engine knobs (see :func:`grouped_motion.run_cycle_engine`).
        reference_frame, min_coverage, min_group_size, max_drift_slen
            Grouping arguments, see :meth:`group_tracks`. Left out, the stored grouping of
            ``by`` is used; given, ``by`` is grouped again with them (the arguments left out
            take :meth:`group_tracks`' defaults) before the analysis. ``min_group_size``
            discards groups with too few
            member tracks (e.g. ``min_group_size=6`` to skip 1–5 sarcomere
            myofibril fragments whose group mean is too noisy to analyse);
            ``max_drift_slen`` drops identity-drifting tracks from the chain
            groupings (myofibril/loi) only.
        """
        if reading not in ('tracks', 'aggregate'):
            raise ValueError(f"reading must be 'tracks' or 'aggregate', got {reading!r}")
        if by not in self._GROUPING_LEVELS:
            raise ValueError(f"Unknown grouping '{by}'. Valid: {self._GROUPING_LEVELS}.")
        grouping_args = {name: value for name, value in dict(
            reference_frame=reference_frame, min_coverage=min_coverage,
            min_group_size=min_group_size, max_drift_slen=max_drift_slen).items() if value is not _STORED}
        grouped = self._gkey(by, 'track_group_id') in self.data
        if by == 'custom':
            if grouping_args:
                raise ValueError("A custom grouping is made by group_tracks(by='custom', labels=...); "
                                 "pass no grouping arguments to analyze('custom').")
            if not grouped:
                raise ValueError("No custom grouping found. Run sarc.motion.group_tracks(by='custom', "
                                 "labels=...) first.")
        elif grouping_args or not grouped:
            self.group_tracks(by=by, **grouping_args)
        if self.metadata.frametime is None:
            raise ValueError('Frame time not defined in metadata. Required for motion analysis.')

        # Refuse to analyze a grouping that no longer matches the current tracks.
        snapshot_ids = np.asarray(self.data.get(self._gkey(by, 'track_ids'), []))
        cur_ids = np.asarray(self.data.get('motion.tracks.ids', []))
        if not np.array_equal(snapshot_ids, cur_ids):
            raise ValueError(f"Tracks changed since the '{by}' grouping was made. Run "
                             f"sarc.motion.group_tracks(by='{by}') again before analyze().")

        kind = by
        n_groups = int(self.data[self._gkey(kind, 'n')])
        gid = np.asarray(self.data[self._gkey(kind, 'track_group_id')])
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
            agg['slen_avg'], frametime=self.metadata.frametime, model_path=model,
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

        params = {
            'grouping_hash': self.data[self._gkey(kind, 'hash')],
            'n_groups': n_groups,
            'reading': reading,
            'max_tracks': max_tracks,
            'max_step_slen': max_step_slen,
            'aggregate': aggregate,
            'slen_lims': list(slen_lims),
            'model': model,
            'threshold': threshold,
            'contr_time_min': contr_time_min,
            'buffer_frames': buffer_frames,
            'min_valid_frames': min_valid_frames,
            'filter_params': filter_params,
        }
        result.update({f'params.motion.analyze.{kind}.{name}': value for name, value in params.items()})
        self.data.update(result)
        logger.info(f"analyze complete ('{kind}', {n_groups} groups).")
        if self.auto_save:
            self.commit()

    def _assert_track_motion_fresh(self, kind: str) -> None:
        """Raise if the analysis of the ``kind`` grouping is missing or stale.

        Guards getters and plots, so a grouping changed after its analysis can never
        silently return the earlier grouping's numbers.
        """
        used = self.data.get(f'params.motion.analyze.{kind}.grouping_hash')
        if used is None:
            raise ValueError(f"No analysis of the '{kind}' grouping found. "
                             f"Run sarc.motion.analyze('{kind}') first.")
        if self.data.get(self._gkey(kind, 'hash')) != used:
            raise ValueError(f"The '{kind}' grouping changed since it was analysed. "
                             f"Run sarc.motion.analyze('{kind}') again.")
        snapshot_ids = np.asarray(self.data.get(self._gkey(kind, 'track_ids'), []))
        if not np.array_equal(snapshot_ids, np.asarray(self.data.get('motion.tracks.ids', []))):
            raise ValueError('Tracks changed since grouping. Run sarc.motion.track -> group_tracks -> analyze again.')

    def get_track_motion(self, group: int, *, kind: str = 'myofibril', analyze: bool = False,
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
        kind : {'myofibril', 'loi'}, optional
            The grouping the group belongs to. Default is 'myofibril'.
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

        if kind not in self._CHAIN_LEVELS:
            raise ValueError(f"get_track_motion needs an ordered grouping, {self._CHAIN_LEVELS}, got '{kind}'.")
        if self._gkey(kind, 'track_group_id') not in self.data:
            raise ValueError(f"No '{kind}' grouping found. Run sarc.motion.group_tracks(by='{kind}') first.")
        n_tracks = int(self.data['motion.tracks.n'])
        gid = np.asarray(self.data[self._gkey(kind, 'track_group_id')]).reshape(-1)
        order = np.asarray(self.data[self._gkey(kind, 'track_group_order')]).reshape(-1)
        members = np.flatnonzero(gid == int(group))
        if members.size == 0:
            raise ValueError(f"No tracks in {kind} group {group} "
                             f'(n_groups={self.data.get(self._gkey(kind, "n"))}).')
        members = members[np.argsort(order[members])]  # head -> tail along the fibre

        slen = np.asarray(self.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)[members]
        # Pass the members' measured positions so each Z-band boundary is placed from
        # its own sarcomere. Accumulating the lengths instead would let a single
        # undefined member blank every boundary below it.
        pos_um = np.asarray(self.data['motion.tracks.positions_um'],
                            dtype=float).reshape(n_tracks, -1, 2)[members]
        # Anchor the chain geometry on the frame the grouping (and hence the
        # head-to-tail order) was built from.
        ref_frame = int(self.data.get(f'params.motion.group_tracks.{kind}.reference_frame', 0))
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
