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


"""Grouping of tracks (pool, M-band, myofibril, LOI, domain, custom) and LOI detection."""

"""Main :class:`SarcAsM` object: sarcomere morphology, full-field 2D tracking, and grouped motion analysis."""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np



logger = logging.getLogger(__name__)

# Import structure modules
from sarcasm.analysis import (
    myofibril_analysis,
    domain_clustering,
    loi_detection,
)


class GroupingMixin:
    """Part of :class:`~sarcasm.motion.MotionAnalysis`; see there."""

    #: Artifacts of the grouping currently in effect -> the leaf they are mirrored
    #: under at ``motion.groups.<kind>.<leaf>`` (``motion.groups.kind`` is not mirrored).
    _GROUPING_MIRRORS = {
        'motion.tracks.group_id': 'track_group_id',
        'motion.tracks.group_order': 'track_group_order',
        'motion.groups.n': 'n',
        'motion.groups.member_counts': 'member_counts',
        'motion.groups.n_vectors_total': 'n_vectors_total',
        'motion.groups.n_vectors_in_long_tracks': 'n_vectors_in_long_tracks',
        'motion.groups.track_ids': 'track_ids',
        'motion.groups.hash': 'hash',
    }

    @classmethod
    def _grouping_key(cls, kind: Optional[str], current_key: str) -> str:
        """Key holding ``current_key``'s value for ``kind``.

        ``kind=None`` means the grouping currently in effect, which lives in the
        unqualified key; naming a kind selects that kind's own mirror.
        """
        if kind is None:
            return current_key
        return f'motion.groups.{kind}.{cls._GROUPING_MIRRORS[current_key]}'

    def _grouped_kinds(self) -> List[str]:
        """Kinds whose grouping is mirrored in this store, in grouping-level order."""
        return [lvl for lvl in self._GROUPING_LEVELS
                if self._grouping_key(lvl, 'motion.tracks.group_id') in self.data]

    def _invalidate_groupings(self) -> None:
        """Drop every grouping artifact, current and per-kind.

        A grouping labels *tracks*, so re-tracking makes it meaningless. Leaving a
        mirror behind would let a stale grouping be read back afterwards.
        """
        stale = list(self._GROUPING_MIRRORS)
        stale += ['motion.groups.kind', 'motion.groups.analyzed_kind',
                  'motion.groups.analyzed_kinds',
                  'params.motion.analyze.grouping_hash',
                  'params.motion.analyze.n_groups']
        for lvl in self._GROUPING_LEVELS:
            stale.extend(self._grouping_key(lvl, key) for key in self._GROUPING_MIRRORS)
            stale.append(f'params.motion.analyze.grouping_hash_{lvl}')
            stale.append(f'params.motion.analyze.n_groups_{lvl}')
        for key in stale:
            if key in self.data:
                del self.data[key]

    _GROUPING_LEVELS = ('pool', 'mband', 'myofibril', 'loi', 'domain', 'custom')

    # Groupings that order their members head-to-tail into a 1D fibre thread.
    # They represent geometry, not just a set, so they keep partially-tracked
    # sarcomeres (a dropped member is a hole in the fibre, not removed noise).
    _CHAIN_LEVELS = ('myofibril', 'loi')

    _CHAIN_MIN_COVERAGE = 0.1

    def _grouping_hash(self, by: str, reference_frame: int, min_coverage: float,
                       min_group_size: int = 1,
                       max_drift_slen: Optional[float] = None,
                       labels: Optional[np.ndarray] = None) -> str:
        """Fingerprint of (current track identity + grouping recipe)."""
        h = hashlib.sha1()
        track_ids = np.ascontiguousarray(np.asarray(self.data.get('motion.tracks.ids', []))).astype(np.int64)
        h.update(track_ids.tobytes())
        h.update(repr((by, int(reference_frame), float(min_coverage), int(min_group_size),
                       None if max_drift_slen is None else float(max_drift_slen))).encode())
        if labels is not None:
            h.update(np.ascontiguousarray(np.asarray(labels)).astype(np.int64).tobytes())
        return h.hexdigest()

    @staticmethod
    def _partition_fibre_chains(fibers: List[List[int]], gid: np.ndarray,
                                order: np.ndarray,
                                group_ids: Optional[List[int]] = None) -> None:
        """Greedy longest-first partition of ordered track chains into groups.

        Each track joins only one fibre; ``gid``/``order`` are written in place
        (group id and rank along the fibre). Shared by 'myofibril' and any other
        line-chain grouping that resolves to ordered track index lists.

        Parameters
        ----------
        fibers : list of list of int
            Ordered track-index chains, head to tail along each fibre.
        gid, order : np.ndarray
            Per-track group id / rank-along-fibre, written in place.
        group_ids : list of int or None, optional
            Group id to assign to each chain, parallel to ``fibers``. ``None``
            (default) numbers the surviving fibres sequentially. Pass explicit ids
            to preserve a fixed label space (e.g. LOI line *i* -> group *i*), which
            keeps the groups aligned with the lines they came from even when a
            chain is dropped for having too few unclaimed tracks.
        """
        idx = sorted(range(len(fibers)), key=lambda i: len(fibers[i]), reverse=True)
        claimed: set = set()
        fib_id = 0
        for i in idx:
            chain = fibers[i]
            unclaimed = [tr for tr in chain if tr not in claimed]
            if len(unclaimed) < 2:
                continue
            this_id = fib_id if group_ids is None else int(group_ids[i])
            for rank, tr in enumerate(unclaimed):
                gid[tr] = this_id
                order[tr] = rank
                claimed.add(tr)
            fib_id += 1

    @staticmethod
    def _assign_points_to_polylines(points: np.ndarray, lines: List[np.ndarray],
                                    max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
        """Assign each point to the nearest polyline within ``max_dist``.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 2)`` array of ``(row, col)`` positions.
        lines : list of np.ndarray
            List of ``(Li, 2)`` polyline vertex arrays (same coordinate space).
        max_dist : float
            Maximum point-to-polyline distance to accept an assignment.

        Returns
        -------
        line_id : np.ndarray
            ``(N,)`` int array; index of the nearest polyline, or -1 if none is
            within ``max_dist``.
        arclen : np.ndarray
            ``(N,)`` float array; arc length of the closest point along that
            polyline (NaN if unassigned), used to order points along the line.
        dist : np.ndarray
            ``(N,)`` float array; perpendicular distance to the assigned polyline
            (inf if unassigned), used to pick the on-line track when several
            candidates share an arc position.
        """
        n = len(points)
        line_id = np.full(n, -1, dtype=np.int64)
        arclen = np.full(n, np.nan, dtype=float)
        best = np.full(n, np.inf, dtype=float)
        pts = np.asarray(points, dtype=float)
        for li, L in enumerate(lines):
            L = np.asarray(L, dtype=float)
            if L.shape[0] < 2:
                continue
            a = L[:-1]                       # (S, 2) segment starts
            seg = L[1:] - a                  # (S, 2) segment vectors
            seg_len2 = (seg * seg).sum(1)    # (S,)
            seg_len = np.sqrt(seg_len2)
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # (S+1,)
            safe = np.where(seg_len2 > 0, seg_len2, 1.0)
            for i in range(n):
                p = pts[i]
                t = ((p - a) * seg).sum(1) / safe
                t = np.clip(t, 0.0, 1.0)
                proj = a + t[:, None] * seg
                d = np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])
                j = int(np.argmin(d))
                if d[j] < best[i] and d[j] <= max_dist:
                    best[i] = d[j]
                    line_id[i] = li
                    arclen[i] = cum[j] + t[j] * seg_len[j]
        return line_id, arclen, best

    def group_tracks(
        self,
        by: str = 'pool',
        *,
        reference_frame: int = 0,
        min_coverage: Optional[float] = None,
        min_group_size: int = 1,
        max_drift_slen: Optional[float] = 1.0,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Assign each track to a group (cheap, inspectable, freely re-callable).

        Writes only label arrays — runs no contraction analysis. Prerequisite:
        :meth:`track`.

        Parameters
        ----------
        by : {'pool', 'mband', 'myofibril', 'loi', 'domain', 'custom'}, optional
            Grouping level. ``'pool'`` = all eligible tracks in one group;
            ``'mband'`` = group tracks by the M-band (midline) they were matched to at
            ``reference_frame`` (laterally-registered sarcomeres); ``'myofibril'`` =
            order tracks into fibre chains from the ``reference_frame`` myofibril
            lines (requires :meth:`~sarcasm.structure.StructureAnalysis.analyze_myofibrils`; sets ``track_group_order``
            = rank along the fibre, enabling the LOI-style per-fibre analysis via
            :meth:`get_track_motion`); ``'loi'`` = like ``'myofibril'`` but uses
            the **curated** lines from :meth:`detect_lois` instead of all myofibril
            lines (the same quality-filtered/clustered subset you would place manual
            LOIs on — far fewer, cleaner fibre groups; run
            ``detect_lois(frame=reference_frame)`` first);
            ``'domain'`` = assign each track to the
            Leiden domain its ``reference_frame`` position falls in (requires
            :meth:`~sarcasm.structure.StructureAnalysis.analyze_domains`; preserves the domain mask's label
            space so the existing domain plots stay aligned); ``'custom'`` = use
            ``labels``. Default is 'pool'.
        reference_frame : int, optional
            Movie-frame whose geometry defines the grouping (must be a tracked
            frame). Used by ``'mband'`` / ``'myofibril'`` / ``'loi'`` / ``'domain'``.
            For ``'loi'`` this should match the ``frame`` passed to
            :meth:`detect_lois`. Default is 0.
        min_coverage : float or None, optional
            Tracks observed in fewer than this fraction of frames are dropped
            (group id ``-1``). ``None`` (default) picks the floor from the
            grouping: 0.1 for the chain kinds ``'myofibril'`` / ``'loi'``, 0.5
            otherwise. A chain represents the fibre's geometry, so dropping a
            partially-tracked sarcomere punches a hole into it rather than
            removing noise; for a group *mean* the strict floor is right.
        min_group_size : int, optional
            Drop groups with fewer than this many member tracks — their members
            become unassigned (group id ``-1``), like a failed ``min_coverage``.
            Useful to discard the many 1–2 sarcomere fragments a fine grouping
            produces (e.g. ``min_group_size=6`` for ``by='myofibril'``), whose
            group-mean traces are too noisy to analyse. For ``'pool'``, ``'mband'``,
            ``'myofibril'`` and ``'custom'`` the surviving groups are renumbered
            contiguously; ``'domain'`` and ``'loi'`` keep their fixed label space
            (the emptied labels simply end up with a member count of 0) so the
            domain/LOI plots stay aligned. Default is 1 (no filtering).
        max_drift_slen : float or None, optional
            Drop tracks whose ``track_drift_um`` exceeds this many sarcomere
            lengths — they have wandered away from the coherent motion of their
            neighbours and most likely changed identity. Applied **only** to the
            chain groupings ``'myofibril'`` and ``'loi'``, where such a track
            breaks the head-to-tail order and can place the same physical
            sarcomere in one chain twice; a drifting track still contributes a
            valid length to a pooled/mband/domain group, so those are left
            untouched. ``None`` disables the filter. Default is 1.0.
        labels : np.ndarray or None, optional
            Required for ``by='custom'``: integer label per track, row-aligned to
            ``track_ids``. Negative labels drop the track. Default is None.

        Notes
        -----
        Stores ``motion.tracks.group_id`` / ``motion.tracks.group_order``
        ``(n_tracks,)``, ``motion.groups.kind``, ``motion.groups.n``,
        ``motion.groups.member_counts``, ``motion.groups.track_ids`` and
        ``motion.groups.hash`` into ``self.data``, and mirrors all of them (bar the
        kind's own name) under ``motion.groups.<by>.*`` so a later grouping of the
        same tracking cannot overwrite this one. The hash changes whenever the tracks
        or the recipe change; a stale :meth:`analyze` result is then
        detected on read.
        """
        self._require_tracks()
        if by not in self._GROUPING_LEVELS:
            raise ValueError(f"Unknown grouping '{by}'. Valid: {self._GROUPING_LEVELS}.")
        reference_frame = self._canonical_frame(reference_frame, name='reference_frame')

        # A chain must span the whole fibre, so it keeps partially-tracked
        # sarcomeres that a group *mean* would rather discard (see the parameter
        # docs). Explicit values are always honoured.
        if min_coverage is None:
            min_coverage = self._CHAIN_MIN_COVERAGE if by in self._CHAIN_LEVELS else 0.5

        n_tracks = int(self.data.get('motion.tracks.n', 0))
        track_ids = np.asarray(self.data.get('motion.tracks.ids', np.arange(n_tracks)))

        if n_tracks == 0:
            gid = np.zeros(0, np.int32)
            order = np.zeros(0, np.int32)
            n_groups = 0
            counts = np.zeros(0, np.int64)
            n_vectors_total = 0
            n_vectors_long = 0
        else:
            slen = np.asarray(self.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)
            T = slen.shape[1]
            # Branch explicitly: .get(default=...) evaluates its default eagerly, so it
            # would pull the whole (n_tracks, T) mask out of the store on every call
            # even though 'motion.tracks.n_frames' is always present.
            if 'motion.tracks.n_frames' in self.data:
                length = np.asarray(self.data['motion.tracks.n_frames'])
            else:
                length = np.asarray(self.data['motion.tracks.observed']).reshape(n_tracks, T).sum(axis=1)
            coverage = length / float(T) if T else np.zeros(n_tracks)
            eligible = coverage >= min_coverage

            # a drifted track would corrupt a head-to-tail chain (duplicate sarcomere);
            # unordered groupings keep it
            if max_drift_slen is not None and by in self._CHAIN_LEVELS:
                drift = self.data.get('motion.tracks.drift_um')
                if drift is not None:
                    drift = np.asarray(drift, dtype=float).reshape(-1)
                    med_slen = float(np.nanmedian(slen[eligible])) if eligible.any() else np.nan
                    if drift.shape[0] == n_tracks and np.isfinite(med_slen):
                        too_far = np.isfinite(drift) & (drift > max_drift_slen * med_slen)
                        n_drop = int((too_far & eligible).sum())
                        if n_drop:
                            logger.info(
                                f'max_drift_slen={max_drift_slen}: dropped {n_drop} track(s) '
                                f'drifting >{max_drift_slen * med_slen:.2f} µm from their '
                                f'neighbours before building the {by} chains.')
                        eligible &= ~too_far

            gid = np.full(n_tracks, -1, dtype=np.int32)
            order = np.zeros(n_tracks, dtype=np.int32)
            fixed_n_groups = None  # set by 'domain'/'loi' to preserve a fixed label space

            if by == 'pool':
                gid[eligible] = 0

            elif by == 'mband':
                rf_idx = self._tracked_frame_index(reference_frame)
                mid_ref = np.asarray(self.data['motion.tracks.midline_id']).reshape(n_tracks, T)[:, rf_idx]
                valid = eligible & (mid_ref >= 0)
                uniq = np.unique(mid_ref[valid])
                remap = {int(m): i for i, m in enumerate(uniq)}
                for i in np.flatnonzero(valid):
                    gid[i] = remap[int(mid_ref[i])]

            elif by == 'domain':
                rf_idx = self._tracked_frame_index(reference_frame)
                mask, n_dom = self._domain_mask_for(reference_frame)
                pos_um_ref = np.asarray(self.data['motion.tracks.positions_um'],
                                        dtype=float).reshape(n_tracks, T, 2)[:, rf_idx]
                use = eligible & np.isfinite(pos_um_ref).all(axis=1)
                idx_use = np.flatnonzero(use)
                if idx_use.size:
                    labels_use = domain_clustering.assign_vectors_to_domains(
                        pos_um_ref[idx_use], mask, self.metadata.pixelsize)
                    hit = labels_use > 0
                    gid[idx_use[hit]] = labels_use[hit] - 1  # mask label j -> group j-1
                # Preserve the full mask label space (0..n_dom-1) so the motion
                # rows stay aligned with plot_sarcomere_domains' labels even if
                # some domains end up with no assigned tracks.
                fixed_n_groups = int(n_dom)

            elif by == 'myofibril':
                rf_idx = self._tracked_frame_index(reference_frame)
                myof_lines = self.data.get('structure.myofibril.lines')
                if myof_lines is None or myof_lines[reference_frame] is None:
                    raise ValueError(
                        f'Myofibril lines not available for reference_frame {reference_frame}. '
                        'Run sarc.structure.analyze_myofibrils first.')
                lines = myof_lines[reference_frame]  # list of ordered vector-index arrays
                det_ref = np.asarray(self.data['motion.tracks.detection_id']).reshape(n_tracks, T)[:, rf_idx]
                # reverse map: ref-frame detection (vector) index -> first eligible track
                det_to_track: Dict[int, int] = {}
                for tr in np.flatnonzero(eligible):
                    d = int(det_ref[tr])
                    if d >= 0:
                        det_to_track.setdefault(d, int(tr))
                # ordered track chains (skip vectors with no matching track)
                fibers = []
                for line in lines:
                    chain = [det_to_track[int(v)] for v in np.asarray(line).ravel()
                             if int(v) in det_to_track]
                    if len(chain) >= 2:
                        fibers.append(chain)
                self._partition_fibre_chains(fibers, gid, order)

            elif by == 'loi':
                # an LOI is a 1-D thread like a myofibril: prefer the detection-index
                # chain it was selected from; lines without one (fitted / hand-drawn)
                # take the geometric path, thinned to one track per sarcomere step
                rf_idx = self._tracked_frame_index(reference_frame)
                loi_data = self.data.get('motion.loi.data')
                loi_lines = None if loi_data is None else loi_data.get('loi_lines')
                if loi_lines is None or len(loi_lines) == 0:
                    raise ValueError(
                        'LOI lines not available. Run detect_lois first, e.g. '
                        'detect_lois(frame=reference_frame).')
                index_lines = None if loi_data is None else loi_data.get('loi_index_lines')
                lines_px = [np.asarray(l, dtype=float).reshape(-1, 2) for l in loi_lines]

                med_slen_um = float(np.nanmedian(slen[eligible])) if eligible.any() else np.nan
                px = self.metadata.pixelsize
                med_slen_px = med_slen_um / px if (np.isfinite(med_slen_um) and px) else 30.0

                if index_lines is not None and len(index_lines) == len(lines_px):
                    # --- chain path: same construction as 'myofibril' -------------
                    det_ref = np.asarray(self.data['motion.tracks.detection_id']).reshape(n_tracks, T)[:, rf_idx]
                    det_to_track: Dict[int, int] = {}
                    for tr in np.flatnonzero(eligible):
                        d = int(det_ref[tr])
                        if d >= 0:
                            det_to_track.setdefault(d, int(tr))
                    fibers, line_ids = [], []
                    for li, chain_idx in enumerate(index_lines):
                        chain = [det_to_track[int(v)] for v in np.asarray(chain_idx).ravel()
                                 if int(v) in det_to_track]
                        if len(chain) >= 2:
                            fibers.append(chain)
                            line_ids.append(li)
                    # explicit ids keep the LOI label space (line i -> group i)
                    self._partition_fibre_chains(fibers, gid, order, group_ids=line_ids)
                else:
                    # --- geometric fallback: hand-drawn / fitted lines ------------
                    pos_px_ref = np.asarray(self.data['motion.tracks.positions_px'],
                                            dtype=float).reshape(n_tracks, T, 2)[:, rf_idx]
                    use = eligible & np.isfinite(pos_px_ref).all(axis=1)
                    idx_use = np.flatnonzero(use)
                    if idx_use.size:
                        # candidates within half a sarcomere of the line, one per
                        # sarcomere-long bin (else the chain becomes a ribbon)
                        line_id, arclen, dist = self._assign_points_to_polylines(
                            pos_px_ref[idx_use], lines_px, 0.5 * med_slen_px)
                        step = max(med_slen_px, 1e-6)
                        bins = np.floor(arclen / step)
                        for li in range(len(lines_px)):
                            sel = np.flatnonzero(line_id == li)
                            # one track per sarcomere-long bin: sort by (bin, distance)
                            # and take the first of each bin — the one on the line.
                            o = sel[np.lexsort((dist[sel], bins[sel]))]
                            keep = o[np.unique(bins[o], return_index=True)[1]]
                            if keep.size < 2:
                                continue
                            gid[idx_use[keep]] = li
                            order[idx_use[keep]] = np.arange(keep.size)
                # preserve the LOI line label space (line i -> group i)
                fixed_n_groups = len(lines_px)

            elif by == 'custom':
                if labels is None:
                    raise ValueError("by='custom' requires `labels` (one integer per track).")
                labels = np.asarray(labels).reshape(-1)
                if labels.shape[0] != n_tracks:
                    raise ValueError(
                        f'labels has length {labels.shape[0]}, expected n_tracks={n_tracks} '
                        '(row-aligned to track_ids).')
                valid = eligible & (labels >= 0)
                uniq = np.unique(labels[valid])
                remap = {int(m): i for i, m in enumerate(uniq)}
                for i in np.flatnonzero(valid):
                    gid[i] = remap[int(labels[i])]

            # drop under-sized groups (members become unassigned); fixed label
            # spaces (domain/loi) keep their numbering, the others are renumbered
            if min_group_size > 1 and (gid >= 0).any():
                raw_counts = np.bincount(gid[gid >= 0])
                too_small = np.flatnonzero(raw_counts < min_group_size)
                if too_small.size:
                    gid[np.isin(gid, too_small)] = -1
                    n_dropped = int(too_small.size)
                    if fixed_n_groups is None and (gid >= 0).any():
                        keep = np.unique(gid[gid >= 0])
                        remap = np.full(int(keep.max()) + 1, -1, dtype=np.int32)
                        remap[keep] = np.arange(keep.size, dtype=np.int32)
                        gid[gid >= 0] = remap[gid[gid >= 0]]
                    logger.info(
                        f'min_group_size={min_group_size}: dropped {n_dropped} group(s) '
                        f'with fewer than {min_group_size} tracks.')

            if fixed_n_groups is not None:
                n_groups = fixed_n_groups
            else:
                n_groups = int(gid.max()) + 1 if (gid >= 0).any() else 0
            counts = np.bincount(gid[gid >= 0], minlength=n_groups).astype(np.int64)

            # QC: observation-weighted share of the detected signal captured by
            # long-lived tracks (the trajectory count is dominated by fragments)
            n_vectors_total = self._n_sarcomere_vectors_tracked()
            n_vectors_long = int(length[eligible].sum())

        grouping_hash = self._grouping_hash(
            by, reference_frame, min_coverage, min_group_size=min_group_size,
            max_drift_slen=max_drift_slen,
            labels=labels if by == 'custom' else None)

        current = {
            'motion.tracks.group_id': gid,
            'motion.tracks.group_order': order,
            'motion.groups.n': n_groups,
            'motion.groups.member_counts': counts,
            'motion.groups.n_vectors_total': int(n_vectors_total),
            'motion.groups.n_vectors_in_long_tracks': int(n_vectors_long),
            'motion.groups.track_ids': np.asarray(track_ids).copy(),
            'motion.groups.hash': grouping_hash,
        }
        # The keys above mean "the grouping currently in effect" and are overwritten by
        # the next group_tracks call; the motion.groups.<by>.* mirrors persist, so
        # several groupings of one tracking stay independently readable.
        per_kind = {self._grouping_key(by, key): value for key, value in current.items()}
        self.data.update({
            **current,
            **per_kind,
            'motion.groups.kind': by,
            'params.motion.group_tracks.by': by,
            'params.motion.group_tracks.reference_frame': reference_frame,
            'params.motion.group_tracks.min_coverage': min_coverage,
            'params.motion.group_tracks.min_group_size': min_group_size,
            'params.motion.group_tracks.max_drift_slen': max_drift_slen,
        })
        logger.info(
            f"group_tracks(by='{by}'): {n_groups} groups, "
            f"{int((gid >= 0).sum())}/{n_tracks} tracks assigned.")
        if self.auto_save:
            self.store_structure_data()

    def _domain_mask_for(self, reference_frame: int) -> Tuple[np.ndarray, int]:
        """Integer-labelled domain mask + n_domains for a reference frame.

        Uses the stored ``domain_mask`` when available, else regenerates it from
        the stored ``domains`` exactly as :meth:`~sarcasm.structure.StructureAnalysis.analyze_domains` /
        :func:`domain_clustering.analyze_domains` do, so the label space (1..n)
        matches what :meth:`Plots.plot_sarcomere_domains` draws.
        """
        if 'structure.domain.members' not in self.data:
            raise ValueError("Sarcomere domains not analyzed. Run sarc.structure.analyze_domains first.")
        domain_frames = self.data.get('params.structure.analyze_domains.frames', [])
        if reference_frame not in list(domain_frames):
            raise ValueError(
                f'reference_frame {reference_frame} was not analyzed for domains. '
                f'Available domain frames: {list(domain_frames)}.')

        if 'structure.domain.mask' in self.data and self.data['structure.domain.mask'][reference_frame] is not None:
            mask = self.data['structure.domain.mask'][reference_frame]
            if hasattr(mask, 'toarray'):
                mask = mask.toarray()
            mask = np.asarray(mask)
        else:
            domains_ref = self.data['structure.domain.members'][reference_frame]
            pos_ref = np.asarray(self.data['structure.sarcomere.pos'][reference_frame])
            ori_ref = np.asarray(self.data['structure.sarcomere.orientation'][reference_frame])
            len_ref = np.asarray(self.data['structure.sarcomere.slen'][reference_frame])
            dilation_radius = self.data.get('params.structure.analyze_domains.dilation_radius', 0.3)
            area_min = self.data.get('params.structure.analyze_domains.area_min', 20.0)
            mask, *_ = domain_clustering.analyze_domains(
                domains_ref, pos_ref, ori_ref, len_ref,
                size=self.metadata.size, pixelsize=self.metadata.pixelsize,
                dilation_radius=dilation_radius, area_min=area_min)
            mask = np.asarray(mask)
        n_domains = int(self.data['structure.domain.n'][reference_frame])
        return mask, n_domains

    def detect_lois(self, frame: int = 0, 
                    n_lois: int = 4, 
                    ratio_seeds: float = 0.1, 
                    persistence: int = 8,
                    threshold_distance: float = 0.3,
                    mode: str = 'longest_in_cluster', 
                    random_seed: Optional[int] = None,
                    number_lims: Tuple[int, int] = (10, 50), 
                    length_lims: Tuple[float, float] = (0, 200),
                    sarcomere_mean_length_lims: Tuple[float, float] = (1, 3),
                    sarcomere_std_length_lims: Tuple[float, float] = (0, 1),
                    midline_mean_length_lims: Tuple[float, float] = (0, 50),
                    midline_std_length_lims: Tuple[float, float] = (0, 50),
                    midline_min_length_lims: Tuple[float, float] = (0, 50), 
                    distance_threshold_lois: float = 2.5,
                    linkage: str = 'single',
                    ) -> None:
        """
        Detect lines of interest (LOIs) — curated sarcomere/myofibril lines.

        Grows candidate lines from seed vectors, filters them by geometric and
        morphological criteria, clusters them, and selects one line per cluster
        (or fitted / random lines). The selected lines are stored as
        ``self.data['motion.loi.data']['loi_lines']`` and consumed by :meth:`group_tracks`
        with ``by='loi'`` to group full-field tracks along the curated fibres.

        Parameters
        ----------
        frame : int, optional
            Index of the frame to analyze. Default is 0.
        n_lois : int, optional
            Number of LOIs to select. Default is 4.
        ratio_seeds : float, optional
            Ratio of sarcomere vectors used as seed vectors for LOI growth.
            Default is 0.1.
        persistence : int, optional
            Number of preceding line points whose mean length and orientation
            predict the next step; larger values give straighter LOIs. Default is 8.
        threshold_distance : float, optional
            Search radius (µm) around the predicted position for the next
            sarcomere vector. Sarcomere vectors are dense M-band midline points,
            so a larger radius lets each step snap sideways along the M-band
            and makes LOIs wavy. Default is 0.3.
        mode : {'fit_straight_line', 'longest_in_cluster', 'random_from_cluster', 'random_line'}, optional
            Mode for selecting LOIs from identified clusters:
            'fit_straight_line' fits a straight line to all midline points in the
            cluster; 'longest_in_cluster' selects the longest (possibly curved)
            line of each of the ``n_lois`` largest clusters (most lines);
            'random_from_cluster' selects a random line of each of the ``n_lois``
            largest clusters; 'random_line' selects random lines passing the filters.
            Default is 'longest_in_cluster'.
        random_seed : int or None, optional
            Random seed for selecting random starting vectors (reproducibility).
            If None, results differ each run. Default is None.
        number_lims : tuple of int, optional
            Limits (min, max) for the number of sarcomeres within an LOI.
            Default is (10, 50).
        length_lims : tuple of float, optional
            Length limits (min, max) for LOIs in µm. Default is (0, 200).
        sarcomere_mean_length_lims : tuple of float, optional
            Limits (min, max) for mean sarcomere length within an LOI.
            Default is (1, 3).
        sarcomere_std_length_lims : tuple of float, optional
            Limits (min, max) for the std of sarcomere lengths within an LOI.
            Default is (0, 1).
        midline_mean_length_lims : tuple of float, optional
            Limits (min, max) for mean midline length within an LOI.
            Default is (0, 50).
        midline_std_length_lims : tuple of float, optional
            Limits (min, max) for the std of midline length within an LOI.
            Default is (0, 50).
        midline_min_length_lims : tuple of float, optional
            Limits (min, max) for minimum midline length within an LOI.
            Default is (0, 50).
        distance_threshold_lois : float, optional
            Hausdorff distance (µm) between lines above which clusters are not
            merged. Default is 2.5.
        linkage : {'complete', 'average', 'single'}, optional
            Linkage criterion for clustering. Default is 'single'.
        """
        if 'structure.sarcomere.pos' not in self.data:
            raise ValueError('Sarcomere length and orientation not yet analyzed. Run sarc.structure.analyze_vectors first.')
        modes = ('fit_straight_line', 'longest_in_cluster', 'random_from_cluster', 'random_line')
        if mode not in modes:
            raise ValueError(f'mode must be one of {modes}, got {mode!r}.')

        frame = self._canonical_frame(frame)
        if self.metadata.n_stack == 1:
            raise ValueError('LOI detection not possible in single images. '
                             'Sarcomere motion tracking is only possible in high-speed movies; (t, x, y) stacks.')

        pos_px = self.data['structure.sarcomere.pos_px'][frame]
        grown = myofibril_analysis.line_growth(
            points_t=pos_px, sarcomere_length_vectors_t=self.data['structure.sarcomere.slen'][frame],
            sarcomere_orientation_vectors_t=self.data['structure.sarcomere.orientation'][frame],
            midline_length_vectors_t=self.data['structure.sarcomere.midline_length'][frame],
            pixelsize=self.metadata.pixelsize, ratio_seeds=ratio_seeds, persistence=persistence,
            threshold_distance=threshold_distance, random_seed=random_seed)
        lines, lines_vectors, _ = loi_detection.filter_lois(
            lois=grown['lines'], loi_features=grown['line_features'],
            lois_vectors=[pos_px[line] for line in grown['lines']],
            number_lims=number_lims, length_lims=length_lims,
            sarcomere_mean_length_lims=sarcomere_mean_length_lims,
            sarcomere_std_length_lims=sarcomere_std_length_lims,
            midline_mean_length_lims=midline_mean_length_lims,
            midline_std_length_lims=midline_std_length_lims,
            midline_min_length_lims=midline_min_length_lims)

        index_lines = None              # a fitted straight line has no detection chain
        if mode == 'random_line':
            loi_lines, len_loi_lines, index_lines = loi_detection.select_random_lois(
                lines=lines, pos_vectors=pos_px, n_lois=n_lois)
        else:
            # Hausdorff distances (µm) between the remaining lines, then clustering
            dist = loi_detection.hausdorff_distance_lois(
                lines_vectors=[np.asarray(v, dtype=float) * self.metadata.pixelsize for v in lines_vectors],
                symmetry_mode='max')
            cluster, n_clusters = loi_detection.cluster_lois(
                hausdorff_dist_matrix=dist, distance_threshold=distance_threshold_lois, linkage=linkage)
            if mode == 'fit_straight_line':
                loi_lines, len_loi_lines = loi_detection.fit_straight_line_to_clusters(
                    lines_vectors=lines_vectors, cluster_labels=cluster, n_clusters=n_clusters,
                    pixelsize=self.metadata.pixelsize, add_length=2, n_lois=n_lois)
            else:
                select = (loi_detection.select_longest_in_cluster if mode == 'longest_in_cluster'
                          else loi_detection.select_random_from_cluster)
                loi_lines, len_loi_lines, index_lines = select(
                    lines=lines, pos_vectors=pos_px, cluster_labels=cluster, n_clusters=n_clusters, n_lois=n_lois)
        # Only what group_tracks(by='loi') and the plots read is stored. loi_index_lines is the
        # ordered detection-index chain of each line, so the grouping follows it head to tail.
        # Plain lists: the store writes this dict as JSON, which takes lists of arrays but not
        # object arrays.
        loi = {'loi_lines': [np.asarray(line, dtype=float) for line in loi_lines],
               'len_loi_lines': np.asarray(len_loi_lines, dtype=float),
               'loi_index_lines': (None if index_lines is None
                                   else [np.asarray(c, dtype=int) for c in index_lines])}
        self.data['motion.loi.data'] = loi
        if self.auto_save:
            self.store_structure_data()
        logger.info(f"detect_lois: selected {len(loi['loi_lines'])} LOI line(s).")
