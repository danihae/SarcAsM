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

"""The groups of one grouping kind, read from the store: ``sarc.motion.groups(kind)``.

A :class:`Group` is the same class for every kind (pool, mband, myofibril, loi, domain, custom).
Besides ``kind``, ``id``, ``ordered``, ``n_members`` and the recording ``sarc``, it holds no results of
its own; its attributes are views of the stored ones, under the same names:

* ``g.<name>`` is the group's row of ``motion.<kind>.<name>`` (``g.slen_avg``, ``g.contr``,
  ``g.beating_rate``, ...); a per-track result (``track_popping``) gives the members' rows.
* ``g.tracks.<name>`` is the members' slice of ``motion.tracks.<name>`` (``ids``, ``slen``,
  ``positions_um``, ...), plus their length change ``delta_slen``, velocity ``vel`` and resting length
  ``equ``, computed against the group's own contractions as ``sarc.motion.analyze`` does.
* ``g.z_pos``: the Z-band positions along the line, for the ordered kinds (myofibril, loi).

Members are in track order, or start to end along the line for the ordered kinds.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from sarcasm.analysis import grouped_motion, heterogeneity

if TYPE_CHECKING:
    from sarcasm.motion import MotionAnalysis

#: Results stored once per grouping, not once per group.
_SHARED = ('oscill_frequencies',)
# motion.tracks.* results with one value per frame, not per track: no member slice.
_PER_FRAME = ('rest_frames',)
#: Member quantities computed from the lengths (not stored): see :class:`GroupTracks`.
_KINEMATICS = ('delta_slen', 'vel', 'equ')


class GroupSet(Sequence):
    """The groups of one grouping kind: ``sarc.motion.groups('myofibril')``; ``groups[i]`` is a :class:`Group`."""

    def __init__(self, motion: "MotionAnalysis", kind: str):
        grouping = motion._grouping(kind)            # raises if the kind was never grouped
        snapshot = np.asarray(motion.data.get(motion._gkey(kind, 'track_ids'), []))
        if not np.array_equal(snapshot, np.asarray(motion.data.get('motion.tracks.ids', []))):
            raise ValueError(f"Tracks changed since the '{kind}' grouping was made. Run "
                             f"sarc.motion.group_tracks(by='{kind}') again.")
        self._motion, self.kind = motion, kind
        self._grouping = grouping

    def __len__(self) -> int:
        return self._grouping['n']

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        n = len(self)
        j = int(i) + n if int(i) < 0 else int(i)
        if not 0 <= j < n:
            raise IndexError(f"{self.kind} group {i} out of range (n={n}).")
        return Group(self._motion, self.kind, j, self._grouping)

    def __repr__(self) -> str:
        return f"<GroupSet {self.kind}: {len(self)} groups>"


class Group:
    """One group of one grouping kind; see the module docstring for its attributes."""

    def __init__(self, motion: "MotionAnalysis", kind: str, gid: int, grouping: Dict[str, np.ndarray]):
        self._motion, self.kind, self.id = motion, kind, int(gid)
        self.sarc = motion._sarc                     # the recording, for images and calibration
        self.ordered = kind in motion._CHAIN_LEVELS
        members = np.flatnonzero(grouping['track_group_id'] == self.id)
        if self.ordered:
            members = members[np.argsort(grouping['track_group_order'][members], kind='stable')]
        self._members = members
        self._n_groups = grouping['n']
        self.n_members = int(members.size)
        self.tracks = GroupTracks(self)

    def __repr__(self) -> str:
        return f"<Group {self.kind} {self.id}: {self.n_members} tracks>"

    def __dir__(self) -> List[str]:
        prefix = f'motion.{self.kind}.'
        stored = [k[len(prefix):] for k in self._motion.data if k.startswith(prefix)]
        return sorted(set(super().__dir__()) | set(stored))

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        key = f'motion.{self.kind}.{name}'
        if key not in self._motion.data:
            raise AttributeError(f"'{name}' is not a result of the {self.kind} grouping (keys motion.{self.kind}.*). "
                                 f"Run sarc.motion.analyze('{self.kind}') if it has not been analysed.")
        self._motion._assert_track_motion_fresh(self.kind)
        value = np.asarray(self._motion.data[key])
        if name in _SHARED:
            return value
        if name.startswith('track_'):
            return value[self._members]
        return value[self.id]

    @property
    def z_pos(self) -> np.ndarray:
        """Z-band positions (µm) along the line, ``(n_members + 1, T)``, for the ordered kinds."""
        if not self.ordered:
            raise ValueError(f"z_pos exists for the ordered kinds {self._motion._CHAIN_LEVELS}, "
                                 f"not for '{self.kind}'.")
        motion = self._motion
        ref_frame = int(motion.data.get(f'params.motion.group_tracks.{self.kind}.reference_frame', 0))
        try:
            ref_idx = motion._tracked_frame_index(ref_frame)
        except (ValueError, KeyError):
            ref_idx = 0
        z_pos, _, _ = grouped_motion.synthesize_loi_chain(
            self.tracks.slen.astype(float), motion.metadata.frametime,
            member_pos=self.tracks.positions_um.astype(float), ref_idx=ref_idx)
        return z_pos


class GroupTracks:
    """The member tracks of a :class:`Group`: ``g.tracks.<name>``."""

    def __init__(self, group: Group):
        self._group = group
        self._kinematics = None

    def __dir__(self) -> List[str]:
        stored = [k[len('motion.tracks.'):] for k in self._group._motion.data if k.startswith('motion.tracks.')]
        return sorted(set(super().__dir__()) | set(stored) | set(_KINEMATICS))

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in _KINEMATICS:
            return self._member_kinematics()[name]
        data = self._group._motion.data
        key = f'motion.tracks.{name}'
        if key not in data:
            raise AttributeError(f"'{name}' is neither a track result (motion.tracks.*) nor one of {_KINEMATICS}.")
        return np.asarray(data[key]) if name in _PER_FRAME else np.asarray(data[key])[self._group._members]

    def _member_kinematics(self) -> Dict[str, np.ndarray]:
        """ΔSL, velocity and resting length of the members, against the group's own contractions."""
        if self._kinematics is None:
            g = self._group
            motion, kind = g._motion, g.kind
            T = int(np.asarray(motion.data['motion.tracks.slen']).shape[1])
            if f'params.motion.analyze.{kind}.grouping_hash' in motion.data:
                motion._assert_track_motion_fresh(kind)
                contr = np.asarray(motion.data[f'motion.{kind}.contr'])[g.id].astype(bool)
            else:                                         # not analysed: rest over every frame
                contr = np.zeros(T, bool)
            filter_params = tuple(motion.data.get(f'params.motion.analyze.{kind}.filter_params', (13, 5)))
            slen_lims = tuple(motion.data.get(f'params.motion.analyze.{kind}.slen_lims', (1.0, 3.0)))
            self._kinematics = heterogeneity.member_kinematics(
                self.slen.astype(float), contr, motion.metadata.frametime,
                filter_params=filter_params, slen_lims=slen_lims)
        return self._kinematics
