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


"""``sarc.motion``: sarcomere motion of a recording, from tracked sarcomere vectors."""

from sarcasm._internal.namespace import _Namespace
from sarcasm.motion.analysis import AnalysisMixin
from sarcasm.motion.grouping import GroupingMixin
from sarcasm.motion.tracking import TrackingMixin
from sarcasm.motion.view import Motion  # noqa: F401  (per-group view; replaced by Group in phase 2)


class MotionAnalysis(TrackingMixin, GroupingMixin, AnalysisMixin, _Namespace):
    """Sarcomere motion of one recording, reached as ``sarc.motion``.

    The pipeline is track -> group -> analyze: :meth:`track` follows every sarcomere
    vector through the movie, :meth:`group_tracks` combines the tracks into groups of one
    kind (pool, M-band, myofibril, LOI, domain or custom labels), and :meth:`analyze`
    reads each group's contractions. Results are written to ``sarc.data`` under
    ``motion.*`` and parameters under ``params.motion.*``.
    """

    def full_analysis(self, by: str = 'pool', frames='all', reference_frame: int = None,
                      fast_movie: bool = False, motion_predictor: str = 'none') -> None:
        """Track and analyse the motion with default parameters, after ``sarc.detect_sarcomeres``.

        Runs ``detect_z_bands_fast_movie`` (if ``fast_movie``), ``sarc.structure.analyze_vectors``,
        :meth:`track`, the structure step the grouping needs at the reference frame
        (``analyze_myofibrils`` for myofibril, ``analyze_domains`` for domain, :meth:`detect_lois` for
        loi) and :meth:`analyze`. The results are written once at the end.

        Parameters
        ----------
        by : {'pool', 'mband', 'myofibril', 'loi', 'domain'}, optional
            The grouping to analyse. Default is 'pool'.
        frames : {'all', int, sequence of int}, optional
            Frames to track. Default is 'all'.
        reference_frame : int or None, optional
            Frame whose geometry defines the grouping; None takes the first tracked frame.
            Default is None.
        fast_movie : bool, optional
            Predict the Z-bands with the 3D U-Net for high-speed movies first. Default is False.
        motion_predictor : {'none', 'flow'}, optional
            Passed to :meth:`track`. Default is 'none'.
        """
        if by == 'custom':
            raise ValueError("A custom grouping needs labels: run group_tracks(by='custom', labels=...) and "
                             "analyze('custom') instead.")
        structure = self._sarc.structure
        frames = self._canonical_frames(frames)
        auto_save = self.auto_save
        self.auto_save = False                 # one store write at the end, not one per step
        try:
            if fast_movie:
                self.detect_z_bands_fast_movie()
            structure.analyze_vectors(frames=frames)
            self.track(frames=frames, motion_predictor=motion_predictor)
            ref = reference_frame if reference_frame is not None else int(self.data['params.motion.track.frames'][0])
            if by == 'myofibril':
                structure.analyze_myofibrils(frames=[ref])
            elif by == 'domain':
                structure.analyze_domains(frames=[ref])
            elif by == 'loi':
                self.detect_lois(frame=ref)
            self.analyze(by, reference_frame=ref)
        finally:
            self.auto_save = auto_save
        self.commit()
