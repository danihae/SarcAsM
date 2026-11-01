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


"""Tracking of sarcomere vectors through the movie (``sarc.motion.track`` and its readers)."""

import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from bio_image_unet.progress import ProgressNotifier



logger = logging.getLogger(__name__)

# Import structure modules
from sarcasm.analysis import (
    detection,
    sarcomere_tracking,
    heterogeneity,
)


class TrackingMixin:
    """Part of :class:`~sarcasm.motion.MotionAnalysis`; see there."""

    def detect_z_bands_fast_movie(self, model_path: Optional[str] = None,
                                  max_patch_size: Union[Tuple[int, int, int], str] = 'auto',
                                  normalization_mode: str = 'all',
                                  clip_thres: Tuple[float, float] = (0., 99.8),
                                  batch_size: Union[int, str] = 'auto',
                                  progress_notifier: ProgressNotifier = ProgressNotifier.progress_notifier_tqdm()) -> None:
        """
        Predict sarcomere z-bands with 3D U-Net for high-speed movies for improved temporal consistency.

        Parameters
        ----------
        model_path : str or None, optional
            ``'generalist'`` (or None) is the bundled 3D model for high-speed movies;
            ``'legacy'`` the 3D model shipped before v1.0.0 (see sarcasm/models/README.md);
            any other string is a path to custom 3D U-Net weights. The resolved model is stored as
            ``params.motion.detect_z_bands_fast_movie.model_path``. Default is None.
        max_patch_size : tuple of int or 'auto', optional
            Maximal patch dimensions ``(n_frames, n_x, n_y)`` for the CNN; each
            must be divisible by 16. 'auto' derives them from free device memory
            and the model. Default is 'auto'.
        normalization_mode : str, optional
            Intensity normalization mode for 3D stacks ('single': each image
            individually, 'all': histogram of full stack, 'first': histogram of
            first image). Default is 'all'.
        clip_thres : tuple of float, optional
            Clip threshold (lower, upper percentiles) for intensity
            normalization. Default is (0., 99.8).
        progress_notifier : ProgressNotifier, optional
            Progress notifier for inclusion in the GUI. Default is
            ProgressNotifier.progress_notifier_tqdm().
        """
        # Delegate to detection module
        info = {}
        masks = {}

        def predict():
            masks.update(detection.detect_z_bands_fast_movie_unet(
                info=info,
                images=self.read_imgs(),
                model_path=model_path,
                model_dir=str(self.model_dir),
                max_patch_size=max_patch_size,
                normalization_mode=normalization_mode,
                clip_thres=clip_thres,
                device=self.device,
                batch_size=batch_size,
                progress_notifier=progress_notifier
            ))

        self._predict_on_gpu(predict, 'detect_z_bands_fast_movie')
        for name, arr in masks.items():
            self.store.write_mask(name, np.asarray(arr))
        _dict = {'params.motion.detect_z_bands_fast_movie.model_path': info.get('model', model_path),
                 'params.motion.detect_z_bands_fast_movie.max_patch_size': max_patch_size,
                 'params.motion.detect_z_bands_fast_movie.patch_size_used': info.get('patch_size'),
                 'params.motion.detect_z_bands_fast_movie.normalization_mode': normalization_mode,
                 'params.motion.detect_z_bands_fast_movie.clip_thres': clip_thres}
        self.data.update(_dict)
        if self.auto_save:
            self.commit()

    def track(
        self,
        frames: Union[str, int, Sequence[int]] = 'all',
        max_disp_along_um: float = 1.0,
        max_disp_perp_um: float = 0.2,
        ori_tol_deg: float = 45.0,
        retire_after_s: Optional[float] = None,
        min_track_duration_s: float = 0.08,
        max_gap_interpolation_s: float = 0.05,
        motion_predictor: str = 'none',
        refine_lengths: bool = True,
        orientation_window_s: Optional[float] = None,
        max_disp_along_rest_um: Optional[float] = 0.8,
        fill_gaps: bool = True,
        fill_min_coverage: float = 0.5,
        progress_notifier: Optional[ProgressNotifier] = ProgressNotifier.progress_notifier_tqdm(),
    ) -> None:
        """2D full-field sarcomere-vector tracking.

        Each sarcomere vector in the first analyzed frame seeds a query point. Every frame, the
        candidate (query point, detection) pairs that pass the anisotropic
        (along-/perpendicular-to-sarcomere) and orientation gates are matched by an
        exact minimum-cost assignment, solved per connected component of the
        candidate graph — which, because the vectors densely sample each M-band
        midline, means each midline row is aligned jointly. A query point with no
        consistent detection records an honest gap frame (``tracks_observed`` False,
        length NaN) and keeps its identity, so a detection dropout of any length no
        longer ends a trajectory. No M-band identity is tracked; anti-convergence
        is guaranteed because each detection is matched at most once.

        Prerequisites: :meth:`~sarcasm.structure.StructureAnalysis.analyze_vectors` must have been run.

        Parameters
        ----------
        frames : {'all', int, sequence of int}, optional
            Frames to track ('all' or a contiguous selection, in any spelling of
            frame numbers; order and repeats do not matter). 'all' resolves to
            every frame that carries sarcomere vectors, which is the frame range
            structure.analyze_vectors actually covered, not necessarily the whole
            movie. Default is 'all'.
        max_disp_along_um : float, optional
            Match-gate tolerance for motion along the sarcomere axis, in µm, measured from
            the track's prediction, regardless of pixel size. On a rest frame the prediction can
            be the rest position (see ``max_disp_along_rest_um``), so the step from the previous
            frame can be larger. Default is 1.0.
        max_disp_perp_um : float, optional
            Match-gate tolerance for motion perpendicular to the sarcomere axis,
            in µm; far tighter than the along gate (a perpendicular jump is a
            swap onto a neighbouring myofibril). Default is 0.2.
        ori_tol_deg : float, optional
            Orientation tolerance for the match gate, in degrees (compared
            modulo π). Default is 45.0.
        retire_after_s : float or None, optional
            Time a track may go unmatched before it is closed, in seconds.
            ``None`` (default) means tracks never retire: an unmatched track is
            carried along by the coherent motion of its neighbourhood, so its
            identity stays valid through a dropout of any length and retiring it
            would only fragment the trajectory. Set a value (e.g. 5.0) for very
            long recordings, where sarcomeres genuinely appear and disappear, to
            bound the track count.
        min_track_duration_s : float, optional
            Minimum accumulated real observation time required to keep a track, in
            seconds. Falls back to 5 real observations when frametime is unknown.
            Default is 0.08.
        progress_notifier : ProgressNotifier or None, optional
            Reports per-frame progress: once for the tracking and once more for the length
            refinement, as two labelled console bars by default. None reports nothing.
            Default is ProgressNotifier.progress_notifier_tqdm().
        max_gap_interpolation_s : float, optional
            Longest gap, in seconds, whose sarcomere length and orientation are
            filled by interpolating between the real observations on either side,
            so brief detection flicker does not punch holes in the per-track
            traces. Interior gaps only, and ``tracks_observed`` stays False on
            filled frames, so coverage and every real-observation metric are
            unaffected. Set to 0 to leave all gap frames NaN. Default is 0.05.
        motion_predictor : {'none', 'flow'}, optional
            How a query point's position is predicted before gating. ``'none'``
            (default): hold the last observed position and advect coasting points
            with their neighbours — right whenever the per-frame motion is small
            against the gates (≥ ~30 fps). ``'flow'``: dense optical flow of the raw
            image between consecutive frames (:class:`sarcasm.analysis.optical_flow.ImageFlowPredictor`),
            applied to every live point along its sarcomere axis (the perpendicular
            component only where it is large). Use it when the frame rate is coarse
            relative to the motion, i.e. when one contraction moves a sarcomere a
            sizeable fraction of its length between consecutive frames: fewer
            fragments, higher coverage, no drift. At high frame rates it reproduces
            ``'none'``. The flow adds computation proportional to the image size.
            Requires OpenCV (``opencv-python-headless``).
        refine_lengths : bool, optional
            After tracking, give each track one sarcomere axis, the median of its
            detected orientations, and re-measure its length along that axis in every
            frame, with the Z-band map and length estimator of
            :meth:`~sarcasm.structure.StructureAnalysis.analyze_vectors`. Sarcomeres do not turn during a
            contraction, but their Z-bands tilt where neighbouring myofibrils shorten by
            different amounts (a shear), and a detection's orientation, the Z-band normal
            in that frame, follows the tilt and jitters on top of it. Normal to tilted
            Z-bands the spacing is shorter than along the filaments, by the cosine of the
            tilt. On a 61-fps movie the Z-bands tilted by 5° (SD) during contractions
            while the sarcomere axes, from the tracked displacement field, turned by less
            than 1°; along the axis the cell-average shortening was 10 % smaller than
            along the detection orientations, and the frame-to-frame length noise fell
            from 19.7 to 15.3 nm. The same profile also gives the midpoint between the
            two Z-bands, and the track's position is moved there: the tracker follows
            integer M-band skeleton pixels, whose +-0.5 px quantisation would otherwise
            show as jitter in the position and Z-band trajectories. The detection values
            are kept in ``motion.tracks.slen_raw`` / ``motion.tracks.orientations_raw``;
            frames where the axis yields no length keep their detection length and
            position. Default is True.
        orientation_window_s : float, optional
            None (default) gives every track one axis. For long recordings in which the
            tissue slowly turns, an interval in seconds gives a running median over that
            interval instead; it ignores the contraction tilt only while contractions
            fill less than half of it, so it should span several beats. With fewer than
            5 frames in the interval, or an unknown frametime, a warning is logged and
            every track gets one axis.
        max_disp_along_rest_um : float or None, optional
            Between beats the tissue returns to where it was, but during a fast beat the
            stripes can shift by about a sarcomere within a few frames and a track lands on a
            neighbouring sarcomere. So on rest frames, found from the median detection length, a track
            is predicted at its rest position, unless the tissue did not return, and this narrower
            along-axis gate applies, in µm, capped at half a sarcomere length. On 14 recordings between 0.06 and 0.22 µm/px
            with a detectable rest, tracks that shifted by more than 0.3 µm between rest periods fell
            from up to 33 % to below 2 %, with no loss of tracks on high-magnification movies. A
            movie without a detectable rest tracks as with None, which switches this off.
            ``motion.tracks.rest_frames`` holds the rest frames found afterwards from the tracked
            lengths, for filling, so it can differ from those used here. Default is 0.8.
        fill_gaps : bool, optional
            With ``refine_lengths``, measure the interior gap frames of well-covered tracks at the
            track's own position (:func:`~sarcasm.analysis.sarcomere_tracking.fill_track_gaps`). Filled
            frames go to ``motion.tracks.slen_filled``, marked in ``motion.tracks.filled``. They follow a
            contraction less reliably than real frames, so ``motion.tracks.slen``,
            ``motion.tracks.observed`` and every analysis of them stay as they are. The position of a filled
            frame does change, to the measured Z-band midpoint, which ``group_tracks`` at a reference frame and
            ``g.z_pos`` read. Default is True.
        fill_min_coverage : float, optional
            Tracks with fewer real observations than this fraction of the frames are not filled;
            they are mostly flickering M-band ends. Default is 0.5.
        """
        if motion_predictor not in ('none', 'flow'):
            raise ValueError(f"motion_predictor must be 'none' or 'flow', got {motion_predictor!r}.")
        frames = self._canonical_frames(frames)
        if 'structure.sarcomere.pos_px' not in self.data:
            raise ValueError('Sarcomere vectors not analyzed. Run sarc.structure.analyze_vectors first.')

        # 'all' = every frame that actually carries vectors (the stored params may be stale)
        pv = self.data['structure.sarcomere.pos_px']
        n_analysable = min(self.metadata.n_stack, len(pv))
        analysed = [t for t in range(n_analysable) if pv[t] is not None]
        list_frames = analysed if frames == 'all' else frames

        if len(list_frames) < 2:
            raise ValueError(
                f'Need at least 2 frames for tracking; sarcomere vectors exist for '
                f'{len(analysed)} of {self.metadata.n_stack} frames (e.g. {analysed[:8]}). '
                'Run detect_sarcomeres and then structure.analyze_vectors on the frames you '
                'want to track — detect_sarcomeres(frames=0) covers the first frame only, '
                'and M-bands and orientation are needed in every tracked frame.')

        # Tracking is temporal: real-time gaps between non-contiguous frames would
        # be treated as single-frame steps (the seconds-valued horizons assume
        # Δt = frametime).
        if not np.all(np.diff(list_frames) == 1):
            raise ValueError(
                f'track requires contiguous frames; got {list_frames}. '
                'The tracker assumes a single-frame step between consecutive entries.')

        # The per-frame vectors must ACTUALLY be present for every tracked frame.
        missing = [t for t in list_frames if t >= len(pv) or pv[t] is None]
        if missing:
            preview = missing[:8]
            raise ValueError(
                f'Sarcomere vectors are missing for {len(missing)} of {len(list_frames)} '
                f'requested frames (e.g. {preview}). structure.analyze_vectors most likely '
                'did not finish (it stores results only after the last frame). Re-run '
                'structure.analyze_vectors() to completion, or pass frames=<a fully analyzed '
                'contiguous range>.')

        # Collect the per-frame vector data that structure.analyze_vectors stored.
        pos_px_all = [
            np.asarray(self.data['structure.sarcomere.pos_px'][t], dtype=np.int32)
            if self.data['structure.sarcomere.pos_px'][t] is not None
            else np.zeros((0, 2), np.int32)
            for t in list_frames
        ]
        mid_all = [
            np.asarray(self.data['structure.sarcomere.midline_id'][t], dtype=np.int64)
            if self.data['structure.sarcomere.midline_id'][t] is not None
            else np.zeros(0, np.int64)
            for t in list_frames
        ]
        slen_all = [
            np.asarray(self.data['structure.sarcomere.slen'][t], dtype=np.float32)
            if self.data['structure.sarcomere.slen'][t] is not None
            else np.zeros(0, np.float32)
            for t in list_frames
        ]
        ori_all = [
            np.asarray(self.data['structure.sarcomere.orientation'][t], dtype=np.float32)
            if self.data['structure.sarcomere.orientation'][t] is not None
            else np.zeros(0, np.float32)
            for t in list_frames
        ]

        predictor = None
        if motion_predictor == 'flow':
            from ..analysis.optical_flow import ImageFlowPredictor
            # tracker step k is list_frames[k] -> list_frames[k + 1]; one frame is read per call
            predictor = ImageFlowPredictor(lambda k: self.read_imgs(frames=int(list_frames[k])))
            logger.info('Tracking with the image-flow motion predictor (raw-image DIS optical flow).')
        logger.info(f'Tracking {len(list_frames)} frames...')
        out = sarcomere_tracking.track_sarcomere_vectors(
            pos_px_all, mid_all, slen_all, ori_all,
            pixelsize=self.metadata.pixelsize,
            frametime=self.metadata.frametime,
            max_disp_along_um=max_disp_along_um,
            max_disp_perp_um=max_disp_perp_um,
            ori_tol_deg=ori_tol_deg,
            retire_after_s=retire_after_s,
            min_track_duration_s=min_track_duration_s,
            max_gap_interpolation_s=max_gap_interpolation_s,
            progress_notifier=progress_notifier,
            predictor=predictor,
            max_disp_along_rest_um=max_disp_along_rest_um,
        )
        refined = refine_lengths and self._refine_track_lengths(
            out, list_frames, orientation_window_s, max_gap_interpolation_s, progress_notifier,
            fill_min_coverage if fill_gaps else None)
        observed = np.asarray(out['motion.tracks.observed'], dtype=bool)
        if 'motion.tracks.filled' not in out:
            out['motion.tracks.slen_filled'] = np.where(observed, out['motion.tracks.slen'], np.nan).astype(np.float32)
            out['motion.tracks.filled'] = np.zeros_like(observed)

        tracking_data = {
            'motion.tracks.n': out['motion.tracks.n'],
            'motion.tracks.ids': out['motion.tracks.ids'],
            'motion.tracks.start_frame': out['motion.tracks.start_frame'],
            'motion.tracks.n_frames': out['motion.tracks.n_frames'],
            'motion.tracks.drift_um': out['motion.tracks.drift_um'],
            'motion.tracks.positions_um': out['motion.tracks.positions_um'],
            'motion.tracks.positions_px': out['motion.tracks.positions_px'],
            'motion.tracks.slen': out['motion.tracks.slen'],
            'motion.tracks.orientations': out['motion.tracks.orientations'],
            'motion.tracks.observed': out['motion.tracks.observed'],
            'motion.tracks.detection_id': out['motion.tracks.detection_id'],
            'motion.tracks.midline_id': out['motion.tracks.midline_id'],
            'motion.tracks.fragmentation_ratio': out['motion.tracks.fragmentation_ratio'],
            'motion.tracks.n_retired': out['motion.tracks.n_retired'],
            'motion.tracks.n_interpolated_gap_frames': out['motion.tracks.n_interpolated_gap_frames'],
            'motion.tracks.rest_frames': out['motion.tracks.rest_frames'],
            'motion.tracks.slen_filled': out['motion.tracks.slen_filled'],
            'motion.tracks.filled': out['motion.tracks.filled'],
            'params.motion.track.frames': list_frames,
            'params.motion.track.max_disp_along_um': max_disp_along_um,
            'params.motion.track.max_disp_perp_um': max_disp_perp_um,
            'params.motion.track.ori_tol_deg': ori_tol_deg,
            'params.motion.track.retire_after_s': retire_after_s,
            'params.motion.track.min_track_duration_s': min_track_duration_s,
            'params.motion.track.max_gap_interpolation_s': max_gap_interpolation_s,
            'params.motion.track.motion_predictor': motion_predictor,
            'params.motion.track.refine_lengths': refine_lengths,
            'params.motion.track.orientation_window_s': orientation_window_s,
            'params.motion.track.max_disp_along_rest_um': max_disp_along_rest_um,
            'params.motion.track.fill_gaps': fill_gaps,
            'params.motion.track.fill_min_coverage': fill_min_coverage,
        }
        raw_keys = ('motion.tracks.slen_raw', 'motion.tracks.orientations_raw')
        if refined:
            tracking_data.update({key: out[key] for key in raw_keys})
        else:                                   # no stale raw arrays from an earlier refined run
            for key in raw_keys:
                if key in self.data:
                    del self.data[key]
        self.data.update(tracking_data)
        # re-tracking changes track identities: drop any prior grouping
        self._invalidate_groupings()
        logger.info(f'Tracked {out["motion.tracks.n"]} sarcomere query points over {len(list_frames)} frames.')
        if out['motion.tracks.n'] == 0:
            logger.warning(self._NO_TRACKS_MESSAGE)
        if self.auto_save:
            self.commit()

    def _orientation_window_frames(self, orientation_window_s: Optional[float]) -> Optional[int]:
        """``orientation_window_s`` in frames (odd), or None for one axis per track, with a warning when the
        interval cannot be used."""
        if orientation_window_s is None:
            return None
        frametime = self.metadata.frametime
        if not frametime or frametime <= 0:
            logger.warning(f'orientation_window_s={orientation_window_s} cannot be converted to frames: the '
                           'frametime is unknown. Every track gets one sarcomere axis instead.')
            return None
        n_window = float(orientation_window_s) / float(frametime)        # frames within the interval
        window_frames = 2 * int(round((n_window - 1) / 2)) + 1           # nearest odd number of frames
        if window_frames < 5:
            logger.warning(f'The orientation window of {orientation_window_s} s holds {n_window:.2f} frames at '
                           f'this frame rate; the running median needs at least 5 (and several beats). Every '
                           f'track gets one sarcomere axis instead; set orientation_window_s >= '
                           f'{5 * float(frametime):.3g} s for a running median.')
            return None
        return window_frames

    def _refine_track_lengths(self, out: dict, list_frames: Sequence[int], orientation_window_s: Optional[float],
                              max_gap_interpolation_s: float,
                              progress_notifier: Optional[ProgressNotifier] = None,
                              fill_min_coverage: Optional[float] = None) -> bool:
        """Replace the tracks' detection lengths/orientations in ``out`` by lengths re-measured along each
        track's sarcomere axis and move the tracked positions to the Z-band midpoints (see ``refine_lengths``);
        the detection lengths and orientations go to the ``*_raw`` keys. With ``fill_min_coverage`` also fill
        the gaps of the tracks with that coverage (see ``fill_gaps``). Returns True."""
        frametime = self.metadata.frametime
        window_frames = self._orientation_window_frames(orientation_window_s)
        p = self.data
        zbands_name = ('zbands_fast_movie' if p.get('params.structure.analyze_vectors.zbands_source')
                       == 'fast_movie_3d' else 'zbands')
        settings = dict(
            slen_lims=tuple(p.get('params.structure.analyze_vectors.slen_lims', (1, 3))),
            interp_factor=int(p.get('params.structure.analyze_vectors.interp_factor', 4)),
            linewidth=float(p.get('params.structure.analyze_vectors.linewidth', 0.3)),
            # stores written before the method was recorded used the default of structure.analyze_vectors
            interpolation_method=str(p.get('params.structure.analyze_vectors.interpolation_method', 'akima')),
            peak_prominence=float(p.get('params.structure.analyze_vectors.peak_prominence', 0.4)))
        raw_slen = np.asarray(out['motion.tracks.slen'], dtype=np.float32)
        raw_ori = np.asarray(out['motion.tracks.orientations'], dtype=np.float32)
        observed = np.asarray(out['motion.tracks.observed'], dtype=bool)
        logger.info('Refining track lengths along ' + ('one sarcomere axis per track' if window_frames is None
                    else f'the running median orientation over {window_frames} frames') + '...')
        # frames are read in order, so read ~64 MB of pages at a time instead of one store access per frame
        height, width = self.metadata.size
        block = max(1, (64 * 2 ** 20) // (4 * int(height) * int(width)))
        pages = {'start': None, 'data': None}

        def read_zbands(k):
            start = k - k % block
            if pages['start'] != start:
                pages['start'] = start
                pages['data'] = self.read_mask(zbands_name, frames=[int(f) for f in list_frames[start:start + block]])
            return pages['data'][k - start]

        slen, ori, pos_px = sarcomere_tracking.refine_track_lengths(
            out['motion.tracks.positions_px'], raw_ori, observed, read_zbands,
            pixelsize=self.metadata.pixelsize, window_frames=window_frames, progress_notifier=progress_notifier,
            **settings)
        axis = ori.copy()
        # a frame whose axis finds no pair of Z-bands keeps its detection measurement
        keep = observed & ~np.isfinite(slen) & np.isfinite(raw_slen)
        slen[keep], ori[keep] = raw_slen[keep], raw_ori[keep]
        ori = ori.astype(np.float32)
        max_gap = (0 if not max_gap_interpolation_s or max_gap_interpolation_s <= 0 or not frametime
                   or frametime <= 0 else max(1, int(round(float(max_gap_interpolation_s) / float(frametime)))))
        n_filled = sarcomere_tracking._interpolate_short_gaps(len(slen), slen, ori, observed, max_gap)
        pos_px = np.asarray(pos_px, dtype=np.float32)
        if fill_min_coverage is not None:
            slen_filled, filled, pos_px = sarcomere_tracking.fill_track_gaps(
                out['motion.tracks.positions_px'], axis, observed, slen, pos_px, out['motion.tracks.rest_frames'],
                read_zbands, pixelsize=self.metadata.pixelsize, min_coverage=fill_min_coverage,
                progress_notifier=progress_notifier, **settings)
            logger.info(f'Filled {int(filled.sum())} gap frames at the tracks\' own positions, in '
                        "'motion.tracks.slen_filled'.")
            out.update({'motion.tracks.slen_filled': slen_filled, 'motion.tracks.filled': filled})
        out.update({'motion.tracks.slen_raw': raw_slen, 'motion.tracks.orientations_raw': raw_ori,
                    'motion.tracks.slen': slen, 'motion.tracks.orientations': ori,
                    'motion.tracks.positions_px': pos_px,
                    'motion.tracks.positions_um': (pos_px * self.metadata.pixelsize).astype(np.float32),
                    'motion.tracks.n_interpolated_gap_frames': int(n_filled)})
        logger.info(f'Refined the lengths of {int(observed.sum())} observed track frames; {int(keep.sum())} '
                    'kept their detection length.')
        return True

    def tracks(self, min_coverage: float = 0.0) -> pd.DataFrame:
        """Tidy per-track summary of the 2D tracker output (one row per track).

        Discoverable accessor over the dense ``tracks_*`` arrays written by
        :meth:`track`, so downstream code does not have to
        reach into raw ``self.data`` and reconstruct shapes/semantics.

        Parameters
        ----------
        min_coverage : float, optional
            Only return tracks whose coverage (observed frames / n_frames)
            is at least this value. Default 0.0 (all kept tracks).

        Returns
        -------
        pandas.DataFrame
            Columns: ``track_id``, ``start_frame``, ``length`` (number of observed
            frames), ``n_frames``, ``coverage``, ``mean_slen``, ``std_slen``,
            ``drift_um`` (how far the track wandered from the coherent motion of
            its neighbours — roughly one sarcomere length means it probably
            changed identity; NaN if too short to score),
            ``start_y_um``, ``start_x_um``, ``ref_midline_id`` (the M-band id at
            the start frame), and per stored grouping ``group_<kind>`` (the track's group
            in that grouping, -1 if unassigned), plus ``order_<kind>`` (rank along the line)
            for the ordered kinds myofibril and loi.
        """
        if 'motion.tracks.slen' not in self.data:   # zero tracks give an empty table
            raise ValueError('No tracks found. Run sarc.motion.track() first.')

        n_tracks = int(self.data.get('motion.tracks.n', 0))
        columns = ['track_id', 'start_frame', 'length', 'n_frames', 'coverage',
                   'mean_slen', 'std_slen', 'drift_um', 'start_y_um', 'start_x_um',
                   'ref_midline_id']
        if n_tracks == 0:
            return pd.DataFrame(columns=columns)

        slen = np.asarray(self.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)
        T = slen.shape[1]
        observed = np.asarray(self.data['motion.tracks.observed']).astype(bool).reshape(n_tracks, T)
        pos = np.asarray(self.data['motion.tracks.positions_um'], dtype=float).reshape(n_tracks, T, 2)
        det_mid = np.asarray(self.data.get('motion.tracks.midline_id', np.full((n_tracks, T), -1))).reshape(n_tracks, T)

        track_ids = np.asarray(self.data.get('motion.tracks.ids', np.arange(n_tracks)))
        start_frame = np.asarray(self.data.get('motion.tracks.start_frame', np.zeros(n_tracks, int))).astype(int)
        length = np.asarray(self.data.get('motion.tracks.n_frames', observed.sum(axis=1))).astype(int)
        coverage = length / float(T) if T else np.zeros(n_tracks)

        rows = np.arange(n_tracks)
        start_pos = pos[rows, start_frame]  # (n_tracks, 2) in (y, x)
        ref_mid = det_mid[rows, start_frame]
        with np.errstate(invalid='ignore'):
            mean_slen = np.nanmean(slen, axis=1)
            std_slen = np.nanstd(slen, axis=1)

        df = pd.DataFrame({
            'track_id': track_ids,
            'start_frame': start_frame,
            'length': length,
            'n_frames': T,
            'coverage': coverage,
            'mean_slen': mean_slen,
            'std_slen': std_slen,
            'drift_um': np.asarray(self.data.get(
                'motion.tracks.drift_um', np.full(n_tracks, np.nan)), dtype=float).reshape(-1),
            'start_y_um': start_pos[:, 0],
            'start_x_um': start_pos[:, 1],
            'ref_midline_id': ref_mid,
        })
        for kind in self._grouped_kinds():
            df[f'group_{kind}'] = np.asarray(self.data[self._gkey(kind, 'track_group_id')]).reshape(-1)[:n_tracks]
            if kind in self._CHAIN_LEVELS:
                df[f'order_{kind}'] = np.asarray(
                    self.data[self._gkey(kind, 'track_group_order')]).reshape(-1)[:n_tracks]

        if min_coverage > 0.0:
            df = df[df['coverage'] >= min_coverage].reset_index(drop=True)
        return df

    def kinematics(self, filter_params: Tuple[int, int] = (13, 5),
                             slen_lims: Tuple[float, float] = (1.2, 3.0)) -> Dict[str, np.ndarray]:
        """Per-track length change and velocity over time.

        The equilibrium (resting) length of each track is its median over the
        non-contracting frames of the pooled cell signal when
        ``analyze(by='pool')`` has run, otherwise over all its frames.
        Feeds the ΔSL and velocity colourings of the napari app, :meth:`Plots.plot_track_vectors` and :meth:`Plots.plot_track_raster`.

        Parameters
        ----------
        filter_params : (int, int), optional
            Savitzky-Golay ``(window_length, polyorder)`` used to smooth the length
            before differentiating. Default is (13, 5).
        slen_lims : (float, float), optional
            Lengths outside this range (µm) are treated as missing. Default is (1.2, 3.0).

        Returns
        -------
        dict
            ``'slen'`` (µm, NaN outside ``slen_lims``), ``'delta_slen'`` (µm) and
            ``'vel'`` (µm/s), each ``(n_tracks, n_frames)`` with NaN on unobserved
            frames; ``'equ'``
            ``(n_tracks,)``; ``'contr'`` the ``(n_frames,)`` contraction mask the
            equilibrium was taken against (all False without a pooled analysis);
            ``'coverage'`` ``(n_tracks,)``.
        """
        self._require_tracks()
        n_tracks = int(self.data.get('motion.tracks.n', 0))
        slen = np.asarray(self.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)
        T = slen.shape[1]
        contr = self.data.get('motion.pool.contr')
        if contr is not None and np.asarray(contr).size == T:
            contr = np.asarray(contr, dtype=bool).reshape(-1)
        else:
            contr = np.zeros(T, dtype=bool)
        kin = heterogeneity.member_kinematics(slen, contr, self.metadata.frametime,
                                              filter_params=filter_params, slen_lims=slen_lims)
        observed = np.asarray(self.data.get('motion.tracks.observed',
                                            np.isfinite(slen))).astype(bool).reshape(n_tracks, T)
        return {'slen': kin['slen'], 'delta_slen': kin['delta_slen'], 'vel': kin['vel'], 'equ': kin['equ'],
                'contr': contr, 'coverage': observed.sum(axis=1) / float(T) if T else np.zeros(n_tracks)}

    # ------------------------------------------------------------------
    # Post-tracking grouping + motion analysis (group_tracks -> analyze):
    # every level is a grouping over the same tracks_slen(t); a fingerprint
    # hard-raises if a grouped result is read after its grouping changed.
    # ------------------------------------------------------------------

    def _n_sarcomere_vectors_tracked(self) -> int:
        """Total number of detected sarcomere-vector observations over the tracked
        frames (``sum_t N_vectors(t)``) — the denominator for the long-track vector
        coverage QC. Uses the stored per-frame ``n_vectors`` count, restricted to the
        frames that were actually tracked; falls back to counting ``pos_vectors``."""
        nv = self.data.get('structure.sarcomere.n_vectors')
        tracked = self.data.get('params.motion.track.frames')
        if nv is not None:
            nv = np.asarray(nv).ravel()
            if tracked is not None:
                idx = [int(f) for f in tracked if 0 <= int(f) < nv.shape[0]]
                return int(nv[idx].sum())
            return int(nv.sum())
        pv = self.data.get('structure.sarcomere.pos')
        if pv is not None:
            return int(sum(len(np.asarray(p)) for p in pv if p is not None))
        return 0

    _NO_TRACKS_MESSAGE = ('The tracking found no tracks: no sarcomere was followed for at least '
                          'min_track_duration_s. Check that it is shorter than the movie and that the sarcomere '
                          'vectors were analyzed on every frame.')

    def _require_tracks(self):
        """Raise a ValueError naming the reason when there are no tracks to work with."""
        if 'motion.tracks.slen' not in self.data:
            raise ValueError('No tracks found. Run sarc.motion.track() first.')
        if int(self.data.get('motion.tracks.n', 0)) == 0:
            raise ValueError(self._NO_TRACKS_MESSAGE)

    def _tracked_frame_index(self, reference_frame: int) -> int:
        """Window index (into the tracks_* time axis) of an absolute movie frame."""
        tracked = self.data.get('params.motion.track.frames')
        if tracked is None:
            return int(reference_frame)
        tracked = [int(f) for f in tracked]
        if reference_frame not in tracked:
            raise ValueError(
                f'reference_frame {reference_frame} was not tracked. '
                f'Tracked frames: {tracked[0]}..{tracked[-1]}.')
        return tracked.index(int(reference_frame))
