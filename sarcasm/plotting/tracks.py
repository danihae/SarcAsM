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

"""Plots of the tracked sarcomere vectors (trajectories, vector overlays, groupings, coverage, raster)."""

import warnings
from typing import Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib_scalebar.scalebar import ScaleBar

from sarcasm.plotting.plot_utils import PlotUtils
from sarcasm.core import SarcAsM
from sarcasm.analysis import heterogeneity
from sarcasm.plotting.structure import StructurePlots
from sarcasm.plotting.structure import _LABEL_DELTA_SL, _LABEL_SL, _LABEL_TIME


class TrackPlots:
    """Plots of the tracked sarcomere vectors (trajectories, vector overlays, groupings, coverage, raster). Part of :class:`~sarcasm.plotting.plots.Plots`."""

    @staticmethod
    def _track_coverage(sarc_obj: SarcAsM) -> np.ndarray:
        """Per-track completeness in percent: observed frames / tracked frames * 100.

        The denominator is the number of frames the tracker ran over (not the
        number of frames the track itself spans), so 100 % means the sarcomere
        was matched to a real detection in *every* frame of the recording.
        Interpolated gap frames are not observations and do not count.

        Parameters
        ----------
        sarc_obj : SarcAsM
            SarcAsM with :meth:`MotionAnalysis.track` results.

        Returns
        -------
        np.ndarray
            Coverage in percent, shape ``(n_tracks,)``.
        """
        if 'motion.tracks.n_frames' not in sarc_obj.data:
            raise ValueError('No tracks found. Run sarc.motion.track() first.')
        n_tracks = int(sarc_obj.data.get('motion.tracks.n', 0))
        n_t = np.asarray(sarc_obj.data['motion.tracks.observed']).reshape(n_tracks, -1).shape[1]
        return np.asarray(sarc_obj.data['motion.tracks.n_frames'], dtype=float) / max(n_t, 1) * 100.0

    @staticmethod
    def plot_tracks(ax: Axes, sarc_obj: SarcAsM, frame: int = 0, color_by: str = 'coverage',
                    kind: Optional[str] = None, cmap: str = 'viridis', linewidth: float = 0.8, only_observed: bool = True,
                    max_tracks: Optional[int] = 2000, alpha: float = 0.8,
                    clim: Optional[Tuple[float, float]] = None,
                    scalebar: bool = True, colorbar: bool = False, title: Optional[str] = None,
                    show_image: bool = False, show_z_bands: bool = False,
                    invert_image: bool = False, invert_z_bands: bool = False,
                    cmap_image: str = 'gray', cmap_z_bands: str = 'Greys_r',
                    alpha_image: float = 1, alpha_z_bands: float = 1):
        """
        Draw each tracked sarcomere as a trajectory line (its centre's path over time).

        One thin polyline per track links the sarcomere-centre positions across
        frames, coloured per track. Because sarcomere centres oscillate by only a
        fraction of a micron, each line is a small local squiggle — together they
        show where the tissue moves and how the tracks are grouped.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            SarcAsM with :meth:`MotionAnalysis.track` results.
        frame : int, optional
            Movie frame used for the **background** image/Z-bands only (the
            trajectories always span the full movie). Default is 0.
        color_by : {'coverage', 'slen', 'group'}, optional
            Per-track colour: observation coverage in percent, the track's mean sarcomere
            length, or the track's group in the ``kind`` grouping. Default is 'coverage'.
        kind : str, optional
            The grouping ``color_by='group'`` shows. Default is None.
        cmap : str, optional
            Colormap for the 'coverage'/'slen' colourings ('group' always uses a
            discrete ``gist_rainbow``). Default is 'viridis'.
        linewidth : float, optional
            Width of the trajectory lines. Default is 0.8.
        only_observed : bool, optional
            Only link frames where the track was actually observed (vs a predicted gap frame)
            (vs a predicted gap frame); the line bridges across the skipped frames.
            Default is True.
        max_tracks : int or None, optional
            Draw at most this many tracks (longest-coverage first) for legibility.
            None draws all. Default is 2000.
        alpha : float, optional
            Line opacity. Default is 0.8.
        clim : tuple of float, optional
            Fixed ``(vmin, vmax)`` for the 'coverage'/'slen' colour scale. If None, the
            scale is stretched to the data range. Default is None.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        colorbar : bool, optional
            Whether to add a colorbar to the plot (only for 'coverage'/'slen'). Default is False.
        title : str, optional
            The title for the plot. Default is None.
        show_image : bool, optional
            Whether to show the raw microscopy image as background. Default is False.
        show_z_bands : bool, optional
            Whether to show the Z-band mask as background. Mutually exclusive with
            ``show_image``. Default is False.
        invert_image : bool, optional
            Reverse the raw-image colormap (e.g. 'gray' -> 'gray_r'). Default is False.
        invert_z_bands : bool, optional
            Reverse the Z-band colormap (e.g. 'Greys_r' -> 'Greys'). Default is False.
        cmap_image : str, optional
            Colormap of the raw image background. Default is 'gray'.
        cmap_z_bands : str, optional
            Colormap of the Z-band background. Default is 'Greys_r'.
        alpha_image : float, optional
            Opacity of the raw image background. Default is 1.
        alpha_z_bands : float, optional
            Opacity of the Z-band background. Default is 1.
        """
        frame = StructurePlots._frame(sarc_obj, frame)
        if 'motion.tracks.positions_px' not in sarc_obj.data:
            raise ValueError('No tracks found. Run sarc.motion.track() first.')
        n_tracks = int(sarc_obj.data.get('motion.tracks.n', 0))
        pos = np.asarray(sarc_obj.data['motion.tracks.positions_px'], dtype=float).reshape(n_tracks, -1, 2)
        observed = np.asarray(sarc_obj.data['motion.tracks.observed']).reshape(n_tracks, -1).astype(bool)

        # Per-track colour scalar (computed once per track, not per frame).
        if color_by == 'slen':
            slen = np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)
            with np.errstate(invalid='ignore'):
                c = np.nanmean(np.where(observed, slen, np.nan), axis=1)
            clabel = 'Sarcomere length [µm]'
        elif color_by == 'group':
            if kind is None:
                raise ValueError("color_by='group' needs the grouping, e.g. kind='mband'.")
            c = sarc_obj.motion._grouping(kind)['track_group_id'].astype(float)
            clabel = f'Group id ({kind})'
        else:  # coverage
            c = TrackPlots._track_coverage(sarc_obj)
            clabel = 'Track coverage [%]'

        # Build one (x, y) polyline per track from its finite (and, if requested,
        # observed) centre positions in time order; gaps are dropped, not split.
        segments, seg_vals = [], []
        for k in range(n_tracks):
            if color_by != 'group' and not np.isfinite(c[k]):
                continue          # no colour value (e.g. an all-gap track for 'slen')
            if color_by == 'group' and c[k] < 0:
                continue          # dropped by min_coverage / unassigned
            keep = np.isfinite(pos[k, :, 0]) & np.isfinite(pos[k, :, 1])
            if only_observed:
                keep &= observed[k]
            if keep.sum() < 2:
                continue          # need >= 2 vertices to draw a line
            p = pos[k][keep]      # (m, 2) yx in time order
            segments.append(np.column_stack([p[:, 1], p[:, 0]]))  # -> (m, 2) as (x, y)
            seg_vals.append(c[k])

        # Keep the longest (most-covered) trajectories first for legibility.
        if max_tracks is not None and len(segments) > max_tracks:
            order = np.argsort([-seg.shape[0] for seg in segments])[:max_tracks]
            segments = [segments[i] for i in order]
            seg_vals = [seg_vals[i] for i in order]

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        lc = None
        norm = None
        if segments:
            vals = np.asarray(seg_vals, dtype=float)
            if color_by == 'group':
                n_groups = max(sarc_obj.motion._grouping(kind)['n'], 1)
                gcm = plt.get_cmap('gist_rainbow')
                colors = gcm((vals.astype(int) % n_groups) / n_groups)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
            else:
                finite = vals[np.isfinite(vals)]
                if clim is not None:
                    norm = plt.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
                elif finite.size:
                    norm = plt.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
                lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm,
                                    linewidths=linewidth, alpha=alpha)
                lc.set_array(vals)
            ax.add_collection(lc)

        if colorbar and lc is not None and color_by != 'group' and norm is not None:
            cb = ax.figure.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(clabel, fontsize=PlotUtils.fontsize - 1)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

    @staticmethod
    def track_vector_segments(sarc_obj: SarcAsM, frame: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z-band-to-Z-band line of every tracked sarcomere in one frame, in image pixels.

        Uses the tracker's centre position, length and orientation of each track; the
        geometry is the one of :meth:`plot_sarcomere_vectors`: endpoints
        ``(x -/+ cos(theta) L/2, y -/+ sin(theta) L/2)`` on the image axes (y pointing down).

        Parameters
        ----------
        sarc_obj : SarcAsM
            SarcAsM with :meth:`MotionAnalysis.track` results.
        frame : int
            Frame index (position in the tracked frame list).
        mask : np.ndarray, optional
            Boolean ``(n_tracks,)`` selection; default = tracks observed in ``frame`` with a
            finite length and orientation.

        Returns
        -------
        np.ndarray
            Segments ``(n_selected, 2, 2)`` as ``[[x0, y0], [x1, y1]]`` for a
            :class:`matplotlib.collections.LineCollection` (the default selection is
            :meth:`track_vector_mask`).
        """
        frame = StructurePlots._frame(sarc_obj, frame)
        n_tracks = int(sarc_obj.data.get('motion.tracks.n', 0))
        t = sarc_obj.motion._tracked_frame_index(frame)
        pos = np.asarray(sarc_obj.data['motion.tracks.positions_px'], dtype=float).reshape(n_tracks, -1, 2)[:, t]
        slen = np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)[:, t]
        ori = np.asarray(sarc_obj.data['motion.tracks.orientations'], dtype=float).reshape(n_tracks, -1)[:, t]
        if mask is None:
            mask = TrackPlots.track_vector_mask(sarc_obj, frame)
        half = 0.5 * slen[mask] / float(sarc_obj.metadata.pixelsize)
        dx, dy = np.cos(ori[mask]) * half, np.sin(ori[mask]) * half
        x, y = pos[mask, 1], pos[mask, 0]
        return np.stack([np.column_stack([x - dx, y - dy]), np.column_stack([x + dx, y + dy])], axis=1)

    @staticmethod
    def track_vector_mask(sarc_obj: SarcAsM, frame: int, only_observed: bool = True) -> np.ndarray:
        """
        Tracks that can be drawn as a vector in ``frame``: finite position, length and
        orientation and, with ``only_observed``, matched to a real detection (no gap or
        interpolated frame).

        Returns
        -------
        np.ndarray
            Boolean mask, shape ``(n_tracks,)``.
        """
        frame = StructurePlots._frame(sarc_obj, frame)
        n_tracks = int(sarc_obj.data.get('motion.tracks.n', 0))
        t = sarc_obj.motion._tracked_frame_index(frame)
        pos = np.asarray(sarc_obj.data['motion.tracks.positions_px'], dtype=float).reshape(n_tracks, -1, 2)[:, t]
        slen = np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)[:, t]
        ori = np.asarray(sarc_obj.data['motion.tracks.orientations'], dtype=float).reshape(n_tracks, -1)[:, t]
        ok = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1]) & np.isfinite(slen) & np.isfinite(ori)
        if only_observed:
            ok &= np.asarray(sarc_obj.data['motion.tracks.observed']).reshape(n_tracks, -1)[:, t].astype(bool)
        return ok

    #: Default colormap, colour limits and label per track quantity (shared with the animations).
    TRACK_COLOR_DEFAULTS = {
        'slen': ('viridis', (1.5, 2.1), 'SL [µm]'),
        'delta_slen': ('RdBu_r', (-0.3, 0.3), 'ΔSL [µm]'),
        'vel': ('RdBu_r', (-8.0, 8.0), 'V [µm/s]'),
        'coverage': ('viridis', (0.0, 100.0), 'Track coverage [%]'),
        'group': ('gist_rainbow', None, 'Group id'),
    }

    @staticmethod
    def track_color_values(sarc_obj: SarcAsM, color_by: str = 'slen',
                           kinematics: Optional[dict] = None, kind: Optional[str] = None) -> np.ndarray:
        """
        Per-track colour values for the vector overlays.

        Parameters
        ----------
        sarc_obj : SarcAsM
            SarcAsM with tracks.
        color_by : {'slen', 'delta_slen', 'vel', 'coverage', 'group'}
            Quantity. ``'delta_slen'`` and ``'vel'`` come from
            :meth:`MotionAnalysis.kinematics` (equilibrium over the quiet frames of the pooled
            signal when available).
        kinematics : dict, optional
            A cached result of :meth:`MotionAnalysis.kinematics` to avoid recomputing it per frame.
        kind : str, optional
            The grouping whose group ids ``color_by='group'`` shows. Default is None.

        Returns
        -------
        np.ndarray
            ``(n_tracks, n_frames)`` for the time-dependent quantities, ``(n_tracks,)`` for
            ``'coverage'`` and ``'group'``.
        """
        n_tracks = int(sarc_obj.data.get('motion.tracks.n', 0))
        if color_by == 'slen':
            return np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)
        if color_by in ('delta_slen', 'vel'):
            kin = kinematics if kinematics is not None else sarc_obj.motion.kinematics()
            return np.asarray(kin[color_by], dtype=float).reshape(n_tracks, -1)
        if color_by == 'coverage':
            return TrackPlots._track_coverage(sarc_obj)
        if color_by == 'group':
            if kind is None:
                raise ValueError("color_by='group' needs the grouping, e.g. kind='mband'.")
            return sarc_obj.motion._grouping(kind)['track_group_id'].astype(float)
        raise ValueError(f"unknown color_by {color_by!r}")

    @staticmethod
    def plot_track_vectors(ax: Axes, sarc_obj: SarcAsM, frame: int = 0, color_by: str = 'slen',
                           kind: Optional[str] = None, cmap: Optional[str] = None, clim: Optional[Tuple[float, float]] = None,
                           linewidth: Optional[float] = None, only_observed: bool = True,
                           scalebar: bool = True, colorbar: bool = False, title: Optional[str] = None,
                           show_image: bool = False, show_z_bands: bool = False,
                           invert_image: bool = False, invert_z_bands: bool = False,
                           cmap_image: str = 'gray', cmap_z_bands: str = 'Greys_r',
                           alpha_image: float = 1, alpha_z_bands: float = 1,
                           kinematics: Optional[dict] = None):
        """
        Draw the tracked sarcomeres of one frame as Z-band-to-Z-band lines coloured by a track quantity.

        The tracked counterpart of :meth:`plot_sarcomere_vectors`: every line spans the sarcomere
        it measures, so with the default line width (the lateral spacing of the vectors) the
        space between neighbouring Z-bands is filled with the colour of its length, length change,
        velocity, coverage or group. The returned collection can be updated frame by frame
        (see :class:`sarcasm.plotting.animation.Animations`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        sarc_obj : SarcAsM
            SarcAsM with :meth:`MotionAnalysis.track` results.
        frame : int, optional
            Frame to draw. Default is 0.
        color_by : {'slen', 'delta_slen', 'vel', 'coverage', 'group'}, optional
            Colour quantity. Default is 'slen'.
        kind : str, optional
            The grouping ``color_by='group'`` shows. Default is None.
        cmap : str, optional
            Colormap; default per quantity from :attr:`TRACK_COLOR_DEFAULTS`.
        clim : tuple of float, optional
            Colour limits; default per quantity from :attr:`TRACK_COLOR_DEFAULTS`.
        linewidth : float, optional
            Line width in points. Default fills the space between the Z-bands (one vector spacing).
        only_observed : bool, optional
            Hide tracks without a real detection in ``frame``. Default is True.
        scalebar, colorbar, title : optional
            As in :meth:`plot_tracks`.
        show_image, show_z_bands, invert_image, invert_z_bands, cmap_image, cmap_z_bands, alpha_image, alpha_z_bands
            Background options as in :meth:`plot_tracks`.
        kinematics : dict, optional
            Cached :meth:`MotionAnalysis.kinematics` result.

        Returns
        -------
        matplotlib.collections.LineCollection
            The drawn vectors (``set_segments`` / ``set_array`` to update).
        """
        frame = StructurePlots._frame(sarc_obj, frame)
        if 'motion.tracks.positions_px' not in sarc_obj.data:
            raise ValueError('No tracks found. Run sarc.motion.track() first.')
        cmap_d, clim_d, clabel = TrackPlots.TRACK_COLOR_DEFAULTS[color_by]
        cmap = cmap or cmap_d
        clim = clim if clim is not None else clim_d
        mask = TrackPlots.track_vector_mask(sarc_obj, frame, only_observed=only_observed)
        values = TrackPlots.track_color_values(sarc_obj, color_by, kinematics, kind=kind)
        vals = values[:, sarc_obj.motion._tracked_frame_index(frame)] if values.ndim == 2 else values
        if color_by == 'group':
            mask &= vals >= 0
        mask &= np.isfinite(vals)
        segments = TrackPlots.track_vector_segments(sarc_obj, frame, mask)

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)
        if linewidth is None:
            # one vector spacing (1 px) expressed in points for the current axes size
            bb = ax.get_window_extent()
            width_px = float(sarc_obj.metadata.size[-1])
            linewidth = max(72.0 * bb.width / ax.figure.dpi / width_px * 1.4, 0.3)
        if color_by == 'group':
            n_groups = max(sarc_obj.motion._grouping(kind)['n'], 1)
            colors = plt.get_cmap(cmap)((vals[mask].astype(int) % n_groups) / n_groups)
            lc = LineCollection(segments, colors=colors, linewidths=linewidth, capstyle='butt')
        else:
            lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=plt.Normalize(*clim),
                                linewidths=linewidth, capstyle='butt')
            lc.set_array(vals[mask])
        ax.add_collection(lc)
        if not (show_image or show_z_bands):
            h, w = sarc_obj.metadata.size[-2:]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect('equal')
        if colorbar and color_by != 'group':
            cb = ax.figure.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(clabel, fontsize=PlotUtils.fontsize - 1)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)
        return lc

    @staticmethod
    def plot_track_groups(ax: Axes, sarc_obj: SarcAsM, kind: str, frame: int = 0, cmap: str = 'gist_rainbow',
                          s: float = 5, show_dropped: bool = True, dropped_color: str = 'lightgrey',
                          scalebar: bool = True, title: Optional[str] = None,
                          show_image: bool = False, show_z_bands: bool = False,
                          invert_image: bool = False, invert_z_bands: bool = False,
                          cmap_image: str = 'gray', cmap_z_bands: str = 'Greys_r',
                          alpha_image: float = 1, alpha_z_bands: float = 1):
        """
        QC view of a track grouping: colour each tracked centre by its group.

        Lets a user eyeball the partition (and the tracks dropped by
        ``min_coverage``, drawn in grey) before running the expensive
        :meth:`MotionAnalysis.analyze`. Requires :meth:`MotionAnalysis.group_tracks`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            SarcAsM with a track grouping (:meth:`MotionAnalysis.group_tracks`).
        kind : str
            The grouping to show ('pool', 'mband', 'myofibril', ...).
        frame : int, optional
            Movie frame to draw. Default is 0.
        cmap : str, optional
            Colormap used to colour the groups. Default is 'gist_rainbow'.
        s : float, optional
            Marker size. Default is 5.
        show_dropped : bool, optional
            Whether to draw tracks dropped by ``min_coverage``. Default is True.
        dropped_color : str, optional
            Colour of the dropped tracks. Default is 'lightgrey'.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        title : str, optional
            The title for the plot. If None, a default title is used. Default is None.
        show_image : bool, optional
            Whether to show the raw microscopy image as background. Default is False.
        show_z_bands : bool, optional
            Whether to show the Z-band mask as background. Mutually exclusive with
            ``show_image``. Default is False.
        invert_image : bool, optional
            Reverse the raw-image colormap (e.g. 'gray' -> 'gray_r'). Default is False.
        invert_z_bands : bool, optional
            Reverse the Z-band colormap (e.g. 'Greys_r' -> 'Greys'). Default is False.
        cmap_image : str, optional
            Colormap of the raw image background. Default is 'gray'.
        cmap_z_bands : str, optional
            Colormap of the Z-band background. Default is 'Greys_r'.
        alpha_image : float, optional
            Opacity of the raw image background. Default is 1.
        alpha_z_bands : float, optional
            Opacity of the Z-band background. Default is 1.
        """
        frame = StructurePlots._frame(sarc_obj, frame)
        grouping = sarc_obj.motion._grouping(kind)
        n_tracks = int(sarc_obj.data.get('motion.tracks.n', 0))
        t = sarc_obj.motion._tracked_frame_index(frame)
        n_groups = grouping['n']

        pos = np.asarray(sarc_obj.data['motion.tracks.positions_px'], dtype=float).reshape(n_tracks, -1, 2)
        gid = grouping['track_group_id']
        yx = pos[:, t]
        finite = np.isfinite(yx[:, 0]) & np.isfinite(yx[:, 1])

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        if show_dropped:
            drop = finite & (gid < 0)
            ax.scatter(yx[drop, 1], yx[drop, 0], c=dropped_color, s=s * 0.6, edgecolors='none', label='dropped')

        assigned = finite & (gid >= 0)
        cm = plt.get_cmap(cmap)
        colors = cm(np.linspace(0, 1, max(n_groups, 1)))
        ax.scatter(yx[assigned, 1], yx[assigned, 0], c=colors[gid[assigned]], s=s, edgecolors='none')

        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title if title is not None else f"Track groups ('{kind}', n={n_groups})",
                     fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_track_myofibrils(ax: Axes, sarc_obj: SarcAsM, frame: int = 0, kind: str = 'myofibril',
                              color_by: str = 'group', cmap: str = 'gist_rainbow',
                              linewidth: float = 1.5, show_points: bool = True, markersize: float = 6,
                              only_observed: bool = False,
                              scalebar: bool = True, colorbar: bool = False, title: Optional[str] = None,
                              show_image: bool = False, show_z_bands: bool = False,
                              invert_image: bool = False, invert_z_bands: bool = False,
                              cmap_image: str = 'gray', cmap_z_bands: str = 'Greys_r',
                              alpha_image: float = 1, alpha_z_bands: float = 1):
        """
        Draw each tracked myofibril as a connected polyline over the image.

        The tracker analogue of :meth:`plot_myofibril_lines`: each group's member
        sarcomeres — ordered head-to-tail by ``track_group_order`` — are linked by a
        line at ``frame``, so you can see where each analyzed group runs and its
        shape. Requires a ``'myofibril'`` grouping
        (:meth:`MotionAnalysis.group_tracks` / :meth:`MotionAnalysis.analyze`
        with ``by='myofibril'``).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            SarcAsM with a myofibril grouping.
        frame : int, optional
            Movie frame to draw. Default is 0.
        kind : {'myofibril', 'loi'}, optional
            The ordered grouping to draw. Default is 'myofibril'.
        color_by : {'group', 'slen', 'beating_rate'}, optional
            Per-group colour: group id, the line's median sarcomere length at this
            frame, or its beating rate (requires ``sarc.motion.analyze(by='myofibril')``).
            Default is 'group'.
        cmap : str, optional
            Colormap. Default is 'gist_rainbow'.
        linewidth : float, optional
            Width of the group polylines. Default is 1.5.
        show_points : bool, optional
            Whether to mark the member sarcomeres as points. Default is True.
        markersize : float, optional
            Size of the member sarcomere markers. Default is 6.
        only_observed : bool, optional
            Only draw members that were actually observed at this frame. Default is False.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        colorbar : bool, optional
            Whether to add a colorbar to the plot (only for metric colourings). Default is False.
        title : str, optional
            The title for the plot. If None, a default title is used. Default is None.
        show_image : bool, optional
            Whether to show the raw microscopy image as background. Default is False.
        show_z_bands : bool, optional
            Whether to show the Z-band mask as background. Mutually exclusive with
            ``show_image``. Default is False.
        invert_image : bool, optional
            Reverse the raw-image colormap (e.g. 'gray' -> 'gray_r'). Default is False.
        invert_z_bands : bool, optional
            Reverse the Z-band colormap (e.g. 'Greys_r' -> 'Greys'). Default is False.
        cmap_image : str, optional
            Colormap of the raw image background. Default is 'gray'.
        cmap_z_bands : str, optional
            Colormap of the Z-band background. Default is 'Greys_r'.
        alpha_image : float, optional
            Opacity of the raw image background. Default is 1.
        alpha_z_bands : float, optional
            Opacity of the Z-band background. Default is 1.
        """
        frame = StructurePlots._frame(sarc_obj, frame)
        if kind not in sarc_obj.motion._CHAIN_LEVELS:
            raise ValueError(f"plot_track_myofibrils draws an ordered grouping, {sarc_obj.motion._CHAIN_LEVELS}, "
                             f"got '{kind}'.")
        grouping = sarc_obj.motion._grouping(kind)
        n_tracks = int(sarc_obj.data['motion.tracks.n'])
        t = sarc_obj.motion._tracked_frame_index(frame)
        n_groups = grouping['n']

        pos = np.asarray(sarc_obj.data['motion.tracks.positions_px'], dtype=float).reshape(n_tracks, -1, 2)[:, t]
        gid = grouping['track_group_id']
        order = grouping['track_group_order']
        observed = np.asarray(sarc_obj.data['motion.tracks.observed']).reshape(n_tracks, -1)[:, t].astype(bool)
        slen_t = np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)[:, t]

        # Per-group colour value.
        metric = None
        clabel = ''
        if color_by == 'beating_rate':
            metric = np.asarray(sarc_obj.data.get(f'motion.{kind}.beating_rate', np.full(n_groups, np.nan)), dtype=float)
            clabel = 'Beating rate [Hz]'
        elif color_by == 'slen':
            with np.errstate(invalid='ignore'):
                metric = np.array([np.nanmedian(slen_t[gid == g]) if np.any(gid == g) else np.nan
                                   for g in range(n_groups)])
            clabel = 'Sarcomere length [µm]'

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        cm = plt.get_cmap(cmap)
        norm = None
        if metric is not None and np.isfinite(metric).any():
            finite = metric[np.isfinite(metric)]
            norm = plt.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

        for g in range(n_groups):
            members = np.flatnonzero(gid == g)
            if members.size < 2:
                continue
            members = members[np.argsort(order[members])]  # head -> tail
            p = pos[members]
            keep = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1])
            if only_observed:
                keep &= observed[members]
            if keep.sum() < 2:
                continue
            p = p[keep]
            if metric is not None:
                c = cm(norm(metric[g])) if (norm is not None and np.isfinite(metric[g])) else 'lightgrey'
            else:
                c = cm(g / max(n_groups - 1, 1))
            ax.plot(p[:, 1], p[:, 0], '-', color=c, lw=linewidth, alpha=0.9)
            if show_points:
                ax.scatter(p[:, 1], p[:, 0], color=c, s=markersize, edgecolors='none', zorder=3)

        if colorbar and norm is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
            sm.set_array([])
            cb = ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(clabel, fontsize=PlotUtils.fontsize - 1)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title if title is not None else f'Tracked myofibrils (n={n_groups})',
                     fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_track_coverage_map(ax: Axes, sarc_obj: SarcAsM, frame: int = 0,
                                cmap: str = 'viridis', linewidth: float = 0.8,
                                max_tracks: Optional[int] = None, alpha: float = 0.9,
                                clim: Tuple[float, float] = (0, 100),
                                scalebar: bool = True, colorbar: bool = True,
                                title: Optional[str] = None, show_image: bool = True,
                                invert_image: bool = True, cmap_image: str = 'gray',
                                alpha_image: float = 1):
        """
        Tracking control plot: where the complete tracks are.

        Draws every track's trajectory over the (inverted) raw image, coloured by
        its completeness — the percentage of tracked frames in which the sarcomere
        was matched to a real detection, 100 % meaning it was observed in every
        frame. The colorbar is the legend. Together with
        :meth:`plot_track_coverage_histogram` (same measure, as a distribution)
        this is the standard QC pair for a tracking run: the histogram says *how
        much* of the field tracks well, this map says *where*.

        A track observed in fewer than two frames has no line to draw and is
        omitted here, but still appears in the histogram.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            SarcAsM with :meth:`MotionAnalysis.track` results.
        frame : int, optional
            Movie frame used for the **background** image only (the trajectories
            always span the full movie). Default is 0.
        cmap : str, optional
            Colormap for the coverage colouring. Default is 'viridis'.
        linewidth : float, optional
            Width of the trajectory lines. Default is 0.8.
        max_tracks : int or None, optional
            Draw at most this many tracks (longest-coverage first). None draws all,
            which is what a control plot should do. Default is None.
        alpha : float, optional
            Line opacity. Default is 0.9.
        clim : tuple of float, optional
            Colour limits in percent. Fixed at (0, 100) so the colours mean the same
            thing across recordings. Default is (0, 100).
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        colorbar : bool, optional
            Whether to add the coverage colorbar (the legend). Default is True.
        title : str, optional
            The title for the plot. If None, a default title is used. Default is None.
        show_image : bool, optional
            Whether to show the raw microscopy image as background. Default is True.
        invert_image : bool, optional
            Reverse the raw-image colormap, so the trajectories sit on a light
            background. Default is True.
        cmap_image : str, optional
            Colormap of the raw image background. Default is 'gray'.
        alpha_image : float, optional
            Opacity of the raw image background. Default is 1.
        """
        frame = StructurePlots._frame(sarc_obj, frame)
        n_tracks = int(sarc_obj.data.get('motion.tracks.n', 0)) if 'motion.tracks.n_frames' in sarc_obj.data else 0
        TrackPlots.plot_tracks(ax, sarc_obj, frame=frame, color_by='coverage', cmap=cmap,
                          linewidth=linewidth, only_observed=True, max_tracks=max_tracks,
                          alpha=alpha, clim=clim, scalebar=scalebar, colorbar=colorbar,
                          title=title if title is not None else f'Track coverage (n={n_tracks})',
                          show_image=show_image, invert_image=invert_image,
                          cmap_image=cmap_image, alpha_image=alpha_image)

    @staticmethod
    def plot_track_coverage_histogram(ax: Axes, sarc_obj: SarcAsM, bins: int = 25,
                                      cmap: Optional[str] = 'viridis', color: str = 'darkslategray',
                                      clim: Tuple[float, float] = (0, 100),
                                      edge_color: str = 'k', rwidth: float = 0.9,
                                      density: bool = False, show_median: bool = True,
                                      title: Optional[str] = None):
        """
        Tracking control plot: distribution of track completeness.

        Histogram of the same per-track measure that
        :meth:`plot_track_coverage_map` colours the trajectories by — the
        percentage of tracked frames with a real detection. All tracks are
        counted, including single-frame fragments. A run dominated by a peak at
        100 % tracked the field through the whole movie; a heavy left tail means
        the tracks fragment (cross-check ``fragmentation_ratio``).

        By default the bars are filled from the same colormap over the same 0-100 %
        range as the map, so one colorbar reads as the legend for both panels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the histogram on.
        sarc_obj : SarcAsM
            SarcAsM with :meth:`MotionAnalysis.track` results.
        bins : int, optional
            Number of histogram bins across ``clim``. Default is 25.
        cmap : str or None, optional
            Colormap used to fill each bar according to its coverage, matching the
            map panel. None falls back to the flat ``color``. Default is 'viridis'.
        color : str, optional
            Flat bar colour, used when ``cmap`` is None. Default is 'darkslategray'.
        clim : tuple of float, optional
            Histogram range in percent, and the colour range when ``cmap`` is set.
            Default is (0, 100).
        edge_color : str, optional
            The color of the edges of the histogram bars. Default is 'k'.
        rwidth : float, optional
            The relative width of the histogram bars. Default is 0.9.
        density : bool, optional
            If True, normalize to a probability density rather than raw counts.
            Default is False.
        show_median : bool, optional
            Whether to mark the median coverage with a dashed line and label.
            Default is True.
        title : str, optional
            The title for the plot. Default is None.
        """
        coverage = TrackPlots._track_coverage(sarc_obj)

        counts, edges, patches = ax.hist(coverage, bins=bins, range=clim, density=density,
                                         rwidth=rwidth, color=color, edgecolor=edge_color)
        if cmap is not None:
            cm = plt.get_cmap(cmap)
            norm = plt.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
            centers = 0.5 * (edges[:-1] + edges[1:])
            for patch, center in zip(patches, centers):
                patch.set_facecolor(cm(norm(center)))

        if show_median and coverage.size:
            med = float(np.median(coverage))
            ax.axvline(med, color='k', ls='--', lw=1)
            # Label on whichever side of the line has room, so a median near
            # either end of the 0-100 % range stays inside the axes.
            right = med > 0.5 * (clim[0] + clim[1])
            ax.annotate(f'median {med:.0f}%', xy=(med, 1), xycoords=('data', 'axes fraction'),
                        xytext=(-4 if right else 4, -3), textcoords='offset points',
                        ha='right' if right else 'left', va='top',
                        fontsize=PlotUtils.fontsize - 1)

        ax.set_xlim(*clim)
        ax.set_xlabel('Track coverage [%]')
        ax.set_ylabel('Frequency' if density else 'Number of tracks')
        if title is not None:
            ax.set_title(title, fontsize=PlotUtils.fontsize)
        PlotUtils.remove_spines(ax)

    @staticmethod
    def plot_track_raster(ax: Axes, sarc_obj: SarcAsM, value: str = 'delta_slen',
                          cycle_average: bool = True, sort_by: str = 'time_to_peak',
                          kind: Optional[str] = None, clim: Optional[Tuple[float, float]] = None,
                          cmap: Optional[str] = None, min_coverage: float = 0.5,
                          show_peak: bool = True, show_contr: bool = True, colorbar: bool = True,
                          title: Optional[str] = None):
        """
        Raster of every tracked sarcomere's time series: one row per sarcomere, time along x.

        With ``cycle_average=True`` (default) each sarcomere's trace is averaged over the
        pooled contraction cycles (onset to next onset), so one beat is shown per row and
        the rows can be ordered by their time-to-peak or amplitude — the timing and
        amplitude heterogeneity across the cell then reads directly from the picture,
        with the peak-time curve drawn on top. With ``cycle_average=False`` the full
        recording is shown with rows ordered by group, which is the view for QC
        (dropouts, drifting rows, which cycles were detected).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        sarc_obj : SarcAsM
            SarcAsM with :meth:`MotionAnalysis.track` results; the cycle
            average needs ``sarc.motion.analyze(by='pool')``.
        value : {'delta_slen', 'slen', 'vel'}, optional
            Length change from equilibrium (µm), length (µm) or velocity (µm/s).
            Default is 'delta_slen'.
        cycle_average : bool, optional
            Average each row over the pooled contraction cycles. Default is True.
        sort_by : {'time_to_peak', 'amplitude', 'group'}, optional
            Row order. ``'time_to_peak'`` / ``'amplitude'`` use the cycle average of
            ``delta_slen`` (earliest / largest shortening first); ``'group'`` orders by
            group id and, for the ordered groupings, head-to-tail. Default is 'time_to_peak'
            (``'group'`` when ``cycle_average`` is False).
        kind : str or None, optional
            Grouping whose labels are used (tracks outside its groups are left out, and
            ``sort_by='group'`` sorts by it); None uses no grouping. Default is None.
        clim : tuple of float or None, optional
            Colour limits. None: symmetric at the 99th percentile of |value| for
            ``delta_slen`` / ``vel``, the 1st-99th percentiles for ``slen``. Default is None.
        cmap : str or None, optional
            Colormap; None picks a diverging map for ``delta_slen`` / ``vel`` and
            'viridis' for ``slen``. Default is None.
        min_coverage : float, optional
            Only tracks observed in at least this fraction of frames are shown.
            Default is 0.5.
        show_peak : bool, optional
            Draw the time-to-peak curve when rows are sorted by it. Default is True.
        show_contr : bool, optional
            Mark the pooled contraction onsets (full recording only). Default is True.
        colorbar : bool, optional
            Whether to add a colorbar. Default is True.
        title : str or None, optional
            Axes title. Default is None.

        Returns
        -------
        matplotlib.image.AxesImage or None
            The raster, or None when there is nothing to draw.
        """
        if value not in ('delta_slen', 'slen', 'vel'):
            raise ValueError(f"value must be 'delta_slen', 'slen' or 'vel', got {value!r}")
        if sort_by not in ('time_to_peak', 'amplitude', 'group'):
            raise ValueError(f"sort_by must be 'time_to_peak', 'amplitude' or 'group', got {sort_by!r}")
        kin = sarc_obj.motion.kinematics()
        series = np.asarray(kin[value], dtype=float)
        n_tracks, T = series.shape
        frametime = sarc_obj.metadata.frametime
        keep = kin['coverage'] >= min_coverage

        gid = order = None
        if kind is not None:
            grouping = sarc_obj.motion._grouping(kind)
            gid, order = grouping['track_group_id'][:n_tracks], grouping['track_group_order'][:n_tracks]
            keep &= gid >= 0
        elif sort_by == 'group':
            raise ValueError("sort_by='group' needs the grouping, e.g. kind='myofibril'.")

        peak_frame = None
        if cycle_average:
            labels = sarc_obj.data.get('motion.pool.labels_contr')
            if labels is None:
                raise ValueError("cycle_average needs the pooled contraction cycles: "
                                 "run sarc.motion.analyze(by='pool') first.")
            onsets, _ = heterogeneity.cycle_windows(np.asarray(labels).reshape(-1))
            if onsets.size < 2:
                raise ValueError('cycle_average needs at least two complete contraction cycles.')
            win = int(np.median(np.diff(onsets)))
            starts = [o for o in onsets if o + win <= T]
            if not starts:
                raise ValueError('No complete contraction cycle inside the recording.')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                data = np.nanmean(np.stack([series[:, o:o + win] for o in starts]), axis=0)
                dsl = np.nanmean(np.stack([np.asarray(kin['delta_slen'])[:, o:o + win] for o in starts]), axis=0)
            keep &= np.isfinite(dsl).all(axis=1)
            time = np.arange(win) * frametime
        else:
            data, dsl, time = series, np.asarray(kin['delta_slen']), np.arange(T) * frametime
            if sort_by != 'group':
                sort_by = 'group'
        rows = np.flatnonzero(keep)
        if rows.size == 0:
            ax.text(0.5, 0.5, 'No tracks', ha='center', va='center', transform=ax.transAxes)
            return None

        if sort_by == 'time_to_peak':
            peak_frame = np.nanargmin(dsl[rows], axis=1)
            rows = rows[np.argsort(peak_frame, kind='stable')]
            peak_frame = np.sort(peak_frame)
        elif sort_by == 'amplitude':
            amp = -np.nanmin(dsl[rows], axis=1)
            rows = rows[np.argsort(-amp, kind='stable')]
        elif gid is not None:
            pos0 = np.asarray(sarc_obj.data['motion.tracks.positions_um'], dtype=float).reshape(n_tracks, T, 2)
            start = np.asarray(sarc_obj.data.get('motion.tracks.start_frame', np.zeros(n_tracks, int))).astype(int)
            yx = pos0[np.arange(n_tracks), start]
            rows = rows[np.lexsort((yx[rows, 1], yx[rows, 0], order[rows], gid[rows]))]

        img = np.ma.masked_invalid(data[rows])
        if clim is None:
            finite = img.compressed()
            if value == 'slen':
                clim = tuple(np.percentile(finite, [1, 99])) if finite.size else (1.4, 2.2)
            else:
                lim = float(np.percentile(np.abs(finite), 99)) if finite.size else 0.2
                clim = (-lim, lim)
        if cmap is None:
            cmap = {'delta_slen': 'RdBu_r', 'vel': 'PuOr_r', 'slen': 'viridis'}[value]
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad('white')
        im = ax.imshow(img, aspect='auto', cmap=cm, vmin=clim[0], vmax=clim[1],
                       extent=(0, time[-1] + frametime, rows.size, 0), interpolation='nearest')

        if show_peak and sort_by == 'time_to_peak' and peak_frame is not None:
            ax.plot(peak_frame * frametime, np.arange(rows.size) + 0.5, c='k', lw=0.6)
        if sort_by == 'group' and gid is not None:
            g_sorted = gid[rows]
            for b in np.flatnonzero(np.diff(g_sorted)) + 1:
                ax.axhline(b, color='k', lw=0.4)
            groups = np.unique(g_sorted)
            if groups.size <= 40:
                ax.set_yticks([np.mean(np.flatnonzero(g_sorted == g)) + 0.5 for g in groups])
                ax.set_yticklabels([str(int(g)) for g in groups], fontsize='x-small')
                ax.set_ylabel(f'Group ({kind})')
            else:
                ax.set_ylabel(f'Sarcomere (by {kind})')
        else:
            ax.set_ylabel({'time_to_peak': 'Sarcomere (sorted by time to peak)',
                           'amplitude': 'Sarcomere (sorted by amplitude)'}.get(sort_by, 'Sarcomere'))
        if show_contr and not cycle_average and kin['contr'].any():
            for f in np.flatnonzero(np.diff(np.r_[0, kin['contr'].astype(int)]) == 1):
                ax.axvline(f * frametime, color='k', lw=0.5, ls=':')
        ax.set_xlabel('Time in cycle [s]' if cycle_average else _LABEL_TIME)
        if colorbar:
            label = {'delta_slen': _LABEL_DELTA_SL, 'slen': _LABEL_SL, 'vel': 'V [µm/s]'}[value]
            plt.colorbar(im, ax=ax, label=label, fraction=0.04, pad=0.02)
        if title is not None:
            ax.set_title(title, fontsize=PlotUtils.fontsize)
        return im
