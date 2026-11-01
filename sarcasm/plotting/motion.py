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

"""Plots of the motion of the groups: all groups of a kind over time, and one group (a ``Group``)."""

import numbers
import os.path
import warnings
from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt, transforms
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from sarcasm.features import MOTION_KIND_LABELS
from sarcasm.plotting.plot_utils import PlotUtils
from sarcasm.core import SarcAsM
from sarcasm.analysis import grouped_motion
from sarcasm.plotting.structure import StructurePlots
from sarcasm.plotting.structure import _LABEL_DELTA_SL, _LABEL_SL, _LABEL_TIME

if TYPE_CHECKING:
    from sarcasm.motion.groups import Group


class MotionPlots:
    """Plots of the motion of the groups: all groups of a kind, and one group. Part of :class:`~sarcasm.plotting.plots.Plots`."""

    # ------------------------------------------------------------------
    # One group (sarc.motion.groups(kind)[i]): its members over time
    # ------------------------------------------------------------------

    @staticmethod
    def _series(g: "Group") -> dict:
        """Time axis, contraction mask, cycle labels and cycle onsets (frames) of a group."""
        T = int(np.asarray(g.tracks.slen).shape[1])
        time = np.arange(T) * g.sarc.metadata.frametime
        contr = np.asarray(g.contr, dtype=bool)
        labels = np.asarray(g.labels_contr)
        n = int(labels.max()) if labels.size else 0
        complete = np.asarray(g.contr_complete)[:n] == 1
        incomplete = np.zeros(T, dtype=bool)
        for k in np.flatnonzero(~complete):
            incomplete |= labels == k + 1
        onsets = np.array([np.flatnonzero(labels == k + 1)[0] for k in range(n)], dtype=int)
        return {'time': time, 'contr': contr, 'incomplete': incomplete, 'onsets': onsets}

    @staticmethod
    def _shade_contr(ax: Axes, g: "Group", t_offset: float = 0.0, color: str = 'lavender',
                     alpha_incomplete: float = 0.45) -> None:
        """Shade the group's contractions on ``ax``; cycles cut by the recording's edges are fainter."""
        s = MotionPlots._series(g)
        if not s['contr'].any():
            return
        time = s['time'] - t_offset
        blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(time, 0, 1, where=s['contr'] & ~s['incomplete'], color=color, transform=blended,
                        linewidth=0)
        if s['incomplete'].any():
            ax.fill_between(time, 0, 1, where=s['incomplete'], color=color, alpha=alpha_incomplete,
                            transform=blended, linewidth=0)

    @staticmethod
    def _contr_window(g: "Group", number_contr, t_lim):
        """Frame window of a view centred on contraction ``number_contr``, clamped to the recording.

        Returns the absolute time limits and the frame slice bounds ``(i0, i1)``, or
        ``((None, None), None, None)`` when ``number_contr`` is None or the group has no contraction.
        """
        s = MotionPlots._series(g)
        if number_contr is None or s['onsets'].size == 0:
            return (None, None), None, None
        frametime = g.sarc.metadata.frametime
        n = s['time'].size
        start = s['onsets'][number_contr] * frametime
        lo = 0.0 if t_lim[0] is None else start + t_lim[0]
        hi = n * frametime if t_lim[1] is None else start + t_lim[1]
        i0 = min(max(0, int(lo / frametime)), n)
        i1 = min(max(i0, int(hi / frametime)), n)
        return (lo, hi), i0, i1

    @staticmethod
    def plot_group_summary(g: "Group", number_contr=0, t_lim=(0, None), t_lim_overlay=(-0.1, 2.9),
                           file_path=None):
        """
        Summary figure of one group: where its members are, their Z-bands (ordered kinds) or lengths,
        their ΔSL, and their ΔSL and velocity over one contraction.

        Parameters
        ----------
        g : Group
            The group, ``sarc.motion.groups(kind)[i]``.
        number_contr : int, optional
            The contraction of the overlay panels. Default is 0.
        t_lim : tuple of float, optional
            The time limits in seconds; ``None`` on either side means the recording boundary.
            Default is (0, None).
        t_lim_overlay : tuple of float, optional
            The time limits of the overlay panels around the contraction onset. Default is (-0.1, 2.9).
        file_path : str, optional
            Save the figure here. Default is None (not saved).

        Returns
        -------
        matplotlib.figure.Figure
        """
        mosaic = """
        aaaccc
        bbbccc
        dddeee
        dddfff
        """
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(PlotUtils.width_2cols, PlotUtils.width_2cols),
                                      constrained_layout=True)
        fig.suptitle(f'{os.path.basename(g.sarc.file_path)}: {MOTION_KIND_LABELS.get(g.kind, g.kind)} {g.id} '
                     f'({g.n_members} sarcomeres)', fontsize=PlotUtils.fontsize)
        ref = int(g.sarc.data.get(f'params.motion.group_tracks.{g.kind}.reference_frame', 0))
        StructurePlots.plot_image(axs['a'], g.sarc, frame=ref)
        StructurePlots.plot_z_bands(axs['b'], g.sarc, frame=ref)
        t = g.sarc.motion._tracked_frame_index(ref)
        yx = np.asarray(g.tracks.positions_px, dtype=float)[:, t]
        for key in ('a', 'b'):
            axs[key].plot(yx[:, 1], yx[:, 0], '-' if g.ordered else '.', c='darkorange', lw=1.2, ms=2)
        if g.ordered:
            MotionPlots.plot_z_pos(axs['c'], g, t_lim=t_lim)
        else:
            MotionPlots.plot_slen(axs['c'], g, t_lim=t_lim)
        MotionPlots.plot_delta_slen(axs['d'], g, stacked=True, t_lim=t_lim)
        MotionPlots.plot_overlay_delta_slen(axs['e'], g, number_contr=number_contr, t_lim=t_lim_overlay)
        MotionPlots.plot_overlay_velocity(axs['f'], g, number_contr=number_contr, t_lim=t_lim_overlay)
        PlotUtils.label_all_panels(axs)
        if file_path is not None:
            fig.savefig(file_path, dpi=PlotUtils.dpi)
        return fig

    @staticmethod
    def plot_z_pos(ax: Axes, g: "Group", number_contr=None, show_contr=True, color='k',
                   t_lim=(0, None), y_lim=(None, None)):
        """
        Plot the Z-band positions along the line of an ordered group (myofibril or loi) over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        g : Group
            An ordered group.
        number_contr : int, optional
            Centre the plot on this contraction; None shows the whole recording. Default is None.
        show_contr : bool, optional
            Whether to shade the contractions. Default is True.
        color : str, optional
            The colour of the trajectories. Default is 'k'.
        t_lim : tuple, optional
            The time limits in seconds; ``None`` on either side means the recording boundary.
            Default is (0, None).
        y_lim : tuple, optional
            The y-axis limits. Default is (None, None).
        """
        tlim, i0, i1 = MotionPlots._contr_window(g, number_contr, t_lim)
        centered = i0 is not None
        time = MotionPlots._series(g)['time']
        z_pos = g.z_pos
        if show_contr:
            MotionPlots._shade_contr(ax, g, t_offset=tlim[0] if centered else 0.0)
        if centered:
            seg = z_pos[:, i0:i1].T
            ax.plot(time[:len(seg)], seg, linewidth=0.75, c=color)
            ax.set_xlim(0, tlim[1] - tlim[0])
        else:
            ax.plot(time, z_pos.T, linewidth=0.75, c=color)
            ax.set_xlim(t_lim)
        ax.set_xlabel(_LABEL_TIME)
        ax.set_ylabel('Z-band position Z(t) [µm]')
        ax.set_ylim((0, None) if y_lim == (None, None) else y_lim)
        PlotUtils.polish_yticks(ax, 5, 2.5)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def _plot_delta_slen_stacked(ax: Axes, g: "Group", frame=None, t_lim=(0, None), y_lim=(-0.3, 0.4), n_rows=6,
                                 n_start=0, show_contr=True):
        """ΔSL of the members, one inset row per sarcomere (head to tail for the ordered kinds)."""
        yticks = [-0.2, 0, 0.2]
        delta_slen = g.tracks.delta_slen
        time = MotionPlots._series(g)['time']
        n_sarc = len(delta_slen)
        if n_start >= n_sarc:
            n_start = 0
        n_rows = min(n_rows, n_sarc - n_start)
        if n_rows <= 0:
            ax.text(0.5, 0.5, 'No sarcomeres', ha='center', va='center', transform=ax.transAxes)
            return
        for i, y in enumerate(np.linspace(0, 1, num=n_rows, endpoint=False)):
            ax_i = ax.inset_axes((0., y, 1, 1 / n_rows - min(0.02, 0.3 / n_rows)))
            ax_i.plot(time, delta_slen[i + n_start], c='k', lw=0.6)
            ax_i.axhline(0, linewidth=1, linestyle=':', c='k')
            if show_contr:
                MotionPlots._shade_contr(ax_i, g)
            if frame is not None:
                ax_i.axvline(time[frame], linestyle='--', c='k')
            ax_i.set_ylim(y_lim)
            ax_i.set_xlim(t_lim)
            if i > 0:
                ax_i.set_xticks([])
            else:
                PlotUtils.polish_xticks(ax_i)
                ax_i.tick_params(axis='x', labelsize='x-small')
            ax_i.set_yticks(yticks)
            ax_i.set_yticklabels(yticks, fontsize='x-small')
        ax.set_xlabel(_LABEL_TIME)
        ax.set_ylabel(_LABEL_DELTA_SL)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.xaxis.label.set_color('k')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')

    @staticmethod
    def _plot_members_over_cycle(ax: Axes, g: "Group", values: np.ndarray, number_contr, t_lim, show_contr):
        """Members' traces (coloured) and their mean (black), whole recording or one contraction."""
        tlim, i0, i1 = MotionPlots._contr_window(g, number_contr, t_lim)
        centered = i0 is not None
        time = MotionPlots._series(g)['time']
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nanmean(values, axis=0)
        if show_contr:
            MotionPlots._shade_contr(ax, g, t_offset=tlim[0] if centered else 0.0)
        ax.set_prop_cycle('color', list(plt.cm.nipy_spectral(np.linspace(0, 1, max(len(values), 1)))))
        if centered:
            seg = values.T[i0:i1]
            ax.plot(time[:len(seg)], seg, linewidth=0.5)
            ax.plot(time[:len(seg)], mean[i0:i1], c='k', linewidth=2)
            ax.set_xlim(0, tlim[1] - tlim[0])
        else:
            ax.plot(time, values.T, linewidth=0.5)
            ax.plot(time, mean, c='k', linewidth=2)
            ax.set_xlim(t_lim)
        ax.set_xlabel(_LABEL_TIME)

    @staticmethod
    def plot_overlay_delta_slen(ax: Axes, g: "Group", number_contr=None, t_lim=(0, None), y_lim=(-0.35, 0.5),
                                show_contr=True):
        """
        Overlay the members' ΔSL (coloured) and their mean (black).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        g : Group
            The group.
        number_contr : int, optional
            Centre on this contraction; None shows the whole recording. Default is None.
        t_lim : tuple, optional
            The time limits in seconds (around the contraction onset when centred). Default is (0, None).
        y_lim : tuple, optional
            The y-axis limits. Default is (-0.35, 0.5).
        show_contr : bool, optional
            Whether to shade the contractions. Default is True.
        """
        MotionPlots._plot_members_over_cycle(ax, g, g.tracks.delta_slen, number_contr, t_lim, show_contr)
        ax.set_ylabel(_LABEL_DELTA_SL)
        ax.set_ylim(y_lim)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_overlay_velocity(ax: Axes, g: "Group", number_contr=None, t_lim=(0, None), y_lim=(-9, 12),
                              show_contr=True):
        """
        Overlay the members' velocity (coloured) and their mean (black).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        g : Group
            The group.
        number_contr : int, optional
            Centre on this contraction; None shows the whole recording. Default is None.
        t_lim : tuple, optional
            The time limits in seconds (around the contraction onset when centred). Default is (0, None).
        y_lim : tuple, optional
            The y-axis limits. Default is (-9, 12).
        show_contr : bool, optional
            Whether to shade the contractions. Default is True.
        """
        MotionPlots._plot_members_over_cycle(ax, g, g.tracks.vel, number_contr, t_lim, show_contr)
        ax.set_ylabel('V [µm/s]')
        ax.set_ylim(y_lim)
        PlotUtils.polish_yticks(ax, 3, 1)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_phase_space(ax: Axes, g: "Group", t_lim=(0, None), number_contr=None, frame=None):
        """
        Plot the members' length change against their velocity (phase space), and the mean.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        g : Group
            The group.
        t_lim : tuple, optional
            The time limits in seconds; ``None`` on either side means the recording boundary.
            Default is (0, None).
        number_contr : int, optional
            One contraction; None overlays the whole recording. Default is None.
        frame : int, optional
            Mark the members at this frame. Default is None.
        """
        delta_slen, vel = g.tracks.delta_slen, g.tracks.vel
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            delta_avg, vel_avg = np.nanmean(delta_slen, axis=0), np.nanmean(vel, axis=0)
        cm = plt.cm.nipy_spectral(np.linspace(0, 1, max(len(delta_slen), 1)))
        _, i0, i1 = MotionPlots._contr_window(g, number_contr, t_lim)
        for i, (vel_i, delta_i) in enumerate(zip(vel, delta_slen)):
            ax.plot(vel_i[i0:i1], delta_i[i0:i1], c='r', alpha=0.35, lw=0.36, zorder=1)
            if isinstance(frame, numbers.Integral):
                ax.scatter(vel_i[frame], delta_i[frame], c=[cm[i]], s=10, zorder=2)
        ax.plot(vel_avg[i0:i1], delta_avg[i0:i1], c='k', lw=1, label='Average')
        legend_elements = [Line2D([0], [0], color='k', lw=2), Line2D([0], [0], color='r', alpha=0.35, lw=0.5)]
        ax.legend(legend_elements, ['Average', 'Individual'], loc='upper right')
        PlotUtils.polish_xticks(ax, 5, 2.5)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        ax.set_xlabel('Velocity $V$ [µm/s]', fontsize=PlotUtils.fontsize)
        ax.set_ylabel(r'Length change $\Delta SL$ [µm]', fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_popping_events(g: "Group", save_name=None):
        """
        Map which member sarcomere pops in which contraction cycle (black = popped), with the fraction of
        members popping per cycle (top) and the fraction of cycles each member pops in (right).

        Popping: lengthening more than ``thres_popping`` beyond the resting length during a contraction
        (``sarc.motion.analyze``). Incomplete cycles are blank.

        Parameters
        ----------
        g : Group
            The group (analysed).
        save_name : str, optional
            File path to save the figure. Default is None (not saved).

        Returns
        -------
        matplotlib.figure.Figure
        """
        events = np.asarray(g.track_popping, dtype=float)
        n_cycles = int(np.asarray(g.n_contr))
        events = events[:, :n_cycles]
        rate_cycle = np.asarray(g.popping_rate_contr, dtype=float)[:n_cycles]
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rate_member = np.nanmean(events, axis=1) if n_cycles else np.zeros(len(events))

        left, width, bottom, height, spacing = 0.1, 0.65, 0.1, 0.65, 0.02
        fig = plt.figure(figsize=(PlotUtils.width_1cols * 0.9, 3.))
        ax = fig.add_axes((left, bottom, width, height))
        ax_histx = fig.add_axes((left, bottom + height + spacing, width, 0.2), sharex=ax)
        ax_histy = fig.add_axes((left + width + spacing, bottom, 0.2, height), sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        if events.size:
            ax.pcolorfast(np.ma.masked_invalid(events), cmap='Greys', vmin=0, vmax=1)
        ax_histx.bar(np.arange(len(rate_cycle)) + 0.5, np.nan_to_num(rate_cycle), color='k', alpha=0.4)
        ax_histy.barh(np.arange(len(rate_member)) + 0.5, np.nan_to_num(rate_member), color='k', alpha=0.4)
        ax.set_xlabel('Contraction cycle [#]')
        ax.set_ylabel('Sarcomere [#]')
        ax.set_xticks(np.arange(n_cycles) + 0.5, np.arange(n_cycles) + 1)
        step = max(1, int(np.ceil(len(rate_member) / 15)))          # at most ~15 readable labels
        rows = np.arange(0, len(rate_member), step)
        ax.set_yticks(rows + 0.5, rows + 1)
        ax_histx.set_ylabel('$f_c(P)$')
        ax_histy.set_xlabel('$f_s(P)$')
        ax_histy.xaxis.set_major_locator(MaxNLocator(2, prune='lower'))
        ax.set_ylim(0, max(len(rate_member), 1))
        ax.set_xlim(0, max(n_cycles, 1))
        ax.grid(axis='x')
        if save_name is not None:
            fig.savefig(save_name, bbox_inches='tight')
        return fig

    @staticmethod
    def _plot_group_stacked(ax, sarc_obj, kind, matrix, group_contr, ylabel,
                            t_lim, y_lim, n_rows, show_contr, hline='equ', label_offset=0,
                            readable_labels=False):
        """Stacked per-group time-series (shared by plot_slen_mean / plot_delta_slen_mean).

        ``hline`` controls the dotted reference line per row: ``'equ'`` = the
        equilibrium length (median over the non-contracting frames), ``'zero'`` =
        the ΔSL=0 line (which *is* the equilibrium in delta space), ``None`` = no line.
        ``label_offset`` shifts the row labels (1 gives the 1-based numbering the
        domain masks use). ``readable_labels`` names the rows by grouping
        ('Myofibril 3', not 'M3') and keeps y tick labels on the bottom row only.
        """
        matrix = np.asarray(matrix, dtype=float)
        n_groups, n_frames = matrix.shape
        time = np.arange(n_frames) * sarc_obj.metadata.frametime
        if n_rows is None:
            n_rows = n_groups
        n_rows = min(n_rows, n_groups)
        if n_rows == 0:
            ax.text(0.5, 0.5, 'No groups', ha='center', va='center', transform=ax.transAxes)
            return
        y_range = y_lim[1] - y_lim[0]
        y_step = y_range / 4
        yticks = [y_lim[0] + y_step, y_lim[0] + 2 * y_step, y_lim[0] + 3 * y_step]
        cm = plt.cm.gist_rainbow(np.linspace(0, 1, n_groups))
        list_y = np.linspace(0, 1, num=n_rows, endpoint=False)
        for i, y in enumerate(list_y):
            g = n_rows - 1 - i
            if g >= n_groups:
                continue
            ax_i = ax.inset_axes((0., y, 1, 1 / n_rows - min(0.02, 0.3 / n_rows)))
            ax_i.plot(time, matrix[g], c=cm[g], lw=0.8)
            if hline == 'zero':
                ax_i.axhline(0, linewidth=0.5, linestyle=':', c='k')
            elif hline == 'equ':
                c = group_contr[g] if group_contr is not None else np.zeros(n_frames, dtype=bool)
                equ_g = grouped_motion.equilibrium_over_quiet(matrix[g], c)
                if np.isfinite(equ_g):
                    ax_i.axhline(equ_g, linewidth=0.5, linestyle=':', c='k')
            if show_contr and group_contr is not None:
                ax_i.fill_between(time, y_lim[0], y_lim[1], where=group_contr[g], color='lavender', alpha=0.7)
            ax_i.set_ylim(y_lim)
            ax_i.set_xlim(t_lim)
            if i > 0:
                ax_i.set_xticks([])
            else:
                PlotUtils.polish_xticks(ax_i)
                ax_i.tick_params(axis='x', labelsize='x-small')
            if readable_labels:
                # one scale for all rows: lowest and highest tick on the bottom row only
                ax_i.set_yticks([yticks[0], yticks[-1]])
                ax_i.set_yticklabels([f'{yt:.2f}' for yt in (yticks[0], yticks[-1])] if i == 0 else [],
                                     fontsize='x-small')
            else:
                ax_i.set_yticks(yticks)
                ax_i.set_yticklabels([f'{yt:.2f}' for yt in yticks], fontsize='x-small')
            row_label = (f'{MOTION_KIND_LABELS.get(kind, kind)} {g + label_offset}' if readable_labels
                         else f'{kind[0].upper()}{g + label_offset}')
            ax_i.text(0.02, 0.85, row_label, transform=ax_i.transAxes,
                      fontsize='x-small', fontweight='bold', color=cm[g])
        ax.set_xlabel(_LABEL_TIME)
        ax.set_ylabel(ylabel)
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.xaxis.label.set_color('k')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')

    @staticmethod
    def _resolve_kind(sarc_obj, kind):
        """Validate that the ``kind`` grouping is analysed and current."""
        if kind is None:
            raise ValueError("Name the grouping to plot, e.g. kind='myofibril'.")
        sarc_obj.motion._assert_track_motion_fresh(kind)
        if f'motion.{kind}.slen_avg' not in sarc_obj.data:
            raise ValueError(f"No '{kind}' track-motion results found. "
                             f"Run sarc.motion.analyze(by='{kind}') first.")
        return kind

    @staticmethod
    def plot_slen_mean(ax: Axes, sarc_obj: SarcAsM, kind: str,
                       t_lim: Tuple[Optional[float], Optional[float]] = (0, None), y_lim: Tuple[float, float] = (1.3, 2.0),
                       n_rows: Optional[int] = None, show_contr: bool = True,
                       use_median: bool = False, readable_labels: bool = False):
        """
        Plot stacked per-group MEAN sarcomere-length time-series from :meth:`MotionAnalysis.analyze`.

        One row per group, each showing that group's aggregated (mean, or median if
        ``use_median``) sarcomere length over time — the signal the contraction
        engine analyses. For ``by='pool'`` this is a single whole-cell trace. See
        :meth:`plot_slen` for the individual member sarcomeres.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The SarcAsM object with track motion analysis results.
        kind : str
            The grouping to plot ('pool', 'mband', 'myofibril', ...).
        t_lim : tuple of float, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple of float, optional
            The y-axis limits for sarcomere length in µm. Default is (1.3, 2.0).
        n_rows : int, optional
            Number of groups to display. If None, all groups are shown. Default is None.
        show_contr : bool, optional
            Whether to shade contraction periods. Default is True.
        use_median : bool, optional
            If True, use median sarcomere length instead of mean. Default is False.
        readable_labels : bool, optional
            Label rows with the grouping name ('Myofibril 3' instead of 'M3') and put
            y tick labels only on the bottom row, for many rows. Default is False.
        """
        kind = MotionPlots._resolve_kind(sarc_obj, kind)
        key = f'motion.{kind}.slen_median' if use_median else f'motion.{kind}.slen_avg'
        matrix = np.asarray(sarc_obj.data[key])
        group_contr = sarc_obj.data.get(f'motion.{kind}.contr', None)
        MotionPlots._plot_group_stacked(ax, sarc_obj, kind, matrix, group_contr, _LABEL_SL,
                                  t_lim, y_lim, n_rows, show_contr, readable_labels=readable_labels)

    @staticmethod
    def plot_delta_slen_mean(ax: Axes, sarc_obj: SarcAsM, kind: str,
                             t_lim: Tuple[Optional[float], Optional[float]] = (0, None), y_lim: Tuple[float, float] = (-0.4, 0.4),
                             n_rows: Optional[int] = None, show_contr: bool = True,
                             use_median: bool = False, readable_labels: bool = False):
        """
        Plot stacked per-group MEAN sarcomere-length *change* (ΔSL) time-series.

        Like :meth:`plot_slen_mean` but plotting ΔSL(t) = SL(t) − equ, where the
        equilibrium ``equ`` is the median group length over the non-contracting
        frames (``<kind>_contr == 0``), matching the LOI ``delta_slen``.
        See :meth:`plot_delta_slen` for the individual member sarcomeres.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The SarcAsM object with track motion analysis results.
        kind : str
            The grouping to plot ('pool', 'mband', 'myofibril', ...).
        t_lim : tuple of float, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple of float, optional
            The y-axis limits for ΔSL in µm. Default is (-0.4, 0.4).
        n_rows : int, optional
            Number of groups to display. If None, all groups are shown. Default is None.
        show_contr : bool, optional
            Whether to shade contraction periods. Default is True.
        use_median : bool, optional
            If True, use median sarcomere length instead of mean. Default is False.
        readable_labels : bool, optional
            See :meth:`plot_slen_mean`. Default is False.
        """
        kind = MotionPlots._resolve_kind(sarc_obj, kind)
        key = f'motion.{kind}.slen_median' if use_median else f'motion.{kind}.slen_avg'
        slen_ts = np.asarray(sarc_obj.data[key], dtype=float)
        contr = np.asarray(sarc_obj.data[f'motion.{kind}.contr']) if f'motion.{kind}.contr' in sarc_obj.data else None
        delta = np.full_like(slen_ts, np.nan)
        for g in range(slen_ts.shape[0]):
            c = contr[g] if contr is not None else np.zeros(slen_ts.shape[1], dtype=bool)
            delta[g] = slen_ts[g] - grouped_motion.equilibrium_over_quiet(slen_ts[g], c)
        MotionPlots._plot_group_stacked(ax, sarc_obj, kind, delta, contr, _LABEL_DELTA_SL,
                                  t_lim, y_lim, n_rows, show_contr, hline='zero',
                                  readable_labels=readable_labels)

    @staticmethod
    def _plot_members(ax, g: "Group", *, mode, t_lim=(0, None), y_lim=None, show_contr=True, show_mean=True, qc_only=False,
                      max_lines=300, color=None, mean_color='k'):
        """Overlay the members of one group (grey) and the group (bold).

        ``mode='slen'`` plots SL(t) and the group average ``slen_avg``; ``mode='delta'`` plots each
        member's ΔSL and the ΔSL of the group average. Members are subsampled (longest coverage first)
        to ``max_lines``.
        """
        MotionPlots._resolve_kind(g.sarc, g.kind)
        if y_lim is None:
            y_lim = (-0.4, 0.4) if mode == 'delta' else (1.6, 2.2)
        time = MotionPlots._series(g)['time']
        contr = np.asarray(g.contr, dtype=bool)
        if g.n_members == 0:
            ax.text(0.5, 0.5, 'No members', ha='center', va='center', transform=ax.transAxes)
            return
        # ΔSL comes from motion.tracks.slen, without filled frames; the lengths are shown with filled frames too
        lengths = (np.where(np.isfinite(g.tracks.slen_filled), g.tracks.slen_filled, g.tracks.slen)
                   if 'motion.tracks.slen_filled' in g.sarc.data else g.tracks.slen)
        member_y = np.asarray(g.tracks.delta_slen if mode == 'delta' else lengths, dtype=float)
        n_total = len(member_y)
        if qc_only:
            member_y = member_y[np.asarray(g.tracks.qc_pass, dtype=bool)]
        avg = np.asarray(g.slen_avg, dtype=float)
        mean_y = avg - grouped_motion.equilibrium_over_quiet(avg, contr) if mode == 'delta' else avg
        if max_lines is not None and len(member_y) > max_lines:
            keep = np.argsort(np.isfinite(member_y).sum(axis=1))[::-1][:max_lines]
            member_y = member_y[keep]
        n_shown = len(member_y)
        if show_contr and contr.any():
            ax.fill_between(time, y_lim[0], y_lim[1], where=contr, color='lavender', alpha=0.6)
        alpha = float(max(0.04, min(0.5, 30.0 / max(n_shown, 1))))
        ax.plot(time, member_y.T, c=color if color is not None else '0.6', lw=0.4, alpha=alpha)
        if show_mean:
            ax.plot(time, mean_y, c=mean_color, lw=2, zorder=3)
        if mode == 'delta':
            ax.axhline(0, lw=0.8, ls=':', c='k')
        ax.set_xlim(t_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(_LABEL_TIME)
        ax.set_ylabel(_LABEL_DELTA_SL if mode == 'delta' else _LABEL_SL)
        PlotUtils.polish_xticks(ax)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        title = f"{MOTION_KIND_LABELS.get(g.kind, g.kind)} {g.id} (n={n_total}"
        ax.set_title(title + (f", showing {n_shown})" if n_shown < n_total else ")"), fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_slen(ax: Axes, g: "Group", *, t_lim: Tuple[Optional[float], Optional[float]] = (0, None),
                  y_lim: Tuple[float, float] = (1.4, 2.2), show_contr: bool = True, show_mean: bool = True,
                  max_lines: Optional[int] = 300, color: Optional[str] = None, mean_color: str = 'k',
                  qc_only: bool = False):
        """
        Plot the sarcomere lengths of the members of one group, with their gaps filled
        (``motion.tracks.slen_filled``), and the group average ``slen_avg`` bold.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        g : Group
            The group, ``sarc.motion.groups(kind)[i]`` (analysed).
        t_lim : tuple of float, optional
            The time limits in seconds; ``None`` on either side means the recording boundary.
            Default is (0, None).
        y_lim : tuple of float, optional
            The y-axis limits in µm. Default is (1.4, 2.2).
        show_contr : bool, optional
            Whether to shade the contractions. Default is True.
        show_mean : bool, optional
            Whether to draw the group average. Default is True.
        max_lines : int, optional
            Draw at most this many members (longest coverage first); None draws all. Default is 300.
        color : str, optional
            Colour of the member traces; None is grey. Default is None.
        mean_color : str, optional
            Colour of the group average. Default is 'k'.
        qc_only : bool, optional
            Draw only members that pass the quality rules (``motion.tracks.qc_pass``). The bold line
            stays the average of all members. Default is False.
        """
        MotionPlots._plot_members(ax, g, mode='slen', t_lim=t_lim, y_lim=y_lim, show_contr=show_contr,
                                  show_mean=show_mean, max_lines=max_lines, color=color, mean_color=mean_color,
                                  qc_only=qc_only)

    @staticmethod
    def plot_delta_slen(ax: Axes, g: "Group", *, stacked: bool = False,
                        t_lim: Tuple[Optional[float], Optional[float]] = (0, None),
                        y_lim: Tuple[float, float] = (-0.4, 0.4), show_contr: bool = True, show_mean: bool = True,
                        max_lines: Optional[int] = 300, color: Optional[str] = None, mean_color: str = 'k',
                        frame: Optional[int] = None, n_rows: int = 6, n_start: int = 0, qc_only: bool = False):
        """
        Plot the length change ΔSL of the members of one group.

        By default the members are overlaid (grey) with the ΔSL of the group average (bold);
        ``stacked=True`` draws one row per member instead (``n_rows`` rows from ``n_start``, head to tail
        for the ordered kinds). ΔSL is the change from each member's resting length, its median over the
        group's quiet frames, without smoothing.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        g : Group
            The group, ``sarc.motion.groups(kind)[i]`` (analysed).
        stacked : bool, optional
            One row per member instead of an overlay. Default is False.
        t_lim : tuple of float, optional
            The time limits in seconds; ``None`` on either side means the recording boundary.
            Default is (0, None).
        y_lim : tuple of float, optional
            The y-axis limits in µm. Default is (-0.4, 0.4).
        show_contr : bool, optional
            Whether to shade the contractions. Default is True.
        show_mean : bool, optional
            Overlay: whether to draw the group's ΔSL. Default is True.
        max_lines : int, optional
            Overlay: draw at most this many members; None draws all. Default is 300.
        qc_only : bool, optional
            Overlay: draw only members that pass the quality rules. The group's ΔSL stays that of
            all members. Default is False.
        color : str, optional
            Overlay: colour of the member traces; None is grey. Default is None.
        mean_color : str, optional
            Overlay: colour of the group's ΔSL. Default is 'k'.
        frame : int, optional
            Stacked: mark this frame with a dashed line. Default is None.
        n_rows : int, optional
            Stacked: number of rows. Default is 6.
        n_start : int, optional
            Stacked: index of the first member shown. Default is 0.
        """
        if stacked:
            return MotionPlots._plot_delta_slen_stacked(ax, g, frame=frame, t_lim=t_lim, y_lim=y_lim,
                                                        n_rows=n_rows, n_start=n_start, show_contr=show_contr)
        MotionPlots._plot_members(ax, g, mode='delta', t_lim=t_lim, y_lim=y_lim, show_contr=show_contr, qc_only=qc_only,
                                  show_mean=show_mean, max_lines=max_lines, color=color, mean_color=mean_color)
