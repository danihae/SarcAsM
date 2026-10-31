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

"""Plots of the motion of the groups (group time series) and of the per-group view."""

import numbers
import os.path
from typing import Union, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt, transforms
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from sarcasm.features import MOTION_KIND_LABELS
from sarcasm.motion import Motion
from sarcasm.plotting.plot_utils import PlotUtils
from sarcasm.core import SarcAsM
from sarcasm.analysis import grouped_motion
from sarcasm.plotting.structure import StructurePlots
from sarcasm.plotting.structure import _LABEL_DELTA_SL, _LABEL_SL, _LABEL_TIME


class MotionPlots:
    """Plots of the motion of the groups (group time series) and of the per-group view. Part of :class:`~sarcasm.plotting.plots.Plots`."""

    @staticmethod
    def plot_loi_summary_motion(motion_obj: Motion, number_contr=0, t_lim=(0, None), t_lim_overlay=(-0.1, 2.9),
                                file_path=None):
        """
        Plot a summary of the motion of the line of interest (LOI).

        Parameters
        ----------
        motion_obj : Motion
            The Motion object to plot.
        number_contr : int, optional
            The index of the contraction to plot. Default is 0.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        t_lim_overlay : tuple of float, optional
            The time limits for the overlay plots in seconds. Default is (-0.1, 2.9).
        file_path : str, optional
            The file path to save the plot. If None, saved to ``summary_loi.png``
            in the LOI folder. Default is None.
        """

        mosaic = """
        aaaccc
        bbbccc
        dddeee
        dddfff
        """

        fig, axs = plt.subplot_mosaic(mosaic, figsize=(PlotUtils.width_2cols, PlotUtils.width_2cols),
                                      constrained_layout=True)
        title = f'File: {motion_obj.file_path}, \nLOI: {motion_obj.loi_name}'
        fig.suptitle(title, fontsize=PlotUtils.fontsize)

        # A- image cell w/ LOI
        StructurePlots.plot_image(axs['a'], motion_obj, show_loi=True)

        # B- U-Net cell w/ LOI
        StructurePlots.plot_z_bands(axs['b'], motion_obj, show_loi=True)

        # C- kymograph and tracked z-lines
        MotionPlots.plot_z_pos(axs['c'], motion_obj, t_lim=t_lim)

        # D- single sarcomere trajs (vel and delta slen)
        MotionPlots.plot_delta_slen(axs['d'], motion_obj, t_lim=t_lim)

        # E- overlay delta slen
        MotionPlots.plot_overlay_delta_slen(axs['e'], motion_obj, number_contr=number_contr, t_lim=t_lim_overlay)

        # F- overlay velocity
        MotionPlots.plot_overlay_velocity(axs['f'], motion_obj, number_contr=number_contr, t_lim=t_lim_overlay)

        PlotUtils.label_all_panels(axs)

        if file_path is None:
            file_path = os.path.join(motion_obj.loi_folder, 'summary_loi.png')
        # The LOI folder is not created at construction, so it may not exist yet —
        # create it on demand, as export_json/store_loi_data do.
        parent = os.path.dirname(os.path.abspath(file_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(file_path, dpi=PlotUtils.dpi)
        plt.show()

    @staticmethod
    def _shade_contr_loi(ax: Axes, motion_obj: Motion, t_offset: float = 0.0,
                         color: str = 'lavender', alpha_incomplete: float = 0.45) -> None:
        """Shade the contraction intervals of a LOI/``Motion`` trace on ``ax``.

        Driven by the boolean ``loi_data['contr']`` mask rather than
        ``zip(start_contr, time_contr)``: the durations are NaN for cycles that are
        incomplete at the recording edges, and ``start_contr`` (one entry per rising
        edge) and ``time_contr`` (one entry per labelled cycle) do not correspond
        one-to-one when a cycle begins on the very first frame. Incomplete cycles are
        drawn at reduced alpha so a truncated cycle is not mistaken for a full one.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to shade. The shading spans the full height via a blended transform.
        motion_obj : Motion
            Motion object holding ``loi_data['time'] / ['contr']``.
        t_offset : float, optional
            Subtracted from the time axis, for the single-contraction views that
            re-zero time on the selected cycle. Default is 0.0.
        color : str, optional
            Shading colour. Default is 'lavender'.
        alpha_incomplete : float, optional
            Alpha for cycles that are incomplete at the recording edges. Default is 0.45.
        """
        contr = motion_obj.loi_data.get('contr')
        time = motion_obj.loi_data.get('time')
        if contr is None or time is None:
            return
        contr = np.asarray(contr, dtype=bool)
        time = np.asarray(time, dtype=float) - t_offset
        if contr.shape[0] != time.shape[0] or not contr.any():
            return
        blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        # Split the mask into complete / incomplete cycles when the flags are present.
        labels = motion_obj.loi_data.get('labels_contr')
        complete_flags = motion_obj.loi_data.get('contr_complete')
        incomplete = np.zeros_like(contr)
        if labels is not None and complete_flags is not None:
            labels = np.asarray(labels)
            for i, ok in enumerate(np.asarray(complete_flags, dtype=bool)):
                if not ok:
                    incomplete |= labels == i + 1
        ax.fill_between(time, 0, 1, where=contr & ~incomplete, color=color,
                        transform=blended, linewidth=0)
        if incomplete.any():
            ax.fill_between(time, 0, 1, where=incomplete, color=color,
                            alpha=alpha_incomplete, transform=blended, linewidth=0)

    @staticmethod
    def _contr_window(motion_obj: Motion, number_contr, t_lim):
        """Frame window of a contraction-centered view, clamped to the recording.

        Parameters
        ----------
        motion_obj : Motion
            Object holding ``loi_data['start_contr']`` / ``['n_contr']`` / ``['time']``.
        number_contr : int or None
            Index of the contraction to center on; None disables centering.
        t_lim : tuple of float or None
            Window around the contraction onset, in seconds (may start negative).
            ``None`` on either side means "up to the recording boundary".

        Returns
        -------
        tlim : tuple
            Absolute time limits, or ``(None, None)`` when not centering.
        i0, i1 : int or None
            Frame slice bounds, or ``(None, None)`` when not centering, so that
            ``arr[i0:i1]`` yields the full series. Both are clamped into
            ``[0, len(time)]`` (a negative start index would silently yield an
            empty slice).
        """
        if number_contr is None or motion_obj.loi_data['n_contr'] <= 0:
            return (None, None), None, None
        start_contr_t = motion_obj.loi_data['start_contr'][number_contr]
        n = len(motion_obj.loi_data['time'])
        frametime = motion_obj.metadata.frametime
        lo = 0.0 if t_lim[0] is None else start_contr_t + t_lim[0]
        hi = n * frametime if t_lim[1] is None else start_contr_t + t_lim[1]
        i0 = min(max(0, int(lo / frametime)), n)
        i1 = min(max(i0, int(hi / frametime)), n)
        return (lo, hi), i0, i1

    @staticmethod
    def plot_z_pos(ax: Axes, motion_obj: Motion, number_contr=None, show_contr=True, color='k',
                   t_lim=(0, None), y_lim=(None, None)):
        """
        Plot the Z-band trajectories of the motion object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        number_contr : int, optional
            The index of the contraction to center the plot on. If None, the full
            time series is shown. Default is None.
        show_contr : bool, optional
            Whether to shade the contraction periods. Default is True.
        color : str, optional
            The color of the trajectories. Default is 'k'.
        t_lim : tuple, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple, optional
            The y-axis limits for the plot. Default is (None, None).
        """
        # plot limits and params
        tlim, i0, i1 = MotionPlots._contr_window(motion_obj, number_contr, t_lim)
        centered = i0 is not None

        # get data
        time = motion_obj.loi_data['time']
        z_pos = motion_obj.loi_data['z_pos']
        # plot contraction cycles
        if show_contr:
            MotionPlots._shade_contr_loi(ax, motion_obj, t_offset=tlim[0] if centered else 0.0)

        # plot trajectories
        if centered:
            seg = z_pos[:, i0:i1].T
            ax.plot(time[:len(seg)], seg, linewidth=0.75, c=color)
            ax.set_xlim(0, tlim[1] - tlim[0])
        else:
            ax.plot(time, z_pos.T, linewidth=0.75, c=color)
            ax.set_xlim(t_lim)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Z-band position Z(t) [µm]')
        if y_lim == (None, None):
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(y_lim)
        PlotUtils.polish_yticks(ax, 5, 2.5)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def _plot_delta_slen_loi(ax: Axes, motion_obj: Motion, frame=None, t_lim=(0, None), y_lim=(-0.3, 0.4), n_rows=6,
                             n_start=1, show_contr=True):
        """
        Plot the change in sarcomere length over time (stacked rows) for a motion object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        frame : int, optional
            Mark this frame with a vertical dashed line. Default is None.
        t_lim : tuple, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple, optional
            The y-axis limits for the plot. Default is (-0.3, 0.4).
        n_rows : int, optional
            The number of sarcomere rows to plot. Default is 6.
        n_start : int, optional
            The starting sarcomere index for the plot. Default is 1.
        show_contr : bool, optional
            Whether to shade the contraction periods. Default is True.
        """
        yticks = [-0.2, 0, 0.2]
        delta_slen = motion_obj.loi_data['delta_slen']
        # a short fibre has fewer sarcomeres than rows: draw the ones it has
        n_sarc = len(delta_slen)
        if n_start >= n_sarc:
            n_start = 0
        n_rows = min(n_rows, n_sarc - n_start)
        if n_rows <= 0:
            ax.text(0.5, 0.5, 'No sarcomeres', ha='center', va='center', transform=ax.transAxes)
            return
        list_y = np.linspace(0, 1, num=n_rows, endpoint=False)
        for i, y in enumerate(list_y):
            ax_i = ax.inset_axes((0., y, 1, 1 / n_rows - min(0.02, 0.3 / n_rows)))
            ax_i.plot(motion_obj.loi_data['time'], delta_slen[i + n_start], c='k', lw=0.6)
            ax_i.axhline(0, linewidth=1, linestyle=':', c='k')
            if show_contr:
                MotionPlots._shade_contr_loi(ax_i, motion_obj)

            if frame is not None:
                ax_i.axvline(motion_obj.loi_data['time'][frame], linestyle='--', c='k')
            ax_i.set_ylim(y_lim)
            ax_i.set_xlim(t_lim)
            if i > 0:
                ax_i.set_xticks([])
            else:
                PlotUtils.polish_xticks(ax_i)
                ax_i.tick_params(axis='x', labelsize='x-small')
            ax_i.set_yticks(yticks)
            ax_i.set_yticklabels(yticks, fontsize='x-small')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\Delta$SL [µm]')
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.xaxis.label.set_color('k')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')

    @staticmethod
    def plot_overlay_delta_slen(ax: Axes, motion_obj: Motion, number_contr=None, t_lim=(0, None), y_lim=(-0.35, 0.5),
                                show_contr=True):
        """
        Plot the sarcomere length change over time for a motion object, overlaying multiple trajectories.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        number_contr : int, optional
            The index of a single contraction to center on. If None, the full time
            series is shown. Default is None.
        t_lim : tuple, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple, optional
            The y-axis limits for the plot. Default is (-0.35, 0.5).
        show_contr : bool, optional
            Whether to shade the contraction periods. Default is True.
        """
        # plot limits and params
        tlim, i0, i1 = MotionPlots._contr_window(motion_obj, number_contr, t_lim)
        centered = i0 is not None
        # get data
        time = motion_obj.loi_data['time']
        delta_slen = motion_obj.loi_data['delta_slen']
        delta_slen_avg = motion_obj.loi_data['delta_slen_avg']
        # plot contraction cycles
        if show_contr:
            MotionPlots._shade_contr_loi(ax, motion_obj, t_offset=tlim[0] if centered else 0.0)

        # colormap
        cm = plt.cm.nipy_spectral(np.linspace(0, 1, len(delta_slen)))
        ax.set_prop_cycle('color', list(cm))

        # plot single and average trajectories
        if centered:
            seg = delta_slen.T[i0:i1]
            ax.plot(time[:len(seg)], seg, linewidth=0.5)
            ax.plot(time[:len(seg)], delta_slen_avg[i0:i1], c='k', linewidth=2,
                    linestyle='-')
            ax.set_xlim(0, tlim[1] - tlim[0])
        else:
            ax.plot(time, delta_slen.T, linewidth=0.5)
            ax.plot(time, delta_slen_avg, c='k', linewidth=2,
                    linestyle='-')
            ax.set_xlim(t_lim)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$\Delta$SL [µm]')
        ax.set_ylim(y_lim)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_overlay_velocity(ax, motion_obj: Motion, number_contr=None, t_lim=(0, None), y_lim=(-9, 12),
                              show_contr=True):
        """
        Plot an overlay of the sarcomere velocity time series of the motion object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        number_contr : int, optional
            The index of a single contraction to center on. If None, the full time
            series is shown. Default is None.
        t_lim : tuple, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple, optional
            The y-axis limits for the plot. Default is (-9, 12).
        show_contr : bool, optional
            Whether to shade the contraction periods. Default is True.
        """
        # plot limits and params
        tlim, i0, i1 = MotionPlots._contr_window(motion_obj, number_contr, t_lim)
        centered = i0 is not None
        # get data
        time = motion_obj.loi_data['time']
        vel = motion_obj.loi_data['vel']
        vel_avg = motion_obj.loi_data['vel_avg']

        # plot contraction cycles
        if show_contr:
            MotionPlots._shade_contr_loi(ax, motion_obj, t_offset=tlim[0] if centered else 0.0)

        # colormap
        cm = plt.cm.nipy_spectral(np.linspace(0, 1, len(vel)))
        ax.set_prop_cycle('color', list(cm))

        # plot single and average trajectories
        if centered:
            seg = vel.T[i0:i1]
            ax.plot(time[:len(seg)], seg, linewidth=0.5)
            ax.plot(time[:len(seg)], vel_avg[i0:i1], c='k', linewidth=2,
                    linestyle='-')
            ax.set_xlim(0, tlim[1] - tlim[0])
        else:
            ax.plot(time, vel.T, linewidth=0.5)
            ax.plot(time, vel_avg, c='k', linewidth=2,
                    linestyle='-')
            ax.set_xlim(t_lim)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('V [µm/s]')
        ax.set_ylim(y_lim)
        PlotUtils.polish_yticks(ax, 3, 1)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_phase_space(ax: Axes, motion_obj: Motion, t_lim=(0, None), number_contr=None, frame=None):
        """
        Plot the sarcomere trajectory in length-change vs. velocity phase space.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        t_lim : tuple, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        number_contr : int, optional
            The index of a single contraction to plot. If None, all contractions are
            overlaid. Default is None.
        frame : int, optional
            If set, mark the individual sarcomeres at this frame as scatter points. Default is None.
        """
        # get data
        delta_slen = motion_obj.loi_data['delta_slen']
        vel = motion_obj.loi_data['vel']
        delta_slen_avg = motion_obj.loi_data['delta_slen_avg']
        vel_avg = motion_obj.loi_data['vel_avg']
        # colormap
        cm = plt.cm.nipy_spectral(np.linspace(0, 1, len(delta_slen)))
        ax.set_prop_cycle('color', list(cm))
        # plot limits and params
        _, i0, i1 = MotionPlots._contr_window(motion_obj, number_contr, t_lim)
        for i, (vel_i, delta_i) in enumerate(zip(vel, delta_slen)):
            ax.plot(vel_i[i0:i1], delta_i[i0:i1], c='r', alpha=0.35, lw=0.36, zorder=1)
            if isinstance(frame, numbers.Integral):
                ax.scatter(vel_i[frame], delta_i[frame], c=cm[i], s=10,
                           zorder=2)

        ax.plot(vel_avg[i0:i1], delta_slen_avg[i0:i1], c='k', lw=1, label='Average')
        legend_elements = [Line2D([0], [0], color='k', lw=2), Line2D([0], [0], color='r', alpha=0.35, lw=0.5)]
        ax.legend(legend_elements, ['Average', 'Individual'], loc='upper right')
        PlotUtils.polish_xticks(ax, 5, 2.5)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        ax.set_xlabel('Velocity $V$ [µm/s]', fontsize=PlotUtils.fontsize)
        ax.set_ylabel(r'Length change $\Delta SL$ [µm]', fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_popping_events(motion_obj: Motion, save_name=None):
        """
        Create a binary event map of the popping events of the motion object.

        Parameters
        ----------
        motion_obj : Motion
            The motion object to plot.
        save_name : str, optional
            File path to save the plot. If None, the plot is not saved. Default is None.
        """
        popping_events = motion_obj.loi_data['popping_events']
        prob_time = motion_obj.loi_data['popping_freq_time']
        prob_sarcomeres = motion_obj.loi_data['popping_freq_sarcomeres']

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.02

        rect_scatter = (left, bottom, width, height)
        rect_histx = (left, bottom + height + spacing, width, 0.2)
        rect_histy = (left + width + spacing, bottom, 0.2, height)

        fig_events = plt.figure(figsize=(PlotUtils.width_1cols * 0.9, 3.))
        ax = fig_events.add_axes(rect_scatter)
        ax_histx = fig_events.add_axes(rect_histx, sharex=ax)
        ax_histy = fig_events.add_axes(rect_histy, sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax.pcolorfast(popping_events, cmap='Greys')
        ax_histx.bar(np.arange(len(prob_time)) + 0.5, prob_time, color='k', alpha=0.4)
        ax_histy.barh(np.arange(len(prob_sarcomeres)) + 0.5, prob_sarcomeres, color='k', alpha=0.4)

        ax.set_xlabel('Contraction cycle [#]')
        ax.set_ylabel('Sarcomere [#]')
        yticks = np.arange(len(prob_sarcomeres))
        ax.set_yticks(yticks + 0.5)
        ax.set_yticklabels(yticks + 1)
        ax_histx.set_ylabel('$f_c(P)$')
        ax_histy.set_xlabel('$f_s(P)$')
        ax.set_ylim(0, None)
        ax.set_xlim(0, None)
        ax.grid()

        if save_name is not None:
            fig_events.savefig(save_name)

    # ------------------------------------------------------------------
    # 2D sarcomere tracking + grouped motion (motion.track ->
    # motion.group_tracks -> motion.analyze).
    # ------------------------------------------------------------------

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
    def _track_group_overlay(ax, sarc_obj, *, mode, group=0, kind=None,
                             t_lim=(0, None), y_lim=None, show_contr=True, show_mean=True,
                             max_lines=300, color=None, mean_color='k'):
        """Overlay the individual member sarcomeres of one track group + the group mean.

        ``mode='slen'`` plots SL(t); ``mode='delta'`` plots ΔSL(t) = SL(t) − equ
        (per-member equilibrium over the group's non-contracting frames). The bold
        overlay is the group aggregate over *all* members; individual member lines
        are subsampled (longest-coverage first) to ``max_lines`` for legibility.
        """
        kind = MotionPlots._resolve_kind(sarc_obj, kind)
        if y_lim is None:
            y_lim = (-0.4, 0.4) if mode == 'delta' else (1.6, 2.2)
        n_tracks = int(sarc_obj.data['motion.tracks.n'])
        grouping = sarc_obj.motion._grouping(kind)
        n_groups = grouping['n']
        if not (0 <= group < max(n_groups, 1)):
            raise ValueError(f'group {group} out of range [0, {n_groups}).')
        gid = grouping['track_group_id']
        slen_tracks = np.asarray(sarc_obj.data['motion.tracks.slen'], dtype=float).reshape(n_tracks, -1)
        T = slen_tracks.shape[1]
        time = np.arange(T) * sarc_obj.metadata.frametime
        agg = np.asarray(sarc_obj.data[f'motion.{kind}.slen_avg'], dtype=float)[group]
        contr = (np.asarray(sarc_obj.data[f'motion.{kind}.contr'])[group]
                 if f'motion.{kind}.contr' in sarc_obj.data else np.zeros(T, dtype=bool))

        members = np.flatnonzero(gid == group)
        if members.size == 0:
            ax.text(0.5, 0.5, 'No members', ha='center', va='center', transform=ax.transAxes)
            return
        n_total = members.size
        if max_lines is not None and members.size > max_lines:
            cov = np.isfinite(slen_tracks[members]).sum(axis=1)
            members = members[np.argsort(cov)[::-1][:max_lines]]
        n_shown = members.size
        member_slen = slen_tracks[members]  # (k, T)

        if mode == 'delta':
            equ_m = grouped_motion.equilibrium_over_quiet(member_slen, contr)
            member_y = member_slen - equ_m[:, None]
            mean_y = agg - grouped_motion.equilibrium_over_quiet(agg, contr)
            ylabel = _LABEL_DELTA_SL
        else:
            member_y = member_slen
            mean_y = agg
            ylabel = _LABEL_SL

        if show_contr and contr.any():
            ax.fill_between(time, y_lim[0], y_lim[1], where=contr, color='lavender', alpha=0.6)
        col = color if color is not None else '0.6'
        alpha = float(max(0.04, min(0.5, 30.0 / max(n_shown, 1))))
        ax.plot(time, member_y.T, c=col, lw=0.4, alpha=alpha)
        if show_mean:
            ax.plot(time, mean_y, c=mean_color, lw=2, zorder=3)
        if mode == 'delta':
            ax.axhline(0, lw=0.8, ls=':', c='k')

        ax.set_xlim(t_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(_LABEL_TIME)
        ax.set_ylabel(ylabel)
        PlotUtils.polish_xticks(ax)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        title = f"{kind} group {group} (n={n_total}"
        title += f", showing {n_shown})" if n_shown < n_total else ")"
        ax.set_title(title, fontsize=PlotUtils.fontsize)

    @staticmethod
    def _plot_slen_loi(ax, motion_obj: Motion, t_lim=(None, None), y_lim=(None, None),
                       show_contr=True, show_mean=True, color=None, mean_color='k'):
        """LOI overlay of individual sarcomere lengths + the average."""
        time = motion_obj.loi_data['time']
        slen = np.asarray(motion_obj.loi_data['slen'], dtype=float)
        slen_avg = motion_obj.loi_data['slen_avg']
        if show_contr:
            MotionPlots._shade_contr_loi(ax, motion_obj)
        col = color if color is not None else '0.6'
        alpha = float(max(0.04, min(0.5, 30.0 / max(len(slen), 1))))
        ax.plot(time, slen.T, c=col, lw=0.4, alpha=alpha)
        if show_mean:
            ax.plot(time, slen_avg, c=mean_color, lw=2, zorder=3)
        ax.set_xlim(t_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(_LABEL_TIME)
        ax.set_ylabel(_LABEL_SL)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_slen(ax: Axes, obj: Union[SarcAsM, Motion], *, group: int = 0, kind: Optional[str] = None,
                  t_lim: Tuple[Optional[float], Optional[float]] = (0, None),
                  y_lim: Optional[Tuple[float, float]] = None,
                  show_contr: bool = True, show_mean: bool = True, max_lines: Optional[int] = 300,
                  color: Optional[str] = None, mean_color: str = 'k'):
        """
        Plot individual sarcomere-length traces with the mean overlaid.

        Polymorphic — pass either object:

        * **SarcAsM** (track grouping): overlays the member sarcomeres of one
          track ``group`` of the ``kind`` grouping, with
          the group aggregate drawn bold. Members are subsampled to ``max_lines``
          (longest-coverage first) for legibility. Requires
          :meth:`MotionAnalysis.analyze`.
        * **Motion** (LOI): overlays the per-sarcomere lengths along the LOI
          plus the average (``t_lim``/``y_lim``/``show_contr``/``show_mean``/``color``/
          ``mean_color`` apply; ``group``/``kind``/``max_lines`` are ignored).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        obj : SarcAsM or Motion
            The object to plot (see polymorphic behaviour above).
        group : int, optional
            Track group index to plot (SarcAsM only). Default is 0.
        kind : str, optional
            The grouping the group belongs to (SarcAsM only, required there).
        t_lim : tuple of float, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple of float, optional
            The y-axis limits for sarcomere length in µm. If None, a ``Motion``
            object is auto-scaled and a track group uses (1.4, 2.2). Default is None.
        show_contr : bool, optional
            Whether to shade contraction periods. Default is True.
        show_mean : bool, optional
            Whether to overlay the bold mean trace. Default is True.
        max_lines : int, optional
            Maximum number of member traces to draw (SarcAsM only). Default is 300.
        color : str, optional
            Colour of the individual traces. If None, a grey is used. Default is None.
        mean_color : str, optional
            Colour of the mean trace. Default is 'k'.
        """
        if isinstance(obj, Motion):
            return MotionPlots._plot_slen_loi(ax, obj, t_lim=t_lim,
                                        y_lim=y_lim if y_lim is not None else (None, None),
                                        show_contr=show_contr, show_mean=show_mean,
                                        color=color, mean_color=mean_color)
        return MotionPlots._track_group_overlay(ax, obj, mode='slen', group=group, kind=kind,
                                          t_lim=t_lim, y_lim=y_lim if y_lim is not None else (1.4, 2.2),
                                          show_contr=show_contr,
                                          show_mean=show_mean, max_lines=max_lines,
                                          color=color, mean_color=mean_color)

    @staticmethod
    def plot_delta_slen(ax: Axes, obj: Union[SarcAsM, Motion], *, group: int = 0, kind: Optional[str] = None,
                        t_lim: Tuple[Optional[float], Optional[float]] = (0, None), y_lim: Tuple[float, float] = (-0.4, 0.4),
                        show_contr: bool = True, show_mean: bool = True, max_lines: Optional[int] = 300,
                        color: Optional[str] = None, mean_color: str = 'k',
                        frame: Optional[int] = None, n_rows: int = 6, n_start: int = 1):
        """
        Plot individual sarcomere-length *change* (ΔSL) traces with the mean overlaid.

        The ΔSL counterpart of :meth:`plot_slen`, polymorphic on ``obj``:

        * **SarcAsM** (track grouping): overlays member ΔSL(t) = SL(t) − equ for
          one track ``group`` (``equ`` = each member's median length over the
          group's non-contracting frames), with the group ΔSL drawn bold. Members
          are subsampled to ``max_lines`` (longest-coverage first) for legibility.
          Requires :meth:`MotionAnalysis.analyze`.
        * **Motion** (LOI): the stacked per-sarcomere ΔSL view,
          one inset row per sarcomere.

        The two branches take different parameters — ``group``/``kind``/``max_lines``/
        ``show_mean``/``color``/``mean_color`` apply to **SarcAsM** only, and
        ``frame``/``n_rows``/``n_start`` to **Motion** only. Parameters that do not
        apply to the object you pass are ignored.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        obj : SarcAsM or Motion
            The object to plot (see polymorphic behaviour above).
        group : int, optional
            Track group index to plot (SarcAsM only). Default is 0.
        kind : str, optional
            The grouping the group belongs to ('pool', 'mband', ...) (SarcAsM only, required there).
        t_lim : tuple of float, optional
            The time limits for the plot in seconds; ``None`` on either side means the
            recording boundary. Default is (0, None).
        y_lim : tuple of float, optional
            The y-axis limits for ΔSL in µm. Default is (-0.4, 0.4).
        show_contr : bool, optional
            Whether to shade contraction periods. Default is True.
        show_mean : bool, optional
            Whether to overlay the bold group ΔSL trace (SarcAsM only). Default is True.
        max_lines : int, optional
            Maximum number of member traces to draw (SarcAsM only). If None, all
            members are drawn. Default is 300.
        color : str, optional
            Colour of the individual traces (SarcAsM only). If None, a grey is used.
            Default is None.
        mean_color : str, optional
            Colour of the mean trace (SarcAsM only). Default is 'k'.
        frame : int, optional
            Mark this frame with a vertical dashed line (Motion only). Default is None.
        n_rows : int, optional
            Number of stacked sarcomere rows to plot (Motion only). Default is 6.
        n_start : int, optional
            Index of the first sarcomere row to plot (Motion only). Default is 1.
        """
        if isinstance(obj, Motion):
            return MotionPlots._plot_delta_slen_loi(ax, obj, frame=frame, t_lim=t_lim, y_lim=y_lim,
                                              n_rows=n_rows, n_start=n_start, show_contr=show_contr)
        return MotionPlots._track_group_overlay(ax, obj, mode='delta', group=group, kind=kind,
                                          t_lim=t_lim, y_lim=y_lim, show_contr=show_contr,
                                          show_mean=show_mean, max_lines=max_lines,
                                          color=color, mean_color=mean_color)
