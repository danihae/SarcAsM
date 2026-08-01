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

"""Plotting functions for SarcAsM and Motion objects."""

import numbers
import os.path
import warnings
from typing import Union, Tuple, Optional, Literal

import numpy as np
from matplotlib import pyplot as plt, transforms
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

from sarcasm._internal.feature_dict import structure_feature_dict
from sarcasm.motion import Motion
from sarcasm.plotting.plot_utils import PlotUtils
from sarcasm.structure import SarcAsM
from sarcasm.analysis import domain_clustering, grouped_motion, myofibril_analysis
from sarcasm.utils import Utils

# Canonical axis labels (kept consistent with the symbols used across the
# package and the feature dictionary: SL = sarcomere length, ΔSL = its change).
_LABEL_TIME = 'Time [s]'
_LABEL_SL = 'Sarcomere length SL [µm]'
_LABEL_DELTA_SL = r'$\Delta$SL [µm]'


class Plots:
    """Plotting functions for SarcAsM and Motion objects."""

    @staticmethod
    def plot_stack_overlay(ax: Axes, sarc_obj: Union[SarcAsM, Motion], frames, plot_func, offset=0.025,
                           spine_color='w', xlim=None, ylim=None):
        """
        Plot a stack of overlaid subplots on a given Axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which the stack is plotted.
        sarc_obj : SarcAsM or Motion
            The object to plot in each subplot.
        frames : list of int
            The frames at which the subplots are created.
        plot_func : callable
            The function used to plot the data in each subplot, called as
            ``plot_func(ax, sarc_obj, frame)``.
        offset : float, optional
            The offset between each subplot. Default is 0.025.
        spine_color : str, optional
            The color of the spines (borders) of each subplot. Default is 'w'.
        xlim : tuple, optional
            The x-axis limits for each subplot. Default is None.
        ylim : tuple, optional
            The y-axis limits for each subplot. Default is None.
        """
        ax.axis('off')
        for i, t in enumerate(frames):
            ax_t = ax.inset_axes((0.1 + offset * i, 0.1 - offset * i, 0.8, 0.8))

            plot_func(ax_t, sarc_obj, t)
            ax_t.spines['bottom'].set_color(spine_color)
            ax_t.spines['top'].set_color(spine_color)
            ax_t.spines['right'].set_color(spine_color)
            ax_t.spines['left'].set_color(spine_color)

            ax_t.set_xlim(xlim)
            ax_t.set_ylim(ylim)

    @staticmethod
    def plot_loi_summary_motion(motion_obj: Motion, number_contr=0, t_lim=(0, 12), t_lim_overlay=(-0.1, 2.9),
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
            The time limits for the plot in seconds. Default is (0, 12).
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
        Plots.plot_image(axs['a'], motion_obj, show_loi=True)

        # B- U-Net cell w/ LOI
        Plots.plot_z_bands(axs['b'], motion_obj, show_loi=True)

        # C- kymograph and tracked z-lines
        Plots.plot_z_pos(axs['c'], motion_obj, t_lim=t_lim)

        # D- single sarcomere trajs (vel and delta slen)
        Plots.plot_delta_slen(axs['d'], motion_obj, t_lim=t_lim)

        # E- overlay delta slen
        Plots.plot_overlay_delta_slen(axs['e'], motion_obj, number_contr=number_contr, t_lim=t_lim_overlay)

        # F- overlay velocity
        Plots.plot_overlay_velocity(axs['f'], motion_obj, number_contr=number_contr, t_lim=t_lim_overlay)

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
    def plot_loi_detection(sarc_obj: SarcAsM, frame: int = 0, file_path: str = None,
                           cmap_z_bands='Greys'):
        """
        Plot all steps of the automated LOI finding algorithm.

        Parameters
        ----------
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        file_path : str, optional
            Path to save the plot. If None, the plot is not saved. Default is None.
        cmap_z_bands : str, optional
            Colormap of Z-bands. Default is 'Greys'.
        """
        mosaic = """
        ac
        bd
        """

        fig, axs = plt.subplot_mosaic(mosaic, figsize=(PlotUtils.width_2cols, PlotUtils.width_1p5cols),
                                      constrained_layout=True)

        if isinstance(sarc_obj.data['params.analyze_sarcomere_vectors.frames'], int):
            frame = sarc_obj.data['params.analyze_sarcomere_vectors.frames']
        elif sarc_obj.data['params.analyze_sarcomere_vectors.frames'] == 'all':
            frame = frame
        else:
            frame = sarc_obj.data['params.analyze_sarcomere_vectors.frames'][frame]

        Plots.plot_z_bands(axs['a'], sarc_obj, frame=frame, cmap=cmap_z_bands)
        Plots.plot_z_bands(axs['c'], sarc_obj, frame=frame, cmap=cmap_z_bands)
        Plots.plot_z_bands(axs['d'], sarc_obj, frame=frame, cmap=cmap_z_bands)

        for i, pos_vectors_i in enumerate(sarc_obj.data['loi_data']['lines_vectors']):
            axs['a'].plot(pos_vectors_i[:, 1], pos_vectors_i[:, 0], c='r', lw=0.2, alpha=0.6)

        axs['b'].hist(sarc_obj.data['loi_data']['hausdorff_dist_matrix'].reshape(-1), bins=100, color='k',
                      alpha=0.75,
                      rwidth=0.75)
        axs['b'].set_xlim(0, 400)
        axs['b'].set_xlabel('Hausdorff distance')
        axs['b'].set_ylabel('# LOI pairs')

        for i, (pos_vectors_i, label_i) in enumerate(zip(sarc_obj.data['loi_data']['lines_vectors'],
                                                         sarc_obj.data['loi_data']['line_cluster'])):
            axs['c'].plot(pos_vectors_i[:, 1], pos_vectors_i[:, 0],
                          c=plt.cm.jet(label_i / sarc_obj.data['loi_data']['n_lines_clusters']), lw=0.2)

        for i, line_i in enumerate(sarc_obj.data['loi_data']['loi_lines']):
            axs['d'].plot(line_i.T[1], line_i.T[0], lw=2, label=i)
        axs['d'].legend(loc='lower left', fontsize='xx-small')

        PlotUtils.label_all_panels(axs, offset=(0.05, 0.9))

        axs['a'].set_title('1. Line growth', ha='left', x=0.02, fontsize=PlotUtils.fontsize + 1, fontweight='bold')
        axs['b'].set_title('2. Pair-wise Hausdorff distance', ha='left', x=0.02, fontsize=PlotUtils.fontsize + 1,
                           fontweight='bold')
        axs['c'].set_title('3. Agglomerative clustering', ha='left', x=0.02, fontsize=PlotUtils.fontsize + 1,
                           fontweight='bold')
        axs['d'].set_title('4. LOI lines', ha='left', x=0.02, fontsize=PlotUtils.fontsize + 1, fontweight='bold')

        if file_path is not None:
            fig.savefig(file_path, dpi=300)
        plt.show()

    @staticmethod
    def plot_image(ax: Axes, sarc_obj: Union[SarcAsM, Motion], frame: int = 0, cmap: str = 'gray',
                   alpha: float = 1, clip_thrs: Tuple[float, float] = (1, 99), scalebar: bool = True,
                   title: Union[None, str] = None, show_loi: bool = False, invert: bool = False,
                   zoom_region: Tuple[int, int, int, int] = None,
                   inset_bounds: Tuple[float, float, float, float] = (0.6, 0.6, 0.4, 0.4)):
        """
        Plot the raw microscopy image of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        cmap : str, optional
            The colormap to use. Default is 'gray'.
        alpha : float, optional
            The opacity of the image. Default is 1.
        clip_thrs : tuple of float, optional
            Clipping thresholds to normalize intensity, in percentiles. Default is (1, 99).
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        title : str, optional
            The title for the plot. Default is None.
        show_loi : bool, optional
            Whether to show the line of interest (LOI). Default is False.
        invert : bool, optional
            If True, reverse the colormap (e.g. 'gray' -> 'gray_r') so bright
            pixels render dark and vice versa. Default is False.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """

        img = sarc_obj.read_imgs(frames=frame)
        img = np.clip(img, np.percentile(img, clip_thrs[0]), np.percentile(img, clip_thrs[1]))

        cmap_use = plt.get_cmap(cmap).reversed() if invert else cmap
        _ = ax.imshow(img, cmap=cmap_use, alpha=alpha)
        ax.set_aspect('equal')
        if show_loi:
            Plots.plot_lois(ax, sarc_obj)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                   width_fraction=0.035, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            ax_inset.imshow(img[y1:y2, x1:x2], cmap=cmap_use)
            ax_inset.set_aspect('equal')
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')
            PlotUtils.change_color_spines(ax_inset, 'w')

            if scalebar:
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w',
                                             sep=1, width_fraction=0.035, location='lower right', scale_loc='top',
                                             font_properties={'size': PlotUtils.fontsize - 1}))

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
    def _draw_background(ax: Axes, sarc_obj: Union[SarcAsM, Motion], frame: int = 0, *,
                         show_image: bool = False, show_z_bands: bool = False,
                         invert_image: bool = False, invert_z_bands: bool = False,
                         cmap_image: str = 'gray', cmap_z_bands: str = 'Greys_r',
                         alpha_image: float = 1, alpha_z_bands: float = 1,
                         clip_thrs: Optional[Tuple[float, float]] = None,
                         scalebar: bool = False):
        """Draw the background of a structure overlay plot.

        Centralises the choice between raw microscopy image and Z-band mask so
        that every overlay plot (:meth:`plot_sarcomere_mask`,
        :meth:`plot_sarcomere_vectors`, ...) renders its background the same
        way, on both the main panel and the zoom inset. By default nothing is
        drawn — the caller opts in via ``show_image`` or ``show_z_bands``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the background on.
        sarc_obj : SarcAsM or Motion
            The sarcomere object to read the image/Z-bands from.
        frame : int, optional
            The frame to draw. Default is 0.
        show_image : bool, optional
            If True, draw the raw microscopy image. Default is False.
        show_z_bands : bool, optional
            If True, draw the Z-band mask. Default is False. Mutually exclusive
            with ``show_image``.
        invert_image : bool, optional
            Reverse the raw-image colormap (e.g. 'gray' -> 'gray_r'). Default is False.
        invert_z_bands : bool, optional
            Reverse the Z-band colormap (e.g. 'Greys_r' -> 'Greys'). Default is False.
        cmap_image : str, optional
            Colormap of the raw image. Default is 'gray'.
        cmap_z_bands : str, optional
            Colormap of the Z-bands. Default is 'Greys_r'.
        alpha_image : float, optional
            Opacity of the raw image. Default is 1.
        alpha_z_bands : float, optional
            Opacity of the Z-bands. Default is 1.
        clip_thrs : tuple of float, optional
            Clipping percentiles forwarded to :meth:`plot_image`. If None,
            ``plot_image``'s default ``(1, 99)`` is used. Default is None.
        scalebar : bool, optional
            Whether to add a scalebar via the background call. Default is False
            (overlay plots add their own scalebar afterwards).
        """
        if show_image and show_z_bands:
            raise ValueError("show_image and show_z_bands are mutually exclusive.")
        if show_image:
            kw = dict(cmap=cmap_image, alpha=alpha_image, invert=invert_image, scalebar=scalebar)
            if clip_thrs is not None:
                kw['clip_thrs'] = clip_thrs
            Plots.plot_image(ax, sarc_obj, frame=frame, **kw)
        elif show_z_bands:
            Plots.plot_z_bands(ax, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands,
                               invert=invert_z_bands, scalebar=scalebar)
        else:
            # No background image: nothing establishes image coordinates, so set them
            # here. Without this the overlay inherits matplotlib's y-up default and
            # renders vertically mirrored w.r.t. the image (and w.r.t. the zoom insets,
            # which invert explicitly), while collection-only plots (plot_tracks) never
            # autoscale at all and come out blank.
            size = getattr(sarc_obj.metadata, 'size', None)
            if size is not None and len(size) >= 2:
                h, w = int(size[-2]), int(size[-1])
                ax.set_xlim(-0.5, w - 0.5)
                ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect('equal')

    @staticmethod
    def plot_z_bands(ax: plt.Axes, sarc_obj: Union[SarcAsM, Motion], frame=0, cmap='Greys_r', zero_transparent=False,
                     alpha=1, scalebar=True, title=None, color_scalebar='w',
                     show_loi=False, invert=False,
                     zoom_region: Tuple[int, int, int, int] = None,
                     inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot the Z-bands of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        cmap : str, optional
            Colormap to use. Default is 'Greys_r'.
        zero_transparent : bool, optional
            Whether to render near-zero pixels (< 0.05) as transparent. Default is False.
        alpha : float, optional
            Opacity of the image. Default is 1.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        title : str, optional
            The title for the plot. Default is None.
        color_scalebar : str, optional
            The color of the scalebar. Default is 'w'.
        show_loi : bool, optional
            Whether to show the line of interest (LOI). Default is False.
        invert : bool, optional
            If True, reverse the colormap (e.g. 'Greys_r' -> 'Greys') so the
            Z-band rendering is flipped. Default is False.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        assert sarc_obj._mask_exists('zbands'), ('Z-band mask not found. Run detect_sarcomeres first.')

        img = sarc_obj._read_mask('zbands', frames=frame)
        if zero_transparent:
            img = np.ma.masked_where(img < 0.05, img)
        cmap_use = plt.get_cmap(cmap).reversed() if invert else cmap
        ax.imshow(img, cmap=cmap_use, alpha=alpha)
        ax.set_aspect('equal')
        if show_loi:
            Plots.plot_lois(ax, sarc_obj)
        if scalebar:
            ax.add_artist(
                ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color=color_scalebar,
                         sep=1, width_fraction=0.035, location='lower right', scale_loc='top',
                         font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            PlotUtils.change_color_spines(ax_inset, 'w')
            ax_inset.imshow(img[y1:y2, x1:x2], cmap=cmap_use, alpha=alpha)
            ax_inset.set_aspect('equal')
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')

    @staticmethod
    def plot_z_bands_midlines(ax: plt.Axes, sarc_obj: Union[SarcAsM, Motion], frame=0, cmap='berlin',
                              alpha=1, scalebar=True, title=None, color_scalebar='w',
                              show_loi=True, zoom_region: Tuple[int, int, int, int] = None,
                              inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot the Z-bands and midlines of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        cmap : str, optional
            Colormap to use. Default is 'berlin'.
        alpha : float, optional
            Opacity of the image. Default is 1.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        title : str, optional
            The title for the plot. Default is None.
        color_scalebar : str, optional
            The color of the scalebar. Default is 'w'.
        show_loi : bool, optional
            Whether to show the line of interest (LOI). Default is True.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        if not sarc_obj._mask_exists('zbands'):
            raise FileNotFoundError("Z-band mask not found. Run detect_sarcomeres first.")

        zbands = sarc_obj._read_mask('zbands', frames=frame)
        midlines = sarc_obj._read_mask('mbands', frames=frame)
        joined = midlines - zbands

        ax.imshow(joined, cmap=cmap, alpha=alpha)
        ax.set_aspect('equal')

        if show_loi:
            Plots.plot_lois(ax, sarc_obj)
        if scalebar:
            ax.add_artist(
                ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color=color_scalebar,
                         sep=1, width_fraction=0.02, location='lower right', scale_loc='top',
                         font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            PlotUtils.change_color_spines(ax_inset, 'w')
            ax_inset.imshow(joined[y1:y2, x1:x2], cmap=cmap, alpha=alpha)
            ax_inset.set_aspect('equal')
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')

            if scalebar:
                ax_inset.add_artist(
                    ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color=color_scalebar,
                             sep=1, width_fraction=0.02, location='lower right', scale_loc='top',
                             font_properties={'size': PlotUtils.fontsize - 1}))

    @staticmethod
    def plot_cell_mask(ax: Axes, sarc_obj: Union[SarcAsM, Motion], frame=0, threshold=0.5, cmap='gray', alpha=1,
                       scalebar=True, title=None):
        """
        Plot the cell mask of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        threshold : float, optional
            Binarization threshold for the cell mask. Default is 0.5.
        cmap : str, optional
            The colormap to use. Default is 'gray'.
        alpha : float, optional
            Opacity of the mask. Default is 1.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        title : str, optional
            The title for the plot. Default is None.
        """
        assert sarc_obj._mask_exists('cell_mask'), ('Cell mask not found. Run detect_sarcomeres first.')

        img = sarc_obj._read_mask('cell_mask', frames=frame) > threshold
        ax.imshow(img, cmap=cmap, alpha=alpha)
        ax.set_aspect('equal')

        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_z_segmentation(ax: Axes, sarc_obj: SarcAsM, frame=0, scalebar=True, shuffle=True,
                            title=None, zoom_region: Tuple[int, int, int, int] = None,
                            inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot the Z-band segmentation result of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        shuffle : bool, optional
            Whether to shuffle the labels. Default is True.
        title : str, optional
            The title for the plot. Default is None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        assert 'z_labels' in sarc_obj.data, 'Z-bands not yet analyzed. Run analyze_z_bands first.'
        assert frame in sarc_obj.data['params.analyze_z_bands.frames'], f'Frame {frame} not yet analyzed.'

        labels = sarc_obj.data['z_labels'][frame].toarray()
        if shuffle:
            labels = Utils.shuffle_labels(labels)
        masked_labels = np.ma.masked_array(labels, mask=(labels == 0))
        cmap = plt.cm.prism
        cmap.set_bad(color=(0, 0, 0, 0))  # Set color for masked values to transparent
        ax.imshow(masked_labels, cmap=cmap)
        ax.set_aspect('equal')
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            ax_inset.imshow(masked_labels[y1:y2, x1:x2], cmap=cmap)
            ax_inset.set_aspect('equal')
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            if scalebar:
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                       width_fraction=0.02, location='lower right', scale_loc='top',
                                       font_properties={'size': PlotUtils.fontsize - 1}))

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

    @staticmethod
    def plot_z_lateral_connections(ax: Axes, sarc_obj: SarcAsM, frame=0, scalebar=True, markersize=1.5,
                                   markersize_inset=3, linewidth=0.25, linewidth_inset=0.5, plot_groups=True,
                                   shuffle=True, title=None, zoom_region: Tuple[int, int, int, int] = None,
                                   inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot lateral Z-band connections of a SarcAsM object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        markersize : float, optional
            The size of the markers of the Z-band ends. Default is 1.5.
        markersize_inset : float, optional
            The size of the markers of the Z-band ends in the inset plot. Default is 3.
        linewidth : float, optional
            The width of the connection lines. Default is 0.25.
        linewidth_inset : float, optional
            The width of the connection lines in the inset plot. Default is 0.5.
        plot_groups : bool, optional
            Whether to show the Z-bands of each lateral group with the same color. Default is True.
        shuffle : bool, optional
            Whether to shuffle the labels. Default is True.
        title : str, optional
            The title for the plot. Default is None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        assert 'z_labels' in sarc_obj.data, 'Z-bands not yet analyzed. Run analyze_z_bands first.'
        assert frame in sarc_obj.data['params.analyze_z_bands.frames'], f'Frame {frame} not yet analyzed.'

        labels = sarc_obj.data['z_labels'][frame].toarray()

        if plot_groups:
            groups = sarc_obj.data['z_lat_groups'][frame]
            labels_plot = np.zeros_like(labels)
            for i, group in enumerate(groups[1:]):
                mask = np.zeros_like(labels, dtype=bool)
                for label in group:
                    mask += (labels == label + 1)
                labels_plot[mask] = i + 1
        else:
            labels_plot = labels

        if shuffle:
            labels_plot = Utils.shuffle_labels(labels_plot)

        z_ends = sarc_obj.data['z_ends'][frame].astype('float32') / sarc_obj.metadata.pixelsize
        z_links = sarc_obj.data['z_lat_links'][frame]
        masked_labels = np.ma.masked_where(labels_plot == 0, labels_plot)
        cmap = plt.cm.prism
        cmap.set_bad(color=(0, 0, 0, 0))
        ax.imshow(masked_labels, cmap=cmap)
        ax.set_aspect('equal')
        for (i, k, j, l) in z_links.T:
            ax.plot([z_ends[i, k, 1], z_ends[j, l, 1]],
                    [z_ends[i, k, 0], z_ends[j, l, 0]],
                    c='k', lw=linewidth, linestyle='-', alpha=1, zorder=2)
        ax.scatter(z_ends[:, 0, 1], z_ends[:, 0, 0], c='k', marker='.', s=markersize, zorder=3, edgecolors='none')
        ax.scatter(z_ends[:, 1, 1], z_ends[:, 1, 0], c='k', marker='.', s=markersize, zorder=3, edgecolors='none')
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            ax_inset.imshow(masked_labels, cmap=cmap)
            ax_inset.set_aspect('equal')
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            for (i, k, j, l) in z_links.T:
                ax_inset.plot([z_ends[i, k, 1], z_ends[j, l, 1]],
                              [z_ends[i, k, 0], z_ends[j, l, 0]],
                              c='k', lw=linewidth_inset, linestyle='-', alpha=0.8, zorder=2)
            ax_inset.scatter(z_ends[:, 0, 1], z_ends[:, 0, 0], c='k', marker='.', s=markersize_inset, zorder=3,
                             edgecolors='none')
            ax_inset.scatter(z_ends[:, 1, 1], z_ends[:, 1, 0], c='k', marker='.', s=markersize_inset, zorder=3,
                             edgecolors='none')
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y2, y1)

            if scalebar:
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                       width_fraction=0.02, location='lower right', scale_loc='top',
                                       font_properties={'size': PlotUtils.fontsize - 1}))

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

    @staticmethod
    def plot_sarcomere_orientation_field(ax1: Axes, ax2: Axes, sarc_obj: SarcAsM, frame=0, cmap='vanimo',
                                         scalebar=True, colorbar=True, shrink_colorbar=0.7, orient_colorbar='vertical',
                                         zoom_region: Tuple[int, int, int, int] = None,
                                         inset_bounds=(0.6, 0.6, 0.4, 0.4),):
        """
        Plot the sarcomere orientation field (X- and Y-components) of the sarcomere object.

        Parameters
        ----------
        ax1 : matplotlib.axes.Axes
            The axes to draw the X-field component on.
        ax2 : matplotlib.axes.Axes
            The axes to draw the Y-field component on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        cmap : str, optional
            The colormap to use. Default is 'vanimo'.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        colorbar : bool, optional
            Whether to add a colorbar to the plot. Default is True.
        shrink_colorbar : float, optional
            The factor by which to shrink the colorbar. Default is 0.7.
        orient_colorbar : {'vertical', 'horizontal'}, optional
            The orientation of the colorbar. Default is 'vertical'.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        assert sarc_obj._mask_exists('orientation'), \
            'Sarcomere orientation map does not exist! Run detect_sarcomeres first.'

        orientation_field = sarc_obj._read_mask('orientation', frames=frame)

        plot1 = ax1.imshow(orientation_field[0], cmap=cmap)
        plot2 = ax2.imshow(orientation_field[1], cmap=cmap)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        if scalebar:
            ax1.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                    width_fraction=0.02, location='lower right', scale_loc='top',
                                    font_properties={'size': PlotUtils.fontsize - 1}))
            ax2.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                    width_fraction=0.02, location='lower right', scale_loc='top',
                                    font_properties={'size': PlotUtils.fontsize - 1}))

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        if colorbar:
            plt.colorbar(plot1, ax=ax1, label=r'X-Field', shrink=shrink_colorbar, orientation=orient_colorbar)
            plt.colorbar(plot2, ax=ax2, label=r'Y-Field', shrink=shrink_colorbar, orientation=orient_colorbar)

        if scalebar:
            ax1.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
            ax2.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset1 = ax1.inset_axes(bounds=inset_bounds)
            ax_inset2 = ax2.inset_axes(bounds=inset_bounds)

            ax_inset1.imshow(orientation_field[0][y1:y2, x1:x2], cmap=cmap)
            ax_inset2.imshow(orientation_field[1][y1:y2, x1:x2], cmap=cmap)
            ax_inset1.set_aspect('equal')
            ax_inset2.set_aspect('equal')
            ax_inset1.set_xticks([])
            ax_inset1.set_yticks([])
            ax_inset2.set_xticks([])
            ax_inset2.set_yticks([])

            PlotUtils.change_color_spines(ax_inset1, c='w')
            PlotUtils.change_color_spines(ax_inset2, c='w')

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax1, xlim=(x1, x2), ylim=(y1, y2), c='w')
            PlotUtils.plot_box(ax2, xlim=(x1, x2), ylim=(y1, y2), c='w')

            if scalebar:
                ax_inset1.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                        width_fraction=0.02, location='lower right', scale_loc='top',
                                        font_properties={'size': PlotUtils.fontsize - 1}))
                ax_inset2.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                        width_fraction=0.02, location='lower right', scale_loc='top',
                                        font_properties={'size': PlotUtils.fontsize - 1}))

    @staticmethod
    def plot_sarcomere_mask(ax: Axes, sarc_obj: SarcAsM, frame=0, cmap='viridis', threshold=0.1,
                            alpha=0.5, clip_thrs=(1, 99.9), scalebar=True,
                            show_image=False, show_z_bands=False,
                            invert_image=False, invert_z_bands=False,
                            cmap_image='gray', cmap_z_bands='Greys_r',
                            alpha_image=1, alpha_z_bands=1,
                            title=None, zoom_region: Tuple[int, int, int, int] = None,
                            inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot the binary mask of sarcomeres, derived from sarcomere vectors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        cmap : str, optional
            The colormap to use for the sarcomere mask. Default is 'viridis'.
        threshold : float, optional
            Binarization threshold for the sarcomere mask. If None, the threshold
            from the sarcomere vector analysis is used. Default is 0.1.
        alpha : float, optional
            The opacity of the sarcomere mask. Default is 0.5.
        clip_thrs : tuple of float, optional
            Clipping thresholds (in percentiles) for the background image, forwarded
            to :meth:`plot_image` when ``show_image=True``. Default is (1, 99.9).
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
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
        title : str, optional
            The title for the plot. Default is None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        assert sarc_obj._mask_exists('sarcomere_mask'), ('No sarcomere masks stored. '
                                                         'Run detect_sarcomeres first.')

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands,
                               clip_thrs=clip_thrs, scalebar=False)

        sarcomere_mask = sarc_obj._read_mask('sarcomere_mask', frames=frame)

        # binarize sarcomere mask
        if threshold is None:
            threshold = sarc_obj.data.get('params.analyze_sarcomere_vectors.threshold_sarcomere_mask')
        sarcomere_mask = sarcomere_mask > threshold

        # plot sarcomere mask
        sarcomere_mask = np.ma.masked_where(sarcomere_mask == 0, sarcomere_mask)

        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=(0, 0, 0, 0))
        ax.imshow(sarcomere_mask, vmin=0, vmax=1, alpha=alpha, cmap=cmap)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='w', sep=1,
                                   width_fraction=0.035, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            Plots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                                   invert_image=invert_image, invert_z_bands=invert_z_bands,
                                   cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                                   alpha_image=alpha_image, alpha_z_bands=alpha_z_bands,
                                   clip_thrs=clip_thrs, scalebar=False)
            ax_inset.set_ylim(y2, y1)
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.imshow(sarcomere_mask, vmin=0, vmax=1, alpha=alpha, cmap=cmap)
            ax_inset.set_aspect('equal')
            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')
            PlotUtils.change_color_spines(ax_inset, 'w')

    @staticmethod
    def plot_sarcomere_vectors(ax: Axes, sarc_obj: SarcAsM, frame=0, color_arrows='k',
                               color_points='darkgreen', s_points=0.5, linewidths=0.5,
                               s_points_inset=0.5, linewidths_inset=0.5, scalebar=True,
                               legend=False, title=None,
                               show_image=False, show_z_bands=False,
                               invert_image=False, invert_z_bands=False,
                               cmap_image='gray', cmap_z_bands='Greys_r',
                               alpha_image=1, alpha_z_bands=1,
                               zoom_region: Tuple[int, int, int, int] = None,
                               inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot a quiver plot of local sarcomere length and orientation from the
        sarcomere vector analysis of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        color_arrows : str, optional
            The color of the arrows. Default is 'k'.
        color_points : str, optional
            The color of the midline points. Default is 'darkgreen'.
        s_points : float, optional
            The size of midline points. Default is 0.5.
        linewidths : float, optional
            The width of the arrow lines. Default is 0.5.
        s_points_inset : float, optional
            The size of midline points in the inset plot. Default is 0.5.
        linewidths_inset : float, optional
            The width of the arrow lines in the inset plot. Default is 0.5.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
        legend : bool, optional
            Whether to add a legend to the plot. Default is False.
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
        title : str, optional
            The title for the plot. Default is None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        assert 'pos_vectors' in sarc_obj.data.keys(), ('Sarcomere vectors not yet calculated, '
                                                                 'run analyze_sarcomere_vectors first.')
        assert frame in sarc_obj.data['params.analyze_sarcomere_vectors.frames'], f'Frame {frame} not yet analyzed.'

        pos_vectors = sarc_obj.data['pos_vectors'][frame] / sarc_obj.metadata.pixelsize
        sarcomere_orientation_vectors = sarc_obj.data['sarcomere_orientation_vectors'][frame]
        sarcomere_length_vectors = sarc_obj.data['sarcomere_length_vectors'][frame] / sarc_obj.metadata.pixelsize
        orientation_vectors = np.asarray(
            [np.cos(sarcomere_orientation_vectors), -np.sin(sarcomere_orientation_vectors)])

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        ax.plot([0, 1], [0, 1], c='k', label='Z-bands', lw=0.5)

        # adjust sarcomere lengths to appear correct in quiver plot
        half_length = sarcomere_length_vectors * 0.5
        headaxislength = 4


        ax.quiver(pos_vectors[:, 1], pos_vectors[:, 0], -orientation_vectors[0] * half_length,
                  orientation_vectors[1] * half_length, width=linewidths, headaxislength=headaxislength, units='xy',
                  angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5, label='Sarcomere vectors')
        ax.quiver(pos_vectors[:, 1], pos_vectors[:, 0], orientation_vectors[0] * half_length,
                  -orientation_vectors[1] * half_length, headaxislength=headaxislength, units='xy',
                  angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5, width=linewidths)

        ax.scatter(pos_vectors[:, 1], pos_vectors[:, 0], marker='.', c=color_points, edgecolors='none', s=s_points * 0.5,
                   label='Midline pos_vectors')

        if legend:
            ax.legend(loc=3, fontsize=PlotUtils.fontsize - 2)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            linewidths *= 10
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)

            Plots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                                   invert_image=invert_image, invert_z_bands=invert_z_bands,
                                   cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                                   alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

            ax_inset.plot([0, 1], [0, 1], c='k', label='Z-bands', lw=0.5)
            ax_inset.scatter(pos_vectors[:, 1], pos_vectors[:, 0], marker='.', c=color_points, edgecolors='none',
                             s=s_points_inset, label='Midline points')
            ax_inset.quiver(pos_vectors[:, 1], pos_vectors[:, 0],
                            -orientation_vectors[0] * half_length,
                            orientation_vectors[1] * half_length, width=linewidths_inset, headaxislength=headaxislength,
                            units='xy', angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5,
                            label='Sarcomere vectors')
            ax_inset.quiver(pos_vectors[:, 1], pos_vectors[:, 0], orientation_vectors[0] * half_length,
                            -orientation_vectors[1] * half_length, headaxislength=headaxislength,
                            units='xy', angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5,
                            width=linewidths_inset)

            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y2, y1)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

            if scalebar:
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k',
                                             sep=1, width_fraction=0.02, location='lower right', scale_loc='top',
                                             font_properties={'size': PlotUtils.fontsize - 1, }))

    @staticmethod
    def plot_sarcomere_domains(ax: Axes, sarc_obj: SarcAsM, frame=0, alpha=0.5, cmap='gist_rainbow',
                               scalebar=True, title=None,
                               show_image=False, show_z_bands=False,
                               invert_image=False, invert_z_bands=False,
                               cmap_image='gray', cmap_z_bands='Greys_r',
                               alpha_image=1, alpha_z_bands=1):
        """
        Plot the sarcomere domains of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        alpha : float, optional
            The opacity of the domain masks. Default is 0.5.
        cmap : str, optional
            The colormap to use for the domain mask. Default is 'gist_rainbow'.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
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
        assert 'n_domains' in sarc_obj.data.keys(), ('Sarcomere domains not analyzed. '
                                                               'Run analyze_sarcomere_domains first.')
        assert frame in sarc_obj.data['params.analyze_sarcomere_domains.frames'], (f'Domains in frame {frame} are not yet '
                                                                           f'analyzed.')
        domains = sarc_obj.data['domains'][frame]
        pos_vectors = sarc_obj.data['pos_vectors'][frame]
        sarcomere_orientation_vectors = sarc_obj.data['sarcomere_orientation_vectors'][frame]
        sarcomere_length_vectors = sarc_obj.data['sarcomere_length_vectors'][frame]
        area_min = sarc_obj.data['params.analyze_sarcomere_domains.area_min']
        dilation_radius = sarc_obj.data['params.analyze_sarcomere_domains.dilation_radius']
        domain_mask, *_ = domain_clustering.analyze_domains(
            domains, pos_vectors, sarcomere_orientation_vectors, sarcomere_length_vectors,
            size=sarc_obj.metadata.size, pixelsize=sarc_obj.metadata.pixelsize,
            dilation_radius=dilation_radius, area_min=area_min)

        domain_mask_masked = np.ma.masked_where(domain_mask == 0, domain_mask)
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=(0, 0, 0, 0))

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        ax.imshow(domain_mask_masked, cmap=cmap, alpha=alpha, vmin=0, vmax=np.nanmax(domain_mask))
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_title(title, fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_myofibril_lines(ax: Axes, sarc_obj: SarcAsM, frame=0, linewidth=1, color_lines='r',
                             linewidth_inset=3, alpha=0.2, scalebar=True, title=None,
                             show_image=False, show_z_bands=False,
                             invert_image=False, invert_z_bands=False,
                             cmap_image='gray', cmap_z_bands='Greys_r',
                             alpha_image=1, alpha_z_bands=1,
                             zoom_region=None, inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot the result of the myofibril line growth algorithm of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        linewidth : float, optional
            The width of the lines. Default is 1.
        color_lines : str, optional
            The color of the lines. Default is 'r'.
        linewidth_inset : float, optional
            The width of the lines in the inset plot. Default is 3.
        alpha : float, optional
            The opacity of the lines. Default is 0.2.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
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
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).
        """
        assert 'myof_lines' in sarc_obj.data.keys(), ('Myofibrils not analyzed. '
                                                                'Run analyze_myofibrils first.')
        assert frame in sarc_obj.data['params.analyze_myofibrils.frames'], f'Frame {frame} not yet analyzed.'

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        lines = sarc_obj.data['myof_lines'][frame]
        pos_vectors = sarc_obj.data['pos_vectors_px'][frame]
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        for i, line_i in enumerate(lines):
            ax.plot(pos_vectors[line_i, 1], pos_vectors[line_i, 0], c=color_lines, alpha=alpha, lw=linewidth)
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)

            Plots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                                   invert_image=invert_image, invert_z_bands=invert_z_bands,
                                   cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                                   alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

            if scalebar:
                ax_inset.add_artist(
                    ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                             width_fraction=0.02, location='lower right', scale_loc='top',
                             font_properties={'size': PlotUtils.fontsize - 1}))
            for i, line_i in enumerate(lines):
                ax_inset.plot(pos_vectors[line_i, 1], pos_vectors[line_i, 0], c='r', alpha=alpha,
                              lw=linewidth_inset)

            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y2, y1)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

    @staticmethod
    def plot_myofibril_length_map(ax: Axes, sarc_obj: SarcAsM, frame=0, vmax=None, alpha=1,
                                  colorbar=True, shrink_colorbar=0.7, orient_colorbar='vertical',
                                  scalebar=True, title=None,
                                  show_image=False, show_z_bands=False,
                                  invert_image=False, invert_z_bands=False,
                                  cmap_image='gray', cmap_z_bands='Greys_r',
                                  alpha_image=1, alpha_z_bands=1,
                                  zoom_region: Tuple[int, int, int, int] = None,
                                  inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot the spatial map of myofibril lengths for a given frame.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        frame : int, optional
            The frame to plot. Default is 0.
        vmax : float, optional
            Maximum value for the colormap. If None, the maximum value in the data is used. Default is None.
        alpha : float, optional
            Opacity of the length map. Default is 1.
        colorbar : bool, optional
            Whether to show the colorbar. Default is True.
        shrink_colorbar : float, optional
            Shrinkage of the colorbar. Default is 0.7.
        orient_colorbar : {'vertical', 'horizontal'}, optional
            Orientation of the colorbar. Default is 'vertical'.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Default is True.
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
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Default is None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Default is (0.6, 0.6, 0.4, 0.4).

        Notes
        -----
        No background image is drawn by default; pass
        ``show_image=True, invert_image=True`` for an inverted raw-image backdrop.
        """
        # create myofibril length map
        assert 'myof_lines' in sarc_obj.data.keys(), ('Myofibrils not yet analyzed. '
                                                                'Run analyze_myofibrils first.')
        assert frame in sarc_obj.data['params.analyze_myofibrils.frames'], f'Frame {frame} not yet analyzed.'

        myof_lines = sarc_obj.data['myof_lines'][frame]
        myof_lengths = sarc_obj.data['myof_length'][frame]
        pos_vectors = sarc_obj.data['pos_vectors'][frame]
        orientation_vectors = sarc_obj.data['sarcomere_orientation_vectors'][frame]
        length_vectors = sarc_obj.data['sarcomere_length_vectors'][frame]
        median_filter_radius = sarc_obj.data['params.analyze_myofibrils.median_filter_radius']
        myof_length_map = myofibril_analysis.create_myofibril_length_map(
            myof_lines=myof_lines, myof_length=myof_lengths,
            pos_vectors=pos_vectors,
            sarcomere_orientation_vectors=orientation_vectors,
            sarcomere_length_vectors=length_vectors,
            size=sarc_obj.metadata.size,
            pixelsize=sarc_obj.metadata.pixelsize,
            median_filter_radius=median_filter_radius)

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        masked_myof_length_map = np.ma.masked_array(myof_length_map, mask=(myof_length_map == 0))
        cmap = plt.cm.inferno
        cmap.set_bad(color=(0, 0, 0, 0))  # Set color for masked values to transparent
        vmin, vmax = 0, np.nanmax(myof_length_map) if vmax is None else vmax
        plot = ax.imshow(masked_myof_length_map, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        ax.set_aspect('equal')
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                                   width_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        if colorbar:
            plt.colorbar(mappable=plot, ax=ax, shrink=shrink_colorbar, orientation=orient_colorbar,
                                label='Myofibril length [µm]')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)

            Plots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                                   invert_image=invert_image, invert_z_bands=invert_z_bands,
                                   cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                                   alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

            ax_inset.imshow(masked_myof_length_map, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
            ax_inset.set_aspect('equal')

            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y2, y1)

            if scalebar:
                ax_inset.add_artist(
                    ScaleBar(sarc_obj.metadata.pixelsize, units='µm', frameon=False, color='k', sep=1,
                             width_fraction=0.02, location='lower right', scale_loc='top',
                             font_properties={'size': PlotUtils.fontsize - 1}))

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

    @staticmethod
    def plot_lois(ax: Axes, sarc_obj: Union[SarcAsM, Motion], color='darkorange', linewidth=2, alpha=0.5):
        """
        Plot all LOI lines of a SarcAsM or Motion object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the LOI lines on.
        sarc_obj : SarcAsM or Motion
            The object to plot.
        color : str, optional
            Color of the lines. Default is 'darkorange'.
        linewidth : float, optional
            Width of the lines. Default is 2.
        alpha : float, optional
            Opacity of the lines. Default is 0.5.
        """
        loi_lines = None

        if hasattr(sarc_obj, 'loi_data'):
            # Extract line data directly from sarc_obj.loi_data. Chains synthesized by
            # SarcAsM.get_track_motion carry no 'line' polyline, so this must not raise.
            line = sarc_obj.loi_data.get('line')
            loi_lines = None if line is None else [line]
        elif hasattr(sarc_obj, 'data') and 'loi_data' in sarc_obj.data:
            # Extract lines from sarc_obj.data['loi_data']
            loi_lines = sarc_obj.data['loi_data'].get('loi_lines', [])

        if loi_lines is not None:
            # Plot each line
            for line in loi_lines:
                ax.plot(line.T[1], line.T[0], color=color, linewidth=linewidth, alpha=alpha)
        else:
            # Raise a warning if no LOI lines are found
            warnings.warn("No LOI lines found in the provided object.", UserWarning)


    @staticmethod
    def plot_histogram_structure(ax: Axes,
                                 sarc_obj: SarcAsM,
                                 feature: str,
                                 frame: int = 0,
                                 bins: int = 20,
                                 density: bool = False,
                                 range: Optional[tuple] = None,
                                 label: Optional[str] = None,
                                 ylabel: Optional[str] = None,
                                 rwidth: float = 0.6,
                                 color: str = 'darkslategray',
                                 edge_color: str = 'k',
                                 align: Literal['mid', 'left', 'right'] = 'mid',
                                 rotate_yticks: bool = False) -> None:
        """
        Plot a histogram of a specified structural feature from a sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the histogram on.
        sarc_obj : SarcAsM
            The instance of SarcAsM class to plot.
        feature : str
            The name of the structural feature to plot.
        frame : int, optional
            The frame index from which to extract the data. Default is 0.
        bins : int, optional
            The number of bins for the histogram. Default is 20.
        density : bool, optional
            If True, normalize the histogram to a probability density rather than raw counts. Default is False.
        range : tuple, optional
            The lower and upper range of the bins. If None, determined from the data. Default is None.
        label : str, optional
            The label for the x-axis. If None, a default label based on the feature is used. Default is None.
        ylabel : str, optional
            The label for the y-axis. Overrides the default label if provided. Default is None.
        rwidth : float, optional
            The relative width of the histogram bars. Default is 0.6.
        color : str, optional
            The fill color of the histogram bars. Default is 'darkslategray'.
        edge_color : str, optional
            The color of the edges of the histogram bars. Default is 'k'.
        align : {'mid', 'left', 'right'}, optional
            The alignment of the histogram bars. Default is 'mid'.
        rotate_yticks : bool, optional
            If True, rotate the y-axis tick labels by 90 degrees. Default is False.
        """
        data = sarc_obj.data[feature][frame]
        # Flatten data if it has more than one dimension
        if data.ndim > 1:
            data = data.flatten()
        # Remove NaN values from the data
        data = data[~np.isnan(data)]

        ax.hist(
            data,
            bins=bins,
            density=density,
            range=range,
            rwidth=rwidth,
            color=color,
            edgecolor=edge_color,
            align=align
        )

        # Use a default label if none is provided
        if label is None:
            label = structure_feature_dict.get(feature, {}).get('name', feature)
        ax.set_xlabel(label)

        # Set y-axis label based on whether density is True
        ax.set_ylabel('Frequency' if density else 'Count')
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if rotate_yticks:
            ax.tick_params(axis='y', labelrotation=90)
            plt.setp(ax.get_yticklabels(), va='center')

        PlotUtils.remove_spines(ax)

    @staticmethod
    def _contr_window(motion_obj: Motion, number_contr, t_lim):
        """Frame window of a contraction-centered view, clamped to the recording.

        Parameters
        ----------
        motion_obj : Motion
            Object holding ``loi_data['start_contr']`` / ``['n_contr']`` / ``['time']``.
        number_contr : int or None
            Index of the contraction to center on; None disables centering.
        t_lim : tuple of float
            Window around the contraction onset, in seconds (may start negative).

        Returns
        -------
        tlim : tuple
            Absolute time limits, or ``(None, None)`` when not centering.
        i0, i1 : int or None
            Frame slice bounds, or ``(None, None)`` when not centering, so that
            ``arr[i0:i1]`` yields the full series.

        Notes
        -----
        A cycle beginning at or near frame 0 — which the incomplete-cycle handling
        deliberately keeps — makes ``tlim[0]`` negative. An unclamped negative start
        index is interpreted as "from the end" and silently produces an EMPTY slice,
        so both bounds are clamped into ``[0, len(time)]``.
        """
        if number_contr is None or motion_obj.loi_data['n_contr'] <= 0:
            return (None, None), None, None
        start_contr_t = motion_obj.loi_data['start_contr'][number_contr]
        tlim = (start_contr_t + t_lim[0], start_contr_t + t_lim[1])
        n = len(motion_obj.loi_data['time'])
        frametime = motion_obj.metadata.frametime
        i0 = min(max(0, int(tlim[0] / frametime)), n)
        i1 = min(max(i0, int(tlim[1] / frametime)), n)
        return tlim, i0, i1

    @staticmethod
    def plot_z_pos(ax: Axes, motion_obj: Motion, number_contr=None, show_contr=True, show_kymograph=False, color='k',
                   t_lim=(None, None), y_lim=(None, None)):
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
        show_kymograph : bool, optional
            Whether to show the kymograph. Default is False.
        color : str, optional
            The color of the trajectories. Default is 'k'.
        t_lim : tuple, optional
            The time limits for the plot in seconds. Default is (None, None).
        y_lim : tuple, optional
            The y-axis limits for the plot. Default is (None, None).
        """
        # plot limits and params
        tlim, i0, i1 = Plots._contr_window(motion_obj, number_contr, t_lim)
        centered = i0 is not None

        if show_kymograph:
            ax.pcolorfast(motion_obj.loi_data['time'], motion_obj.loi_data['x_pos'], motion_obj.loi_data['y_int'].T,
                          cmap='Greys')
        # get data
        time = motion_obj.loi_data['time']
        z_pos = motion_obj.loi_data['z_pos']
        # plot contraction cycles
        if show_contr:
            Plots._shade_contr_loi(ax, motion_obj, t_offset=tlim[0] if centered else 0.0)

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
    def _plot_delta_slen_loi(ax: Axes, motion_obj: Motion, frame=None, t_lim=(0, 12), y_lim=(-0.3, 0.4), n_rows=6,
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
            The time limits for the plot in seconds. Default is (0, 12).
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
        list_y = np.linspace(0, 1, num=n_rows, endpoint=False)
        for i, y in enumerate(list_y):
            ax_i = ax.inset_axes((0., y, 1, 1 / n_rows - min(0.02, 0.3 / n_rows)))
            ax_i.plot(motion_obj.loi_data['time'], delta_slen[i + n_start], c='k', lw=0.6)
            ax_i.axhline(0, linewidth=1, linestyle=':', c='k')
            if show_contr:
                Plots._shade_contr_loi(ax_i, motion_obj)

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
        ax.set_ylabel('$\Delta$SL [µm]')
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
            The time limits for the plot in seconds. Default is (0, 1).
        y_lim : tuple, optional
            The y-axis limits for the plot. Default is (-0.35, 0.5).
        show_contr : bool, optional
            Whether to shade the contraction periods. Default is True.
        """
        # plot limits and params
        tlim, i0, i1 = Plots._contr_window(motion_obj, number_contr, t_lim)
        centered = i0 is not None
        # get data
        time = motion_obj.loi_data['time']
        delta_slen = motion_obj.loi_data['delta_slen']
        delta_slen_avg = motion_obj.loi_data['delta_slen_avg']
        # plot contraction cycles
        if show_contr:
            Plots._shade_contr_loi(ax, motion_obj, t_offset=tlim[0] if centered else 0.0)

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
        ax.set_ylabel('$\Delta$SL [µm]')
        ax.set_ylim(y_lim)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_overlay_velocity(ax, motion_obj: Motion, number_contr=None, t_lim=(0, 0.9), y_lim=(-9, 12),
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
            The time limits for the plot in seconds. Default is (0, 0.9).
        y_lim : tuple, optional
            The y-axis limits for the plot. Default is (-9, 12).
        show_contr : bool, optional
            Whether to shade the contraction periods. Default is True.
        """
        # plot limits and params
        tlim, i0, i1 = Plots._contr_window(motion_obj, number_contr, t_lim)
        centered = i0 is not None
        # get data
        time = motion_obj.loi_data['time']
        vel = motion_obj.loi_data['vel']
        vel_avg = motion_obj.loi_data['vel_avg']

        # plot contraction cycles
        if show_contr:
            Plots._shade_contr_loi(ax, motion_obj, t_offset=tlim[0] if centered else 0.0)

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
            ax.set_xlim(0, time.max())
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('V [µm/s]')
        ax.set_ylim(y_lim)
        PlotUtils.polish_yticks(ax, 3, 1)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_domain_timeseries(ax: Axes, sarc_obj: SarcAsM, t_lim: Tuple[float, float] = (0, 12),
                               y_lim: Tuple[float, float] = (1.6, 2.2), n_rows: Optional[int] = None,
                               show_contr: bool = True, use_median: bool = False):
        """
        Plots domain sarcomere length time-series in a stacked multi-subplot layout.

        Each domain's sarcomere length time-series is shown in a separate row, with optional
        contraction period shading and a dotted line at the domain's equilibrium length
        (its median length over the non-contracting frames). Same rendering as
        :meth:`plot_slen_mean`, with the rows numbered like the domain masks.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The SarcAsM object with domain motion analysis results.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds. Default is (0, 12).
        y_lim : tuple of float, optional
            The y-axis limits for sarcomere length in µm. Default is (1.6, 2.2).
        n_rows : int, optional
            Number of domains to display. If None, all domains are shown. Default is None.
        show_contr : bool, optional
            Whether to shade contraction periods. Default is True.
        use_median : bool, optional
            If True, use median sarcomere length instead of mean. Default is False.

        Raises
        ------
        ValueError
            If domain motion analysis has not been run.
        """
        # Validate prerequisites
        if 'domain_slen_timeseries' not in sarc_obj.data:
            raise ValueError("Domain motion analysis not run. Call analyze_track_motion(by='domain') first.")

        # Get data
        key = 'domain_slen_median_timeseries' if use_median else 'domain_slen_timeseries'
        slen_timeseries = np.asarray(sarc_obj.data[key], dtype=float)
        domain_contr = sarc_obj.data.get('domain_contr', None)
        Plots._plot_group_stacked(ax, sarc_obj, 'domain', slen_timeseries, domain_contr, _LABEL_SL,
                                  t_lim, y_lim, n_rows, show_contr, label_offset=1)

    @staticmethod
    def plot_overlay_domain_timeseries(ax: Axes, sarc_obj: SarcAsM, t_lim: Tuple[float, float] = (0, 12),
                                       y_lim: Tuple[float, float] = (1.4, 2.2), show_contr: bool = True,
                                       show_average: bool = True, use_median: bool = False,
                                       domain_indices: Optional[list] = None):
        """
        Plots domain sarcomere length time-series as overlaid trajectories.

        All domain time-series are plotted on the same axes with different colors,
        optionally with an average line and contraction period shading.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            The SarcAsM object with domain motion analysis results.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds. Default is (0, 12).
        y_lim : tuple of float, optional
            The y-axis limits for sarcomere length in µm. Default is (1.4, 2.2).
        show_contr : bool, optional
            Whether to shade contraction periods (union of all domain contractions). Default is True.
        show_average : bool, optional
            Whether to show the average across all domains. Default is True.
        use_median : bool, optional
            If True, use median sarcomere length instead of mean. Default is False.
        domain_indices : list of int, optional
            Domain indices (0-based) to plot. If None, all domains are plotted. Default is None.

        Raises
        ------
        ValueError
            If domain motion analysis has not been run.
        """
        # Validate prerequisites
        if 'domain_slen_timeseries' not in sarc_obj.data:
            raise ValueError("Domain motion analysis not run. Call analyze_track_motion(by='domain') first.")

        # Get data
        if use_median:
            slen_timeseries = sarc_obj.data['domain_slen_median_timeseries']
        else:
            slen_timeseries = sarc_obj.data['domain_slen_timeseries']
        n_domains, n_frames = slen_timeseries.shape
        time = np.arange(n_frames) * sarc_obj.metadata.frametime

        # Select domains to plot
        if domain_indices is None:
            domain_indices = list(range(n_domains))
        domain_indices = [i for i in domain_indices if 0 <= i < n_domains]

        # Get contraction data if available
        domain_contr = sarc_obj.data.get('domain_contr', None)

        # Shade contraction periods (union across selected domains)
        if show_contr and domain_contr is not None:
            any_contr = np.any(domain_contr[domain_indices], axis=0)
            ax.fill_between(time, y_lim[0], y_lim[1], where=any_contr, color='lavender', alpha=0.5)

        # Domain colormap
        cm = plt.cm.gist_rainbow(np.linspace(0, 1, n_domains))

        # Plot individual domain trajectories
        for domain_idx in domain_indices:
            ax.plot(time, slen_timeseries[domain_idx], c=cm[domain_idx], lw=0.8,
                    label=f'Domain {domain_idx + 1}', alpha=0.8)

        # Plot average trajectory
        if show_average and len(domain_indices) > 1:
            avg_slen = np.nanmean(slen_timeseries[domain_indices], axis=0)
            ax.plot(time, avg_slen, c='k', lw=2, linestyle='-', label='Average')

        # Configure axes
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Sarcomere length [µm]')
        ax.set_xlim(t_lim)
        ax.set_ylim(y_lim)
        PlotUtils.polish_xticks(ax)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)

        # Add legend
        ax.legend(loc='upper right', fontsize='x-small')

    @staticmethod
    def plot_phase_space(ax: Axes, motion_obj: Motion, t_lim=(0, 4), number_contr=None, frame=None):
        """
        Plot the sarcomere trajectory in length-change vs. velocity phase space.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        t_lim : tuple, optional
            The time limits for the plot in seconds. Default is (0, 4).
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
        _, i0, i1 = Plots._contr_window(motion_obj, number_contr, t_lim)
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
        ax.set_ylabel('Length change $\Delta SL$ [µm]', fontsize=PlotUtils.fontsize)

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
    # 2D sarcomere tracking + grouped motion (track_sarcomere_vectors ->
    # group_tracks -> analyze_track_motion).
    # ------------------------------------------------------------------

    @staticmethod
    def plot_tracks(ax: Axes, sarc_obj: SarcAsM, frame: int = 0, color_by: str = 'coverage',
                    cmap: str = 'viridis', linewidth: float = 0.8, only_observed: bool = True,
                    max_tracks: Optional[int] = 2000, alpha: float = 0.8,
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
            SarcAsM with :meth:`SarcAsM.track_sarcomere_vectors` results.
        frame : int, optional
            Movie frame used for the **background** image/Z-bands only (the
            trajectories always span the full movie). Default is 0.
        color_by : {'coverage', 'slen', 'group'}, optional
            Per-track colour: observation coverage, the track's mean sarcomere length, or
            ``track_group_id`` (requires :meth:`SarcAsM.group_tracks`). Default is 'coverage'.
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
        if 'tracks_positions_px' not in sarc_obj.data:
            raise ValueError('No tracks found. Run track_sarcomere_vectors first.')
        n_tracks = int(sarc_obj.data.get('n_tracks', 0))
        pos = np.asarray(sarc_obj.data['tracks_positions_px'], dtype=float).reshape(n_tracks, -1, 2)
        observed = np.asarray(sarc_obj.data['tracks_observed']).reshape(n_tracks, -1).astype(bool)
        n_t = pos.shape[1]

        # Per-track colour scalar (computed once per track, not per frame).
        if color_by == 'slen':
            slen = np.asarray(sarc_obj.data['tracks_slen'], dtype=float).reshape(n_tracks, -1)
            with np.errstate(invalid='ignore'):
                c = np.nanmean(np.where(observed, slen, np.nan), axis=1)
            clabel = 'Sarcomere length [µm]'
        elif color_by == 'group':
            if 'track_group_id' not in sarc_obj.data:
                raise ValueError("color_by='group' requires group_tracks() first.")
            c = np.asarray(sarc_obj.data['track_group_id']).reshape(-1).astype(float)
            clabel = 'Group id'
        else:  # coverage
            c = np.asarray(sarc_obj.data['track_lengths'], dtype=float) / float(n_t)
            clabel = 'Track coverage'

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

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        lc = None
        norm = None
        if segments:
            vals = np.asarray(seg_vals, dtype=float)
            if color_by == 'group':
                n_groups = max(int(sarc_obj.data.get('n_groups', 0)), 1)
                gcm = plt.get_cmap('gist_rainbow')
                colors = gcm((vals.astype(int) % n_groups) / n_groups)
                lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
            else:
                finite = vals[np.isfinite(vals)]
                if finite.size:
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
    def plot_track_groups(ax: Axes, sarc_obj: SarcAsM, frame: int = 0, cmap: str = 'gist_rainbow',
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
        :meth:`SarcAsM.analyze_track_motion`. Requires :meth:`SarcAsM.group_tracks`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            SarcAsM with a track grouping (:meth:`SarcAsM.group_tracks`).
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
        if 'track_group_id' not in sarc_obj.data:
            raise ValueError('No track grouping found. Run group_tracks(...) first.')
        n_tracks = int(sarc_obj.data.get('n_tracks', 0))
        t = sarc_obj._tracked_frame_index(frame)
        n_groups = int(sarc_obj.data.get('n_groups', 0))

        pos = np.asarray(sarc_obj.data['tracks_positions_px'], dtype=float).reshape(n_tracks, -1, 2)
        gid = np.asarray(sarc_obj.data['track_group_id']).reshape(-1)
        yx = pos[:, t]
        finite = np.isfinite(yx[:, 0]) & np.isfinite(yx[:, 1])

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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
        kind = sarc_obj.data.get('group_kind', '')
        ax.set_title(title if title is not None else f"Track groups ('{kind}', n={n_groups})",
                     fontsize=PlotUtils.fontsize)

    @staticmethod
    def _plot_group_stacked(ax, sarc_obj, kind, matrix, group_contr, ylabel,
                            t_lim, y_lim, n_rows, show_contr, hline='equ', label_offset=0):
        """Stacked per-group time-series (shared by plot_slen_mean / plot_delta_slen_mean).

        ``hline`` controls the dotted reference line per row: ``'equ'`` = the
        equilibrium length (median over the non-contracting frames), ``'zero'`` =
        the ΔSL=0 line (which *is* the equilibrium in delta space), ``None`` = no line.
        ``label_offset`` shifts the row labels (1 gives the 1-based numbering the
        domain masks use).
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
            ax_i.set_yticks(yticks)
            ax_i.set_yticklabels([f'{yt:.2f}' for yt in yticks], fontsize='x-small')
            ax_i.text(0.02, 0.85, f'{kind[0].upper()}{g + label_offset}', transform=ax_i.transAxes,
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
        """Resolve and validate the grouping kind, defaulting to the last analyzed one."""
        sarc_obj._assert_track_motion_fresh()
        if kind is None:
            kind = sarc_obj.data.get('track_motion_kind')
        if f'{kind}_slen_timeseries' not in sarc_obj.data:
            raise ValueError(f"No '{kind}' track-motion results found. "
                             f"Run analyze_track_motion(by='{kind}') first.")
        return kind

    @staticmethod
    def plot_slen_mean(ax: Axes, sarc_obj: SarcAsM, kind: Optional[str] = None,
                       t_lim: Tuple[float, float] = (0, 12), y_lim: Tuple[float, float] = (1.3, 2.0),
                       n_rows: Optional[int] = None, show_contr: bool = True,
                       use_median: bool = False):
        """
        Plot stacked per-group MEAN sarcomere-length time-series from :meth:`SarcAsM.analyze_track_motion`.

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
        kind : str, optional
            Grouping prefix ('pool', 'mband', ...). If None, the last analyzed
            grouping (``track_motion_kind``) is used. Default is None.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds. Default is (0, 12).
        y_lim : tuple of float, optional
            The y-axis limits for sarcomere length in µm. Default is (1.3, 2.0).
        n_rows : int, optional
            Number of groups to display. If None, all groups are shown. Default is None.
        show_contr : bool, optional
            Whether to shade contraction periods. Default is True.
        use_median : bool, optional
            If True, use median sarcomere length instead of mean. Default is False.
        """
        kind = Plots._resolve_kind(sarc_obj, kind)
        key = f'{kind}_slen_median_timeseries' if use_median else f'{kind}_slen_timeseries'
        matrix = np.asarray(sarc_obj.data[key])
        group_contr = sarc_obj.data.get(f'{kind}_contr', None)
        Plots._plot_group_stacked(ax, sarc_obj, kind, matrix, group_contr, _LABEL_SL,
                                  t_lim, y_lim, n_rows, show_contr)

    @staticmethod
    def plot_delta_slen_mean(ax: Axes, sarc_obj: SarcAsM, kind: Optional[str] = None,
                             t_lim: Tuple[float, float] = (0, 12), y_lim: Tuple[float, float] = (-0.4, 0.4),
                             n_rows: Optional[int] = None, show_contr: bool = True,
                             use_median: bool = False):
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
        kind : str, optional
            Grouping prefix ('pool', 'mband', ...). If None, the last analyzed
            grouping (``track_motion_kind``) is used. Default is None.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds. Default is (0, 12).
        y_lim : tuple of float, optional
            The y-axis limits for ΔSL in µm. Default is (-0.4, 0.4).
        n_rows : int, optional
            Number of groups to display. If None, all groups are shown. Default is None.
        show_contr : bool, optional
            Whether to shade contraction periods. Default is True.
        use_median : bool, optional
            If True, use median sarcomere length instead of mean. Default is False.
        """
        kind = Plots._resolve_kind(sarc_obj, kind)
        key = f'{kind}_slen_median_timeseries' if use_median else f'{kind}_slen_timeseries'
        slen_ts = np.asarray(sarc_obj.data[key], dtype=float)
        contr = np.asarray(sarc_obj.data[f'{kind}_contr']) if f'{kind}_contr' in sarc_obj.data else None
        delta = np.full_like(slen_ts, np.nan)
        for g in range(slen_ts.shape[0]):
            c = contr[g] if contr is not None else np.zeros(slen_ts.shape[1], dtype=bool)
            delta[g] = slen_ts[g] - grouped_motion.equilibrium_over_quiet(slen_ts[g], c)
        Plots._plot_group_stacked(ax, sarc_obj, kind, delta, contr, _LABEL_DELTA_SL,
                                  t_lim, y_lim, n_rows, show_contr, hline='zero')

    @staticmethod
    def _track_group_overlay(ax, sarc_obj, *, mode, group=0, kind=None,
                             t_lim=(0, 12), y_lim=None, show_contr=True, show_mean=True,
                             max_lines=300, color=None, mean_color='k'):
        """Overlay the individual member sarcomeres of one track group + the group mean.

        ``mode='slen'`` plots SL(t); ``mode='delta'`` plots ΔSL(t) = SL(t) − equ
        (per-member equilibrium over the group's non-contracting frames). The bold
        overlay is the group aggregate over *all* members; individual member lines
        are subsampled (longest-coverage first) to ``max_lines`` for legibility.
        """
        kind = Plots._resolve_kind(sarc_obj, kind)
        if y_lim is None:
            y_lim = (-0.4, 0.4) if mode == 'delta' else (1.6, 2.2)
        n_tracks = int(sarc_obj.data['n_tracks'])
        n_groups = int(sarc_obj.data.get('n_groups', 0))
        if not (0 <= group < max(n_groups, 1)):
            raise ValueError(f'group {group} out of range [0, {n_groups}).')
        gid = np.asarray(sarc_obj.data['track_group_id']).reshape(-1)
        slen_tracks = np.asarray(sarc_obj.data['tracks_slen'], dtype=float).reshape(n_tracks, -1)
        T = slen_tracks.shape[1]
        time = np.arange(T) * sarc_obj.metadata.frametime
        agg = np.asarray(sarc_obj.data[f'{kind}_slen_timeseries'], dtype=float)[group]
        contr = (np.asarray(sarc_obj.data[f'{kind}_contr'])[group]
                 if f'{kind}_contr' in sarc_obj.data else np.zeros(T, dtype=bool))

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
            Plots._shade_contr_loi(ax, motion_obj)
        col = color if color is not None else '0.6'
        alpha = float(max(0.04, min(0.5, 30.0 / max(len(slen), 1))))
        ax.plot(time, slen.T, c=col, lw=0.4, alpha=alpha)
        if show_mean:
            ax.plot(time, slen_avg, c=mean_color, lw=2, zorder=3)
        ax.set_xlim(t_lim if t_lim != (None, None) else (float(time.min()), float(time.max())))
        ax.set_ylim(y_lim)
        ax.set_xlabel(_LABEL_TIME)
        ax.set_ylabel(_LABEL_SL)
        PlotUtils.polish_xticks(ax)

    @staticmethod
    def plot_slen(ax: Axes, obj: Union[SarcAsM, Motion], *, group: int = 0, kind: Optional[str] = None,
                  t_lim: Tuple[float, float] = (0, 12), y_lim: Tuple[float, float] = (1.4, 2.2),
                  show_contr: bool = True, show_mean: bool = True, max_lines: Optional[int] = 300,
                  color: Optional[str] = None, mean_color: str = 'k'):
        """
        Plot individual sarcomere-length traces with the mean overlaid.

        Polymorphic — pass either object:

        * **SarcAsM** (track grouping): overlays the member sarcomeres of one
          track ``group`` (``kind`` defaults to the last analyzed grouping), with
          the group aggregate drawn bold. Members are subsampled to ``max_lines``
          (longest-coverage first) for legibility. Requires
          :meth:`SarcAsM.analyze_track_motion`.
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
            Grouping prefix (SarcAsM only). If None, the last analyzed grouping is
            used. Default is None.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds. Default is (0, 12).
        y_lim : tuple of float, optional
            The y-axis limits for sarcomere length in µm. Default is (1.4, 2.2).
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
            return Plots._plot_slen_loi(ax, obj, t_lim=t_lim if t_lim != (0, 12) else (None, None),
                                        y_lim=y_lim if y_lim != (1.4, 2.2) else (None, None),
                                        show_contr=show_contr, show_mean=show_mean,
                                        color=color, mean_color=mean_color)
        return Plots._track_group_overlay(ax, obj, mode='slen', group=group, kind=kind,
                                          t_lim=t_lim, y_lim=y_lim, show_contr=show_contr,
                                          show_mean=show_mean, max_lines=max_lines,
                                          color=color, mean_color=mean_color)

    @staticmethod
    def plot_delta_slen(ax: Axes, obj: Union[SarcAsM, Motion], *, group: int = 0, kind: Optional[str] = None,
                        t_lim: Tuple[float, float] = (0, 12), y_lim: Tuple[float, float] = (-0.4, 0.4),
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
          Requires :meth:`SarcAsM.analyze_track_motion`.
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
            Grouping prefix ('pool', 'mband', ...) (SarcAsM only). If None, the last
            analyzed grouping is used. Default is None.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds. Default is (0, 12).
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
            return Plots._plot_delta_slen_loi(ax, obj, frame=frame, t_lim=t_lim, y_lim=y_lim,
                                              n_rows=n_rows, n_start=n_start, show_contr=show_contr)
        return Plots._track_group_overlay(ax, obj, mode='delta', group=group, kind=kind,
                                          t_lim=t_lim, y_lim=y_lim, show_contr=show_contr,
                                          show_mean=show_mean, max_lines=max_lines,
                                          color=color, mean_color=mean_color)

    @staticmethod
    def plot_track_myofibrils(ax: Axes, sarc_obj: SarcAsM, frame: int = 0,
                              color_by: str = 'group', cmap: str = 'gist_rainbow',
                              linewidth: float = 1.5, show_points: bool = True, markersize: float = 6,
                              only_observed: bool = False,
                              scalebar: bool = True, colorbar: bool = False, title: Optional[str] = None,
                              show_image: bool = False, show_z_bands: bool = False,
                              invert_image: bool = False, invert_z_bands: bool = False,
                              cmap_image: str = 'gray', cmap_z_bands: str = 'Greys_r',
                              alpha_image: float = 1, alpha_z_bands: float = 1):
        """
        Draw each tracked myofibril (fibre) as a connected polyline over the image.

        The tracker analogue of :meth:`plot_myofibril_lines`: each fibre's member
        sarcomeres — ordered head-to-tail by ``track_group_order`` — are linked by a
        line at ``frame``, so you can see where each analyzed fibre runs and its
        shape. Requires a ``'myofibril'`` grouping
        (:meth:`SarcAsM.group_tracks` / :meth:`SarcAsM.analyze_track_motion`
        with ``by='myofibril'``).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
            SarcAsM with a myofibril grouping.
        frame : int, optional
            Movie frame to draw. Default is 0.
        color_by : {'group', 'slen', 'beating_rate'}, optional
            Per-fibre colour: fibre id, the fibre's median sarcomere length at this
            frame, or its beating rate (requires ``analyze_track_motion(by='myofibril')``).
            Default is 'group'.
        cmap : str, optional
            Colormap. Default is 'gist_rainbow'.
        linewidth : float, optional
            Width of the fibre polylines. Default is 1.5.
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
        if sarc_obj.data.get('group_kind') != 'myofibril':
            raise ValueError("plot_track_myofibrils requires a 'myofibril' grouping. "
                             "Run group_tracks(by='myofibril') first.")
        n_tracks = int(sarc_obj.data['n_tracks'])
        t = sarc_obj._tracked_frame_index(frame)
        n_groups = int(sarc_obj.data.get('n_groups', 0))

        pos = np.asarray(sarc_obj.data['tracks_positions_px'], dtype=float).reshape(n_tracks, -1, 2)[:, t]
        gid = np.asarray(sarc_obj.data['track_group_id']).reshape(-1)
        order = np.asarray(sarc_obj.data['track_group_order']).reshape(-1)
        observed = np.asarray(sarc_obj.data['tracks_observed']).reshape(n_tracks, -1)[:, t].astype(bool)
        slen_t = np.asarray(sarc_obj.data['tracks_slen'], dtype=float).reshape(n_tracks, -1)[:, t]

        # Per-fibre colour value.
        metric = None
        clabel = ''
        if color_by == 'beating_rate':
            metric = np.asarray(sarc_obj.data.get('myofibril_beating_rate', np.full(n_groups, np.nan)), dtype=float)
            clabel = 'Beating rate [Hz]'
        elif color_by == 'slen':
            with np.errstate(invalid='ignore'):
                metric = np.array([np.nanmedian(slen_t[gid == g]) if np.any(gid == g) else np.nan
                                   for g in range(n_groups)])
            clabel = 'Sarcomere length [µm]'

        Plots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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
