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

"""Plots of the sarcomere structure of a recording (masks, Z-bands, vectors, myofibrils, domains)."""

import warnings
from typing import Union, Tuple, Optional, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar

from sarcasm.features import structure_feature_dict
from sarcasm.plotting.plot_utils import PlotUtils
from sarcasm.core import SarcAsM
from sarcasm.analysis import domain_clustering, myofibril_analysis
from sarcasm.utils import Utils

_LABEL_TIME = 'Time [s]'
_LABEL_SL = 'Sarcomere length SL [µm]'
_LABEL_DELTA_SL = r'$\Delta$SL [µm]'


class StructurePlots:
    """Plots of the sarcomere structure of a recording (masks, Z-bands, vectors, myofibrils, domains). Part of :class:`~sarcasm.plotting.plots.Plots`."""

    @staticmethod
    def _frame(sarc_obj, frame) -> int:
        """The frame argument of a plot as one validated frame number: an int, numpy integer, 0-d array or
        one-element sequence (``[k]`` from a parsed frame list) all name frame ``k``."""
        if hasattr(sarc_obj, '_canonical_frame'):
            return sarc_obj._canonical_frame(frame)
        return int(frame)

    @staticmethod
    def plot_image(ax: Axes, sarc_obj: SarcAsM, frame: int = 0, cmap: str = 'gray',
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
        sarc_obj : SarcAsM
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
        frame = StructurePlots._frame(sarc_obj, frame)

        img = sarc_obj.read_imgs(frames=frame)
        img = np.clip(img, np.percentile(img, clip_thrs[0]), np.percentile(img, clip_thrs[1]))

        cmap_use = plt.get_cmap(cmap).reversed() if invert else cmap
        _ = ax.imshow(img, cmap=cmap_use, alpha=alpha)
        ax.set_aspect('equal')
        if show_loi:
            StructurePlots.plot_lois(ax, sarc_obj)
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
    def _draw_background(ax: Axes, sarc_obj: SarcAsM, frame: int = 0, *,
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
        sarc_obj : SarcAsM
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
            StructurePlots.plot_image(ax, sarc_obj, frame=frame, **kw)
        elif show_z_bands:
            StructurePlots.plot_z_bands(ax, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands,
                               invert=invert_z_bands, scalebar=scalebar)
        else:
            # no imshow to establish image coordinates: set them (y down, full extent)
            size = getattr(sarc_obj.metadata, 'size', None)
            if size is not None and len(size) >= 2:
                h, w = int(size[-2]), int(size[-1])
                ax.set_xlim(-0.5, w - 0.5)
                ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect('equal')

    @staticmethod
    def plot_z_bands(ax: plt.Axes, sarc_obj: SarcAsM, frame=0, cmap='Greys_r', zero_transparent=False,
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
        sarc_obj : SarcAsM
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert sarc_obj.mask_exists('zbands'), ('Z-band mask not found. Run detect_sarcomeres first.')

        img = sarc_obj.read_mask('zbands', frames=frame)
        if zero_transparent:
            img = np.ma.masked_where(img < 0.05, img)
        cmap_use = plt.get_cmap(cmap).reversed() if invert else cmap
        ax.imshow(img, cmap=cmap_use, alpha=alpha)
        ax.set_aspect('equal')
        if show_loi:
            StructurePlots.plot_lois(ax, sarc_obj)
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
    def plot_z_bands_midlines(ax: plt.Axes, sarc_obj: SarcAsM, frame=0, cmap='berlin',
                              alpha=1, scalebar=True, title=None, color_scalebar='w',
                              show_loi=True, zoom_region: Tuple[int, int, int, int] = None,
                              inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plot the Z-bands and midlines of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
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
        frame = StructurePlots._frame(sarc_obj, frame)
        if not sarc_obj.mask_exists('zbands'):
            raise FileNotFoundError("Z-band mask not found. Run detect_sarcomeres first.")

        zbands = sarc_obj.read_mask('zbands', frames=frame)
        midlines = sarc_obj.read_mask('mbands', frames=frame)
        joined = midlines - zbands

        ax.imshow(joined, cmap=cmap, alpha=alpha)
        ax.set_aspect('equal')

        if show_loi:
            StructurePlots.plot_lois(ax, sarc_obj)
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
    def plot_cell_mask(ax: Axes, sarc_obj: SarcAsM, frame=0, threshold=0.5, cmap='gray', alpha=1,
                       scalebar=True, title=None):
        """
        Plot the cell mask of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : SarcAsM
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert sarc_obj.mask_exists('cell_mask'), ('Cell mask not found. Run detect_sarcomeres first.')

        img = sarc_obj.read_mask('cell_mask', frames=frame) > threshold
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert 'structure.zbands.labels' in sarc_obj.data, 'Z-bands not yet analyzed. Run sarc.structure.analyze_z_bands first.'
        assert frame in sarc_obj.data['params.structure.analyze_z_bands.frames'], f'Frame {frame} not yet analyzed.'

        labels = sarc_obj.data['structure.zbands.labels'][frame].toarray()
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert 'structure.zbands.labels' in sarc_obj.data, 'Z-bands not yet analyzed. Run sarc.structure.analyze_z_bands first.'
        assert frame in sarc_obj.data['params.structure.analyze_z_bands.frames'], f'Frame {frame} not yet analyzed.'

        labels = sarc_obj.data['structure.zbands.labels'][frame].toarray()

        if plot_groups:
            groups = sarc_obj.data['structure.zbands.lat_groups'][frame]
            labels_plot = np.zeros_like(labels)
            for i, group in enumerate(groups):
                mask = np.zeros_like(labels, dtype=bool)
                for label in group:
                    mask += (labels == label + 1)
                labels_plot[mask] = i + 1
        else:
            labels_plot = labels

        if shuffle:
            labels_plot = Utils.shuffle_labels(labels_plot)

        z_ends = sarc_obj.data['structure.zbands.ends'][frame].astype('float32') / sarc_obj.metadata.pixelsize
        z_links = sarc_obj.data['structure.zbands.lat_links'][frame]
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert sarc_obj.mask_exists('orientation'), \
            'Sarcomere orientation map does not exist! Run detect_sarcomeres first.'

        orientation_field = sarc_obj.read_mask('orientation', frames=frame)

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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert sarc_obj.mask_exists('sarcomere_mask'), ('No sarcomere masks stored. '
                                                         'Run detect_sarcomeres first.')

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands,
                               clip_thrs=clip_thrs, scalebar=False)

        sarcomere_mask = sarc_obj.read_mask('sarcomere_mask', frames=frame)

        # binarize sarcomere mask
        if threshold is None:
            threshold = sarc_obj.data.get('params.structure.analyze_vectors.threshold_sarcomere_mask')
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
            StructurePlots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert 'structure.sarcomere.pos' in sarc_obj.data.keys(), ('Sarcomere vectors not yet calculated, '
                                                                 'run sarc.structure.analyze_vectors first.')
        assert frame in sarc_obj.data['params.structure.analyze_vectors.frames'], f'Frame {frame} not yet analyzed.'

        pos_vectors = sarc_obj.data['structure.sarcomere.pos'][frame] / sarc_obj.metadata.pixelsize
        sarcomere_orientation_vectors = sarc_obj.data['structure.sarcomere.orientation'][frame]
        sarcomere_length_vectors = sarc_obj.data['structure.sarcomere.slen'][frame] / sarc_obj.metadata.pixelsize
        orientation_vectors = np.asarray(
            [np.cos(sarcomere_orientation_vectors), -np.sin(sarcomere_orientation_vectors)])

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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

            StructurePlots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert 'structure.domain.n' in sarc_obj.data.keys(), ('Sarcomere domains not analyzed. '
                                                               'Run sarc.structure.analyze_domains first.')
        assert frame in sarc_obj.data['params.structure.analyze_domains.frames'], (f'Domains in frame {frame} are not yet '
                                                                           f'analyzed.')
        domains = sarc_obj.data['structure.domain.members'][frame]
        pos_vectors = sarc_obj.data['structure.sarcomere.pos'][frame]
        sarcomere_orientation_vectors = sarc_obj.data['structure.sarcomere.orientation'][frame]
        sarcomere_length_vectors = sarc_obj.data['structure.sarcomere.slen'][frame]
        area_min = sarc_obj.data['params.structure.analyze_domains.area_min']
        dilation_radius = sarc_obj.data['params.structure.analyze_domains.dilation_radius']
        domain_mask, *_ = domain_clustering.analyze_domains(
            domains, pos_vectors, sarcomere_orientation_vectors, sarcomere_length_vectors,
            size=sarc_obj.metadata.size, pixelsize=sarc_obj.metadata.pixelsize,
            dilation_radius=dilation_radius, area_min=area_min)

        domain_mask_masked = np.ma.masked_where(domain_mask == 0, domain_mask)
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=(0, 0, 0, 0))

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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
        frame = StructurePlots._frame(sarc_obj, frame)
        assert 'structure.myofibril.lines' in sarc_obj.data.keys(), ('Myofibrils not analyzed. '
                                                                'Run sarc.structure.analyze_myofibrils first.')
        assert frame in sarc_obj.data['params.structure.analyze_myofibrils.frames'], f'Frame {frame} not yet analyzed.'

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
                               invert_image=invert_image, invert_z_bands=invert_z_bands,
                               cmap_image=cmap_image, cmap_z_bands=cmap_z_bands,
                               alpha_image=alpha_image, alpha_z_bands=alpha_z_bands, scalebar=False)

        lines = sarc_obj.data['structure.myofibril.lines'][frame]
        pos_vectors = sarc_obj.data['structure.sarcomere.pos_px'][frame]
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

            StructurePlots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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
        frame = StructurePlots._frame(sarc_obj, frame)
        # create myofibril length map
        assert 'structure.myofibril.lines' in sarc_obj.data.keys(), ('Myofibrils not yet analyzed. '
                                                                'Run sarc.structure.analyze_myofibrils first.')
        assert frame in sarc_obj.data['params.structure.analyze_myofibrils.frames'], f'Frame {frame} not yet analyzed.'

        myof_lines = sarc_obj.data['structure.myofibril.lines'][frame]
        myof_lengths = sarc_obj.data['structure.myofibril.length'][frame]
        pos_vectors = sarc_obj.data['structure.sarcomere.pos'][frame]
        orientation_vectors = sarc_obj.data['structure.sarcomere.orientation'][frame]
        length_vectors = sarc_obj.data['structure.sarcomere.slen'][frame]
        median_filter_radius = sarc_obj.data['params.structure.analyze_myofibrils.median_filter_radius']
        myof_length_map = myofibril_analysis.create_myofibril_length_map(
            myof_lines=myof_lines, myof_length=myof_lengths,
            pos_vectors=pos_vectors,
            sarcomere_orientation_vectors=orientation_vectors,
            sarcomere_length_vectors=length_vectors,
            size=sarc_obj.metadata.size,
            pixelsize=sarc_obj.metadata.pixelsize,
            median_filter_radius=median_filter_radius)

        StructurePlots._draw_background(ax, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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

            StructurePlots._draw_background(ax_inset, sarc_obj, frame=frame, show_image=show_image, show_z_bands=show_z_bands,
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
    def plot_lois(ax: Axes, sarc_obj: SarcAsM, color='darkorange', linewidth=2, alpha=0.5):
        """
        Plot the lines of interest of a recording (``motion.lois.lines``).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the LOI lines on.
        sarc_obj : SarcAsM
            The object to plot.
        color : str, optional
            Color of the lines. Default is 'darkorange'.
        linewidth : float, optional
            Width of the lines. Default is 2.
        alpha : float, optional
            Opacity of the lines. Default is 0.5.
        """
        loi_lines = None
        if hasattr(sarc_obj, 'data') and 'motion.lois.lines' in sarc_obj.data:
            loi_lines = sarc_obj.data['motion.lois.lines']

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
        frame = StructurePlots._frame(sarc_obj, frame)
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
