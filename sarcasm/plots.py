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


import numbers
import os.path
import warnings
from typing import Union, Tuple, Optional, Literal

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, transforms
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib_scalebar.scalebar import ScaleBar
from tifffile import tifffile

from sarcasm.feature_dict import structure_feature_dict
from sarcasm.motion import Motion
from sarcasm.plot_utils import PlotUtils
from sarcasm.structure import Structure
from sarcasm.utils import Utils


class Plots:
    """
    Class with plotting functions for Structure and Motion objects
    """

    @staticmethod
    def plot_stack_overlay(ax: Axes, sarc_obj: Union[Structure, Motion], frames, plot_func, offset=0.025,
                           spine_color='w', xlim=None, ylim=None):
        """
        Plot a stack of overlayed subplots on a given Axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object on which the stack should be plotted.
        sarc_obj : Structure
            Data to be plotted in each subplot, which can be an instance of Structure or Motion.
        frames : list
            The frames at which the subplots should be created.
        plot_func : function
            The function used to plot the data in each subplot, e.g.
        offset : float, optional
            The offset between each subplot. Defaults to 0.025.
        spine_color : str, optional
            The color of the spines (borders) of each subplot. Defaults to 'w' (white).
        xlim : tuple, optional
            The x-axis limits for each subplot. Defaults to None.
        ylim : tuple, optional
            The y-axis limits for each subplot. Defaults to None.
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
                                filename=None):
        """
        Plots a summary of the motion of the line of interest (LOI).

        Parameters
        ----------
        motion_obj : Motion
            The Motion object to plot.
        number_contr : int, optional
            The number of contractions to plot. Defaults to 0.
        t_lim : tuple of float, optional
            The time limits for the plot in seconds. Defaults to (0, 12).
        t_lim_overlay : tuple of float, optional
            The time limits for the overlay plots in seconds. Defaults to (-0.1, 2.9)
        filename : str, optional
            The filename to save the plot. Defaults to None.
        """

        mosaic = """
        aaaccc
        bbbccc
        dddeee
        dddfff
        """

        fig, axs = plt.subplot_mosaic(mosaic, figsize=(PlotUtils.width_2cols, PlotUtils.width_2cols),
                                      constrained_layout=True)
        title = f'File: {motion_obj.filepath}, \nLOI: {motion_obj.loi_name}'
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

        if filename is None:
            filename = os.path.join(motion_obj.loi_folder, 'summary_loi.png')
        fig.savefig(filename, dpi=PlotUtils.dpi)
        plt.show()

    @staticmethod
    def plot_loi_detection(sarc_obj: Structure, frame: int = 0, filepath: str = None,
                           cmap_z_bands='Greys'):
        """
        Plots all steps of automated LOI finding algorithm

        Parameters
        ----------
        sarc_obj : Structure
            Instance of Structure class
        frame: int
            The time point to plot.
        filepath: str
            Path to save the plot. If None, plot is not saved.
        cmap_z_bands : str, optional
            Colormap of Z-bands. Defaults to 'Greys'.
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

        if filepath is not None:
            fig.savefig(filepath, dpi=300)
        plt.show()

    @staticmethod
    def plot_image(ax: Axes, sarc_obj: Union[Structure, Motion], frame: int = 0, cmap: str = 'grey',
                   alpha: float = 1, clip_thrs: Tuple[float, float] = (1, 99), scalebar: bool = True,
                   title: Union[None, str] = None, show_loi: bool = False,
                   zoom_region: Tuple[int, int, int, int] = None,
                   inset_bounds: Tuple[float, float, float, float] = (0.6, 0.6, 0.4, 0.4)):
        """
        Plots microscopy raw image of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        cmap : matplotlib.cm.Colormap, optional
            The colormap to use. Defaults to 'gray'.
        alpha : float, optional
            The transparency to use. Defaults to 1.
        clip_thrs : tuple, optional
            Clipping thresholds to normalize intensity, in percentiles. Defaults to (1, 99).
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        title : str, optional
            The title for the plot. Defaults to None.
        show_loi : bool, optional
            Whether to show the line of interest (LOI). Defaults to True.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
        """

        img = sarc_obj.read_imgs(frames=frame)
        img = np.clip(img, np.percentile(img, clip_thrs[0]), np.percentile(img, clip_thrs[1]))

        plot = ax.imshow(img, cmap=cmap, alpha=alpha)
        if show_loi:
            Plots.plot_lois(ax, sarc_obj)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            ax_inset.imshow(img[y1:y2, x1:x2], cmap='gray')
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')
            PlotUtils.change_color_spines(ax_inset, 'w')

            if scalebar:
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w',
                                             sep=1, height_fraction=0.02, location='lower right', scale_loc='top',
                                             font_properties={'size': PlotUtils.fontsize - 1}))

    @staticmethod
    def plot_z_bands(ax: plt.Axes, sarc_obj: Union[Structure, Motion], frame=0, cmap='Greys_r', zero_transparent=False,
                     alpha=1, scalebar=True, title=None, color_scalebar='w',
                     show_loi=False, zoom_region: Tuple[int, int, int, int] = None,
                     inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots the Z-bands of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        cmap : matplotlib.cm.Colormap, optional
            Colormap to use. Defaults to 'Greys_r'.
        alpha : float, optional
            Alpha value to change opacity of image. Defaults to 1
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        title : str, optional
            The title for the plot. Defaults to None.
        show_loi : bool, optional
            Whether to show the line of interest (LOI). Defaults to True.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
        """
        assert os.path.exists(sarc_obj.file_zbands), ('Z-band mask not found. Run predict_z_bands first.')

        img = tifffile.imread(sarc_obj.file_zbands, key=frame)
        if zero_transparent:
            img = np.ma.masked_where(img < 0.05, img)
        ax.imshow(img, cmap=cmap, alpha=alpha)
        if show_loi:
            Plots.plot_lois(ax, sarc_obj)
        if scalebar:
            ax.add_artist(
                ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color=color_scalebar,
                         sep=1, height_fraction=0.02, location='lower right', scale_loc='top',
                         font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            PlotUtils.change_color_spines(ax_inset, 'w')
            ax_inset.imshow(img[y1:y2, x1:x2], cmap=cmap, alpha=alpha)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')

    def plot_z_bands_midlines(ax: plt.Axes, sarc_obj: Union[Structure, Motion], frame=0, cmap='berlin',
                              alpha=1, scalebar=True, title=None, color_scalebar='w',
                              show_loi=True, zoom_region: Tuple[int, int, int, int] = None,
                              inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots the Z-bands and midlines of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        cmap : matplotlib.cm.Colormap, optional
            Colormap to use. Defaults to 'Blues_r'.
        alpha : float, optional
            Alpha value to change opacity of image. Defaults to 1
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        title : str, optional
            The title for the plot. Defaults to None.
        show_loi : bool, optional
            Whether to show the line of interest (LOI). Defaults to True.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
        """
        assert os.path.exists(sarc_obj.file_zbands), ('Z-band mask not found. Run predict_z_bands first.')

        zbands = tifffile.imread(sarc_obj.file_zbands, key=frame)
        midlines = tifffile.imread(sarc_obj.file_mbands, key=frame)
        joined = midlines - zbands

        ax.imshow(joined, cmap=cmap, alpha=alpha)

        if show_loi:
            Plots.plot_lois(ax, sarc_obj)
        if scalebar:
            ax.add_artist(
                ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color=color_scalebar,
                         sep=1, height_fraction=0.02, location='lower right', scale_loc='top',
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
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')

            if scalebar:
                ax_inset.add_artist(
                    ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color=color_scalebar,
                             sep=1, height_fraction=0.02, location='lower right', scale_loc='top',
                             font_properties={'size': PlotUtils.fontsize - 1}))

    @staticmethod
    def plot_cell_mask(ax: Axes, sarc_obj: Union[Structure, Motion], frame=0, threshold=0.5, cmap='gray', alpha=1,
                       scalebar=True, title=None):
        """
        Plots the cell mask of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        threshold : float, optional
            Binarization threshold to use for cell mask. Defaults to 0.5.
        cmap : matplotlib.colors.Colormap, optional
            The colormap to use. Defaults to 'gray'
        alpha : float, optional
            Transparency value to change opacity of mask. Defaults to 0.5.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        title : str, optional
            The title for the plot. Defaults to None.
        """
        assert os.path.exists(sarc_obj.file_cell_mask), ('Cell mask not found. Run predict_cell_mask first.')

        img = tifffile.imread(sarc_obj.file_cell_mask, key=frame) > threshold
        ax.imshow(img, cmap=cmap, alpha=alpha)

        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_z_segmentation(ax: Axes, sarc_obj: Structure, frame=0, scalebar=True, shuffle=True,
                            title=None, zoom_region: Tuple[int, int, int, int] = None,
                            inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots the Z-band segmentation result of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure
            The instance of Structure class to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        shuffle : bool, optional
            Whether to shuffle the labels. Defaults to True.
        title : str, optional
            The title for the plot. Defaults to None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
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
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            ax_inset.imshow(masked_labels[y1:y2, x1:x2], cmap=cmap)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            if scalebar:
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                       height_fraction=0.02, location='lower right', scale_loc='top',
                                       font_properties={'size': PlotUtils.fontsize - 1}))

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

    @staticmethod
    def plot_z_lateral_connections(ax: Axes, sarc_obj: Structure, frame=0, scalebar=True, markersize=1.5,
                                   markersize_inset=3, linewidth=0.25, linewidth_inset=0.5, plot_groups=True,
                                   shuffle=True, title=None, zoom_region: Tuple[int, int, int, int] = None,
                                   inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots lateral Z-band connections of a Structure object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure
            The instance of Structure object to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        markersize : int, optional
            The size of the markers of the Z-band ends. Defaults to 5.
        markersize_inset : int, optional
            The size of the markers of the Z-band ends in the inset plot. Defaults to 5.
        linewidth : int, optional
            The width of the connection lines. Defaults to 0.25.
        linewidth : int, optional
            The width of the connection lines in the inset plot. Defaults to 0.5.
        plot_groups : bool
            Whether to show the Z-bands of each lateral group with the same color. Defaults to True.
        shuffle : bool, optional
            Whether to shuffle the labels. Defaults to True.
        title : str, optional
            The title for the plot. Defaults to None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
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

        z_ends = sarc_obj.data['z_ends'][frame].astype('float32') / sarc_obj.metadata['pixelsize']
        z_links = sarc_obj.data['z_lat_links'][frame]
        masked_labels = np.ma.masked_where(labels_plot == 0, labels_plot)
        cmap = plt.cm.prism
        cmap.set_bad(color=(0, 0, 0, 0))
        ax.imshow(masked_labels, cmap=cmap)
        for (i, k, j, l) in z_links.T:
            ax.plot([z_ends[i, k, 1], z_ends[j, l, 1]],
                    [z_ends[i, k, 0], z_ends[j, l, 0]],
                    c='k', lw=linewidth, linestyle='-', alpha=1, zorder=2)
        ax.scatter(z_ends[:, 0, 1], z_ends[:, 0, 0], c='k', marker='.', s=markersize, zorder=3, edgecolors='none')
        ax.scatter(z_ends[:, 1, 1], z_ends[:, 1, 0], c='k', marker='.', s=markersize, zorder=3, edgecolors='none')
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            ax_inset.imshow(masked_labels, cmap=cmap)
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
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                       height_fraction=0.02, location='lower right', scale_loc='top',
                                       font_properties={'size': PlotUtils.fontsize - 1}))

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

    @staticmethod
    def plot_sarcomere_orientation_field(ax1: Axes, ax2: Axes, sarc_obj: Structure, frame=0, cmap='vanimo',
                                         scalebar=True, colorbar=True, shrink_colorbar=0.7, orient_colorbar='vertical',
                                         zoom_region: Tuple[int, int, int, int] = None,
                                         inset_bounds=(0.6, 0.6, 0.4, 0.4),):
        """
            Plots sarcomere orientation field of the sarcomere object.

            Parameters
            ----------
            ax1 : matplotlib.axes.Axes
                The axes to draw the plot on.
            sarc_obj : object
                The instance of Structure class to plot.
            frame : int, optional
                The frame to plot. Defaults to 0.
            scalebar : bool, optional
                Whether to add a scalebar to the plot. Defaults to True.
            colorbar : bool, optional
                Whether to add a colorbar to the plot. Defaults to True.
            shrink_colorbar : float, optional
                The factor by which to shrink the colorbar. Defaults to 0.7.
            orient_colorbar : str, optional
                The orientation of the colorbar ('horizontal' or 'vertical'). Defaults to 'vertical'.
            zoom_region : tuple of int, optional
                The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
            inset_bounds : tuple of float, optional
                Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
            """
        assert os.path.exists(
            sarc_obj.file_orientation), 'Sarcomere orientation map does not exist! Run predict_sarcomeres first.'

        orientation_field = tifffile.imread(sarc_obj.file_orientation, key=[frame * 2, frame * 2 + 1])

        plot1 = ax1.imshow(orientation_field[0], cmap=cmap)
        plot2 = ax2.imshow(orientation_field[1], cmap=cmap)

        if scalebar:
            ax1.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                    height_fraction=0.02, location='lower right', scale_loc='top',
                                    font_properties={'size': PlotUtils.fontsize - 1}))
            ax2.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                    height_fraction=0.02, location='lower right', scale_loc='top',
                                    font_properties={'size': PlotUtils.fontsize - 1}))

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        if colorbar:
            plt.colorbar(plot1, ax=ax1, label=r'X-Field', shrink=shrink_colorbar, orientation=orient_colorbar)
            plt.colorbar(plot2, ax=ax2, label=r'Y-Field', shrink=shrink_colorbar, orientation=orient_colorbar)

        if scalebar:
            ax1.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
            ax2.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset1 = ax1.inset_axes(bounds=inset_bounds)
            ax_inset2 = ax2.inset_axes(bounds=inset_bounds)

            ax_inset1.imshow(orientation_field[0][y1:y2, x1:x2], cmap=cmap)
            ax_inset2.imshow(orientation_field[1][y1:y2, x1:x2], cmap=cmap)
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
                ax_inset1.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                                        height_fraction=0.02, location='lower right', scale_loc='top',
                                        font_properties={'size': PlotUtils.fontsize - 1}))
                ax_inset2.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                                        height_fraction=0.02, location='lower right', scale_loc='top',
                                        font_properties={'size': PlotUtils.fontsize - 1}))

    @staticmethod
    def plot_sarcomere_mask(ax: Axes, sarc_obj: Structure, frame=0, cmap='viridis', threshold=0.1,
                            show_z_bands=False, alpha=0.5, cmap_z_bands='gray', alpha_z_bands=1, clip_thrs=(1, 99.9),
                            title=None, zoom_region: Tuple[int, int, int, int] = None,
                            inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots binary mask of sarcomeres, derived from sarcomere vectors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure
            The instance of Structure class to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        cmap : str, optional
            The colormap to use. Defaults to 'viridis'
        show_z_bands : bool, optional
            Whether to show Z-bands. If False, the raw image is shown. Defaults to False.
        alpha : float, optional
            The transparency of sarcomere mask. Defaults to 0.5.
        cmap_z_bands : bool, optional
            Colormap for Z-bands. Defaults to 'gray'.
        alpha_z_bands : float, optional
            Alpha value of Z-bands. Defaults to 1.
        clip_thrs : tuple of float, optional
            Clipping threshold for image in background. Defaults to (1, 99.9). Only if show_z_bands is False.
        title : str, optional
            The title for the plot. Defaults to None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
        """
        assert os.path.exists(sarc_obj.file_sarcomere_mask), ('No sarcomere masks stored. '
                                                              'Run sarc_obj.analyze_sarcomere_vectors ')

        if show_z_bands:
            Plots.plot_z_bands(ax, sarc_obj, alpha=alpha_z_bands, frame=frame)
        else:
            Plots.plot_image(ax, sarc_obj, frame=frame, clip_thrs=clip_thrs)

        sarcomere_mask = tifffile.imread(sarc_obj.file_sarcomere_mask, key=frame)

        # binarize sarcomere mask
        if threshold is None:
            threshold = sarc_obj.data.get('params.analyze_sarcomere_vectors.threshold_sarcomere_mask')
        sarcomere_mask = sarcomere_mask > threshold

        # plot sarcomere mask
        sarcomere_mask = np.ma.masked_where(sarcomere_mask == 0, sarcomere_mask)

        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=(0, 0, 0, 0))
        ax.imshow(sarcomere_mask, vmin=0, vmax=1, alpha=alpha, cmap=cmap)
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)
            if show_z_bands:
                Plots.plot_z_bands(ax_inset, sarc_obj, alpha=alpha_z_bands, cmap=cmap_z_bands, frame=frame)
            else:
                Plots.plot_image(ax_inset, sarc_obj, frame=frame, clip_thrs=clip_thrs)
            ax_inset.set_ylim(y2, y1)
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.imshow(sarcomere_mask, vmin=0, vmax=1, alpha=alpha, cmap=cmap)
            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='w')
            PlotUtils.change_color_spines(ax_inset, 'w')

    @staticmethod
    def plot_sarcomere_vectors(ax: Axes, sarc_obj: Structure, frame=0, color_arrows='k',
                               color_points='darkgreen', s_points=0.5, linewidths=0.5,
                               s_points_inset=0.5, linewidths_inset=0.5, scalebar=True,
                               legend=False, show_image=False, cmap_z_bands='Purples', alpha_z_bands=1, title=None,
                               zoom_region: Tuple[int, int, int, int] = None,
                               inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots quiver plot reflecting local sarcomere length and orientation based on sarcomere vector analysis result
        of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure
            The instance of Structure class to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        color_arrows : str, optional
            The color of the arrows. Defaults to 'mediumpurple'.
        color_points : str, optional
            The color of the points. Defaults to 'darkgreen'.
        s_points : float, optional
            The size of midline points. Defaults to 0.5.
        linewidths : float, optional
            The width of the arrow lines. Defaults to 0.0005.
        s_points_inset : float, optional
            The size of midline points. Defaults to 0.5.
        linewidths_inset : float, optional
            The width of the arrow lines in the inset plot. Defaults to 0.0001.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        legend : bool, optional
            Whether to add a legend to the plot. Defaults to False.
        show_image : bool, optional
            Whether to show the image (True) or the Z-bands (False). Defaults to False.
        cmap_z_bands : str, optional
            Colormap of Z-bands. Defaults to 'Greys'.
        alpha_z_bands : float, optional
            Alpha value of Z-bands. Defaults to 1.
        title : str, optional
            The title for the plot. Defaults to None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
        """
        assert 'pos_vectors' in sarc_obj.data.keys(), ('Sarcomere vectors not yet calculated, '
                                                                 'run analyze_sarcomere_vectors first.')
        assert frame in sarc_obj.data['params.analyze_sarcomere_vectors.frames'], f'Frame {frame} not yet analyzed.'

        pos_vectors = sarc_obj.data['pos_vectors'][frame] / sarc_obj.metadata['pixelsize']
        sarcomere_orientation_vectors = sarc_obj.data['sarcomere_orientation_vectors'][frame]
        sarcomere_length_vectors = sarc_obj.data['sarcomere_length_vectors'][frame] / sarc_obj.metadata[
            'pixelsize']
        orientation_vectors = np.asarray(
            [np.cos(sarcomere_orientation_vectors), -np.sin(sarcomere_orientation_vectors)])

        if show_image:
            Plots.plot_image(ax, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)
        else:
            Plots.plot_z_bands(ax, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)

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
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            linewidths *= 10
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)

            if show_image:
                Plots.plot_image(ax_inset, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)
            else:
                Plots.plot_z_bands(ax_inset, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)

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
                ax_inset.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k',
                                             sep=1, height_fraction=0.02, location='lower right', scale_loc='top',
                                             font_properties={'size': PlotUtils.fontsize - 1, }))

    @staticmethod
    def plot_sarcomere_domains(ax: Axes, sarc_obj: Structure, frame=0, alpha=0.5, cmap='gist_rainbow',
                               scalebar=True, plot_raw_data=False, cmap_z_bands='Greys', alpha_z_bands=1, title=None):
        """
        Plots the sarcomere domains of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure
            The instance of Structure class to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        alpha : float, optional
            The transparency of the domain masks. Defaults to 0.3.
        cmap : str, optional
            The colormap to use. Defaults to 'gist_rainbow'.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        plot_raw_data : bool, optional
            Whether to plot the raw data. Defaults to False.
        cmap_z_bands : str, optional
            Colormap for Z-bands. Defaults to 'Greys'.
        alpha_z_bands : float, optional
            Transparency of Z-bands. Defaults to 1.
        title : str, optional
            The title for the plot. Defaults to None.

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
        domain_mask = sarc_obj._analyze_domains(domains, pos_vectors=pos_vectors,
                                                 sarcomere_length_vectors=sarcomere_length_vectors,
                                                 sarcomere_orientation_vectors=sarcomere_orientation_vectors,
                                                 size=sarc_obj.metadata['size'],
                                                 pixelsize=sarc_obj.metadata['pixelsize'],
                                                 dilation_radius=dilation_radius, area_min=area_min)[0]

        domain_mask_masked = np.ma.masked_where(domain_mask == 0, domain_mask)
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=(0, 0, 0, 0))

        if plot_raw_data:
            Plots.plot_image(ax, sarc_obj, frame=frame, scalebar=False, alpha=alpha_z_bands, cmap=cmap_z_bands)
        else:
            Plots.plot_z_bands(ax, sarc_obj, cmap=cmap_z_bands, alpha=alpha_z_bands, frame=frame, scalebar=False)

        ax.imshow(domain_mask_masked, cmap=cmap, alpha=alpha, vmin=0, vmax=np.nanmax(domain_mask))

        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        ax.set_title(title, fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_myofibril_lines(ax: Axes, sarc_obj: Structure , frame=0, show_z_bands=True, linewidth=1, color_lines='r',
                             linewidth_inset=3, alpha=0.2, cmap_z_bands='Greys', alpha_z_bands=1,
                             scalebar=True, title=None, zoom_region=None, inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots result of myofibril line growth algorithm of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure or Motion
            The sarcomere object to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        show_z_bands : bool
            Whether or not to show Z-bands. Defaults to True
        linewidth : float, optional
            The width of the lines. Defaults to 1.
        color_lines : str
            Color of lines. Defaults to 'r'
        linewidth_inset : float, optional
            Thickness of the lines in inset. Defaults to 1.
        alpha : float, optional
            The transparency of the lines. Defaults to 0.2.
        cmap_z_bands : str, optional
            Colormap of Z-bands. Defaults to 'Greys'.
        alpha_z_bands : float, optional
            Transparency of Z-bands. Defaults to 1.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        title : str, optional
            The titlefor the plot. Defaults to None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
        """
        assert 'myof_lines' in sarc_obj.data.keys(), ('Myofibrils not analyzed. '
                                                                'Run analyze_myofibrils first.')
        assert frame in sarc_obj.data['params.analyze_myofibrils.frames'], f'Frame {frame} not yet analyzed.'

        if show_z_bands:
            Plots.plot_z_bands(ax, sarc_obj, cmap=cmap_z_bands, frame=frame, alpha=alpha_z_bands)
        else:
            Plots.plot_image(ax, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)

        lines = sarc_obj.data['myof_lines'][frame]
        pos_vectors = sarc_obj.data['pos_vectors_px'][frame]
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
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

            if show_z_bands:
                Plots.plot_z_bands(ax_inset, sarc_obj, cmap=cmap_z_bands, frame=frame)
            else:
                Plots.plot_image(ax_inset, sarc_obj, frame=frame, cmap=cmap_z_bands)

            if scalebar:
                ax_inset.add_artist(
                    ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                             height_fraction=0.02, location='lower right', scale_loc='top',
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
    def plot_myofibril_length_map(ax: Axes, sarc_obj: Structure, frame=0, vmax=None, alpha=1,
                                  show_z_bands=False, cmap_z_bands='Greys', alpha_z_bands=1,
                                  colorbar=True, shrink_colorbar=0.7, orient_colorbar='vertical',
                                  scalebar=True, title=None, zoom_region: Tuple[int, int, int, int] = None,
                                  inset_bounds=(0.6, 0.6, 0.4, 0.4)):
        """
        Plots the spatial map of myofibril lengths for a given frame.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : Structure
            The instance of Structure class to plot.
        frame : int, optional
            The frame to plot. Defaults to 0.
        vmax : float, optional
            Maximum value for the colormap. If None, the maximum value in the data is used. Defaults to None.
        alpha : float, optional
            Transparency of the colormap. Defaults to 1.
        show_z_bands : bool, optional
            Whether to show Z-band mask, else raw image is shown. Defaults to False.
        cmap_z_bands : str, optional
            Colormap of Z-bands. Defaults to 'Greys'.
        alpha_z_bands : float, optional
            Transparency of Z-bands or raw image. Defaults to 1.
        colorbar : bool, optional
            Whether to show the colorbar. Defaults to True.
        shrink_colorbar: float, optional
            Shrinkage of the colorbar. Defaults to 0.7.
        orient_colorbar : str, optional
            Orientation of the colorbar. Defaults to 'vertical'.
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        title : str, optional
            The title for the plot. Defaults to None.
        zoom_region : tuple of int, optional
            The region to zoom in on, specified as (x1, x2, y1, y2). Defaults to None.
        inset_bounds : tuple of float, optional
            Bounds of inset axis, specified as (x0, y0, width, height). Defaults to (0.6, 0.6, 0.4, 0.4).
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
        myof_length_map = sarc_obj.create_myofibril_length_map(myof_lines=myof_lines, myof_length=myof_lengths,
                                                      pos_vectors=pos_vectors,
                                                      sarcomere_orientation_vectors=orientation_vectors,
                                                      sarcomere_length_vectors=length_vectors,
                                                      size=sarc_obj.metadata['size'],
                                                      pixelsize=sarc_obj.metadata['pixelsize'],
                                                      median_filter_radius=median_filter_radius)

        if show_z_bands:
            Plots.plot_z_bands(ax, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)
        else:
            Plots.plot_image(ax, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)

        masked_myof_length_map = np.ma.masked_array(myof_length_map, mask=(myof_length_map == 0))
        cmap = plt.cm.inferno
        cmap.set_bad(color=(0, 0, 0, 0))  # Set color for masked values to transparent
        vmin, vmax = 0, np.nanmax(myof_length_map) if vmax is None else vmax
        plot = ax.imshow(masked_myof_length_map, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        if scalebar:
            ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                                   height_fraction=0.02, location='lower right', scale_loc='top',
                                   font_properties={'size': PlotUtils.fontsize - 1}))
        if colorbar:
            cbar = plt.colorbar(mappable=plot, ax=ax, shrink=shrink_colorbar, orientation=orient_colorbar,
                                label='Myofibril length [µm]')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=PlotUtils.fontsize)

        # Add inset axis if zoom_region is specified
        if zoom_region:
            x1, x2, y1, y2 = zoom_region
            ax_inset = ax.inset_axes(bounds=inset_bounds)

            if show_z_bands:
                Plots.plot_z_bands(ax_inset, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)
            else:
                Plots.plot_image(ax_inset, sarc_obj, frame=frame, cmap=cmap_z_bands, alpha=alpha_z_bands)

            ax_inset.imshow(masked_myof_length_map, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)

            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y2, y1)

            if scalebar:
                ax_inset.add_artist(
                    ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                             height_fraction=0.02, location='lower right', scale_loc='top',
                             font_properties={'size': PlotUtils.fontsize - 1}))

            # Mark the zoomed region on the main plot
            PlotUtils.plot_box(ax, xlim=(x1, x2), ylim=(y1, y2), c='k')

    @staticmethod
    def plot_lois(ax: Axes, sarc_obj: Union[Structure, Motion], color='darkorange', linewidth=2, alpha=0.5):
        """
        Plot all LOI lines for Structure object and LOI line Motion object.

        Parameters
        ----------
        ax : matplotlib axis
            Axis on which to plot the LOI lines
        sarc_obj : Structure or Motion
            Object of Structure or Motion class
        color : str
            Color of lines
        linewidth : float
            Width of lines
        alpha : float
            Transparency of lines
        """
        loi_lines = None

        if hasattr(sarc_obj, 'loi_data'):
            # Extract line data directly from sarc_obj.loi_data
            loi_lines = [sarc_obj.loi_data['line']]
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
                                 sarc_obj: Structure,
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
        Plots the histogram of a specified structural feature from a sarcomere object on a given Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to draw the histogram.
        sarc_obj : Structure
            The instance of Structure class to plot.
        feature : str
            The name of the structural feature to plot.
        frame : int, optional
            The frame index from which to extract the data. Defaults to 0.
        bins : int, optional
            The number of bins for the histogram. Defaults to 20.
        density : bool, optional
            If True, the histogram is normalized to show the probability density rather than raw counts.
            Defaults to False.
        range : tuple, optional
            The lower and upper range of the bins. If not provided, the range is determined from the data.
        label : str, optional
            The label for the x-axis. If not specified, a default label based on the feature will be used.
        ylabel : str, optional
            The label for the y-axis. Overrides the default label if provided.
        rwidth : float, optional
            The relative width of the histogram bars. Defaults to 0.7.
        color : str, optional
            The fill color of the histogram bars. Defaults to 'darkslategray'.
        edge_color : str, optional
            The color of the edges of the histogram bars. Defaults to 'k'.
        align : str, optional
            The alignment of the histogram bars. Defaults to 'mid'.
        rotate_yticks : bool, optional
            If True, rotates the y-axis tick labels by 90 degrees for improved readability.
            Defaults to False.
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
    def plot_z_pos(ax: Axes, motion_obj: Motion, number_contr=None, show_contr=True, show_kymograph=False, color='k',
                   t_lim=(None, None), y_lim=(None, None)):
        """
        Plots the z-band trajectories of the motion object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        show_contr : bool, optional
            Whether to show the contractions. Defaults to True.
        show_kymograph : bool, optional
            Whether to show the kymograph. Defaults to False.
        color : str, optional
            The color of the plot. Defaults to 'k'.
        t_lim : tuple, optional
            The time limits for the plot. Defaults to (None, None).
        y_lim : tuple, optional
            The y limits for the plot. Defaults to (None, None).
        """
        # plot limits and params
        if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
            start_contr_t = motion_obj.loi_data['start_contr'][number_contr]
            tlim = (start_contr_t + t_lim[0], start_contr_t + t_lim[1])
            idxlim = (int(tlim[0] / motion_obj.metadata['frametime']), int(tlim[1] / motion_obj.metadata['frametime']))
        else:
            tlim, idxlim = (None, None), (None, None)

        if show_kymograph:
            ax.pcolorfast(motion_obj.loi_data['time'], motion_obj.loi_data['x_pos'], motion_obj.loi_data['y_int'].T,
                          cmap='Greys')
        # get data
        time = motion_obj.loi_data['time']
        z_pos = motion_obj.loi_data['z_pos']
        # plot contraction cycles
        if show_contr:
            for start_i, time_i in zip(motion_obj.loi_data['start_contr'],
                                       motion_obj.loi_data['time_contr']):
                end_i = start_i + time_i
                if number_contr is not None:
                    start_i -= tlim[0]
                    end_i -= tlim[0]
                ax.fill_betweenx([0, 1], [start_i, start_i], [end_i, end_i], color='lavender',
                                 transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

        # plot trajectories
        if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
            ax.plot(time[:idxlim[1] - idxlim[0]], z_pos[:, idxlim[0]:idxlim[1]], linewidth=0.75, c=color)
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
        PlotUtils.polish_xticks(ax, 2, 1)

    @staticmethod
    def plot_delta_slen(ax: Axes, motion_obj: Motion, frame=None, t_lim=(0, 12), y_lim=(-0.3, 0.4), n_rows=6,
                        n_start=1, show_contr=True):
        """
        Plots the change in sarcomere length over time for a motion object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        frame : int or None, optional
            Show frame with vertical dashed line, in frames. Defaults to None.
        t_lim : tuple, optional
            The time limits for the plot. Defaults to (0, 12).
        y_lim : tuple, optional
            The y limits for the plot. Defaults to (-0.3, 0.4).
        n_rows : int, optional
            The number of rows for the plot. Defaults to 6.
        n_start : int, optional
            The starting index for the plot. Defaults to 1.
        show_contr : bool, optional
            Whether to show the systoles. Defaults to True.
        """
        yticks = [-0.2, 0, 0.2]
        delta_slen = motion_obj.loi_data['delta_slen']
        list_y = np.linspace(0, 1, num=n_rows, endpoint=False)
        for i, y in enumerate(list_y):
            ax_i = ax.inset_axes((0., y, 1, 1 / n_rows - 0.02))
            ax_i.plot(motion_obj.loi_data['time'], delta_slen[i + n_start], c='k', lw=0.6)
            ax_i.axhline(0, linewidth=1, linestyle=':', c='k')
            if show_contr:
                for start_i, time_i in zip(motion_obj.loi_data['start_contr'],
                                           motion_obj.loi_data['time_contr']):
                    end_i = start_i + time_i
                    ax_i.fill_betweenx([-1, 1], [start_i, start_i], [end_i, end_i], color='lavender')

            if i > 0:
                ax_i.set_xticks([])
            else:
                PlotUtils.polish_xticks(ax_i, 1, 0.5)
            if frame is not None:
                ax_i.axvline(motion_obj.loi_data['time'][frame], linestyle='--', c='k')
            ax_i.set_ylim(y_lim)
            ax_i.set_xlim(t_lim)
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
    def plot_overlay_delta_slen(ax: Axes, motion_obj: Motion, number_contr=None, t_lim=(0, 1), y_lim=(-0.35, 0.5),
                                show_contr=True):
        """
        Plots the sarcomere length change over time for a motion object, overlaying multiple trajectories.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        number_contr : int, optional
            The number of contractions to overlay. If None, all contractions are overlaid. Defaults to None.
        t_lim : tuple, optional
            The time limits for the plot. Defaults to (0, 1).
        y_lim : tuple, optional
            The y limits for the plot. Defaults to (-0.35, 0.45).
        show_contr : bool, optional
            Whether to show the contractions. Defaults to True.
        """
        # plot limits and params
        if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
            start_contr_t = motion_obj.loi_data['start_contr'][number_contr]
            tlim = (start_contr_t + t_lim[0], start_contr_t + t_lim[1])
            idxlim = (int(tlim[0] / motion_obj.metadata['frametime']), int(tlim[1] / motion_obj.metadata['frametime']))
        else:
            tlim, idxlim = (None, None), (None, None)
        # get data
        time = motion_obj.loi_data['time']
        delta_slen = motion_obj.loi_data['delta_slen']
        delta_slen_avg = motion_obj.loi_data['delta_slen_avg']
        # plot contraction cycles
        if show_contr:
            for start_i, time_i in zip(motion_obj.loi_data['start_contr'],
                                       motion_obj.loi_data['time_contr']):
                end_i = start_i + time_i
                if number_contr is not None:
                    start_i -= tlim[0]
                    end_i -= tlim[0]
                ax.fill_betweenx([0, 1], [start_i, start_i], [end_i, end_i], color='lavender',
                                 transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

        # colormap
        cm = plt.cm.nipy_spectral(np.linspace(0, 1, len(delta_slen)))
        ax.set_prop_cycle('color', list(cm))

        # plot single and average trajectories
        if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
            ax.plot(time[:idxlim[1] - idxlim[0]], delta_slen.T[idxlim[0]:idxlim[1]], linewidth=0.5)
            ax.plot(time[:idxlim[1] - idxlim[0]], delta_slen_avg[idxlim[0]:idxlim[1]], c='k', linewidth=2,
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
        PlotUtils.polish_xticks(ax, 0.5, 0.25)

    @staticmethod
    def plot_overlay_velocity(ax, motion_obj: Motion, number_contr=None, t_lim=(0, 0.9), y_lim=(-9, 12),
                              show_contr=True):
        """
        Plots overlay of sarcomere velocity time series of the motion object

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        number_contr : int, optional
            The number of contractions to overlay. If None, all contractions are overlaid. Defaults to None.
        t_lim : tuple, optional
            The time limits for the plot. Defaults to (0, 0.9).
        y_lim : tuple, optional
            The y limits for the plot. Defaults to (-7, 10).
        show_contr : bool, optional
            Whether to show the contractions. Defaults to True.
        """
        # plot limits and params
        if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
            start_contr_t = motion_obj.loi_data['start_contr'][number_contr]
            tlim = (start_contr_t + t_lim[0], start_contr_t + t_lim[1])
            idxlim = (int(tlim[0] / motion_obj.metadata['frametime']), int(tlim[1] / motion_obj.metadata['frametime']))
        else:
            tlim, idxlim = (None, None), (None, None)
        # get data
        time = motion_obj.loi_data['time']
        vel = motion_obj.loi_data['vel']
        vel_avg = motion_obj.loi_data['vel_avg']

        # plot contraction cycles
        if show_contr:
            for start_i, time_i in zip(motion_obj.loi_data['start_contr'],
                                       motion_obj.loi_data['time_contr']):
                end_i = start_i + time_i
                if number_contr is not None:
                    start_i -= tlim[0]
                    end_i -= tlim[0]
                ax.fill_betweenx([0, 1], [start_i, start_i], [end_i, end_i], color='lavender',
                                 transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

        # colormap
        cm = plt.cm.nipy_spectral(np.linspace(0, 1, len(vel)))
        ax.set_prop_cycle('color', list(cm))

        # plot single and average trajectories
        if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
            ax.plot(time[:idxlim[1] - idxlim[0]], vel.T[idxlim[0]:idxlim[1]], linewidth=0.5)
            ax.plot(time[:idxlim[1] - idxlim[0]], vel_avg[idxlim[0]:idxlim[1]], c='k', linewidth=2,
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
        ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))

    @staticmethod
    def plot_phase_space(ax: Axes, motion_obj: Motion, t_lim=(0, 4), number_contr=None, frame=None):
        """
        Plots sarcomere trajectory in length-change velocity phase space

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        motion_obj : Motion
            The motion object to plot.
        t_lim : tuple, optional
            The time limits for the plot. Defaults to (0, 4).
        number_contr : int, optional
            The number of contractions to overlay. If None, all contractions are overlaid. Defaults to None.
        frame : int, optional
            The frame number to plot the individual sarcomeres in phase space. Defaults to None.
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
        if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
            start_contr_t = motion_obj.loi_data['start_contr'][number_contr]
            tlim = (start_contr_t + t_lim[0], start_contr_t + t_lim[1])
            idxlim = (int(tlim[0] / motion_obj.metadata['frametime']), int(tlim[1] / motion_obj.metadata['frametime']))
        else:
            tlim, idxlim = (None, None), (None, None)
        for i, (vel_i, delta_i) in enumerate(zip(vel, delta_slen)):
            ax.plot(vel_i[idxlim[0]:idxlim[1]], delta_i[idxlim[0]:idxlim[1]], c='r', alpha=0.35, lw=0.2, zorder=1)
            if isinstance(frame, numbers.Integral):
                ax.scatter(vel_i[frame], delta_i[frame], c=cm[i], s=10,
                           zorder=2)

        ax.plot(vel_avg[idxlim[0]:idxlim[1]], delta_slen_avg[idxlim[0]:idxlim[1]], c='k', lw=1, label='Average')
        legend_elements = [Line2D([0], [0], color='k', lw=2), Line2D([0], [0], color='r', alpha=0.35, lw=0.5)]
        ax.legend(legend_elements, ['Average', 'Individual'], loc='upper right')
        PlotUtils.polish_xticks(ax, 5, 2.5)
        PlotUtils.polish_yticks(ax, 0.2, 0.1)
        ax.set_xlabel('Velocity $V$ [µm/s]', fontsize=PlotUtils.fontsize)
        ax.set_ylabel('Length change $\Delta SL$ [µm]', fontsize=PlotUtils.fontsize)

    @staticmethod
    def plot_popping_events(motion_obj: Motion, save_name=None):
        """
        Create binary event map of popping events of the motion object.

        Parameters
        ----------
        motion_obj : Motion
            The motion object to plot.
        save_name : str, optional
            The name to save the plot as. If None, the plot is not saved. Defaults to None.
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
