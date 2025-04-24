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



from typing import Union

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, FuncFormatter
from scipy.stats import gaussian_kde

from sarcasm import SarcAsM, Motion


class PlotUtils:
    """Helper function and parameters for plotting"""

    # Plot parameters
    fontsize = 8
    markersize = 3
    labelpad = 1
    dpi = 600
    save_format = 'png'
    width_1cols = 3.5
    width_1p5cols = 5
    width_2cols = 7.1

    def __init__(self):
        # Apply plot parameters
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        plt.rcParams.update({
            'font.size': self.fontsize,
            'axes.labelpad': self.labelpad,
            'font.family': 'arial'
        })

    @staticmethod
    def label_all_panels(axs: dict, offset=(-0.1, 1.1), color='k'):
        """
        Labels all panels in a given dictionary of Axes objects.

        Parameters:
            axs (dict): A dictionary of Axes objects.
            offset (tuple, optional): The x and y offset for the labels. Defaults to (-0.1, 1.1).
            color (str, optional): The color of the labels. Defaults to 'k' (black).
        """
        for key in axs.keys():
            PlotUtils.label_panel(axs[key], key, offset=offset, color=color)

    @staticmethod
    def label_panel(ax: Axes, label: str, offset=(-0.1, 1.1), color='k'):
        """
        Labels a single panel with the specified label.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object representing the panel to be labeled.
            label (str): The label to be displayed.
            offset (tuple, optional): The x and y offset for the label. Defaults to (-0.1, 1.1).
            color (str, optional): The color of the label. Defaults to 'k' (black).
        """
        ax.text(offset[0], offset[1], label, transform=ax.transAxes,
                fontsize=PlotUtils.fontsize + 1, fontweight='black', va='top', ha='right', color=color)

    @staticmethod
    def remove_all_spines(axs: dict):
        """
        Removes the spines from all panels in a given dictionary of Axes objects.

        Parameters:
            axs (dict): A dictionary of Axes objects.
        """
        for key in axs.keys():
            PlotUtils.remove_spines(axs[key])

    @staticmethod
    def remove_spines(ax):
        """
        Removes the spines from a single panel.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object representing the panel.
        """
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    @staticmethod
    def change_color_spines(ax, c='w', linewidth=1):
        """
        Changes the color of the spines (borders) of a single panel.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object representing the panel.
            c (str, optional): The color of the spines. Defaults to 'w' (white).
        """
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
            spine.set_color(c)

    @staticmethod
    def remove_ticks(ax):
        """
        Removes the ticks (markings) from both x-axis and y-axis of a single panel.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object representing the panel.
        """
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def polish_xticks(ax, major, minor, pad=3, radian=False):
        """
        Formats and polishes the x-ticks (markings) of a single panel.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object representing the panel.
            major (float): The major tick spacing.
            minor (float): The minor tick spacing.
            pad (float, optional): The padding between the x-axis and the tick labels. Defaults to 3.
            radian (bool, optional): Flag to format ticks in radians. Defaults to False.
        """
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ax.tick_params(axis='x', pad=pad)

        if radian:
            def radian_formatter(x, pos):
                mapping = {
                    0: '0',
                    np.pi / 4: r'$\frac{\pi}{4}$',
                    -np.pi / 4: r'$-\frac{\pi}{4}$',
                    np.pi / 2: r'$\frac{\pi}{2}$',
                    -np.pi / 2: r'$-\frac{\pi}{2}$',
                    3 * np.pi / 4: r'$\frac{3\pi}{4}$',
                    -3 * np.pi / 4: r'$-\frac{3\pi}{4}$',
                    np.pi: r'$\pi$',
                    -np.pi: r'$-\pi$',
                    5 * np.pi / 4: r'$\frac{5\pi}{4}$',
                    -5 * np.pi / 4: r'$-\frac{5\pi}{4}$',
                    3 * np.pi / 2: r'$\frac{3\pi}{2}$',
                    -3 * np.pi / 2: r'$-\frac{3\pi}{2}$',
                    7 * np.pi / 4: r'$\frac{7\pi}{4}$',
                    -7 * np.pi / 4: r'$-\frac{7\pi}{4}$',
                    2 * np.pi: r'$2\pi$',
                    -2 * np.pi: r'$-2\pi$'
                }
                return mapping.get(x, f'{x / np.pi:.1f}π')

            ax.xaxis.set_major_formatter(FuncFormatter(radian_formatter))
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

    @staticmethod
    def polish_yticks(ax, major, minor, pad=3):
        """
        Formats and polishes the y-ticks (markings) of a single panel.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object representing the panel.
            major (float): The major tick spacing.
            minor (float): The minor tick spacing.
            pad (float, optional): The padding between the y-axis and the tick labels. Defaults to 3.
        """
        ax.yaxis.set_major_locator(MultipleLocator(major))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_minor_locator(MultipleLocator(minor))
        ax.tick_params(axis='y', pad=pad)

    @staticmethod
    def plot_box(ax, xlim, ylim, c='w', lw=1, linestyle='-'):
        """
        Plots a box around a given area defined by the x-axis and y-axis limits.

        Parameters:
            ax (matplotlib.axes.Axes): The Axes object representing the panel.
            xlim (tuple): The x-axis limits of the box.
            ylim (tuple): The y-axis limits of the box.
            c (str, optional): The color of the box. Defaults to 'w' (white).
            lw (float, optional): The linewidth of the box. Defaults to 1.
            linestyle (str, optional): The linestyle of the box. Defaults to '-' (solid line).
        """
        ax.plot((xlim[0], xlim[1]), (ylim[0], ylim[0]), c=c, lw=lw, linestyle=linestyle)
        ax.plot((xlim[0], xlim[1]), (ylim[1], ylim[1]), c=c, lw=lw, linestyle=linestyle)
        ax.plot((xlim[0], xlim[0]), (ylim[0], ylim[1]), c=c, lw=lw, linestyle=linestyle)
        ax.plot((xlim[1], xlim[1]), (ylim[1], ylim[0]), c=c, lw=lw, linestyle=linestyle)

    @staticmethod
    def jitter(x, y, width=0.02):
        """Adds a small amount of random noise to the x-coordinates of the points to prevent overlap.

        Parameters
        ----------
        x : array-like
            The x-coordinates of the points.
        y : array-like
            The y-coordinates of the points.
        width : float, optional
            The maximum width of the random noise. Defaults to 0.02.

        Returns
        -------
        array-like
            The jittered x-coordinates of the points.
        """
        # Estimate the density of y
        density = gaussian_kde(y)
        y_density = density(y)

        # Scale the width of the jitter by the density of y
        jitter_width = width * y_density / y_density.max()

        return x + np.random.uniform(-jitter_width, jitter_width, size=x.shape)

    def boxplot_with_points(ax, data, labels, width=0.1, alpha=0.5, s=10, whis=(5, 95), rotation=90):
        """
        Creates a boxplot with scattered points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        data : array-like
            The data to plot.
        labels : list of str
            The labels for the boxplots.
        width : float, optional
            The width of the boxplots. Defaults to 1.25.
        alpha : float, optional
            The alpha value for the points. Defaults to 0.005.
        s : int, optional
            The size of the points. Defaults to 10.
        whis : tuple of float, optional
            The whisker range for the boxplots. Defaults to (5, 95).
        rotation : int, optional
            The rotation angle for the x-axis labels. Defaults to 90.

        Returns
        -------
        None
        """
        # make boxplot
        ax.boxplot(data, showfliers=False, labels=labels, zorder=1, whis=whis)

        # plot points in background
        for i, d in enumerate(data):
            x = PlotUtils.jitter(np.ones_like(d) * (i + 1), d, width=width)
            ax.scatter(x, d, c='k', alpha=alpha, zorder=0, s=s)

        # rotate labels
        ax.tick_params(axis='x', labelrotation=rotation)

    @staticmethod
    def plot_func_to_img(sarc_obj: Union[SarcAsM, Motion], plot_func, img_filename, figsize=(6, 6), scalebar=False,
                         dpi=300):
        """
        Generates a plot using a specified plotting function and saves it as an image file.

        Parameters
        ----------
        sarc_obj : object
            The object containing data to be plotted, which can be an instance of SarcAsM or Motion
        plot_func : function
            The plotting function that takes an Axes object, a sarcomere object, and a boolean indicating
            whether to include a scalebar. It should plot the data onto the provided Axes object.
        img_filename : str
            The filename (including path) where the image will be saved.
        figsize : tuple of int or float, optional
            The size of the figure (width, height) in inches. Default is (6, 6).
        scalebar : bool, optional
            Indicates whether a scalebar should be included in the plot. Default is False.
        dpi : int, optional
            Quality of the plot. Default is 300.
        """

        # create matplotlib figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # create plot with plot_func
        plot_func(ax=ax, sarc_obj=sarc_obj, scalebar=scalebar)

        # Remove axes
        ax.axis('off')

        # Save the figure without edges or padding
        fig.savefig(img_filename, bbox_inches='tight', pad_inches=0, dpi=dpi)

        # Close the figure
        plt.close(fig)
