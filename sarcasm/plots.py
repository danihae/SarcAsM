import os.path

import matplotlib as mpl
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt, transforms
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.stats import gaussian_kde
from tifffile import tifffile

from .utils import shuffle_labels

# plot params
fontsize = 8
markersize = 3
labelpad = 1
dpi = 600
save_format = 'png'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

width_1cols = 3.5
width_1p5cols = 5
width_2cols = 7.1
plt.rcParams.update({'font.size': fontsize, 'axes.labelpad': labelpad, 'font.family': 'arial'})
if 'seaborn-v0_8-paper' in mpl.style.available:
    plt.style.use('seaborn-v0_8-paper')

# dict with feature names
feature_names = {'z_intensity': 'Z-band intensity [a.u.]', 'z_intensity_mean': 'Z-band intensity [a.u.]',
                 'z_intensity_std': 'Z-band intensity STD [a.u.]', 'z_length': 'Z-band length [µm]',
                 'z_length_mean': 'Z-band length [µm]', 'z_length_std': 'Z-band length STD [µm]',
                 'z_length_max': 'Z-band length MAX [µm]', 'z_ratio_intensity': 'Z-band intensity ratio',
                 'z_straightness': 'Z-band straightness', 'z_straightness_mean': 'Z-band straightness',
                 'z_straightness_std': 'Z-band straightness STD',
                 'z_lat_neighbors': 'Lateral neighbors [#]', 'z_lat_neighbors_mean': 'Lateral neighbors [#]',
                 'z_lat_neighbors_std': 'Lateral neighbors STD [#]',
                 'z_lat_alignment': 'Z-band lat. alignment', 'z_lat_alignment_mean': 'Z-band lat. alignment',
                 'z_lat_alignment_std': 'Z-band lat. alignment STD',
                 'z_lat_dist': 'Z-band lat. dist. [µm]', 'z_lat_dist_mean': 'Z-band lat. dist. [µm]',
                 'z_lat_dist_std': 'Z-band lat. dist. STD [µm]',
                 'sarcomere_area': 'Sarcomere area [µm$^2$]', 'sarcomere_area_ratio': 'Sarcomere area ratio',
                 'cell_area': 'Cell area [µm$^2$]', 'cell_area_ratio': 'Cell area ratio',
                 'sarcomere_length': 'Sarcomere length [µm]',
                 'sarcomere_length_mean': 'Sarcomere length [µm]', 'sarcomere_length_std': 'Sarc. length STD [µm]',
                 'sarcomere_orientation': 'Sarcomere orient. [°]', 'sarcomere_oop': 'Sarcomere OOP',
                 'sarcomere_orientation_mean': 'Sarcomere orient. [°]',
                 'sarcomere_orientation_std': 'Sarcomere orient. STD [°]',
                 'myof_length': 'Myofibril length [µm]', 'myof_length_mean': 'Myofibril length [µm]',
                 'myof_length_std': 'Myofibril length STD [µm]',
                 'myof_length_max': 'Myofibril length MAX [µm]', 'myof_length_median': 'Myofibril length MEDIAN [µm]',
                 'myof_msc': 'Myofibril MSC', 'myof_msc_mean': 'Myofibril MSC', 'myof_msc_std': 'Myofibril MSC STD',
                 'myof_msc_median': 'Myofibril MSC MEDIAN', 'domain_area': 'Domain area [µm$^2$]',
                 'domain_area_mean': 'Domain area [µm$^2$]', 'domain_area_std': 'Domain area STD [µm$^2$]',
                 'domain_area_median': 'Domain area MEDIAN [µm$^2$]', 'domain_oop': 'Domain OOP',
                 'domain_oop_mean': 'Domain OOP', 'domain_oop_median': 'Domain OOP', 'domain_oop_std': 'Domain OOP'}

feature_names_motion = {'contr_max': '$\Delta SL_-$', 'contr_max_avg': '$\overline{\Delta SL}_-$',
                        'elong_max': '$\Delta SL_+$', 'elong_max_avg': '$\overline{\Delta SL}_+$',
                        'vel_contr_max': '$V_-$', 'vel_contr_max_avg': '$\overline{V}_-$',
                        'vel_elong_max': '$V_+$', 'vel_elong_max_avg': '$\overline{V}_+$',
                        'equ': 'Resting length [µm]', 'frequency': 'Frequency [Hz]', 'periodicity': 'Aperiodicity [s]',
                        'time_to_peak': 'Time-to-peak single [s]', 'time_to_peak_avg': 'Time-to-peak avg. [s]',
                        'popping_freq': 'Popping frequency', 'ratio_delta_slen_mutual_serial': 'R$_{\Delta SL}$',
                        'ratio_vel_mutual_serial': 'R$_{V}$', 'corr_delta_slen_serial': 'r$_s, \Delta SL$',
                        'corr_delta_slen_mutual': 'r$_m, \Delta SL$', 'corr_vel_serial': 'r$_s, V$',
                        'corr_vel_mutual': 'r$_m, V$', 'smi': 'Surplus motion index'}


def label_all_panels(axs, offset=(-0.1, 1.1), color='k'):
    """
    Labels all panels in a given dictionary of Axes objects.

    Parameters:
        axs (dict): A dictionary of Axes objects.
        offset (tuple, optional): The x and y offset for the labels. Defaults to (-0.1, 1.1).
        color (str, optional): The color of the labels. Defaults to 'k' (black).
    """
    for key in axs.keys():
        if not key.isupper():
            continue
        label_panel(axs[key], key, offset=offset, color=color)


def label_panel(ax, label, offset=(-0.1, 1.1), color='k'):
    """
    Labels a single panel with the specified label.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object representing the panel to be labeled.
        label (str): The label to be displayed.
        offset (tuple, optional): The x and y offset for the label. Defaults to (-0.1, 1.1).
        color (str, optional): The color of the label. Defaults to 'k' (black).
    """
    ax.text(offset[0], offset[1], label, transform=ax.transAxes,
            fontsize=fontsize + 1, fontweight='black', va='top', ha='right', color=color)


def remove_all_spines(axs):
    """
    Removes the spines from all panels in a given dictionary of Axes objects.

    Parameters:
        axs (dict): A dictionary of Axes objects.
    """
    for key in axs.keys():
        remove_spines(axs[key])


def remove_spines(ax):
    """
    Removes the spines from a single panel.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object representing the panel.
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def change_color_spines(ax, c='w'):
    """
    Changes the color of the spines (borders) of a single panel.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object representing the panel.
        c (str, optional): The color of the spines. Defaults to 'w' (white).
    """
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['right'].set_color(c)
    ax.spines['left'].set_color(c)


def remove_ticks(ax):
    """
    Removes the ticks (markings) from both x-axis and y-axis of a single panel.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object representing the panel.
    """
    ax.set_xticks([])
    ax.set_yticks([])


def polish_xticks(ax, major, minor, pad=3):
    """
    Formats and polishes the x-ticks (markings) of a single panel.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object representing the panel.
        major (float): The major tick spacing.
        minor (float): The minor tick spacing.
        pad (float, optional): The padding between the x-axis and the tick labels. Defaults to 3.
    """
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(axis='x', pad=pad)


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


def plot_stack_overlay(ax, sarc_obj, timepoints, plot_func, offset=0.025, spine_color='w', xlim=None, ylim=None):
    """
    Plots a stack of overlayed subplots on a given Axes object.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object on which the stack should be plotted.
        sarc_obj: The object or data to be plotted in each subplot.
        timepoints (list): The timepoints at which the subplots should be created.
        plot_func: The function used to plot the data in each subplot.
        offset (float, optional): The offset between each subplot. Defaults to 0.025.
        spine_color (str, optional): The color of the spines (borders) of each subplot. Defaults to 'w' (white).
        xlim (tuple, optional): The x-axis limits for each subplot. Defaults to None.
        ylim (tuple, optional): The y-axis limits for each subplot. Defaults to None.
    """
    ax.axis('off')
    for i, t in enumerate(timepoints):
        ax_t = ax.inset_axes([0.1 + offset * i, 0.1 - offset * i, 0.8, 0.8])

        plot_func(ax_t, sarc_obj, t)
        ax_t.spines['bottom'].set_color(spine_color)
        ax_t.spines['top'].set_color(spine_color)
        ax_t.spines['right'].set_color(spine_color)
        ax_t.spines['left'].set_color(spine_color)

        ax_t.set_xlim(xlim)
        ax_t.set_ylim(ylim)


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
        x = jitter(np.ones_like(d) * (i + 1), d, width=width)
        ax.scatter(x, d, c='k', alpha=alpha, zorder=0, s=s)

    # rotate labels
    ax.tick_params(axis='x', labelrotation=rotation)


def plot_summary_structure(sarc_obj, save_format='png'):
    """
    Plots a summary of the structure of the sarcomere object.

    Parameters
    ----------
    sarc_obj : object
        The sarcomere object to plot.
    save_format : str, optional
        The format to save the plot. Defaults to 'png'.
    """

    mosaic = """
    AAADDD
    BBBEEE
    CCCFFF
    GGHHII
    JJKKLL
    """

    fig, axs = plt.subplot_mosaic(mosaic=mosaic, figsize=(width_2cols, width_2cols))

    # image
    title = f'{sarc_obj.filename}'
    fig.suptitle(title)

    plot_image(axs['A'], sarc_obj)
    plot_z_bands(axs['B'], sarc_obj)
    plot_z_segmentation(axs['C'], sarc_obj)
    plot_sarcomere_lengths(axs['D'], sarc_obj)
    plot_sarcomere_orientations(axs['E'], sarc_obj)
    plot_myofibrils(axs['F'], sarc_obj)

    plot_histogram_structure(axs['G'], sarc_obj, feature='z_length', label='Z-band lengths [µm]')
    plot_histogram_structure(axs['H'], sarc_obj, feature='z_intensity', label='Z-band intensity [a.u.]')
    plot_histogram_structure(axs['I'], sarc_obj, feature='z_straightness', label='Z-band straightness [a.u.]')
    plot_histogram_structure(axs['J'], sarc_obj, feature='sarcomere_length_points', label='Sarcomere lengths [µm]')
    plot_histogram_structure(axs['K'], sarc_obj, feature='sarcomere_orientation_points',
                             label='Sarcomere orientation [°]')
    plot_histogram_structure(axs['L'], sarc_obj, feature='myof_line_lengths', label='Myofibril lengths [µm]')

    label_all_panels(axs)
    plt.tight_layout()

    fig.savefig(sarc_obj.analysis_folder + 'summary_structure.' + save_format, dpi=dpi)
    plt.show()


def plot_loi_summary_motion(motion_obj, number_contr=0, t_lim=(-0.1, 3), filename=None):
    """
    Plots a summary of the motion of the line of interest (LOI).

    Parameters
    ----------
    motion_obj : object
        The motion object to plot.
    number_contr : int, optional
        The number of contractions to plot. Defaults to 0.
    t_lim : tuple of float, optional
        The time limits for the plot. Defaults to (0, 0.9).
    filename : str, optional
        The filename to save the plot. Defaults to None.
    """

    mosaic = """
    ACC
    BCC
    DDE
    DDF
    """

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(width_2cols, width_2cols))
    title = f'File: {motion_obj.filename}, LOI: {motion_obj.loi_name}'
    fig.suptitle(title, fontsize=fontsize)

    # A- image cell w/ LOI
    plot_image(axs['A'], motion_obj)

    # B- U-Net cell w/ LOI
    plot_z_bands(axs['B'], motion_obj)

    # C- kymograph and tracked z-lines
    plot_z_pos(axs['C'], motion_obj)

    # D- single sarcomere trajs (vel and delta slen)
    plot_delta_slen(axs['D'], motion_obj)

    # E- overlay delta slen
    plot_overlay_delta_slen(axs['E'], motion_obj, number_contr=number_contr, t_lim=t_lim)

    # F- overlay velocity
    plot_overlay_velocity(axs['F'], motion_obj, number_contr=number_contr, t_lim=t_lim)

    label_all_panels(axs)

    plt.tight_layout()
    if filename is None:
        filename = motion_obj.loi_folder + 'summary_loi.png'
    fig.savefig(filename, dpi=dpi)
    plt.show()


def plot_loi_detection(sarc_obj, timepoint=0, filepath=None):
    """Plots all steps of automated LOI finding algorithm"""
    mosaic = """
    a
    b
    c
    d
    """

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(width_1cols, 4.6), constrained_layout=True, dpi=300)

    points = sarc_obj.structure.data['points'][timepoint]

    if isinstance(sarc_obj.structure.data['params.wavelet_timepoints'], int):
        frame = sarc_obj.structure.data['params.wavelet_timepoints']
    elif sarc_obj.structure.data['params.wavelet_timepoints'] == 'all':
        frame = timepoint
    else:
        frame = sarc_obj.structure.data['params.wavelet_timepoints'][timepoint]

    plot_z_bands(axs['a'], sarc_obj, timepoint=frame, invert=True)
    plot_z_bands(axs['c'], sarc_obj, timepoint=frame, invert=True)
    plot_z_bands(axs['d'], sarc_obj, timepoint=frame, invert=True)

    for i, loi_i in enumerate(sarc_obj.structure.data['lines']):
        if sarc_obj.structure.data['quality'][timepoint] == 1:
            axs['a'].plot(points[1, loi_i], points[0, loi_i], c='r', lw=0.2, alpha=0.6)

    axs['b'].hist(sarc_obj.structure.data['hausdorff_dist_matrix'].reshape(-1), bins=100, color='k', alpha=0.75, rwidth=0.75)
    axs['b'].set_xlim(0, 400)
    axs['b'].set_xlabel('Hausdorff distance')
    axs['b'].set_ylabel('# LOI pairs')

    for i, loi_i in enumerate(sarc_obj.structure.data['good_lois']):
        label_i = sarc_obj.structure.data['good_loi_cluster'][i]
        axs['c'].plot(sarc_obj.structure.data['good_lois_pos'][i].T[1], sarc_obj.structure.data['good_lois_pos'][i].T[0],
                      c=plt.cm.jet(label_i / sarc_obj.structure.data['num_good_loi_clusters']), lw=0.2)

    for i, line_i in enumerate(sarc_obj.structure.data['loi_lines']):
        axs['d'].plot(line_i.T[0], line_i.T[1], lw=2, label=i)
    axs['d'].legend(title='LOI #', bbox_to_anchor=(0.5, -0.75), loc='lower center', borderaxespad=0., ncol=6)

    label_all_panels(axs, offset=(0.05, 0.9))

    axs['a'].set_title('1. Line growth', ha='left', x=0.02, fontsize=fontsize + 1, fontweight='bold')
    axs['b'].set_title('2. Pair-wise Hausdorff distance', ha='left', x=0.02, fontsize=fontsize + 1, fontweight='bold')
    axs['c'].set_title('3. Agglomerative clustering', ha='left', x=0.02, fontsize=fontsize + 1, fontweight='bold')
    axs['d'].set_title('4. Fit lines through clusters', ha='left', x=0.02, fontsize=fontsize + 1, fontweight='bold')

    if filepath is None:
        fig.savefig(sarc_obj.analysis_folder + 'LOIs.png', dpi=300)
    plt.show()


def plot_func_to_img(sarc_obj, plot_func, img_filename, figsize=(6, 6), scalebar=False, dpi=300):
    """
    Generates a plot using a specified plotting function and saves it as an image file.

    Parameters
    ----------
    sarc_obj : object
        The sarcomere object containing data to be plotted.
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


def plot_image(ax, sarc_obj, timepoint=0, clip_thrs=(1, 99), rotate=False, scalebar=True, title=None, show_loi=True):
    """
    Plots microscopy raw image of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    clip_thrs : tuple of int, optional
        The thresholds to clip the image. Defaults to (1, 98).
    rotate : bool, optional
        Whether to rotate the image. Defaults to False.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    title : str, optional
        The title for the plot. Defaults to None.
    show_loi : bool, optional
        Whether to show the line of interest (LOI). Defaults to True.
    """

    img = sarc_obj.structure.read_imgs(timepoint=timepoint)
    if rotate:
        img = img.T
    img = np.clip(img, np.percentile(img, clip_thrs[0]), np.percentile(img, clip_thrs[1]))
    img = 1 - img / np.max(img)
    plot = ax.imshow(img, cmap='Greys')
    if hasattr(sarc_obj, 'loi_data') and show_loi:
        line = sarc_obj.loi_data['line']
        if rotate:
            ax.plot(line.T[1], line.T[0], color='r', linewidth=2, alpha=0.5)
        else:
            ax.plot(line.T[0], line.T[1], color='r', linewidth=2, alpha=0.5)
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                               height_fraction=0.04, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)


def plot_z_bands(ax, sarc_obj, timepoint=0, rotate=False, invert=False, alpha=1, scalebar=True, title=None,
                 show_loi=True):
    """
    Plots the Z-bands of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    rotate : bool, optional
        Whether to rotate the image. Defaults to False.
    invert : bool, optional
        Whether to invert the image. Defaults to False.
    alpha : float, optional
        Alpha value to change opacity of image. Defaults to 1
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    title : str, optional
        The title for the plot. Defaults to None.
    show_loi : bool, optional
        Whether to show the line of interest (LOI). Defaults to True.
    """
    assert os.path.exists(sarc_obj.file_sarcomeres), ('Z-band mask not found. Run predict_z_bands first.')

    img = tifffile.imread(sarc_obj.file_sarcomeres, key=timepoint)
    if invert:
        img = 255 - img
    if rotate:
        img = img.T
    ax.imshow(img, cmap='gray', alpha=alpha)
    if hasattr(sarc_obj, 'loi_data') and show_loi:
        line = sarc_obj.loi_data['line']
        if rotate:
            ax.plot(line.T[1], line.T[0], color='r', linewidth=2, alpha=0.5)
        else:
            ax.plot(line.T[0], line.T[1], color='r', linewidth=2, alpha=0.5)
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k' if invert else 'w',
                               sep=1, height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)


def plot_cell_area(ax, sarc_obj, timepoint=0, rotate=False, invert=False, scalebar=True, title=None):
    """
    Plots the cell area of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    rotate : bool, optional
        Whether to rotate the image. Defaults to False.
    invert : bool, optional
        Whether to invert the image. Defaults to False.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    title : str, optional
        The title for the plot. Defaults to None.
    """
    assert os.path.exists(sarc_obj.file_cell_mask), ('Cell mask not found. Run predict_cell_area first.')

    img = tifffile.imread(sarc_obj.file_cell_mask, key=timepoint)
    if invert:
        img = 255 - img
    if rotate:
        img = img.T
    ax.imshow(img, cmap='gray')
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='w', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)


def plot_z_segmentation(ax, sarc_obj, timepoint=0, scalebar=True, shuffle=True, title=None):
    """
    Plots the Z-band segmentation result of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    shuffle : bool, optional
        Whether to shuffle the labels. Defaults to True.
    title : str, optional
        The title for the plot. Defaults to None.
    """
    assert 'z_labels' in sarc_obj.structure.data.keys(), ('Z-bands not yet analyzed. '
                                                     'Run analyze_z_bands first.')

    labels = sarc_obj.structure.data['z_labels'][timepoint].toarray()
    if shuffle:
        labels = shuffle_labels(labels)
    labels_plot = labels.astype('float16')
    labels_plot[labels == 0] = np.nan
    ax.imshow(labels_plot, cmap='prism')
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)


def plot_z_dist_alignment(ax, sarc_obj, timepoint=0, scalebar=True, markersize=5, linewidth=1, shuffle=True,
                          title=None):
    """
    Plots lateral Z-band distance and alignment of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    markersize : int, optional
        The size of the markers. Defaults to 5.
    linewidth : int, optional
        The width of the lines. Defaults to 1.
    shuffle : bool, optional
        Whether to shuffle the labels. Defaults to True.
    title : str, optional
        The title for the plot. Defaults to None.
    """
    assert 'z_labels' in sarc_obj.structure.data.keys(), ('Z-bands not yet analyzed. '
                                                     'Run analyze_z_bands first.')

    labels = sarc_obj.structure.data['z_labels'][timepoint].toarray()
    if shuffle:
        labels = shuffle_labels(labels)
    z_ends = sarc_obj.structure.data['z_ends'][timepoint] / sarc_obj.metadata['pixelsize']
    z_links = sarc_obj.structure.data['z_links'][timepoint]
    labels_plot = labels.copy().astype('float32')
    labels_plot[labels == 0] = np.nan
    ax.imshow(labels_plot, cmap='prism')
    for (i, k, j, l) in z_links.T:
        ax.plot([z_ends[i, k, 1], z_ends[j, l, 1]],
                [z_ends[i, k, 0], z_ends[j, l, 0]],
                c='gray', lw=linewidth, linestyle='-', alpha=1, zorder=2)
    ax.scatter(z_ends[:, 0, 1], z_ends[:, 0, 0], c='k', marker='o', s=markersize, zorder=3)
    ax.scatter(z_ends[:, 1, 1], z_ends[:, 1, 0], c='k', marker='o', s=markersize, zorder=3)
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)


def plot_wavelet_bank(ax, sarc_obj, gap=0.005):
    """
    Plots a wavelet filter bank with two channels (red and blue) on a given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the wavelet bank.
    sarc_obj : object
        An SarcAsM object containing the wavelet bank in its 'structure' dict.
    gap : float, optional
        The gap size between individual plots as a fraction of figure size. Default is 0.005.

    Returns
    -------
    None
    """
    assert 'wavelet_bank' in sarc_obj.structure.data.keys(), ('No wavelet bank stored. '
                                                         'Run sarc_obj.analyze_sarcomere_length_orient '
                                                         'with save_all=True.')

    bank = sarc_obj.structure.data['wavelet_bank']
    if bank is None:
        raise ValueError('Wavelet bank is not saved. Run sarc_obj.analyze_sarcomere_length_orient with save_all=True.')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    rows, cols = bank.shape[:2]
    for i in range(rows):
        for j in range(cols):
            # Calculate bounds for the inset axes, including gaps
            x = j / cols + gap / 2
            y = 1 - (i + 1) / rows + gap / 2  # Inverting y to start from top left
            width = 1 / cols - gap
            height = 1 / rows - gap

            # Create an inset axis for each filter
            inset = ax.inset_axes([x, y, width, height])

            # Plot the filter
            kernel_0, kernel_1 = bank[i, j, 0], bank[i, j, 1]

            inset.imshow(kernel_0 - kernel_1, cmap='seismic', aspect='equal')
            inset.set_xticks([])
            inset.set_yticks([])


def plot_wavelet_score(ax, sarc_obj, timepoint=0, score_threshold=None, lim=(1.6, 2.1), scalebar=True, colorbar=True,
                       shrink_colorbar=0.7, title=None):
    """
        Plots the maximal wavelet score of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : object
            The sarcomere object to plot.
        timepoint : int, optional
            The timepoint to plot. Defaults to 0.
        score_threshold : float, optional
            The threshold for the score. If None, the threshold from the sarcomere object is used. Defaults to None.
        lim : tuple, optional
            The limits for the colorbar. Defaults to (1.6, 2.1).
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        colorbar : bool, optional
            Whether to add a colorbar to the plot. Defaults to True.
        shrink_colorbar : float, optional
            The factor by which to shrink the colorbar. Defaults to 0.7.
        title : str, optional
            The title for the plot. Defaults to None.
        """
    assert 'wavelet_max_score' in sarc_obj.structure.data.keys(), ('No wavelet stors map stored. '
                                                              'Run sarc_obj.analyze_sarcomere_length_orient '
                                                              'with save_all=True.')

    max_score = sarc_obj.structure.data['wavelet_max_score'][timepoint].copy()
    if score_threshold is None:
        score_threshold = sarc_obj.structure.data['params.score_threshold'][timepoint]
    max_score[max_score < score_threshold] = np.nan
    plot = ax.imshow(max_score, vmin=lim[0], vmax=lim[1], cmap='gray')
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(plot, ax=ax, label='C score', shrink=shrink_colorbar)
    ax.set_title(title, fontsize=fontsize)


def plot_sarcomere_lengths(ax, sarc_obj, timepoint=0, score_threshold=None, lim=(1.6, 2.1), scalebar=True,
                           colorbar=True, shrink_colorbar=0.7, orient_colorbar='vertical', title=None):
    """
    Plots the sarcomere length obtained by wavelet analysis of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    score_threshold : float, optional
        The threshold for the score. If None, the threshold from the sarcomere object is used. Defaults to None.
    lim : tuple, optional
        The limits for the colorbar. Defaults to (1.6, 2.1).
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    colorbar : bool, optional
        Whether to add a colorbar to the plot. Defaults to True.
    shrink_colorbar : float, optional
        The factor by which to shrink the colorbar. Defaults to 0.7.
    orient_colorbar : str, optional
            The orientation of the colorbar ('horizontal' or 'vertical'). Defaults to 'vertical'.
    title : str, optional
        The title for the plot. Defaults to None.
    """
    assert 'wavelet_sarcomere_length' in sarc_obj.structure.data.keys(), ('No sarcomere length map stored. '
                                                                     'Run sarc_obj.analyze_sarcomere_length_orient '
                                                                     'with save_all=True.')

    length = sarc_obj.structure.data['wavelet_sarcomere_length'][timepoint].copy()
    max_score = sarc_obj.structure.data['wavelet_max_score'][timepoint].copy()
    if score_threshold is None:
        score_threshold = sarc_obj.structure.data['params.score_threshold'][timepoint]
    length[max_score < score_threshold] = np.nan
    plot = ax.imshow(length, vmin=lim[0], vmax=lim[1], cmap='viridis')
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(plot, ax=ax, label='Length SL [µm]', shrink=shrink_colorbar, orientation=orient_colorbar)
    ax.set_title(title, fontsize=fontsize)


def plot_sarcomere_orientations(ax, sarc_obj, timepoint=0, score_threshold=None, lim=(-90, 90), radians=False,
                                scalebar=True, colorbar=True, shrink_colorbar=0.7, orient_colorbar='vertical',
                                title=None):
    """
        Plots sarcomere orientation obtained by wavelet analysis of the sarcomere object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw the plot on.
        sarc_obj : object
            The sarcomere object to plot.
        timepoint : int, optional
            The timepoint to plot. Defaults to 0.
        score_threshold : float, optional
            The threshold for the score. If None, the threshold from the sarcomere object is used. Defaults to None.
        lim : tuple, optional
            The limits for the colorbar. Defaults to (-90, 90).
        scalebar : bool, optional
            Whether to add a scalebar to the plot. Defaults to True.
        colorbar : bool, optional
            Whether to add a colorbar to the plot. Defaults to True.
        shrink_colorbar : float, optional
            The factor by which to shrink the colorbar. Defaults to 0.7.
        orient_colorbar : str, optional
            The orientation of the colorbar ('horizontal' or 'vertical'). Defaults to 'vertical'.
        title : str, optional
            The title for the plot. Defaults to None.
        """
    assert 'wavelet_sarcomere_orientation' in sarc_obj.structure.data.keys(), ('No sarcomere orientation map stored. '
                                                                          'Run sarc_obj.analyze_sarcomere_length_orient '
                                                                          'with save_all=True.')

    orientation = sarc_obj.structure.data['wavelet_sarcomere_orientation'][timepoint].copy()
    if not radians:
        orientation = np.degrees(orientation)
    max_score = sarc_obj.structure.data['wavelet_max_score'][timepoint].copy()
    if score_threshold is None:
        score_threshold = sarc_obj.structure.data['params.score_threshold'][timepoint]
    orientation[max_score < score_threshold] = np.nan
    plot = ax.imshow(orientation, vmin=lim[0], vmax=lim[1], cmap='hsv')
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(plot, ax=ax, label=r'Angle $\theta$ [°]', shrink=shrink_colorbar, orientation=orient_colorbar)
    ax.set_title(title, fontsize=fontsize)


def plot_sarcomere_area(ax, sarc_obj, timepoint=0, cmap='viridis', show_z_bands=False, alpha=0.5, invert_z_bands=True,
                        alpha_z_bands=1):
    """
    Plots binary mask of sarcomeres, derived from sarcomere vectors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    cmap : str, optional
        The colormap to use. Defaults to 'viridis'
    show_z_bands : bool, optional
        Whether to show Z-bands. If False, the raw image is shown. Defaults to False.
    alpha : float, optional
        The transparency of sarcomere mask. Defaults to 0.5.
    invert_z_bands : bool, optional
        Whether to invert binary mask of Z-bands. Defaults to True. Only applied if show_z_bands is True.
    alpha_z_bands : float, optional
        Alpha value of Z-bands. Defaults to 1.
    """
    assert os.path.exists(sarc_obj.file_sarcomere_mask), ('No sarcomere masks stored. '
                                                          'Run sarc_obj.analyze_sarcomere_length_orient ')

    _timepoints = sarc_obj.structure.data['params.wavelet_timepoints']
    if _timepoints is not 'all':
        _timepoint = _timepoints[timepoint]
    else:
        _timepoint = timepoint

    sarcomere_mask = tifffile.imread(sarc_obj.file_sarcomere_mask, key=timepoint)

    if show_z_bands:
        plot_z_bands(ax, sarc_obj, invert=invert_z_bands, alpha=alpha_z_bands, timepoint=_timepoint)
    else:
        plot_image(ax, sarc_obj, timepoint=_timepoint)
    ax.imshow(sarcomere_mask, vmin=0, vmax=1, alpha=alpha, cmap=cmap)


def plot_sarcomere_vectors(ax, sarc_obj, timepoint=0, color_arrows='mediumpurple',
                           color_points='darkgreen', style='half', s_points=0.5, linewidths=0.001, scalebar=True,
                           legend=False, invert_z_bands=True, alpha_z_bands=1, title=None):
    """
    Plots quiver plot reflecting local sarcomere length and orientation based on
    wavelet analysis result of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    color_arrows : str, optional
        The color of the arrows. Defaults to 'mediumpurple'.
    color_points : str, optional
        The color of the points. Defaults to 'darkgreen'.
    style : str, optional
        The style of arrows ('half', 'full'). Defaults to 'half'.
    s_points : float, optional
        The size of the points. Defaults to 0.5.
    linewidths : float, optional
        The width of the arrow lines. Defaults to 1.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    legend : bool, optional
        Whether to add a legend to the plot. Defaults to False.
    invert_z_bands : bool, optional
        Whether to invert color of Z-bands. Defaults to True.
    alpha_z_bands : float, optional
        Alpha value of Z-bands. Defaults to 1.
    title : str, optional
        The title for the plot. Defaults to None.
    """
    assert 'points' in sarc_obj.structure.data.keys(), ('Sarcomere vectors not yet calculated, '
                                                   'run analyze_sarcomere_length_orient first.')

    points = sarc_obj.structure.data['points'][timepoint]
    sarcomere_orientation_points = sarc_obj.structure.data['sarcomere_orientation_points'][timepoint]
    sarcomere_length_points = sarc_obj.structure.data['sarcomere_length_points'][timepoint] / sarc_obj.metadata['pixelsize']
    orientation_vectors = np.asarray([np.cos(sarcomere_orientation_points), np.sin(sarcomere_orientation_points)])

    _timepoints = sarc_obj.structure.data['params.wavelet_timepoints']
    if _timepoints == 'all':
        plot_z_bands(ax, sarc_obj, invert=invert_z_bands, alpha=alpha_z_bands, timepoint=timepoint)
    else:
        plot_z_bands(ax, sarc_obj, invert=invert_z_bands, alpha=alpha_z_bands, timepoint=_timepoints[timepoint])

    ax.plot([0, 1], [0, 1], c='k', label='Z-bands', lw=0.5)
    ax.scatter(points[1], points[0], marker='o', c=color_points, s=s_points, label='Midline points')
    if style == 'half':
        ax.quiver(points[1], points[0], -orientation_vectors[0] * sarcomere_length_points * 0.5,
                  orientation_vectors[1] * sarcomere_length_points * 0.5, width=linewidths,
                  angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5, label='Sarcomere vectors')
        ax.quiver(points[1], points[0], orientation_vectors[0] * sarcomere_length_points * 0.5,
                  -orientation_vectors[1] * sarcomere_length_points * 0.5,
                  angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5, width=linewidths)
    if style == 'full':
        ax.quiver(points[1] - sarcomere_length_points * orientation_vectors[0] * 0.5,
                  points[0] + sarcomere_length_points * orientation_vectors[1] * 0.5,
                  orientation_vectors[0] * sarcomere_length_points * 1,
                  -orientation_vectors[1] * sarcomere_length_points * 1, width=linewidths,
                  angles='xy', scale_units='xy', scale=1, color=color_arrows, alpha=0.5)

    if legend:
        ax.legend(loc=3, fontsize=fontsize - 2)
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=fontsize)


def plot_sarcomere_domains_points(ax, sarc_obj, timepoint=0, scalebar=True, markersize=1, plot_raw_data=False,
                                  show_oop=True,
                                  title=None):
    """
    Plots the sarcomere domains of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    markersize : int, optional
        The size of the markers. Defaults to 1.
    plot_raw_data : bool, optional
        Whether to plot the raw data. Defaults to False.
    show_oop : bool, optional
        Whether to show the out of plane component. Defaults to True.
    title : str, optional
        The title for the plot. Defaults to None.

    """
    assert 'n_domains' in sarc_obj.structure.data.keys(), ('Sarcomere domains not analyzed. '
                                                      'Run analyze_sarcomere_domains first.')

    n_domains = sarc_obj.structure.data['n_domains'][timepoint]
    domains = sarc_obj.structure.data['domains'][timepoint]
    domain_oop = sarc_obj.structure.data['domain_oop'][timepoint]
    points = sarc_obj.structure.data['points'][timepoint]

    _timepoints = sarc_obj.structure.data['params.wavelet_timepoints']
    if _timepoints == 'all':
        timepoint_plot = timepoint
    else:
        timepoint_plot = _timepoints[timepoint]
    if plot_raw_data:
        plot_image(ax, sarc_obj, timepoint=timepoint_plot)
    else:
        plot_z_bands(ax, sarc_obj, invert=True, timepoint=timepoint_plot)

    cm = mpl.colormaps['jet'].resampled(n_domains)
    for i, domain_i in enumerate(domains):
        points_i = points[:, list(domain_i)].T
        ax.scatter(points_i.T[1], points_i.T[0],
                   color=cm(i), s=markersize)
        if show_oop:
            ax.text((np.mean(points_i.T[1]) + 3),
                    (np.mean(points_i.T[0]) + 8),
                    s=np.round(domain_oop[i], 3), fontsize=fontsize, weight='bold')
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_title(title, fontsize=fontsize)


def plot_sarcomere_domains(ax, sarc_obj, timepoint=0, alpha=0.5, cmap='gist_rainbow',
                           scalebar=True, plot_raw_data=False, title=None):
    """
    Plots the sarcomere domains of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    alpha : float, optional
        The transparency of the domain masks. Defaults to 0.3.
    cmap : str, optional
        The colormap to use. Defaults to 'gist_rainbow'.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    plot_raw_data : bool, optional
        Whether to plot the raw data. Defaults to False.
    title : str, optional
        The title for the plot. Defaults to None.

    """
    assert 'n_domains' in sarc_obj.structure.data.keys(), ('Sarcomere domains not analyzed. '
                                                      'Run analyze_sarcomere_domains first.')

    domain_mask = sarc_obj.structure.data['domain_mask'][timepoint].toarray().astype(float)
    domain_mask[domain_mask == 0] = np.nan

    _timepoints = sarc_obj.structure.data['params.wavelet_timepoints']
    if _timepoints == 'all':
        timepoint_plot = timepoint
    else:
        timepoint_plot = _timepoints[timepoint]
    if plot_raw_data:
        plot_image(ax, sarc_obj, timepoint=timepoint_plot)
    else:
        plot_z_bands(ax, sarc_obj, invert=True, timepoint=timepoint_plot)

    ax.imshow(domain_mask, cmap=cmap, alpha=alpha, vmin=0, vmax=np.nanmax(domain_mask))

    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.04, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_title(title, fontsize=fontsize)


def plot_myofibrils(ax, sarc_obj, timepoint=0, show_z_bands=True, linewidth=1, alpha=0.2, scalebar=True, title=None):
    """
    Plots result of myofibril line growth algorithm of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    show_z_bands : bool
        Whether or not to show Z-bands. Defaults to True
    linewidth : int, optional
        The width of the lines. Defaults to 1.
    alpha : float, optional
        The transparency of the lines. Defaults to 0.2.
    scalebar : bool, optional
        Whether to add a scalebar to the plot. Defaults to True.
    title : str, optional
        The titlefor the plot. Defaults to None.
    """
    assert 'myof_lines' in sarc_obj.structure.data.keys(), ('Myofibrils not analyzed. '
                                                       'Run analyze_myofibrils first.')

    _timepoints = sarc_obj.structure.data['params.wavelet_timepoints']
    if show_z_bands:
        if _timepoints == 'all':
            plot_z_bands(ax, sarc_obj, invert=True, timepoint=timepoint)
        else:
            plot_z_bands(ax, sarc_obj, invert=True, timepoint=_timepoints[timepoint])
    lines = sarc_obj.structure.data['myof_lines'][timepoint]
    points = sarc_obj.structure.data['points'][timepoint]
    if scalebar:
        ax.add_artist(ScaleBar(sarc_obj.metadata['pixelsize'], units='µm', frameon=False, color='k', sep=1,
                               height_fraction=0.07, location='lower right', scale_loc='top',
                               font_properties={'size': fontsize - 1}))
    ax.set_xticks([])
    ax.set_yticks([])
    for i, line_i in enumerate(lines):
        ax.plot(points[1, line_i], points[0, line_i], c='r', alpha=alpha, lw=linewidth)
    ax.set_title(title, fontsize=fontsize)


def plot_lois(ax, sarc_obj, color='g', linewidth=2):
    """
    Plot LOI lines.

    Parameters
    ----------
    ax : matplotlib axis
        Axis on which to plot the LOI lines
    sarc_obj : SarcAsM object
        Object of SarcAsM class
    color : str
        Color of lines
    linewidth : float
        Width of lines
    """
    loi_lines = sarc_obj.structure.data['loi_data']['loi_lines']

    for line in loi_lines:
        ax.plot(line.T[0], line.T[1], color=color, linewidth=linewidth)


def plot_histogram_structure(ax, sarc_obj, feature, timepoint=0, label=None, bins=20, range=None):
    """
    Plots the histogram of a structural feature of the sarcomere object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    sarc_obj : object
        The sarcomere object to plot.
    feature : str
        The feature to plot.
    timepoint : int, optional
        The timepoint to plot. Defaults to 0.
    label : str, optional
        The label for the x-axis. Defaults to None.
    bins : int, optional
        The number of bins for the histogram. Defaults to 20.
    range : tuple, optional
        The range for the histogram. Defaults to None.
    """
    data = sarc_obj.structure.data[feature][timepoint]
    if len(data.shape) > 1:
        data = data.flatten()
    data = data[~np.isnan(data)]
    ax.hist(data, rwidth=0.8, color='k', alpha=0.7, density=True, bins=bins,
            range=range)
    ax.set_xlabel(label)
    ax.set_ylabel('Frequency')
    remove_spines(ax)


def analyze_noise_filter_plot(i, time, z_pos, z_pos_filt, slen, slen_filt, vel, residuals, std, tlim,
                              save_folder=None):
    """
    Plots the results of the signal noise analysis and smoothing

    Parameters
    ----------
    i : int
        The index to analyze.
    time : array-like
        The time data.
    z_pos : array-like
        The z position data.
    z_pos_filt : array-like
        The filtered z position data.
    slen : array-like
        The slen data.
    slen_filt : array-like
        The filtered slen data.
    vel : array-like
        The velocity data.
    residuals : array-like
        The residuals data.
    std : float
        The standard deviation.
    tlim : tuple
        The time limits for the plot.
    save_folder : str, optional
        The folder to save the plot to. If None, the plot is not saved. Defaults to None.
    """
    mosaic = """
    AA
    BB
    CC
    DE
    """
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(width_2cols, width_2cols * 0.8))
    axs['A'].plot(time, z_pos[i], c='k', lw=0.25, marker='x')
    axs['A'].plot(time, z_pos[i + 1], c='k', lw=0.25, marker='x')
    axs['A'].plot(time, z_pos_filt[i], c='r', lw=1)
    axs['A'].plot(time, z_pos_filt[i + 1], c='r', lw=1)
    axs['B'].plot(time, slen[i], c='k', lw=0.25, marker='x')
    axs['B'].plot(time, slen_filt[i], c='r', lw=1)
    axs['C'].plot(time, vel[i], c='r', lw=1)
    axs['C'].axhline(0, lw=1, c='k', linestyle='--')

    axs['A'].set_xlim(tlim)
    axs['B'].set_xlim(tlim)
    axs['C'].set_xlim(tlim)

    axs['A'].set_ylabel('Z-band position [µm]')
    axs['B'].set_ylabel('Sarc. length [µm]')
    axs['C'].set_ylabel('Sarc. vel. [µm/s]')
    axs['C'].set_xlabel('Time [s]')

    axs['D'].hist(residuals, bins=150, label=f'$\sigma$ = {np.round(std, 4)} nm')
    axs['D'].set_xlabel('Z-band pos. (raw-filt) [µm]')
    axs['D'].set_ylabel('Occurance [#]')
    axs['D'].legend()

    # q-q plot
    qq = sm.qqplot(residuals / std, line='45', ax=axs['E'])
    label_all_panels(axs)
    remove_all_spines(axs)

    plt.tight_layout()
    if save_folder is not None:
        fig.savefig(save_folder + f'noise_filter_{i}.pdf', dpi=dpi)


def plot_z_pos(ax, motion_obj, number_contr=None, show_contr=True, show_kymograph=False, color='k', t_lim=(None, None),
               y_lim=(None, None)):
    """
    Plots the z-band trajectories of the motion object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    motion_obj : object
        The motion object to plot.
    show_contractions : bool, optional
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
        for start_i, end_i in zip(motion_obj.loi_data['start_contr'][:-1],
                                  motion_obj.loi_data['start_quiet']):
            if number_contr is not None:
                start_i -= tlim[0]
                end_i -= tlim[0]
            ax.fill_betweenx([0, 1], [start_i, start_i], [end_i, end_i], color='lavender',
                             transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

    # plot trajectories
    if number_contr is not None and motion_obj.loi_data['n_contr'] > 0:
        ax.plot(time[:idxlim[1] - idxlim[0]], z_pos.T[idxlim[0]:idxlim[1]], linewidth=0.75, c=color)
        ax.set_xlim(0, tlim[1] - tlim[0])
    else:
        ax.plot(time, z_pos.T, linewidth=0.75, c=color)
        ax.set_xlim(t_lim)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Z-band position Z(t) [µm]')
    ax.set_ylim(y_lim)
    polish_yticks(ax, 5, 2.5)
    polish_xticks(ax, 2, 1)


def plot_delta_slen(ax, motion_obj, tlim=(0, 12), ylim=(-0.3, 0.4), n_rows=6, n_start=1, show_contr=True):
    """
    Plots the change in sarcomere length over time for a motion object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    motion_obj : object
        The motion object to plot.
    tlim : tuple, optional
        The time limits for the plot. Defaults to (0, 12).
    ylim : tuple, optional
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
        ax_i = ax.inset_axes([0., y, 1, 1 / n_rows - 0.02])
        ax_i.plot(motion_obj.loi_data['time'], delta_slen[i + n_start], c='k', lw=0.6)
        ax_i.axhline(0, linewidth=1, linestyle=':', c='k')
        if show_contr:
            for start_j, end_j in zip(motion_obj.loi_data['start_contr'][:-1],
                                      motion_obj.loi_data['start_quiet']):
                ax_i.fill_betweenx([-1, 1], [start_j, start_j], [end_j, end_j], color='lavender')
        if i > 0:
            ax_i.set_xticks([])
        ax_i.set_ylim(ylim)
        ax_i.set_xlim(tlim)
        ax_i.set_yticks(yticks)
        if show_contr:
            for start_i, end_i in zip(motion_obj.loi_data['start_contr'][:-1],
                                      motion_obj.loi_data['start_quiet']):
                ax.fill_betweenx([0, 1], [start_i, start_i], [end_i, end_i], color='lavender',
                                 transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))
        if i > 0:
            ax_i.set_xticks([])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('$\Delta$SL [µm]')
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.xaxis.label.set_color('k')
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')


def plot_overlay_delta_slen(ax, motion_obj, number_contr=None, t_lim=(0, 1), y_lim=(-0.35, 0.45), show_contr=True):
    """
    Plots the sarcomere length change over time for a motion object, overlaying multiple trajectories.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    motion_obj : object
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
        for start_i, end_i in zip(motion_obj.loi_data['start_contr'][:-1],
                                  motion_obj.loi_data['start_quiet']):
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
    polish_yticks(ax, 0.2, 0.1)
    polish_xticks(ax, 0.5, 0.25)


def plot_overlay_velocity(ax, motion_obj, number_contr=None, t_lim=(0, 0.9), y_lim=(-7, 10), show_contr=True):
    """
    Plots overlay of sarcomere velocity time series of the motion object

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    motion_obj : object
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
        for start_i, end_i in zip(motion_obj.loi_data['start_contr'][:-1],
                                  motion_obj.loi_data['start_quiet']):
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


def plot_phase_space(ax, motion_obj, t_lim=(0, 4), number_contr=None):
    """
    Plots sarcomere trajectory in length-change velocity phase space

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the plot on.
    motion_obj : object
        The motion object to plot.
    t_lim : tuple, optional
        The time limits for the plot. Defaults to (0, 4).
    number_contr : int, optional
        The number of contractions to overlay. If None, all contractions are overlaid. Defaults to None.
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
    for vel_i, delta_i in zip(vel, delta_slen):
        ax.plot(vel_i[idxlim[0]:idxlim[1]], delta_i[idxlim[0]:idxlim[1]], c='r', alpha=0.35, lw=0.5)

    ax.plot(vel_avg[idxlim[0]:idxlim[1]], delta_slen_avg[idxlim[0]:idxlim[1]], c='k', lw=2, label='Average')
    legend_elements = [Line2D([0], [0], color='k', lw=2), Line2D([0], [0], color='r', alpha=0.35, lw=0.5)]
    ax.legend(legend_elements, ['Average', 'Individual'], loc='upper right')
    polish_xticks(ax, 5, 2.5)
    polish_yticks(ax, 0.2, 0.1)
    ax.set_xlabel('Velocity $V$ [µm/s]', fontsize=fontsize)
    ax.set_ylabel('Length change $\Delta SL$ [µm]', fontsize=fontsize)


def plot_popping_events(motion_obj, save_name=None):
    """
    Create binary event map of popping events of the motion object.

    Parameters
    ----------
    motion_obj : object
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

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    fig_events = plt.figure(figsize=(width_1cols * 0.9, 3.))
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

    if save_name is None:
        fig_events.savefig(motion_obj.loi_folder + 'popping_events.pdf')
    else:
        fig_events.savefig(save_name)
