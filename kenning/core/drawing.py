# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrappers for drawing plots for reports.
"""

import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
from typing import List, Tuple, Optional, Dict, Union, Iterable
import numpy as np
import itertools
from pathlib import Path
from matplotlib import gridspec
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from math import floor, pi
from scipy.signal import savgol_filter
from contextlib import contextmanager
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.resources import reports


BOKEH_THEME_FILE = path(reports, 'bokeh_theme.yml')
MATPLOTLIB_THEME_FILE = path(reports, 'matplotlib_theme_rc')

RED = '#d52a2a'
GREEN = '#1c7d4d'
# Creating colormap for confusion matrix
cmap_values = np.ones((256, 4))
for channel in range(3):
    pos = 1 + 2*channel
    cmap_values[:, channel] = np.linspace(
        int(RED[pos:pos + 2], 16),
        int(GREEN[pos:pos + 2], 16), 256)
cmap_values[:, :3] /= 255
RED_GREEN_CMAP = ListedColormap(cmap_values, name='red_green_colormap')

IMMATERIAL_COLORS = [
    "#ef5552",  # red
    # "#e92063",  # pink
    "#ab47bd",  # purple
    "#7e56c2",  # deep-purple
    "#4051b5",  # indigo
    "#2094f3",  # blue
    "#00bdd6",  # cyan
    "#009485",  # teal
    "#4cae4f",  # green
    "#cbdc38",  # lime
    # "#ffec3d",  # yellow
    "#ffa724",  # orange
    "#795649",  # brown
    "#546d78",  # deep-blue
]


def get_comparison_color_scheme(n_colors: int) -> List[Tuple]:
    """
    Creates default color schema to use for comparison plots (such as violin
    plot, bubble chart etc.)

    Parameters
    ----------
    n_colors : int
        Number of colors to return

    Returns
    -------
    List of colors to use for plotting
    """
    CMAP_NAME = "nipy_spectral"
    cmap = plt.get_cmap(CMAP_NAME)
    return [cmap(i) for i in np.linspace(0.0, 1.0, n_colors)]


def time_series_plot(
        outpath: Optional[Path],
        title: str,
        xtitle: str,
        xunit: str,
        ytitle: str,
        yunit: str,
        xdata: List,
        ydata: List,
        trimxvalues: bool = True,
        skipfirst: bool = False,
        figsize: Tuple = (15, 8.5),
        bins: int = 20):
    """
    Draws time series plot.

    Used i.e. for timeline of resource usage.

    It also draws the histogram of values that appeared throughout the
    experiment.

    Parameters
    ----------
    outpath : Optional[Path]
        Output path for the plot image. If None, the plot will be displayed.
    title : str
        Title of the plot
    xtitle : str
        Name of the X axis
    xuint : str
        Unit for the X axis
    ytitle : str
        Name of the Y axis
    yunit : str
        Unit for the Y axis
    xdata : List
        The values for X dimension
    ydata : List
        The values for Y dimension
    trimxvalues : bool
        True if all values for the X dimension should be subtracted by
        the minimal value on this dimension
    skipfirst : bool
        True if the first entry should be removed from plotting.
    figsize : Tuple
        The size of the figure
    bins : int
        Number of bins for value histograms
    """
    start = 1 if skipfirst else 0
    xdata = np.array(xdata[start:], copy=True)
    ydata = np.array(ydata[start:], copy=True)
    if trimxvalues:
        minx = min(xdata)
        xdata = [x - minx for x in xdata]
    fig, (axplot, axhist) = plt.subplots(
        ncols=2,
        tight_layout=True,
        figsize=figsize,
        sharey=True,
        gridspec_kw={'width_ratios': (8, 3)}
    )
    if title:
        fig.suptitle(title, fontsize='x-large')
    axplot.scatter(xdata, ydata, c='purple', alpha=0.5)
    xlabel = xtitle
    if xunit is not None:
        xlabel += f' [{xunit}]'
    ylabel = ytitle
    if yunit is not None:
        ylabel += f' [{yunit}]'
    axplot.set_xlabel(xlabel, fontsize='large')
    axplot.set_ylabel(ylabel, fontsize='large')
    axplot.grid()

    axhist.hist(ydata, bins=bins, orientation='horizontal', color='purple')
    axhist.set_xscale('log')
    axhist.set_xlabel('Value histogram', fontsize='large')
    axhist.grid(which='both')
    plt.setp(axhist.get_yticklabels(), visible=False)

    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath)
    plt.close()


def draw_multiple_time_series(
        outpath: Optional[Path],
        title: str,
        xdata: Dict[str, List],
        xtitle: str,
        ydata: Dict[str, List],
        ytitle: str,
        skipfirst: bool = False,
        smooth: Optional[int] = None,
        figsize: Tuple = (11, 8.5),
        colors: Optional[List] = None,
        outext: Iterable[str] = ['png'],
):
    """
    Draws multiple time series plots.

    Parameters
    ----------
    outpath : Optional[Path]
        Path where the plot will be saved. If None, the plot will be displayed.
    title : str
        Title of the plot
    xdata : Dict[str, List]
        Mapping between name of the model and x coordinates of samples.
    xtitle : str
        Name of the X axis
    ydata : Dict[str, List]
        Mapping between name of the model and y coordinates of samples.
    ytitle : str
        Name of the Y axis
    skipfirst : bool
        True if the first entry should be removed from plotting.
    smooth : Optional[int]
        If None, raw point coordinates are plotted in a line plot.
        If int, samples are plotted in a scatter plot in a background,
        and smoothing is performed with Savitzky–Golay filter to the line,
        where `smooth` is the window size parameter.
    figsize : Tuple
        The size of the figure
    colors : Optional[List]
        List with colors which should be used to draw plots
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    start = 1 if skipfirst else 0
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize="x-large")
    if colors is None:
        colors = get_comparison_color_scheme(len(xdata))
    for color, (samplename, sample) in zip(colors, ydata.items()):
        x_sample = xdata[samplename][start:]
        x_sample = np.array(x_sample)
        x_sample -= np.min(x_sample)
        y_sample = sample[start:]
        if smooth is None:
            ax.plot(x_sample, y_sample, label=samplename, color=color)
        else:
            ax.scatter(x_sample, y_sample, alpha=0.15, marker='.',
                       s=10, color=color)
            smoothed = savgol_filter(y_sample, smooth, 3)
            ax.plot(x_sample, smoothed, label=samplename,
                    linewidth=3, color=color)

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.legend()
    plt.grid()

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(f"{outpath}.{ext}")
    plt.close()


def draw_violin_comparison_plot(
        outpath: Optional[Path],
        title: str,
        xnames: List[str],
        data: Dict[str, List],
        colors: Optional[List] = None,
        outext: Iterable[str] = ['png'],
):
    """
    Draws violin plots comparing different metrics.

    Parameters
    ----------
    outpath : Optional[Path]
        Path where the plot will be saved without extension. If None,
        the plot will be displayed.
    title : str
        Title of the plot
    xnames : List[str]
        Names of the metrics in order
    data : Dict[str, List]
        Map between name of the model and list of metrics to visualize
    colors : Optional[List]
        List with colors which should be used to draw plots
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """

    num_plots = len(xnames)
    legend_lines, legend_labels = [], []
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 3.5*num_plots))
    axs = axs.flatten()
    bbox_extra = []
    if title:
        bbox_extra.append(fig.suptitle(title, fontsize='x-large'))
    if colors is None:
        colors = get_comparison_color_scheme(len(data))

    for i, (samplename, samples) in enumerate(data.items()):
        for ax, metric_sample in zip(axs, samples):
            vp = ax.violinplot(metric_sample, positions=[i], vert=False)
            for body in vp['bodies']:
                body.set_color(colors[i])
            vp['cbars'].set_color(colors[i])
            vp['cmins'].set_color(colors[i])
            vp['cmaxes'].set_color(colors[i])
        # dummy plot used to create a legend
        line, = plt.plot([], label=samplename, color=colors[i])
        legend_lines.append(line)
        legend_labels.append(samplename)

    for ax, metricname in zip(axs, xnames):
        ax.set_title(metricname)
        ax.tick_params(
            axis='y',
            which='both',
            left=False,
            labelleft=False
        )

    bbox_extra.append(fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        fontsize="large",
        ncol=2
    ))

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(
                f"{outpath}.{ext}",
                bbox_extra_artists=bbox_extra,
                bbox_inches='tight')
    plt.close()


def draw_radar_chart(
        outpath: Optional[Path],
        title: str,
        data: Dict[str, List],
        labelnames: List,
        figsize: Tuple = (11, 12),
        colors: Optional[List] = None,
        outext: Iterable[str] = ['png'],
):
    """
    Draws radar plot.

    Parameters
    ----------
    outpath : Optional[Path]
        Path where the plot will be saved. If None, the plot will be displayed.
    title : str
        Title of the plot
    data : Dict[str, List]
        Map between name of the model and list of metrics to visualize
    labelnames : List[str]
        Names of the labels in order
    figsize : Optional[Tuple]
        The size of the plot
    colors : Optional[List]
        List with colors which should be used to draw plots
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    n_categories = len(labelnames)

    angles = [n / n_categories * 2 * pi for n in range(n_categories)]
    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, labelnames)
    ax.set_rlabel_position(1 / (n_categories * 2) * 2 * pi)
    ax.set_yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"])
    ax.set_ylim((0, 1.0))
    bbox_extra = []
    if title:
        bbox_extra.append(fig.suptitle(title, fontsize='x-large'))
    if colors is None:
        colors = get_comparison_color_scheme(len(data))

    angles += [0]
    linestyles = ['-', '--', '-.', ':']
    for i, (color, (samplename, sample)) in enumerate(zip(colors, data.items())):  # noqa: E501
        sample += sample[:1]
        ax.plot(
            angles,
            sample,
            label=samplename,
            color=color,
            alpha=0.5,
            linestyle=linestyles[i % len(linestyles)]
        )
        ax.fill(
            angles,
            sample,
            alpha=0.1,
            color=color
        )
    bbox_extra.append(
        plt.legend(fontsize="large", bbox_to_anchor=[0.50, -0.05],
                   loc="upper center", ncol=2))

    angles = np.array(angles)
    angles[np.cos(angles) <= -1e-5] += pi
    angles = np.rad2deg(angles)
    for i in range(n_categories):
        label = ax.get_xticklabels()[i]
        labelname, angle = labelnames[i], angles[i]
        x, y = label.get_position()
        lab = ax.text(
            x, y, labelname,
            transform=label.get_transform(),
            ha=label.get_ha(),
            va=label.get_va()
        )
        lab.set_rotation(-angle)
        lab.set_fontsize('large')
        bbox_extra.append(lab)
    ax.set_xticklabels([])

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(
                f"{outpath}.{ext}",
                bbox_extra_artists=bbox_extra,
                bbox_inches='tight')
    plt.close()


def draw_bubble_plot(
        outpath: Optional[Path],
        title: str,
        xdata: List[float],
        xlabel: str,
        ydata: List[float],
        ylabel: str,
        bubblesize: List[float],
        bubblename: List[str],
        figsize: Tuple = (11, 10),
        colors: Optional[List] = None,
        outext: Iterable[str] = ['png'],
):
    """
    Draws bubble plot

    Parameters
    ----------
    outpath : Optional[Path]
        Path where the plot will be saved. If None, the plot will be displayed.
    title : str
        Title of the plot
    xdata : List[float]
        The values for X dimension
    xlabel : str
        Name of the X axis
    ydata : List[float]
        The values for Y dimension
    ylabel : str
        Name of the Y axis
    bubblesize : List[float]
        Sizes of subsequent bubbles
    bubblename : List[str]
        Labels for consecutive bubbles
    figsize : Tuple
        The size of the plot
    colors : Optional[List]
        List with colors which should be used to draw plots
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if colors is None:
        colors = get_comparison_color_scheme(len(xdata))
    markers = []
    bbox_extra = []
    maxsize = max(bubblesize)
    minsize = min(bubblesize)
    for x, y, bsize, label, c in zip(xdata, ydata, bubblesize,
                                     bubblename, colors):
        size = (bsize - minsize) / (maxsize - minsize + 1) * 100
        marker = ax.scatter(x, y, s=(15 + size**1.75), label=label, color=c,
                            alpha=0.5, edgecolors='black')
        markers.append(marker)
    legend = ax.legend(
        loc='upper center',
        handles=markers,
        bbox_to_anchor=[0.5, -0.08],
        ncol=2
    )
    for handler in legend.legendHandles:
        handler.set_sizes([40.0])
    ax.add_artist(legend)
    bbox_extra.append(legend)

    bubblemarkers, bubblelabels = [], []
    for i in [0, 25, 50, 75, 100]:
        bubblemarker = ax.scatter(
            [], [], s=(15 + i**1.75), color='None',
            edgecolors=mpl.rcParams['legend.labelcolor'])
        bubblemarkers.append(bubblemarker)
        bubblelabels.append(f"{(minsize + i / 100 * (maxsize - minsize)) / 1024 ** 2:.4f} MB")  # noqa: E501
    bubblelegend = ax.legend(
        bubblemarkers,
        bubblelabels,
        handletextpad=3,
        labelspacing=4.5,
        borderpad=3,
        title="Model size",
        frameon=False,
        bbox_to_anchor=[1.05, 0.5],
        loc='center left'
    )
    bubblelegend._legend_box.sep = 20
    ax.add_artist(bubblelegend)
    bbox_extra.append(bubblelegend)

    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + 0.05, box.width * 0.85, box.height - 0.05])

    if title:
        bbox_extra.append(fig.suptitle(title))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(
                f"{outpath}.{ext}",
                bbox_extra_artists=bbox_extra,
                bbox_inches='tight'
            )
    plt.close()


def _value_to_nondiagonal_color(
        value: Union[float, np.ndarray], cmap) -> np.ndarray:
    """
    Calculates colors of non-diagonal cells in confusion matrix.

    Parameters
    ----------
    value : float | np.ndarray
        Values from confusion matrix
    cmap :
        Color map assosiating values with colors

    Returns
    -------
    np.ndarray :
        Calcualted colors
    """
    color = np.asarray(cmap(1 - np.log2(99*value + 1) / np.log2(100)))
    color[..., 3] = np.log2(99*value + 1) / np.log2(100)
    return color


def draw_confusion_matrix(
        confusion_matrix: np.ndarray,
        outpath: Optional[Path],
        title: str,
        class_names: List[str],
        cmap=None,
        figsize: Optional[Tuple] = None,
        dpi: Optional[int] = None,
        backend: str = 'matplotlib',
        outext: Iterable[str] = ['png'],
):
    """
    Creates a confusion matrix plot.

    Parameters
    ----------
    confusion_matrix : ArrayLike
        Square numpy matrix containing the confusion matrix.
        0-th axis stands for ground truth, 1-st axis stands for predictions
    outpath : Optional[Path]
        Path where the plot will be saved. If None, the plot will be displayed.
    title : str
        Title of the plot
    class_names : List[str]
        List of the class names
    cmap : Any
        Color map for the plot
    figsize : Optional[Tuple]
        The size of the plot
    dpi : Optional[int]
        The dpi of the plot
    backend : str
        Which library should be used to generate plot - bokeh or matplotlib
    outext : Iterable[str]
        List with files extensions, should be supported by chosen backend
    """
    available_backends = ('matplotlib', 'bokeh')
    assert backend in available_backends, (
        f"Backend has to be one of: {' '.join(available_backends)}")
    if cmap is None:
        if len(class_names) < 50:
            cmap = plt.get_cmap('BuPu')
        else:
            cmap = plt.get_cmap('nipy_spectral_r')

    confusion_matrix = np.array(confusion_matrix, dtype=np.float32, copy=True)

    # compute sensitivity
    correctactual = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    correctactual = correctactual.reshape(1, len(class_names))

    # compute precision
    correctpredicted = \
        confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
    correctpredicted = correctpredicted.reshape(len(class_names), 1)
    # change nan to 0
    correctpredicted[np.isnan(correctpredicted)] = 0.0

    # compute overall accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # normalize confusion matrix
    confusion_matrix /= confusion_matrix.sum(axis=0)
    confusion_matrix = confusion_matrix.transpose()
    # change nan to 0
    confusion_matrix[np.isnan(confusion_matrix)] = 0.0

    # Calculate colors for confusion matrix
    colors = np.zeros(confusion_matrix.shape + (4,))
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            if i == j:
                colors[i, j] = cmap(confusion_matrix[i, j])
            else:
                colors[i, j] = _value_to_nondiagonal_color(
                    confusion_matrix[i, j], cmap
                )

    if backend == "bokeh" or "html" in outext:
        draw_confusion_matrix_bokeh(
            output_path=outpath,
            class_names=class_names,
            confusion_matrix=confusion_matrix,
            confusion_matrix_colors=colors,
            sensitivity=correctactual,
            precision=correctpredicted,
            accuracy=accuracy,
            width=figsize[0] if figsize else None,
            height=figsize[1] if figsize else None,
            title=title,
            cmap=cmap,
            formats=outext if backend == 'bokeh' else ['html'])
        outext = set(outext)
        outext.discard('html')
    draw_confusion_matrix_matplotlib(
        output_path=outpath,
        class_names=class_names,
        confusion_matrix=confusion_matrix,
        confusion_matrix_colors=colors,
        sensitivity=correctactual,
        precision=correctpredicted,
        accuracy=accuracy,
        figsize=figsize,
        dpi=dpi,
        title=title,
        cmap=cmap,
        outext=outext)


def draw_confusion_matrix_matplotlib(
    confusion_matrix: np.ndarray,
    confusion_matrix_colors: np.ndarray,
    sensitivity: np.ndarray,
    precision: np.ndarray,
    accuracy: np.ndarray,
    class_names: List[str],
    figsize: Optional[Tuple[int]] = None,
    dpi: Optional[int] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    cmap=None,
    outext: Iterable[str] = ['png'],
):
    """
    Function drawing interactive confusion matrix with bokeh backend.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Values of confusion matrix, from 0 to 1
    confusion_matrix_colors : np.ndarray
        Colors for calculated based on confusion matrix
    sensitivity : np.ndarray
        Ordered values with sensitivity
    precision : np.ndarray
        Ordered values with precision
    accuracy : np.ndarray | float
        Overall accuracy
    class_names : List[str]
        List with names of classes
    figsize : Optional[Tuple[int]]
        Tuple with width and height of figure
    output_path : str | None
        Path to the file, where plot will be saved to. If not specified,
        result won't be saved
    title : str | None
        Title of the plot
    cmap :
        Color map which will be used for drawing plot. If not specified,
        'RdYlGn' color map from matplotlib will be chosen
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """

    if figsize is None:
        figsize = [35, 35]

    if dpi is None:
        dpi = 216

    # create axes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    vectors = 1
    if len(class_names) >= 50:
        vectors = 0
    gs = gridspec.GridSpec(
        len(class_names) + vectors, len(class_names) + vectors
    )
    axConfMatrix = fig.add_subplot(gs[0:len(class_names), 0:len(class_names)])
    plots = [axConfMatrix]
    if len(class_names) < 50:
        axPredicted = fig.add_subplot(
            gs[len(class_names), 0:len(class_names)],
            sharex=axConfMatrix
        )
        axActual = fig.add_subplot(
            gs[0:len(class_names), len(class_names)],
            sharey=axConfMatrix
        )
        axTotal = fig.add_subplot(
            gs[len(class_names), len(class_names)],
            sharex=axActual,
            sharey=axPredicted
        )
        plots = [axPredicted, axConfMatrix, axActual, axTotal]
    # define ticks for classes
    ticks = np.arange(len(class_names))

    # configure and draw confusion matrix
    if len(class_names) < 50:
        axConfMatrix.set_xticks(ticks)
        axConfMatrix.set_xticklabels(
            class_names,
            fontsize='large',
            rotation=90
        )
        axConfMatrix.set_yticks(ticks)
        axConfMatrix.set_yticklabels(class_names, fontsize='large')
        axConfMatrix.xaxis.set_ticks_position('top')
    else:
        # plt.setp(axConfMatrix.get_yticklabels(), visible=False)
        # plt.setp(axConfMatrix.get_xticklabels(), visible=False)
        axConfMatrix.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False
        )
    axConfMatrix.xaxis.set_label_position('top')
    axConfMatrix.set_xlabel(
        'Actual class', fontsize='x-large', fontweight='bold')
    axConfMatrix.set_ylabel(
        'Predicted class', fontsize='x-large', fontweight='bold')
    img = axConfMatrix.imshow(
        confusion_matrix_colors,
        # norm=colors.PowerNorm(0.5),
        interpolation='nearest',
        cmap=cmap,
        aspect='auto',
        vmin=0.0,
        vmax=1.0
    )

    if len(class_names) < 50:
        # add percentages for confusion matrix
        for i, j in itertools.product(
                range(len(class_names)),
                range(len(class_names))):
            txt = axConfMatrix.text(
                j, i,
                ('100' if confusion_matrix[i, j] == 1.0
                    else f'{100.0 * confusion_matrix[i,j]:3.1f}'),
                ha='center',
                va='center',
                color='black',
                fontsize='medium')
            txt.set_path_effects([
                patheffects.withStroke(linewidth=5, foreground='w')
            ])

        # configure and draw sensitivity percentages
        axPredicted.set_xticks(ticks)
        axPredicted.set_yticks([0])
        axPredicted.set_xlabel(
            'Sensitivity', fontsize='large', fontweight='bold')
        axPredicted.imshow(
            sensitivity,
            interpolation='nearest',
            cmap='RdYlGn' if cmap is None else cmap,
            aspect='auto',
            vmin=0.0,
            vmax=1.0
        )
        for i in range(len(class_names)):
            txt = axPredicted.text(
                i, 0,
                ('100' if sensitivity[0, i] == 1.0
                    else f'{100.0 * sensitivity[0, i]:3.1f}'),
                ha='center',
                va='center',
                color='black',
                fontsize='medium')
            txt.set_path_effects([
                patheffects.withStroke(linewidth=5, foreground='w')
            ])

        # configure and draw precision percentages
        axActual.set_xticks([0])
        axActual.set_yticks(ticks)
        axActual.set_ylabel(
            'Precision', fontsize='large', fontweight='bold')
        axActual.yaxis.set_label_position('right')
        axActual.imshow(
            precision,
            interpolation='nearest',
            cmap='RdYlGn' if cmap is None else cmap,
            aspect='auto',
            vmin=0.0,
            vmax=1.0
        )
        for i in range(len(class_names)):
            txt = axActual.text(
                0, i,
                ('100' if precision[i, 0] == 1.0
                    else f'{100.0 * precision[i, 0]:3.1f}'),
                ha='center',
                va='center',
                color='black',
                fontsize='medium')
            txt.set_path_effects([
                patheffects.withStroke(linewidth=5, foreground='w')
            ])

        # configure and draw total accuracy
        axTotal.set_xticks([0])
        axTotal.set_yticks([0])
        axTotal.set_xlabel(
            'Accuracy', fontsize='large', fontweight='bold')
        axTotal.imshow(
            np.array([[accuracy]]),
            interpolation='nearest',
            cmap='RdYlGn' if cmap is None else cmap,
            aspect='auto',
            vmin=0.0,
            vmax=1.0
        )
        txt = axTotal.text(
            0, 0,
            f'{100 * accuracy:3.1f}',
            ha='center',
            va='center',
            color='black',
            fontsize='medium'
        )
        txt.set_path_effects([
            patheffects.withStroke(linewidth=5, foreground='w')
        ])

        # disable axes for other matrices than confusion matrix
        for a in (axPredicted, axActual, axTotal):
            plt.setp(a.get_yticklabels(), visible=False)
            plt.setp(a.get_xticklabels(), visible=False)

    # draw colorbar for confusion matrix
    cbar = fig.colorbar(
        img,
        ax=plots,
        shrink=0.5,
        ticks=np.linspace(0.0, 1.0, 11),
        pad=0.1
    )
    cbar.ax.set_yticks(np.linspace(0.0, 1.0, 11),
                       labels=list(range(0, 101, 10)))
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize('medium')
    suptitlehandle = None
    if title:
        suptitlehandle = fig.suptitle(
            f'{title} (ACC={accuracy:.5f})',
            fontsize='xx-large'
        )
    if output_path is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(
                f"{output_path}.{ext}",
                dpi=dpi,
                bbox_inches='tight',
                bbox_extra_artists=[
                    suptitlehandle] if suptitlehandle else None,
                pad_inches=0.1,
            )
    plt.close()


def _create_custom_hover_template(
        names: List[str],
        values: List[str] = None,
        units: List[str] = None
) -> str:
    """
    Function creating custom template for tooltip displaying when hover
    event occurs. This tooltip is part of bokeh features

    Parameters
    ----------
    names : List[str]
        List with names, displayed before values
    values : List[str]
        List with names of fields (in source object) containing values
    units : List[str]
        List with units, displayed afrer values

    Returns
    -------
    str :
        HTML template for tooltip
    """
    if values is None:
        values = names
    if units is None:
        units = ['' for _ in names]
    else:
        units = ['' if unit is None else unit for unit in units]
    template = """
    <tr class="bk-tooltip-entry">
        <td class="bk-tt-entry-name">%s</td>
        <td class="bk-tt-entry-value">@{%s}%s</td>
    </tr>
    """
    result = "<table>"
    for name, value, unit in zip(names, values, units):
        result += template % (name, value, unit)
    result += "</table>"
    return result


def draw_confusion_matrix_bokeh(
    confusion_matrix: np.ndarray,
    confusion_matrix_colors: np.ndarray,
    sensitivity: np.ndarray,
    precision: np.ndarray,
    accuracy: np.ndarray,
    class_names: List[str],
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    cmap=None,
    formats: Tuple[str] = ('html',),
):
    """
    Function drawing interactive confusion matrix with bokeh backend.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Values of confusion matrix, from 0 to 1
    confusion_matrix_colors : np.ndarray
        Colors for calculated based on confusion matrix
    sensitivity : np.ndarray
        Ordered values with sensitivity
    precision : np.ndarray
        Ordered values with precision
    accuracy : np.ndarray | float
        Overall accuracy
    class_names : List[str]
        List with names of classes
    width : int
        Width of the generated plot
    height : int
        Height of the generated plot
    output_path : str | None
        Path to the file, where plot will be saved to. If not specified,
        result won't be saved
    title : str | None
        Title of the plot
    cmap :
        Color map which will be used for drawing plot. If not specified,
        'RdYlGn' color map from matplotlib will be chosen
    formats : Tuple[str]
        Tuple with formats names
    """
    from bokeh.plotting import figure, output_file, save, row, show
    from bokeh.models import ColumnDataSource, HoverTool, FactorRange, Range1d
    from bokeh.layouts import Spacer, gridplot
    from bokeh.io import export_png, export_svg

    if cmap is None:
        cmap = plt.get_cmap('RdYlGn')

    if width is None:
        width = 900
    if height is None:
        height = 778

    # === Confusion Matrix ===

    # Calculate confusion matrix sizes
    cm_width = int(width / (1 + 1/15 + 1/13 + 1/11))
    cm_height = int(height / (1 + 1/15))

    # Prepare figure
    confusion_matrix_fig = figure(
        title=None, x_range=FactorRange(
            factors=class_names, bounds=(0, len(class_names))),
        y_range=FactorRange(
            factors=class_names[::-1], bounds=(0, len(class_names))),
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location=None,
        x_axis_location="above",
        width=cm_width,
        height=cm_height,
        output_backend='webgl',
    )

    # Preprocess data
    confusion_matrix_colors = np.rot90(
        confusion_matrix_colors, k=-1).reshape((-1, 4))
    coords = np.array(list(itertools.product(
        class_names, class_names)), dtype=str)
    coords[:, 1] = coords[::-1, 1]
    percentage = np.rot90(confusion_matrix, k=-1).flatten() * 100
    source = ColumnDataSource(data={
        'Actual class': coords[:, 0],
        'Predicted class': coords[:, 1],
        'color': confusion_matrix_colors,
        'Percentage': percentage,
    })

    # Draw confusion matrix
    confusion_matrix_fig.rect(
        x='Actual class', y='Predicted class',
        color='color',
        line_color=None,
        width=1, height=1,
        source=source,)

    # Set labels and styles
    confusion_matrix_fig.xaxis.axis_label = "Actual class"
    confusion_matrix_fig.yaxis.axis_label = "Predicted class"
    if len(class_names) < 50:
        confusion_matrix_fig.xaxis.major_label_orientation = 'vertical'
    else:
        confusion_matrix_fig.xaxis.major_label_text_alpha = 0.0
        confusion_matrix_fig.yaxis.major_label_text_alpha = 0.0
        confusion_matrix_fig.xaxis.major_tick_line_alpha = 0.0
        confusion_matrix_fig.yaxis.major_tick_line_alpha = 0.0
    confusion_matrix_fig.xaxis.axis_line_alpha = 0.0
    confusion_matrix_fig.yaxis.axis_line_alpha = 0.0
    confusion_matrix_fig.grid.visible = False

    # Set custom tooltips
    confusion_matrix_fig.add_tools(HoverTool(
        tooltips=_create_custom_hover_template(
            ["Actual class", "Predicted class", "Percentage"],
            units=[None, None, '%']
        )
    ))

    # === Sensitivity ===

    # Prepare figure
    sensitivity_fig = figure(
        title=None,
        x_range=confusion_matrix_fig.x_range,
        y_range=FactorRange(factors=['Sensivity'], bounds=(0, 1)),
        width=confusion_matrix_fig.width,
        height=confusion_matrix_fig.height // 15,
        toolbar_location=None,
        output_backend='webgl',
    )

    # Preprocess data
    cc = cmap(sensitivity).reshape((-1, 4))
    sensitivity_source = ColumnDataSource(data={
        'y': ['Sensivity' for _ in class_names],
        'Class': class_names,
        'color': cc,
        "Sensitivity": sensitivity.flatten() * 100,
    })

    # Draw sensitivity
    sensitivity_fig.rect(
        x='Class', y='y', color='color',
        source=sensitivity_source,
        line_color='black',
        line_width=0.1,
        width=1,
        height=1,
    )

    # Add label and custom tooltip
    sensitivity_fig.xaxis.axis_label = "Sensitivity"
    sensitivity_fig.add_tools(HoverTool(
        tooltips=_create_custom_hover_template(
            ["Class", "Sensitivity"],
            units=[None, '%']
        ),
        attachment='above',
    ))

    # === Precision ===

    # Prepare figure
    precision_fig = figure(
        title=None,
        x_range=FactorRange(factors=['Precision'], bounds=(0, 1)),
        y_range=confusion_matrix_fig.y_range,
        width=confusion_matrix_fig.width // 15,
        height=confusion_matrix_fig.height,
        toolbar_location=None,
        y_axis_location='right',
        output_backend='webgl',
    )

    # Preprocess data
    cc2 = cmap(precision).reshape((-1, 4))
    precision_source = ColumnDataSource(data={
        'x': ['Precision' for _ in class_names],
        'Class': class_names,
        'color': cc2,
        'Precision': precision.flatten() * 100,
    })

    # Draw sensitivity
    precision_fig.rect(
        x='x',
        y='Class',
        color='color',
        source=precision_source,
        height=1,
        width=1,
        line_color='black',
        line_width=0.1,
    )

    # Add label and custom tooltip
    precision_fig.yaxis.axis_label = "Precision"
    precision_fig.add_tools(HoverTool(
        tooltips=_create_custom_hover_template(
            ["Class", "Precision"],
            units=[None, '%']
        ),
        attachment='left',
    ))

    # === Accuracy ===

    # Prepare figure
    accuracy_fig = figure(
        title=None,
        x_range=FactorRange(factors=['x'], bounds=(0, 1)),
        y_range=FactorRange(factors=['y'], bounds=(0, 1)),
        width=precision_fig.width,
        height=sensitivity_fig.height,
        toolbar_location=None,
        output_backend='webgl',
    )

    # Preprocess data
    c = cmap(accuracy)
    color_str = (f"#{int(255 * c[0]):02X}{int(255 * c[1]):02X}"
                 f"{int(255 * c[2]):02X}")
    accuracy_source = ColumnDataSource(data={
        'x': ['x'], 'y': ['y'],
        'Accuracy': [float(accuracy) * 100],
    })

    # Draw sensitivity
    accuracy_fig.rect(
        x='x',
        y='y',
        color=color_str,
        source=accuracy_source,
        width=1,
        height=1,
        line_color='black',
        line_width=0.1,
    )

    # Add label and custom tooltip
    accuracy_fig.xaxis.axis_label = "ACC"
    accuracy_fig.add_tools(HoverTool(
        tooltips=_create_custom_hover_template(
            ['Accuracy'], units=['%']
        ),
        attachment='above',
    ))

    # Set style for Sensitivity, Precision and Accuracy
    for fig in (sensitivity_fig, precision_fig, accuracy_fig):
        fig.yaxis.major_label_text_alpha = 0.0
        fig.xaxis.major_label_text_alpha = 0.0
        fig.yaxis.major_tick_line_alpha = 0.0
        fig.xaxis.major_tick_line_alpha = 0.0
        fig.xaxis.axis_line_alpha = 0.0
        fig.yaxis.axis_line_alpha = 0.0
        fig.grid.visible = False

    # === Scale ===

    # Prepare figure
    scale_fig = figure(
        title=None,
        x_range=['color'],
        y_range=Range1d(0.0, 100.0),
        width=confusion_matrix_fig.width // 11,
        height=height // 2,
        tools="",
        toolbar_location=None,
        x_axis_location='above',
        y_axis_location='right',
        margin=(height // 4, 0, height // 4, 0),
        output_backend='webgl',
    )
    # Draw scale
    scale_fig.hbar(
        y=np.linspace(0.0, 100.0, 256),
        left=[0.0] * 256,
        right=[1.0] * 256,
        color=cmap(np.linspace(0.0, 1.0, 256))
    )

    # Set styles for scale
    scale_fig.xaxis.major_tick_line_alpha = 0.0
    scale_fig.yaxis.major_tick_line_alpha = 0.0
    scale_fig.yaxis.minor_tick_line_alpha = 0.0
    scale_fig.xaxis.axis_line_alpha = 0.0
    scale_fig.yaxis.axis_line_alpha = 0.0
    scale_fig.xaxis.major_label_text_alpha = 0.0

    # === Saving to file ===

    grid_fig = gridplot(
        [
            [confusion_matrix_fig, precision_fig,],
            [sensitivity_fig, accuracy_fig,]
        ],
        merge_tools=True,
        toolbar_location='above',
        toolbar_options={'logo': None},
    )
    plot_with_scale = row(
        grid_fig,
        Spacer(width=confusion_matrix_fig.width // 13),
        scale_fig,
    )
    if output_path is None:
        show(plot_with_scale)
        return

    if 'html' in formats:
        output_file(f"{output_path}.html", mode='inline')
        save(plot_with_scale)

    grid_fig = gridplot(
        [
            [confusion_matrix_fig, precision_fig,],
            [sensitivity_fig, accuracy_fig,]
        ]
    )
    plot_with_scale = row(
        grid_fig,
        Spacer(width=confusion_matrix_fig.width // 13),
        scale_fig,
    )
    if 'png' in formats:
        export_png(plot_with_scale, f"{output_path}.png")
    if 'svg' in formats:
        export_svg(plot_with_scale, f"{output_path}.svg")


def recall_precision_curves(
        outpath: Optional[Path],
        title: str,
        lines: List[Tuple[List, List]],
        class_names: List[str],
        figsize: Tuple = (15, 15),
        outext: Iterable[str] = ['png'],
):
    """
    Draws Recall-Precision curves for AP measurements.

    Parameters
    ----------
    outpath : Optional[Path]
        Output path for the plot image. If None, the plot will be displayed.
    title : str
        Title of the plot
    lines : List[List[List]]
        Per-class list of tuples with list of recall values and precision
        values
    class_names : List[str]
        List of the class names
    figsize : Tuple
        The size of the figure
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 1, len(class_names))]
    linestyles = ['-', '--', '-.', ':']
    for i, (cls, line) in enumerate(zip(class_names, lines)):
        ax.plot(
            line[0], line[1],
            label=cls, c=colors[i], linewidth=3,
            linestyle=linestyles[i % len(linestyles)],
            alpha=0.8
        )
    legendhandle = ax.legend(
        bbox_to_anchor=(0.5, -0.3),
        loc='lower center',
        ncol=10
    )
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_xlim((0.0, 1.01))
    ax.set_ylim((0.0, 1.01))
    ax.grid('on')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_title(title)

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            fig.savefig(
                f"{outpath}.{ext}",
                bbox_extra_artists=[legendhandle],
                bbox_inches='tight'
            )
    plt.close()


def true_positive_iou_histogram(
        outpath: Optional[Path],
        title: str,
        lines: List[float],
        class_names: List[str],
        figsize: Tuple = (10, 25),
        colors: Optional[List] = None,
        color_offset: int = 0,
        outext: Iterable[str] = ['png'],
):
    """
    Draws per-class True Positive IoU precision plot

    Parameters
    ----------
    outpath : Optional[Path]
        Output path for the plot image. If None, the plot will be displayed.
    title : str
        Title of the plot
    lines : List[float]
        Per-class list of floats with IoU values
    class_names : List[str]
        List of the class names
    figsize : Tuple
        The size of the figure
    colors : Optional[List]
        List with colors which should be used to draw plots
    color_offset : int
        How many colors from default color list should be skipped
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    if colors is None:
        color = 'purple'
    else:
        color = colors[color_offset]
    plt.figure(figsize=figsize)
    plt.barh(
        class_names,
        np.array(lines),
        orientation='horizontal',
        color=color
    )
    plt.ylim((-1, len(class_names)))
    plt.yticks(np.arange(0, len(class_names)))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('IoU precision')
    plt.ylabel('classes')
    if title:
        plt.title(f'{title}')
    plt.tight_layout()

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(f"{outpath}.{ext}")
    plt.close()


def true_positives_per_iou_range_histogram(
        outpath: Optional[Path],
        title: str,
        lines: List[float],
        range_fraction: float = 0.05,
        figsize: Tuple = (10, 10),
        colors: Optional[List] = None,
        color_offset: int = 0,
        outext: Iterable[str] = ['png'],
):
    """
    Draws histogram of True Positive IoU values

    Parameters
    ----------
    outpath : Optional[Path]
        Output path for the plot image. If None, the plot will be displayed.
    title : str
        Title of the plot
    lines : List[float]
        All True Positive IoU values
    range_fraction : float
        Fraction by which the range should be divided (1/number_of_segments)
    figsize : Tuple
        The size of the figure
    colors : Optional[List]
        List with colors which should be used to draw plots
    color_offset : int
        How many colors from default color list should be skipped
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    if colors is None:
        color = 'purple'
    else:
        color = colors[color_offset]
    lower_bound = floor(10*min(lines)) / 10
    x_range = np.arange(lower_bound, 1.01, (1 - lower_bound) * range_fraction)
    plt.figure(figsize=figsize)
    plt.hist(
        lines,
        x_range,
        color=color
    )
    plt.xlabel('IoU ranges')
    plt.xticks(x_range, rotation=45)
    plt.ylabel('Number of masks in IoU range')
    if title:
        plt.title(f'{title}')
    plt.tight_layout()

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(f"{outpath}.{ext}")
    plt.close()


def recall_precision_gradients(
        outpath: Optional[Path],
        title: str,
        lines: List[Tuple[List, List]],
        class_names: List[str],
        aps: List[float],
        map: float,
        figsize: Tuple = (10, 25),
        cmap=None,
        outext: Iterable[str] = ['png'],
):
    """
    Draws per-class gradients of precision dependent to recall.

    Provides per-class AP and mAP values.

    Parameters
    ----------
    outpath : Optional[Path]
        Output path for the plot image. If None, the plot will be displayed.
    title : str
        Title of the plot
    lines : List[Tuple[List, List]]
        Per-class list of tuples with list of recall values and precision
        values
    class_names : List[str]
        List of the class names
    aps : List[float]
        Per-class AP values
    figsize : Tuple
        The size of the figure
    cmap : Any
        Color map for the plot
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    if cmap is None:
        cmap = plt.get_cmap('RdYlGn')
    plt.figure(figsize=figsize)
    clsticks = []
    for i, (cls, line, averageprecision) \
            in enumerate(zip(class_names, lines, aps)):
        clscoords = np.ones(len(line[0])) * i
        points = np.array([line[0], clscoords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(0, 1.0)
        )
        lc.set_array(line[1])
        lc.set_linewidth(10)
        plt.gca().add_collection(lc)
        clsticks.append(f'{cls} (AP={averageprecision:.4f})')
    plt.ylim((-1, len(class_names)))
    plt.yticks(np.arange(0, len(clsticks)), labels=clsticks)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('recall')
    plt.ylabel('classes')
    plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1.0), cmap=cmap),
        orientation='horizontal',
        label='precision',
        fraction=0.1,
        pad=0.05
    )
    if title:
        plt.title(f'{title} (mAP={map})')
    plt.tight_layout()

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(f"{outpath}.{ext}")
    plt.close()


def draw_plot(
        outpath: Optional[Path],
        title: str,
        xtitle: str,
        xunit: str,
        ytitle: str,
        yunit: str,
        lines: List[Tuple[List, List]],
        linelabels: Optional[List[str]] = None,
        figsize: Tuple = (15, 15),
        colors: Optional[List] = None,
        color_offset: int = 0,
        outext: Iterable[str] = ['png'],
):
    """
    Draws plot.

    Parameters
    ----------
    outpath : Optional[Path]
        Output path for the plot image. If None, the plot will be displayed.
    title : str
        Title of the plot
    xtitle : str
        Name of the X axis
    xuint : str
        Unit for the X axis
    ytitle : str
        Name of the Y axis
    yunit : str
        Unit for the Y axis
    lines : List[Tuple[List, List]]
        Per-class list of tuples with list of recall values and precision
        values
    linelabels : Optional[List[str]]
        Optional list of labels naming each line
    figsize : Tuple
        The size of the figure
    colors : Optional[List]
        List with colors which should be used to draw plots
    color_offset : int
        How many colors from default color list should be skipped
    outext : Iterable[str]
        List with files extensions, should be supported by matplotlib
    """
    plt.figure(figsize=figsize)

    bbox_extra = []
    if colors is None:
        color = 'purple'
    else:
        color = colors[color_offset]
    for color, line in zip(colors, lines):
        plt.plot(line[0], line[1], c=color, linewidth=3)
    xlabel = xtitle
    if xunit is not None:
        xlabel += f' [{xunit}]'
    ylabel = ytitle
    if yunit is not None:
        ylabel += f' [{yunit}]'
    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel(ylabel, fontsize='large')
    plt.grid()
    if title:
        bbox_extra.append(plt.title(title))
    if linelabels is not None:
        bbox_extra.append(
            plt.legend(
                linelabels,
                loc='upper center',
                bbox_to_anchor=[0.5, -0.06],
                ncols=2)
        )

    if outpath is None:
        plt.show()
    else:
        for ext in outext:
            plt.savefig(
                f"{outpath}.{ext}",
                bbox_extra_artists=bbox_extra,
                bbox_inches='tight')
    plt.close()


@contextmanager
def choose_theme(
        custom_bokeh_theme: Union[bool, str, Path] = False,
        custom_matplotlib_theme: Union[bool, str, Path] = False,
):
    """
    Context manager, allowing to temporaly set theme

    Parameter
    ---------
    custom_bokeh_theme : bool | str | Path
        If True uses BOKEH_THEME_FILE, if str or Path uses file specified
        by this path
    custom_matplotlib_theme : bool | str | Path
        If True uses MATPLOTLIB_THEME_FILE, if str or Path uses file specified
        by this path
    """
    # Backup current setups
    if custom_bokeh_theme:
        from bokeh.io import curdoc
        from bokeh.themes import Theme
        _copy_bokeh_theme = curdoc().theme
        # Set theme for bokeh
        if isinstance(custom_bokeh_theme, bool):
            with BOKEH_THEME_FILE as bokeh_theme_file:
                filename = bokeh_theme_file
        else:
            filename = custom_bokeh_theme
        theme = Theme(filename=filename)
        curdoc().theme = theme

    # Create temporary context for matplotlib
    with mpl.rc_context():
        # Set matplotlib theme
        if custom_matplotlib_theme:
            if isinstance(custom_matplotlib_theme, bool):
                with MATPLOTLIB_THEME_FILE as matplotlib_theme_file:
                    filename = matplotlib_theme_file
            else:
                filename = custom_matplotlib_theme
            plt.style.use(filename)

        yield
    # Cleanup
    if custom_bokeh_theme:
        curdoc().theme = _copy_bokeh_theme
