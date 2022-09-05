"""
Wrappers for drawing plots for reports.
"""

from matplotlib import pyplot as plt
from matplotlib import patheffects
from typing import List, Tuple, Optional, Dict
import numpy as np
import itertools
from pathlib import Path
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from math import floor


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
    skipfirst: bool
        True if the first entry should be removed from plotting.
    figsize: Tuple
        The size of the figure
    bins: int
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


def draw_violin_comparison_plot(
        outpath: Optional[Path],
        title: str,
        xnames: List[str],
        data: Dict[str, List]
):
    """
    Draws violin plots comparing different metrics.

    Parameters
    ----------
    outpath : Optional[Path]
        Path where the plot will be saved. If None, the plot will be displayed.
    title : str
        Title of the plot
    xnames : List[str]
        Names of the metrics in order
    data : Dict[str, List]
        Map between name of the model and list of metrics to visualize
    """

    num_plots = len(xnames)
    legend_lines, legend_labels = [], []
    fig, axs = plt.subplots(1, num_plots, figsize=(3*num_plots, 10))
    axs = axs.flatten()
    fig.suptitle(title, fontsize='x-large')
    cmap = plt.get_cmap("gnuplot")
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(data))]

    for i, (samplename, samples) in enumerate(data.items()):
        for ax, metric_sample in zip(axs, samples):
            vp = ax.violinplot(metric_sample, positions=[i])
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
            axis='x',
            which='both',
            bottom=False,
            labelbottom=False
        )

    fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        fontsize="large"
    )
    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath)


def draw_confusion_matrix(
        confusion_matrix: np.ndarray,
        outpath: Optional[Path],
        title: str,
        class_names: List[str],
        cmap=None,
        figsize: Optional[Tuple] = None,
        dpi: Optional[int] = None):
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
    """
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

    # compute overall accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # normalize confusion matrix
    confusion_matrix /= confusion_matrix.sum(axis=0)
    confusion_matrix = confusion_matrix.transpose()

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
    axConfMatrix.set_xlabel('Actual class', fontsize='x-large')
    axConfMatrix.set_ylabel('Predicted class', fontsize='x-large')
    img = axConfMatrix.imshow(
        confusion_matrix,
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
        axPredicted.set_xlabel('Sensitivity', fontsize='large')
        axPredicted.imshow(
            correctactual,
            interpolation='nearest',
            cmap='RdYlGn',
            aspect='auto',
            vmin=0.0,
            vmax=1.0
        )
        for i in range(len(class_names)):
            txt = axPredicted.text(
                i, 0,
                ('100' if correctactual[0, i] == 1.0
                    else f'{100.0 * correctactual[0, i]:3.1f}'),
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
        axActual.set_ylabel('Precision', fontsize='large')
        axActual.yaxis.set_label_position('right')
        axActual.imshow(
            correctpredicted,
            interpolation='nearest',
            cmap='RdYlGn',
            aspect='auto',
            vmin=0.0,
            vmax=1.0
        )
        for i in range(len(class_names)):
            txt = axActual.text(
                0, i,
                ('100' if correctpredicted[i, 0] == 1.0
                    else f'{100.0 * correctpredicted[i, 0]:3.1f}'),
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
        axTotal.set_xlabel('Accuracy', fontsize='large')
        axTotal.imshow(
            np.array([[accuracy]]),
            interpolation='nearest',
            cmap='RdYlGn',
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
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize('medium')
    suptitlehandle = fig.suptitle(
        f'{title} (ACC={accuracy:.5f})',
        fontsize='xx-large'
    )
    if outpath is None:
        plt.show()
    else:
        plt.savefig(
            outpath,
            dpi=dpi,
            bbox_inches='tight',
            bbox_extra_artists=[suptitlehandle],
            pad_inches=0.1
        )


def recall_precision_curves(
        outpath: Optional[Path],
        title: str,
        lines: List[Tuple[List, List]],
        class_names: List[str],
        figsize: Tuple = (15, 15)):
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
    figsize: Tuple
        The size of the figure
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
        fig.savefig(
            outpath,
            bbox_extra_artists=[legendhandle],
            bbox_inches='tight'
        )


def true_positive_iou_histogram(
        outpath: Optional[Path],
        title: str,
        lines: List[float],
        class_names: List[str],
        figsize: Tuple = (10, 25)):
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
    figsize: Tuple
        The size of the figure
    """
    plt.figure(figsize=figsize)
    plt.barh(
        class_names,
        np.array(lines),
        orientation='horizontal',
        color='purple'
    )
    plt.ylim((-1, len(class_names)))
    plt.yticks(np.arange(0, len(class_names)))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('IoU precision')
    plt.ylabel('classes')
    plt.title(f'{title}')
    plt.tight_layout()

    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath)


def true_positives_per_iou_range_histogram(
        outpath: Optional[Path],
        title: str,
        lines: List[float],
        range_fraction: float = 0.05,
        figsize: Tuple = (10, 10)):
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
    figsize: Tuple
        The size of the figure
    """
    lower_bound = floor(min(lines)*10)/10
    x_range = np.arange(lower_bound, 1.01, (1-lower_bound)*range_fraction)
    plt.figure(figsize=figsize)
    plt.hist(
        lines,
        x_range,
        color='purple'
    )
    plt.xlabel('IoU ranges')
    plt.xticks(x_range, rotation=45)
    plt.ylabel('Number of masks in IoU range')
    plt.title(f'{title}')
    plt.tight_layout()

    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath)


def recall_precision_gradients(
        outpath: Optional[Path],
        title: str,
        lines: List[Tuple[List, List]],
        class_names: List[str],
        aps: List[float],
        figsize: Tuple = (10, 25)):
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
    aps: List[float]
        Per-class AP values
    figsize: Tuple
        The size of the figure
    """
    plt.figure(figsize=figsize)
    clsticks = []
    for i, (cls, line, averageprecision) \
            in enumerate(zip(class_names, lines, aps)):
        clscoords = np.ones(len(line[0])) * i
        points = np.array([line[0], clscoords]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap='RdYlGn',
            norm=plt.Normalize(0, 1.0)
        )
        lc.set_array(line[1])
        lc.set_linewidth(10)
        plt.gca().add_collection(lc)
        clsticks.append(f'{cls} (AP={averageprecision})')
    plt.ylim((-1, len(class_names)))
    plt.yticks(np.arange(0, len(clsticks)), labels=clsticks)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('recall')
    plt.ylabel('classes')
    plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1.0), cmap='RdYlGn'),
        orientation='horizontal',
        label='precision',
        fraction=0.1,
        pad=0.05
    )
    plt.title(f'{title} (mAP={np.mean(aps)})')
    plt.tight_layout()

    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath)


def draw_plot(
        outpath: Optional[Path],
        title: str,
        xtitle: str,
        xunit: str,
        ytitle: str,
        yunit: str,
        line: Tuple[List, List],
        figsize: Tuple = (15, 15)):
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
    line : Tuple[List, List]
        Per-class list of tuples with list of recall values and precision
        values
    figsize: Tuple
        The size of the figure
    """
    plt.figure(figsize=figsize)
    plt.plot(line[0], line[1], c='purple', linewidth=3)
    xlabel = xtitle
    if xunit is not None:
        xlabel += f' [{xunit}]'
    ylabel = ytitle
    if yunit is not None:
        ylabel += f' [{yunit}]'
    plt.xlabel(xlabel, fontsize='large')
    plt.ylabel(ylabel, fontsize='large')
    plt.grid()
    plt.title(title)

    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath)
