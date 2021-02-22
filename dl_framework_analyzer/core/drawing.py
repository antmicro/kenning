"""
Wrappers for drawing plots for reports.
"""

from matplotlib import pyplot as plt
from matplotlib import patheffects
from typing import List
import numpy as np
import itertools


def create_line_plot(
        outpath: str,
        title: str,
        xtitle: str,
        xunit: str,
        ytitle: str,
        yunit: str,
        xdata: List,
        ydata: List,
        trimxvalues: bool = True):
    """
    Draws single line plot.

    Used i.e. for timeline of resource usage.

    Parameters
    ----------
    outpath : str
        Output path for the plot image
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
    """
    plt.figure(tight_layout=True, figsize=(20, 20))
    plt.title(title)
    if trimxvalues:
        minx = min(xdata)
        xdata = [x - minx for x in xdata]
    plt.plot(xdata, ydata, 'b-', alpha=0.6)
    plt.plot(xdata, ydata, 'bo')
    plt.xlabel(f'{xtitle} [{xunit}]')
    plt.ylabel(f'{ytitle} [{yunit}]')
    plt.grid()

    plt.savefig(outpath)


def draw_confusion_matrix(
        confusion_matrix,
        outpath: str,
        title: str,
        class_names: List[str],
        normalized: bool = True,
        add_summary_columns: bool = True,
        cmap=None):
    """
    Creates a confusion matrix plot.
    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    plt.figure(tight_layout=True, figsize=(20, 20))
    plt.title(f'{title} (ACC={accuracy})')

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)
    plt.xlabel('Actual class')
    plt.ylabel('Predicted class')

    if normalized:
        confusion_matrix /= confusion_matrix.sum(axis=0)

    plt.imshow(
        confusion_matrix,
        interpolation='nearest',
        cmap=cmap)

    for i, j in itertools.product(
            range(len(class_names)),
            range(len(class_names))):
        if normalized:
            txt = plt.text(
                i, j,
                f'{confusion_matrix[i,j]:0.3f}',
                horizontalalignment='center',
                color='black',
                fontsize=9)
            txt.set_path_effects([
                patheffects.withStroke(linewidth=3, foreground='w')
            ])
        else:
            txt = plt.text(
                i, j,
                f'{confusion_matrix[i,j]}',
                horizontalalignment='center',
                color='black',
                fontsize=9)
            txt.set_path_effects([
                patheffects.withStroke(linewidth=3, foreground='w')
            ])

    plt.tight_layout()
    plt.savefig(outpath)
