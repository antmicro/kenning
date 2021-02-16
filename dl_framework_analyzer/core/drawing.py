"""
Wrappers for drawing plots for reports.
"""

from matplotlib import pyplot as plt
from typing import List


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
    plt.figure(tight_layout=True)
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
