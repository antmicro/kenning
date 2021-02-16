"""
Wrappers for drawing plots for reports.
"""

from matplotlib import pyplot as plt

def create_line_plot(
        outpath,
        title,
        xtitle,
        xunit,
        ytitle,
        yunit,
        xdata,
        ydata,
        trimxvalues=True):
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
