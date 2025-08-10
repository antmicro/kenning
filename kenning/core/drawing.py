# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrappers for drawing plots for reports.
"""

import itertools
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import matplotlib as mpl
import numpy as np
from matplotlib import gridspec, patheffects
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike

from kenning.resources import reports
from kenning.scenarios.manage_cache import format_size
from kenning.utils.logger import KLogger

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


BOKEH_THEME_FILE = path(reports, "bokeh_theme.yml")
BOKEH_PLOT_WIDTH = 90  # in viewport width (vw)
MATPLOTLIB_THEME_FILE = path(reports, "matplotlib_theme_rc")
MATPLOTLIB_DPI = 120
DEFAULT_PLOT_SIZE = 1000
MATPLOTLIB_FONT_SIZE = 12
SERVIS_PLOT_OPTIONS = {
    "figsize": (DEFAULT_PLOT_SIZE, DEFAULT_PLOT_SIZE * 2 // 3),
    "plottype": "scatter",
    "backend": "matplotlib",
}


plt.rc("font", size=MATPLOTLIB_FONT_SIZE)  # default text sizes
plt.rc("axes", titlesize=MATPLOTLIB_FONT_SIZE)  # axes title
plt.rc("axes", labelsize=1.5 * MATPLOTLIB_FONT_SIZE)  # x and y labels
plt.rc("xtick", labelsize=MATPLOTLIB_FONT_SIZE)  # x tick labels
plt.rc("ytick", labelsize=MATPLOTLIB_FONT_SIZE)  # y tick labels
plt.rc("legend", fontsize=MATPLOTLIB_FONT_SIZE)  # legend
plt.rc("figure", titlesize=2 * MATPLOTLIB_FONT_SIZE)  # figure title

RED = "#d52a2a"
GREEN = "#1c7d4d"
# Creating colormap for confusion matrix
cmap_values = np.ones((256, 4))
for channel in range(3):
    pos = 1 + 2 * channel
    cmap_values[:, channel] = np.linspace(
        int(RED[pos : pos + 2], 16), int(GREEN[pos : pos + 2], 16), 256
    )
cmap_values[:, :3] /= 255
RED_GREEN_CMAP = ListedColormap(
    cmap_values.tolist(), name="red_green_colormap"
)

KENNING_COLORS = [
    "#00E58D",  # gree500
    "#0093E5",  # blue500
    "#E56000",  # orange500
    "#007F8C",  # teal500
    "#159500",  # lime500
    "#DE1135",  # red500
    "#9E1FDA",  # purple500
    "#E59700",  # yellow500
]


@dataclass
class Plot(ABC, object):
    """
    Generic plot class.

    Parameters
    ----------
    width : int
        Width of the plot.
    height : int
        Height of the plot.
    title : Optional[str]
        Title of the plot.
    colors : Optional[List]
        List with colors which should be used to draw plots.
    color_offset : int
        How many colors from default color list should be skipped.
    cmap : Optional[Any]
        Color map for the plot.
    """

    width: int
    height: int
    title: Optional[str] = None
    colors: Optional[List] = None
    cmap: Optional[Any] = None

    def plot(
        self,
        output_path: Optional[Path] = None,
        output_formats: Iterable[str] = ("png",),
        backend: str = "matplotlib",
    ):
        """
        Display or export plot.

        Parameters
        ----------
        output_path : Optional[Path]
            Path where the plot will be saved without extension. If None,
            the plot will be displayed.
        output_formats : Iterable[str]
            Iterable with files extensions, should be supported by given
            framework.
        backend : str
            Which library should be used to generate plot - bokeh or
            matplotlib.
        """
        if backend == "bokeh" or "html" in output_formats:
            try:
                self.plot_bokeh(
                    output_path,
                    output_formats if backend == "bokeh" else ("html",),
                )
            except ImportError as e:
                KLogger.error(f"bokeh backend is not available: {e}")
            except NotImplementedError:
                KLogger.error(
                    "bokeh rendering not implemented for this plot type"
                )
            output_formats = set(output_formats)
            output_formats.discard("html")

        self.plot_matplotlib(output_path, output_formats)

    def _plt_figsize(
        self, ratio: Tuple[float, float] = (1, 1)
    ) -> Tuple[float, float]:
        return (
            ratio[0] * self.width / MATPLOTLIB_DPI,
            ratio[1] * self.height / MATPLOTLIB_DPI,
        )

    @abstractmethod
    def plot_bokeh(
        self,
        output_path: Optional[Path],
        output_formats: Iterable[str],
    ):
        """
        Create plot using bokeh backend.

        Parameters
        ----------
        output_path : Optional[Path]
            Path where the plot will be saved without extension. If None,
            the plot will be displayed.
        output_formats : Iterable[str]
            Iterable with files extensions, should be supported by bokeh.
        """
        ...

    @abstractmethod
    def plot_matplotlib(
        self,
        output_path: Optional[Path],
        output_formats: Iterable[str],
    ):
        """
        Create plot using matplotlib backend.

        Parameters
        ----------
        output_path : Optional[Path]
            Path where the plot will be saved without extension. If None,
            the plot will be displayed.
        output_formats : Iterable[str]
            List Iterable files extensions, should be supported by matplotlib.
        """
        ...

    def _create_bokeh_legend_fig(
        self,
        legend_items: List,
        safety_offset: int = 0,
        click_policy: str = "none",
        reverse: bool = False,
        max_height: str = "15vh",
    ) -> Any:
        """
        Creates bokeh figure with legend.

        Parameters
        ----------
        legend_items : List
            List with LegendItems.
        safety_offset : int
            Offset to be added to the figure.
        click_policy : str
            Click policy of legend items.
        reverse : bool
            Whether legend items order should be reversed.
        max_height : str
            CSS value used as max-height.

        Returns
        -------
        Any
            Bokeh figure with legend.
        """
        from bokeh.models import Legend, Range1d
        from bokeh.plotting import figure

        # Line width + margin + padding + label width
        legend_length = [
            20 + 20 + 10 + 6 * len(x.label.value) for x in legend_items
        ]

        # Iterate over length of labels to find the number of columns
        # that would fit under the plot
        legend_columns = len(legend_length)
        for i in range(len(legend_length) - 1):
            for j in range(i + 1, len(legend_length)):
                if sum(legend_length[i:j]) > self.width:
                    if legend_columns > j - i - 1:
                        legend_columns = j - i - 1
                    break
        legend_columns = max(1, legend_columns)

        # Creating fake figure for legend
        legend_fig = figure(
            min_border_left=0,
            frame_height=100 * len(legend_items) // legend_columns,
            toolbar_location=None,
            aspect_ratio=None,
            height_policy="min",
            width_policy="max",
            styles={
                "width": "100%",
                "max-height": max_height,
                "overflow": "clip",
            },
            css_classes=["legend"],
        )

        merged_legend_items = {}

        for item in legend_items:
            label = str(item.label.value)
            # Without renderers, no color will be shown
            # next to a legend item.
            legend_fig.renderers.extend(item.renderers)

            if label not in merged_legend_items:
                merged_legend_items[label] = item.renderers
            else:
                merged_legend_items[label].extend(item.renderers)

        legend_fig.toolbar.logo = None

        if reverse:
            legend_items = list(reversed(legend_items))

        legend = Legend(
            items=legend_items,
            orientation="vertical",
            location="left",
            click_policy=click_policy,
            ncols=legend_columns,
            styles={
                "width": "100%",
                "max-height": max_height,
            },
        )

        legend_fig.xaxis.visible = False
        legend_fig.yaxis.visible = False
        legend_fig.outline_line_alpha = 0.0
        legend_fig.add_layout(legend, place="center")
        legend_fig.background_fill_color = None
        legend_fig.x_range = Range1d(0, 0)

        return legend_fig

    @staticmethod
    def _add_global_css(html_file: Path, additional_css: str) -> None:
        """
        Inject a custom CSS into an HTML file.

        Insert the CSS before the closing </head> tag.
        Modifies the HTML file in-place.

        Parameters
        ----------
        html_file : Path
            Path to an HTML file.
        additional_css : str
            CSS code that should be injected globally.
        """
        style_tag = f"<style> {additional_css} </style>\n"
        with open(html_file, "r") as fd:
            html_content = fd.read()

        head_end_index = html_content.find("</head>")
        if head_end_index != -1:
            html_content = (
                html_content[:head_end_index]
                + style_tag
                + html_content[head_end_index:]
            )

        with open(html_file, "w") as fd:
            fd.write(html_content)

    @staticmethod
    def _output_bokeh_figure(
        bokeh_figure: Any,
        output_path: Optional[Path] = None,
        formats: Iterable[str] = ("html",),
    ) -> None:
        """
        Shows or exports bokeh figure.

        Parameters
        ----------
        bokeh_figure : Any
            Bokeh figure.
        output_path : Optional[Path]
            Path where figure should be saved.
        formats : Iterable[str]
            Iterable with formats names.
        """
        from bokeh.io import export_png, export_svg
        from bokeh.plotting import output_file, save, show

        if output_path is None:
            show(bokeh_figure)
            return

        if "html" in formats:
            output_file(f"{output_path}.html", mode="inline")
            save(bokeh_figure)
        if "png" in formats:
            export_png(bokeh_figure, filename=f"{output_path}.png")
        if "svg" in formats:
            export_svg(bokeh_figure, filename=f"{output_path}.svg")

    @staticmethod
    def _output_matplotlib_figure(
        bbox_extra: List[Any],
        output_path: Optional[Path] = None,
        formats: Iterable[str] = ("html",),
    ) -> None:
        """
        Shows or exports matplotlib figure.

        Parameters
        ----------
        bbox_extra : List[Any]
            Extra artist to be included.
        output_path : Optional[Path]
            Path where figure should be saved.
        formats : Iterable[str]
            Iterable with formats names.
        """
        if output_path is None:
            plt.show()
        else:
            for ext in formats:
                plt.savefig(
                    f"{output_path}.{ext}",
                    dpi=MATPLOTLIB_DPI,
                    bbox_extra_artists=bbox_extra,
                    bbox_inches="tight",
                )
        plt.close()

    @staticmethod
    def _get_comparison_color_scheme(n_colors: int) -> List[Tuple]:
        """
        Creates default color schema to use for comparison plots (such as
        violin plot, bubble chart etc.).

        Parameters
        ----------
        n_colors : int
            Number of colors to return.

        Returns
        -------
        List[Tuple]
            List of colors to use for plotting.
        """
        cmap = plt.get_cmap("nipy_spectral")
        return [cmap(i) for i in np.linspace(0.0, 1.0, n_colors)]

    @staticmethod
    def _create_custom_hover_template(
        names: List[str],
        values: Optional[List[str]] = None,
        units: Optional[List[Optional[str]]] = None,
    ) -> str:
        """
        Function creating custom template for tooltip displaying when hover
        event occurs. This tooltip is part of bokeh features.

        Parameters
        ----------
        names : List[str]
            List with names, displayed before values.
        values : Optional[List[str]]
            List with names of fields (in source object) containing values.
        units : Optional[List[Optional[str]]]
            List with units, displayed after values.

        Returns
        -------
        str
            HTML template for tooltip.
        """
        if values is None:
            values = [f"@{{{name}}}" for name in names]
        else:
            values = [
                f"@{{{value[1:]}}}" if value[0] == "@" else value
                for value in values
            ]
        if units is None:
            units = ["" for _ in names]
        else:
            units = ["" if unit is None else unit for unit in units]
        template = """
        <tr class="bk-tooltip-entry">
            <td class="bk-tt-entry-name">%s</td>
            <td class="bk-tt-entry-value">%s%s</td>
        </tr>
        """
        result = "<table>"
        for name, value, unit in zip(names, values, units):
            result += template % (name, value, unit)
        result += "</table>"
        return result

    @staticmethod
    def _matplotlib_color_to_bokeh(
        color: Tuple[float, float, float, float],
    ) -> Tuple[int, int, int, float]:
        """
        Converts color from matplotlib format to bokeh format.

        Parameters
        ----------
        color : Tuple[float, float, float, float]
            Color in matplotlib format.

        Returns
        -------
        Tuple[int, int, int, float]
            Color in bokeh format.
        """
        return (
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
            color[3],
        )


class ViolinComparisonPlot(Plot):
    """
    Violin plots comparing different metrics.
    """

    def __init__(
        self,
        metric_data: Dict[str, List],
        metric_labels: List[str],
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE // 3,
        colors: Optional[List] = None,
        color_offset: int = 0,
    ):
        """
        Violin plots comparing different metrics.

        Parameters
        ----------
        metric_data : Dict[str, List]
            Map between name of the model and list of metrics to visualize.
        metric_labels : List[str]
            Names of the metrics in order.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        colors : Optional[List]
            List with colors which should be used to draw plots.
        color_offset : int
            How many colors from default color list should be skipped.
        """
        if colors is None:
            colors = self._get_comparison_color_scheme(
                len(metric_data) + color_offset
            )

        super().__init__(width, height, title, colors=colors[color_offset:])

        self.metric_data = metric_data
        self.metric_labels = metric_labels

    def plot_matplotlib(
        self, output_path: Path, output_formats: Iterable[str]
    ):
        num_plots = len(self.metric_labels)
        legend_lines, legend_labels = [], []
        fig, axs = plt.subplots(
            num_plots,
            1,
            figsize=self._plt_figsize((1, num_plots)),
            dpi=MATPLOTLIB_DPI,
        )
        if num_plots == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        bbox_extra = []
        if self.title:
            bbox_extra.append(fig.suptitle(self.title))

        for i, (color, (sample_name, samples)) in enumerate(
            zip(self.colors, self.metric_data.items())
        ):
            for ax, metric_sample in zip(axs, samples):
                vp = ax.violinplot(metric_sample, positions=[i], vert=False)
                for body in vp["bodies"]:
                    body.set_color(color)
                vp["cbars"].set_color(color)
                vp["cmins"].set_color(color)
                vp["cmaxes"].set_color(color)
            # dummy plot used to create a legend
            (line,) = plt.plot([], label=sample_name, color=color)
            legend_lines.append(line)
            legend_labels.append(sample_name)

        for ax, metric_name in zip(axs, self.metric_labels):
            ax.set_title(metric_name)
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        bbox_extra.append(
            fig.legend(
                legend_lines,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=[0.5, 0],
                ncol=2,
            )
        )
        plt.tight_layout()

        self._output_matplotlib_figure(
            bbox_extra,
            output_path,
            output_formats,
        )

    def plot_bokeh(self, output_path: Path, output_formats: Iterable[str]):
        from bokeh.layouts import gridplot, layout
        from bokeh.models import (
            ColumnDataSource,
            Div,
            LegendItem,
            Patch,
        )
        from bokeh.plotting import figure
        from scipy.stats import gaussian_kde

        margins = (0, 20, 0, 10)
        fig_num = len(self.metric_labels)

        violin_figs = {
            metric_label: figure(
                title=metric_label,
                tools="pan,box_zoom,wheel_zoom,reset,save",
                toolbar_location=None,
                output_backend="webgl",
                sizing_mode="scale_both",
                margin=margins,
                match_aspect=True,
                height_policy="fit",
                width_policy="max",
                css_classes=["plot", "violin"],
                styles={"max-height": f"{70 // fig_num}vh"},
            )
            for metric_label in self.metric_labels
        }

        legend_items: List[LegendItem] = []
        for i, (color, (sample_name, samples)) in enumerate(
            zip(self.colors, self.metric_data.items())
        ):
            renderers = []
            for name, sample in zip(self.metric_labels, samples):
                x_min = min(sample)
                x_max = max(sample)
                x = np.linspace(x_min, x_max, 1000)
                if x_min == x_max:
                    y = np.full(x.shape, 0.45)
                else:
                    kde = gaussian_kde(sample)
                    y = kde.pdf(x)
                    y *= 0.45 / max(y)

                # Create the violin plot
                source = ColumnDataSource(
                    data=dict(
                        x=np.hstack([x, x[::-1]]),
                        y=i + np.hstack([y, -y[::-1]]),
                    )
                )
                renderer = violin_figs[name].add_glyph(
                    source,
                    Patch(
                        x="x",
                        y="y",
                        fill_color=color,
                        fill_alpha=0.5,
                        line_color=color,
                    ),
                )
                renderers.append(renderer)

                padding_percentage = 0.10
                padding = padding_percentage * (x_max - x_min)

                # Ensure a minimal width of a plot.
                minimal_width = (0.1) ** 10
                x_start = x_min - padding
                x_end = x_max + padding
                current_width = x_end - x_start
                if current_width < minimal_width:
                    missing_width = minimal_width - current_width
                    half_of_missing_width = missing_width / 2
                    x_start -= half_of_missing_width
                    x_end += half_of_missing_width

                # Add lines for min and max
                for line_start, line_end in (
                    ([x_min, x_max], [i, i]),
                    ([x_min, x_min], [i - 0.2, i + 0.2]),
                    ([x_max, x_max], [i - 0.2, i + 0.2]),
                ):
                    violin_figs[name].line(
                        line_start,
                        line_end,
                        color=color,
                        line_width=2,
                    )
            legend_items.append(
                LegendItem(label=sample_name, renderers=renderers)
            )

        # Adjust X range to the smallest and largest outlier plus paddings.
        padding_percentage = 0.05
        min_max = self._find_min_max_for_each_plot(self.metric_data)
        for i, value in enumerate(violin_figs.values()):
            x_min = min_max[i][0]
            x_max = min_max[i][1]
            padding = padding_percentage * (x_max - x_min)

            # Ensure a minimal width of a plot.
            minimal_width = (0.1) ** 10
            x_start = x_min - padding
            x_end = x_max + padding
            current_width = x_end - x_start
            if current_width < minimal_width:
                missing_width = minimal_width - current_width
                half_of_missing_width = missing_width / 2
                x_start -= half_of_missing_width
                x_end += half_of_missing_width
            value.x_range.start = x_start
            value.x_range.end = x_end

        legend_fig = self._create_bokeh_legend_fig(
            legend_items, click_policy="hide", reverse=True
        )

        if self.title is not None:
            violin_figs[self.metric_labels[0]].add_layout(
                Div(text=self.title), "above"
            )

        grid_fig = gridplot(
            [[violin_figs[name]] for name in self.metric_labels],
            merge_tools=False,
            toolbar_location=None,
            toolbar_options={"logo": None},
            sizing_mode="stretch_width",
        )
        grid_fig.css_classes = ["violin-plots"]
        grid_fig.styles = {"max-height": "70vh"}

        final_fig = layout(
            children=[[grid_fig], [legend_fig]],
            sizing_mode="scale_width",
        )

        self._output_bokeh_figure(
            final_fig,
            output_path,
            output_formats,
        )

    def _find_min_max_for_each_plot(
        self, metrics_data: Dict[str, List[List[float]]]
    ) -> List[Tuple[float, float]]:
        """
        Find min and max values for each violin plot.

        Parameters
        ----------
        metrics_data : Dict[str, List[List[float]]]
            Metrics data for each violin plot.

        Returns
        -------
        List[Tuple[float, float]]
            List of (min, max) tuples.
            Each index corresponds to a violin plot.
        """
        index_lists = []

        for key, lists in metrics_data.items():
            while len(index_lists) < len(lists):
                index_lists.append([])

            for index, sublist in enumerate(lists):
                index_lists[index].extend(sublist)

        # Calculate min and max for each index list.
        result = []
        for index_list in index_lists:
            if not index_list:
                continue
            min_value = min(index_list)
            max_value = max(index_list)
            result.append((min_value, max_value))

        return result


class RadarChart(Plot):
    """
    Radar chart comparing different metrics.
    """

    def __init__(
        self,
        metric_data: Dict[str, List],
        metric_labels: List[str],
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE,
        colors: Optional[List] = None,
        color_offset: int = 0,
    ):
        """
        Radar chart comparing different metrics.

        Parameters
        ----------
        metric_data : Dict[str, List]
            Map between name of the model and list of metrics to visualize.
        metric_labels : List[str]
            Names of the metrics in order.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        colors : Optional[List]
            List with colors which should be used to draw plots.
        color_offset : int
            How many colors from default color list should be skipped.
        """
        if colors is None:
            colors = self._get_comparison_color_scheme(
                len(metric_data) + color_offset
            )

        super().__init__(width, height, title, colors=colors[color_offset:])

        self.metric_data = metric_data
        self.metric_labels = metric_labels

    def plot_matplotlib(
        self, output_path: Path, output_formats: Iterable[str]
    ):
        n_metrics = len(self.metric_labels)

        angles = [n / n_metrics * 2 * pi for n in range(n_metrics)]
        fig, ax = plt.subplots(
            1,
            1,
            figsize=self._plt_figsize(),
            dpi=MATPLOTLIB_DPI,
            subplot_kw={"projection": "polar"},
        )
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles, self.metric_labels)
        ax.set_rlabel_position(1 / (n_metrics * 2) * 2 * pi)
        ax.set_yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"])
        ax.set_ylim((0, 1.0))
        bbox_extra = []
        if self.title:
            bbox_extra.append(fig.suptitle(self.title))

        angles += [0]
        linestyles = ["-", "--", "-.", ":"]
        for i, (color, (sample_name, sample)) in enumerate(
            zip(self.colors, self.metric_data.items())
        ):
            sample += sample[:1]
            ax.plot(
                angles,
                sample,
                label=sample_name,
                color=color,
                linestyle=linestyles[i % len(linestyles)],
            )
            ax.fill(
                angles,
                sample,
                color=color,
                alpha=0.25,
            )
        bbox_extra.append(
            plt.legend(
                bbox_to_anchor=[0.50, -0.05],
                loc="upper center",
                ncol=2,
            )
        )

        angles = np.array(angles)
        angles[np.cos(angles) <= -1e-5] += pi
        angles = np.rad2deg(angles)
        for i in range(n_metrics):
            label = ax.get_xticklabels()[i]
            labelname, angle = self.metric_labels[i], angles[i]
            x, y = label.get_position()
            lab = ax.text(
                x,
                y,
                labelname,
                transform=label.get_transform(),
                ha=label.get_ha(),
                va=label.get_va(),
            )
            lab.set_rotation(-angle)
            lab.set_fontsize("large")
            bbox_extra.append(lab)
        ax.set_xticklabels([])

        self._output_matplotlib_figure(
            bbox_extra,
            output_path,
            output_formats,
        )

    def plot_bokeh(self, output_path: Path, output_formats: Iterable[str]):
        from bokeh.layouts import layout
        from bokeh.models import (
            ColumnDataSource,
            HoverTool,
            Label,
            LegendItem,
            Patch,
        )
        from bokeh.plotting import figure

        radius = 300

        radar_fig = figure(
            title=self.title,
            tools="reset,save",
            toolbar_location="above",
            width=self.width,
            height=self.height,
            max_width=self.width * 2,
            max_height=self.height,
            match_aspect=True,
            output_backend="webgl",
            sizing_mode="scale_both",
            height_policy="min",
            width_policy="max",
            styles={
                "max-height": "70vh",
            },
        )

        radar_fig.grid.visible = False
        radar_fig.xaxis.visible = False
        radar_fig.yaxis.visible = False
        radar_fig.background_fill_alpha = 0
        radar_fig.toolbar.logo = None

        for i in [25, 50, 75, 100]:
            radar_fig.circle(
                x=0,
                y=0,
                radius=radius * i / 100,
                line_color="black" if i == 100 else "gray",
                fill_color="black",
                fill_alpha=0.05 if i == 100 else 0,
            )
            if i != 100:
                radar_fig.add_layout(
                    Label(x=0, y=radius * i / 100, text=f"{i}%")
                )

        for i, name in enumerate(self.metric_labels):
            a = i / len(self.metric_labels) * 2 * np.pi
            radar_fig.line(
                [0, radius * np.cos(a + 0.5 * np.pi)],
                [0, radius * np.sin(a + 0.5 * np.pi)],
                color="gray",
            )
            flip = 0.5 * np.pi < a < 1.5 * np.pi
            text_a = a - np.pi if flip else a
            radar_fig.add_layout(
                Label(
                    x=(radius + 12) * np.cos(a + 0.5 * np.pi),
                    y=(radius + 12) * np.sin(a + 0.5 * np.pi),
                    text=name,
                    angle=text_a,
                    text_align="center",
                    text_baseline="middle",
                )
            )

        sorted_metric_data = list(enumerate(self.metric_data.items()))
        sorted_metric_data.sort(key=lambda m: sum(m[1][1]), reverse=True)

        legend_items = {name: [] for name in self.metric_data.keys()}
        for color_id, (sample_name, samples) in sorted_metric_data:
            x = []
            y = []
            for j, sample in enumerate(samples):
                a = j * 2 * np.pi / len(self.metric_labels)
                x.append(sample * radius * np.cos(a + 0.5 * np.pi))
                y.append(sample * radius * np.sin(a + 0.5 * np.pi))

            renderer = radar_fig.add_glyph(
                ColumnDataSource(
                    data=dict(
                        x=x,
                        y=y,
                    )
                ),
                Patch(
                    x="x",
                    y="y",
                    fill_color=self.colors[color_id],
                    fill_alpha=0.5,
                    line_color=self.colors[color_id],
                ),
            )
            radar_fig.add_tools(
                HoverTool(
                    renderers=[renderer],
                    tooltips=self._create_custom_hover_template(
                        self.metric_labels,
                        values=[f"{100 * s:.2f}" for s in samples],
                        units=["%" for _ in samples],
                    ),
                    toggleable=False,
                )
            )
            legend_items[sample_name].append(renderer)

        legend_items = [
            LegendItem(label=label, renderers=renderers)
            for label, renderers in legend_items.items()
        ]

        # dummy patch for correct initial zoom
        radar_fig.add_glyph(
            ColumnDataSource(
                data=dict(
                    x=[radius * 1.05, 0, -radius * 1.05, 0],
                    y=[0, radius * 1.05, 0, -radius * 1.05],
                )
            ),
            Patch(
                x="x",
                y="y",
                fill_alpha=0,
                line_alpha=0,
            ),
        )

        legend_fig = self._create_bokeh_legend_fig(
            legend_items, click_policy="hide"
        )

        RadarChart._remove_redirect_on_click(radar_fig)

        final_fig = layout(
            [[radar_fig], [legend_fig]], sizing_mode="scale_width"
        )
        self._output_bokeh_figure(final_fig, output_path, output_formats)

    @staticmethod
    def _remove_redirect_on_click(bokeh_figure: object):
        """
        Remove a redirection occurring after clicking on a plot.

        It requires `figclass: remove-href` in MyST `{figure}`.
        The rationale for removing the redirection is that the legend
        of a figure needs to be clickable without side effects
        of redirecting to the image of a plot.

        Parameters
        ----------
        bokeh_figure : object
            Bokeh figure, from which a redirection should be removed.
            Actual type is `plotting.bokeh.figure`, but it could not be
            specified without importing bokeh module.

        Raises
        ------
        TypeError
            If `bokeh_figure` has type other than figure or its subtypes.
        """
        from bokeh.events import MouseEnter
        from bokeh.models import CustomJS
        from bokeh.plotting import figure

        def remove_href() -> CustomJS:
            return CustomJS(
                code=(
                    'document.querySelector(".prevent-redirection a")'
                    '.removeAttribute("href");'
                )
            )

        if not isinstance(bokeh_figure, figure):
            raise TypeError(
                f"{bokeh_figure.__qualname__} has to be of type {type(figure)}"
                f" in {RadarChart._remove_redirect_on_click.__qualname__}()."
            )

        bokeh_figure.js_on_event(MouseEnter, remove_href())


class BubblePlot(Plot):
    """
    Bubble plot comparing three metrics.
    """

    def __init__(
        self,
        x_data: List[float],
        x_label: str,
        y_data: List[float],
        y_label: str,
        size_data: List[float],
        size_label: str,
        bubble_labels: List[str],
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE,
        colors: Optional[List] = None,
        color_offset: int = 0,
    ):
        """
        Create bubble plot.

        Parameters
        ----------
        x_data : List[float]
            The values for X dimension.
        x_label : str
            Name of the X axis.
        y_data : List[float]
            The values for Y dimension.
        y_label : str
            Name of the Y axis.
        size_data : List[float]
            Sizes of subsequent bubbles.
        size_label : str
            Name of the size values.
        bubble_labels : List[str]
            Labels for consecutive bubbles.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        colors : Optional[List]
            List with colors which should be used to draw plots.
        color_offset : int
            How many colors from default color list should be skipped.
        """
        if colors is None:
            colors = self._get_comparison_color_scheme(
                len(x_data) + color_offset
            )

        super().__init__(width, height, title, colors=colors[color_offset:])

        self.x_data = x_data
        self.x_label = x_label
        self.y_data = y_data
        self.y_label = y_label
        self.size_data = size_data
        self.size_label = size_label
        self.bubble_labels = bubble_labels

        x_range = max(x_data) - min(x_data) + 1e-9
        y_range = max(y_data) - min(y_data) + 1e-9

        self.x_lim = (
            min(x_data) - 0.15 * x_range,
            max(x_data) + 0.15 * x_range,
        )
        self.y_lim = (
            min(y_data) - 0.15 * y_range,
            max(y_data) + 0.15 * y_range,
        )

        max_size = max(size_data)
        min_size = min(size_data)

        self.bubble_size = (
            (np.array(size_data) - min_size) / (max_size - min_size + 1) * 100
        ).tolist()

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=self._plt_figsize(),
            dpi=MATPLOTLIB_DPI,
        )

        markers = []
        bbox_extra = []
        for x, y, size, label, color in zip(
            self.x_data,
            self.y_data,
            self.bubble_size,
            self.bubble_labels,
            self.colors,
        ):
            marker = ax.scatter(
                x,
                y,
                s=15 + size**1.75,
                label=label,
                color=color,
                alpha=0.5,
                edgecolors="black",
            )
            markers.append(marker)

        legend = ax.legend(
            loc="upper center",
            handles=markers,
            bbox_to_anchor=[0.5, -0.08],
            ncol=2,
        )
        for handler in legend.legend_handles:
            handler.set_sizes([40.0])
        ax.add_artist(legend)
        bbox_extra.append(legend)

        maxsize = max(self.size_data)
        minsize = min(self.size_data)
        bubblemarkers, bubblelabels = [], []
        for i in [0, 25, 50, 75, 100]:
            bubblemarker = ax.scatter(
                [],
                [],
                s=(15 + i**1.75),
                color="None",
                edgecolors=mpl.rcParams["legend.labelcolor"],
            )
            bubblemarkers.append(bubblemarker)
            bubblelabels.append(
                format_size(minsize + i / 100 * (maxsize - minsize))
            )
        bubblelegend = ax.legend(
            bubblemarkers,
            bubblelabels,
            handletextpad=3,
            labelspacing=4.5,
            borderpad=3,
            title=self.size_label,
            frameon=False,
            bbox_to_anchor=[1.05, 0.5],
            loc="center left",
        )
        bubblelegend._legend_box.sep = 20
        ax.add_artist(bubblelegend)
        bbox_extra.append(bubblelegend)

        box = ax.get_position()
        ax.set_position(
            [
                box.x0,
                box.y0 + 0.05,
                box.width * 0.85,
                box.height - 0.05,
            ]
        )

        if self.title:
            bbox_extra.append(fig.suptitle(self.title))

        ax.set_xlim(*self.x_lim)
        ax.set_ylim(*self.y_lim)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        self._output_matplotlib_figure(
            bbox_extra,
            output_path,
            output_formats,
        )

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.layouts import layout
        from bokeh.models import (
            ColumnDataSource,
            HoverTool,
            LegendItem,
            Range1d,
        )
        from bokeh.plotting import figure

        source = ColumnDataSource(
            dict(
                x=self.x_data,
                y=self.y_data,
                size=15 + np.array(self.bubble_size),
                model_size=[format_size(s) for s in self.size_data],
                color=self.colors[: len(self.x_data)],
                name=self.bubble_labels,
            )
        )
        margins = (0, 20, 0, 10)
        bubbleplot_fig = figure(
            title=self.title,
            x_range=Range1d(*self.x_lim),
            y_range=Range1d(*self.y_lim),
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location="above",
            width=self.width,
            height=self.height,
            x_axis_label=self.x_label,
            y_axis_label=self.y_label,
            output_backend="webgl",
            sizing_mode="scale_width",
            height_policy="fit",
            width_policy="max",
            css_classes=["plot"],
            margin=margins,
            styles={
                "max-height": "70vh",
            },
        )
        bubbleplot_fig.toolbar.logo = None

        scatter_renderer = bubbleplot_fig.scatter(
            x="x",
            y="y",
            size="size",
            fill_color="color",
            line_color="black",
            source=source,
        )

        # The legend of a bubble plot.
        legend_items = [
            LegendItem(label=label, renderers=[scatter_renderer], index=i)
            for i, label in enumerate(self.bubble_labels)
        ]

        legend_fig = self._create_bokeh_legend_fig(
            legend_items, max_height="10vh"
        )

        # custom tooltips
        bubbleplot_fig.add_tools(
            HoverTool(
                tooltips=self._create_custom_hover_template(
                    ["Model", self.size_label, self.x_label, self.y_label],
                    values=["@name", "@model_size", "@x", "@y"],
                )
            )
        )

        final_fig = layout(
            [[bubbleplot_fig], [legend_fig]], sizing_mode="stretch_width"
        )

        self._output_bokeh_figure(
            final_fig,
            output_path,
            output_formats,
        )


class ConfusionMatrixPlot(Plot):
    """
    Confusion matrix plot.
    """

    def __init__(
        self,
        confusion_matrix: ArrayLike,
        class_names: List[str],
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE,
        cmap: Optional[Any] = None,
    ):
        """
        Create a confusion matrix plot.

        Parameters
        ----------
        confusion_matrix : ArrayLike
            Square numpy matrix containing the confusion matrix. 0-th axis
            stands for ground truth, 1-st axis stands for predictions.
        class_names : List[str]
            List of the class names.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        cmap : Optional[Any]
            Color map for the plot.
        """
        if cmap is None:
            if len(class_names) < 50:
                cmap = plt.get_cmap("BuPu")
            else:
                cmap = plt.get_cmap("nipy_spectral_r")

        super().__init__(width, height, title, cmap=cmap)

        self.class_names = class_names

        confusion_matrix = np.array(confusion_matrix, dtype=np.float32)

        # compute sensitivity
        sensitivity = confusion_matrix.diagonal() / confusion_matrix.sum(
            axis=1
        )
        sensitivity = sensitivity.reshape(1, len(class_names))

        # compute precision
        precision = confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
        precision = precision.reshape(len(class_names), 1)
        # change nan to 0
        precision[np.isnan(precision)] = 0.0

        # compute overall accuracy
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        # normalize confusion matrix
        confusion_matrix /= confusion_matrix.sum(axis=0)
        confusion_matrix = confusion_matrix.transpose()
        # change nan to 0
        confusion_matrix[np.isnan(confusion_matrix)] = 0.0

        def _value_to_nondiagonal_color(
            value: Union[float, np.ndarray], cmap: Optional[Any]
        ) -> np.ndarray:
            """
            Calculates colors of non-diagonal cells in confusion matrix.

            Parameters
            ----------
            value : Union[float, np.ndarray]
                Values from confusion matrix.
            cmap : Optional[Any]
                Color map associating values with colors.

            Returns
            -------
            np.ndarray
                Calculated colors.
            """
            color = np.asarray(
                cmap(1 - np.log2(99 * value + 1) / np.log2(100))
            )
            color[..., 3] = np.log2(99 * value + 1) / np.log2(100)
            return color

        # Cal[culate colors for confusion matrix
        colors = np.zeros(confusion_matrix.shape + (4,))
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                if i == j:
                    colors[i, j] = cmap(confusion_matrix[i, j])
                else:
                    colors[i, j] = _value_to_nondiagonal_color(
                        confusion_matrix[i, j], cmap
                    )

        self.confusion_matrix = confusion_matrix
        self.confusion_matrix_colors = colors
        self.sensitivity = sensitivity
        self.accuracy = accuracy
        self.precision = precision

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        fig = plt.figure(figsize=self._plt_figsize(), dpi=MATPLOTLIB_DPI)

        percent_font_size = (
            0.6
            * MATPLOTLIB_FONT_SIZE
            * min(1, 20 / self.confusion_matrix.shape[0])
        )

        vectors = 1
        if len(self.class_names) >= 50:
            vectors = 0
        gs = gridspec.GridSpec(
            len(self.class_names) + vectors, len(self.class_names) + vectors
        )
        axConfMatrix = fig.add_subplot(
            gs[0 : len(self.class_names), 0 : len(self.class_names)]
        )
        plots = [axConfMatrix]
        if len(self.class_names) < 50:
            axPredicted = fig.add_subplot(
                gs[len(self.class_names), 0 : len(self.class_names)],
                sharex=axConfMatrix,
            )
            axActual = fig.add_subplot(
                gs[0 : len(self.class_names), len(self.class_names)],
                sharey=axConfMatrix,
            )
            axTotal = fig.add_subplot(
                gs[len(self.class_names), len(self.class_names)],
                sharex=axActual,
                sharey=axPredicted,
            )
            plots = [axPredicted, axConfMatrix, axActual, axTotal]
        # define ticks for classes
        ticks = np.arange(len(self.class_names))

        # configure and draw confusion matrix
        if len(self.class_names) < 50:
            axConfMatrix.set_xticks(ticks)
            axConfMatrix.set_xticklabels(
                self.class_names,
                rotation=90,
                fontweight="bold",
            )
            axConfMatrix.set_yticks(ticks)
            axConfMatrix.set_yticklabels(
                self.class_names,
                fontweight="bold",
            )
            axConfMatrix.xaxis.set_ticks_position("top")
        else:
            axConfMatrix.tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )
        axConfMatrix.xaxis.set_label_position("top")
        axConfMatrix.set_xlabel(
            "Actual class",
            fontweight="bold",
        )
        axConfMatrix.set_ylabel(
            "Predicted class",
            fontweight="bold",
        )
        img = axConfMatrix.imshow(
            self.confusion_matrix_colors,
            interpolation="nearest",
            cmap=self.cmap,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )

        if len(self.class_names) < 50:
            # add percentages for confusion matrix
            for i, j in itertools.product(
                range(len(self.class_names)), range(len(self.class_names))
            ):
                txt = axConfMatrix.text(
                    j,
                    i,
                    (
                        "100"
                        if self.confusion_matrix[i, j] == 1.0
                        else f"{100.0 * self.confusion_matrix[i, j]:3.1f}"
                    ),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=percent_font_size,
                )
                txt.set_path_effects(
                    [patheffects.withStroke(linewidth=2, foreground="w")]
                )

            # configure and draw sensitivity percentages
            axPredicted.set_xticks(ticks)
            axPredicted.set_yticks([0])
            axPredicted.set_xlabel("Sensitivity", fontweight="bold")
            axPredicted.imshow(
                self.sensitivity,
                interpolation="nearest",
                cmap="RdYlGn" if self.cmap is None else self.cmap,
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )
            for i in range(len(self.class_names)):
                txt = axPredicted.text(
                    i,
                    0,
                    (
                        "100"
                        if self.sensitivity[0, i] == 1.0
                        else f"{100.0 * self.sensitivity[0, i]:3.1f}"
                    ),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=percent_font_size,
                )
                txt.set_path_effects(
                    [patheffects.withStroke(linewidth=2, foreground="w")]
                )

            # configure and draw precision percentages
            axActual.set_xticks([0])
            axActual.set_yticks(ticks)
            axActual.set_ylabel("Precision", fontweight="bold")
            axActual.yaxis.set_label_position("right")
            axActual.imshow(
                self.precision,
                interpolation="nearest",
                cmap="RdYlGn" if self.cmap is None else self.cmap,
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )
            for i in range(len(self.class_names)):
                txt = axActual.text(
                    0,
                    i,
                    (
                        "100"
                        if self.precision[i, 0] == 1.0
                        else f"{100.0 * self.precision[i, 0]:3.1f}"
                    ),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=percent_font_size,
                )
                txt.set_path_effects(
                    [patheffects.withStroke(linewidth=2, foreground="w")]
                )

            # configure and draw total accuracy
            axTotal.set_xticks([0])
            axTotal.set_yticks([0])
            axTotal.set_xlabel("Accuracy", fontweight="bold")
            axTotal.imshow(
                np.array([[self.accuracy]]),
                interpolation="nearest",
                cmap="RdYlGn" if self.cmap is None else self.cmap,
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )
            txt = axTotal.text(
                0,
                0,
                f"{100 * self.accuracy:3.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=percent_font_size,
            )
            txt.set_path_effects(
                [patheffects.withStroke(linewidth=2, foreground="w")]
            )

            # disable axes for other matrices than confusion matrix
            for a in (axPredicted, axActual, axTotal):
                plt.setp(a.get_yticklabels(), visible=False)
                plt.setp(a.get_xticklabels(), visible=False)

        # draw colorbar for confusion matrix
        cbar = fig.colorbar(
            img, ax=plots, shrink=0.5, ticks=np.linspace(0.0, 1.0, 11), pad=0.1
        )
        cbar.ax.set_yticks(
            np.linspace(0.0, 1.0, 11), labels=list(range(0, 101, 10))
        )
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize("medium")
        suptitlehandle = None
        if self.title:
            suptitlehandle = fig.suptitle(
                f"{self.title} (ACC={self.accuracy:.5f})",
            )

        self._output_matplotlib_figure(
            [suptitlehandle] if suptitlehandle else [],
            output_path,
            output_formats,
        )

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.layouts import gridplot
        from bokeh.models import (
            ColumnDataSource,
            FactorRange,
            HoverTool,
            Range1d,
        )
        from bokeh.plotting import figure

        max_width_percentage = "100%"
        matrix_css_sizes = {
            "min-width": "80%",
            "max-width": max_width_percentage,
            "aspect-ration": "1 / 1",
            "max-height": "70vh",
        }

        # Prepare figure
        confusion_matrix_fig = figure(
            title=self.title,
            x_range=FactorRange(
                factors=self.class_names, bounds=(0, len(self.class_names))
            ),
            y_range=FactorRange(
                factors=self.class_names[::-1],
                bounds=(0, len(self.class_names)),
            ),
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location=None,
            x_axis_location="above",
            output_backend="webgl",
            sizing_mode="scale_both",
            css_classes=["plot", "confusion-matrix"],
            styles=matrix_css_sizes,
        )

        # Preprocess data
        confusion_matrix_colors = np.rot90(
            self.confusion_matrix_colors, k=-1
        ).reshape((-1, 4))
        coords = np.array(
            list(itertools.product(self.class_names, self.class_names)),
            dtype=str,
        )
        coords[:, 1] = coords[::-1, 1]
        percentage = np.rot90(self.confusion_matrix, k=-1).flatten() * 100
        source = ColumnDataSource(
            data={
                "Actual class": coords[:, 0],
                "Predicted class": coords[:, 1],
                "color": confusion_matrix_colors,
                "Percentage": percentage,
            }
        )

        # Draw confusion matrix
        confusion_matrix_fig.rect(
            x="Actual class",
            y="Predicted class",
            color="color",
            line_color=None,
            width=1,
            height=1,
            source=source,
        )

        # Set labels and styles
        confusion_matrix_fig.xaxis.axis_label = "Actual class"
        confusion_matrix_fig.yaxis.axis_label = "Predicted class"
        if len(self.class_names) < 50:
            confusion_matrix_fig.xaxis.major_label_orientation = "vertical"
        else:
            confusion_matrix_fig.xaxis.major_label_text_alpha = 0.0
            confusion_matrix_fig.yaxis.major_label_text_alpha = 0.0
            confusion_matrix_fig.xaxis.major_tick_line_alpha = 0.0
            confusion_matrix_fig.yaxis.major_tick_line_alpha = 0.0
        confusion_matrix_fig.xaxis.axis_line_alpha = 0.0
        confusion_matrix_fig.yaxis.axis_line_alpha = 0.0
        confusion_matrix_fig.grid.visible = False

        # Set custom tooltips
        confusion_matrix_fig.add_tools(
            HoverTool(
                tooltips=self._create_custom_hover_template(
                    ["Actual class", "Predicted class", "Percentage"],
                    units=[None, None, "%"],
                )
            )
        )

        # === Sensitivity ===

        # Prepare figure
        sensitivity_fig = figure(
            title=None,
            x_range=confusion_matrix_fig.x_range,
            y_range=FactorRange(factors=["Sensitivity"], bounds=(0, 1)),
            toolbar_location=None,
            output_backend="webgl",
            sizing_mode="scale_width",
            styles={
                "height": "5vh",
                "min-width": "80%",
                "max-width": max_width_percentage,
                "max-height": "5vh",
            },
        )

        # Preprocess data
        sensitivity_color = self.cmap(self.sensitivity).reshape((-1, 4))
        sensitivity_source = ColumnDataSource(
            data={
                "y": ["Sensitivity" for _ in self.class_names],
                "Class": self.class_names,
                "color": sensitivity_color,
                "Sensitivity": self.sensitivity.flatten() * 100,
            }
        )

        # Draw sensitivity
        sensitivity_fig.rect(
            x="Class",
            y="y",
            color="color",
            source=sensitivity_source,
            line_color="black",
            line_width=0.1,
            width=1,
            height=1,
        )

        # Add label and custom tooltip
        sensitivity_fig.xaxis.axis_label = "Sensitivity"
        sensitivity_fig.add_tools(
            HoverTool(
                tooltips=self._create_custom_hover_template(
                    ["Class", "Sensitivity"], units=[None, "%"]
                ),
                attachment="above",
            )
        )

        # === Precision ===

        # Prepare figure
        precision_fig = figure(
            title=None,
            x_range=FactorRange(factors=["Precision"], bounds=(0, 1)),
            y_range=confusion_matrix_fig.y_range,
            toolbar_location=None,
            y_axis_location="right",
            output_backend="webgl",
            sizing_mode="scale_height",
            match_aspect=True,
            styles={
                "width": "5vh",
                "max-width": "5vh",
            },
        )

        # Preprocess data
        precision_color = self.cmap(self.precision).reshape((-1, 4))
        precision_source = ColumnDataSource(
            data={
                "x": ["Precision" for _ in self.class_names],
                "Class": self.class_names,
                "color": precision_color,
                "Precision": self.precision.flatten() * 100,
            }
        )

        # Draw sensitivity
        precision_fig.rect(
            x="x",
            y="Class",
            color="color",
            source=precision_source,
            height=1,
            width=1,
            line_color="black",
            line_width=0.1,
        )

        # Add label and custom tooltip
        precision_fig.yaxis.axis_label = "Precision"
        precision_fig.add_tools(
            HoverTool(
                tooltips=self._create_custom_hover_template(
                    ["Class", "Precision"], units=[None, "%"]
                ),
                attachment="left",
            )
        )

        # === Accuracy ===

        # Prepare figure
        accuracy_fig = figure(
            title=None,
            x_range=FactorRange(factors=["x"], bounds=(0, 1)),
            y_range=FactorRange(factors=["y"], bounds=(0, 1)),
            output_backend="webgl",
            sizing_mode="fixed",
            toolbar_location=None,
            styles={
                "width": "5vh",
                "height": "5vh",
                "max-height": "5vh",
            },
        )

        # Preprocess data
        accuracy_color = self.cmap(self.accuracy)
        color_str = (
            f"#{int(255 * accuracy_color[0]):02X}"
            f"{int(255 * accuracy_color[1]):02X}"
            f"{int(255 * accuracy_color[2]):02X}"
        )
        accuracy_source = ColumnDataSource(
            data={
                "x": ["x"],
                "y": ["y"],
                "Accuracy": [float(self.accuracy) * 100],
            }
        )

        # Draw sensitivity
        accuracy_fig.rect(
            x="x",
            y="y",
            color=color_str,
            source=accuracy_source,
            width=1,
            height=1,
            line_color="black",
            line_width=0.1,
        )

        # Add label and custom tooltip
        accuracy_fig.xaxis.axis_label = "ACC"
        accuracy_fig.add_tools(
            HoverTool(
                tooltips=self._create_custom_hover_template(
                    ["Accuracy"], units=["%"]
                ),
                attachment="above",
            )
        )

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
        def create_color_scale_figure() -> figure:
            """
            Create a color scale figure spanning from 0% (red)
            to 100% (green).
            """
            # Prepare figure
            scale_fig = figure(
                title=None,
                x_range=Range1d(0.0, 100.0),
                y_range=["color"],
                tools="",
                toolbar_location=None,
                x_axis_location="below",
                y_axis_location="left",
                output_backend="webgl",
                sizing_mode="scale_width",
                margin=(40, 0, 0, 0),
                styles={
                    "height": "5vh",
                    "max-width": max_width_percentage,
                    "min-width": "80%",
                    "max-height": "5vh",
                },
            )

            # Draw scale
            scale_fig.vbar(
                x=np.linspace(0.0, 100.0, 256),
                top=100,
                width=1.0,
                color=self.cmap(np.linspace(0.0, 1.0, 256)),
            )

            # Set styles for scale
            scale_fig.xaxis.major_tick_line_alpha = 0.0
            scale_fig.yaxis.major_tick_line_alpha = 0.0
            scale_fig.yaxis.minor_tick_line_alpha = 0.0
            scale_fig.xaxis.axis_line_alpha = 0.0
            scale_fig.yaxis.axis_line_alpha = 0.0
            scale_fig.xaxis.major_label_text_alpha = 1.0

            # Change axis captions from numbers to percentages,
            # e.g.: 10 -> 10%.
            scale_fig.xaxis.major_label_overrides = {
                i: f"{i}%" for i in range(0, 120, 20)
            }

            return scale_fig

        # === Saving to file ===
        grid_fig = gridplot(
            [
                [confusion_matrix_fig, precision_fig],
                [sensitivity_fig, accuracy_fig],
                [create_color_scale_figure(), None],
            ],
            merge_tools=True,
            toolbar_location=None,
            sizing_mode="scale_width",
        )
        grid_fig.cols = ["6fr", "1fr"]

        self._output_bokeh_figure(
            grid_fig,
            output_path,
            output_formats,
        )


class RecallPrecisionCurvesPlot(Plot):
    """
    Recall-Precision curves for AP measurements.
    """

    def __init__(
        self,
        lines: List[Tuple[List, List]],
        class_names: List[str],
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE,
        cmap: Optional[Any] = None,
    ):
        """
        Create Recall-Precision curves for AP measurements.

        Parameters
        ----------
        lines : List[Tuple[List, List]]
            Per-class list of tuples with list of recall values and precision
            values.
        class_names : List[str]
            List of the class names.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        cmap : Optional[Any]
            Color map for the plot.
        """
        if cmap is None:
            cmap = plt.get_cmap("nipy_spectral_r")

        super().__init__(width, height, title, cmap=cmap)

        self.lines = lines
        self.class_names = class_names

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        fig = plt.figure(figsize=self._plt_figsize(), dpi=MATPLOTLIB_DPI)
        ax = fig.add_subplot(111)
        colors = [
            self.cmap(i) for i in np.linspace(0, 1, len(self.class_names))
        ]
        linestyles = ["-", "--", "-.", ":"]
        for i, (cls, line) in enumerate(zip(self.class_names, self.lines)):
            ax.plot(
                line[0],
                line[1],
                label=cls,
                c=colors[i],
                linewidth=3,
                linestyle=linestyles[i % len(linestyles)],
                alpha=0.8,
            )
        ncol = 6
        bbox_to_anchor = (
            0.5,
            -0.15 - (np.ceil(len(self.class_names) / ncol) - 1) * 0.04,
        )
        legendhandle = ax.legend(
            bbox_to_anchor=bbox_to_anchor, loc="lower center", ncol=ncol
        )
        ax.set_aspect("equal")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim((0.0, 1.01))
        ax.set_ylim((0.0, 1.01))
        ax.grid("on")
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_title(self.title)

        self._output_matplotlib_figure(
            [legendhandle],
            output_path,
            output_formats,
        )

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.models import Legend, Range1d
        from bokeh.plotting import figure

        precision_fig = figure(
            title=self.title,
            x_range=Range1d(0, 1),
            y_range=Range1d(0, 1),
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location="above",
            width=self.width,
            height=self.height,
            x_axis_label="recall",
            y_axis_label="precision",
            output_backend="webgl",
            max_width=self.width,
            max_height=self.height,
            match_aspect=True,
            sizing_mode="scale_both",
            height_policy="fit",
            width_policy="max",
            css_classes=["plot"],
        )

        colors = [
            self.cmap(i) for i in np.linspace(0, 1, len(self.class_names))
        ]
        linestyles = ["solid", "dashed", "dotted", "dotdash", "dashdot"]

        line_renderers = []
        for i, ((x, y), c) in enumerate(zip(self.lines, colors)):
            line_renderers.append(
                precision_fig.line(
                    x,
                    y,
                    color=self._matplotlib_color_to_bokeh(c),
                    line_width=2.0,
                    line_dash=linestyles[i % len(linestyles)],
                )
            )

        legend_ncol = 6
        for i in range((len(line_renderers) + legend_ncol - 1) // legend_ncol):
            labels = self.class_names[i * legend_ncol : (i + 1) * legend_ncol]
            renderers = line_renderers[i * legend_ncol : (i + 1) * legend_ncol]
            legend = Legend(
                items=[
                    (label, [renderer])
                    for label, renderer in zip(labels, renderers)
                ],
                orientation="horizontal",
                label_text_font="Lato",
                click_policy="mute",
                location="left",
            )
            legend.label_width = 120
            precision_fig.add_layout(legend, "below")

        self._output_bokeh_figure(precision_fig, output_path, output_formats)


class TruePositiveIoUHistogram(Plot):
    """
    Per-class True Positive IoU precision plot.
    """

    def __init__(
        self,
        iou_data: List[float],
        class_names: List[str],
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: Optional[int] = None,
        colors: Optional[List] = None,
        color_offset: int = 0,
    ):
        """
        Create per-class True Positive IoU precision plot.

        Parameters
        ----------
        iou_data : List[float]
            Per-class list of floats with IoU values.
        class_names : List[str]
            List of the class names.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : Optional[int]
            Height of the plot, if None, it will be calculated based on the
            number of classes.
        colors : Optional[List]
            List with colors which should be used to draw plots.
        color_offset : int
            How many colors from default color list should be skipped.
        """
        if height is None:
            height = 200 + 30 * len(class_names)
        if colors is None:
            colors = self._get_comparison_color_scheme(1 + color_offset)

        super().__init__(width, height, title, colors=colors[color_offset:])

        self.iou_data = iou_data
        self.class_names = class_names

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        plt.figure(figsize=self._plt_figsize(), dpi=MATPLOTLIB_DPI)
        plt.barh(
            self.class_names,
            np.array(self.iou_data),
            orientation="horizontal",
            color=self.colors[0],
        )
        plt.ylim((-1, len(self.class_names)))
        plt.yticks(
            np.arange(0, len(self.class_names)), labels=self.class_names
        )
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlabel("IoU precision")
        plt.ylabel("classes")
        if self.title:
            plt.title(f"{self.title}")

        self._output_matplotlib_figure(
            [],
            output_path,
            output_formats,
        )

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.models import ColumnDataSource, HoverTool, Range1d
        from bokeh.plotting import figure

        source = ColumnDataSource(
            data=dict(
                x=self.iou_data,
                y=self.class_names,
                y_idx=list(range(0, len(self.class_names))),
            )
        )

        hist_fig = figure(
            title=self.title,
            x_range=Range1d(0, 1),
            y_range=Range1d(-1, len(self.class_names)),
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location="above",
            width=self.width,
            height=self.height,
            x_axis_label="IoU precision",
            y_axis_label="classes",
            output_backend="webgl",
            max_width=self.width,
            max_height=self.height,
            match_aspect=True,
            sizing_mode="scale_both",
            height_policy="fit",
            width_policy="max",
            css_classes=["plot"],
        )

        hbar = hist_fig.hbar(
            y="y_idx",
            left=0,
            right="x",
            fill_color=self.colors[0],
            source=source,
        )

        hist_fig.yaxis.ticker = list(range(0, len(self.class_names)))
        hist_fig.yaxis.major_label_overrides = {
            i: label for i, label in enumerate(self.class_names)
        }

        hist_fig.add_tools(
            HoverTool(
                renderers=[hbar],
                tooltips=self._create_custom_hover_template(
                    ["Class", "IoU precision"], values=["@y", "@x"]
                ),
            )
        )

        self._output_bokeh_figure(hist_fig, output_path, output_formats)


class TruePositivesPerIoURangeHistogram(Plot):
    """
    Histogram of True Positive IoU values.
    """

    def __init__(
        self,
        iou_data: List[float],
        range_fraction: float = 0.05,
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE,
        colors: Optional[List] = None,
        color_offset: int = 0,
    ):
        """
        Create histogram of True Positive IoU values.

        Parameters
        ----------
        iou_data : List[float]
            All True Positive IoU values.
        range_fraction : float
            Fraction by which the range should be divided
            (1/number_of_segments).
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        colors : Optional[List]
            List with colors which should be used to draw plots.
        color_offset : int
            How many colors from default color list should be skipped.
        """
        if colors is None:
            colors = self._get_comparison_color_scheme(color_offset)

        super().__init__(width, height, title, colors=colors[color_offset:])

        self.iou_data = iou_data
        self.range_fraction = range_fraction
        self.x_range = np.arange(0, 1.1, 0.1)

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        plt.figure(
            figsize=self._plt_figsize(),
            dpi=MATPLOTLIB_DPI,
        )
        plt.hist(self.iou_data, self.x_range, color=self.colors[0])
        plt.xlabel("IoU ranges")
        plt.xticks(self.x_range, rotation=45)
        plt.ylabel("Number of masks in IoU range")
        if self.title:
            plt.title(f"{self.title}")

        self._output_matplotlib_figure([], output_path, output_formats)

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.plotting import figure

        hist_fig = figure(
            title=self.title,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location="above",
            width=self.width // 2,
            height=self.height,
            x_axis_label="IoU ranges",
            y_axis_label="Number of masks in IoU range",
            output_backend="webgl",
            max_width=self.width // 2,
            max_height=self.height,
            match_aspect=True,
            sizing_mode="scale_both",
            height_policy="fit",
            width_policy="max",
            css_classes=["plot"],
        )

        hist = np.histogram(self.iou_data, bins=self.x_range)[0]

        source = ColumnDataSource(
            dict(
                x_left=list(self.x_range[:-1]),
                x_right=list(self.x_range[1:]),
                x_mid=list(self.x_range[:-1] + self.x_range[1:] / 2),
                y=hist,
            )
        )

        vbar = hist_fig.vbar(
            x="x_mid",
            bottom=0,
            top="y",
            width=self.x_range[1] - self.x_range[0],
            fill_color=self.colors[0],
            source=source,
        )

        hist_fig.xaxis.ticker = self.x_range
        hist_fig.xaxis.major_label_orientation = "vertical"

        hist_fig.add_tools(
            HoverTool(
                renderers=[vbar],
                tooltips=self._create_custom_hover_template(
                    ["Number of masks", "min IoU", "max IoU"],
                    values=["@y", "@x_left", "@x_right"],
                ),
            )
        )

        self._output_bokeh_figure(hist_fig, output_path, output_formats)


class RecallPrecisionGradients(Plot):
    """
    Per-class gradients of precision dependent to recall.
    """

    def __init__(
        self,
        lines: List[Tuple[List, List]],
        class_names: List[str],
        avg_precisions: List[float],
        mean_avg_precision: float,
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: Optional[int] = None,
        cmap: Optional[Any] = None,
    ):
        """
        Create per-class gradients of precision dependent to recall.

        Provide per-class AP and mAP values.

        Parameters
        ----------
        lines : List[Tuple[List, List]]
            Per-class list of tuples with list of recall values and precision
            values.
        class_names : List[str]
            List of the class names.
        avg_precisions : List[float]
            Per-class AP values.
        mean_avg_precision : float
            The mAP value.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : Optional[int]
            Height of the plot. If None, height is calculated based on
            the amount of classes.
        cmap : Optional[Any]
            Color map for the plot.
        """
        if cmap is None:
            cmap = plt.get_cmap("RdYlGn")
        if height is None:
            height = 200 + 30 * len(class_names)

        super().__init__(width, height, title, cmap=cmap)

        self.lines = lines
        self.class_names = class_names
        self.avg_precisions = avg_precisions
        self.mean_avg_precision = mean_avg_precision

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        plt.figure(figsize=self._plt_figsize(), dpi=MATPLOTLIB_DPI)
        clsticks = []
        for i, (cls, line, averageprecision) in enumerate(
            zip(self.class_names, self.lines, self.avg_precisions)
        ):
            clscoords = np.ones(len(line[0])) * i
            points = np.array([line[0], clscoords]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments, cmap=self.cmap, norm=plt.Normalize(0, 1.0)
            )
            lc.set_array(line[1])
            lc.set_linewidth(10)
            plt.gca().add_collection(lc)
            clsticks.append(f"{cls} (AP={averageprecision:.4f})")
        plt.ylim((-1, len(self.class_names)))
        plt.yticks(np.arange(0, len(clsticks)), labels=clsticks)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlabel("recall")
        plt.ylabel("classes")
        ax = plt.gca()
        PCM = ax.get_children()[2]
        plt.colorbar(
            PCM,
            ax=ax,
            orientation="vertical",
            label="precision",
            fraction=0.1,
            pad=0.05,
        )
        if self.title:
            plt.title(f"{self.title} (mAP={self.mean_avg_precision})")

        self._output_matplotlib_figure(
            [],
            output_path,
            output_formats,
        )

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.models import (
            ColorBar,
            ColumnDataSource,
            LinearColorMapper,
            Range1d,
        )
        from bokeh.plotting import figure

        gradient_fig = figure(
            title=f"{self.title} (mAP={self.mean_avg_precision})",
            x_range=Range1d(0, 1),
            y_range=Range1d(-1, len(self.class_names)),
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location="above",
            width=self.width,
            height=self.height,
            x_axis_label="recall",
            y_axis_label="classes",
            output_backend="webgl",
            max_width=self.width,
            max_height=self.height,
            match_aspect=True,
            sizing_mode="scale_both",
            height_policy="fit",
            width_policy="max",
            css_classes=["plot"],
        )
        color_mapper = LinearColorMapper(
            palette=[
                self._matplotlib_color_to_bokeh(color)
                for color in self.cmap.colors
            ],
            low=0,
            high=1,
        )
        for i, line in enumerate(self.lines):
            x = np.linspace(0, 1, len(line[0]) + 2)
            y = np.ones(len(line[0]) + 2) * i
            source = ColumnDataSource(
                dict(
                    xs=[x[i : i + 2] for i in range(len(x) - 2)],
                    ys=[y[i : i + 2] for i in range(len(y) - 2)],
                    precision=line[1],
                )
            )
            gradient_fig.multi_line(
                xs="xs",
                ys="ys",
                line_color={"field": "precision", "transform": color_mapper},
                line_width=10,
                source=source,
            )

        gradient_fig.xaxis.ticker = np.arange(0, 1.1, 0.1)
        gradient_fig.yaxis.ticker = list(range(0, len(self.class_names)))
        gradient_fig.yaxis.major_label_overrides = {
            i: f"{label} (AP={ap:.4f})"
            for i, (label, ap) in enumerate(
                zip(self.class_names, self.avg_precisions)
            )
        }

        color_bar = ColorBar(
            color_mapper=color_mapper,
            title="Precision",
            border_line_color=None,
            background_fill_alpha=0,
        )
        gradient_fig.add_layout(color_bar, "below")

        self._output_bokeh_figure(gradient_fig, output_path, output_formats)


class LinePlot(Plot):
    """
    Line plot.
    """

    def __init__(
        self,
        lines: List[Tuple[List, List]],
        x_label: str,
        y_label: str,
        x_unit: Optional[str] = None,
        y_unit: Optional[str] = None,
        lines_labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE,
        colors: Optional[List] = None,
        color_offset: int = 0,
        x_scale: Literal["linear", "log"] = "linear",
        y_scale: Literal["linear", "log"] = "linear",
        dashed: Optional[List[bool]] = None,
        add_points: bool = False,
    ):
        """
        Create line plot.

        Parameters
        ----------
        lines : List[Tuple[List, List]]
            Per-class list of tuples with list of recall values and precision
            values.
        x_label : str
            Name of the X axis.
        y_label : str
            Name of the Y axis.
        x_unit : Optional[str]
            Unit for the X axis.
        y_unit : Optional[str]
            Unit for the Y axis.
        lines_labels : Optional[List[str]]
            Optional list of labels naming each line.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        colors : Optional[List]
            List with colors which should be used to draw plots.
        color_offset : int
            How many colors from default color list should be skipped.
        x_scale : Literal["linear", "log"]
            The scale of the x-axis.
        y_scale : Literal["linear", "log"]
            The scale of the y-axis.
        dashed : Optional[List[bool]]
            Whether lines should be dashed (by default false). It has to be
            the same length as lines.
        add_points : bool
            Whether data should be marked with points.
        """
        if colors is None:
            colors = self._get_comparison_color_scheme(
                len(lines) + color_offset
            )

        super().__init__(width, height, title, colors=colors[color_offset:])

        self.x_label = x_label if x_unit is None else f"{x_label} [{x_unit}]"
        self.y_label = y_label if y_unit is None else f"{y_label} [{y_unit}]"
        self.lines = lines
        self.lines_labels = lines_labels
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.add_points = add_points
        self.dashed = dashed
        if self.dashed is None:
            self.dashed = [False for _ in self.lines]
        assert len(self.dashed) == len(
            self.lines
        ), "`dashed` has to be defined for each line"

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        plt.figure(figsize=self._plt_figsize(), dpi=MATPLOTLIB_DPI)

        bbox_extra = []

        marker = "." if self.add_points else None
        for color, (x, y), dashed in zip(self.colors, self.lines, self.dashed):
            if dashed:
                plt.plot(
                    x,
                    y,
                    c=color,
                    linewidth=2,
                    marker=marker,
                    linestyle="dashed",
                    fillstyle="none",
                )
            else:
                plt.plot(x, y, c=color, linewidth=2, marker=marker)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid()
        plt.xscale(self.x_scale)
        plt.yscale(self.y_scale)
        if self.title:
            bbox_extra.append(plt.title(self.title))
        ncols = 2
        bbox_to_anchor = [0.5, -0.06]
        if self.lines_labels is not None:
            bbox_to_anchor[1] = -(
                np.ceil(len(self.lines_labels) / ncols) - 1
            ) * 0.04
        if self.lines_labels is not None:
            bbox_extra.append(
                plt.legend(
                    self.lines_labels,
                    loc="upper center",
                    bbox_to_anchor=bbox_to_anchor,
                    ncols=ncols,
                )
            )

        self._output_matplotlib_figure(
            bbox_extra,
            output_path,
            output_formats,
        )

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.layouts import gridplot
        from bokeh.models import LegendItem, Range1d
        from bokeh.plotting import figure

        safety_offset = 20

        plot_fig = figure(
            title=self.title,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location="above",
            width=self.width,
            height=self.height,
            max_height=self.height,
            x_axis_label=self.x_label,
            y_axis_label=self.y_label,
            output_backend="webgl",
            sizing_mode="scale_width",
            height_policy="fit",
            width_policy="max",
            css_classes=["plot"],
            x_axis_type=self.x_scale,
            y_axis_type=self.y_scale,
            styles={
                "height": "40vh",
            },
        )
        plot_fig.toolbar.logo = None

        renderers = [
            [
                plot_fig.line(
                    x,
                    y,
                    color=color,
                    line_width=2.0,
                    line_dash="dashed" if dashed else "solid",
                ),
            ]
            + (
                [
                    plot_fig.scatter(
                        x,
                        y,
                        color=color,
                        size=6 if dashed else 4,
                        fill_alpha=0.0 if dashed else 1.0,
                    ),
                ]
                if self.add_points
                else []
            )
            for color, (x, y), dashed in zip(
                self.colors, self.lines, self.dashed
            )
        ]
        # Adjust range of x-axis
        x_min = min(
            min([x for x in xs if not np.isnan(x)], default=float("inf"))
            for xs, _ in self.lines
        )
        x_max = max(
            max([x for x in xs if not np.isnan(x)], default=float("-inf"))
            for xs, _ in self.lines
        )
        diff = (x_max - x_min) * 0.05
        if diff < 1e-8:
            diff = 0.25
        plot_fig.x_range = Range1d(x_min - diff, x_max + diff)

        # Adjust the range of X values.
        plot_range = LinePlot.determine_range(
            [x for x, _ in self.lines], padding_percentage=0.10
        )
        plot_fig.x_range = Range1d(*plot_range)

        if self.lines_labels is not None:
            legend_data = [
                LegendItem(label=label, renderers=renderer)
                for label, renderer in zip(self.lines_labels, renderers)
            ]

            legend_fig = self._create_bokeh_legend_fig(
                legend_data,
                safety_offset=safety_offset,
                click_policy="hide",
                max_height="40vh",
            )

            plot_fig = gridplot(
                children=[[plot_fig], [legend_fig]], sizing_mode="scale_both"
            )

        self._output_bokeh_figure(plot_fig, output_path, output_formats)

    @staticmethod
    def determine_range(
        X: List[List[float]], padding_percentage: float
    ) -> Tuple[float, float]:
        """
        Determine an X range of values for a given padding.

        Parameters
        ----------
        X : List[List[float]]
            List of lists of x values.
        padding_percentage : float
            A number of percents of padding, which should be applied
            to ends of the plot. Half of the padding is applied to
            the left end and the second half - to the right.

        Returns
        -------
        Tuple[float, float]
            Padded lower and upper bound of a range of values.
        """
        global_max = -99999999
        global_min = +99999999
        for x in X:
            local_max = max(x)
            if local_max > global_max:
                global_max = local_max

            local_min = min(x)
            if local_min < global_min:
                global_min = local_min

        padding = padding_percentage * (global_max - global_min)
        global_min -= padding / 2
        global_max += padding / 2

        # Prevent the plot from having zero width.
        if global_min == global_max:
            global_max += 0.1
            global_min -= 0.1

        return (global_min, global_max)


class Barplot(Plot):
    """
    Barplot.
    """

    def __init__(
        self,
        x_data: List[Any],
        y_data: Dict[str, List[Union[int, float]]],
        x_label: str,
        y_label: str,
        x_unit: Optional[str] = None,
        y_unit: Optional[str] = None,
        max_bars_matplotlib: Optional[int] = None,
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE * 5 // 6,
        colors: Optional[List] = None,
        color_offset: int = 0,
        vertical_x_labels: bool = True,
    ):
        """
        Create barplot.

        Parameters
        ----------
        x_data : List[Any]
            List of x labels for bars.
        y_data : Dict[str, List[Union[int, float]]]
            Dictionary of values.
        x_label : str
            Name of the X axis.
        y_label : str
            Name of the Y axis.
        x_unit : Optional[str]
            Unit for the X axis.
        y_unit : Optional[str]
            Unit for the Y axis.
        max_bars_matplotlib : Optional[int]
            Max number of bars for matplotlib backend.
        title : Optional[str]
            Title of the plot.
        width : int
            Width of the plot.
        height : int
            Height of the plot.
        colors : Optional[List]
            List with colors which should be used to draw plots.
        color_offset : int
            How many colors from default color list should be skipped.
        vertical_x_labels : bool
            Whether labels on x-axis should be vertiacal.
        """
        if colors is None:
            colors = self._get_comparison_color_scheme(
                len(y_data) + color_offset
            )

        super().__init__(width, height, title, colors=colors[color_offset:])

        self.x_label = x_label if x_unit is None else f"{x_label} [{x_unit}]"
        self.y_label = y_label if y_unit is None else f"{y_label} [{y_unit}]"
        self.x_data = x_data
        self.y_data = y_data
        combined_data = []
        for v in y_data.values():
            combined_data.extend(v)
        self.y_data_std = np.std(combined_data)
        self.max_bars_matplotlib = max_bars_matplotlib
        self.bar_width = 0.8 / len(self.y_data)
        if len(self.y_data) == 1:
            self.bar_offset = [0.0]
        else:
            self.bar_offset = np.linspace(
                -0.4 + self.bar_width / 2,
                0.4 - self.bar_width / 2,
                len(self.y_data),
            ).tolist()
        self.vertical_x_labels = vertical_x_labels

    def plot_matplotlib(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        x_data = self.x_data
        y_data = self.y_data
        if self.max_bars_matplotlib is not None:
            x_data = self.x_data[: self.max_bars_matplotlib]
            y_data = {
                name: values[: self.max_bars_matplotlib]
                for name, values in self.y_data.items()
            }

        plt.figure(figsize=self._plt_figsize(), dpi=MATPLOTLIB_DPI)

        x_range = np.arange(0, len(x_data))

        bbox_extra = []
        for i, (label, values) in enumerate(y_data.items()):
            plt.bar(
                x_range + self.bar_offset[i],
                values,
                width=self.bar_width,
                color=self.colors[i],
                label=label,
            )

        if self.vertical_x_labels:
            plt.xticks(x_range, x_data, rotation=90)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.ticklabel_format(style="plain", axis="y")
        plt.grid()
        if self.title:
            bbox_extra.append(plt.title(self.title))
        if len(y_data) > 1:
            bbox_extra.append(
                plt.legend(
                    list(self.y_data.keys()),
                    loc="upper center",
                    bbox_to_anchor=[0.5, -0.2],
                    ncols=2,
                )
            )

        self._output_matplotlib_figure(
            bbox_extra,
            output_path,
            output_formats,
        )

    def plot_bokeh(
        self, output_path: Optional[Path], output_formats: Iterable[str]
    ):
        from bokeh.models import HoverTool, Range1d
        from bokeh.plotting import figure
        from bokeh.transform import dodge

        y_max = max([max(y) for y in self.y_data.values()]) * 1.01
        y_min = 0
        # If data have similar values, decrease range
        if self.y_data_std / y_max < 0.1:
            y_min = max(min([min(y) for y in self.y_data.values()]) * 0.95, 0)

        barplot_fig = figure(
            title=self.title,
            x_range=self.x_data,
            y_range=Range1d(y_min, y_max),
            tools="pan,box_zoom,wheel_zoom,reset,save",
            toolbar_location="above",
            width=self.width,
            height=self.height,
            x_axis_label=self.x_label,
            y_axis_label=self.y_label,
            output_backend="webgl",
            max_width=self.width,
            max_height=self.height,
            match_aspect=True,
            sizing_mode="scale_both",
            height_policy="fit",
            width_policy="max",
            css_classes=["plot"],
            styles={
                "max-width": "100%",
                "max-height": "80vh",
            },
        )
        barplot_fig.toolbar.logo = None

        data = dict(self.y_data, xdata=self.x_data)

        for i, label in enumerate(self.y_data.keys()):
            vbar = barplot_fig.vbar(
                x=dodge(
                    "xdata", self.bar_offset[i], range=barplot_fig.x_range
                ),
                top=label,
                source=data,
                bottom=0,
                fill_color=self.colors[i],
                width=self.bar_width,
                legend_label=label,
            )
            tooltips = [(self.x_label, "@xdata"), (self.y_label, f"@{label}")]
            if len(self.y_data) > 1:
                tooltips.insert(0, ("File", label))

            barplot_fig.add_tools(
                HoverTool(
                    renderers=[vbar],
                    tooltips=self._create_custom_hover_template(
                        [t[0] for t in tooltips],
                        values=[t[1] for t in tooltips],
                    ),
                )
            )

        if self.vertical_x_labels:
            barplot_fig.xaxis.major_label_orientation = "vertical"

        self._output_bokeh_figure(barplot_fig, output_path, output_formats)


@contextmanager
def choose_theme(
    custom_bokeh_theme: Union[bool, str, Path] = False,
    custom_matplotlib_theme: Union[bool, str, Path] = False,
) -> Generator[None, None, None]:
    """
    Context manager, allowing to temporally set theme.

    Parameters
    ----------
    custom_bokeh_theme : Union[bool, str, Path]
        If True uses BOKEH_THEME_FILE, if str or Path uses file specified
        by this path.
    custom_matplotlib_theme : Union[bool, str, Path]
        If True uses MATPLOTLIB_THEME_FILE, if str or Path uses file specified
        by this path.

    Yields
    ------
    None
        Theme context
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
        mpl.use("Agg")
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
