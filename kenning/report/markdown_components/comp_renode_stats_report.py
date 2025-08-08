# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for renode statistics comparison report generation.
"""

from collections import defaultdict
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from kenning.core.metrics import compute_renode_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def comparison_renode_stats_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    draw_titles: bool = True,
    colors: Optional[List] = None,
    **kwargs: Any,
) -> str:
    """
    Creates Renode comparison stats section of the report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project
        involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    color_offset : int
        How many colors from default color list should be skipped.
    draw_titles : bool
        Should titles be drawn on the plot.
    colors : Optional[List]
        Colors used for plots.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_multiple_time_series_plot

    from kenning.core.drawing import SERVIS_PLOT_OPTIONS, Barplot, LinePlot

    def retrieve_non_zero_profiler_data(
        measurementsdata: List[Dict], keys: List[str] = []
    ) -> Tuple[List, List, List]:
        ydata = []
        xdata = []
        labels = []
        for m in measurementsdata:
            data = m
            for k in keys:
                if k in data.keys():
                    data = data[k]
                else:
                    data = None
                    break
            if data is None:
                continue

            if sum(data) == 0:
                continue

            ydata.append(data)
            xdata.append(m["profiler_timestamps"])
            labels.append(m["model_name"])

        return xdata, ydata, labels

    KLogger.info("Running comparison_renode_stats_report")

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }

    metrics = compute_renode_metrics(measurementsdata)

    # opcode counter barplot
    if "sorted_opcode_counters" in metrics:
        opcode_counters = metrics["sorted_opcode_counters"]
        instr_barplot_path = imgdir / "instr_barplot_comparison"

        Barplot(
            title="Instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=opcode_counters["opcodes"],
            y_data=opcode_counters["counters"],
            colors=colors,
            max_bars_matplotlib=32,
        ).plot(instr_barplot_path, image_formats)
        report_variables["instrbarpath"] = get_plot_wildcard_path(
            instr_barplot_path, root_dir
        )

    # vector opcode counter barplot
    if "sorted_vector_opcode_counters" in metrics:
        vector_opcode_counters = metrics["sorted_vector_opcode_counters"]
        vector_instr_barplot_path = imgdir / "vector_instr_barplot_comparison"

        Barplot(
            title="Vector instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=vector_opcode_counters["opcodes"],
            y_data=vector_opcode_counters["counters"],
            colors=colors,
            max_bars_matplotlib=32,
        ).plot(vector_instr_barplot_path, image_formats)

        report_variables["vectorinstrbarpath"] = get_plot_wildcard_path(
            vector_instr_barplot_path, root_dir
        )

    # executed instructions plot
    report_variables["executedinstrplotpath"] = {}

    all_cpus = set()

    for data in measurementsdata:
        all_cpus = all_cpus.union(data["executed_instructions"].keys())

    for cpu in all_cpus:
        xdata, ydata, labels = retrieve_non_zero_profiler_data(
            measurementsdata, ["executed_instructions", cpu]
        )

        paths = {}

        executed_instructions_plot_path = (
            imgdir / f"executed_instructions_{cpu}_plot_comparison"
        )

        render_multiple_time_series_plot(
            ydatas=[ydata],
            xdatas=[xdata],
            title=f"Executed instructions for {cpu} comparison"
            if draw_titles
            else None,
            subtitles=None,
            xtitles=["Interval timestamp"],
            xunits=["s"],
            ytitles=["Executed instructions"],
            yunits=["1/s"],
            legend_labels=labels,
            outpath=executed_instructions_plot_path,
            outputext=image_formats,
            **SERVIS_PLOT_OPTIONS,
        )
        paths["persec"] = get_plot_wildcard_path(
            executed_instructions_plot_path, root_dir
        )

        cum_executed_instructions_plot_path = (
            imgdir / f"cumulative_executed_instructions_{cpu}_plot_comparison"
        )

        LinePlot(
            title=f"Executed instructions for {cpu}" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total executed instructions",
            lines=[(x, np.cumsum(y)) for x, y in zip(xdata, ydata)],
            lines_labels=labels,
            colors=SERVIS_PLOT_OPTIONS["colormap"],
        ).plot(cum_executed_instructions_plot_path, image_formats)
        paths["cumulative"] = get_plot_wildcard_path(
            cum_executed_instructions_plot_path, root_dir
        )

        if "executedinstrplotpath" not in report_variables:
            report_variables["executedinstrplotpath"] = {}

        report_variables["executedinstrplotpath"][cpu] = paths

    # memory accesses plot
    if any(("memory_accesses" in data for data in measurementsdata)):
        for access_type in ("read", "write"):
            paths = {}

            memory_access_plot_path = (
                imgdir / f"memory_{access_type}s_plot_comparison"
            )

            render_multiple_time_series_plot(
                ydatas=[
                    [m["memory_accesses"]["read"] for m in measurementsdata]
                ],
                xdatas=[[m["profiler_timestamps"] for m in measurementsdata]],
                title="Memory reads comparison" if draw_titles else None,
                subtitles=None,
                xtitles=["Interval timestamp"],
                xunits=["s"],
                ytitles=["Memory reads"],
                yunits=["1/s"],
                legend_labels=[m["model_name"] for m in measurementsdata],
                outpath=memory_access_plot_path,
                outputext=image_formats,
                **SERVIS_PLOT_OPTIONS,
            )
            paths["persec"] = get_plot_wildcard_path(
                memory_access_plot_path, root_dir
            )

            cum_memory_access_plot_path = (
                imgdir / f"cumulative_memory_{access_type}s_plot_comparison"
            )

            LinePlot(
                title=f"Memory {access_type}s" if draw_titles else None,
                x_label="Interval timestamp",
                x_unit="s",
                y_label=f"Total memory {access_type}s",
                lines=[
                    (
                        m["profiler_timestamps"],
                        np.cumsum(m["memory_accesses"][access_type]),
                    )
                    for m in measurementsdata
                ],
                lines_labels=[m["model_name"] for m in measurementsdata],
                colors=SERVIS_PLOT_OPTIONS["colormap"],
            ).plot(cum_memory_access_plot_path, image_formats)
            paths["cumulative"] = get_plot_wildcard_path(
                cum_memory_access_plot_path, root_dir
            )

            if "memoryaccessesplotpath" not in report_variables:
                report_variables["memoryaccessesplotpath"] = {}

            report_variables["memoryaccessesplotpath"][access_type] = paths

    # peripheral accesses plot
    report_variables["peripheralaccessesplotpath"] = {}

    all_peripherals = set()

    for data in measurementsdata:
        all_peripherals = all_peripherals.union(
            data["peripheral_accesses"].keys()
        )

    for peripheral in all_peripherals:
        paths = defaultdict(dict)

        for access_type in ("read", "write"):
            xdata, ydata, labels = retrieve_non_zero_profiler_data(
                measurementsdata,
                ["peripheral_accesses", peripheral, access_type],
            )

            if not len(ydata):
                continue

            peripheral_access_plot_path = (
                imgdir / f"{peripheral}_{access_type}s_plot_comparison"
            )

            render_multiple_time_series_plot(
                ydatas=[ydata],
                xdatas=[xdata],
                title=f"{peripheral} reads comparison"
                if draw_titles
                else None,
                subtitles=None,
                xtitles=["Interval timestamp"],
                xunits=["s"],
                ytitles=[f"{peripheral} {access_type}s"],
                yunits=["1/s"],
                legend_labels=labels,
                outpath=peripheral_access_plot_path,
                outputext=image_formats,
                **SERVIS_PLOT_OPTIONS,
            )
            paths[access_type]["persec"] = get_plot_wildcard_path(
                peripheral_access_plot_path, root_dir
            )

            cum_peripheral_access_plot_path = (
                imgdir
                / f"cumulative_{peripheral}_{access_type}s_plot_comparison"
            )

            LinePlot(
                title=f"{peripheral} {access_type}s" if draw_titles else None,
                x_label="Interval timestamp",
                x_unit="s",
                y_label=f"Total {peripheral} {access_type}s",
                lines=[(x, np.cumsum(y)) for x, y in zip(xdata, ydata)],
                lines_labels=labels,
                colors=SERVIS_PLOT_OPTIONS["colormap"],
            ).plot(cum_peripheral_access_plot_path, image_formats)
            paths[access_type]["cumulative"] = get_plot_wildcard_path(
                cum_peripheral_access_plot_path, root_dir
            )

        if len(paths):
            report_variables["peripheralaccessesplotpath"][peripheral] = paths

    # exceptions plot
    xdata, ydata, labels = retrieve_non_zero_profiler_data(
        measurementsdata, ["exceptions"]
    )

    if len(ydata):
        paths = {}

        exceptions_plot_path = imgdir / "exceptions_plot_comparison"

        render_multiple_time_series_plot(
            ydatas=[ydata],
            xdatas=[xdata],
            title="Exceptions comparison" if draw_titles else None,
            subtitles=None,
            xtitles=["Interval timestamp"],
            xunits=["s"],
            ytitles=["Exceptions count"],
            yunits="1/s",
            legend_labels=labels,
            outpath=exceptions_plot_path,
            outputext=image_formats,
            **SERVIS_PLOT_OPTIONS,
        )
        paths["persec"] = get_plot_wildcard_path(
            exceptions_plot_path, root_dir
        )

        cum_exceptions_plot_path = (
            imgdir / "cumulative_exceptions_plot_comparison"
        )

        LinePlot(
            title="Total exceptions" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total exceptions",
            lines=[
                (
                    m["profiler_timestamps"],
                    np.cumsum(m["exceptions"]),
                )
                for m in measurementsdata
            ],
            lines_labels=labels,
            colors=SERVIS_PLOT_OPTIONS["colormap"],
        ).plot(cum_exceptions_plot_path, image_formats)

        paths["cumulative"] = get_plot_wildcard_path(
            cum_exceptions_plot_path, root_dir
        )

        report_variables["exceptionsplotpath"] = paths

    with path(reports, "renode_stats_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )
