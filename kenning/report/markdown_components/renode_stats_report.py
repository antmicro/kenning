# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for renode statistics report generation.
"""

import copy
from collections import defaultdict
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from kenning.core.metrics import compute_renode_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def renode_stats_report(
    measurementsdata: Dict,
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    draw_titles: bool = True,
    colors: Optional[List] = None,
    color_offset: int = 0,
    **kwargs: Any,
) -> str:
    """
    Creates Renode stats section of the report.

    Parameters
    ----------
    measurementsdata : Dict
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    root_dir : Path
        Path to the root of the documentation project
        involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    colors : Optional[List]
        Colors used for plots.
    color_offset : int
        How many colors from default color list should be skipped.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_time_series_plot_with_histogram

    from kenning.core.drawing import SERVIS_PLOT_OPTIONS, Barplot, LinePlot

    KLogger.info(
        f'Running renode_stats_report for {measurementsdata["model_name"]}'
    )
    # shift colors to match color_offset
    plot_options = copy.deepcopy(SERVIS_PLOT_OPTIONS)
    plot_options["colormap"] = plot_options["colormap"][color_offset:]

    metrics = compute_renode_metrics([measurementsdata])
    measurementsdata |= metrics

    # opcode counter barplot
    if "sorted_opcode_counters" in measurementsdata:
        opcode_counters = measurementsdata["sorted_opcode_counters"]
        instr_barplot_path = imgdir / f"{imgprefix}instr_barplot"

        Barplot(
            title="Instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=opcode_counters["opcodes"],
            y_data=opcode_counters["counters"],
            colors=colors,
            color_offset=color_offset,
            max_bars_matplotlib=32,
        ).plot(instr_barplot_path, image_formats)
        measurementsdata["instrbarpath"] = get_plot_wildcard_path(
            instr_barplot_path, root_dir
        )

    # vector opcode counter barplot
    if "sorted_vector_opcode_counters" in measurementsdata:
        vector_opcode_counters = measurementsdata[
            "sorted_vector_opcode_counters"
        ]
        vector_instr_barplot_path = imgdir / f"{imgprefix}vector_instr_barplot"

        Barplot(
            title="Vector instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=vector_opcode_counters["opcodes"],
            y_data=vector_opcode_counters["counters"],
            colors=colors,
            color_offset=color_offset,
            max_bars_matplotlib=32,
        ).plot(vector_instr_barplot_path, image_formats)
        measurementsdata["vectorinstrbarpath"] = get_plot_wildcard_path(
            vector_instr_barplot_path, root_dir
        )

    # executed instructions plot
    for cpu, data in measurementsdata.get("executed_instructions", {}).items():
        paths = {}

        executed_instructions_plot_path = (
            imgdir / f"{imgprefix}executed_instructions_{cpu}_plot"
        )

        render_time_series_plot_with_histogram(
            ydata=data,
            xdata=measurementsdata["profiler_timestamps"],
            title=f"Executed instructions for {cpu}" if draw_titles else None,
            xtitle="Interval timestamp",
            xunit="s",
            ytitle="Executed instructions",
            yunit="1/s",
            outpath=executed_instructions_plot_path,
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )
        paths["persec"] = get_plot_wildcard_path(
            executed_instructions_plot_path, root_dir
        )

        cum_executed_instructions_plot_path = (
            imgdir / f"{imgprefix}cumulative_executed_instructions_{cpu}_plot"
        )

        LinePlot(
            lines=[(measurementsdata["profiler_timestamps"], np.cumsum(data))],
            title=f"Executed instructions for {cpu}" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total executed instructions",
            colors=plot_options["colormap"],
        ).plot(cum_executed_instructions_plot_path, image_formats)
        paths["cumulative"] = get_plot_wildcard_path(
            cum_executed_instructions_plot_path, root_dir
        )

        if "executedinstrplotpath" not in measurementsdata:
            measurementsdata["executedinstrplotpath"] = {}

        measurementsdata["executedinstrplotpath"][cpu] = paths

    # memory accesses plot
    for access_type in ("read", "write"):
        if "memory_access" not in measurementsdata or not len(
            measurementsdata["memory_accesses"][access_type]
        ):
            continue

        paths = {}

        memory_access_plot_path = (
            imgdir / f"{imgprefix}memory_{access_type}s_plot"
        )

        render_time_series_plot_with_histogram(
            ydata=measurementsdata["memory_accesses"][access_type],
            xdata=measurementsdata["profiler_timestamps"],
            title=f"Memory {access_type}s" if draw_titles else None,
            xtitle="Interval timestamp",
            xunit="s",
            ytitle=f"Memory {access_type}s",
            yunit="1/s",
            outpath=memory_access_plot_path,
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )
        paths["persec"] = get_plot_wildcard_path(
            memory_access_plot_path, root_dir
        )

        cum_memory_access_plot_path = (
            imgdir / f"{imgprefix}cumulative_memory_{access_type}s_plot"
        )

        LinePlot(
            title=f"Memory {access_type}s" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label=f"Total memory {access_type}s",
            lines=[
                (
                    measurementsdata["profiler_timestamps"],
                    np.cumsum(
                        measurementsdata["memory_accesses"][access_type]
                    ),
                )
            ],
            colors=plot_options["colormap"],
        ).plot(cum_memory_access_plot_path, image_formats)
        paths["cumulative"] = get_plot_wildcard_path(
            cum_memory_access_plot_path, root_dir
        )

        if "memoryaccessesplotpath" not in measurementsdata:
            measurementsdata["memoryaccessesplotpath"] = {}

        measurementsdata["memoryaccessesplotpath"][access_type] = paths

    # peripheral accesses plot
    for peripheral, measurements in measurementsdata.get(
        "peripheral_accesses", {}
    ).items():
        paths = defaultdict(dict)

        for access_type in ("read", "write"):
            if not sum(measurements[access_type]):
                continue

            peripheral_access_plot_path = (
                imgdir / f"{imgprefix}_{peripheral}_{access_type}s_plot"
            )

            render_time_series_plot_with_histogram(
                ydata=measurements[access_type],
                xdata=measurementsdata["profiler_timestamps"],
                title=f"{peripheral} {access_type}s" if draw_titles else None,
                xtitle="Interval timestamp",
                xunit="s",
                ytitle=f"{peripheral} {access_type}s",
                yunit="1/s",
                outpath=peripheral_access_plot_path,
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            paths[access_type]["persec"] = get_plot_wildcard_path(
                peripheral_access_plot_path, root_dir
            )

            cum_peripheral_access_plot_path = (
                imgdir
                / f"{imgprefix}cumulative_{peripheral}_{access_type}s_plot"
            )

            LinePlot(
                title=f"{peripheral} {access_type}s" if draw_titles else None,
                x_label="Interval timestamp",
                x_unit="s",
                y_label=f"Total {peripheral} {access_type}s",
                lines=[
                    (
                        measurementsdata["profiler_timestamps"],
                        np.cumsum(measurements[access_type]),
                    )
                ],
                colors=plot_options["colormap"],
            ).plot(cum_peripheral_access_plot_path, image_formats)
            paths[access_type]["cumulative"] = get_plot_wildcard_path(
                cum_peripheral_access_plot_path, root_dir
            )

        if len(paths):
            if "peripheralaccessesplotpath" not in measurementsdata:
                measurementsdata["peripheralaccessesplotpath"] = {}
            measurementsdata["peripheralaccessesplotpath"][peripheral] = paths

    # exceptions plot
    if "exceptions" in measurementsdata and sum(
        measurementsdata["exceptions"]
    ):
        paths = {}

        exceptions_plot_path = imgdir / f"{imgprefix}exceptions_plot"

        render_time_series_plot_with_histogram(
            ydata=measurementsdata["exceptions"],
            xdata=measurementsdata["profiler_timestamps"],
            title="Exceptions" if draw_titles else None,
            xtitle="Interval timestamp",
            xunit="s",
            ytitle="Exceptions count",
            yunit="1/s",
            outpath=exceptions_plot_path,
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )
        paths["persec"] = get_plot_wildcard_path(
            exceptions_plot_path, root_dir
        )
        cum_exceptions_plot_path = (
            imgdir / f"{imgprefix}cumulative_exceptions_plot"
        )

        LinePlot(
            title="Total xceptions" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total exceptions",
            y_unit=None,
            lines=[
                (
                    measurementsdata["profiler_timestamps"],
                    np.cumsum(measurementsdata["exceptions"]),
                )
            ],
            colors=plot_options["colormap"],
        ).plot(cum_exceptions_plot_path, image_formats)

        paths["cumulative"] = get_plot_wildcard_path(
            cum_exceptions_plot_path, root_dir
        )

        measurementsdata["exceptionsplotpath"] = paths

    with path(reports, "renode_stats.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        ), metrics
