# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for performance report generation.
"""

import copy
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, Set

from kenning.core.metrics import compute_performance_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def performance_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates performance section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
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
    color_offset : int
        How many colors from default color list should be skipped.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_time_series_plot_with_histogram

    from kenning.core.drawing import SERVIS_PLOT_OPTIONS

    KLogger.info(
        f'Running performance_report for {measurementsdata["model_name"]}'
    )
    metrics = compute_performance_metrics(measurementsdata)
    measurementsdata |= metrics

    # Shifting colors to match color_offset
    plot_options = copy.deepcopy(SERVIS_PLOT_OPTIONS)
    plot_options["colormap"] = plot_options["colormap"][color_offset:]
    if plot_options["backend"] == "bokeh":
        plot_options["figsize"] = "responsive"

    inference_step = None
    if "target_inference_step" in measurementsdata:
        KLogger.info("Using target measurements for inference time")
        inference_step = "target_inference_step"
    elif "protocol_inference_step" in measurementsdata:
        KLogger.info("Using protocol measurements for inference time")
        inference_step = "protocol_inference_step"
    else:
        KLogger.warning("No inference time measurements in the report")

    if inference_step:
        plot_path = imgdir / f"{imgprefix}inference_time"
        render_time_series_plot_with_histogram(
            ydata=measurementsdata[inference_step],
            xdata=measurementsdata[f"{inference_step}_timestamp"],
            title="Inference time" if draw_titles else None,
            xtitle="Time",
            xunit="s",
            ytitle="Inference time",
            yunit="s",
            outpath=str(plot_path),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        measurementsdata["inferencetimepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

        measurementsdata["inferencetime"] = measurementsdata[inference_step]

    if "session_utilization_mem_percent" in measurementsdata:
        KLogger.info("Using target measurements memory usage percentage")
        plot_path = imgdir / f"{imgprefix}cpu_memory_usage"
        render_time_series_plot_with_histogram(
            ydata=measurementsdata["session_utilization_mem_percent"],
            xdata=measurementsdata["session_utilization_timestamp"],
            title="Memory usage" if draw_titles else None,
            xtitle="Time",
            xunit="s",
            ytitle="Memory usage",
            yunit="%",
            outpath=str(plot_path),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        measurementsdata["memusagepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )
    else:
        KLogger.warning("No memory usage measurements in the report")

    if "session_utilization_cpus_percent" in measurementsdata:
        KLogger.info("Using target measurements CPU usage percentage")
        plot_path = imgdir / f"{imgprefix}cpu_usage"
        render_time_series_plot_with_histogram(
            ydata=measurementsdata["session_utilization_cpus_percent_avg"],
            xdata=measurementsdata["session_utilization_timestamp"],
            title="Average CPU usage" if draw_titles else None,
            xtitle="Time",
            xunit="s",
            ytitle="Average CPU usage",
            yunit="%",
            outpath=str(plot_path),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        measurementsdata["cpuusagepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )
    else:
        KLogger.warning("No CPU usage measurements in the report")

    if "session_utilization_gpu_mem_utilization" in measurementsdata:
        KLogger.info("Using target measurements GPU memory usage percentage")
        plot_path = imgdir / f"{imgprefix}gpu_memory_usage"
        gpumemmetric = "session_utilization_gpu_mem_utilization"
        if len(measurementsdata[gpumemmetric]) == 0:
            KLogger.warning(
                "Incorrectly collected data for GPU memory utilization"
            )
        else:
            render_time_series_plot_with_histogram(
                ydata=measurementsdata[gpumemmetric],
                xdata=measurementsdata["session_utilization_gpu_timestamp"],
                title="GPU memory usage" if draw_titles else None,
                xtitle="Time",
                xunit="s",
                ytitle="GPU memory usage",
                yunit="%",
                outpath=str(plot_path),
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            measurementsdata["gpumemusagepath"] = get_plot_wildcard_path(
                plot_path, root_dir
            )
    else:
        KLogger.warning("No GPU memory usage measurements in the report")

    if "session_utilization_gpu_utilization" in measurementsdata:
        KLogger.info("Using target measurements GPU utilization")
        plot_path = imgdir / f"{imgprefix}gpu_usage"
        if len(measurementsdata["session_utilization_gpu_utilization"]) == 0:
            KLogger.warning("Incorrectly collected data for GPU utilization")
        else:
            render_time_series_plot_with_histogram(
                ydata=measurementsdata["session_utilization_gpu_utilization"],
                xdata=measurementsdata["session_utilization_gpu_timestamp"],
                title="GPU Utilization" if draw_titles else None,
                xtitle="Time",
                xunit="s",
                ytitle="Utilization",
                yunit="%",
                outpath=str(plot_path),
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            measurementsdata["gpuusagepath"] = get_plot_wildcard_path(
                plot_path, root_dir
            )
    else:
        KLogger.warning("No GPU utilization measurements in the report")

    with path(reports, "performance.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        ), metrics
