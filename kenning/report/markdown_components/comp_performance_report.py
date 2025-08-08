# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for performance comparison report generation.
"""

import copy
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from kenning.core.metrics import compute_performance_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def comparison_performance_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates performance comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project
        involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_multiple_time_series_plot

    from kenning.core.drawing import (
        SERVIS_PLOT_OPTIONS,
        RadarChart,
        ViolinComparisonPlot,
    )

    KLogger.info("Running comparison_performance_report")

    metric_names = {
        "inference_step": ("Inference time", "s"),
        "session_utilization_mem_percent": ("Memory usage", "%"),
        "session_utilization_cpus_percent_avg": ("CPU usage", "%"),
        "session_utilization_gpu_mem_utilization": (
            "GPU memory usage",
            "%",
        ),
        "session_utilization_gpu_utilization": ("GPU usage", "%"),
    }
    common_metrics = set(metric_names.keys())
    hardware_usage_metrics = common_metrics - {"inference_step"}
    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }
    names = [data["model_name"] for data in measurementsdata]
    report_variables["model_names"] = names

    plot_options = copy.deepcopy(SERVIS_PLOT_OPTIONS)
    if plot_options["backend"] == "bokeh":
        plot_options["figsize"] = "responsive"

    available_metrics = None
    for data in measurementsdata:
        performance_metrics = compute_performance_metrics(data)
        report_variables[data["model_name"]] = performance_metrics

        metrics = set(performance_metrics.keys())
        if available_metrics is None:
            available_metrics = metrics

        if "target_inference_step" in data:
            data["inference_step"] = data["target_inference_step"]
            data["inference_step_timestamp"] = data[
                "target_inference_step_timestamp"
            ]
        elif "protocol_inference_step" in data:
            data["inference_step"] = data["protocol_inference_step"]
            data["inference_step_timestamp"] = data[
                "protocol_inference_step_timestamp"
            ]

        if "session_utilization_cpus_percent" in data:
            data["session_utilization_cpus_percent_avg"] = performance_metrics[
                "session_utilization_cpus_percent_avg"
            ]

        gpumetrics = [
            "session_utilization_gpu_mem_utilization",
            "session_utilization_gpu_utilization",
        ]
        for gpumetric in gpumetrics:
            if gpumetric in data and len(data[gpumetric]) == 0:
                del data[gpumetric]

        modelmetrics = set(data.keys())
        common_metrics &= modelmetrics
    if "session_utilization_cpus_percent_avg" in available_metrics:
        available_metrics.remove("session_utilization_cpus_percent_avg")
    report_variables["available_metrics"] = available_metrics

    for metric, (metric_name, unit) in metric_names.items():
        metric_data = {}
        if metric_name == "Inference time":
            timestamp_key = "inference_step_timestamp"
        elif metric_name in ("GPU usage", "GPU memory usage"):
            timestamp_key = "session_utilization_gpu_timestamp"
        else:
            timestamp_key = "session_utilization_timestamp"
        if timestamp_key not in data:
            KLogger.warning(
                f'Missing measurement "{timestamp_key}" \
                in the measurements '
                f"file. Can't provide benchmarks for {metric_name}"
            )
            continue
        timestamps = {
            data["model_name"]: data[timestamp_key]
            for data in measurementsdata
        }

        for data in measurementsdata:
            if metric in data:
                metric_data[data["model_name"]] = data[metric]
        if len(metric_data) > 1:
            plot_path = imgdir / f"{metric}_comparison"
            render_multiple_time_series_plot(
                ydatas=[list(metric_data.values())],
                xdatas=[list(timestamps.values())],
                title=f"{metric_name} comparison" if draw_titles else None,
                subtitles=None,
                xtitles=["Time"],
                xunits=["s"],
                ytitles=[metric_name],
                yunits=[unit],
                legend_labels=list(metric_data.keys()),
                outpath=plot_path,
                outputext=image_formats,
                **plot_options,
            )
            report_variables[f"{metric}_path"] = get_plot_wildcard_path(
                plot_path, root_dir
            )

    common_metrics = sorted(list(common_metrics))
    visualizationdata = {}
    for data in measurementsdata:
        visualizationdata[data["model_name"]] = [
            data[metric] for metric in common_metrics
        ]

    plot_path = imgdir / "mean_performance_comparison"
    ViolinComparisonPlot(
        title="Performance comparison plot" if draw_titles else None,
        metric_data=visualizationdata,
        metric_labels=[
            f"{metric_names[metric][0]} [{metric_names[metric][1]}]"
            for metric in common_metrics
        ],
        colors=colors,
    ).plot(plot_path, image_formats)
    report_variables["meanperformancepath"] = get_plot_wildcard_path(
        plot_path, root_dir
    )

    hardware_usage_metrics = sorted(list(hardware_usage_metrics))
    measurements_metrics = set()
    for data in measurementsdata:
        measurements_metrics = measurements_metrics.union(data.keys())

    if set(hardware_usage_metrics).intersection(measurements_metrics):
        usage_visualization = {}
        for data in measurementsdata:
            usage_visualization[data["model_name"]] = [
                np.mean(data.get(metric, 0)) / 100
                for metric in hardware_usage_metrics
            ]

        plot_path = imgdir / "hardware_usage_comparison"
        RadarChart(
            title="Resource usage comparison" if draw_titles else None,
            metric_data=usage_visualization,
            metric_labels=[
                metric_names[metric][0] for metric in hardware_usage_metrics
            ],
            colors=colors,
        ).plot(plot_path, image_formats)
        report_variables["hardwareusagepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

    with path(reports, "performance_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )
