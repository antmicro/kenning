# Copyright (c) 2025-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for performance report generation.
"""

import copy
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import numpy as np

from kenning.core.drawing import SERVIS_PLOT_OPTIONS
from kenning.core.metrics import compute_performance_metrics
from kenning.core.report import PerformanceReportScope
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger

FULL_NAME: Dict[str, str] = {
    "cpu_mem": "RAM",
    "cpu": "CPU",
    "gpu_mem": "GPU memory",
    "gpu": "GPU",
}


def generate_resource_usage_plot(
    scope: PerformanceReportScope,
    resource_name: str,
    meas: Dict[str, Any],
    root_dir: Path,
    img_dir: Path,
    imgprefix: str,
    draw_titles: bool,
    image_formats: Set[str],
    plot_options: Dict[str, Any],
) -> bool:
    """
    Creates a usage plot in performance section of the report.

    Parameters
    ----------
    scope : PerformanceReportScope
        Resource usage metrics collection scope.
    resource_name : str
        Name of resource, which measurements should be plotted.
    meas : Dict[str, Any]
        Statistics from the Measurements class.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    img_dir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    draw_titles : bool
        Should titles be drawn on the plot.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    plot_options : Dict[str, Any]
        Additional plot configuration.

    Returns
    -------
    bool
        True if plot was generated successfully.
    """
    import servis

    prefix = "session_utilization"
    device_name = resource_name.split("_")[0]
    scope_str = scope.value
    plot_path = img_dir / f"{imgprefix}{resource_name}_usage"
    title = f"{FULL_NAME[resource_name]} utilization"

    if scope == PerformanceReportScope.ALL:
        if f"{prefix}_system_{resource_name}_percent" not in meas:
            KLogger.warning(
                f"System {FULL_NAME[resource_name]} usage is not present in"
                " measurements."
            )
            return False
        if f"{prefix}_process_{resource_name}_percent" not in meas:
            KLogger.warning(
                f"Process {FULL_NAME[resource_name]} usage is not present in"
                " measurements. Falling back to system"
                f" {FULL_NAME[resource_name]} usage."
            )
            return generate_resource_usage_plot(
                PerformanceReportScope.SYSTEM,
                resource_name,
                meas,
                root_dir,
                img_dir,
                imgprefix,
                draw_titles,
                image_formats,
                plot_options,
            )

        if device_name == "gpu":
            if not len(meas[f"{prefix}_system_gpu_percent"]):
                KLogger.warning(
                    "Incorrectly collected data for system GPU usage."
                )
                return False
            if not len(meas[f"{prefix}_process_gpu_percent"]):
                KLogger.warning(
                    "Incorrectly collected data for process GPU usage."
                )
                return False
        KLogger.info(
            f"Using system and process {FULL_NAME[resource_name]}"
            " usage measurements."
        )

        servis.render_multiple_time_series_plot(
            ydatas=[
                [
                    meas[f"{prefix}_system_{resource_name}_percent"],
                    meas[f"{prefix}_process_{resource_name}_percent"],
                ]
            ],
            xdatas=[
                [
                    meas[f"{prefix}_system_{device_name}_timestamp"],
                    meas[f"{prefix}_process_{device_name}_timestamp"],
                ]
            ],
            subtitles=[""],
            title=title if draw_titles else "",
            xtitles=["Time"],
            xunits=["s"],
            ytitles=[title],
            yunits=["%"],
            outpath=plot_path,
            skipfirst=True,
            outputext=list(image_formats),
            legend_labels=["System", "Process"],
            **plot_options,
        )
    else:
        if (
            scope == PerformanceReportScope.SYSTEM
            and f"{prefix}_{scope_str}_{resource_name}_percent" not in meas
        ):
            KLogger.warning(
                f"{scope_str.capitalize()} {FULL_NAME[resource_name]} usage is"
                " not present in measurements."
            )
            return False

        if (
            scope == PerformanceReportScope.PROCESS
            and f"{prefix}_{scope_str}_{resource_name}_percent" not in meas
        ):
            KLogger.warning(
                f"Process {FULL_NAME[resource_name]} usage is not present"
                " in measurements. Falling back to system"
                f" {FULL_NAME[resource_name]} usage."
            )
            return generate_resource_usage_plot(
                PerformanceReportScope.SYSTEM,
                resource_name,
                meas,
                root_dir,
                img_dir,
                imgprefix,
                draw_titles,
                image_formats,
                plot_options,
            )

        if device_name == "gpu":
            if not len(meas[f"{prefix}_{scope_str}_gpu_percent"]):
                KLogger.warning(
                    f"Incorrectly collected data for {scope_str} GPU usage."
                )
                return False

        KLogger.info(
            f"Using {scope_str} {FULL_NAME[resource_name]} usage measurements."
        )
        servis.render_time_series_plot_with_histogram(
            ydata=meas[f"{prefix}_{scope_str}_{resource_name}_percent"],
            xdata=meas[f"{prefix}_{scope_str}_{device_name}_timestamp"],
            title=title if draw_titles else "",
            xtitle="Time",
            xunit="s",
            ytitle=title,
            yunit="%",
            outpath=plot_path,
            skipfirst=True,
            outputext=list(image_formats),
            **plot_options,
        )

    meas[f"{resource_name}_{scope_str}_usage_path"] = get_plot_wildcard_path(
        plot_path, root_dir
    )
    return True


def performance_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    draw_titles: bool = True,
    performance_report_scope: PerformanceReportScope = (
        PerformanceReportScope.PROCESS
    ),
    **kwargs: Any,
) -> Tuple[str, Dict]:
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
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    color_offset : int
        How many colors from default color list should be skipped.
    draw_titles : bool
        Should titles be drawn on the plot.
    performance_report_scope : PerformanceReportScope
        Resource usage metrics collection scope.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Tuple[str, Dict]
        Content of the report in MyST format, a dict of measurements.
    """
    import servis

    KLogger.info(
        f'Running performance_report for {measurementsdata["model_name"]}'
    )

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
        KLogger.warning("No inference time measurements in the report.")

    if cycles := measurementsdata.get("cycles", []):
        cycles = np.asarray(cycles)
        plot_path = imgdir / f"{imgprefix}cpu_memory_usage"
        servis.render_time_series_plot_with_histogram(
            ydata=cycles,
            xdata=list(range(cycles.shape[0])),
            title="Cycles for inference" if draw_titles else None,
            xtitle="Dispatch",
            ytitle="Cycles",
            outpath=str(plot_path),
            skipfirst=False,
            outputext=image_formats,
            **plot_options,
        )
        measurementsdata["cycles_series_plot"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

        total_number_of_cycles = np.sum(cycles)
        if inference_step:
            inference_count = len(measurementsdata[inference_step])
            mean_cycles = total_number_of_cycles / inference_count
            measurementsdata["mean_cycles"] = mean_cycles
        measurementsdata["total_cycles"] = total_number_of_cycles

        # If cycles are available, other metrics do not matter
        with path(reports, "cycles.md") as reporttemplate:
            return create_report_from_measurements(
                reporttemplate, measurementsdata
            ), {}

    measurementsdata[
        "performance_report_scope"
    ] = performance_report_scope.value
    metrics = compute_performance_metrics(measurementsdata)
    measurementsdata |= metrics

    if inference_step:
        plot_path = imgdir / f"{imgprefix}inference_time"
        servis.render_time_series_plot_with_histogram(
            ydata=measurementsdata[inference_step],
            xdata=measurementsdata[f"{inference_step}_timestamp"],
            title="Inference time" if draw_titles else "",
            xtitle="Time",
            xunit="s",
            ytitle="Inference time",
            yunit="s",
            outpath=plot_path,
            skipfirst=True,
            outputext=list(image_formats),
            **plot_options,
        )

        measurementsdata["inferencetimepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

        measurementsdata["inferencetime"] = measurementsdata[inference_step]

    if (
        "input_preprocess_step" in measurementsdata
        and "output_postprocess_step" in measurementsdata
    ):
        title = "Preprocessing and postprocessing time"
        plot_path = imgdir / f"{imgprefix}_pre_post_process"
        servis.render_multiple_time_series_plot(
            ydatas=[
                [
                    measurementsdata["input_preprocess_step"],
                    measurementsdata["output_postprocess_step"],
                ]
            ],
            xdatas=[
                [
                    measurementsdata["input_preprocess_step_timestamp"],
                    measurementsdata["output_postprocess_step_timestamp"],
                ]
            ],
            subtitles=[""],
            title=title if draw_titles else "",
            xtitles=["Time"],
            xunits=["s"],
            ytitles=["Processing time"],
            yunits=["s"],
            outpath=plot_path,
            skipfirst=True,
            outputext=list(image_formats),
            legend_labels=["Preprocess", "Postprocess"],
            **plot_options,
        )

        measurementsdata["pre_post_process_path"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

    generate_resource_usage_plot(
        performance_report_scope,
        "cpu_mem",
        measurementsdata,
        root_dir,
        imgdir,
        imgprefix,
        draw_titles,
        image_formats,
        plot_options,
    )

    generate_resource_usage_plot(
        performance_report_scope,
        "cpu",
        measurementsdata,
        root_dir,
        imgdir,
        imgprefix,
        draw_titles,
        image_formats,
        plot_options,
    )

    generate_resource_usage_plot(
        performance_report_scope,
        "gpu_mem",
        measurementsdata,
        root_dir,
        imgdir,
        imgprefix,
        draw_titles,
        image_formats,
        plot_options,
    )

    generate_resource_usage_plot(
        performance_report_scope,
        "gpu",
        measurementsdata,
        root_dir,
        imgdir,
        imgprefix,
        draw_titles,
        image_formats,
        plot_options,
    )

    with path(reports, "performance.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        ), metrics
