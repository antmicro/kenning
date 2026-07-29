# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for generic quality comparison report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from kenning.core.metrics import (
    Metric,
    compute_performance_metrics,
)
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def comparison_generic_quality_report(
    type_to_display: str,
    compute_metric_func: Callable[[Dict], Dict[str, List]],
    default_quality_metric: Metric,
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    main_quality_metric: Optional[Metric] = None,
    metrics_for_radar: Optional[List[Metric]] = None,
    **kwargs: Any,
) -> str:
    """
    Creates a generic model quality comparison section of report. Plots
    different quality metrics and their relation to model performance.

    Parameters
    ----------
    type_to_display: str
        Report type that will be displayed in the section header, e.g.
        "Regression" or "Depth estimation".
    compute_metric_func : Callable[[Dict], Dict[str, List]]
        Function used to obtain metrics from raw measurements.
    default_quality_metric : Metric
        The metric to be used as the main quality metric if user requests the
        default one (passes None to main_quality_metric).
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    main_quality_metric : Optional[Metric]
        Metric presented on Y-axis on bubble plot.
    metrics_for_radar : Optional[List[Metric]]
        List of metrics to use for radar plot. By default, all available
        metrics are used.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    report_variables, _ = _create_comparison_generic_quality_report(**locals())

    with path(reports, "generic_quality_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )


def _create_comparison_generic_quality_report(
    type_to_display: str,
    compute_metric_func: Callable[[Dict], Dict[str, List]],
    default_quality_metric: Metric,
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    main_quality_metric: Optional[Metric] = None,
    metrics_for_radar: Optional[List[Metric]] = None,
    **kwargs: Any,
) -> Tuple[Dict, bool]:
    from kenning.report.markdown_components.utils import plot_all_comparisons

    KLogger.info("Running generic_quality_report")

    if main_quality_metric is None:
        main_quality_metric = default_quality_metric

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
        "type_to_display": type_to_display,
    }
    names = [data["model_name"] for data in measurementsdata]

    plot_variables = {}
    mean_inference_times = []
    model_sizes = []
    skip_inference_metrics = False

    available_metrics = None
    max_metrics = {}

    for data in measurementsdata:
        computed_metrics = compute_metric_func(data)
        model_metrics = {}

        for metric in computed_metrics:
            model_metrics[metric] = computed_metrics[metric]
            if (
                metric not in max_metrics
                or computed_metrics[metric] > max_metrics[metric]
            ):
                max_metrics[metric] = computed_metrics[metric]

        if available_metrics is None:
            available_metrics = list(model_metrics.keys())
        else:
            # Intersection maintaining order
            available_metrics = [
                metric
                for metric in available_metrics
                if metric in model_metrics
            ]

        performance_metrics = compute_performance_metrics(data)

        if "inferencetime_mean" not in performance_metrics:
            skip_inference_metrics = True
        else:
            model_inferencetime_mean = performance_metrics[
                "inferencetime_mean"
            ]
            mean_inference_times.append(model_inferencetime_mean)
            model_metrics["inferencetime_mean"] = model_inferencetime_mean

        if "compiled_model_size" in data:
            model_sizes.append(data["compiled_model_size"])
            model_metrics["size"] = data["compiled_model_size"]
        else:
            KLogger.warning(
                "Missing information about model size in measurements"
                " - computing size based on average RAM usage"
            )
            model_sizes.append(
                performance_metrics["session_utilization_mem_percent_mean"]
            )
            model_metrics["size"] = performance_metrics[
                "session_utilization_mem_percent_mean"
            ]

        plot_variables[data["model_name"]] = model_metrics

    report_variables["available_metrics"] = available_metrics

    if not skip_inference_metrics:
        report_variables |= plot_all_comparisons(
            all_display_metrics=plot_variables,
            main_quality_metric=main_quality_metric,
            metrics_for_radar=metrics_for_radar,
            available_metrics=available_metrics,
            mean_inference_times=mean_inference_times,
            model_sizes=model_sizes,
            imgdir=imgdir,
            root_dir=root_dir,
            image_formats=image_formats,
            colors=colors,
            draw_titles=draw_titles,
        )

    report_variables["model_names"] = names
    report_variables = {
        **report_variables,
        **plot_variables,
    }

    report_variables["bubble_plot_metric"] = main_quality_metric.value
    report_variables["max_metrics"] = max_metrics

    return report_variables, skip_inference_metrics
