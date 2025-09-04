# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for classification comparison report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from kenning.core.metrics import (
    CLASSIFICATION_METRICS,
    Metric,
    compute_classification_metrics,
    compute_performance_metrics,
)
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def comparison_classification_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    main_quality_metric: Metric = Metric.ACC,
    metrics_for_radar: Optional[List[Metric]] = None,
    **kwargs: Any,
) -> str:
    """
    Creates classification comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation
        project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    main_quality_metric : Metric
        Metric presented on Y-axis on bubble plot.
    metrics_for_radar : Optional[List[Metric]]
        List of metrics to use for radar plot. By default,
        all available metrics are used.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from kenning.core.drawing import Barplot, BubblePlot, RadarChart

    KLogger.info("Running comparison_classification_report")
    # HTML plots format unsupported, removing html

    # check that each measurements have the same classes
    for data in measurementsdata:
        assert (
            measurementsdata[0]["class_names"] == data["class_names"]
        ), "Invalid class names in measurements"

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }
    names = [data["model_name"] for data in measurementsdata]

    metric_visualization = {}
    bubble_plot_data, mean_inference_time, model_sizes = [], [], []
    skip_inference_metrics = False
    available_metrics = set(CLASSIFICATION_METRICS)
    max_metrics = {}
    for data in measurementsdata:
        performance_metrics = compute_performance_metrics(data)
        if "inferencetime_mean" not in performance_metrics:
            skip_inference_metrics = True
            break

        classification_metrics = compute_classification_metrics(data)
        model_metrics = {}
        if Metric.ACC in classification_metrics:
            model_metrics["accuracy"] = classification_metrics[Metric.ACC]
        if "inferencetime_mean" in performance_metrics:
            model_metrics["inferencetime_mean"] = performance_metrics[
                "inferencetime_mean"
            ]
        metrics = []
        for metric in CLASSIFICATION_METRICS:
            if metric not in classification_metrics:
                continue
            model_metrics[metric] = classification_metrics[metric]
            metrics.append(metric)
            if (
                metric not in max_metrics
                or classification_metrics[metric] > max_metrics[metric]
            ):
                max_metrics[metric] = classification_metrics[metric]
        available_metrics = available_metrics.intersection(metrics)
        bubble_plot_data.append(model_metrics[main_quality_metric])

        model_inferencetime_mean = performance_metrics["inferencetime_mean"]
        mean_inference_time.append(model_inferencetime_mean)

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

        metric_visualization[data["model_name"]] = model_metrics

    if not skip_inference_metrics:
        if main_quality_metric not in available_metrics:
            KLogger.error(
                f"{main_quality_metric} not available"
                " for all models, using accuracy"
            )
            main_quality_metric = Metric.ACC
        plot_path = imgdir / "accuracy_vs_inference_time"
        BubblePlot(
            title=f"{main_quality_metric.value} vs Mean inference time"
            if draw_titles
            else None,
            x_data=mean_inference_time,
            x_label="Mean inference time [s]",
            y_data=[
                metric_visualization[name][main_quality_metric]
                for name in names
            ],
            y_label=main_quality_metric.value,
            size_data=model_sizes,
            size_label="Model size",
            bubble_labels=names,
            colors=colors,
        ).plot(plot_path, image_formats)
        report_variables["bubbleplotpath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

        plot_path = imgdir / "classification_metric_comparison"
        if metrics_for_radar is None:
            metrics_for_radar = available_metrics
        if not available_metrics.issuperset(metrics_for_radar):
            KLogger.error(
                f"{set(metrics_for_radar).difference(available_metrics)} "
                "are not available for all models"
            )
            metrics_for_radar = available_metrics.intersection(
                metrics_for_radar
            )
        RadarChart(
            title="Metric comparison" if draw_titles else None,
            metric_data={
                model: [metrics[metric] for metric in metrics_for_radar]
                for model, metrics in metric_visualization.items()
            },
            metric_labels=[metric.value for metric in metrics_for_radar],
            colors=colors,
        ).plot(plot_path, image_formats)
        report_variables["radarchartpath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )
        # preserve the original order
        metric_visualization["available_metrics"] = [
            metric for metric in list(Metric) if metric in available_metrics
        ]
        report_variables["model_names"] = names
        report_variables = {
            **report_variables,
            **metric_visualization,
        }

    if "predictions" in measurementsdata[0] and (
        "eval_confusion_matrix" not in measurementsdata[0]
    ):
        predictions = [measurementsdata[0]["class_names"]] + [
            data["predictions"] for data in measurementsdata
        ]
        predictions = list(zip(*predictions))
        predictions.sort(key=lambda x: (sum(x[1:]), x[0]), reverse=True)
        predictions = list(zip(*predictions))
        predictions_data = {
            name: data for name, data in zip(names, predictions[1:])
        }
        predictions_batplot_path = imgdir / "predictions"
        Barplot(
            title="Predictions barplot" if draw_titles else None,
            x_label="Class",
            y_label="Percentage",
            y_unit="%",
            x_data=predictions[0],
            y_data=predictions_data,
            colors=colors,
        ).plot(predictions_batplot_path, image_formats)
        report_variables["predictionsbarpath"] = get_plot_wildcard_path(
            predictions_batplot_path, root_dir
        )
    elif skip_inference_metrics:
        KLogger.warning(
            "No inference measurements available, "
            "skipping report generation"
        )
        return ""

    report_variables["bubble_plot_metric"] = main_quality_metric.value
    report_variables["max_metrics"] = max_metrics
    with path(reports, "classification_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )
