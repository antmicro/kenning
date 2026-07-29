# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing helper functions for markdown report generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

from kenning.core.drawing import BubblePlot, RadarChart
from kenning.core.metrics import Metric
from kenning.report.markdown_components.general import get_plot_wildcard_path
from kenning.utils.logger import KLogger


def plot_all_comparisons(
    all_display_metrics: Dict[str, Dict[Metric, float]],
    main_quality_metric: Metric,
    metrics_for_radar: Optional[List[Metric]],
    available_metrics: List[Metric],
    mean_inference_times: List,
    model_sizes: List,
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List],
    draw_titles: bool = True,
) -> Dict[str, str]:
    """
    Displays bubble and radar plots with model comparisons.

    Parameters
    ----------
    all_display_metrics : Dict[str, Dict[Metric, float]]
        Dictionary from model names to all model metrics.
    main_quality_metric: Metric
        The quality metric to be shown on a radar plot.
    metrics_for_radar : Optional[List[Metric]]
        Set of metrics to be included in the plot.
    available_metrics: List[Metric]
        Set of metrics which are defined for all models.
    mean_inference_times : List
        Inference times for all models.
    model_sizes: List
        Sizes of all models.
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

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot types to their paths.
    """
    result = {}

    if main_quality_metric in available_metrics:
        result["bubbleplotpath"] = plot_bubble_metric_vs_performance(
            quality_metrics=[
                model_metrics[main_quality_metric]
                for model_metrics in all_display_metrics.values()
            ],
            mean_inference_times=mean_inference_times,
            model_sizes=model_sizes,
            model_names=list(all_display_metrics.keys()),
            quality_metric=main_quality_metric,
            imgdir=imgdir,
            root_dir=root_dir,
            image_formats=image_formats,
            colors=colors,
            draw_titles=draw_titles,
        )
    else:
        KLogger.error(
            f"{main_quality_metric} not available for all models, skipping "
            "bubble plot"
        )

    result["radarchartpath"] = plot_radar_comparison(
        all_display_metrics=all_display_metrics,
        metrics_for_radar=metrics_for_radar,
        available_metrics=available_metrics,
        imgdir=imgdir,
        root_dir=root_dir,
        image_formats=image_formats,
        colors=colors,
        draw_titles=draw_titles,
    )

    return result


def plot_bubble_metric_vs_performance(
    quality_metrics: List,
    mean_inference_times: List,
    model_sizes: List,
    model_names: List[str],
    quality_metric: Metric,
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List],
    draw_titles: bool = True,
) -> str:
    """
    Creates a comparison bubble plot showing the relation between model
    quality, size and inference time.

    Parameters
    ----------
    quality_metrics : List
        Values of the quality metric for all models.
    mean_inference_times : List
        Inference times for all models.
    model_sizes: List
        Sizes of all models.
    model_names: List[str]
        Names of all models.
    quality_metric: Metric
        The name of the quality metric which values are passed as
        quality_metrics.
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

    Returns
    -------
    str
        Path to the generated plot.
    """
    plot_path = imgdir / f"{quality_metric.name.lower()}_vs_inference_time"

    BubblePlot(
        title=f"{quality_metric.value} vs Mean inference time"
        if draw_titles
        else None,
        x_data=mean_inference_times,
        x_label="Mean inference time [s]",
        y_data=quality_metrics,
        y_label=quality_metric.value,
        size_data=model_sizes,
        size_label="Model size",
        bubble_labels=model_names,
        colors=colors,
    ).plot(plot_path, image_formats)

    return get_plot_wildcard_path(plot_path, root_dir)


def plot_radar_comparison(
    all_display_metrics: Dict[str, Dict[Metric, float]],
    metrics_for_radar: Optional[List[Metric]],
    available_metrics: List[Metric],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List],
    draw_titles: bool = True,
) -> str:
    """
    Creates a radar plot comparing different metrics of multiple models.

    Parameters
    ----------
    all_display_metrics : Dict[str, Dict[Metric, float]]
        Dictionary from model names to all model metrics.
    metrics_for_radar : Optional[List[Metric]]
        Set of metrics to be included in the plot.
    available_metrics: List[Metric]
        Set of metrics which are defined for all models.
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

    Returns
    -------
    str
        Path to the generated plot.
    """
    plot_path = imgdir / "classification_metric_comparison"

    if metrics_for_radar is None:
        metrics_for_radar = available_metrics

    if not set(available_metrics).issuperset(metrics_for_radar):
        KLogger.error(
            f"{set(metrics_for_radar).difference(available_metrics)} "
            "are not available for all models"
        )
        metrics_for_radar = [
            metric
            for metric in available_metrics
            if metric in metrics_for_radar
        ]

    RadarChart(
        title="Metric comparison" if draw_titles else None,
        metric_data={
            model: [metrics[metric] for metric in metrics_for_radar]
            for model, metrics in all_display_metrics.items()
        },
        metric_labels=[metric.value for metric in metrics_for_radar],
        colors=colors,
    ).plot(plot_path, image_formats)

    return get_plot_wildcard_path(plot_path, root_dir)
