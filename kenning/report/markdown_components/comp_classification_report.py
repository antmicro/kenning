# Copyright (c) 2025-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for classification comparison report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from kenning.core.metrics import (
    Metric,
    compute_classification_metrics,
)
from kenning.report.markdown_components.comp_generic_quality_report import (
    _create_comparison_generic_quality_report,
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
    main_quality_metric: Optional[Metric] = None,
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
    main_quality_metric : Optional[Metric]
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
    from kenning.core.drawing import Barplot

    KLogger.info("Running comparison_classification_report")
    # HTML plots format unsupported, removing html

    # check that each measurements have the same classes
    for data in measurementsdata:
        assert (
            measurementsdata[0]["class_names"] == data["class_names"]
        ), "Invalid class names in measurements"

    (
        report_variables,
        skip_inference_metrics,
    ) = _create_comparison_generic_quality_report(
        type_to_display="Classification",
        compute_metric_func=compute_classification_metrics,
        default_quality_metric=Metric.ACC,
        measurementsdata=measurementsdata,
        imgdir=imgdir,
        root_dir=root_dir,
        image_formats=image_formats,
        colors=colors,
        draw_titles=draw_titles,
        main_quality_metric=main_quality_metric,
        metrics_for_radar=metrics_for_radar,
    )

    if "predictions" in measurementsdata[0] and (
        "eval_confusion_matrix" not in measurementsdata[0]
    ):
        names = report_variables["model_names"]

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

    with path(reports, "classification_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )
