# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for classification report generation.
"""


from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from kenning.core.metrics import compute_classification_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def classification_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates classification quality section of the report.

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
    cmap : Optional[Any]
        Color map to be used in the plots
        (matplotlib.colors.ListedColormap)
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
    from kenning.core.drawing import Barplot, ConfusionMatrixPlot

    KLogger.info(
        f'Running classification report for \
        {measurementsdata["model_name"]}'
    )
    metrics = compute_classification_metrics(measurementsdata)
    measurementsdata |= metrics
    available_metrics = list(metrics.keys())

    if "eval_confusion_matrix" in measurementsdata:
        KLogger.info("Using confusion matrix")
        confusion_path = imgdir / f"{imgprefix}confusion_matrix"
        ConfusionMatrixPlot(
            title="Confusion matrix" if draw_titles else None,
            confusion_matrix=measurementsdata["eval_confusion_matrix"],
            class_names=measurementsdata["class_names"],
            cmap=cmap,
        ).plot(confusion_path, image_formats)
        measurementsdata["confusionpath"] = get_plot_wildcard_path(
            confusion_path, root_dir
        )
    elif "predictions" in measurementsdata:
        KLogger.info("Using predictions")

        predictions = list(
            zip(
                measurementsdata["predictions"],
                measurementsdata["class_names"],
            )
        )
        predictions.sort(key=lambda x: x[0], reverse=True)

        predictions = list(zip(*predictions))

        predictions_path = imgdir / f"{imgprefix}predictions"
        Barplot(
            title="Predictions" if draw_titles else None,
            x_label="Class",
            y_label="Percentage",
            y_unit="%",
            x_data=list(predictions[1]),
            y_data={"predictions": list(predictions[0])},
            colors=colors,
        ).plot(predictions_path, image_formats)
        measurementsdata["predictionspath"] = get_plot_wildcard_path(
            predictions_path, root_dir
        )
    else:
        KLogger.error(
            "Confusion matrix and predictions \
                not present for classification "
            "report"
        )
        return "", metrics

    measurementsdata["available_metrics"] = available_metrics
    with path(reports, "classification.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        ), metrics
