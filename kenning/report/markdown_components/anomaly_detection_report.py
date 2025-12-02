# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Module used for anomaly detection report generation stage.
"""
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from kenning.core.drawing import LinePlot
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports


def anomaly_detection_report(
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
    Creates Fault Detection Rate and False Alarm rate plots.
    """
    if "metrics_per_threshold" not in measurementsdata:
        return "", {}

    metrics_per_threshold = measurementsdata["metrics_per_threshold"]

    thresholds, fdrs, fars, adds = [], [], [], []

    for threshold, metrics in metrics_per_threshold.items():
        thresholds.append(float(threshold))
        fdrs.append(metrics["fdr"])
        fars.append(metrics["far"])
        adds.append(metrics["add"])

    anomalyfarfdrplot = imgdir / f"{imgprefix}anomalyfarfdrplot"
    LinePlot(
        lines=[
            (thresholds, fars),
            (thresholds, fdrs),
        ],
        x_label="Threshold",
        y_label="Percentage",
        y_unit="%",
        lines_labels=["False Alarm Rate", "Fault Detection Rate"],
        colors=colors,
    ).plot(anomalyfarfdrplot, image_formats)
    measurementsdata["anomalyfarfdrplot"] = get_plot_wildcard_path(
        anomalyfarfdrplot, root_dir
    )

    anomalyaddsplot = imgdir / f"{imgprefix}anomalyaddsplot"
    LinePlot(
        lines=[
            (thresholds, adds),
        ],
        x_label="Threshold",
        y_label="Delay",
        lines_labels=["Average Detection Delay"],
        colors=colors,
    ).plot(anomalyaddsplot, image_formats)
    measurementsdata["anomalyaddsplot"] = get_plot_wildcard_path(
        anomalyaddsplot, root_dir
    )
    with path(reports, "anomaly.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        ), {}
