# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for detection comparison report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def comparison_detection_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates detection comparison section of report.

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
    KLogger.info("Running comparison_detection_report")

    from kenning.core.drawing import LinePlot
    from kenning.datasets.helpers.detection_and_segmentation import (
        compute_map_per_threshold,
    )

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
        "model_names": [],
    }

    visualization_data = []
    for data in measurementsdata:
        thresholds = np.arange(0.2, 1.05, 0.05)
        mapvalues = compute_map_per_threshold(data, thresholds)
        max_map = max(mapvalues)
        max_thr = thresholds[np.argmax(mapvalues)].round(2)

        report_variables[data["model_name"]] = {}
        report_variables[data["model_name"]]["best_map"] = max_map
        report_variables[data["model_name"]]["best_map_thr"] = max_thr

        visualization_data.append((thresholds, mapvalues))
        report_variables["model_names"].append(data["model_name"])

    plot_path = imgdir / "detection_map_thresholds"
    LinePlot(
        title=(
            "mAP values comparison over different threshold values"
            if draw_titles
            else None
        ),
        x_label="threshold",
        y_label="mAP",
        lines=visualization_data,
        lines_labels=report_variables["model_names"],
        colors=colors,
    ).plot(plot_path, image_formats)
    report_variables["mapcomparisonpath"] = get_plot_wildcard_path(
        plot_path, root_dir
    )

    with path(reports, "detection_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )
