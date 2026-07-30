# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for depth estimation quality report generation.
"""


from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from kenning.core.drawing import ImageGallery
from kenning.core.metrics import compute_depth_estimation_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def depth_estimation_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates depth estimation quality section of the report.

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
    cmap : Optional[Any]
        Color map to be used in the plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs: Any
        Additional arguments (not used here).

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    KLogger.info(
        f'Running depth estimation report for {measurementsdata["model_name"]}'
    )

    report_variables = {}

    metrics = compute_depth_estimation_metrics(measurementsdata)

    report_variables |= metrics
    report_variables["metrics"] = list(metrics.keys())

    unit = measurementsdata.get("depth_unit", None)

    if unit:
        unit = f"[{unit}]"

    report_variables["sample_categories"] = ["best", "worst"]

    for category in report_variables["sample_categories"]:
        _create_category_plots(
            measurementsdata=measurementsdata,
            category_prefix=category,
            report_variables=report_variables,
            unit=unit,
            imgdir=imgdir,
            image_formats=image_formats,
            root_dir=root_dir,
        )

    with path(reports, "depth_estimation.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        ), report_variables


def _create_category_plots(
    measurementsdata: Dict[str, Any],
    category_prefix: str,
    report_variables: Dict,
    unit: Optional[str],
    imgdir: Path,
    image_formats: Set[str],
    root_dir: Path,
):
    images = _get_as_list_of_numpy(
        measurementsdata, f"depth_{category_prefix}_sample_images"
    )
    truths = _get_as_list_of_numpy(
        measurementsdata, f"depth_{category_prefix}_sample_truths"
    )
    preds = _get_as_list_of_numpy(
        measurementsdata, f"depth_{category_prefix}_sample_preds"
    )

    plot_paths = []

    for img, truth, pred, i in zip(images, truths, preds, range(len(images))):
        plot_path = imgdir / f"depth_prediction_{category_prefix}_{i}"

        err = np.abs(truth - pred)

        ImageGallery(
            [img, err, truth, pred],
            n_cols=2,
            image_titles=["Image", "Abs error", "Ground truth", "Prediction"],
            force_common_scale=True,
            cbar_label=unit,
        ).plot(plot_path, image_formats)

        plot_paths.append(get_plot_wildcard_path(plot_path, root_dir))

    report_variables[f"{category_prefix}_plot_paths"] = plot_paths


def _get_as_list_of_numpy(data: Dict[str, Any], key: str) -> List[np.ndarray]:
    return [np.load(path) for path in data.get(key, [])]
