# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for depth estimation quality report generation.
"""


from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from kenning.core.metrics import compute_depth_estimation_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
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

    metrics = compute_depth_estimation_metrics(measurementsdata)

    with path(reports, "depth_estimation.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, metrics
        ), metrics
