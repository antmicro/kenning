# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for LLM tokens per second comparison report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from kenning.core.drawing import Barplot
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports


def comparison_llm_performance_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates tokens per second comparison section of report.

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
    values = {}
    for measurement in measurementsdata:
        tokens_per_second = [
            token / time
            for token, time in zip(
                measurement["tokens"], measurement["target_inference_step"]
            )
        ]
        values[measurement["model_name"]] = [np.mean(tokens_per_second)]

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }

    Barplot(
        title="Tokens per second" if draw_titles else None,
        x_label="Model",
        y_label="Tokens per second",
        x_data=[""],
        y_data=values,
        colors=colors,
    ).plot(imgdir, image_formats)
    report_variables[
        "barplot_tokens_per_second_comparison"
    ] = get_plot_wildcard_path(imgdir, root_dir)

    with path(reports, "llm_performance_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )
