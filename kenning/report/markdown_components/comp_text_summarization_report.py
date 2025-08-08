# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for text summarization comparison report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from kenning.core.metrics import compute_text_summarization_metrics
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def comparison_text_summarization_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates text summarization comparison section of report.

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
    from kenning.core.drawing import Barplot

    KLogger.info("Running comparison_text_summarization_report")

    rouge_data = {}
    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }

    for data in measurementsdata:
        metrics = compute_text_summarization_metrics(data)
        if not metrics:
            KLogger.warning(
                "No metrics were computed for the text summarization task "
                + f'for {data["model_name"]} in a comparison report'
            )
            continue

        rouge_keys = [key for key in metrics.keys() if key.startswith("rouge")]
        rouge_scores = [metrics[key] for key in rouge_keys]
        rouge_data[data["model_name"]] = rouge_scores

    rouge_path = imgdir / "rouge"

    if len(rouge_data.keys()) < 2:
        KLogger.error(
            f"Not enough metrics ({len(rouge_data.keys())}) were computed "
            + "for the comparison text summarization task"
        )
        return ""

    Barplot(
        title="Rouge scores" if draw_titles else None,
        x_label="Rouge",
        y_label="Score",
        y_unit="%",
        x_data=rouge_keys,
        y_data=rouge_data,
        colors=colors,
    ).plot(rouge_path, image_formats)
    report_variables["barplot_rouge_path_comparison"] = get_plot_wildcard_path(
        rouge_path, root_dir
    )

    with path(reports, "text_summarization_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )
