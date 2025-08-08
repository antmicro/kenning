# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for text summarization report generation.
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


def text_summarization_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    color_offset: int = 0,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates text summarization section of the report.

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
    colors : Optional[List]
        Colors to be used in the plots.
    color_offset : int
        How many colors from default color list should be skipped.
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

    KLogger.info(
        "Running text_summarization_report for "
        + measurementsdata["model_name"]
    )
    report_variables = {
        "model_name": measurementsdata["model_name"],
        "report_name": measurementsdata["report_name"],
        "report_name_simple": measurementsdata["report_name_simple"],
    }

    metrics = compute_text_summarization_metrics(measurementsdata)
    if not metrics:
        KLogger.error(
            "No metrics were computed for the text summarization task "
            + f'for {measurementsdata["model_name"]}'
        )
        return ""

    rouge_keys = [key for key in metrics.keys() if key.startswith("rouge")]
    rouge_scores = [metrics[key] for key in rouge_keys]

    if "predictions" in measurementsdata:
        from random import sample

        NUM_OF_EXAMPLES = 10

        KLogger.info(
            f"Including {NUM_OF_EXAMPLES} example predictions to "
            "the text summarization report"
        )
        report_variables["example_predictions"] = sample(
            measurementsdata["predictions"], NUM_OF_EXAMPLES
        )

    rouge_path = imgdir / f"{imgprefix}rouge"
    Barplot(
        title="Rouge scores" if draw_titles else None,
        x_label="Rouge",
        y_label="Score",
        y_unit="%",
        x_data=rouge_keys,
        y_data={"scores": rouge_scores},
        colors=colors,
        color_offset=color_offset,
    ).plot(rouge_path, image_formats)
    report_variables["barplot_rouge_path"] = get_plot_wildcard_path(
        rouge_path, root_dir
    )

    with path(reports, "text_summarization.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        ), metrics
