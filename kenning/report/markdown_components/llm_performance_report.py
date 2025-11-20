# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for LLM tokens per second report generation.
"""

from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

from kenning.report.markdown_components.general import (
    create_report_from_measurements,
)
from kenning.resources import reports


def llm_performance_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    **kwargs: Any,
) -> str:
    """
    Creates tokens per second section of the report.

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
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    tokens_per_second = [
        token / time
        for token, time in zip(
            measurementsdata["tokens"],
            measurementsdata["target_inference_step"],
        )
    ]
    measurementsdata["tokens_per_second_mean"] = np.mean(tokens_per_second)
    measurementsdata["tokens_per_second_std"] = np.std(tokens_per_second)
    measurementsdata["tokens_per_second_median"] = np.median(tokens_per_second)
    measurementsdata["tokens_per_second_min"] = np.min(tokens_per_second)
    measurementsdata["tokens_per_second_max"] = np.max(tokens_per_second)

    with path(reports, "llm_performance.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        ), {}
