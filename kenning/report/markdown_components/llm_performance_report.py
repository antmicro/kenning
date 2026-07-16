# Copyright (c) 2025-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for LLM tokens per second report generation.
"""

from importlib.resources import path
from typing import Any, Dict, Tuple

import numpy as np

from kenning.report.markdown_components.general import (
    create_report_from_measurements,
)
from kenning.resources import reports


def llm_performance_report(
    measurementsdata: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[str, Dict]:
    """
    Creates tokens per second section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Statistics from the Measurements class.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Tuple[str, Dict]
        Content of the report in MyST format, unused dict of measurements.
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
