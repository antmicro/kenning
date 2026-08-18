# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module generating report sections with data
gathered from compilation or optimization.
"""

from typing import Any, Dict, Tuple

from kenning.report.markdown_components.iree_compilation import (
    comparison_iree_compilation_report,
    iree_compilation_report,
)

COMPILER_REPORTS = {
    "iree": iree_compilation_report,
}

COMPARISON_COMPILER_REPORTS = {
    "iree": comparison_iree_compilation_report,
}


def compilation_report(
    measurementsdata: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[str, Dict]:
    """
    Generates report sections based on data gathered
    during compilation or optimization.
    """
    report, data = "", {}
    compilation_metadata = measurementsdata["compilation_metadata"]
    for key, func in COMPILER_REPORTS.items():
        if key not in compilation_metadata:
            continue
        r, d = func(measurementsdata, **kwargs)
        report += r
        data |= d
    return report, data


def comparison_compilation_report(
    measurementsdata: Dict[str, Any],
    **kwargs: Any,
) -> str:
    """
    Generates comparison report sections based on data gathered
    during compilation or optimization.
    """
    report = ""
    compilation_metadata = measurementsdata["compilation_metadata"]
    for key, func in COMPARISON_COMPILER_REPORTS.items():
        if key not in compilation_metadata:
            continue
        report += func(measurementsdata, **kwargs)
    return report, {}
