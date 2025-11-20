# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A package that contains useful modules used for markdown report generation.
"""

from kenning.report.markdown_components.automl_report import automl_report
from kenning.report.markdown_components.classification_report import (
    classification_report,
)
from kenning.report.markdown_components.comp_classification_report import (
    comparison_classification_report,
)
from kenning.report.markdown_components.comp_detection_report import (
    comparison_detection_report,
)
from kenning.report.markdown_components.comp_llm_performance_report import (
    comparison_llm_performance_report,
)
from kenning.report.markdown_components.comp_performance_report import (
    comparison_performance_report,
)
from kenning.report.markdown_components.comp_renode_stats_report import (
    comparison_renode_stats_report,
)
from kenning.report.markdown_components.comp_text_summarization_report import (
    comparison_text_summarization_report,
)
from kenning.report.markdown_components.detection_report import (
    detection_report,
)
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    generate_html_report,
    get_plot_wildcard_path,
)
from kenning.report.markdown_components.llm_performance_report import (
    llm_performance_report,
)
from kenning.report.markdown_components.performance_report import (
    performance_report,
)
from kenning.report.markdown_components.renode_stats_report import (
    renode_stats_report,
)
from kenning.report.markdown_components.text_summarization_report import (
    text_summarization_report,
)

__all__ = [
    "automl_report",
    "classification_report",
    "detection_report",
    "llm_performance_report",
    "performance_report",
    "renode_stats_report",
    "text_summarization_report",
    "create_report_from_measurements",
    "get_plot_wildcard_path",
    "comparison_classification_report",
    "comparison_detection_report",
    "comparison_llm_performance_report",
    "comparison_performance_report",
    "comparison_renode_stats_report",
    "comparison_text_summarization_report",
    "generate_html_report",
]
