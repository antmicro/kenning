# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions to generate Markdown reports from templates and Measurements objects.
"""

from jinja2 import Template
from pathlib import Path
from typing import Dict, List


def create_report_from_measurements(
    template: Path, measurementsdata: Dict[str, List]
) -> str:
    """
    Creates a report from template and measurements data.

    It additionally provides methods for quality metrics.

    Parameters
    ----------
    template : Path
        Path to the Jinja template.
    measurementsdata : Dict[str, List]
        Dictionary describing measurements taken during benchmark.

    Returns
    -------
    str :
        Content of the report.
    """
    with open(template, "r") as resourcetemplatefile:
        resourcetemplate = resourcetemplatefile.read()
        tm = Template(resourcetemplate)

        content = tm.render(data=measurementsdata)

        return content
