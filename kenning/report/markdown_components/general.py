# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with general function used by different modules and functions in
markdown_components package.
"""

from pathlib import Path
from typing import Dict, List

from kenning.core.metrics import Metric


def get_plot_wildcard_path(plot_path: Path, root_dir: Path) -> str:
    """
    Generate wildcard plot path relative to given directory
    which can be used in report.

    Parameters
    ----------
    plot_path : Path
        Path to the saved plot.
    root_dir : Path
        Report root directory.

    Returns
    -------
    str
        Universal plot path relative to report root directory.
    """
    return str(
        plot_path.with_suffix(plot_path.suffix + ".*").relative_to(root_dir)
    )


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
    str
        Content of the report.
    """
    with open(template, "r") as resourcetemplatefile:
        from jinja2 import Template

        resourcetemplate = resourcetemplatefile.read()
        tm = Template(resourcetemplate)

        content = tm.render(data=measurementsdata, zip=zip, Metric=Metric)

        return content
