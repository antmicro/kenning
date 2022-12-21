"""
Functions to generate Markdown reports from templates and Measurements objects.
"""

from jinja2 import Template
from pathlib import Path
from typing import Dict, List


def create_report_from_measurements(
        template: Path,
        measurementsdata: Dict[str, List]):
    """
    Creates a report from template and measurements data.

    It additionaly provides methods for quality metrics.

    Parameters
    ----------
    template : Path
        Path to the Jinja template
    measurementsdata : Dict[str, List]
        dictionary describing measurements taken during benchmark
    """
    with open(template, 'r') as resourcetemplatefile:
        resourcetemplate = resourcetemplatefile.read()
        tm = Template(resourcetemplate)

        content = tm.render(
            data=measurementsdata
        )

        return content
