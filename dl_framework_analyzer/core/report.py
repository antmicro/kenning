"""
Functions to generate RST reports from templates and Measurements objects.
"""

from jinja2 import Template
from pathlib import Path
from typing import Dict, List


def create_report_from_measurements(
        template: Path,
        measurementsdata: Dict[str, List],
        result: Path):
    """
    Creates a report file from template and measurements data.

    Parameters
    ----------
    template : Path
        Path to the Jinja template
    measurementsdata : Dict[str, List]
        dictionary describing measurements taken during benchmark
    result : Path
        Path to the output
    """
    with open(template, 'r') as resourcetemplatefile:
        resourcetemplate = resourcetemplatefile.read()
        tm = Template(resourcetemplate)
        content = tm.render(
            data=measurementsdata
        )
        with open(result, 'w') as out:
            out.write(content)
