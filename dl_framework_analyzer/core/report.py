from jinja2 import Template
from pathlib import Path
from typing import Dict, List


def create_report_from_measurements(
        template: Path,
        measurementsdata: Dict[str, List],
        result: Path):
    with open(template, 'r') as resourcetemplatefile:
        resourcetemplate = resourcetemplatefile.read()
        tm = Template(resourcetemplate)
        content = tm.render(
            data=measurementsdata
        )
        with open(result, 'w') as out:
            out.write(content)
