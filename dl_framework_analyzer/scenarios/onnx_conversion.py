import argparse
from collections import defaultdict
from jinja2 import Template
from pathlib import Path
from typing import List
from importlib.resources import path

from dl_framework_analyzer.core.onnxconversion import SupportStatus
from dl_framework_analyzer.core.onnxconversion import ONNXConversion
from dl_framework_analyzer.resources import reports
from dl_framework_analyzer.utils.class_loader import load_class
from dl_framework_analyzer.utils import logger


def generate_onnx_support_grid(
        converterslist: List[ONNXConversion],
        modelsdir: Path):
    supportlist = []
    for converter in converterslist:
        supportlist += converter.check_conversions(modelsdir)

    frameworkheaders = [f'{x.framework} (ver. {x.version})' for x in converterslist]
    
    supportgrid = defaultdict(dict)

    for el in supportlist:
        framework_and_ver = f'{el.framework} (ver. {el.version})'
        val = f'{el.exported} / {el.imported}'
        supportgrid[el.model][framework_and_ver] = val

    for model in supportgrid.keys():
        for f in frameworkheaders:
            if f not in supportgrid[model]:
                supportgrid[model][f] = 'Not provided / Not provided'

    return frameworkheaders, dict(supportgrid)


def create_onnx_support_report(
        converterslist: List[ONNXConversion],
        modelsdir: Path,
        output: Path):
    headers, grid = generate_onnx_support_grid(converterslist, modelsdir)
    
    with path(reports, 'onnx-conversion-support-grid.rst') as reportpath:
        with open(reportpath, 'r') as templatefile:
            template = templatefile.read()
            tm = Template(template)
            content = tm.render(
                headers=headers,
                grid=grid
            )
            with open(output, 'w') as out:
                out.write(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'modelsdir',
        help='Path to the directory where generated models will be stored',
        type=Path
    )
    parser.add_argument(
        'output',
        help='Path to the output RST file with ONNX support grid',
        type=Path
    )
    parser.add_argument(
        '--converters-list',
        help='List to the ONNXConversion-based classes from which the report will be generated',  # noqa: E501
        required=True,
        nargs='+'
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args = parser.parse_args()

    logger.set_verbosity(args.verbosity)

    args.modelsdir.mkdir(parents=True, exist_ok=True)

    converterslist = []

    for converter in args.converters_list:
        converterslist.append(load_class(converter)())

    create_onnx_support_report(
        converterslist,
        args.modelsdir,
        args.output)
