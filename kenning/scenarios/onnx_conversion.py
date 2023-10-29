#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The script for creating ONNX import/export support matrix.

It expects a list of the implemented ONNXConversion classes and generates
the Markdown file with support matrix table.
"""

import sys
import argparse
from collections import defaultdict
from jinja2 import Template
from pathlib import Path
from typing import List, Tuple, Dict
from importlib.resources import path

from kenning.core.onnxconversion import ONNXConversion
from kenning.resources import reports
from kenning.utils.class_loader import load_class, get_command
from kenning.utils.logger import KLogger


def generate_onnx_support_grid(
    converterslist: List[ONNXConversion], modelsdir: Path
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Creates support matrix for ONNX import/export functions.

    Parameters
    ----------
    converterslist : List[ONNXConversion]
        List of specialized ONNXConversion objects for various frameworks to
        check.
    modelsdir : Path
        Path to the temporarily created ONNX models.

    Returns
    -------
    Tuple[List[str], Dict[str, Dict[str, str]]]
        Tuple of list of frameworks and their versions,
        dictionary with models, frameworks and their export/import support.
    """
    supportlist = []
    for converter in converterslist:
        supportlist += converter.check_conversions(modelsdir)

    frameworkheaders = [
        f"{x.framework} (ver. {x.version})" for x in converterslist
    ]  # noqa: E501

    supportgrid = defaultdict(dict)

    for el in supportlist:
        framework_and_ver = f"{el.framework} (ver. {el.version})"
        val = f"{el.exported} / {el.imported}"
        supportgrid[el.model][framework_and_ver] = val

    for model in supportgrid.keys():
        for f in frameworkheaders:
            if f not in supportgrid[model]:
                supportgrid[model][f] = "Not provided / Not provided"

    return frameworkheaders, dict(supportgrid)


def create_onnx_support_report(
    converterslist: List[ONNXConversion],
    modelsdir: Path,
    output: Path,
    command: str = "",
):
    """
    Creates a Mardown file summarizing the ONNX support grid.
    """
    headers, grid = generate_onnx_support_grid(converterslist, modelsdir)

    with path(reports, "onnx-conversion-support-grid.md") as reportpath:
        with open(reportpath, "r") as templatefile:
            template = templatefile.read()
            tm = Template(template)
            content = tm.render(headers=headers, grid=grid, command=command)
            with open(output, "w") as out:
                out.write(content)


def main(argv):  # noqa: D103
    command = get_command(argv)

    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        "modelsdir",
        help="Path to the directory where generated models will be stored",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Path to the output .md file with ONNX support grid",
        type=Path,
    )
    parser.add_argument(
        "--converters-list",
        help="List to the ONNXConversion-based classes from which the report will be generated",  # noqa: E501
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    args = parser.parse_args(argv[1:])

    KLogger.set_verbosity(args.verbosity)

    args.modelsdir.mkdir(parents=True, exist_ok=True)

    converterslist = []

    for converter in args.converters_list:
        converterslist.append(load_class(converter)())

    create_onnx_support_report(
        converterslist, args.modelsdir, args.output, command
    )


if __name__ == "__main__":
    main(sys.argv)
