# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions to generate Markdown reports from templates and Measurements objects.
"""

import sys
from abc import ABC
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

from kenning.core.metrics import Metric
from kenning.utils.args_manager import ArgumentsHandler

if sys.version_info.minor < 9:
    pass
else:
    pass


# REPORT_TYPES:
PERFORMANCE = "performance"
CLASSIFICATION = "classification"
DETECTION = "detection"
TEXT_SUMMARIZATION = "text_summarization"
RENODE = "renode_stats"
REPORT_TYPES = [
    PERFORMANCE,
    CLASSIFICATION,
    DETECTION,
    RENODE,
    TEXT_SUMMARIZATION,
]


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


class Report(ArgumentsHandler, ABC):
    """
    Generate report based on measurements data gathered during
    the pipeline operation, but for now it is used for arguments
    retrieving from arguments flags and YAML scenario for RenderReport class.
    """

    arguments_structure = {
        "measurements": {
            "description": "Path to the JSON files with measurements",
            "type": Path | list[Path],
            "default": [None],
        },
        "report_name": {
            "description": "Name of the report",
            "type": str,
            "default": None,
        },
        "report_path": {
            "description": "Path to the output MyST file",
            "type": Path,
            "default": None,
        },
        "to_html": {
            "description": "Generate HTML version of the report, \
                it can receive path to the folder where HTML will be saved",
            "type": bool | Path,
            "default": False,
        },
        "root_dir": {
            "description": "Path to root directory for documentation \
                (paths in the MyST file are relative to this directory)",
            "type": Path,
            "default": None,
        },
        "report_types": {
            "description": "List of types that implement this report",
            "type": list[str],
            # "enum":REPORT_TYPES,
            "default": None,
        },
        "img_dir": {
            "description": "Path to the directory where images will be stored",
            "type": Path,
            "default": None,
        },
        "model_names": {
            "description": "Names of the models used to create measurements\
                  in order",
            "type": str,
            "default": None,
        },
        "only_png_images": {
            "description": "Forcing to generate images only in PNG format, \
                if not specified also images in HTML will be generated",
            "type": bool,
            "default": False,
        },
        "comparison_only": {
            "description": "Creates only sections with comparisons\
                  of metrics and time series",
            "type": bool,
            "default": False,
        },
        "skip_unoptimized_model": {
            "description": "Do not use measurements \
                of unoptimized model",
            "type": bool,
            "default": False,
        },
        "smaller_header": {
            "description": "Use smaller size for header \
                containing report name",
            "type": bool,
            "default": False,
        },
        "save_summary": {
            "description": "Saves JSON file with summary data from the report\
                 to file specified in report-path with suffix `.summary.json`",
            "type": bool,
            "default": False,
        },
        "skip_general_information": {
            "description": "Removes beginning sections listing \
                used configuration and commands",
            "type": bool,
            "default": False,
        },
        "verbosity": {
            "description": "Verbosity level",
            "type": str,
            "enum": [
                "NOTSET",
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ],
            "default": "INFO",
        },
    }

    def __init__(
        self,
        measurements: list[Path] | Path = [None],
        report_name: str = None,
        report_path: Optional[Path] = None,
        to_html: bool | Path = False,
        root_dir: Optional[Path] = None,
        report_types: list[str] = None,
        img_dir: Optional[Path] = None,
        model_names: Optional[str] = None,
        only_png_images: bool = False,
        comparison_only: bool = False,
        skip_unoptimized_model: bool = False,
        smaller_header: bool = False,
        save_summary: bool = False,
        skip_general_information: bool = False,
        verbosity: str = "INFO",
    ):
        """c
        Constructs platform and reads its data from platforms.yaml.

        Parameters
        ----------
        measurements : list[Path] | Path
            Path to the JSON files with measurements.
        report_name : str
            Name of the report.
        report_path: Optional[Path]
            Path to the output MyST file.
        to_html: bool | Path
            Generate HTML version of the report, it can receive path
            to the folder where HTML will be saved.
        root_dir: Optional[Path]
            Path to root directory for documentation
            (paths in the MyST file are relative to this directory)
        report_types: list[str]
            List of types that implement this report.
        img_dir: Optional[Path]
            Path to the directory where images will be stored.
        model_names: Optional[str]
            Names of the models used to create measurements in order.
        only_png_images: bool
            Forcing to generate images only in PNG format,
            if not specified also images in HTML will be generated.
        comparison_only: bool
            Creates only sections with comparisons of metrics and time series.
        skip_unoptimized_model: bool
            Do not use measurements of unoptimized model.
        smaller_header: bool
            Use smaller size for header containing report name.
        save_summary: bool
            Saves JSON file with summary data from the report,
            to file specified in report-path with suffix `.summary.json`.
        skip_general_information: bool
            Removes beginning sections listing used configuration and commands.
        verbosity: str
            Verbosity level.
        """
        self.measurements = measurements
        self.report_name = report_name
        self.report_path = report_path
        self.to_html = to_html
        self.root_dir = root_dir
        self.report_types = report_types
        self.img_dir = img_dir
        self.model_names = model_names
        self.only_png_images = only_png_images
        self.comparison_only = comparison_only
        self.skip_unoptimized_model = skip_unoptimized_model
        self.smaller_header = smaller_header
        self.save_summary = save_summary
        self.skip_general_information = skip_general_information
        self.verbosity = verbosity

    @classmethod
    def from_argparse(cls, args: Namespace) -> "Report":
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : Namespace
            Arguments from ArgumentParser object.

        Returns
        -------
        Report
            Object of class Report.
        """
        return super().from_argparse(args)

    @classmethod
    def from_json(cls, json_dict: Dict) -> "Report":
        """
        Constructor wrapper that takes the parameters from json dict.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.

        Returns
        -------
        Report
            Object of class Report.
        """
        return super().from_json(json_dict)
