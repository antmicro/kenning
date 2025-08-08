# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions to generate Markdown reports from templates and Measurements objects.
"""

import sys
from abc import ABC, abstractmethod
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
        "root_dir": {
            "description": "Path to root directory for documentation \
                (paths in the MyST file are relative to this directory)",
            "type": Path,
            "default": None,
        },
        "model_names": {
            "description": "Names of the models used to create measurements\
                  in order",
            "type": str,
            "default": None,
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
        verbosity: str = "INFO",
    ):
        """
        Constructs report.

        Parameters
        ----------
        measurements : list[Path] | Path
            Path to the JSON files with measurements.
        report_name : str
            Name of the report.
        report_path: Optional[Path]
            Path to the output MyST file.
        verbosity: str
            Verbosity level.
        """
        self.measurements = measurements
        self.report_name = report_name
        self.report_path = report_path
        self.verbosity = verbosity

    @abstractmethod
    def generate_report(self):
        """
        Generate report.
        """
        ...

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
