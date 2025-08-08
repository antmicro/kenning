# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Functions to generate Markdown reports from templates and Measurements objects.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentError, Namespace
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from kenning.core.measurements import Measurements
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger


# REPORT_TYPES:
class ReportTypes(str, Enum):
    """
    Enum that describes report types used for report generation.
    """

    PERFORMANCE = "performance"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    TEXT_SUMMARIZATION = "text_summarization"
    RENODE = "renode_stats"


class Report(ArgumentsHandler, ABC):
    """
    Generate report based on measurements data gathered during
    the pipeline operation. It is a base class for possible
    types of report for different scenarios.
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
        "report_types": {
            "description": "List of types that implement this report",
            "type": list[str],
            "default": None,
        },
        "automl_stats": {
            "description": "Path to the JSON file with statistics\
                  during the AutoML run",
            "type": Path,
            "default": False,
        },
    }

    def __init__(
        self,
        measurements: Path | List[Path] = [None],
        report_name: Optional[str] = None,
        report_types: Optional[List[str]] = None,
        automl_stats: Optional[Path] = None,
    ):
        """
        Constructs report.

        Parameters
        ----------
        measurements : Path | List[Path]
            Path to the JSON files with measurements.
        report_name : Optional[str]
            Name of the report.
        report_types: Optional[List[str]]
            List of types that implement this report.
        automl_stats : Optional[Path]
            Path to the JSON file with statistics during the AutoML run
        """
        self.report_types = report_types

        if type(measurements) is not list:
            self.measurements = [measurements]
        else:
            self.measurements = measurements

        self.report_name = report_name
        self.automl_stats = automl_stats

    @abstractmethod
    def generate_report(
        self,
        subcommands: Optional[List[str]] = None,
        command: Optional[List[str]] = None,
        automl_stats: Optional[Path] = None,
    ) -> None:
        """
        Generate report.

        Parameters
        ----------
        subcommands : Optional[List[str]]
            Used subcommands from parsed arguments.
        command : Optional[List[str]]
            A list of arguments from command line.
        automl_stats : Optional[Path]
            Path to the JSON file with statistics during the AutoML run

        Returns
        -------
        None
        """
        ...

    @staticmethod
    def get_model_name(filepath: Path) -> str:
        """
        Generates the name of the model. Path to the measurements
        file is used for name generation.

        Parameters
        ----------
        filepath : Path
            Path to the measurements file.

        Returns
        -------
        str
            Name of the model used when generating the report.
        """
        return str(filepath).replace("/", ".")

    @staticmethod
    def deduce_report_types(measurements_data: List[Dict]) -> List[str]:
        """
        Deduces what type of report should be generated
        based on measurements data.

        Report type is chosen only when all measurements data are compatible
        with it.

        Parameters
        ----------
        measurements_data : List[Dict]
            List with measurements data from which the
            report will be generated.

        Returns
        -------
        List[str]
            List with types of report
        """
        report_types = []

        def _append_type_if(_type: str, func: Callable):
            if all(map(func, measurements_data)):
                report_types.append(_type)

        _append_type_if(
            ReportTypes.CLASSIFICATION,
            lambda data: "eval_confusion_matrix" in data,
        )
        _append_type_if(
            ReportTypes.DETECTION,
            lambda data: any(
                nested_data.startswith("eval_gtcount") for nested_data in data
            ),
        )
        _append_type_if(
            ReportTypes.TEXT_SUMMARIZATION,
            lambda data: any([key.startswith("rouge") for key in data.keys()]),
        )
        _append_type_if(
            ReportTypes.PERFORMANCE,
            lambda data: "target_inference_step" in data
            or "protocol_inference_step" in data,
        )
        _append_type_if(
            ReportTypes.RENODE, lambda data: "opcode_counters" in data
        )

        if len(report_types) == 0:
            KLogger.error(
                "There is no report type which is "
                "suitable for all measurements"
            )
            return []

        KLogger.info(f"Following report types were deduced: {report_types}")
        return report_types

    @staticmethod
    def deduce_report_name(
        measurements_data: List[Dict], report_types: List[str]
    ) -> str:
        """
        Deduces simple report name based on measurements and its type.

        Parameters
        ----------
        measurements_data : List[Dict]
            List with measurements data from which the report
            will be generated.
        report_types : List[str]
            List with types of report.

        Returns
        -------
        str
            Report name
        """
        model_names = [d["model_name"] for d in measurements_data[:-1]]

        if len(measurements_data) > 1:
            report_name = (
                "Comparison of "
                f"{', '.join(model_names)}"
                f" and {measurements_data[-1]['model_name']}"
            )
        elif "report_name" in measurements_data[0]:
            report_name = measurements_data[0]["report_name"]
        elif len(report_types) > 1:
            report_name = (
                f"{', '.join(report_types[:-1])} and "
                f"{report_types[-1]} of {measurements_data[0]['model_name']}"
            )
        else:
            report_name = (
                f"{report_types[0]} of "
                f"{measurements_data[0]['model_name']}"
            )
        report_name = report_name[0].upper() + report_name[1:]

        KLogger.info(f"Report name: {report_name}")
        return report_name

    @classmethod
    def load_measurements_for_report(
        cls,
        measurements_files: List[str],
        skip_unoptimized_model: bool,
        model_names: Optional[List[str]] = None,
        report_types: Optional[List[str]] = None,
        automl_stats_file: Optional[Path] = None,
    ) -> Tuple[Dict, List[str], Optional[Dict]]:
        """
        Loads all files with measurements and prepares list of report types.

        Parameters
        ----------
        measurements_files: List[str]
            List of the files with measurements
        skip_unoptimized_model: bool
            If False, the original native model measurements
            should be collected
        model_names: Optional[List[str]]
            List of model names for measurements
            that should be displayed in the report
        report_types: Optional[List[str]]
            Types of reports (performance, clasisfication, ...) to include
        automl_stats_file : Optional[Path]
            Path to the JSON file with statistics during the AutoML run.

        Returns
        -------
        Dict
            Measurements data to use for report
        List[str]
            List of report types, either passed from arguments or derived
            from measurements data
        Optional[Dict]
            Optional dictionary with AutoML statistics.

        Raises
        ------
        argparse.ArgumentError :
            Raised when report types cannot be deduced from measurements data
        """
        measurementsdata = []
        for i, measurementspath in enumerate(measurements_files):
            if not os.path.exists(measurementspath):
                continue

            with open(measurementspath, "r") as measurementsfile:
                measurements = json.load(measurementsfile)
            if model_names is not None:
                measurements["model_name"] = model_names[i]
            elif "model_name" not in measurements:
                measurements["model_name"] = cls.get_model_name(
                    measurementspath
                )
            measurements["model_name"] = measurements["model_name"].replace(
                " ", "_"
            )
            # Append measurements of unoptimized data separately
            if (
                not skip_unoptimized_model
                and Measurements.UNOPTIMIZED in measurements
            ):
                unoptimized = measurements.pop(Measurements.UNOPTIMIZED)
                if "model_name" not in unoptimized:
                    unoptimized[
                        "model_name"
                    ] = f"unoptimized {measurements['model_name']}"
                unoptimized[Measurements.UNOPTIMIZED] = True
                measurementsdata.append(unoptimized)
            measurementsdata.append(measurements)

        report_types = report_types
        if not report_types:
            report_types = cls.deduce_report_types(measurementsdata)
        if report_types is None:
            raise ArgumentError(
                None,
                "Report types cannot be deduced. Please specify "
                "'--report-types' or make sure the path is correct "
                "measurements were chosen.",
            )

        for measurements in measurementsdata:
            if "build_cfg" in measurements:
                measurements["build_cfg"] = json.dumps(
                    measurements["build_cfg"], indent=4
                ).split("\n")

            if "report_name" not in measurements:
                measurements["report_name"] = cls.deduce_report_name(
                    [measurements], report_types
                )
            measurements["report_name_simple"] = re.sub(
                r"[\W]",
                "",
                measurements["report_name"].lower().replace(" ", "_"),
            )

        automl_stats = None
        # Load AutoML stats
        if automl_stats_file:
            with automl_stats_file.open("r") as fd:
                automl_stats = json.load(fd)

        return measurementsdata, report_types, automl_stats

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
