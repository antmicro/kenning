# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A package that provides markdown and HTML report generator
implementation of core report class.
"""

import sys
from pathlib import Path
from typing import Optional

if sys.version_info.minor < 9:
    pass
else:
    pass
from kenning.core.report import (
    Report,
)


class MarkdownReport(Report):
    """
    Class responsible for markdown report generation.
    """

    arguments_structure = {
        "to_html": {
            "description": "Generate HTML version of the report, \
                it can receive path to the folder where HTML will be saved",
            "type": bool | Path,
            "default": False,
        },
        "img_dir": {
            "description": "Path to the directory where images will be stored",
            "type": Path,
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
        "report_types": {
            "description": "List of types that implement this report",
            "type": list[str],
            # "enum":REPORT_TYPES,
            "default": None,
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
        super().__init__(measurements, report_name, report_path, verbosity)

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

    def generate_report(self):
        """
        Generate report.
        """
        pass
