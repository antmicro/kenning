# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A package that provides markdown and HTML report generator
implementation of core report class.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from matplotlib.colors import to_hex

from kenning.cli.command_template import AUTOML
from kenning.core.metrics import Metric
from kenning.report.markdown_components import (
    automl_report,
    classification_report,
    comparison_classification_report,
    comparison_detection_report,
    comparison_performance_report,
    comparison_renode_stats_report,
    comparison_text_summarization_report,
    create_report_from_measurements,
    detection_report,
    generate_html_report,
    performance_report,
    renode_stats_report,
    text_summarization_report,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.core.drawing import (
    KENNING_COLORS,
    RED_GREEN_CMAP,
    SERVIS_PLOT_OPTIONS,
    Plot,
    choose_theme,
)
from kenning.core.measurements import Measurements
from kenning.core.report import Report, ReportTypes


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
            "overridable": True,
        },
        "report_path": {
            "description": "Path to the output MyST file",
            "type": Path,
            "default": None,
            "required": True,
            "overridable": True,
        },
        "root_dir": {
            "description": "Path to root directory for documentation \
                (paths in the MyST file are relative to this directory)",
            "type": Path,
            "default": None,
            "overridable": True,
        },
        "img_dir": {
            "description": "Path to the directory where images will be stored",
            "type": Path,
            "default": None,
            "overridable": True,
        },
        "only_png_images": {
            "description": "Forcing to generate images only in PNG format, \
                if not specified also images in HTML will be generated",
            "type": bool,
            "default": False,
            "overridable": True,
        },
        "model_names": {
            "description": "Names of the models used to create measurements\
                  in order",
            "type": str,
            "default": None,
            "overridable": True,
        },
        "comparison_only": {
            "description": "Creates only sections with comparisons\
                  of metrics and time series",
            "type": bool,
            "default": False,
            "overridable": True,
        },
        "skip_unoptimized_model": {
            "description": "Do not use measurements \
                of unoptimized model",
            "type": bool,
            "default": False,
            "overridable": True,
        },
        "smaller_header": {
            "description": "Use smaller size for header \
                containing report name",
            "type": bool,
            "default": False,
            "overridable": True,
        },
        "save_summary": {
            "description": "Saves JSON file with summary data from the report\
                 to file specified in report-path with suffix `.summary.json`",
            "type": bool,
            "default": False,
            "overridable": True,
        },
        "skip_general_information": {
            "description": "Removes beginning sections listing \
                used configuration and commands",
            "type": bool,
            "default": False,
            "overridable": True,
        },
        "main_quality_metric": {
            "description": "Option that allows you to select a metric "
            "to compare models against",
            "type": str,
            "enum": [metric.name.lower() for metric in list(Metric)],
            "default": Metric.ACC.name.lower(),
            "overridable": True,
        },
    }

    def __init__(
        self,
        measurements: List[Path] | Path = [None],
        report_name: Optional[str] = None,
        report_path: Path = None,
        to_html: bool | Path = False,
        root_dir: Optional[Path] = None,
        report_types: List[str] = None,
        img_dir: Optional[Path] = None,
        model_names: Optional[str] = None,
        only_png_images: bool = False,
        comparison_only: bool = False,
        skip_unoptimized_model: bool = False,
        smaller_header: bool = False,
        save_summary: bool = False,
        skip_general_information: bool = False,
        automl_stats: Optional[Path] = None,
        main_quality_metric: str = Metric.ACC.name.lower(),
    ):
        super().__init__(measurements, report_name, report_types, automl_stats)

        self.to_html = to_html
        self.report_path = report_path
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.model_names = model_names
        self.only_png_images = only_png_images
        self.comparison_only = comparison_only
        self.skip_unoptimized_model = skip_unoptimized_model
        self.smaller_header = smaller_header
        self.save_summary = save_summary
        self.skip_general_information = skip_general_information

        KLogger.debug(f"Report measurements: {self.measurements}")

        self.measurementsdata = {}

        self.main_quality_metric = Metric[main_quality_metric.upper()]

        if self.to_html:
            if isinstance(self.to_html, Path):
                self.to_html = Path(self.to_html).with_suffix("")
            else:
                self.to_html = Path(self.report_path).with_suffix("")

        if (
            self.to_html
            and not report_path.exists()
            and not self.measurements[0]
        ):
            raise argparse.ArgumentError(
                None,
                "HTML report cannot be generated, file from "
                "'--report-path' does not exist. Please, make sure the "
                "path is correct or use '--measurements' to generate new "
                "report.",
            )

        if self.measurements[0] is None and not self.to_html:
            raise argparse.ArgumentError(
                None,
                "'--measurements' have to be defined to generate new report. "
                "If only HTML version from existing report has to be "
                "rendered, please use '--to-html' flag",
            )

        if self.comparison_only and len(self.measurements) <= 1:
            KLogger.warn(
                "'--comparison-only' used, but only one measurements file"
                " provided - creating standard report"
            )
            self.comparison_only = False

        if self.root_dir is None and self.report_path is not None:
            self.root_dir = self.report_path.parent.absolute()

        if not self.img_dir and self.root_dir is not None:
            self.img_dir = self.root_dir / "img"

        if self.model_names is not None and len(self.measurements) != len(
            self.model_names
        ):
            KLogger.warning(
                "Number of model names differ from number of measurements! "
                "Ignoring --model-names argument"
            )
            self.model_names = None

        self.image_formats = {"png"}
        if not self.only_png_images:
            self.image_formats |= {"html"}

        self._report_types = report_types

    def generate_markdown_report(
        self, command: List[str] = [], draw_titles: bool = True
    ):
        """
        Generates an MyST report based on Measurements data.

        The report is saved to the file in ``outputpath``.

        Parameters
        ----------
        command : List[str]
            Full command used to render this report, split into separate lines.
        draw_titles : bool
            Should titles be drawn on the plot.
        """
        rep = ReportTypes

        reptypes = {
            rep.PERFORMANCE: performance_report,
            rep.CLASSIFICATION: classification_report,
            rep.DETECTION: detection_report,
            rep.RENODE: renode_stats_report,
            rep.TEXT_SUMMARIZATION: text_summarization_report,
        }
        comparereptypes = {
            rep.PERFORMANCE: comparison_performance_report,
            rep.CLASSIFICATION: comparison_classification_report,
            rep.DETECTION: comparison_detection_report,
            rep.RENODE: comparison_renode_stats_report,
            rep.TEXT_SUMMARIZATION: comparison_text_summarization_report,
        }

        header_data = {
            "report_name": self.report_name,
            "model_names": [],
            "command": [],
            "smaller_header": self.smaller_header,
        }

        for model_data in filter(
            lambda x: not x.get(Measurements.UNOPTIMIZED),
            self.measurementsdata,
        ):
            header_data["model_names"].append(model_data["model_name"])
            if "command" in model_data:
                header_data["command"] += model_data["command"] + [""]
            header_data[model_data["model_name"]] = model_data

        # add command only if previous one is not the same
        if any(c1 != c2 for c1, c2 in zip(header_data["command"], command)):
            header_data["command"].extend(command)

        content = ""

        if not self.skip_general_information:
            with path(reports, "header.md") as reporttemplate:
                content += create_report_from_measurements(
                    reporttemplate, header_data
                )

        if self.automl_stats:
            content += automl_report(
                self.automl_stats,
                self.img_dir,
                self.root_dir,
                self.image_formats,
                self.colors,
                draw_titles,
            )

        models_metrics = {}
        if len(self.measurementsdata) > 1:
            for _type in self.report_types:
                content += comparereptypes[_type](
                    self.measurementsdata,
                    self.img_dir,
                    self.root_dir,
                    self.image_formats,
                    cmap=self.cmap,
                    colors=self.colors,
                    draw_titles=draw_titles,
                    main_quality_metric=self.main_quality_metric,
                )
        if not self.comparison_only or self.save_summary:
            for _type in self.report_types:
                for i, model_data in enumerate(self.measurementsdata):
                    if model_data["model_name"] not in models_metrics:
                        models_metrics[model_data["model_name"]] = {
                            "metrics": [],
                            "scenarioPath": model_data.get("cfg_path", None),
                        }
                    if len(self.measurementsdata) > 1:
                        imgprefix = (
                            model_data["model_name"].replace(" ", "_") + "_"
                        )
                    else:
                        imgprefix = ""
                    additional_content, metrics = reptypes[_type](
                        model_data,
                        self.img_dir,
                        imgprefix,
                        self.root_dir,
                        self.image_formats,
                        color_offset=i,
                        cmap=self.cmap,
                        colors=self.colors,
                        draw_titles=draw_titles,
                    )
                    for metric_name, metric in metrics.items():
                        models_metrics[model_data["model_name"]][
                            "metrics"
                        ].append(
                            {
                                "type": _type,
                                "name": metric_name,
                                "value": metric,
                            }
                        )
                    if not self.comparison_only:
                        content += additional_content

        content = re.sub(r"[ \t]+$", "", content, 0, re.M)

        with open(self.report_path, "w") as out:
            out.write(content)
        if self.save_summary:
            report_summary = []
            for name, data in models_metrics.items():
                report_summary.append(data | {"modelName": name})
            with open(
                self.report_path.with_suffix(".summary.json"), "w"
            ) as out:
                json.dump(report_summary, out)

    def generate_report(
        self,
        subcommands: Optional[List[str]] = None,
        command: Optional[List[str]] = None,
    ):
        """
        Generate report.

        Parameters
        ----------
        subcommands : Optional[List[str]]
            Used subcommands from parsed arguments.
        command : Optional[List[str]]
            A list of arguments from command line.

        Raises
        ------
        argparse.ArgumentError
            if there is missing or wrong arguments
        """
        KLogger.debug(f"Measurements {self.measurements}")

        if self.measurements[0]:
            (
                self.measurementsdata,
                self.report_types,
                self.automl_stats,
            ) = Report.load_measurements_for_report(
                measurements_files=self.measurements,
                model_names=self.model_names,
                skip_unoptimized_model=self.skip_unoptimized_model,
                report_types=self.report_types,
                automl_stats_file=self.automl_stats,
            )

        if self.report_name is None and len(self.measurementsdata) > 0:
            self.report_name = Report.deduce_report_name(
                self.measurementsdata, self._report_types
            )

        # Fill missing colors with ones generated from nipy_spectral

        self.colors = KENNING_COLORS

        if len(self.measurementsdata) > len(self.colors):
            self.colors += [
                to_hex(c)
                for c in Plot._get_comparison_color_scheme(
                    len(self.measurementsdata) - len(self.colors)
                )
            ]

        SERVIS_PLOT_OPTIONS["colormap"] = self.colors
        self.cmap = RED_GREEN_CMAP

        if (
            not self.to_html
            and self.comparison_only
            and len(self.measurements) <= 1
            and AUTOML not in subcommands
        ):
            raise argparse.ArgumentError(
                None,
                "'--comparison-only' applies only if there are more "
                "than one measurements' file.",
            )

        if self.measurements[0]:
            self.img_dir.mkdir(parents=True, exist_ok=True)

            with choose_theme(
                custom_bokeh_theme=True,
                custom_matplotlib_theme=True,
            ):
                self.generate_markdown_report(command, draw_titles=False)

        if self.to_html:
            generate_html_report(
                self.report_path, self.to_html, KLogger.level == "DEBUG"
            )
