#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that generates report files based on Measurements JSON output.

It requires providing the report type and JSON file to extract data from.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from argcomplete import FilesCompleter

from kenning.cli.completers import ClassPathCompleter
from kenning.utils.resource_manager import ResourceURI

if sys.version_info.minor < 9:
    pass
else:
    pass
from kenning.cli.command_template import (
    AUTOML,
    DEFAULT_GROUP,
    REPORT,
    ArgumentsGroups,
    CommandTemplate,
    ParserHelpException,
    generate_command_type,
)
from kenning.core.report import Report
from kenning.report.markdown_report import MarkdownReport
from kenning.utils.class_loader import (
    ConfigKey,
    get_command,
    load_class_by_type,
    objs_from_json,
)
from kenning.utils.logger import KLogger

FILE_CONFIG = "Inference configuration with JSON/YAML file"
FLAG_CONFIG = "Inference configuration with flags"
ARGS_GROUPS = {
    FILE_CONFIG: f"Configuration with pipeline defined in JSON/YAML file. This section is not compatible with '{FLAG_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
    FLAG_CONFIG: f"Configuration with flags. This section is not compatible with '{FILE_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
}


class RenderReport(CommandTemplate):
    """
    Command-line template for rendering reports.
    """

    parse_all = False
    description = __doc__.split("\n\n")[0]
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(RenderReport, RenderReport).configure_parser(
            parser, command, types, groups
        )

        other_group = groups[DEFAULT_GROUP]
        # Group specific for this scenario,
        # doesn't have to be added to global groups
        required_prefix = "* "
        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)

        groups[FILE_CONFIG].add_argument(
            "--json-cfg",
            "--cfg",
            help=f"{required_prefix}The path to the input JSON file with configuration of the report",  # noqa: E501
            type=ResourceURI,
        ).completer = FilesCompleter(
            allowednames=("*.json", "*.yaml", "*.yml")
        )

        if AUTOML not in types:
            other_group.add_argument(
                "--automl-stats",
                help="Path to the JSON file with statistics during the AutoML run",  # noqa: E501
                type=Path,
                default=None,
            )

        other_group = groups[FLAG_CONFIG]
        other_group.add_argument(
            "--report-cls",
            help="Report type that will be used in report generation",
            default="MarkdownReport",
        ).completer = ClassPathCompleter(REPORT)

        return parser, groups

    @staticmethod
    def _fill_missing_namespace_args(args: argparse.Namespace):
        if "json_cfg" not in args:
            args.json_cfg = None
        if "evaluate_unoptimized" not in args:
            args.evaluate_unoptimized = False

    @staticmethod
    def prepare_args(
        args: argparse.Namespace, required_flags: List[str]
    ) -> argparse.Namespace:
        """
        Prepares and validates parased arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments.
        required_flags : List[str]
            Flags required for this command.

        Returns
        -------
        argparse.Namespace
            Validated parsed arguments.
        """
        RenderReport._fill_missing_namespace_args(args)
        return args

    @staticmethod
    def _run_from_cfg(
        args: argparse.Namespace,
        command: List[str],
        not_parsed: List[str] = [],
        **kwargs,
    ):
        from kenning.cli.config import get_used_subcommands

        with open(args.json_cfg, "r") as f:
            json_cfg = yaml.safe_load(f)

        if ConfigKey.report.name not in json_cfg.keys():
            KLogger.debug(
                f"No {ConfigKey.report.name} in config file ,"
                " loading from command line"
            )
            # Get it from argument line if definition not present in cfg file
            return RenderReport._run_from_flags(args, command, not_parsed)

        # set default type for Report
        if (
            ConfigKey.report.name in json_cfg.keys()
            and "type" not in json_cfg[ConfigKey.report.name].keys()
        ):
            json_cfg[ConfigKey.report.name]["type"] = args.report_cls
        elif ConfigKey.report.name not in json_cfg.keys():
            json_cfg[ConfigKey.report.name] = {
                "parameters": {},
                "type": args.report_cls,
            }

        report_type = json_cfg[ConfigKey.report.name]["type"]

        KLogger.debug(f"Selected report type: {report_type}")

        objs = objs_from_json(
            json_cfg, set([ConfigKey.report]), override=(args, not_parsed)
        )

        report = objs[ConfigKey.report]

        subcommands = get_used_subcommands(args)

        return report.generate_report(subcommands, command)

    @staticmethod
    def _run_from_flags(
        args: argparse.Namespace,
        command: List[str],
        not_parsed: List[str] = [],
        **kwargs,
    ):
        from kenning.cli.config import get_used_subcommands

        reportcls: Report = load_class_by_type(
            getattr(args, "report_cls", None), REPORT
        )

        if reportcls is None:
            reportcls = MarkdownReport

        KLogger.debug(f"Using report type {reportcls}")

        parser = argparse.ArgumentParser(
            " ".join(map(lambda x: x.strip(), get_command(with_slash=False)))
            + "\n",
            parents=[reportcls.form_argparse(args)[0]],
            add_help=False,
        )

        if args.help:
            raise ParserHelpException(parser)

        for key, value in parser.parse_known_args(not_parsed, namespace=args)[
            0
        ].__dict__.items():
            setattr(args, key, value)

        report = reportcls.from_argparse(args)

        subcommands = get_used_subcommands(args)

        return report.generate_report(subcommands, command)

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        command = get_command()

        flag_config_args = []

        if hasattr(args, "parsed_report"):
            KLogger.debug(
                "Parsed report has been found, using already parsed report"
            )
            report = args.parsed_report

            from kenning.cli.config import get_used_subcommands

            subcommands = get_used_subcommands(args)

            return report.generate_report(subcommands, command)

        args = RenderReport.prepare_args(args, flag_config_args)

        if args.json_cfg is not None:
            KLogger.debug("Running using parameters from config file")
            if args.help:
                raise ParserHelpException
            return RenderReport._run_from_cfg(
                args, command, not_parsed, **kwargs
            )
        return RenderReport._run_from_flags(
            args, command, not_parsed, **kwargs
        )


if __name__ == "__main__":
    sys.exit(RenderReport.scenario_run())
