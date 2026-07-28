#!/usr/bin/env python

# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
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

from argcomplete import FilesCompleter

from kenning.utils.resource_manager import ResourceURI

if sys.version_info.minor < 9:
    pass
else:
    pass
from kenning.cli.command_template import (
    AUTOML,
    DEFAULT_GROUP,
    OPTIMIZE,
    TEST,
    TRAIN,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.parser import get_used_subcommands
from kenning.utils.class_loader import (
    ConfigKey,
    get_command,
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
        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)

        groups[FILE_CONFIG].add_argument(
            "--json-cfg",
            "--cfg",
            help="The path to the input JSON file with configuration of the report",  # noqa: E501
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

        block_types = [ConfigKey.report]
        if OPTIMIZE not in types and TEST not in types and TRAIN not in types:
            block_types.append(ConfigKey.model_wrapper)

        other_group = groups[FLAG_CONFIG]
        CommandTemplate.add_block_flags_to_argparse(other_group, block_types)

        return parser, groups

    @staticmethod
    def _fill_missing_namespace_args(args: argparse.Namespace):
        if "json_cfg" not in args:
            args.json_cfg = None
        if "evaluate_unoptimized" not in args:
            args.evaluate_unoptimized = False

    @staticmethod
    def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
        """
        Prepares and validates parased arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments.

        Returns
        -------
        argparse.Namespace
            Validated parsed arguments.
        """
        RenderReport._fill_missing_namespace_args(args)
        return args

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        command = get_command()
        if hasattr(args, "parsed_report"):
            KLogger.debug(
                "Parsed report has been found, using already parsed report"
            )
            report = args.parsed_report

            subcommands = get_used_subcommands(args)

            return report.generate_report(subcommands, command)

        objs = RenderReport.initialize_blocks(
            args,
            not_parsed,
            [ConfigKey.report, ConfigKey.model_wrapper],
        )

        report = objs[ConfigKey.report]

        subcommands = get_used_subcommands(args)

        return report.generate_report(subcommands, command)


if __name__ == "__main__":
    sys.exit(RenderReport.scenario_run())
