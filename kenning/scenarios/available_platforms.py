#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that provides information about available platforms.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

import json

import yaml

import kenning.resources.platforms as platforms
from kenning.cli.command_template import (
    GROUP_SCHEMA,
    LIST,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)


def read_platform_definitions(
    verbosity: str = "list", as_json: bool = False
) -> List[str]:
    """
    Reads and formats available platform information.

    Parameters
    ----------
    verbosity : str
        Verbosity mode, available options:
        Indicates the detail of the information returned.
        'list'  -  list platforms with their resc path and
                    corresponding default platforms.
        'all'   - list platfors with all information available

    as_json : bool
        Returns all information on platforms in JSON format

    Returns
    -------
    List[str]
        List of formatted strings to be printed out later
    """
    with path(platforms, "platforms.yml") as defs_path:
        with open(defs_path, "r") as f:
            data = yaml.safe_load(f)

    if as_json:
        data_json = json.dumps(data, ensure_ascii=False, indent=4)
        return [data_json]

    platforms_info = []
    for platform, info in data.items():

        def _add_single_detail(k, v):
            platforms_info.append(f"- `{k.replace('_', ' ').capitalize()}`\n")
            platforms_info.append(f"    - {v}\n")

        def _add_list_details(k, v):
            platforms_info.append(f"- `{k.replace('_', ' ').capitalize()}`\n")
            for val in v:
                platforms_info.append(f"    - {val}\n")

        platforms_info.append(f"# {info['display_name']}\n")

        _add_single_detail("platform_resc_path", info["platform_resc_path"])
        _add_single_detail("default_platform", info["default_platform"])

        del (
            info["platform_resc_path"],
            info["default_platform"],
            info["display_name"],
        )

        if verbosity == "all":
            for k, v in info.items():
                if isinstance(v, list):
                    _add_list_details(k, v)
                else:
                    _add_single_detail(k, v)

    return platforms_info


class AvailablePlatformsCommand(CommandTemplate):
    """
    Command template for providing available platforms details.
    """

    parse_all = True
    description = __doc__.split("\n\n")[0]
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
        resolve_conflict: bool = False,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            AvailablePlatformsCommand,
            AvailablePlatformsCommand,
        ).configure_parser(parser, command, types, groups)

        list_group = parser.add_argument_group(GROUP_SCHEMA.format(LIST))

        list_group.add_argument(
            "-v",
            help="Display all available platforms details.",
            action="store_true",
        )
        list_group.add_argument(
            "--json",
            help="Display all platforms data in JSON format.",
            action="store_true",
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        verbosity = "list"
        if args.v:
            verbosity = "all"

        resulting_output = read_platform_definitions(verbosity, args.json)

        if args.json:
            print(resulting_output[0])
            return 0

        resulting_content = "".join(resulting_output)

        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        md = Markdown(resulting_content)
        console.print(md)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    result = AvailablePlatformsCommand.scenario_run()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    sys.exit(result)
