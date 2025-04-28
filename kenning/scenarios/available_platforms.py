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

import yaml

import kenning.resources.platforms as platforms
from kenning.cli.command_template import (
    GROUP_SCHEMA,
    LIST,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)


def read_platform_definitions(verbosity: str = "list") -> List[str]:
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

    Returns
    -------
    List[str]
        List of formatted strings to be printed out later
    """
    with path(platforms, "platforms.yml") as defs_path:
        with open(defs_path, "r") as f:
            data = yaml.safe_load(f)

    platforms_info = []
    for platform, info in data.items():
        details_names = []
        details_values = []

        def _add_list_details(k, v):
            details_names.append("")
            details_values.append("")
            details_names.append(k.replace("_", " ").capitalize() + ":")
            details_values.append(v[0])
            for val in v[1:]:
                details_names.append("")
                details_values.append(val)

        def _add_single_detail(k, v):
            details_names.append(k.replace("_", " ").capitalize() + ":")
            details_values.append(v)
            if len(str(v)) >= 80:
                details_names.append("")
                details_values.append("")

        display_name = info["display_name"]

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

        platforms_info.append([display_name, details_names, details_values])

    max_detail_name_len = 0
    for i in range(len(platforms_info)):
        display_n, details_ns, details_vs = platforms_info[i]
        max_detail_name_len = max(
            max_detail_name_len, max(map(lambda x: len(x), details_ns))
        )

    for i in range(len(platforms_info)):
        display_n, details_ns, details_vs = platforms_info[i]
        platforms_info[i][1] = list(
            map(lambda x: f"{x:<{max_detail_name_len + 4}}", details_ns)
        )

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
            help="Display all available platforms details",
            action="store_true",
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        verbosity = "list"
        if args.v:
            verbosity = "all"

        resulting_output = read_platform_definitions(verbosity)

        for platform_name, details_names, details_values in resulting_output:
            print("\n" + platform_name + "\n")
            for d_n, d_v in zip(details_names, details_values):
                print(f"\t{d_n}{d_v}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    result = AvailablePlatformsCommand.scenario_run()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    sys.exit(result)
