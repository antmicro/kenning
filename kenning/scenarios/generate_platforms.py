# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Script preparing platform definitions based on different sources.

Currently supported sources:
* Zephyr - it requires prepared full repository and SDK,
as well as installed `kenning[zephyr]` dependencies.
Moreover, `ZEPHYR_BASE` environmental variable has to point to
the cloned Zephyr repo.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from argcomplete.completers import FilesCompleter

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.config import GENERATE_PLATFORMS
from kenning.utils.logger import KLogger

ZEPHYR = "zephyr"
SOURCE_GROUP_TEMPLATE = (
    "Arguments for '{}' source, the ones with '*' are required"
)


class GeneratePlatformsCommand(CommandTemplate):
    """
    Command generating platform definitions.
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
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            GeneratePlatformsCommand, GeneratePlatformsCommand
        ).configure_parser(parser, command, types, groups)

        generate_options = parser.add_argument_group(
            GROUP_SCHEMA.format(GENERATE_PLATFORMS)
        )

        # Common arguments
        generate_options.add_argument(
            "source",
            help="The source of a data for platform definitions generations",
            choices=[ZEPHYR],
        )
        generate_options.add_argument(
            "--platforms",
            help="Path where YAML file with platforms definitions should be saved",  # noqa: E501
            type=Path,
            required=True,
        ).completer = FilesCompleter(
            allowednames=("yaml", "yml"), directories=False
        )
        required_prefix = "* "

        # zephyr-specific arguments
        zephyr_options = parser.add_argument_group(
            SOURCE_GROUP_TEMPLATE.format(ZEPHYR)
        )
        zephyr_options.add_argument(
            "--zephyr-base",
            help=f"{required_prefix}Path to the directory with Zephyr, by default uses value from ZEPHYR_BASE variable",  # noqa: E501
            type=str,
            default=os.environ.get("ZEPHYR_BASE", None),
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        KLogger.set_verbosity(args.verbosity)
        RUN_METHODS = {
            ZEPHYR: GeneratePlatformsCommand.run_zephyr,
        }
        RUN_METHODS[args.source](args, **kwargs)

    @staticmethod
    def run_zephyr(args: argparse.Namespace, **kwargs):
        if (
            not args.zephyr_base
            or not (zephyr_base := Path(args.zephyr_base)).exists()
        ):
            raise argparse.ArgumentError(
                None,
                "--zephyr-base (or `ZEPHYR_BASE` environmental variable) "
                "has to point to existing Zephyr directory",
            )
        os.environ["ZEPHYR_BASE"] = str(zephyr_base.resolve())

        from kenning.utils.generate_platforms import get_platforms_definitions

        platforms = get_platforms_definitions()

        with args.platforms.open("w") as fd:
            yaml.dump(platforms, fd)
