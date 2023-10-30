#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script for fuzzy search Kenning classes.
"""

import argparse
import os
import shutil
import sys
from typing import List, Optional, Tuple

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    SEARCH,
    ArgumentsGroups,
    CommandTemplate,
)
from kenning.scenarios.list_classes import ListClassesRunner, list_classes
from kenning.utils.logger import KLogger


class FuzzySearchClass(CommandTemplate):
    """
    Class containing `kenning search` subcommand logic.

    It uses `fzf` program to search through Kenning classes.
    """

    parse_all = True
    description = __doc__.split("\n\n")[0]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            FuzzySearchClass, FuzzySearchClass
        ).configure_parser(
            parser,
            command,
            types,
            groups,
        )
        group = parser.add_argument_group(GROUP_SCHEMA.format(SEARCH))
        group.add_argument(
            "pattern",
            help="Searched pattern",
            type=str,
            nargs="*",
            const=None,
        )
        group.add_argument(
            "--no-preview",
            help="Disable preview with class information",
            action="store_true",
        )
        group.add_argument(
            "--fzf-args",
            help="Additional arguments used for 'fzf' command",
            nargs=argparse.REMAINDER,
            default=[],
        )
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        if not shutil.which("fzf"):
            KLogger.error(
                "'fzf' cannot be found, please make sure it is installed"
            )
            return 1
        # Get all Kenning classes
        classes = [
            name.strip()
            for name in list_classes(
                ListClassesRunner.base_class_arguments, "list"
            )
            if name.lstrip().startswith("kenning.")
        ]

        # Run fuzzy search
        cls_str = "\n".join(classes)
        os.system(
            f"echo '{cls_str}' | fzf {' '.join(args.fzf_args)}"
            + (
                " --preview 'python3 -m kenning.scenarios.class_info {}"
                " --verbosity ERROR'"
                if not args.no_preview
                else ""
            )
            + (f" --query '{' '.join(args.pattern)}'" if args.pattern else ""),
        )


if __name__ == "__main__":
    sys.exit(FuzzySearchClass.scenario_run())
