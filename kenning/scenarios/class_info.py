#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script that provides information about a given kenning class.

More precisely, it displays:
* module and class docstring
* imported dependencies, including information if they are available or not
* supported input and output formats (lots of the classes provide such
  information one way or the other)
* node's parameters, with their help and default values
"""
import argparse
import os
import sys
from typing import List, Optional, Tuple

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    INFO,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.completers import ClassPathCompleter
from kenning.utils.class_info import generate_class_info
from kenning.utils.class_loader import (
    get_module_path,
    is_class_name,
)


class ClassInfoRunner(CommandTemplate):
    """
    Command template for providing Kenning class details.
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
            ClassInfoRunner, ClassInfoRunner
        ).configure_parser(parser, command, types, groups)

        info_group = parser.add_argument_group(GROUP_SCHEMA.format(INFO))

        info_group.add_argument(
            "target",
            help="Module-like path of the module or class "
            "(e.g. kenning.optimizers.onnx)",
            type=str,
        ).completer = ClassPathCompleter()
        info_group.add_argument(
            "--docstrings",
            help="Display class docstrings",
            action="store_true",
        )
        info_group.add_argument(
            "--dependencies",
            help="Display class dependencies",
            action="store_true",
        )
        info_group.add_argument(
            "--input-formats",
            help="Display class input formats",
            action="store_true",
        )
        info_group.add_argument(
            "--output-formats",
            help="Display output formats",
            action="store_true",
        )
        info_group.add_argument(
            "--argument-formats",
            help="Display the argument specification",
            action="store_true",
        )
        info_group.add_argument(
            "--load-class-with-args",
            help="Provide arguments in the format specified by argument "
            "structure to create an instance of the specified class "
            "(e.g. --model-path model.onnx). "
            "This option can be used to gain access to more detailed "
            "information than just static code analysis.",
            nargs=argparse.REMAINDER,
        )
        info_group.add_argument(
            "-md",
            "-markdown",
            help="Display information in a raw Markdown",
            action="store_true",
        )
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        from kenning.cli.config import USED_SUBCOMMANDS

        args_dict = {
            k: v
            for k, v in vars(args).items()
            if v is not None
            and k
            not in (
                "help",
                "verbosity",
                "__seq_0",
                USED_SUBCOMMANDS,
            )
        }
        del args_dict["md"]

        # if no flags are given, set all of them to True (display everything)
        if not any([v for v in args_dict.values() if isinstance(v, bool)]):
            for k, v in args_dict.items():
                args_dict[k] = True if isinstance(v, bool) else v

        # Handle a class name without a module path.
        target = args_dict["target"]
        if is_class_name(target):
            args_dict["target"] = get_module_path(target)

        resulting_output = generate_class_info(**args_dict)

        if args.md:
            for line in resulting_output:
                print(line, end="")
        else:
            from rich.console import Console
            from rich.markdown import Markdown

            resulting_content = "".join(resulting_output)
            console = Console()
            md = Markdown(resulting_content)
            console.print(md)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    ret = ClassInfoRunner.scenario_run()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    sys.exit(ret)
