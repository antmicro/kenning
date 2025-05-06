#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Script collects and lists available subclasses in Kenning, based on the
provided base class.
"""

import argparse
import errno
import os
import sys
from functools import cache
from typing import List, Optional, Tuple, Type

from argcomplete.completers import ChoicesCompleter

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    LIST,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.utils.class_info import generate_class_info, get_class_description
from kenning.utils.class_loader import (
    AUTOML,
    DATA_CONVERTERS,
    DATA_PROVIDERS,
    DATASETS,
    MODEL_WRAPPERS,
    ONNX_CONVERSIONS,
    OPTIMIZERS,
    OUTPUT_COLLECTORS,
    PLATFORMS,
    RUNNERS,
    RUNTIME_PROTOCOLS,
    RUNTIMES,
    get_all_subclasses,
    get_base_classes_dict,
)


@cache
def get_subclasses_dict(base_classes: frozenset[str]) -> dict[Type, List[str]]:
    """
    Get subclasses of the provided base classes.

    Parameters
    ----------
    base_classes : frozenset[str]
        A frozen set of base classes, for which subclasses will be listed.

    Returns
    -------
    dict[Type, List[str]]
        Dictionary with base classes as keys and a list
        of corresponding subclasses as values (in a textual form).
    """
    subclasses_dict = {}
    kenning_base_classes = get_base_classes_dict()

    for base_class in base_classes:
        subclasses = get_all_subclasses(
            module_path=kenning_base_classes[base_class][0],
            cls=kenning_base_classes[base_class][1],
            raise_exception=False,
            import_classes=False,
        )

        subclasses_dict[kenning_base_classes[base_class][1]] = [
            f"{module}.{class_name}"
            for class_name, module in subclasses
            if not class_name[0].startswith("_")
        ]
    return subclasses_dict


def list_classes(
    base_classes: List[str],
    verbosity: str = "list",
    prefix: str = "",
) -> List[str]:
    """
    Lists classes of given module, displays their parameters and descriptions.

    Parameters
    ----------
    base_classes : List[str]
        List of Kenning base classes subclasses of which will be listed
    verbosity : str
        Verbosity mode, available options:
        'list' - just list subclasses,
        'docstrings' - display class docstrings and their dependencies,
        'all' - list subclasses along with their docstring,
        dependencies, input/output/argument formats
        'autocomplete' - path of subclasses with descriptions
    prefix : str
        List classes which contains prefix

    Returns
    -------
    List[str]
        List of formatted strings to be printed out later
    """
    kenning_base_classes = get_base_classes_dict()
    subclasses_dict = get_subclasses_dict(frozenset(base_classes))

    # list of strings to be printed later
    resulting_output = []

    for base_class in base_classes:
        if kenning_base_classes[base_class][1] not in subclasses_dict.keys():
            continue

        if verbosity != "autocomplete":
            resulting_output.append(
                f"# {base_class.title()} "
                f"(in `{kenning_base_classes[base_class][0]}`)\n\n"
            )

        subclass_list = subclasses_dict[kenning_base_classes[base_class][1]]

        for subclass in subclass_list:
            if not subclass.startswith(prefix):
                continue
            module_path = ".".join(subclass.split(".")[:-1])
            class_name = subclass.split(".")[-1]

            if verbosity == "autocomplete":
                extracted_description = get_class_description(
                    target=module_path, class_name=class_name
                )
                description_lines = extracted_description.strip().split("\n")
                abbrev_description = []
                for line in description_lines:
                    if len(line) == 0:
                        break
                    abbrev_description.append(line)
                abbrev_description = " ".join(abbrev_description)
                if len(abbrev_description) == 0:
                    abbrev_description = class_name
                resulting_output.append((subclass, abbrev_description))

            elif verbosity == "list":
                resulting_output.append(f"* {subclass}\n")

            elif verbosity == "docstrings":
                output = generate_class_info(
                    target=module_path,
                    class_name=class_name,
                    docstrings=True,
                    dependencies=True,
                    input_formats=False,
                    output_formats=False,
                    argument_formats=False,
                )

                resulting_output += output

            elif verbosity == "all":
                output = generate_class_info(
                    target=module_path,
                    class_name=class_name,
                    docstrings=True,
                    dependencies=True,
                    input_formats=True,
                    output_formats=True,
                    argument_formats=True,
                )

                resulting_output += output

        if verbosity == "list":
            resulting_output.append("\n")

    return resulting_output


class ListClassesRunner(CommandTemplate):
    """
    Command template for listing available Kenning classes.
    """

    parse_all = True
    description = __doc__.split("\n\n")[0]
    ID = generate_command_type()

    base_class_arguments = [
        PLATFORMS,
        OPTIMIZERS,
        RUNNERS,
        DATA_PROVIDERS,
        DATA_CONVERTERS,
        DATASETS,
        MODEL_WRAPPERS,
        ONNX_CONVERSIONS,
        OUTPUT_COLLECTORS,
        PLATFORMS,
        RUNTIME_PROTOCOLS,
        RUNTIMES,
        AUTOML,
    ]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            ListClassesRunner, ListClassesRunner
        ).configure_parser(parser, command, types, groups)

        list_group = parser.add_argument_group(GROUP_SCHEMA.format(LIST))

        available_choices_string = "[ "
        for base_class in ListClassesRunner.base_class_arguments:
            available_choices_string += f"{base_class}, "
        available_choices_string = available_choices_string[:-2]
        available_choices_string += " ]"

        list_group.add_argument(
            "base_classes",
            help="Base classes of a certain group of modules. List of zero or"
            " more base classes. Providing zero base classes will print"
            " information about all of them. The default verbosity will"
            " only list found subclasses.\n\nAvailable choices: "
            f"{available_choices_string}",
            nargs="*",
        ).completer = ChoicesCompleter(ListClassesRunner.base_class_arguments)

        list_group.add_argument(
            "-v",
            help="Also display class docstrings along with dependencies and"
            " their availability",
            action="store_true",
        )
        list_group.add_argument(
            "-vv",
            help="Display all available information, that is: docstrings,"
            " dependencies, input and output formats and specification of"
            " the arguments",
            action="store_true",
        )
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        for base_class in args.base_classes:
            if base_class not in ListClassesRunner.base_class_arguments:
                print(f"{base_class} is not a valid base class argument")
                sys.exit(errno.EINVAL)

        verbosity = "list"

        if args.v:
            verbosity = "docstrings"
        if args.vv:
            verbosity = "all"

        resulting_output = list_classes(
            args.base_classes
            if len(args.base_classes) > 0
            else ListClassesRunner.base_class_arguments,
            verbosity=verbosity,
        )
        resulting_content = "".join(resulting_output)

        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        md = Markdown(resulting_content)
        console.print(md)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    result = ListClassesRunner.scenario_run()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    sys.exit(result)
