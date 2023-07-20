#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Script collects and lists available subclasses in Kenning, based on the
provided base class.
"""
import argparse
import os
import sys
from typing import List, Optional, Dict, Tuple
import errno

from kenning.scenarios.class_info import generate_class_info
from kenning.utils.class_loader import get_all_subclasses,\
    get_base_classes_dict

from kenning.cli.command_template import (
    CommandTemplate, GROUP_SCHEMA, LIST)

from kenning.utils import logger


def list_classes(base_classes: List[str], verbosity='list') -> List[str]:
    """
    Lists classes of given module, displays their parameters and descriptions

    Parameters
    ----------
    base_classes: str
        List of Kenning base classes subclasses of which will be listed
    verbosity: str
        Verbosity mode, available options:
        'list' - just list subclasses,
        'docstrings' - display class docstrings and their dependencies,
        'all' - list subclasses along with their docstring,
        dependencies, input/output/argument formats

    Returns
    -------
    List[str]: List of formatted strings to be printed out later
    """

    kenning_base_classes = get_base_classes_dict()

    subclasses_dict = {}

    # list of strings to be printed later
    resulting_output = []

    for base_class in base_classes:
        subclasses = get_all_subclasses(
            module_path=kenning_base_classes[base_class][0],
            cls=kenning_base_classes[base_class][1],
            raise_exception=False,
            import_classes=False)

        subclasses_dict[kenning_base_classes[base_class][1]] = \
            [f'{module}.{class_name}' for class_name, module in
             subclasses if class_name[0] != '_']

    for base_class in base_classes:
        if not kenning_base_classes[base_class][1] in subclasses_dict.keys():
            continue

        resulting_output.append(f'{base_class.title()} '
                                f'(in {kenning_base_classes[base_class][0]}):'
                                f'\n\n')

        subclass_list = subclasses_dict[kenning_base_classes[base_class][1]]

        for subclass in subclass_list:
            module_path = '.'.join(subclass.split('.')[:-1])
            class_name = subclass.split('.')[-1]

            if verbosity == 'list':
                resulting_output.append(f'    {subclass}\n')

            if verbosity == 'docstrings':
                output = generate_class_info(
                    target=module_path,
                    class_name=class_name,
                    docstrings=True,
                    dependencies=True,
                    input_formats=False,
                    output_formats=False,
                    argument_formats=False)

                resulting_output += output

            if verbosity == 'all':
                output = generate_class_info(
                    target=module_path,
                    class_name=class_name,
                    docstrings=True,
                    dependencies=True,
                    input_formats=True,
                    output_formats=True,
                    argument_formats=True)

                resulting_output += output

        if verbosity == 'list':
            resulting_output.append('\n')

    return resulting_output


class ListClassesRunner(CommandTemplate):
    parse_all = True
    description = __doc__.split('\n\n')[0]

    base_class_arguments = [
        'optimizers',
        'runners',
        'dataproviders',
        'datasets',
        'modelwrappers',
        'onnxconversions',
        'outputcollectors',
        'runtimes',
    ]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Dict[str, argparse._ArgumentGroup] = None,
    ) -> Tuple[argparse.ArgumentParser, Dict]:
        parser, groups = super(
            ListClassesRunner, ListClassesRunner
        ).configure_parser(
            parser,
            command,
            types,
            groups
        )

        list_group = parser.add_argument_group(GROUP_SCHEMA.format(LIST))

        available_choices_string = '\n'
        for base_class in ListClassesRunner.base_class_arguments:
            available_choices_string += f'  * {base_class}\n'
        available_choices_string = available_choices_string[:-2]
        available_choices_string += '\n'

        list_group.add_argument(
            'base_classes',
            help='Base classes of a certain group of modules. List of zero or'
                 ' more base classes. Providing zero base classes will print'
                 ' information about all of them. The default verbosity will'
                 ' only list found subclasses.\n\nAvailable choices: '
                 f'{available_choices_string}',
            nargs='*',
        )

        list_group.add_argument(
            '-v',
            help='Also display class docstrings along with dependencies and'
                 ' their availability',
            action='store_true'
        )
        list_group.add_argument(
            '-vv',
            help='Display all available information, that is: docstrings,'
                 ' dependencies, input and output formats and specification of'
                 ' the arguments',
            action='store_true'
        )
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        logger.set_verbosity(args.verbosity)

        for base_class in args.base_classes:
            if base_class not in ListClassesRunner.base_class_arguments:
                print(f'{base_class} is not a valid base class argument')
                sys.exit(errno.EINVAL)

        verbosity = 'list'

        if args.v:
            verbosity = 'docstrings'
        if args.vv:
            verbosity = 'all'

        resulting_output = list_classes(
            args.base_classes if len(args.base_classes) > 0 else ListClassesRunner.base_class_arguments,  # noqa: E501
            verbosity=verbosity
        )

        for line in resulting_output:
            print(line, end='')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    result = ListClassesRunner.scenario_run()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    sys.exit(result)
