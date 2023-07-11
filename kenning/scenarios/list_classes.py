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
from typing import List
import errno

from kenning.scenarios.class_info import generate_class_info
from kenning.utils.class_loader import get_all_subclasses,\
    get_base_classes_dict


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
    List of formatted strings to be printed out later
    """

    kenning_base_classes = get_base_classes_dict()

    subclasses_dict = {}

    # list of strings to be printed later
    resulting_output = []

    for base_class in base_classes:
        subclasses = get_all_subclasses(
            modulepath=kenning_base_classes[base_class][0],
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
                # print(f'\t{subclass}')

            if verbosity == 'docstrings':
                output = generate_class_info(target=module_path,
                                             class_name=class_name,
                                             docstrings=True,
                                             dependencies=True,
                                             input_formats=False,
                                             output_formats=False,
                                             argument_formats=False)
                resulting_output += output

            if verbosity == 'all':
                output = generate_class_info(target=module_path,
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


def main(argv):
    parser = argparse.ArgumentParser(argv[0],
                                     formatter_class=argparse.
                                     RawTextHelpFormatter)

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

    available_choices_string = '\n'
    for base_class in base_class_arguments:
        available_choices_string += f'  * {base_class}\n'
    available_choices_string = available_choices_string[:-2]
    available_choices_string += '\n'

    parser.add_argument(
        'base_classes',
        help=f'Base classes of a certain group of modules. List of zero or '
             f'more base classes. Providing zero base classes will print '
             f'information about all of them. The default verbosity will only '
             f'list found subclasses.\n\nAvailable choices: '
             f'{available_choices_string}',
        nargs='*',
    )

    parser.add_argument(
        '-v',
        help='Also display class docstrings along with dependencies and their'
             ' availability',
        action='store_true'
    )
    parser.add_argument(
        '-vv',
        help='Display all available information. That includes: docstrings,'
             ' dependencies, input and output formats and specification of '
             'the arguments',
        action='store_true'
    )

    args = parser.parse_args(argv[1:])

    for base_class in args.base_classes:
        if base_class not in base_class_arguments:
            print(f'{base_class} is not a valid base class argument')
            return errno.EINVAL

    verbosity = 'list'
    if args.v:
        verbosity = 'docstrings'
    if args.vv:
        verbosity = 'all'

    resulting_output = []

    resulting_output = list_classes(
        args.base_classes if len(args.base_classes) > 0 else base_class_arguments,  # noqa: E501
        verbosity=verbosity
    )

    for line in resulting_output:
        print(line, end='')

    return 0


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ret = main(sys.argv)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    sys.exit(ret)
