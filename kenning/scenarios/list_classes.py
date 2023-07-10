#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Script scrapping and listing available subclasses in kenning, based on the
provided base class.
"""
import argparse
import os
import sys
from typing import List

from kenning.core.dataprovider import DataProvider
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.onnxconversion import ONNXConversion
from kenning.core.optimizer import Optimizer
from kenning.core.outputcollector import OutputCollector
from kenning.core.runner import Runner
from kenning.core.runtime import Runtime
from kenning.scenarios.class_info import generate_class_info
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.logger import get_logger


def list_classes(base_classes: List[str], verbosity='list'):
    """
    Lists classes of given module, displays their parameters and descriptions

    Parameters
    ----------
    base_classes: str
        List of kenning base classes subclasses of which will be listed
    verbosity: str
        Verbosity mode, available options:
        'list' - just list subclasses,
         'docstrings' - display class docstrings and their dependencies,
         'everything' - list subclasses along with their docstring,
        dependencies, input/output/argument formats
    """

    kenning_base_classes = {
        'optimizers': ('kenning.compilers', Optimizer),
        'runners': ('kenning.runners', Runner),
        'dataproviders': ('kenning.dataproviders', DataProvider),
        'datasets': ('kenning.datasets', Dataset),
        'modelwrappers': ('kenning.modelwrappers', ModelWrapper),
        'onnxconversions': ('kenning.onnxconverters', ONNXConversion),
        'outputcollectors': ('kenning.outputcollectors', OutputCollector),
        'runtimes': ('kenning.runtimes', Runtime)}

    logger = get_logger()
    logger.setLevel('ERROR')

    subclasses_dict = {}

    for base_class in base_classes:
        subclasses = get_all_subclasses(
            modulepath=kenning_base_classes[base_class][0],
            cls=kenning_base_classes[base_class][1],
            raise_exception=False,
            import_classes=False)

        subclasses_dict[kenning_base_classes[base_class][1]] = \
            [f'{module}.{class_name}' for class_name, module in
             subclasses if class_name[0] != '_']

    logger.setLevel('INFO')

    for base_class in base_classes:
        if not kenning_base_classes[base_class][1] in subclasses_dict.keys():
            continue

        print(f'{base_class.title()} '
              f'(in {kenning_base_classes[base_class][0]}):\n')

        subclass_list = subclasses_dict[kenning_base_classes[base_class][1]]

        for subclass in subclass_list:
            module_path = '.'.join(subclass.split('.')[:-1])
            class_name = subclass.split('.')[-1]

            if verbosity == 'list':
                print(f'\t{subclass}')

            if verbosity == 'docstrings':
                generate_class_info(target=module_path, class_name=class_name,
                                    docstrings=True, dependencies=True,
                                    input_formats=False, output_formats=False,
                                    argument_formats=False)

            if verbosity == 'everything':
                generate_class_info(target=module_path, class_name=class_name,
                                    docstrings=True, dependencies=True,
                                    input_formats=True, output_formats=True,
                                    argument_formats=True)
                print()

        if verbosity == 'list':
            print()


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

    available_choices_string = '['
    for base_class in base_class_arguments:
        available_choices_string += f'{base_class}, '
    available_choices_string = available_choices_string[:-2]
    available_choices_string += ']'

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
            return

    verbosity = 'list'
    if args.v:
        verbosity = 'docstrings'
    if args.vv:
        verbosity = 'everything'

    if len(args.base_classes) == 0:
        list_classes(base_class_arguments, verbosity=verbosity)
        return

    list_classes(args.base_classes, verbosity=verbosity)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main(sys.argv)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
