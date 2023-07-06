#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Script scrapping and listing available classes in kenning.
"""
import argparse
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
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.logger import get_logger


def list_classes(base_classes: List[str]):
    """
    Lists classes of given module, displays their parameters and descriptions

    Parameters
    ----------
    base_classes: str
        # TODO

    """

    kenning_base_classes = {
        'optimizers': ('kenning.compilers', Optimizer),
        'runners': ('kenning.runners', Runner),
        'dataproviders': ('kenning.dataproviders', DataProvider),
        'datasets': ('kenning.datasets', Dataset),
        'modelwrappers': ('kenning.modelwrappers', ModelWrapper),
        'onnxconversions': ('kenning.onnxconversions', ONNXConversion),
        'outputcollectors': ('kenning.outputcollectors', OutputCollector),
        'runtimes': ('kenning.runtimes', Runtime)}

    logger = get_logger()
    logger.setLevel('ERROR')

    all_subclasses = []
    for base_class in base_classes:
        subclasses = get_all_subclasses(
            modulepath=kenning_base_classes[base_class][0],
            cls=kenning_base_classes[base_class][1],
            raise_exception=False)

        for cls in subclasses:
            all_subclasses.append(f'{cls.__module__}.{cls.__qualname__}')

    logger.setLevel('INFO')

    for subclass in all_subclasses:
        print(subclass)


def main(argv):
    parser = argparse.ArgumentParser(argv[0])

    parser.add_argument(
        '--optimizers',
        help='',
        action='store_true',
    )
    parser.add_argument(
        '--runners',
        help='',
        action='store_true',
    )
    parser.add_argument(
        '--dataproviders',
        help='',
        action='store_true',
    )
    parser.add_argument(
        '--datasets',
        help='',
        action='store_true',
    )
    parser.add_argument(
        '--modelwrappers',
        help='',
        action='store_true',
    )
    parser.add_argument(
        '--onnxconversions',
        help='',
        action='store_true',
    )
    parser.add_argument(
        '--outputcollectors',
        help='',
        action='store_true',
    )
    parser.add_argument(
        '--runtimes',
        help='',
        action='store_true',
    )

    args = parser.parse_args(argv[1:])

    if not any(args.__dict__.values()):
        print('No base classes given')

    list_classes([base_class for base_class in args.__dict__.keys()
                  if args.__dict__[base_class]])


if __name__ == '__main__':
    main(sys.argv)
