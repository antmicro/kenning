#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The script for training models given in ModelWrapper object with dataset given
in Dataset object.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    TEST,
    TRAIN,
    ArgumentsGroups,
    CommandTemplate,
    ParserHelpException,
    generate_command_type,
)
from kenning.cli.completers import DATASETS, MODEL_WRAPPERS, ClassPathCompleter
from kenning.utils.class_loader import get_command, load_class


class TrainModel(CommandTemplate):
    """
    Command template for training models with ModelWrapper.
    """

    parse_all = False
    description = __doc__[:-1]
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(TrainModel, TrainModel).configure_parser(
            parser, command, types, groups, TEST in types
        )
        # other_group = groups[DEFAULT_GROUP]
        train_group = parser.add_argument_group(GROUP_SCHEMA.format(TRAIN))

        train_group.add_argument(
            "--modelwrapper-cls",
            help="ModelWrapper-based class with inference implementation to import",  # noqa: E501
            required=True,
        ).completer = ClassPathCompleter(MODEL_WRAPPERS)
        train_group.add_argument(
            "--dataset-cls",
            help="Dataset-based class with dataset to import",
            required=True,
        ).completer = ClassPathCompleter(DATASETS)
        train_group.add_argument(
            "--batch-size",
            help="The batch size for training",
            type=int,
            required=True,
        )
        train_group.add_argument(
            "--learning-rate",
            help="The learning rate for training",
            type=float,
            required=True,
        )
        train_group.add_argument(
            "--num-epochs",
            help="Number of epochs to train for",
            type=int,
            required=True,
        )
        train_group.add_argument(
            "--logdir",
            help="Path to the training logs directory",
            type=Path,
            required=True,
        )
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        modelwrappercls = (
            load_class(args.modelwrapper_cls)
            if args.modelwrapper_cls
            else None
        )
        datasetcls = load_class(args.dataset_cls) if args.dataset_cls else None

        parser = argparse.ArgumentParser(
            " ".join(map(lambda x: x.strip(), get_command(with_slash=False))),
            parents=[]
            + ([modelwrappercls.form_argparse()[0]] if modelwrappercls else [])
            + ([datasetcls.form_argparse()[0]] if datasetcls else []),
            add_help=False,
        )

        if args.help:
            raise ParserHelpException(parser)
        args = parser.parse_args(not_parsed, namespace=args)

        dataset = datasetcls.from_argparse(args)
        model = modelwrappercls.from_argparse(dataset, args, from_file=False)

        args.logdir.mkdir(parents=True, exist_ok=True)

        model.prepare_model()
        model.train_model(
            args.batch_size, args.learning_rate, args.num_epochs, args.logdir
        )
        model.save_model(model.get_path())


if __name__ == "__main__":
    sys.exit(TrainModel.scenario_run(sys.argv))
