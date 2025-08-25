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
from typing import List, Optional, Tuple

import yaml
from argcomplete.completers import FilesCompleter

from kenning.cli.command_template import (
    TEST,
    ArgumentsGroups,
    CommandTemplate,
    ParserHelpException,
    generate_command_type,
)
from kenning.cli.completers import DATASETS, ClassPathCompleter
from kenning.core.model import ModelWrapper
from kenning.core.platform import Platform
from kenning.utils.args_manager import ensure_exclusive_cfg_or_flags
from kenning.utils.class_loader import (
    MODEL_WRAPPERS,
    PLATFORMS,
    ConfigKey,
    get_command,
    load_class,
    obj_from_json,
)
from kenning.utils.resource_manager import ResourceURI

FILE_CONFIG = "Train configuration with JSON/YAML file"
FLAG_CONFIG = "Train configuration with flags"
ARGS_GROUPS = {
    FILE_CONFIG: f"Configuration with parameters defined in JSON/YAML file. This section is not compatible with '{FLAG_CONFIG}'. Arguments with '*' are required",  # noqa: E501
    FLAG_CONFIG: f"Configuration with flags. This section is not compatible with '{FILE_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
}


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
        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)

        # required prefix
        def _(x):
            return f"* {x}"

        groups[FILE_CONFIG].add_argument(
            "--json-cfg",
            "--cfg",
            help=_(
                "The path to the input JSON file with configuration of the inference"  # noqa: E501
            ),
            type=ResourceURI,
        ).completer = FilesCompleter(
            allowednames=("*.json", "*.yaml", "*.yml")
        )
        groups[FLAG_CONFIG].add_argument(
            "--modelwrapper-cls",
            help=_(
                "ModelWrapper-based class with inference implementation to import"  # noqa: E501
            ),
        ).completer = ClassPathCompleter(MODEL_WRAPPERS)
        groups[FLAG_CONFIG].add_argument(
            "--dataset-cls",
            help=_("Dataset-based class with dataset to import"),
        ).completer = ClassPathCompleter(DATASETS)
        groups[FLAG_CONFIG].add_argument(
            "--platform-cls",
            help="Platform-based class that wraps platform being tested",
        ).completer = ClassPathCompleter(PLATFORMS)

        return parser, groups

    @staticmethod
    def prepare_args(args: argparse.Namespace):
        """
        Prepares and validates parased arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments.
        """
        TrainModel._ensure_args_in_namespace(args)
        TrainModel._ensure_exclusive_cfg_or_flags(args)

    @staticmethod
    def _ensure_args_in_namespace(args):
        if "json_cfg" not in args:
            args.json_cfg = None

    @staticmethod
    def _ensure_exclusive_cfg_or_flags(args: argparse.Namespace):
        flag_config_args = ("modelwrapper_cls", "dataset_cls")
        ensure_exclusive_cfg_or_flags(args, flag_config_args)

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        TrainModel.prepare_args(args)
        if args.json_cfg:
            if args.help:
                raise ParserHelpException
            return TrainModel._run_from_cfg(args, not_parsed, **kwargs)
        return TrainModel._run_from_flags(args, not_parsed, **kwargs)

    @staticmethod
    def _run_from_cfg(
        args: argparse.Namespace, not_parsed: List[str] = [], **kwargs
    ):
        if not_parsed:
            raise argparse.ArgumentError(
                None, f"unrecognized arguments: {' '.join(not_parsed)}"
            )

        with open(args.json_cfg, "r") as f:
            cfg = yaml.safe_load(f)

        dataset = obj_from_json(cfg, ConfigKey.dataset)
        model = obj_from_json(
            cfg, ConfigKey.model_wrapper, dataset=dataset, from_file=False
        )
        platform = obj_from_json(cfg, ConfigKey.platform)

        TrainModel._run(model, platform)

    @staticmethod
    def _run_from_flags(
        args: argparse.Namespace, not_parsed: List[str] = [], **kwargs
    ):
        modelwrappercls = (
            load_class(args.modelwrapper_cls)
            if args.modelwrapper_cls
            else None
        )
        datasetcls = load_class(args.dataset_cls) if args.dataset_cls else None
        platformcls = (
            load_class(args.platform_cls) if args.platform_cls else None
        )

        parser = argparse.ArgumentParser(
            " ".join(map(lambda x: x.strip(), get_command(with_slash=False))),
            parents=[]
            + (
                [modelwrappercls.form_argparse(args)[0]]
                if modelwrappercls
                else []
            )
            + ([datasetcls.form_argparse(args)[0]] if datasetcls else [])
            + ([platformcls.form_argparse(args)[0]] if platformcls else []),
            add_help=False,
        )

        if args.help:
            raise ParserHelpException(parser)
        args = parser.parse_known_args(not_parsed, namespace=args)

        dataset = datasetcls.from_argparse(args[0])

        model = modelwrappercls.from_argparse(
            dataset, args[0], from_file=False
        )
        platform = None
        if platformcls:
            platform = platformcls.from_argparse(args[0])

        TrainModel._run(model, platform)

    @staticmethod
    def _run(model: ModelWrapper, platform: Optional[Platform]):
        if platform:
            model.read_platform(platform)
        model.prepare_model()
        model.train_model()
        model.save_model(model.get_path())


if __name__ == "__main__":
    sys.exit(TrainModel.scenario_run(sys.argv))
