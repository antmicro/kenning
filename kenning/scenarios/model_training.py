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

from argcomplete.completers import FilesCompleter

from kenning.cli.command_template import (
    TEST,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.core.exceptions import ConfigurationError
from kenning.dispatcher.block_config import (
    ConfigKey,
    set_block_direct_argument,
)
from kenning.utils.class_loader import (
    objs_from_full_dict_config,
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
        CommandTemplate.add_block_flags_to_argparse(
            groups[FLAG_CONFIG],
            [ConfigKey.model_wrapper, ConfigKey.dataset, ConfigKey.platform],
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        config = TrainModel.parse_configuration(
            args,
            not_parsed,
            [ConfigKey.model_wrapper, ConfigKey.platform, ConfigKey.dataset],
        )
        set_block_direct_argument(
            "from_file", False, config, ConfigKey.model_wrapper
        )
        objs = objs_from_full_dict_config(config)

        if ConfigKey.model_wrapper not in objs:
            raise ConfigurationError(
                "Missing ModelWrapper (required for training)"
            )

        model = objs[ConfigKey.model_wrapper]
        if ConfigKey.platform in objs:
            model.read_platform(objs[ConfigKey.platform])
        model.prepare_model()
        model.train_model()
        model.save_model(model.get_path())


if __name__ == "__main__":
    sys.exit(TrainModel.scenario_run(sys.argv))
