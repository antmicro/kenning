# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing template for creating commands and their names.
"""

import argparse
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from kenning.cli.parser import HELP_FLAGS, Parser, ParserHelpException
from kenning.dispatcher.block_config import (
    CONFIG_KEY_TO_CLS_FLAG,
    KenningBlockConfigDict,
    apply_default_blocks_by_block_type,
    argparse_to_config_dict,
    filter_block_types_from_config_dict,
    merge_config_dicts,
    yaml_or_json_to_config_dict,
)
from kenning.utils.class_loader import (
    ConfigKey,
    classes_from_full_dict_config,
    objs_from_full_dict_config,
    parse_classes,
)
from kenning.utils.logger import KLogger

# Subcommands:
OPTIMIZE = "optimize"
TRAIN = "train"
TEST = "test"
REPORT = "report"
VISUAL_EDITOR = "visual-editor"
FLOW = "flow"
SERVER = "server"
FINE_TUNE = "fine-tune-optimizers"
LIST = "list"
INFO = "info"
CACHE = "cache"
SEARCH = "search"
COMPLETION = "completion"
AUTOML = "automl"
ROS = "ros"
DOWNLOAD_RESOURCES = "download-resources"
GENERATE_PLATFORMS = "generate-platforms"
AVAILABLE_PLATFORMS = "available-platforms"
HELP = {
    "flags": HELP_FLAGS,
    "msg": "show this help message and exit",
}

# Groups:
DEFAULT_GROUP = "common arguments"
GROUP_SCHEMA = "'{}' arguments"
# Internal variable for generating command ID
_COMMAND_ID = -1


ArgumentsGroups = Dict[str, argparse._ArgumentGroup]


def generate_command_type() -> int:
    """
    Generates consecutive IDs for CommandTemplate.
    """
    global _COMMAND_ID
    _COMMAND_ID += 1
    return _COMMAND_ID


class CommandTemplate(ABC):
    """
    A template which make scenarios compatible with Kenning CLI.
    """

    parse_all: bool
    description: Union[str, Dict[str, str]]
    ID = generate_command_type()

    current_command: str = ""

    default_block_classes: Dict[ConfigKey, str] = {
        ConfigKey.report: "MarkdownReport",
        ConfigKey.platform: "LocalPlatform",
    }

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
        resolve_conflict: bool = False,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        """
        Configures parser to accept needed arguments and flags
        for the scenario.

        Parameters
        ----------
        parser : Optional[argparse.ArgumentParser]
            Parser to which flags and arguments should be added.
        command : Optional[str]
            Name of the command or script used by parser.
        types : List[str]
            Used subcommands with current run.
        groups : Optional[ArgumentsGroups]
            Groups of arguments used by parser.
        resolve_conflict : bool
            Indicates if parser should try to resolve conflicts instead of
            raising an error.

        Returns
        -------
        Tuple[argparse.ArgumentParser, ArgumentsGroups]
            Tuple of configured parser and argument groups
        """
        if parser is None:
            parser = Parser(
                command,
                conflict_handler="resolve" if resolve_conflict else "error",
                add_help=False,
            )

        groups = CommandTemplate.add_groups(parser, groups, [DEFAULT_GROUP])

        groups[DEFAULT_GROUP].add_argument(
            *HELP["flags"],
            help=HELP["msg"],
            action="store_true",
        )
        groups[DEFAULT_GROUP].add_argument(
            "--verbosity",
            help="Verbosity level",
            choices=[
                "NOTSET",
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ],
            default="INFO",
        )

        return parser, groups

    @staticmethod
    def add_groups(
        parser: argparse.ArgumentParser,
        groups: Optional[ArgumentsGroups],
        new_groups: Union[List[str], Dict[str, str]],
    ) -> ArgumentsGroups:
        """
        Add empty argument groups with provided titles.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to which flags and arguments should be added.
        groups : Optional[ArgumentsGroups]
            Groups of arguments used by parser.
        new_groups : Union[List[str], Dict[str, str]]
            List of groups titles to be added or dictionary with titles as keys
            and descriptions as values.

        Returns
        -------
        ArgumentsGroups
            Argument groups with new groups added.

        Raises
        ------
        TypeError
            Rased when given value is of invalid type
        """
        if groups is None:
            groups = dict()

        if isinstance(new_groups, dict):
            for title, description in new_groups.items():
                if title not in groups:
                    groups[title] = parser.add_argument_group(
                        title, description
                    )
        elif isinstance(new_groups, list):
            for title in new_groups:
                if title not in groups:
                    groups[title] = parser.add_argument_group(title)
        else:
            raise TypeError(f"Invalid type of new_groups: {type(new_groups)}")

        return groups

    @staticmethod
    def add_block_flags_to_argparse(
        flag_group: argparse._ArgumentGroup, block_types: List[ConfigKey]
    ):
        """
        Adds to the chosen argparse group CLI flags for setting blocks of
        selected block types (such as --modelwrapper-cls).

        Parameters
        ----------
        flag_group: argparse._ArgumentGroup
            Group that the flags will be added to.
        block_types: List[ConfigKey]
            Block types, flags for which are supposed to be parsed.
        """
        from kenning.cli.completers import (
            AUTOML,
            DATASETS,
            MODEL_WRAPPERS,
            OPTIMIZERS,
            PLATFORMS,
            RUNTIME_PROTOCOLS,
            RUNTIMES,
            ClassPathCompleter,
        )

        help_and_completer_map = {
            ConfigKey.optimizers: (
                OPTIMIZERS,
                "Optimizer-based class with compiling routines to import",
            ),
            ConfigKey.dataset: (
                DATASETS,
                "Dataset-based class with dataset to import",
            ),
            ConfigKey.model_wrapper: (
                MODEL_WRAPPERS,
                "ModelWrapper-based class with inference implementation to import",  # noqa: E501
            ),
            ConfigKey.platform: (
                PLATFORMS,
                "Platform-based class that wraps platform being tested",
            ),
            ConfigKey.report: (
                REPORT,
                "Report-based class with report generation code",
            ),
            ConfigKey.runtime: (
                RUNTIMES,
                "Runtime-based class with the implementation of model runtime",
            ),
            ConfigKey.protocol: (
                RUNTIME_PROTOCOLS,
                "Protocol-based class with the implementation of communication between inference tester and inference runner",  # noqa: E501
            ),
            ConfigKey.automl: (
                AUTOML,
                "AutoML-based class with AutoML flow implementation",
            ),
        }
        for block_type in block_types:
            flag_group.add_argument(
                CONFIG_KEY_TO_CLS_FLAG[block_type],
                help=help_and_completer_map[block_type][1],
                default=None,
            ).completer = ClassPathCompleter(
                help_and_completer_map[block_type][0]
            )

    @staticmethod
    @abstractmethod
    def run(
        args: argparse.Namespace, not_parsed: List[str] = [], **kwargs: Any
    ) -> Optional[int]:
        """
        The method containing logic of the scenario.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed and validated arguments used for this scenario.
        not_parsed : List[str]
            Additional arguments which haven't been parsed yet.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Optional[int]
            Status of executed scenario.
        """
        ...

    @classmethod
    def scenario_run(cls, argv: Optional[List[str]] = None) -> Optional[int]:
        """
        The method for running command as a scenario.

        Is manages arguments and help message.

        Parameters
        ----------
        argv : Optional[List[str]]
            Argument used for the scenario

        Returns
        -------
        Optional[int]
            Status of executed scenario

        Raises
        ------
        ParserHelpException
            Raised when help is requested in arguments
        """
        if argv is None:
            argv = sys.argv
        parser, _ = cls.configure_parser()

        try:
            if cls.parse_all:
                args, not_parsed = parser.parse_args(argv[1:]), []
                if args.help:
                    raise ParserHelpException
            else:
                args, not_parsed = parser.parse_known_args(argv[1:])

            KLogger.set_verbosity(args.verbosity)
            return cls.run(args, not_parsed=not_parsed)
        except ParserHelpException as ex:
            ex.print(parser)

    @staticmethod
    def get_overridable(subcommands: List[str]) -> List[ConfigKey]:
        return []

    @classmethod
    def parse_configuration(
        cls,
        args: argparse.Namespace,
        not_parsed: List[str],
        keys: List[ConfigKey],
    ) -> KenningBlockConfigDict:
        """
        Converts parsed command line flags (including an optional path to
        YAML/JSON config file given under --cfg flag) to a standard Kenning
        configuration dict.

        Parameters
        ----------
        args: argparse.Namespace
            Parsed CLI flags.
        not_parsed: List[str]
            Flags that were not matched and parsed by argparse. Since we do
            not know what blocks will be chosen before parsing some of the
            arguments, we cannot parse flag parameters specific to blocks
            right away - therefore argparse parsing has to be done in 2 stages.
        keys: List[ConfigKey]
            List of block types required by the scenario.

        Returns
        -------
        KenningBlockConfigDict
            Standard-format Kenning configuration dict with parsed blocks and
            parameters. See 'block_config' module for details on the format.
        """
        KLogger.info(f"Parsing configuration for scenario {cls.__name__}...")
        args = cls.prepare_args(args)
        config = argparse_to_config_dict(args)
        if args.json_cfg is not None:
            KLogger.debug(f"Config file detected at {args.json_cfg}.")
            with open(args.json_cfg, "r") as f:
                cfg = yaml.safe_load(f)
                config = merge_config_dicts(
                    yaml_or_json_to_config_dict(cfg), config
                )
        config = apply_default_blocks_by_block_type(
            cls.default_block_classes, config
        )

        def flatten_list(irregular_list):
            return (
                [
                    element
                    for item in irregular_list
                    for element in flatten_list(item)
                ]
                if type(irregular_list) is list
                else [irregular_list]
            )

        classes = flatten_list(
            list(classes_from_full_dict_config(config).values())
        )
        args = parse_classes(
            classes,
            args,
            not_parsed,
        )
        config = merge_config_dicts(config, argparse_to_config_dict(args))
        config = filter_block_types_from_config_dict(keys, config)
        KLogger.info(
            f"Collected blocks: {[block.__name__ for block in classes]}"
        )
        return config

    @classmethod
    def initialize_blocks(
        cls,
        args: argparse.Namespace,
        not_parsed: List[str],
        keys: List[ConfigKey],
    ) -> Dict[ConfigKey, Any]:
        """
        Converts parsed command line flags (including an optional path to
        YAML/JSON config file given under --cfg flag) to a standard Kenning
        configuration dict. Then uses this dict to create block objects with
        the required parameters.

        Parameters
        ----------
        args: argparse.Namespace
            Parsed CLI flags.
        not_parsed: List[str]
            Flags that were not matched and parsed by argparse. Since we do
            not know what blocks will be chosen before parsing some of the
            arguments, we cannot parse flag parameters specific to blocks
            right away - therefore argparse parsing has to be done in 2 stages.
        keys: List[ConfigKey]
            List of block types required by the scenario.

        Returns
        -------
        Dict[ConfigKey, Any]
            Block objects sorted by block type. For ConfigKey.optimizers block
            type, the value is a list of objects (because there can be many
            optimizers). For all other block types, the value is an object.
        """
        KLogger.info(f"Building Kenning blocks for scenario {cls.__name__}...")
        return objs_from_full_dict_config(
            cls.parse_configuration(args, not_parsed, keys)
        )
