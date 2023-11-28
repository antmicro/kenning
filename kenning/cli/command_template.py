# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing template for creating commands and their names.
"""

import argparse
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from kenning.cli.parser import HELP_FLAGS, Parser, ParserHelpException
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
HELP = {
    "flags": HELP_FLAGS,
    "msg": "show this help message and exit",
}

# Groups:
DEFAULT_GROUP = "common arguments"
GROUP_SCHEMA = "'{}' arguments"


ArgumentsGroups = Dict[str, argparse._ArgumentGroup]


class CommandTemplate(ABC):
    """
    A template which make scenarios compatible with Kenning CLI.
    """

    parse_all: bool
    description: Union[str, Dict[str, str]]

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
