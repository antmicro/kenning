# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing template for creating commands and their names
"""

import sys
import argparse
from abc import abstractstaticmethod, ABC
from typing import Dict, Optional, List, Union, Tuple

from kenning.cli.parser import Parser, ParserHelpException, HELP_FLAGS


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
HELP = {
    "flags": HELP_FLAGS,
    "msg": "show this help message and exit",
}

# Groups:
DEFAULT_GROUP = "common arguments"
GROUP_SCHEMA = "'{}' arguments"


class CommandTemplate(ABC):
    """
    A template which make scenarios compatible with Kenning CLI
    """

    parse_all: bool
    description: Union[str, Dict[str, str]]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Dict[str, argparse._ArgumentGroup] = None,
        resolve_conflict: bool = False,
    ) -> Tuple[argparse.ArgumentParser, Dict]:
        """
        Configures parser to accept needed arguments and flags
        for the scenario.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to which flags and arguments should be added.
        command : Optional[str]
            Name of the command or script used by parser.
        types : List[str]
            Used subcommands with current run.
        groups : Dict[str, argparse._ArgumentGroup]
            Groups of arguments used by parser.
        resolve_conflict : bool
            Should created parser resolve confilct instead of raising errors?

        Returns
        -------
        argparse.ArgumentParser : Configured parser
        """
        if parser is None:
            parser = Parser(
                command,
                conflict_handler='resolve' if resolve_conflict else 'error',
                add_help=False,
            )

        if groups is None:
            groups = dict()
        if DEFAULT_GROUP not in groups:
            groups[DEFAULT_GROUP] = parser.add_argument_group(DEFAULT_GROUP)

        groups[DEFAULT_GROUP].add_argument(
            *HELP["flags"],
            help=HELP["msg"],
            action="store_true",
        )
        groups[DEFAULT_GROUP].add_argument(
            '--verbosity',
            help='Verbosity level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO'
        )

        return parser, groups

    @abstractstaticmethod
    def run(
        args: argparse.Namespace,
        not_parsed: List[str] = [],
        **kwargs
    ) -> Optional[int]:
        """
        The method containing logic of the scenario.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed and validated arguments used for this scenario
        not_parsed : List[str]
            Additinal arguments which haven't been parsed yet

        Returns
        -------
        Optional[int] :
            Status of executed scenario
        """
        raise NotImplementedError

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
        Optional[int] :
            Status of executed scenario
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

            return cls.run(args, not_parsed=not_parsed)
        except ParserHelpException as ex:
            ex.print(parser)
