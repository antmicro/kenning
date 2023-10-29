# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with custom parser, which does not exit when custom `-h/--help` flag
is used and some required arguments are missing.
"""

from __future__ import annotations
import sys
import argparse
from gettext import gettext
from typing import Optional, List

from kenning.cli.formatter import Formatter

HELP_FLAGS = ("-h", "--help")


def print_help_from_parsers(
    prog: str,
    parents: List[argparse.ArgumentParser],
    description: Optional[str] = None,
):
    """
    Prints help message based on specified parsers.

    Parameters
    ----------
    prog : str
        Name of the program, printed at the beginning of the usage
    parents : List[argparse.ArgumentParser]
        Parsers which should be included in help message
    description : Optional[str]
        Additional information about program, printed after the usage
    """
    Parser(
        prog,
        parents=parents,
        formatter_class=Formatter,
        description=description,
        add_help=False,
    ).print_help()


class ParserHelpException(BaseException):
    """
    Exception used for printing help message based on parsers
    created during runtime.
    """

    def __init__(
        self,
        parser: Optional[argparse.ArgumentParser] = None,
        error: Optional[str] = None,
    ):
        """
        ParserHelpException constructor.

        Parameters
        ----------
        parser : Optional[argparse.ArgumentParser]
            Parser which should be included in help message
        error : Optional[str]
            Error message that will be printed after help
        """
        self.parser = parser
        self.error = error

    def print(
        self,
        parser: Parser,
        parents: Optional[List[argparse.ArgumentParser]] = None,
        description: Optional[str] = None,
    ):
        """
        Prints help messaged based on exception.

        Parameters
        ----------
        parser : Parser
            Base parser of the program
        parents : Optional[List[argparse.ArgumentParser]]
            Other parsers which should be included in help message
        description : Optional[str]
            Additional information about program, printed after the usage
        """
        if parents is None:
            parents = [parser]
        if description is None:
            description = parser.description
        print_help_from_parsers(
            self.parser.prog if self.parser else parser.prog,
            parents=parents
            + (
                [self.parser]
                if self.parser and self.parser not in parents
                else []
            ),
            description=description,
        )
        if self.error:
            parser.error(self.error, early_exit=True)


class Parser(argparse.ArgumentParser):
    """
    Custom Parser, which raises ParserHelpException instead of printing
    help message and exiting the program.
    """

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = gettext("unrecognized arguments: {}")
            # new: raise exception for help message
            if args.help:
                raise ParserHelpException(error=msg.format(" ".join(argv)))
            else:
                self.error(msg.format(" ".join(argv)))
        return args

    def error(self, message: str, early_exit=False, print_usage=True):
        args = {"prog": self.prog, "message": message}
        error = gettext("%(prog)s: error: %(message)s\n") % args
        # new: end program when early_exit is True
        # when help flag is present raise exception
        if early_exit:
            self.exit(0, error)
        if any(help in sys.argv[1:] for help in HELP_FLAGS):
            raise ParserHelpException(self, message)
        if print_usage:
            self.print_usage(sys.stderr)
        self.exit(2, error)
