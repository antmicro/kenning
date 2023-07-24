# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with main logic of Kenning CLI
"""

import argparse
import sys
from typing import List, Dict, Generator, Tuple

from kenning.cli.formatter import Formatter
from kenning.cli.config import (
    HELP,
    SEQUENCED_COMMANDS,
    BASIC_COMMANDS,
    AVAILABLE_COMMANDS,
    MAP_COMMAND_TO_SCENARIO,
    SUBCOMMANDS,
)
from kenning.utils.excepthook import \
    MissingKenningDependencies, find_missing_optional_dependency


SUB_DEST_FORM = "__seq_{}"


def get_all_sequences(
    sequence: List[str],
) -> Generator[List[str], None, None]:
    """
    Yields possible sequences of commands.

    Parameters
    ----------
    sequence : List[str]
        Sequence with commands in right order

    Yields
    ------
    List[str] :
        Sequence of commands
    """
    for i in range(len(sequence)):
        yield sequence[i:]


def create_subcommands(
    subparser: argparse._SubParsersAction,
    names: List[str],
    number: int = 0
) -> Dict[Tuple[str], argparse.ArgumentParser]:
    """
    Creates nested subcommands from list of names.

    Parameters
    ----------
    subparser : argparse._SubParsersAction
        Object which can create parsers
    names : List[str]
        Sequence of subcommands
    number : int
        Depth of subparser

    Returns
    -------
    Dict[Tuple[str], argparse.ArgumentParser] :
        Dictionary of parsers associated with sequence of subcommands
    """
    parsers = {}
    parser = None
    for i, name in enumerate(names):
        if parser:
            subparser = parser.add_subparsers(
                title=SUBCOMMANDS, dest=SUB_DEST_FORM.format(number + i))
        desc = MAP_COMMAND_TO_SCENARIO[name].description
        if not isinstance(desc, str):
            desc = desc[name]
        parser = subparser.add_parser(
            name,
            help=desc.split('.', 1)[0],
            add_help=False,
        )
        parsers[tuple(names[:i + 1])] = parser
    return parsers


def setup_base_parser() -> Tuple[argparse.ArgumentParser, Dict[Tuple[str], argparse.ArgumentParser]]:  # noqa: E501
    """
    Sets up parser containing only subcommands and help message.

    Returns
    -------
    argparse.ArgumentParser :
        Created parser
    Dict[Tuple[str], argparse.ArgumentParser] :
        Dictionary of parsers associated with sequence of subcommands
    """
    parser = argparse.ArgumentParser(
        prog="kenning",
        description="Command-line interface for Kenning",
        conflict_handler='resolve',
        formatter_class=Formatter,
        add_help=False,
    )
    parsers = {}
    subparsers = parser.add_subparsers(
        title=SUBCOMMANDS, dest=SUB_DEST_FORM.format(0))

    flag_group = parser.add_argument_group('Flags')
    flag_group.add_argument(
        *HELP["flags"],
        action='store_true',
        help=HELP["msg"],
    )

    for sequence in get_all_sequences(SEQUENCED_COMMANDS):
        parsers.update(create_subcommands(subparsers, sequence))

    for subcommand in BASIC_COMMANDS:
        parsers[(subcommand,)] = subparsers.add_parser(
            subcommand,
            help=MAP_COMMAND_TO_SCENARIO[subcommand].description.split('.')[0],
            add_help=False,
        )
    return parser, parsers


def main():
    """
    The entrypoint of Kenning CLI.

    Creates and manages parsers, runs subcommands, and handle errors.
    """
    parser, parsers = setup_base_parser()

    # Get only subcommands and help
    i = 1
    while i < len(sys.argv) and \
            sys.argv[i] in AVAILABLE_COMMANDS:
        i += 1

    # Parse subcommands and help
    args, rem = parser.parse_known_args(sys.argv[1:i])
    args, rem = parser.parse_known_args(args=rem, namespace=args)
    rem += sys.argv[i:]

    # Retrieve parsed subcommands
    i = 0
    subcommands = []
    while getattr(args, SUB_DEST_FORM.format(i), None) is not None:
        subcommands.append(getattr(args, SUB_DEST_FORM.format(i)))
        i += 1

    if not subcommands:
        parser.print_help()
        if rem:
            parser.exit(
                2, f"{parser.prog}: error: '{rem[0]}' doesn't match any subcommand")  # noqa: E501
        return

    # Creating parent-parsers from scenarios and completing description
    groups = {}
    parse_all = True
    used_configs = set()
    parents, description = [], []
    seq_description = len(subcommands) > 1
    if seq_description:
        description.append("Running in sequence:")
    for subcommand in subcommands:
        scenario = MAP_COMMAND_TO_SCENARIO[subcommand]
        if isinstance(scenario.description, Dict):
            desc = scenario.description[subcommand]
        else:
            desc = scenario.description[1:]
        description.append(
            ('- ' if seq_description else '') + desc)
        if scenario.configure_parser not in used_configs:
            parents.append(scenario.configure_parser(
                types=subcommands, groups=groups)[0]
            )
            used_configs.add(scenario.configure_parser)
            parse_all = parse_all and scenario.parse_all

    # The main parser without subcommands
    parser = argparse.ArgumentParser(
        "kenning "+" ".join(subcommands),
        conflict_handler='resolve',
        parents=parents,
        formatter_class=Formatter,
        description='\n'.join(description),
        add_help=False,
    )

    # Print help, including possible subcommands
    if args.help:
        argparse.ArgumentParser(
            "kenning "+" ".join(subcommands),
            conflict_handler='resolve',
            parents=[parsers[tuple(subcommands)], parser],
            formatter_class=Formatter,
            description='\n'.join(description),
            add_help=False,
        ).print_help()
        return

    # Parse arguments
    if parse_all:
        args = parser.parse_args(
            args=sys.argv[1 + len(subcommands):], namespace=args)
        rem = []
    else:
        # Parse only known args if scenario will add more arguments
        args, rem = parser.parse_known_args(
            sys.argv[1 + len(subcommands):], namespace=args)

    # Run subcommands
    used_functions = set()
    for subcommand in subcommands:
        run = MAP_COMMAND_TO_SCENARIO[subcommand].run
        if run in used_functions:
            continue
        try:
            run(args, not_parsed=rem)
        except ModuleNotFoundError as er:
            extras = find_missing_optional_dependency(er.name)
            if not extras:
                raise
            er = MissingKenningDependencies(
                name=er.name, path=er.path, optional_dependencies=extras)
            raise

        except argparse.ArgumentError as er:
            parser.error(er.message)

        used_functions.add(run)
