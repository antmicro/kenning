# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
# PYTHON_ARGCOMPLETE_OK

"""
Module with main logic of Kenning CLI.
"""

import argparse
import sys
from typing import Dict

from kenning.cli.autocompletion import configure_autocomplete
from kenning.cli.config import (
    AVAILABLE_COMMANDS,
    MAP_COMMAND_TO_SCENARIO,
    SUB_DEST_FORM,
    setup_base_parser,
)
from kenning.cli.parser import (
    Parser,
    ParserHelpException,
    print_help_from_parsers,
)
from kenning.utils.excepthook import (
    MissingKenningDependencies,
    find_missing_optional_dependency,
)
from kenning.utils.logger import KLogger


def main():
    """
    The entrypoint of Kenning CLI.

    Creates and manages parsers, runs subcommands, and handle errors.
    """
    verbosity = "WARNING"
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--verbosity"):
            if "=" in arg:
                _, verbosity = arg.split("=")
            elif len(sys.argv) > i + 1:
                verbosity = sys.argv[i + 1]
            else:
                raise ValueError(
                    f"Argument {arg} requires a value, e.g. --verbosity=INFO"
                )
            break
    KLogger.set_verbosity(level=verbosity)
    configure_autocomplete()
    parser, parsers = setup_base_parser()

    # Get only subcommands and help
    i = 1
    while i < len(sys.argv) and sys.argv[i] in AVAILABLE_COMMANDS:
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
                2,
                f"{parser.prog}: error: '{rem[0]}' doesn't match any subcommand",  # noqa: E501
            )
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
        description.append(("- " if seq_description else "") + desc)
        if scenario.configure_parser not in used_configs:
            parents.append(
                scenario.configure_parser(types=subcommands, groups=groups)[0]
            )
            used_configs.add(scenario.configure_parser)
            parse_all = parse_all and scenario.parse_all
    description = "\n".join(description)

    # The main parser without subcommands
    parser = Parser(
        "kenning " + " ".join(subcommands),
        conflict_handler="resolve",
        parents=parents,
        description=description,
        add_help=False,
    )

    def _print_help():
        """
        Prints help message for non sequenced subcommands.
        """
        print_help_from_parsers(
            "kenning " + " ".join(subcommands),
            [parsers[tuple(subcommands)], parser],
            description,
        )

    # Print help, including possible subcommands
    if args.help:
        _print_help()
        return

    errors = []
    # Parse arguments
    if parse_all:
        try:
            args = parser.parse_args(
                args=sys.argv[1 + len(subcommands) :], namespace=args
            )
        except ParserHelpException:
            args.help = True
        if args.help:
            _print_help()
            return
        rem = []
    else:
        # Parse only known args if scenario will add more arguments
        try:
            args, rem = parser.parse_known_args(
                sys.argv[1 + len(subcommands) :], namespace=args
            )
        except ParserHelpException as ex:
            errors.append(ex.error)

    # Run subcommands
    used_functions = set()
    for subcommand in subcommands:
        run = MAP_COMMAND_TO_SCENARIO[subcommand].run
        if run in used_functions:
            continue
        try:
            try:
                result = run(args, not_parsed=rem)
                if result:
                    parser.error(
                        f"`{subcommand}` subcommand did not end successfully",
                        print_usage=False,
                    )
            except ModuleNotFoundError as er:
                extras = find_missing_optional_dependency(er.name)
                if not extras:
                    raise
                er = MissingKenningDependencies(
                    name=er.name, path=er.path, optional_dependencies=extras
                )
                raise

            except argparse.ArgumentError as er:
                parser.error(er.message)
                return 2
        except ParserHelpException as ex:
            # Prepare combined errors
            if ex.error is not None:
                errors.append(ex.error)
            if len(errors) > 1:
                ex.error = "\n"
                for error in errors:
                    ex.error += f"- {error}\n"
            elif errors:
                ex.error = errors[0]
            # Print help with errors
            ex.print(
                parser,
                [parsers[tuple(subcommands)], parser],
                description,
            )
            return

        used_functions.add(run)
