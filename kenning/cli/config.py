# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with Kenning CLI configuration.

It contains specification which command can be used in sequence
and mapping to classes extending CommandTemplate.
"""

import argparse
from typing import Dict, Generator, List, Tuple, Type, Union

from kenning.cli.command_template import (
    AUTOML,
    AVAILABLE_PLATFORMS,
    CACHE,
    COMPLETION,
    FINE_TUNE,
    FLOW,
    GENERATE_PLATFORMS,
    HELP,
    INFO,
    LIST,
    OPTIMIZE,
    REPORT,
    SEARCH,
    SERVER,
    TEST,
    TRAIN,
    VISUAL_EDITOR,
    CommandTemplate,
)
from kenning.cli.formatter import Formatter
from kenning.core.exceptions import ConfigurationError
from kenning.scenarios import (
    automl,
    available_platforms,
    class_info,
    configure_autocompletion,
    fuzzy_search_class,
    generate_platforms,
    inference_server,
    inference_tester,
    json_flow_runner,
    list_classes,
    manage_cache,
    model_training,
    optimization_runner,
    pipeline_manager_client,
    render_report,
)


def _sequence(*args):
    """
    Representation of subcommands that can be used in a sequence.
    """
    return [*args]


def _either(*args):
    """
    Representation of subcommands that can be used exclusively - this or that.
    """
    return tuple([*args])


def _optional(arg):
    """
    Subcommand arguments that can be optionally skipped.
    """
    return _either(arg, None)


# Combination of nested Tuples and Lists which together create the logical
# representation of sequenced commands structure.
# Each possible path that can be taken should be a valid and correct set of
# subcomands.
SEQUENCED_COMMANDS = _either(
    _sequence(
        _either(
            _optional(TRAIN), _sequence(_optional(AUTOML), _optional(OPTIMIZE))
        ),
        _either(_sequence(TEST, _optional(REPORT)), _optional(TEST)),
    ),
    _sequence(_optional(AUTOML), REPORT),
)

# Subcommands that can be used one at the time
BASIC_COMMANDS = (
    AVAILABLE_PLATFORMS,
    FLOW,
    SERVER,
    VISUAL_EDITOR,
    FINE_TUNE,
    LIST,
    CACHE,
    INFO,
    SEARCH,
    COMPLETION,
    GENERATE_PLATFORMS,
)
# All available subcommands and help flags
AVAILABLE_COMMANDS = (
    AUTOML,
    OPTIMIZE,
    TRAIN,
    TEST,
    REPORT,
    *BASIC_COMMANDS,
    *HELP["flags"],
)
# Connection between subcommand and its logic (extending CommandTemplate)
MAP_COMMAND_TO_SCENARIO: Dict[str, Type[CommandTemplate]] = {
    AUTOML: automl.AutoMLCommand,
    FINE_TUNE: optimization_runner.OptimizationRunner,
    FLOW: json_flow_runner.FlowRunner,
    SEARCH: fuzzy_search_class.FuzzySearchClass,
    INFO: class_info.ClassInfoRunner,
    LIST: list_classes.ListClassesRunner,
    OPTIMIZE: inference_tester.InferenceTester,
    REPORT: render_report.RenderReport,
    SERVER: inference_server.InferenceServerRunner,
    TEST: inference_tester.InferenceTester,
    TRAIN: model_training.TrainModel,
    CACHE: manage_cache.ManageCacheRunner,
    VISUAL_EDITOR: pipeline_manager_client.PipelineManagerClient,
    COMPLETION: configure_autocompletion.ConfigureCompletion,
    GENERATE_PLATFORMS: generate_platforms.GeneratePlatformsCommand,
    AVAILABLE_PLATFORMS: available_platforms.AvailablePlatformsCommand,
}
# Name of the subcommand group -- displayed in help message
SUBCOMMANDS = "Subcommands"
# Destination of subcommands
SUB_DEST_FORM = "__seq_{}"
# Destination of used subcommands list
USED_SUBCOMMANDS = "__seq"


def get_used_subcommands(
    args: argparse.Namespace,
) -> List[str]:
    """
    Returns used subcommands from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    List[str]
        List with used subcommands.
    """
    return getattr(args, USED_SUBCOMMANDS, [])


def get_all_sequences(
    sequence: Union[List, Tuple]
) -> Generator[Tuple[str], None, None]:
    """
    Yields possible sequences of commands.

    Parameters
    ----------
    sequence : Union[List, Tuple]
        Logical representation of sequenced commands.
        A list of lists and tuples where each leaf is a string.
        Every List is treated as an logical AND between sequences of commands.
        Every tuple is treated as an OR between sequences of commands.
        Every path through this graph should be a valid
        cli subcommand configuration.

    Examples
    --------
        [automl, train] -> returns [('automl', 'train')]
        [optimize, (test, None)] -> returns [('optimize', 'test'), ('optimize')]
        [(automl, train, None), optimize] -> returns [('automl', 'optimize), ('train', 'optimize'), ('optimize')]

    Yields
    ------
    Tuple[str]
        Possible sequence of commands
    """  # noqa: E501
    if sequence is None:
        yield tuple()
    elif isinstance(sequence, list) and len(sequence) == 0:
        yield tuple()
    # yield tuple with this string
    elif isinstance(sequence, str):
        yield tuple([sequence])
    # yield processed from first element and combine with the rest
    elif isinstance(sequence, list):
        prefix = get_all_sequences(sequence[0])
        for pre in prefix:
            for sub in get_all_sequences(sequence[1:]):
                yield tuple([*pre, *sub])
    # yield processed from each entry
    elif isinstance(sequence, tuple):
        for sub in sequence:
            yield from get_all_sequences(sub)


def create_subcommands(
    names: Tuple[str],
    subcommand_parsers: Dict[Tuple[str], argparse.ArgumentParser],
    subcommand_groups: Dict[Tuple[str], argparse._SubParsersAction],
    number: int = 0,
    with_arguments: bool = False,
) -> Dict[Tuple[str], argparse.ArgumentParser]:
    """
    Creates nested subcommands from list of names.

    Parameters
    ----------
    names : Tuple[str]
        Sequence of subcommands
    subcommand_parsers : Dict[Tuple[str], argparse.ArgumentParser]
        Parsers created so far
    subcommand_groups : Dict[Tuple[str], argparse._SubParsersAction]
        Maps parsers created so far to objects which can create parsers.
    number : int
        Depth of subparser
    with_arguments : bool
        Create parsers with configured arguments

    Returns
    -------
    Dict[Tuple[str], argparse.ArgumentParser]
        Dictionary of parsers associated with sequence of subcommands

    Raises
    ------
    ConfigurationError
        if no root subcommand_group was set in subcommand_groups
    """
    groups = {}
    parsers = {}
    parser = None
    for i, name in enumerate(names):
        if names[: i + 1] in subcommand_parsers:
            parser = subcommand_parsers[names[: i + 1]]
            continue

        if names[:i] in subcommand_groups:
            subparser = subcommand_groups[names[:i]]
        elif parser:
            subparser = parser.add_subparsers(
                title=SUBCOMMANDS, dest=SUB_DEST_FORM.format(number + i)
            )
            subcommand_groups.update({names[:i]: subparser})
        else:
            raise ConfigurationError("no root subcommand_group was set")

        desc = MAP_COMMAND_TO_SCENARIO[name].description
        if not isinstance(desc, str):
            desc = desc[name]

        parser = subparser.add_parser(
            name,
            help=desc.split(".", 1)[0].strip("\n"),
            add_help=False,
            conflict_handler="resolve" if with_arguments else "error",
        )
        if with_arguments:
            for n in names[: i + 1]:
                parser, groups = MAP_COMMAND_TO_SCENARIO[n].configure_parser(
                    types=[n], parser=parser
                )

        parsers[tuple(names[: i + 1])] = parser
    return parsers


def setup_base_parser(
    with_arguments: bool = False
) -> Tuple[argparse.ArgumentParser, Dict[Tuple[str], argparse.ArgumentParser]]:
    """
    Sets up parser containing only subcommands and help message.

    Parameters
    ----------
    with_arguments : bool
        Create parsers with configured arguments

    Returns
    -------
    parser: argparse.ArgumentParser
        Created parser
    subcommands: Dict[Tuple[str], argparse.ArgumentParser]
        Dictionary of parsers associated with sequence of subcommands
    """
    parser = argparse.ArgumentParser(
        prog="kenning",
        description="Command-line interface for Kenning",
        conflict_handler="resolve",
        formatter_class=Formatter,
        add_help=False,
    )
    parsers = {}

    flag_group = parser.add_argument_group("Flags")
    flag_group.add_argument(
        *HELP["flags"],
        action="store_true",
        help=HELP["msg"],
    )

    # define root subcommand group
    subparsers = parser.add_subparsers(
        title=SUBCOMMANDS, dest=SUB_DEST_FORM.format(0)
    )
    subcommand_groups = {tuple(): subparsers}

    sequences = set(get_all_sequences(SEQUENCED_COMMANDS))
    for sequence in sorted(sequences):
        parsers.update(
            create_subcommands(
                sequence,
                parsers,
                subcommand_groups,
                with_arguments=with_arguments,
            )
        )

    for subcommand in BASIC_COMMANDS:
        parsers[(subcommand,)] = subparsers.add_parser(
            subcommand,
            help=MAP_COMMAND_TO_SCENARIO[subcommand]
            .description.split(".")[0]
            .strip("\n")
            .replace("\n", " "),
            add_help=False,
        )
        if with_arguments:
            MAP_COMMAND_TO_SCENARIO[subcommand].configure_parser(
                parsers[(subcommand,)]
            )
    return parser, parsers
