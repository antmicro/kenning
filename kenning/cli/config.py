# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with Kenning CLI configuration.

It contains specification which command can be used in sequence
and mapping to classes extending CommandTemplate.
"""

import argparse
from typing import List, Generator, Dict, Tuple, Type, Optional

from kenning.cli.formatter import Formatter
from kenning.cli.command_template import (
    CommandTemplate,
    COMPLETION,
    FINE_TUNE,
    FLOW,
    SEARCH,
    HELP,
    INFO,
    LIST,
    OPTIMIZE,
    REPORT,
    SERVER,
    TEST,
    TRAIN,
    CACHE,
    VISUAL_EDITOR,
)
from kenning.scenarios import (
    class_info,
    inference_server,
    inference_tester,
    json_flow_runner,
    list_classes,
    manage_cache,
    optimization_runner,
    pipeline_manager_client,
    render_report,
    model_training,
    fuzzy_search_class,
    configure_autocompletion,
)


# Subcommands that can be used in sequence list in structure
# defining possible order
SEQUENCED_COMMANDS = ([[TRAIN, OPTIMIZE], TEST, REPORT],)
# Subcommands that can be used one at the time
BASIC_COMMANDS = (
    FLOW,
    SERVER,
    VISUAL_EDITOR,
    FINE_TUNE,
    LIST,
    CACHE,
    INFO,
    SEARCH,
    COMPLETION,
)
# All available subcommands and help flags
AVAILABLE_COMMANDS = (
    OPTIMIZE,
    TRAIN,
    TEST,
    REPORT,
    *BASIC_COMMANDS,
    *HELP["flags"],
)
# Connection between subcommand and its logic (extending CommandTemplate)
MAP_COMMAND_TO_SCENARIO: Dict[str, Type[CommandTemplate]] = {
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
}
# Name of the subcommand group -- displayed in help message
SUBCOMMANDS = "Subcommands"
# Destination of subcommands
SUB_DEST_FORM = "__seq_{}"


def get_all_sequences(
    sequence: List[List[str]],
    prefix: Optional[List[str]] = None,
) -> Generator[Tuple[str], None, None]:
    """
    Yields possible sequences of commands.

    Parameters
    ----------
    sequence : List[List[str]]
        Sequence with commands in right order
    prefix : Optional[List[str]]
        Prefix appended to the results

    Yields
    ------
    List[str] :
        Sequence of commands
    """
    if prefix is None:
        prefix = []
    if not sequence:
        for i in range(len(prefix)):
            yield tuple(prefix[i:])
        return
    for item in (
        sequence[0] if isinstance(sequence[0], List) else [sequence[0]]
    ):
        yield from get_all_sequences(sequence[1:], prefix + [item])


def create_subcommands(
    subparser: argparse._SubParsersAction,
    names: List[str],
    number: int = 0,
    with_arguments: bool = False,
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
    with_arguments : bool
        Create parsers with configured arguments

    Returns
    -------
    Dict[Tuple[str], argparse.ArgumentParser] :
        Dictionary of parsers associated with sequence of subcommands
    """
    groups = {}
    parsers = {}
    parser = None
    for i, name in enumerate(names):
        if parser:
            subparser = parser.add_subparsers(
                title=SUBCOMMANDS, dest=SUB_DEST_FORM.format(number + i)
            )
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
    argparse.ArgumentParser :
        Created parser
    Dict[Tuple[str], argparse.ArgumentParser] :
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
    subparsers = parser.add_subparsers(
        title=SUBCOMMANDS, dest=SUB_DEST_FORM.format(0)
    )

    flag_group = parser.add_argument_group("Flags")
    flag_group.add_argument(
        *HELP["flags"],
        action="store_true",
        help=HELP["msg"],
    )

    sequences = set()
    for sequence in SEQUENCED_COMMANDS:
        sequences.update(get_all_sequences(sequence))
    for sequence in sorted(sequences, key=lambda x: x[0]):
        parsers.update(
            create_subcommands(
                subparsers, sequence, with_arguments=with_arguments
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
