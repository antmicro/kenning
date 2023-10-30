# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with custom autocompletion class and configuration.
"""

import argparse
from typing import List

from argcomplete.finders import CompletionFinder

from kenning.cli.config import AVAILABLE_COMMANDS, setup_base_parser
from kenning.utils.class_loader import load_class

# Subcommands without help
ALL_SUBCOMMANDS = AVAILABLE_COMMANDS[:-2]
# Names of flags which takes class paths
CLASS_FLAG_NAMES = (
    "modelwrapper_cls",
    "protocol_cls",
    "runtime_cls",
    "dataset_cls",
    "compiler_cls",
)


class CustomCompletion(CompletionFinder):
    """
    Introduces Kenning-specific completion patterns.

    Extends default argcomplete class with:

    * Generating completions for dynamically specified class
    * Mutually exclusive groups -- '--json-cfg' and '--*-cls'
    * Completing subcommands only before other flags
    * Preventing flags duplication
    """

    def _get_completions(
        self,
        comp_words: List[str],
        cword_prefix: str,
        cword_prequote,
        last_wordbreak_pos,
    ) -> List[str]:
        # Create new parser to find used classes
        parser = argparse.ArgumentParser(add_help=False)
        for flag in CLASS_FLAG_NAMES:
            parser.add_argument(
                f'--{flag.replace("_", "-")}', nargs="?", const=None
            )
        args, _ = parser.parse_known_args(comp_words)

        # Create parsers for used classes
        parsers = []
        for name in CLASS_FLAG_NAMES:
            if getattr(args, name, None):
                _class = None
                try:
                    _class = load_class(getattr(args, name))
                except Exception:
                    pass
                if _class:
                    parsers.append(_class.form_argparse()[0])

        if parsers:
            # Choose last subparser
            subcommands = [arg for arg in comp_words if arg in ALL_SUBCOMMANDS]
            subparser = self._parser
            for subcommand in subcommands:
                subactions = [
                    action
                    for action in subparser._actions
                    if isinstance(action, argparse._SubParsersAction)
                ]
                if subactions and subcommand in subactions[0].choices:
                    subparser = subactions[0].choices[subcommand]
                else:
                    subparser = None
                    break
            # Extend parser with arguments for classes
            if subparser:
                subactions[0].choices[
                    subcommands[-1]
                ] = argparse.ArgumentParser(
                    subparser.prog, parents=[subparser] + parsers
                )

        completions = super()._get_completions(
            comp_words, cword_prefix, cword_prequote, last_wordbreak_pos
        )

        # JSON and flag config are mutually exclusive
        if "--json-cfg" in comp_words:
            completions = [
                arg
                for arg in completions
                if not (arg.startswith("--") and arg.endswith("-cls"))
            ]
        elif any(
            arg.startswith("--") and arg.endswith("-cls") for arg in comp_words
        ):
            completions.remove(
                "--json-cfg"
            ) if "--json-cfg" in completions else None

        # Do not complete subcommands after flags
        # Do not duplicate already used flags
        used_flags = set(word for word in comp_words if word.startswith("-"))
        subcommands = {arg for arg in completions if arg in ALL_SUBCOMMANDS}
        completions = [
            *(subcommands if not used_flags else []),
            *(set(completions) - subcommands - used_flags),
        ]

        return completions


def configure_autocomplete():
    """
    Function creating new parser with arguments and configuring autocompletion.
    """
    parser, _ = setup_base_parser(with_arguments=True)
    CustomCompletion()(parser)
