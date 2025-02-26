# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with custom autocompletion class and configuration.
"""

import argparse
from pathlib import Path
from typing import List

import yaml
from argcomplete.finders import CompletionFinder

from kenning.cli.config import (
    AVAILABLE_COMMANDS,
    USED_SUBCOMMANDS,
    setup_base_parser,
)
from kenning.utils.class_loader import ConfigKey, load_class, load_class_by_key

# Subcommands without help
ALL_SUBCOMMANDS = AVAILABLE_COMMANDS[:-2]
# Names of flags which takes class paths
CLASS_FLAG_NAMES = (
    "modelwrapper_cls",
    "protocol_cls",
    "runtime_cls",
    "dataset_cls",
    "compiler_cls",
    "platform_cls",
)
CLASS_JSON_KEYS = (
    ConfigKey.model_wrapper,
    ConfigKey.protocol,
    ConfigKey.runtime,
    ConfigKey.dataset,
    ConfigKey.platform,
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
        parser.add_argument("--json-cfg", "--cfg")
        args, _ = parser.parse_known_args(comp_words)
        subcommands = [arg for arg in comp_words if arg in ALL_SUBCOMMANDS]
        setattr(args, USED_SUBCOMMANDS, subcommands)

        # Create parsers for used classes
        parsers = []
        cfg = None
        cfg_path = getattr(args, "json_cfg", None)
        if cfg_path and Path(cfg_path).is_file():
            with open(cfg_path) as f:
                try:
                    cfg = yaml.safe_load(f)
                except yaml.YAMLError:
                    pass

        if cfg:
            # Load the config and inspect classes
            for key in CLASS_JSON_KEYS:
                _class = None
                try:
                    _class = load_class_by_key(cfg, key)
                except Exception:
                    pass
                if _class:
                    parsers.append(
                        _class.form_argparse(args, override_only=True)[0]
                    )

        if not parsers:
            for name in CLASS_FLAG_NAMES:
                if getattr(args, name, None):
                    _class = None
                    try:
                        _class = load_class(getattr(args, name))
                    except Exception:
                        pass
                    if _class:
                        parsers.append(_class.form_argparse(args)[0])

        if parsers:
            # Choose last subparser
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
                    subparser.prog,
                    parents=[subparser] + parsers,
                    add_help=False,
                )

        completions = super()._get_completions(
            comp_words, cword_prefix, cword_prequote, last_wordbreak_pos
        )

        # JSON and flag config are mutually exclusive
        if "--json-cfg" in comp_words or "--cfg" in comp_words:
            completions = [
                arg
                for arg in completions
                if not (arg.startswith("--") and arg.endswith("-cls"))
            ]
            if "--json-cfg" in completions:
                completions.remove("--json-cfg")
            if "--cfg" in completions:
                completions.remove("--cfg")
        elif any(
            arg.startswith("--") and arg.endswith("-cls") for arg in comp_words
        ):
            if "--json-cfg" in completions:
                completions.remove("--json-cfg")
            if "--cfg" in completions:
                completions.remove("--cfg")

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
