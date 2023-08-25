# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script helping to configure autocompletion for Kenning CLI.
"""

import sys
import argparse
from typing import Optional, List, Tuple
import subprocess
from pathlib import Path

from kenning.cli.command_template import (
    ArgumentsGroups, CommandTemplate, GROUP_SCHEMA, COMPLETION
)

BASH = "bash"
ZSH = "zsh"
FISH = "fish"
BASH_ZSH_DONE = False


def configure_bash_zsh():
    """
    Configures autocompletion for Bash and Zsh, using default
    `argcomplete` script `activate-global-python-argcomplete`.
    """
    global BASH_ZSH_DONE
    if BASH_ZSH_DONE:
        return

    subprocess.call(["activate-global-python-argcomplete"])
    BASH_ZSH_DONE = True


def configure_fish():
    """
    Configures autocompletion for Fish, using `argcomplete`
    script `register-python-argcomplete`.
    """
    kenning_completion_path = \
        Path.home() / Path(".config/fish/completions/kenning.fish")
    print(f"Fish completion script will be saved at {kenning_completion_path}")
    proceed = input("OK to proceed? [y/n] ")
    if not proceed[0].lower() == 'y':
        print("Fish configuration cancelled")
        return
    with open(kenning_completion_path, "w+") as fd:
        subprocess.call(
            "register-python-argcomplete --shell fish kenning".split(),
            stdout=fd,
        )
    print("Please restart your shell "
          "or source the installed file to activate it.")


CONFIGURE_SHELL = {
    BASH: configure_bash_zsh,
    ZSH: configure_bash_zsh,
    FISH: configure_fish,
}


class ConfigureCompletion(CommandTemplate):
    parse_all = True
    description = __doc__.strip('\n')

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
        resolve_conflict: bool = False,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            ConfigureCompletion, ConfigureCompletion
        ).configure_parser(
            parser, command, types, groups
        )

        group = parser.add_argument_group(GROUP_SCHEMA.format(COMPLETION))
        group.add_argument(
            "shell",
            help="Which shell should be configured",
            type=str,
            choices=(BASH, ZSH, FISH),
            nargs="+",
        )

        return parser, groups

    @staticmethod
    def run(
        args: argparse.Namespace,
        **kwargs
    ):
        for shell in args.shell:
            CONFIGURE_SHELL[shell]()


if __name__ == "__main__":
    sys.exit(ConfigureCompletion.scenario_run())
