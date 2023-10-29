# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script for managing Kenning's cache. It can list cached files, clear cache
and print cache settings.
"""
import argparse
import sys
from typing import List, Optional, Tuple

from kenning.cli.command_template import (
    CACHE,
    GROUP_SCHEMA,
    ArgumentsGroups,
    CommandTemplate,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceManager


def format_size(size: int, unit: str = "B"):
    """
    Return string with proper unit.

    Parameters
    ----------
    size : int
        Value to be formatted.
    unit : str
        Value units.

    Returns
    -------
    str
        String with properly formatted units.
    """
    for prefix in ("", "K", "M", "G", "T"):
        if abs(size) < 1024.0:
            return f"{size:3.1f}{prefix}{unit}"
        size /= 1000.0

    return f"{size:.1f}P{unit}"


class ManageCacheRunner(CommandTemplate):
    """
    Command template for managing Kenning's cache.
    """

    parse_all = False
    description = __doc__.split("\n\n")[0]

    action_arguments = [
        "list_files",
        "clear",
        "settings",
    ]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            ManageCacheRunner, ManageCacheRunner
        ).configure_parser(parser, command, types, groups)

        list_group = parser.add_argument_group(GROUP_SCHEMA.format(CACHE))

        list_group.add_argument(
            "action",
            help="Action to be performed",
            choices=ManageCacheRunner.action_arguments,
        )
        list_group.add_argument(
            "-v", help="Display full paths", action="store_true"
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        KLogger.set_verbosity(args.verbosity)

        resource_manager = ResourceManager()

        if "list_files" == args.action:
            files = resource_manager.list_cached_files()
            files.sort(key=lambda f: f.stat().st_size, reverse=True)
            total_size = 0
            print("Cached files:")
            for file in files:
                size = file.stat().st_size
                total_size += size
                if args.v:
                    path = str(file.resolve())
                else:
                    path = file.name
                print(f"\t{format_size(size):>8} {path}")
            print(
                f"\nTotal size: {format_size(total_size)} / "
                f"{format_size(resource_manager.max_cache_size)}"
            )

        elif "clear" == args.action:
            files = resource_manager.list_cached_files()
            total_size = 0
            for file in files:
                total_size += file.stat().st_size

            resource_manager.clear_cache()
            print(f"Cleared {format_size(total_size)}")

        elif "settings" == args.action:
            settings_str = "Cache settings:\n"
            available_settings = (
                (
                    "cache directory",
                    ResourceManager.CACHE_DIR_ENV_VAR,
                    str(resource_manager.cache_dir),
                ),
                (
                    "max size",
                    ResourceManager.MAX_CACHE_SIZE_ENV_VAR,
                    format_size(resource_manager.max_cache_size),
                ),
            )
            alignment = max(
                len(f"\t{name} ({env_var})")
                for name, env_var, _ in available_settings
            )
            for name, env_var, value in available_settings:
                settings_str += (
                    f"\t{name} ({env_var})".ljust(alignment) + f" :\t{value}\n"
                )
            settings_str += (
                "\nSet "
                + ", ".join(env_var for _, env_var, _ in available_settings)
                + " environment variables to change defaults."
            )
            print(settings_str)

        else:
            print(
                f'Invalid action: {args.action}. Available actions: '
                f'{", ".join(ManageCacheRunner.action_arguments)}.'
            )


def main(argv):  # noqa: D103
    parser, _ = ManageCacheRunner.configure_parser(command=argv[0])
    args, _ = parser.parse_known_args(argv[1:])

    ManageCacheRunner.run(args)


if __name__ == "__main__":
    main(sys.argv)
