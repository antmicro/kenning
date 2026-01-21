# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that downloads remote resources associated with the scenario.
"""

import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml
from argcomplete import FilesCompleter

from kenning.cli.command_template import (
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceManager, ResourceURI

FILE_CONFIG = "Inference configuration with JSON/YAML file"
FLAG_CONFIG = "Inference configuration with flags"
ARGS_GROUPS = {
    FILE_CONFIG: f"Configuration with pipeline defined in JSON/YAML file. \
    This section is not compatible with '{FLAG_CONFIG}'. Arguments with '*' \
        are required.",
    FLAG_CONFIG: f"Configuration with flags. This section is \
          not compatible with '{FILE_CONFIG}'. Arguments with '*' \
            are required.",
}


class DownloadResources(CommandTemplate):
    """
    Command-line template used to pre fetch
    kenning resources required by configs.
    """

    parse_all = False
    description = __doc__.split("\n\n")[0]
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            DownloadResources, DownloadResources
        ).configure_parser(parser, command, types, groups, True)

        required_prefix = "* "

        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)

        groups[FILE_CONFIG].add_argument(
            "--json-cfg",
            "--cfg",
            help=f"{required_prefix}The set of config file paths that will be \
                  search for any resources",
            type=ResourceURI,
            nargs="+",
        ).completer = FilesCompleter(
            allowednames=("*.json", "*.yaml", "*.yml")
        )

        return parser, groups

    @staticmethod
    def check_for_uri(path: str) -> bool:
        formatted: str = path.strip()

        return re.search(DownloadResources.VALID_FILE_REGEX, formatted)

    @staticmethod
    def check_for_uri_string(obj) -> bool:
        return isinstance(obj, str) and DownloadResources.check_for_uri(obj)

    @staticmethod
    def traverse_dictionary(cfg: Dict) -> List[str]:
        """
        Function that search config for any resources to download.

        Parameters
        ----------
        cfg : Dict
            A config that will be checked

        Returns
        -------
        List[str]
            A list of found resources to download.
        """
        found_uris: list[str] = []

        if isinstance(cfg, List):
            dict_to_check: list[dict] = cfg
        else:
            dict_to_check: list[dict] = [cfg]

        resource_manager = ResourceManager()

        for obj in dict_to_check:
            for key in obj.keys():
                if isinstance(obj[key], dict):
                    dict_to_check.append(obj[key])
                elif isinstance(obj[key], list):
                    for k in obj[key]:
                        if resource_manager.check_for_uri(k):
                            found_uris.append(k)
                        elif isinstance(k, dict):
                            dict_to_check.append(k)
                elif resource_manager.check_for_uri(obj[key]):
                    found_uris.append(obj[key])

        return found_uris

    @staticmethod
    def run(
        args: argparse.Namespace, not_parsed: List[str] = [], **kwargs: Any
    ):
        # Download resources

        if args.json_cfg is None:
            KLogger.error("No configs files provided!")
            return

        configs_to_check = args.json_cfg

        uris_to_download: List[str] = []

        for config in configs_to_check:
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)

                uris_to_download.extend(
                    DownloadResources.traverse_dictionary(cfg)
                )

        if len(uris_to_download) == 0:
            KLogger.warning("No uris found in provided config files!")
            return

        resource_manager = ResourceManager()

        for uri in uris_to_download:
            KLogger.info(f"Found uri: {uri}")
            KLogger.info("Attempting downloads")

            resource_manager.get_resource(uri)
