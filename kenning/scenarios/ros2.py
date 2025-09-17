# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that initialize ROS 2 environment and ROS 2 Node
used by all ROS 2 related components.
"""

import argparse
from typing import Any, List, Optional, Tuple

from kenning.cli.command_template import (
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)


class ROS2Initializer(CommandTemplate):
    """
    Command-line template used for initializing ROS 2 for Kenning.
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
            ROS2Initializer, ROS2Initializer
        ).configure_parser(parser, command, types, groups, True)

        return parser, groups

    @staticmethod
    def run(
        args: argparse.Namespace, not_parsed: List[str] = [], **kwargs: Any
    ):
        # Start ROS 2 node

        from kenning.utils.ros2_global_context import ROS2GlobalContext

        ROS2GlobalContext.init_node()

        args.json_cfg = ROS2GlobalContext.get_config_file_path()

        ROS2GlobalContext.start_node()

        # clean already parsed ROS 2 arguments
        not_parsed[:] = ROS2GlobalContext.clean_args_list(not_parsed)
