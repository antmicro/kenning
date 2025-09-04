# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base ROS2 action parser template.
"""


from typing import Any


class ROS2ActionParser:
    """
    A template class for all ROS2 Action parser.
    """

    # types that are associated with ROS2 Action i.e ,
    # Action result can be converted to.
    associated_type = None

    # types of ROS 2 action that parser is associated with
    associated_action_type = None

    @staticmethod
    def from_any(x: Any) -> Any:
        """
        A function that takes value x and
        convert it to appropriate ROS2 Action
        type defined in kenning messages.

        Parameters
        ----------
        x : Any
            Data to parse.

        Returns
        -------
        Any
            Parsed ROS2 Action type.
        """
        ...

    @staticmethod
    def to_associated_type(x: Any) -> Any:
        """
        A function that takes result of
        ROS 2 action and converts them
        into associated value.

        Parameters
        ----------
        x : Any
            Result to parse.

        Returns
        -------
        Any
            Associated type instance
        """
        ...
