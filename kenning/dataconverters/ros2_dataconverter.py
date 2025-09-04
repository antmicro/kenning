# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A DataConverter-derived class used to manipulate the data using the
SegmentationAction object for compatibility between surronding blocks
during runtime.
"""

from typing import Any, Dict

from kenning.core.dataconverter import DataConverter
from kenning.core.exceptions import KenningDataConverterError
from kenning.utils.class_loader import load_class
from kenning.utils.logger import KLogger


class ROS2DataConverter(DataConverter):
    """
    Converts input and output data for Instance Segmentation to ROS 2 topics.
    """

    arguments_structure: Dict[str, str] = {
        "ros2_message_type": {
            "description": "ROS2 message type handheld by data converter",
            "type": str,
            "required": True,
        }
    }

    def __init__(self, ros2_message_type: str):
        """
        Initializes the ModelWrapperDataConverter object.

        Parameters
        ----------
        ros2_message_type : str
            ROS2 message type handheld by data converter.

        Raises
        ------
        ROS2ActionTypeNotFound
            Exception raised when specific ROS2
            action type has been not found.
        """
        super().__init__()

        KLogger.debug(f"Trying to import type at {ros2_message_type}")
        # try to import described type and right parser for it:

        ros2_message_type = load_class(ros2_message_type)

        from kenning.utils.ros2_actions_parsers.parser_list import (
            ROS2_ACTION_TO_PARSER,
        )

        if ros2_message_type in ROS2_ACTION_TO_PARSER.keys():
            self.ros2_parser = ROS2_ACTION_TO_PARSER[ros2_message_type]
        else:
            raise KenningDataConverterError(
                f"Type: {ros2_message_type} not found"
            )

    def to_next_block(self, data: Any) -> Any:
        """
        Converts input frames to segmentation action goal.
        Assumes that input data has BGR8 encoding.

        Parameters
        ----------
        data : Any
            The input data to be converted.

        Returns
        -------
        Any
            The converted segmentation action goal.
        """
        return self.ros2_parser.from_any(data)

    def to_previous_block(self, data: Any) -> Any:
        """
        Converts segmentation action result to SegmObject list. Assumes that if
        more than one frame is present, the output is for sequence of frames.

        Parameters
        ----------
        data : Any
            Result of the segmentation action.

        Returns
        -------
        Any
            The converted data.

        Raises
        ------
        KenningDataConverterError
            When invalid input data is provided.
        """
        # Check for success, we can make a so called standard that
        # every action definition used by CVNodes should have success parameter

        if not hasattr(data, "success"):
            raise KenningDataConverterError("Invalid input data format")

        if not data.success:
            raise KenningDataConverterError("ROS2 action failed")

        # Convert ROS2 Action result to appropriate type
        result = self.ros2_parser.to_associated_type(data)

        return result
