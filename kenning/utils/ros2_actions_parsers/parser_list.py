# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides list of ROS2 Action parsers with associated ROS2 Kenning action types.
"""

from kenning_computer_vision_msgs.action import SegmentationAction

from kenning.utils.ros2_actions_parsers.segmentation_action import (
    SegmentationActionParser,
)

"""
Dictionary used for deducing right parser type for ROS2 Data converter.
"""
ROS2_ACTION_TO_PARSER = {SegmentationAction: SegmentationActionParser}
