# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for ROS2 inference.
"""


from kenning.core.platform import Platform


class ROS2Platform(Platform):
    """
    ROS 2 platform, for now it is just replacement
    for LocalPlatform that only set needs_protocol to True.
    """

    needs_protocol = True

    def get_default_protocol(self):
        from kenning.protocols.ros2 import ROS2Protocol

        return ROS2Protocol(
            node_name="kenning_node",
            process_action_type_str="kenning_computer_vision_msgs.action.SegmentationAction",
            process_action_name="cvnode_process",
        )

    def get_time(self) -> float:
        from kenning.utils.ros2_global_context import ROS2GlobalContext

        return (
            float(ROS2GlobalContext.node.get_clock().now().nanoseconds) / 10e9
        )
