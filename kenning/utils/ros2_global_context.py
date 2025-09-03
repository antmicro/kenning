# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
File with ROS2 related helpers.
"""

from threading import Thread

import rclpy
from rclpy.node import Node

from kenning.utils.logger import KLogger

KENNING_ROS_NODE_NAME = "kenning"


class ROS2GlobalContext:
    """
    A class that holds static ROS 2 Node used by other ROS 2 related modules.
    """

    node = None

    node_thread = None

    @classmethod
    def init_node(cls):
        """
        Function that initialize global ROS 2 Node.
        """
        if not rclpy.ok():
            KLogger.debug("Initializing ROS 2 global context")

            rclpy.init()

        KLogger.debug("Creating global ROS 2 Node...")

        cls.node = Node(
            KENNING_ROS_NODE_NAME, allow_undeclared_parameters=True
        )

        KLogger.debug("Created ROS 2 Node!")

        KLogger.debug("Initializing ROS 2 Node thread.")
        # create node loop that will spin kenning node in the background
        cls.node_thread = Thread(target=cls.node_loop)
        cls.node_thread.start()
        KLogger.info("Started ROS2 Node!")

    @classmethod
    def init_parameters(cls):
        """
        Function that initialize ROS 2 parameters from.
        """
        KLogger.debug(f"Parameters: {cls.node.get_parameters()}")

    @classmethod
    def node_loop(cls):
        """
        Function that spins node.
        """
        while rclpy.ok():
            rclpy.spin_once(cls.node, timeout_sec=0.001)

    @classmethod
    def stop_node(cls):
        """
        Function that shutdown global context and stop ROS 2 node
        execution.
        """
        KLogger.info("Shutting down ROS 2 Node")

        rclpy.shutdown()
        cls.node.destroy_node()

        cls.node_thread.join()

        cls.node = None
        cls.node_thread = None

    @classmethod
    def node_name(cls) -> str:
        """
        Function that gets used Node name.

        Returns
        -------
        str
            A name of the created Node.
        """
        return cls.node.get_name()
