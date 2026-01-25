# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
File with ROS2 related helpers.
"""

from threading import Thread
from typing import Any, List, Optional

import rclpy
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python

from kenning.utils.logger import KLogger
from kenning.utils.singleton import Singleton

KENNING_ROS_NODE_NAME = "kenning"

KENNING_ROS_LOGGER_BACKEND_NAME = "ros2_backend"


class _ROS2GlobalContext(metaclass=Singleton):
    """
    A class that holds static ROS 2 Node used by other ROS 2 related modules.
    """

    def __init__(self):
        self.node = None
        self.node_thread = None
        self.logger = None

    def init_node(self) -> List[str]:
        """
        Function that initialize global ROS 2 Node.
        """
        if not rclpy.ok():
            KLogger.debug("Initializing ROS 2 global context")

            rclpy.init()

        KLogger.debug("Creating global ROS 2 Node...")

        self.node = Node(
            KENNING_ROS_NODE_NAME,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.logger = self.node.get_logger()

        KLogger.debug("Created ROS 2 Node")

        # add ROS2 logger backend

        KLogger.add_custom_backend(
            KENNING_ROS_LOGGER_BACKEND_NAME, self.logger_backend
        )

    def start_node(self):
        if self.node_thread is not None:
            KLogger.debug("ROS 2 Node is already started")
            return

        KLogger.debug("Initializing ROS 2 Node thread.")
        # create node loop that will spin kenning node in the background
        self.node_thread = Thread(target=self.node_loop)
        self.node_thread.daemon = True
        self.node_thread.start()
        KLogger.info("Started ROS 2 Node")

    def clean_args_list(self, args: Optional[List[str]] = None):
        return rclpy.utilities.remove_ros_args(args)

    def logger_backend(self, msg: str):
        if self.logger is None:
            return

        self.logger.info(msg)

    def get_param(self, name: str) -> Any:
        """
        Try to get parameter from ROS 2.

        Parameters
        ----------
        name : str
            A name of parameter we try to get.

        Returns
        -------
        Any
            An parameter value.
        """
        param = parameter_value_to_python(
            self.node.get_parameter_or(name, None).get_parameter_value()
        )

        KLogger.debug(f"Parameter {param} of: {name}")

        if param is None:
            return None

        return param

    def get_config_file_path(self) -> str:
        """
        Get path to Kenning pipeline config file.

        Returns
        -------
        str
            A string with path to config file.
        """
        cfg = self.node.get_parameter_or(
            "config_file", Parameter("config_file", value="")
        ).get_parameter_value()

        return cfg.string_value

    def get_logging_level(self) -> LoggingSeverity:
        """
        Function to get logging level from ROS 2 Node.

        Returns
        -------
        LoggingSeverity
            A current logging level.
        """
        return self.node.get_logger().get_effective_level()

    def init_parameters(self):
        """
        Function that initialize ROS 2 parameters from.
        """
        KLogger.debug(f"Parameters: {self.node.get_parameters()}")

    def node_loop(self):
        """
        Function that spins node.
        """
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.001)

    def stop_node(self):
        """
        Function that shutdown global context and stop ROS 2 node
        execution.
        """
        if rclpy.ok():
            KLogger.info("Shutting down ROS 2 Node")

            rclpy.shutdown()
            self.node.destroy_node()

            self.node_thread.join()

            self.node = None
            self.node_thread = None

            KLogger.remove_custom_backend(KENNING_ROS_LOGGER_BACKEND_NAME)

    def node_name(self) -> str:
        """
        Function that gets used Node name.

        Returns
        -------
        str
            A name of the created Node.
        """
        return self.node.get_name()

    def __del__(self):
        self.stop_node()


ROS2GlobalContext = _ROS2GlobalContext()
