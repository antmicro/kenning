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
    def get_param(cls, name: str) -> Any:
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
            cls.node.get_parameter_or(name, None).get_parameter_value()
        )

        KLogger.debug(f"Parameter {param} of: {name}")

        if param is None:
            return None

        return param

    @classmethod
    def get_config_file_path(cls) -> str:
        """
        Get path to Kenning pipeline config file.

        Returns
        -------
        str
            A string with path to config file.
        """
        cfg = cls.node.get_parameter_or(
            "config_file", Parameter("config_file", value="")
        ).get_parameter_value()

        return cfg.string_value

    @classmethod
    def get_logging_level(cls) -> LoggingSeverity:
        """
        Function to get logging level from ROS 2 Node.

        Returns
        -------
        LoggingSeverity
            A current logging level.
        """
        return cls.node.get_logger().get_effective_level()

    @classmethod
    def get_pipeline(cls) -> List[str]:
        """
        Function that returns command sequence that will
        be used by Kenning.

        Returns
        -------
        List[str]
            A sequence of commands that will be executed by kenning.

        Raises
        ------
        Exception
            Raised when unknown command is passed to pipeline.
        """
        pipeline = cls.node.get_parameter_or(
            "pipeline", Parameter("pipeline", value=[])
        ).get_parameter_value()

        pipeline = pipeline.string_array_value

        for cmd in pipeline:
            if cmd not in AVAILABLE_COMMANDS:
                raise Exception(f"Unknown pipeline command: {cmd}")

        return pipeline

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
