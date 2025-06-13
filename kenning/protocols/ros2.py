# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ROS2-based inference communication protocol.

Requires 'rclpy' and 'kenning_computer_vision_msgs' packages to be sourced
in the environment.
"""

import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Tuple, TypeVar

import rclpy
from rclpy.action import ActionClient
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node

from kenning.core.measurements import Measurements, timemeasurements
from kenning.core.protocol import Protocol
from kenning.utils.class_loader import load_class
from kenning.utils.logger import KLogger

GoalHandle = TypeVar("GoalHandle", bound=_rclpy.ActionGoalHandle)


class ROS2Protocol(Protocol):
    """
    A ROS2-based runtime protocol for communication.
    It supports only a client side of the runtime protocol.

    Protocol is implemented using ROS2 services and action.
    """

    arguments_structure = {
        "node_name": {
            "description": "Name of the ROS2 node",
            "type": str,
            "required": True,
        },
        "process_action_type_str": {
            "description": "Import path to the action class to use",
            "type": str,
            "required": True,
        },
        "process_action_name": {
            "description": "Name of the action to process via",
            "type": str,
            "required": True,
        },
        "model_service_type_str": {
            "description": "Import path to the service class for MODEL request to use",  # noqa: E501
            "type": str,
            "required": False,
            "default": "std_srvs.srv.Trigger",
        },
        "model_service_name": {
            "description": "Name of the service to upload the model and prepare node",  # noqa: E501
            "type": str,
            "required": False,
            "default": "cvnode_prepare",
        },
        "measurements_service_type_str": {
            "description": "Import path to the service class for MEASUREMENTS request to use",  # noqa: E501
            "type": str,
            "required": False,
            "default": "std_srvs.srv.Trigger",
        },
        "measurements_service_name": {
            "description": "Name of the service to get measurements",
            "type": str,
            "required": False,
            "default": "cvnode_measurements",
        },
    }

    def __init__(
        self,
        node_name: str,
        process_action_type_str: str,
        process_action_name: str,
        model_service_type_str: str = "std_srvs.srv.Trigger",
        model_service_name: str = "cvnode_prepare",
        measurements_service_type_str: str = "std_srvs.srv.Trigger",
        measurements_service_name: str = "cvnode_measurements",
    ):
        """
        Initializes ROS2Protocol object.

        Parameters
        ----------
        node_name : str
            Name of the ROS2 node.
        process_action_type_str : str
            Import path to the action class to use.
        process_action_name : str
            Name of the action to process via.
        model_service_type_str : str
            Import path to the service class for MODEL request to use.
        model_service_name : str
            Name of the service to upload the model and prepare node.
        measurements_service_type_str: str
            Import path to the service class for MEASUREMENTS request to use.
        measurements_service_name: str
            Name of the service to get measurements.
        """
        # ROS2 node
        self.node = None
        self.node_name = node_name

        # Action for data processing routine
        self.process_action = None
        self.process_action_name = process_action_name
        self.process_action_type_str = process_action_type_str
        self._process_action_type = load_class(process_action_type_str)

        # Node prepare service
        self.model_service = None
        self.model_service_name = model_service_name
        self.model_service_type_str = model_service_type_str
        self._model_service_type = load_class(model_service_type_str)

        # Measurements service
        self.measurements_service = None
        self.measurements_service_name = measurements_service_name
        self.measurements_service_type_str = measurements_service_type_str
        self._measurements_service_type = load_class(
            measurements_service_type_str
        )

        # Last message's future
        self.future = None

        super().__init__()

    def log_info(self, message: str):
        """
        Sends the info message to loggers.

        Parameters
        ----------
        message : str
            Message to be sent.
        """
        KLogger.info(message)
        if self.node is not None:
            self.node.get_logger().info(message)

    def log_debug(self, message: str):
        """
        Sends the debug message to loggers.

        Parameters
        ----------
        message : str
            Message to be sent.
        """
        KLogger.debug(message)
        if self.node is not None:
            self.node.get_logger().debug(message)

    def log_warning(self, message: str):
        """
        Sends the warning message to loggers.

        Parameters
        ----------
        message : str
            Message to be sent.
        """
        KLogger.warning(message)
        if self.node is not None:
            self.node.get_logger().warning(message)

    def log_error(self, message: str):
        """
        Sends the error message to loggers.

        Parameters
        ----------
        message : str
            Message to be sent.
        """
        KLogger.error(message)
        if self.node is not None:
            self.node.get_logger().error(message)

    def initialize_client(self):
        self.log_debug(f"Initializing action client node {self.node_name}")

        if not rclpy.ok():
            rclpy.init()
        self.node = Node(self.node_name)
        self.process_action = ActionClient(
            self.node,
            self._process_action_type,
            self.process_action_name,
            callback_group=None,
        )

        self.model_service = self.node.create_client(
            self._model_service_type,
            self.model_service_name,
        )

        self.measurements_service = self.node.create_client(
            self._measurements_service_type,
            self.measurements_service_name,
        )

        self.log_debug("Successfully initialized client")
        return True

    def upload_io_specification(self, path: Path) -> bool:
        self.log_warning(
            "IO specification is not supported in ROS2 protocol. " "Skipping."
        )
        return True

    def upload_model(self, path: Path) -> bool:
        self.log_debug("Uploading model")
        if self.model_service is None:
            self.log_error("Model service is not initialized")
            return False
        elif not self.model_service.wait_for_service(timeout_sec=1.0):
            self.log_error("Model service not available")
            return False

        request = self._model_service_type.Request()
        future = self.model_service.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        result = future.result()
        if result is None:
            self.log_error("Model service call failed")
            return False
        self.log_debug("Model uploaded successfully")
        return True

    def upload_input(self, data: GoalHandle) -> bool:
        """
        Initializes processing goal and sends it to the action server.

        Parameters
        ----------
        data : GoalHandle
            Goal to be sent to the action server containing the input data.
            Should be a ROS2 action goal compatible with the action server.

        Returns
        -------
        bool
            True if the message was sent successfully, False otherwise.
        """
        if self.process_action is None:
            self.log_error("Process action is not initialized")
            return False
        elif not self.process_action.wait_for_server(timeout_sec=1.0):
            self.log_error("Action server not available")
            return False

        self.log_debug("Uploading input")
        self.future = self.process_action.send_goal_async(data)
        rclpy.spin_until_future_complete(self.node, self.future)
        result = self.future.result()
        if result is None:
            self.log_error("Input upload failed")
            return False
        if not result.accepted:
            self.log_error("Input upload rejected")
            return False
        self.log_debug("Input uploaded successfully")
        return True

    def request_processing(
        self, get_time_func: Callable[[], float] = perf_counter
    ) -> bool:
        if self.future is None:
            self.log_error("No input uploaded")
            return False

        self.future = self.future.result().get_result_async()
        timemeasurements("protocol_inference_step", get_time_func)(
            rclpy.spin_until_future_complete
        )(self.node, self.future)
        return True

    def download_statistics(self, final: bool = False) -> Measurements:
        measurements = Measurements()
        if final is False:
            return measurements

        self.log_debug("Downloading statistics")

        if self.measurements_service is None:
            self.log_error("Measurements service is not initialized")
            return measurements
        elif not self.measurements_service.wait_for_service(timeout_sec=1.0):
            self.log_error("Measurements service not available")
            return measurements

        request = self._measurements_service_type.Request()
        future = self.measurements_service.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        result = future.result()
        if result is None:
            self.log_error("Measurements service call failed")
            return measurements

        jsondata = json.loads(result.message)
        measurements += jsondata
        return measurements

    def download_output(self) -> Tuple[bool, Optional[GoalHandle]]:
        """
        Gets result from processing goal from the target ROS2 node.

        Returns
        -------
        Tuple[bool, Optional[GoalHandle]]
            Tuple with status (True if successful)
            and received processing goal handle result.
        """
        if self.future is None:
            self.log_error("No input uploaded")
            return False, None

        KLogger.debug("Downloading output")
        result = self.future.result()
        self.future = None
        if result is None:
            self.log_error("Output download failed")
            return False, None
        self.log_debug("Output downloaded successfully")
        return True, result.result

    def disconnect(self):
        self.log_debug("Disconnecting node")
        self.node.destroy_node()
        self.log_debug("Successfully disconnected")

    def initialize_server(self) -> bool:
        raise NotImplementedError
