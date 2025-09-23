# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ROS2-based inference communication protocol.

Requires 'rclpy' and 'kenning_computer_vision_msgs' packages to be sourced
in the environment.
"""

import json
import time
from asyncio import Future
from pathlib import Path
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from rclpy.impl.implementation_singleton import (
        rclpy_implementation as _rclpy,
    )

from kenning.core.exceptions import NotSupportedError
from kenning.core.measurements import Measurements, timemeasurements
from kenning.core.protocol import (
    Protocol,
    ServerDownloadCallback,
    ServerUploadCallback,
)
from kenning.dataconverters.ros2_dataconverter import ROS2DataConverter
from kenning.utils.class_loader import load_class
from kenning.utils.logger import KLogger

GoalHandle = TypeVar("GoalHandle", bound="_rclpy.ActionGoalHandle")


class ROS2Protocol(Protocol):
    """
    A ROS2-based runtime protocol for communication.
    It supports only a client side of the runtime protocol.

    Protocol is implemented using ROS2 services and action.
    """

    arguments_structure = {
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
        process_action_type_str: str,
        process_action_name: str,
        model_service_type_str: str = "std_srvs.srv.Trigger",
        model_service_name: str = "cvnode_prepare",
        measurements_service_type_str: str = "std_srvs.srv.Trigger",
        measurements_service_name: str = "cvnode_measurements",
        timeout: int = -1,
    ):
        """
        Initializes ROS2Protocol object.

        Parameters
        ----------
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
        timeout : int
            Response receive timeout in seconds. If negative, then waits for
            responses forever.
        """
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

        super().__init__(timeout=timeout)

    def _wait_for_future(self, future: Future) -> bool:
        """
        A function that will wait for future
        and terminate if timeout passed.

        Parameters
        ----------
        future : Future
            A future that we are waiting for.

        Return
        ------
        bool
            True if future completed, False when timeout is reached.
        """
        start = time.perf_counter()
        while not future.done():
            if self.timeout > -1 and (
                (time.perf_counter() - start) >= self.timeout
            ):
                return False

        return True

    def deduce_data_converter_from_io_spec(
        self, io_specification: Optional[Union[Dict, Path]] = None
    ) -> ROS2DataConverter:
        KLogger.debug(
            "Loading ros2 data converter with "
            f"type: {self.process_action_type_str}"
        )

        return ROS2DataConverter(
            ros2_message_type=self.process_action_type_str
        )

    def initialize_client(self):
        from kenning.utils.ros2_global_context import ROS2GlobalContext

        KLogger.debug(
            f"Initializing action client node {ROS2GlobalContext.node_name()}"
        )

        from rclpy.action import ActionClient

        self.process_action = ActionClient(
            ROS2GlobalContext.node,
            self._process_action_type,
            self.process_action_name,
            callback_group=None,
        )

        self.model_service = ROS2GlobalContext.node.create_client(
            self._model_service_type,
            self.model_service_name,
        )

        self.measurements_service = ROS2GlobalContext.node.create_client(
            self._measurements_service_type,
            self.measurements_service_name,
        )

        KLogger.debug("Successfully initialized client")
        return True

    def upload_io_specification(self, path: Path) -> bool:
        KLogger.warning(
            "IO specification is not supported in ROS2 protocol. " "Skipping."
        )
        return True

    def upload_model(self, path: Path) -> bool:
        KLogger.debug("Uploading model")
        if self.model_service is None:
            KLogger.error("Model service is not initialized")
            return False
        elif not self.model_service.wait_for_service(timeout_sec=1.0):
            KLogger.error("Model service not available")
            return False

        request = self._model_service_type.Request()
        future = self.model_service.call_async(request)

        success = self._wait_for_future(future)

        if not success or future.cancelled():
            future.cancel()
            KLogger.error("Model service call timed out")
            return False

        result = future.result()

        if result is None:
            KLogger.error("Model service call failed")
            return False
        KLogger.debug("Model uploaded successfully")
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
            KLogger.error("Process action is not initialized")
            return False
        elif not self.process_action.wait_for_server(timeout_sec=1.0):
            KLogger.error("Action server not available")
            return False

        KLogger.debug("Uploading input")
        self.future = self.process_action.send_goal_async(data)

        success = self._wait_for_future(self.future)

        if not success or self.future.cancelled():
            self.future.cancel()
            KLogger.error("Input upload timed out")
            return False

        result = self.future.result()
        if self.future is None:
            KLogger.error("Input upload failed")
            return False
        if not result.accepted:
            KLogger.error("Input upload rejected")
            return False
        KLogger.debug("Input uploaded successfully")
        return True

    def request_processing(
        self, get_time_func: Callable[[], float] = perf_counter
    ) -> bool:
        if self.future is None:
            KLogger.error("No input uploaded")
            return False

        def _request_processing(future):
            self._wait_for_future(future)

        if not self.future.done() or self.future.cancelled():
            self.future.cancel()
            KLogger.error("Input upload timed out")
            self.future = None
            return False

        self.future = self.future.result().get_result_async()
        timemeasurements("protocol_inference_step", get_time_func)(
            _request_processing
        )(self.future)
        return True

    def download_statistics(self, final: bool = False) -> Measurements:
        measurements = Measurements()
        if final is False:
            return measurements

        KLogger.debug("Downloading statistics")

        if self.measurements_service is None:
            KLogger.error("Measurements service is not initialized")
            return measurements
        elif not self.measurements_service.wait_for_service(timeout_sec=1.0):
            KLogger.error("Measurements service not available")
            return measurements

        request = self._measurements_service_type.Request()
        future = self.measurements_service.call_async(request)

        success = self._wait_for_future(future)

        if not success or future.cancelled():
            future.cancel()
            KLogger.error("Measurements service call timed out")
            return False

        result = future.result()

        if result is None:
            KLogger.error("Measurements service call failed")
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
            KLogger.error("No input uploaded")
            return False, None

        KLogger.debug("Downloading output")
        result = self.future.result()
        self.future = None
        if result is None:
            KLogger.error("Output download failed")
            return False, None
        KLogger.debug("Output downloaded successfully")
        return True, result.result

    def disconnect(self):
        from kenning.utils.ros2_global_context import ROS2GlobalContext

        KLogger.debug("Disconnecting node")
        ROS2GlobalContext.node.destroy_client(self.model_service)
        ROS2GlobalContext.node.destroy_client(self.measurements_service)
        KLogger.debug("Successfully disconnected")

    def initialize_server(
        self,
        client_connected_callback: Optional[Callable[Any, None]] = None,
        client_disconnected_callback: Optional[Callable[None, None]] = None,
    ) -> bool:
        raise NotSupportedError(
            "Kenning Server cannot use ROS2 (the protocol is client-side only)"
        )

    def request_optimization(
        self,
        model_path: Path,
        get_time_func: Callable[[], float] = time.perf_counter,
    ) -> Tuple[bool, Optional[bytes]]:
        raise NotSupportedError(
            "ROS2 protocol does not support request optimization."
        )

    def upload_optimizers(self, optimizers_cfg: Dict[str, Any]) -> bool:
        raise NotSupportedError(
            "ROS2 protocol does not support upload opimizers."
        )

    def upload_runtime(self, path: Path) -> bool:
        raise NotSupportedError(
            "ROS2 protocol does not support upload runtime."
        )

    def start_sending_logs(self):
        raise NotSupportedError("ROS2 protocol does not support sending logs.")

    def stop_sending_logs(self):
        raise NotSupportedError("ROS2 protocol does not support sending logs.")

    def listen_to_server_logs(self):
        raise NotSupportedError("ROS2 protocol does not support sending logs.")

    def serve(
        self,
        upload_input_callback: Optional[ServerUploadCallback] = None,
        upload_model_callback: Optional[ServerUploadCallback] = None,
        process_input_callback: Optional[ServerUploadCallback] = None,
        download_output_callback: Optional[ServerDownloadCallback] = None,
        download_stats_callback: Optional[ServerDownloadCallback] = None,
        upload_iospec_callback: Optional[ServerUploadCallback] = None,
        upload_optimizers_callback: Optional[ServerUploadCallback] = None,
        upload_unoptimized_model_callback: Optional[
            ServerUploadCallback
        ] = None,
        download_optimized_model_callback: Optional[
            ServerDownloadCallback
        ] = None,
        upload_runtime_callback: Optional[ServerUploadCallback] = None,
    ):
        raise NotSupportedError(
            "Kenning Server cannot use ROS2 (the protocol is client-side only)"
        )
