# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A Dataprovider-derived class used to interface with a
ROS2 CameraNode.
"""

from abc import ABC, abstractmethod
from threading import Event
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    import sensor_msgs.msg

from kenning.core.dataprovider import DataProvider
from kenning.core.exceptions import KenningDataProviderError


class ROS2DataProvider(DataProvider, ABC):
    """
    Provides frames collected from ROS 2 topic to Kenning nodes.
    """

    arguments_structure = {
        "topic_name": {
            "description": "Name of the topic to receive messages from",
            "type": str,
            "required": True,
        }
    }

    def __init__(
        self,
        topic_name: str,
        message_type: Any,
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        self._topic_name = topic_name
        self._topic_subscriber = None
        self._message_type = message_type

        self._data = None
        self._triggered = Event()

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    def prepare(self):
        from kenning.utils.ros2_global_context import ROS2GlobalContext

        self._topic_subscriber = ROS2GlobalContext.node.create_subscription(
            self._message_type, self._topic_name, self._topic_callback, 2
        )

    def detach_from_source(self):
        self._topic_subscriber.destroy()

    def fetch_input(self) -> "sensor_msgs.msg.Image":
        if self._topic_subscriber is None:
            raise KenningDataProviderError("ROS2 Subscriber not initialized")

        self._triggered.clear()

        self._triggered.wait()

        return self._data

    def _topic_callback(self, msg: "sensor_msgs.msg.Image"):
        """
        Callback function for ROS2 topic subscriber.
        Sets the _triggered flag to True and stores the received message.

        Parameters
        ----------
        msg : sensor_msgs.msg.Image
            Received message.
        """
        self._triggered.set()
        self._data = msg

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_io_specification(self) -> Dict[str, List[Dict]]:
        ...
