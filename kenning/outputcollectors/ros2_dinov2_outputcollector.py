# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
An OutputCollector-derived class used to broadcast DinoV2 output to ROS2 topic.

Requires 'rclpy' and 'kenning_computer_vision_msgs' packages to be sourced in
the environment.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from kenning_computer_vision_msgs.msg import (
        FrameDepthEstimationMsg,
        VideoFrameMsg,
    )

from kenning.core.exceptions import NotSupportedError
from kenning.core.outputcollector import OutputCollector


class ROS2DinoV2OutputCollector(OutputCollector):
    """
    ROS2 output collector that collects data from DinoV2 model and publishes
    it to a ROS2 topic.
    """

    arguments_structure = {
        "topic_name": {
            "description": "Name of the ROS2 topic for messages to be published to",  # noqa: E501
            "type": str,
            "default": "depth_frame",
        },
        "input_color_format": {
            "description": "Color format of the input images (RGB, BGR, GRAY)",
            "type": str,
            "required": False,
            "default": "RGB",
        },
        "input_memory_layout": {
            "description": "Memory layout of the input images (NHWC or NCHW)",
            "type": str,
            "required": False,
            "default": "NHWC",
        },
    }

    def __init__(
        self,
        topic_name: str = "depth_frame",
        input_color_format: Optional[str] = "RGB",
        input_memory_layout: Optional[str] = "NHWC",
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Creates ROS2DinoV2OutputCollector object.

        Parameters
        ----------
        topic_name : str
            Name of the ROS2 topic for messages to be published to.
        input_color_format : Optional[str]
            Color format of the input images (RGB, BGR, GRAY). Defaults to RGB.
        input_memory_layout : Optional[str]
            Memory layout of the input images (NHWC or NCHW). Defaults to NHWC.
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this runner.
        """
        self._topic_name = topic_name
        self._input_color_format = input_color_format
        self._input_memory_layout = input_memory_layout

        self._topic_publisher = None  # ROS2 topic publisher

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

        self.prepare()

    def prepare(self):
        from kenning_computer_vision_msgs.msg import FrameDepthEstimationMsg

        from kenning.utils.ros2_global_context import ROS2GlobalContext

        self._topic_publisher = ROS2GlobalContext.node.create_publisher(
            FrameDepthEstimationMsg, self._topic_name, 2
        )

    def run(self, inputs: Dict[str, Tuple[int, str]]) -> Dict[str, str]:
        if self._topic_publisher.get_subscription_count() == 0:
            return {}

        frame = inputs["frame_original"][0]
        depth = inputs["depth_data"][0]
        dino_msg = self._create_dino_msg(frame, depth)
        self._topic_publisher.publish(dino_msg)
        return {}

    def detach_from_output(self):
        self._topic_publisher.destroy()

    def should_close(self):
        return False

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification()

    @classmethod
    def parse_io_specification_from_json(cls, json_dict: Dict) -> Dict:
        return cls._get_io_specification()

    @classmethod
    def _get_io_specification(cls) -> Dict[str, List[Dict]]:
        """
        Creates runner IO specification with given parameters.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output layers specification.
        """
        return {
            "input": [
                {
                    "name": "frame_original",
                    "shape": (1, -1, -1, 3),
                    "dtype": "uint8",
                },
                {
                    "name": "depth_data",
                    "shape": [1, -1, -1],
                    "dtype": "float32",
                },
            ],
            "output": [],
        }

    def _create_dino_msg(
        self, original_frame: np.ndarray, predictions: np.ndarray
    ) -> "FrameDepthEstimationMsg":
        """
        Creates ROS2 frame depth estimation message from inputs.

        Parameters
        ----------
        original_frame: np.ndarray
            Raw original frame image.
        predictions: np.ndarray
            Depth data received from the model.

        Returns
        -------
        FrameDepthEstimationMsg
            FrameDepthEstimationMsg initialized from given data.
        """
        from kenning_computer_vision_msgs.msg import FrameDepthEstimationMsg

        dino_msg = FrameDepthEstimationMsg()
        dino_msg.frame = self._create_frame_msg(
            original_frame, self._topic_name
        )
        dino_msg.depth.rows = predictions.shape[0]
        dino_msg.depth.cols = predictions.shape[1]
        dino_msg.depth.data = predictions.ravel().tolist()
        return dino_msg

    def _create_frame_msg(
        self, image: np.ndarray, video_id: str
    ) -> "VideoFrameMsg":
        """
        Creates ROS2 Image message from given image.

        Parameters
        ----------
        image : np.ndarray
            Image to be converted to ROS2 Image message.
        video_id: str
            Frame ID that will be attached to this message.

        Returns
        -------
        VideoFrameMsg
            VideoFrameMsg message initialized with the given data
        """
        import sensor_msgs.msg
        from kenning_computer_vision_msgs.msg import VideoFrameMsg

        from kenning.utils.ros2_global_context import ROS2GlobalContext

        if self._input_memory_layout == "NCHW":
            image = np.transpose(image, (1, 2, 0))

        image = image.squeeze()
        message = VideoFrameMsg()

        frame = sensor_msgs.msg.Image()

        frame.header.stamp = ROS2GlobalContext.node.get_clock().now().to_msg()
        frame.header.frame_id = video_id
        frame.height, frame.width = image.shape[0], image.shape[1]
        frame.encoding = self._color_format_to_encoding(
            self._input_color_format
        )
        frame.is_bigendian = False
        frame.step = frame.width * image.shape[2]
        frame._data = image.tobytes()

        message.frame = frame
        message.video_id = video_id

        return message

    def _color_format_to_encoding(self, color_format: str) -> str:
        """
        Converts color format to ROS2 Image encoding.

        Parameters
        ----------
        color_format : str
            Color format to be converted to ROS2 Image encoding.

        Returns
        -------
        str
            Supported ROS2 image encoding.

        Raises
        ------
        ValueError
            Raised when unknown color format is provided
        """
        encodings = {
            "RGB": "rgb8",
            "BGR": "bgr8",
            "GRAY": "mono8",
        }
        if color_format not in encodings:
            raise ValueError("Unknown color format")
        return encodings[color_format]

    def process_output(self, input_data, output_data):
        raise NotSupportedError("Output processing is not supported.")
