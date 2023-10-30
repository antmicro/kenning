# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A Dataprovider-derived class used to interface with a
ROS2 CameraNode.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import sensor_msgs.msg
from rclpy.node import Node

from kenning.core.dataprovider import DataProvider
from kenning.utils.args_manager import get_parsed_json_dict


class ROS2CameraNodeDataProvider(DataProvider):
    """
    Provides frames collected from ROS 2 topic to Kenning nodes.
    """

    arguments_structure = {
        "node_name": {
            "description": "Name of the ROS2 node",
            "type": str,
            "required": True,
        },
        "topic_name": {
            "description": "Name of the topic to receive messages from",
            "type": str,
            "required": True,
        },
        "color_format": {
            "argparse_name": "--color-format",
            "description": "Color format for the processed image. "
            "If not set, uses original image format."
            "Possible values: RGB, BGR, GRAY",
            "type": str,
            "required": False,
            "default": None,
        },
        "output_width": {
            "argparse_name": "--output-width",
            "description": "Width to resize processed image to."
            "If not set, uses original image width",
            "type": int,
            "required": False,
            "default": None,
        },
        "output_height": {
            "argparse_name": "--output-height",
            "description": "Height to resize processed image to."
            "If not set, uses original image height",
            "type": int,
            "required": False,
            "default": None,
        },
        "output_memory_layout": {
            "argparse_name": "--output-memory-layout",
            "description": "Memory layout of the output image."
            "Possible values: NHWC, NCHW",
            "type": str,
            "required": False,
            "default": "NHWC",
        },
    }

    def __init__(
        self,
        node_name: str,
        topic_name: str,
        color_format: Optional[str] = None,
        output_width: int = None,
        output_height: int = None,
        output_memory_layout: str = "NHWC",
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        self._node = None
        self._node_name = node_name

        self._topic_name = topic_name
        self._topic_subscriber = None

        self._color_format = color_format
        self._output_width = output_width
        self._output_height = output_height
        self._output_memory_layout = output_memory_layout

        self._data = None
        self._supported_color_formats = ("RGB", "BGR", "GRAY")

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    def prepare(self):
        if not rclpy.ok():
            rclpy.init()
        self._node = Node(self._node_name)

        self._topic_subscriber = self._node.create_subscription(
            sensor_msgs.msg.Image, self._topic_name, self._topic_callback, 2
        )

    def detach_from_source(self):
        self._topic_subscriber.destroy()
        self._node.destroy_node()
        rclpy.shutdown()

    def fetch_input(self) -> sensor_msgs.msg.Image:
        if self._topic_subscriber is None:
            raise ROS2DataproviderException("Subscriber not initialized")

        self._triggered = False
        while not self._triggered:
            rclpy.spin_once(self._node)
        return self._data

    def preprocess_input(
        self, data: sensor_msgs.msg.Image
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts ROS2 Image message to numpy array.

        Parameters
        ----------
        data : sensor_msgs.msg.Image
            ROS2 Image message.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing transformed image and original image.
        """
        img_orig = np.frombuffer(data.data, dtype=np.uint8)

        img_orig = img_orig.reshape(data.height, data.width, -1)

        if self._color_format in self._supported_color_formats:
            image_format = self._detect_image_format(data.encoding)
            if image_format != self._color_format:
                img_orig = self._convert_image_format(
                    img_orig, image_format, self._color_format
                )
        img_size = (data.width, data.height)
        if self._output_width is not None:
            img_size = (img_size[0], self._output_width)
        if self._output_height is not None:
            img_size = (self._output_height, img_size[1])
        img = cv2.resize(img_orig, img_size)
        img = img.astype(np.float32) / 255.0

        if self._output_memory_layout == "NCHW":
            img_orig = np.transpose(img_orig, (2, 0, 1))
            img = np.transpose(img, (2, 0, 1))

        return np.expand_dims(img, axis=0), np.expand_dims(img_orig, axis=0)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img_msg = self.fetch_input()
        frame, frame_original = self.preprocess_input(img_msg)
        return {"frame": frame, "frame_original": frame_original}

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(
            self._color_format,
            self._output_width,
            self._output_height,
            self._output_memory_layout,
            self.outputs,
        )

    def parse_io_specification_from_json(cls, json_dict: Dict) -> Dict:
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        outputs = {}
        if parsed_json_dict.has_key("outputs"):
            outputs = parsed_json_dict["outputs"]
        return cls._get_io_specification(
            parsed_json_dict["color_format"],
            parsed_json_dict["output_width"],
            parsed_json_dict["output_height"],
            parsed_json_dict["output_memory_layout"],
            outputs,
        )

    def _topic_callback(self, msg: sensor_msgs.msg.Image):
        """
        Callback function for ROS2 topic subscriber.
        Sets the _triggered flag to True and stores the received message.

        Parameters
        ----------
        msg : sensor_msgs.msg.Image
            Received message.
        """
        self._triggered = True
        self._data = msg

    def _detect_image_format(self, encoding: str) -> str:
        """
        Detects image format from ROS2 encoding.

        Parameters
        ----------
        encoding : str
            ROS2 encoding format.

        Returns
        -------
        str
            Image format. Possible values: RGB, BGR, GRAY.

        Raises
        ------
        ROS2DataproviderException
            If encoding is not supported.
        """
        if encoding == "rgb8":
            return "RGB"
        elif encoding == "bgr8" or encoding == "8UC3":
            return "BGR"
        elif encoding == "mono8" or encoding == "8UC1":
            return "GRAY"

        else:
            raise ROS2DataproviderException(
                "Unsupported image format: {}".format(encoding)
            )

    def _convert_image_format(
        self, img: np.ndarray, src_format: str, dst_format: str
    ) -> np.ndarray:
        """
        Converts image format from a source format to a destination format.

        Parameters
        ----------
        img : np.ndarray
            Image to be converted.
        src_format : str
            Source image format. Possible values: RGB, BGR, GRAY.
        dst_format : str
            Destination image format. Possible values: RGB, BGR, GRAY.

        Returns
        -------
        np.ndarray
            Converted image.
        """
        if src_format == dst_format:
            return img
        elif dst_format == "GRAY":
            if src_format == "RGB":
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            if src_format == "GRAY":
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                return np.flip(img, axis=2)

    def _get_io_specification(
        cls,
        color_format: str,
        output_width: int,
        output_height: int,
        output_memory_layout: str,
        outputs: Optional[Dict[str, str]] = {},
    ) -> Dict[str, List[Dict]]:
        """
        Creates runner IO specification from provided parameters.

        Parameters
        ----------
        color_format : str
            Color format of the input image. Possible values: RGB, BGR, GRAY.
        output_width : int
            Width of the output image.
        output_height : int
            Height of the output image.
        output_memory_layout : str
            Memory layout of the output image. Possible values: NCHW, NHWC.
        outputs : Optional[Dict[str, str]]
            Dictionary of output names and their types.

        Returns
        -------
        Dict[str, List[Dict]]
            Runner IO specification.
        """
        channels = 3
        height = -1
        width = -1
        if color_format == "GRAY":
            channels = 1
        if output_width is not None:
            width = output_width
        if output_height is not None:
            height = output_height
        if output_memory_layout == "NHWC":
            frame_shape = (1, height, width, channels)
        else:
            frame_shape = (1, channels, height, width)
        return {
            "input": [],
            "output": [
                {"name": "frame", "shape": frame_shape, "dtype": "float32"},
                {
                    "name": "frame_original",
                    "shape": (1, -1, -1, -1),
                    "dtype": "uint8",
                },
            ],
        }


class ROS2DataproviderException(Exception):
    """
    Exception to be raised when ROS2CameraNodeDataProvider misbehaves.
    """

    def __init__(self, message):
        super().__init__(message)
