# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
An OutputCollector-derived class used to broadcast YOLACT output to ROS2 topic.

Requires 'rclpy' and 'cvnode_msgs' packages to be sourced in the environment.
"""

import numpy as np
import rclpy
import sensor_msgs.msg
from cvnode_msgs.msg import SegmentationMsg

from rclpy.node import Node
from typing import Dict, Tuple, List, Optional

from kenning.core.outputcollector import OutputCollector
from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.utils.args_manager import get_parsed_json_dict


class ROS2YolactOutputCollector(OutputCollector):
    """
    ROS2 output collector that collects data from YOLACT model and publishes
    it to a ROS2 topic.
    """

    arguments_structure = {
        'node_name': {
            'description': 'Name for the ROS2 node',
            'type': str,
            'required': True,
        },
        'topic_name': {
            'description': 'Name of the ROS2 topic for messages to be published to',  # noqa: E501
            'type': str,
            'required': True,
        },
        'input_color_format': {
            'description': 'Color format of the input images (RGB, BGR, GRAY)',
            'type': str,
            'required': False,
            'default': 'RGB',
        },
        'input_memory_layout': {
            'description': 'Memory layout of the input images (NHWC or NCHW)',
            'type': str,
            'required': False,
            'default': 'NHWC',
        },
    }

    def __init__(self,
                 node_name: str,
                 topic_name: str,
                 input_color_format: Optional[str] = 'RGB',
                 input_memory_layout: Optional[str] = 'NHWC',
                 inputs_sources: Dict[str, Tuple[int, str]] = {},
                 inputs_specs: Dict[str, Dict] = {},
                 outputs: Dict[str, str] = {}):
        """
        Creates ROS2YolactOutputCollector object.

        Parameters
        ----------
        node_name : str
            Name for the ROS2 node.
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
        self._node_name = node_name
        self._topic_name = topic_name
        self._input_color_format = input_color_format
        self._input_memory_layout = input_memory_layout

        self._node = None               # ROS2 node to be spinned
        self._topic_publisher = None    # ROS2 topic publisher

        super().__init__(
                inputs_sources=inputs_sources,
                inputs_specs=inputs_specs,
                outputs=outputs
        )

        self.prepare()

    def prepare(self):
        if not rclpy.ok():
            rclpy.init()
        self._node = Node(self._node_name)

        self._topic_publisher = self._node.create_publisher(
                SegmentationMsg,
                self._topic_name,
                2
        )

    def run(self, inputs: Dict[str, Tuple[int, str]]) -> Dict[str, str]:
        if (self._topic_publisher.get_subscription_count() == 0):
            return {}

        y = inputs['output'][0] if inputs['output'] else []
        yolact_msg = self._create_yolact_msg(inputs['frame'],
                                             y)
        self._topic_publisher.publish(yolact_msg)
        return {}

    def detach_from_output(self):
        self._topic_publisher.destroy()
        self._node.destroy_node()
        rclpy.shutdown()

    def should_close(self):
        return False

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self._input_memory_layout)

    @classmethod
    def parse_io_specification_from_json(cls, json_dict: Dict) -> Dict:
        parameterschema = cls.for_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        return cls._get_io_specification(
                parsed_json_dict['input_memory_layout']
        )

    @classmethod
    def _get_io_specification(cls, input_memory_layout: str):
        """
        Creates runner IO specification with given parameters.

        Parameters
        ----------
        input_memory_layout : str
            Memory layout of the input images (NHWC or NCHW).

        Returns
        -------
        Dict[str, List[Dict]] :
            Dictionary that conveys input and output layers specification.
        """
        return {
            'input': [
                {
                    'name': 'frame_original',
                    'shape': (1, -1, -1, -1),
                    'dtype': 'uint8'
                },
            ],
            'output': [],
        }

    def _extract_yolact_output(self, y: List[SegmObject],
                               yolact_msg: SegmentationMsg):
        """
        Extracts YOLACT output from given list of SegmObject.

        Parameters
        ----------
        y : List[SegmObject]
            List of SegmObject to be converted to numpy arrays.
        yolact_msg : SegmentationMsg
            SegmentationMsg to be filled with masks, boxes, scores and classes.
        """
        classes = []
        masks = np.array([], dtype=np.uint8)
        scores = []
        boxes = np.array([], dtype=np.float32)
        for obj in y:
            classes.append(obj.clsname)
            scores.append(float(obj.score))
            masks = np.concatenate((masks, obj.mask.flatten().astype(np.uint8)
                                    ))
            boxes = np.concatenate((boxes, (obj.xmin, obj.ymin, obj.xmax,
                                            obj.ymax)))
        yolact_msg._classes = classes
        yolact_msg._masks = masks
        yolact_msg._scores = scores
        yolact_msg._num_dets = len(classes)
        yolact_msg._boxes.extend(boxes)

    def _create_yolact_msg(self, image: np.ndarray, y: List[SegmObject]
                           ) -> SegmentationMsg:
        """
        Creates SegmentationMsg from given image and YOLACT output.

        Parameters
        ----------
        image : np.ndarray
            Image to be converted to ROS2 Image message.
        y : List[SgmeObject]
            List of SegmObject to be converted to numpy arrays.

        Returns
        -------
        SegmentationMsg : Filled SegmentationMsg message ready to be published.
        """
        yolact_msg = SegmentationMsg()
        yolact_msg.frame = self._create_frame_msg(image)
        self._extract_yolact_output(y, yolact_msg)
        return yolact_msg

    def _create_frame_msg(self, image: np.ndarray) -> sensor_msgs.msg.Image:
        """
        Creates ROS2 Image message from given image.

        Parameters
        ----------
        image : np.ndarray
            Image to be converted to ROS2 Image message.

        Returns
        -------
        sensor_msgs.msg.Image : Image message filled with given image data.
        """
        image = image.squeeze()
        if self._input_memory_layout == 'NCHW':
            image = np.transpose(image, (1, 2, 0))

        message = sensor_msgs.msg.Image()

        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = 'camera_frame'
        message.height, message.width = image.shape[0], image.shape[1]
        message.encoding = self._color_format_to_encoding(
                self._input_color_format
        )
        message.is_bigendian = False
        message.step = message.width * image.shape[2]
        message._data = image.tobytes()
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
        str : Supported ROS2 image encoding.
        """
        encodings = {
                'RGB': 'rgb8',
                'BGR': 'bgr8',
                'GRAY': 'mono8',
                }
        if color_format not in encodings:
            raise ValueError('Unknown color format')
        return encodings[color_format]
