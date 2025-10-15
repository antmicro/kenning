# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
An OutputCollector-derived class used to broadcast YOLACT output to ROS2 topic.

Requires 'rclpy' and 'kenning_computer_vision_msgs' packages to be sourced in
the environment.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from kenning_computer_vision_msgs.msg import (
        FramePoseEstimationMsg,
        PoseEstimationMsg,
        VideoFrameMsg,
    )

from kenning.core.exceptions import NotSupportedError
from kenning.core.outputcollector import OutputCollector
from kenning.datasets.helpers.pose_estimation import Pose
from kenning.utils.args_manager import get_parsed_json_dict


class ROS2PoseOutputCollector(OutputCollector):
    """
    ROS2 output collector that collects data from pose estimation models
    like MMPose.
    """

    arguments_structure = {
        "topic_name": {
            "description": "Name of the ROS2 topic for messages to be published to",  # noqa: E501
            "type": str,
            "required": True,
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
        topic_name: str,
        input_color_format: Optional[str] = "RGB",
        input_memory_layout: Optional[str] = "NHWC",
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Creates ROS2PoseOutputCollector object.

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
        from kenning_computer_vision_msgs.msg import FramePoseEstimationMsg

        from kenning.utils.ros2_global_context import ROS2GlobalContext

        self._topic_publisher = ROS2GlobalContext.node.create_publisher(
            FramePoseEstimationMsg, self._topic_name, 2
        )

    def run(self, inputs: Dict[str, Tuple[int, str]]) -> Dict[str, str]:
        if self._topic_publisher.get_subscription_count() == 0:
            return {}

        y = inputs["output"][0] if inputs["output"] else []
        yolact_msg = self._create_pose_msg(inputs["frame_original"], y)
        self._topic_publisher.publish(yolact_msg)
        return {}

    def detach_from_output(self):
        self._topic_publisher.destroy()

    def should_close(self):
        return False

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self._input_memory_layout)

    @classmethod
    def parse_io_specification_from_json(cls, json_dict: Dict) -> Dict:
        parameterschema = cls.for_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        return cls._get_io_specification(
            parsed_json_dict["input_memory_layout"]
        )

    @classmethod
    def _get_io_specification(
        cls, input_memory_layout: str
    ) -> Dict[str, List[Dict]]:
        """
        Creates runner IO specification with given parameters.

        Parameters
        ----------
        input_memory_layout : str
            Memory layout of the input images (NHWC or NCHW).

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output layers specification.
        """
        return {
            "input": [
                {
                    "name": "frame_original",
                    "shape": (1, -1, -1, -1),
                    "dtype": "uint8",
                },
            ],
            "output": [],
        }

    def _extract_pose_output(
        self, y: List[Pose], pose_msg: "PoseEstimationMsg"
    ):
        """
        Extracts Pose output from given list of Pose.

        Parameters
        ----------
        y : List[Pose]
            List of SegmObject to be converted to numpy arrays.
        pose_msg : PoseEstimationMsg
            PoseEstimationMsg to be filled with poses and
            boxes.
        """
        from kenning_computer_vision_msgs.msg import (
            BoxMsg,
            KeyPoint2dMsg,
            MaskMsg,
            PoseEstimationMsg,
            PoseMsg,
            SegmentationMsg,
        )

        if not isinstance(y, list):
            y = [y]

        estimation = PoseEstimationMsg()

        segmentation = SegmentationMsg()

        poses = []

        classes, scores, masks, boxes = [], [], [], []

        for obj in y:
            keypoints = []

            for point in obj.keypoints:
                keypoint = KeyPoint2dMsg()

                keypoint.x = point.x
                keypoint.y = point.y
                keypoint.id = point.id

                keypoints.append(keypoint)

            pose = PoseMsg()

            pose.keypoints = keypoints

            poses.append(pose)

            bbox = BoxMsg()

            bbox.xmin = obj.segm.xmin
            bbox.ymin = obj.segm.ymin
            bbox.xmax = obj.segm.xmax
            bbox.ymax = obj.segm.ymax

            boxes.append(bbox)

            mask = MaskMsg()

            if obj.segm.mask is not None:
                obj_mask = obj.segm.mask.astype(np.uint8)

                mask._data = obj_mask.flatten()
                mask.dimension = [obj_mask.shape[0], obj_mask.shape[1]]

            classes.append(obj.segm.clsname)

            scores.append(obj.segm.score)

        estimation.poses = poses

        segmentation.boxes = boxes
        segmentation.masks = masks
        segmentation.classes = classes
        segmentation.scores = scores

        estimation.segmentation = segmentation

        pose_msg.estimation = estimation

    def _create_pose_msg(
        self, image: np.ndarray, y: List[Pose]
    ) -> "FramePoseEstimationMsg":
        """
        Creates FramePoseEstimationMsg from given image and  output.

        Parameters
        ----------
        image : np.ndarray
            Image to be converted to ROS2 Image message.
        y : List[Pose]
            List of Pose to be converted to numpy arrays.

        Returns
        -------
        FramePoseEstimationMsg
            Filled FramePoseEstimationMsg message ready to be published.
        """
        from kenning_computer_vision_msgs.msg import FramePoseEstimationMsg

        pose_msg = FramePoseEstimationMsg()
        pose_msg.frame = self._create_frame_msg(image)
        self._extract_pose_output(y, pose_msg)
        return pose_msg

    def _create_frame_msg(self, image: np.ndarray) -> "VideoFrameMsg":
        """
        Creates ROS2 Image message from given image.

        Parameters
        ----------
        image : np.ndarray
            Image to be converted to ROS2 Image message.

        Returns
        -------
        VideoFrameMsg
            VideoFrameMsg message filled with given image data.
        """
        image = image.squeeze()
        if self._input_memory_layout == "NCHW":
            image = np.transpose(image, (1, 2, 0))

        import sensor_msgs.msg
        from kenning_computer_vision_msgs.msg import VideoFrameMsg

        from kenning.utils.ros2_global_context import ROS2GlobalContext

        message = VideoFrameMsg()

        frame = sensor_msgs.msg.Image()

        frame.header.stamp = ROS2GlobalContext.node.get_clock().now().to_msg()
        frame.header.frame_id = "camera_frame"
        frame.height, frame.width = image.shape[0], image.shape[1]
        frame.encoding = self._color_format_to_encoding(
            self._input_color_format
        )
        frame.is_bigendian = False
        frame.step = frame.width * image.shape[2]
        frame._data = image.tobytes()

        message.frame = frame
        message.video_id = "camera_frame"

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
