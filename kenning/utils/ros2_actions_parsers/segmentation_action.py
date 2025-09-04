# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base ROS2 action parser template.
"""


from typing import TYPE_CHECKING, Any, List

import numpy as np

if TYPE_CHECKING:
    from kenning_computer_vision_msgs.action import SegmentationAction

from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.utils.ros2_actions_parsers.base import ROS2ActionParser


class SegmentationActionParser(ROS2ActionParser):
    """
    A template class for all ROS2 Action parser.
    """

    associated_type = SegmObject

    associated_action_type = "SegmentationAction"

    @staticmethod
    def from_any(x: Any) -> "SegmentationAction.Goal":
        """
        A function that takes value x and
        convert it to appropriate ROS2 Action
        type defined in kenning messages.

        Parameters
        ----------
        x : Any
            Data to parse.

        Returns
        -------
        SegmentationAction.Goal
            Parsed ROS2 Action type.
        """
        from kenning_computer_vision_msgs.action import SegmentationAction
        from kenning_computer_vision_msgs.msg import VideoFrameMsg
        from sensor_msgs.msg import Image

        def prepare_image(frame: np.ndarray) -> Image:
            assert (
                len(frame.shape) == 3
            ), "Input data must be 3-dimensional and have BGR8 encoding"

            if frame.shape[2] > frame.shape[0]:
                frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame *= 255
                frame = frame.astype(np.uint8)

            img = Image()
            img._height = frame.shape[0]
            img._width = frame.shape[1]
            img._encoding = "bgr8"
            img._step = frame.shape[1] * frame.shape[2]
            img._data = frame.tobytes()
            return img

        goal = SegmentationAction.Goal()
        for frame_data in x:
            frame_msg = VideoFrameMsg()
            frame_msg.frame_id = 0
            frame_msg.video_id = "0"
            if type(frame_data) is dict:
                frame_msg._frame = prepare_image(frame_data["data"])
                frame_msg.frame_id = frame_data["frame_id"]
                frame_msg.video_id = frame_data["video_id"]
            else:
                frame_msg._frame = prepare_image(frame_data)
            goal._input.append(frame_msg)
        return goal

    @staticmethod
    def to_associated_type(
        x: "SegmentationAction.Result"
    ) -> List[List[SegmObject]]:
        """
        A function that takes result of
        ROS 2 action and converts them
        into associated value.

        Parameters
        ----------
        x : SegmentationAction.Result
            Result to parse.

        Returns
        -------
        List[List[SegmObject]]
            Associated type instance
        """
        from kenning_computer_vision_msgs.action import SegmentationAction

        if isinstance(x, SegmentationAction.Result):
            # perform conversion
            results = []

            for prediction in x.segmentations:
                prediction_res = []
                for i in range(len(prediction.classes)):
                    clsname = prediction.classes[i]
                    score = prediction.scores[i]
                    box = prediction.boxes[i]
                    xmin = box.xmin
                    ymin = box.ymin
                    xmax = box.xmax
                    ymax = box.ymax
                    mask = np.frombuffer(
                        prediction.masks[i]._data, dtype=np.uint8
                    )
                    mask = mask.reshape(
                        prediction.masks[i].dimension[0],
                        prediction.masks[i].dimension[1],
                    )
                    prediction_res.append(
                        SegmObject(
                            clsname,
                            None,
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                            mask,
                            score,
                            False,
                        )
                    )
                results.append(prediction_res)

        return [results]
