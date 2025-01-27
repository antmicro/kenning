# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A DataConverter-derived class used to manipulate the data using the
SegmentationAction object for compatibility between surronding blocks
during runtime.
"""

from typing import Any, Dict, List, Union

import numpy as np
from kenning_computer_vision_msgs.action import SegmentationAction
from kenning_computer_vision_msgs.msg import VideoFrameMsg
from sensor_msgs.msg import Image

from kenning.core.dataconverter import DataConverter
from kenning.datasets.helpers.detection_and_segmentation import SegmObject


class ROS2SegmentationDataConverter(DataConverter):
    """
    Converts input and output data for Instance Segmentation to ROS 2 topics.
    """

    def __init__(self):
        """
        Initializes the ModelWrapperDataConverter object.
        """
        super().__init__()

    def to_next_block(
        self, data: List[Union[Dict[str, Any], np.array]]
    ) -> SegmentationAction.Goal:
        """
        Converts input frames to segmentation action goal.
        Assumes that input data has BGR8 encoding.

        Parameters
        ----------
        data : List[Union[Dict[str, Any], np.array]]
            The input data to be converted.

        Returns
        -------
        SegmentationAction.Goal
            The converted segmentation action goal.
        """

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
        for frame_data in data:
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

    def to_previous_block(
        self, data: SegmentationAction.Result
    ) -> List[List[SegmObject]]:
        """
        Converts segmentation action result to SegmObject list.
        Assumes that if more than one frame is present, the output is for
        sequence of frames.

        Parameters
        ----------
        data : SegmentationAction.Result
            Result of the segmentation action.

        Returns
        -------
        List[List[SegmObject]]
            The converted data.
        """
        assert data.success, "Segmentation action failed"
        result = []
        for prediction in data.segmentations:
            prediction_res = []
            for i in range(len(prediction.classes)):
                clsname = prediction.classes[i]
                score = prediction.scores[i]
                box = prediction.boxes[i]
                xmin = box.xmin
                ymin = box.ymin
                xmax = box.xmax
                ymax = box.ymax
                mask = np.frombuffer(prediction.masks[i]._data, dtype=np.uint8)
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
            result.append(prediction_res)
        return result
