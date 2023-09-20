# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A DataConverter-derived class used to manipulate the data using the
ModelWrapper object for compatibility between RuntimeProtocol and Dataset
during runtime.
"""

from typing import List

import numpy as np
from kenning_computer_vision_msgs.action import SegmentationAction
from sensor_msgs.msg import Image

from kenning.core.dataconverter import DataConverter
from kenning.datasets.helpers.detection_and_segmentation import SegmObject


class ROS2SegmentationDataConverter(DataConverter):

    def __init__(self):
        """
        Initializes the ModelWrapperDataConverter object.
        """
        super().__init__()

    def to_message(self, data: List[np.ndarray]) -> SegmentationAction.Goal:
        """
        Converts input frames to segmentation action goal.

        Parameters
        ----------
        data : List[np.ndarray]
            The input data to be converted.

        Returns
        -------
        SegmentationAction.Goal :
            The converted segmentation action goal.
        """
        goal = SegmentationAction.Goal()
        for frame in data:
            item_shape = frame.shape
            if (len(item_shape) > 2) and (item_shape[-1] > item_shape[-3]):
                frame = np.moveaxis(frame, -3, -1)
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame *= 255
                frame = frame.astype(np.uint8)
            img = Image()
            img._height = frame.shape[0]
            img._width = frame.shape[1]
            img._encoding = 'bgr8'
            img._step = frame.shape[1] * frame.shape[2]
            img._data = frame.tobytes()
            goal._input.append(img)
        return goal

    def from_message(self, data: SegmentationAction.Result
                     ) -> List[List[SegmObject]]:
        """
        Converts segmentation action result to SegmObject list.

        Parameters
        ----------
        data : SegmentationAction.Result
            Result of the segmentation action.

        Returns
        -------
        List[List[SegmObject]] :
            The converted data.
        """
        assert data.success, 'Segmentation action failed'
        result = []
        for frame in data.output:
            tmp_res = []
            for i in range(len(frame.classes)):
                clsname = frame.classes[i]
                score = frame.scores[i]
                box = frame.boxes[i]
                xmin = box.xmin
                ymin = box.ymin
                xmax = box.xmax
                ymax = box.ymax
                mask = np.frombuffer(frame.masks[i]._data,
                                     dtype=np.uint8)
                mask = mask.reshape(
                    frame.masks[i].dimension[0],
                    frame.masks[i].dimension[1])
                tmp_res.append(SegmObject(clsname, None,
                                          xmin, ymin, xmax, ymax,
                                          mask, score, False))
            result.append(tmp_res)
        return result
