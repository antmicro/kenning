# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A Dataprovider-derived class used to interface with a
ROS2 CameraNode.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from kenning.dataproviders.ros2_camera_node_data_provider import (
    ROS2CameraNodeDataProvider,
)
from kenning.datasets.helpers.detection_and_segmentation import (
    FrameSegmObject,
    SegmObject,
)


class ROS2SegmentationDataProvider(ROS2CameraNodeDataProvider):
    """
    Provides frames collected from ROS 2 topic to Kenning nodes.
    """

    arguments_structure = {}

    def __init__(
        self,
        topic_name: str,
        color_format: Optional[str] = None,
        output_width: int = None,
        output_height: int = None,
        output_memory_layout: str = "NHWC",
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        from kenning_computer_vision_msgs.msg import FrameSegmentationMsg

        super().__init__(
            topic_name=topic_name,
            color_format=color_format,
            output_width=output_width,
            output_height=output_height,
            output_memory_layout=output_memory_layout,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
            message_type=FrameSegmentationMsg,
        )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        frame_segmentation = self.fetch_input()

        img_msg = frame_segmentation.frame.frame

        frame, frame_original = self.preprocess_input(img_msg)

        segmentation = frame_segmentation.segmentation

        segm_output = []

        for i in range(len(segmentation.classes)):
            mask = np.frombuffer(segmentation.masks[i].data, dtype=np.uint8)
            mask = mask.reshape(
                (
                    segmentation.masks[i].dimension[0],
                    segmentation.masks[i].dimension[1],
                )
            )

            segm = SegmObject(
                clsname=segmentation.classes[i],
                maskpath=None,
                xmin=segmentation.boxes[i].xmin,
                ymin=segmentation.boxes[i].ymin,
                xmax=segmentation.boxes[i].xmax,
                ymax=segmentation.boxes[i].ymax,
                mask=mask,
                score=segmentation.scores[i],
                iscrowd=False,
            )

            segm_output.append(segm)

        return {
            "frame": frame,
            "frame_original": frame_original,
            "segmentations": [
                FrameSegmObject(frame=frame[0], segments=segm_output)
            ],
        }

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        io_spec = super().get_io_specification()

        io_spec["output"].append(
            {
                "name": "segmentations",
                "type": "List",
                "dtype": "kenning.datasets.helpers."
                "detection_and_segmentation.FrameSegmObject",
            },
        )

        return io_spec
