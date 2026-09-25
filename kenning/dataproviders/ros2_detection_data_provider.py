# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A Dataprovider-derived class used to interface with a
ROS2 CameraNode.
"""

from typing import Any, Dict, List, Optional, Tuple

from kenning.dataproviders.ros2_camera_node_data_provider import (
    ROS2CameraNodeDataProvider,
)
from kenning.datasets.helpers.detection_and_segmentation import (
    FrameSegmObject,
    SegmObject,
)


class ROS2DetectionDataProvider(ROS2CameraNodeDataProvider):
    """
    Provides frames and object detections collected from a ROS 2 topic
    to Kenning nodes.
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
        frame_detection = self.fetch_input()

        img_msg = frame_detection.frame.frame

        frame, frame_original = self.preprocess_input(img_msg)

        detection = frame_detection.segmentation

        segm_output = []

        for i in range(len(detection.boxes)):
            segm = SegmObject(
                clsname=detection.classes[i],
                maskpath=None,
                xmin=detection.boxes[i].xmin,
                ymin=detection.boxes[i].ymin,
                xmax=detection.boxes[i].xmax,
                ymax=detection.boxes[i].ymax,
                mask=None,
                score=detection.scores[i],
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
