# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
An OutputCollector-derived class used to visualize pose
estimated by pose estimation models.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from kenning.core.outputcollector import OutputCollector
from kenning.datasets.helpers.pose_estimation import Pose
from kenning.utils.visualization_tools import draw_pose, generate_color


class PoseVisualizer(OutputCollector):
    """
    Visualizes detection predictions for video sequences.
    """

    arguments_structure = {
        "output_width": {
            "argparse_name": "--output-width",
            "description": "Output image width",
            "type": int,
            "required": False,
        },
        "output_height": {
            "argparse_name": "--output-height",
            "description": "Output image height",
            "type": int,
            "required": False,
        },
        "point_radius": {
            "description": "A radius size of the pose points",
            "type": int,
            "default": 3,
        },
        "bbox_thickness": {
            "description": "A thickness of bounding box walls.",
            "type": int,
            "default": 2,
        },
        "line_thickness": {
            "description": "A thickness of line in pose.",
            "type": int,
            "default": 2,
        },
        "save_to_file": {
            "argparse_name": "--save-to-file",
            "description": "If output should be saved to file",
            "type": bool,
            "required": False,
            "default": False,
        },
        "save_path": {
            "argparse_name": "--save-path",
            "description": "Output image height",
            "type": str,
            "required": False,
            "default": "./",
        },
    }

    def __init__(
        self,
        output_width: int = 1024,
        output_height: int = 576,
        point_radius: int = 3,
        bbox_thickness: int = 2,
        line_thickness: int = 2,
        save_to_file: bool = False,
        save_path: Path = "./",
        save_fps: int = 25,
        window_name: str = "stream",
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Creates the detection visualizer.

        Parameters
        ----------
        output_width : int
            Width of the output.
        output_height : int
            Height of the output.
        point_radius : int
            A radius size of the pose points.
        bbox_thickness : int
            A thickness of bounding box walls.
        line_thickness : int
            A thickness of line in pose.
        save_to_file : bool
            True if frames should be saved to file. In other case
            they are presented using opencv.
        save_path : Path
            Path where frames should be saved.
        save_fps : int
            Frames pre second of the saved video.
        window_name : str
            Name of opencv window.
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        self.window_name = window_name
        self.output_width = output_width
        self.output_height = output_height
        self.point_radius = point_radius
        self.bbox_thickness = bbox_thickness
        self.line_thickness = line_thickness
        self.save_to_file = save_to_file
        self.save_fps = save_fps
        self.out = None
        self.save_path = Path(save_path)
        if save_to_file:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(
                str(self.save_path),
                codec,
                self.save_fps,
                (self.output_width, self.output_height),
            )
        self.font_scale = 0.7
        self.font_thickness = 2
        self.color_dict = defaultdict(generate_color)

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    def cleanup(self):
        self.detach_from_output()

    def visualize_data(
        self, input_data: np.ndarray, output_data: List[Pose]
    ) -> np.ndarray:
        """
        Method used to add visualizations of the models output.
        It draws bounding boxes, aswell as keypoints.

        Parameters
        ----------
        input_data : np.ndarray
            The original image.
        output_data : List[Pose]
            List of estimated poses represented as Pose.

        Returns
        -------
        np.ndarray
            The modified image with visualizations drawn.
        """
        if not isinstance(output_data, list):
            output_data = [output_data]

        out_img = input_data
        for i, pose in enumerate(output_data):
            points_color = self.color_dict[i]
            bbox_color = self.color_dict[i]

            out_img = draw_pose(
                out_img,
                pose,
                self.point_radius,
                self.bbox_thickness,
                self.line_thickness,
                points_color,
                bbox_color,
            )

        return out_img

    def process_output(
        self,
        input_data: np.ndarray,  # since the original frames are passed in, this should always be HWC, uint8  # noqa: E501
        output_data: List[List[Pose]],
    ):
        """
        Method used to visualize predicted classes on input images.

        Parameters
        ----------
        input_data : np.ndarray
            The original image.
        output_data : List[List[Pose]]
            List of found objects represented as Pose.
        """
        # TODO: consider adding support for variable batch sizes
        output_data = output_data[0]
        input_data = input_data[0]

        if input_data.shape[0] == 3:
            # convert to channel-last
            input_data = input_data.transpose(1, 2, 0)

        frame = self.visualize_data(input_data, output_data)

        frame = cv2.resize(frame, (self.output_width, self.output_height))
        if self.save_to_file:
            frame = (frame * 255).astype(np.uint8)
            self.out.write(frame)
        else:
            cv2.imshow(self.window_name, frame)

    def should_close(self) -> bool:
        return cv2.waitKey(1) == 27

    def detach_from_output(self):
        if self.out:
            self.out.release()
        else:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                # no window to close
                pass

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return {
            "input": [
                {
                    "name": "frame",
                    "shape": [(1, -1, -1, 3), (1, 3, -1, -1)],
                    "dtype": "float32",
                },
                {
                    "name": "pose_data",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.pose_estimation.Pose",  # noqa: E501
                    },
                },
            ],
            "output": [],
        }

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        input_data = inputs["frame"]
        output_data = inputs["pose_data"]
        self.process_output(input_data, output_data)
        return {}
