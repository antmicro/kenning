# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A OutputCollector-derived class used to visualize bounding box
data on input images and display/save them.
"""

import colorsys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from kenning.core.outputcollector import OutputCollector
from kenning.datasets.helpers.detection_and_segmentation import DetectObject


def generate_color() -> Tuple[float, float, float]:
    """
    Generates a random RGB color.

    Returns
    -------
    Tuple[float, float, float]
        Color in (r,g,b) format.
    """
    return colorsys.hsv_to_rgb(np.random.rand(), 1, 1)


class DetectionVisualizer(OutputCollector):
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

    def compute_coordinates(
        self, coord: Tuple[float, float]
    ) -> Tuple[int, int]:
        """
        Computes coordinates in pixel-position form from 0-1 floats.

        Parameters
        ----------
        coord : Tuple[float, float]
            The 0-1 ranged coordinates.

        Returns
        -------
        Tuple[int, int]
            Size-based coordinates.
        """
        return (
            int(coord[0] * self.output_width),
            int(coord[1] * self.output_height),
        )

    def visualize_data(
        self, input_data: np.ndarray, output_data: List[DetectObject]
    ) -> np.ndarray:
        """
        Method used to add visualizations of the models output.
        It draws bounding boxes, class names and score onto
        the original image.

        Parameters
        ----------
        input_data : np.ndarray
            The original image.
        output_data : List[DetectObject]
            List of found objects represented as DetectObjects.

        Returns
        -------
        np.ndarray
            The modified image with visualizations drawn.
        """
        out_img = input_data
        for i in output_data:
            low_pair = self.compute_coordinates((i.xmin, i.ymin))
            high_pair = self.compute_coordinates((i.xmax, i.ymax))
            out_img = cv2.rectangle(
                out_img, low_pair, high_pair, self.color_dict[i.clsname], 2
            )
            out_img = cv2.putText(
                out_img,
                i.clsname,
                (low_pair[0], low_pair[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color_dict[i.clsname],
                self.font_thickness,
            )
            out_img = cv2.putText(
                out_img,
                "score: {:.2f}".format(i.score),
                (low_pair[0] + 5, high_pair[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color_dict[i.clsname],
                self.font_thickness,
            )
        return out_img

    def process_output(
        self,
        input_data: np.ndarray,  # since the original frames are passed in, this should always be HWC, uint8  # noqa: E501
        output_data: List[List[DetectObject]],
    ):
        """
        Method used to visualize predicted classes on input images.

        Parameters
        ----------
        input_data : np.ndarray
            The original image.
        output_data : List[List[DetectObject]]
            List of found objects represented as DetectObjects.
        """
        # TODO: consider adding support for variable batch sizes
        output_data = output_data[0]
        input_data = input_data[0]

        if input_data.shape[0] == 3:
            # convert to channel-last
            input_data = input_data.transpose(1, 2, 0)

        input_data = cv2.resize(
            input_data, (self.output_width, self.output_height)
        )
        if self.save_to_file:
            frame = self.visualize_data(input_data, output_data)
            frame = (frame * 255).astype(np.uint8)
            self.out.write(frame)
        else:
            cv2.imshow(
                self.window_name, self.visualize_data(input_data, output_data)
            )

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
                {"name": "detection_data", "type": "List[DetectObject]"},
            ],
            "output": [],
        }

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        input_data = inputs["frame"]
        output_data = inputs["detection_data"]
        self.process_output(input_data, output_data)
        return {}
