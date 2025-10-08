# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Class recording output that it got directly to mp4 or other video format.
"""

import threading
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from kenning.core.outputcollector import OutputCollector
from kenning.datasets.helpers.depth_estimation import render_depth
from kenning.utils.args_manager import get_parsed_json_dict

_FONT_SCALE = 1.5
_FONT_SIZE = 16
_PADDING = 8
_SIDE_PANEL_WIDTH = 512
_SCORE_COLUMN_WIDTH = 80


class VideoRecorder(OutputCollector, ABC):
    """
    Class for recording output from a runner.
    """

    setup_gui_lock = threading.Lock()

    arguments_structure = {
        "input_color_format": {
            "argparse_name": "--input-color-format",
            "description": "Color format of provided frame (RGB or BGR)",
            "enum": ["BGR", "RGB"],
            "default": "BGR",
        },
        "input_memory_layout": {
            "argparse_name": "--input-memory-layout",
            "description": "Memory layout of provided frame (NCHW or NHWC)",
            "enum": ["NHWC", "NCHW"],
            "default": "NHWC",
        },
        "output_file": {
            "argparse_name": "--output-file",
            "description": "File where output video will be saved",
            "type": Path,
            "default": "out.mp4",
        },
        "video_width": {
            "argparse_name": "--video-width",
            "description": "Width of image output",
            "type": int,
            "default": 416,
            "required": False,
        },
        "video_height": {
            "argparse_name": "--video-height",
            "description": "Height of image output",
            "type": int,
            "default": 416,
            "required": False,
        },
        "output_framerate": {
            "argparse_name": "--output-framerate",
            "description": "What should be the output framerate of the video",
            "type": int,
            "default": 30,
        },
        "video_codec": {
            "argparse_name": "--video-codec",
            "description": "What codec should be used when saving video",
            "type": str,
            "default": "mp4v",
        },
    }

    def __init__(
        self,
        input_color_format: str = "BGR",
        input_memory_layout: str = "NHWC",
        output_file: Path = "out.mp4",
        video_width: int = 416,
        video_height: int = 416,
        output_framerate: int = 30,
        video_codec: str = "mp4v",
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Base class for video recorder.

        Parameters
        ----------
        input_color_format : str
            Color format of provided frame (BGR or RGB).
        input_memory_layout : str
            Memory layout of provided frame (NCHW or NHWC).
        output_file: Path
            File to which output will be recorded.
        video_width : int
            Width of the video saved.
        video_height : int
            Height of the video saved.
        output_framerate : int
            Framerate at which video should be saved.
        video_codec : str
            Video codec used to save the video.
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        assert input_color_format in ["RGB", "BGR"]
        assert input_memory_layout in ["NHWC", "NCHW"]
        self.input_color_format = input_color_format
        self.input_memory_layout = input_memory_layout
        self.input_width = video_width
        self.input_height = video_height
        self.out_file = output_file
        self.video = cv2.VideoWriter(
            str(self.out_file),
            cv2.VideoWriter_fourcc(*video_codec),
            output_framerate,
            (video_width, video_height),
        )

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    @classmethod
    def _get_io_specification(
        cls, input_memory_layout: str
    ) -> Dict[str, List[Dict]]:
        """
        Creates runner IO specification from chosen parameters.

        Parameters
        ----------
        input_memory_layout : str
            Constructor argument.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output layers specification.
        """
        return {
            "input": [
                {
                    "name": "model_out",
                    "type": "Any",
                },
                {
                    "name": "frame_original",
                    "shape": (1, -1, -1, -1),
                    "dtype": "float32",
                },
            ],
            "output": [],
        }

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self.input_memory_layout)

    @classmethod
    def parse_io_specification_from_json(cls, json_dict):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        return cls._get_io_specification(
            parsed_json_dict["input_memory_layout"]
        )

    def cleanup(self):
        self.video.release()

    def detach_from_output(self):
        self.video.release()

    def should_close(self) -> bool:
        return False

    def get_output_data(self, inputs: Dict[str, Any]) -> Any:
        """
        Retrieves data specific to visualizer from inputs.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Visualized inputs.

        Returns
        -------
        Any
            Data specific to visualizer.
        """
        return inputs

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out = inputs["model_out"][0]
        frame = inputs["frame_original"][0]
        out = self.process_output(frame, out)
        self.video.write(np.array(out).astype("uint8"))
        return {}


class DepthVideoRecorder(VideoRecorder):
    """
    Class for recording Depth output from a runner.
    """

    def __init__(
        self,
        input_color_format: str = "BGR",
        input_memory_layout: str = "NHWC",
        output_file: Path = "out.mp4",
        video_width: int = 416,
        video_height: int = 416,
        output_framerate: int = 30,
        video_codec: str = "mp4v",
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Base class for video depth recorder.

        Parameters
        ----------
        input_color_format : str
            Color format of provided frame (BGR or RGB).
        input_memory_layout : str
            Memory layout of provided frame (NCHW or NHWC).
        output_file: Path
            File to which output will be recorded.
        video_width : int
            Width of the video saved.
        video_height : int
            Height of the video saved.
        output_framerate : int
            Framerate at which video should be saved.
        video_codec : str
            Video codec used to save the video.
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        super().__init__(
            input_color_format=input_color_format,
            input_memory_layout=input_memory_layout,
            output_file=output_file,
            video_width=video_width,
            video_height=video_height,
            output_framerate=output_framerate,
            video_codec=video_codec,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    @classmethod
    def _get_io_specification(
        cls, input_memory_layout: str
    ) -> Dict[str, List[Dict]]:
        """
        Creates runner IO specification from chosen parameters.

        Parameters
        ----------
        input_memory_layout : str
            Constructor argument.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output layers specification.
        """
        return {
            "input": [
                {
                    "name": "model_out",
                    "shape": [1, -1, -1],
                    "dtype": "float32",
                },
                {
                    "name": "frame_original",
                    "shape": (1, -1, -1, -1),
                    "dtype": "float32",
                },
            ],
            "output": [],
        }

    def process_output(
        self, input_data: List[np.ndarray], output_data: List[Any]
    ) -> np.ndarray:
        """
        Method used to prepare data for saving to file.

        Parameters
        ----------
        input_data : List[np.ndarray]
            List of input images.
        output_data : List[Any]
            List of data used for visualization.

        Returns
        -------
        np.ndarray
            Rendered Image.
        """
        return cv2.resize(
            render_depth(output_data),
            (self.input_width, self.input_height),
        )
