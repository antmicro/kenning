# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A DataProvider-derived class used to interface with a
camera, video file or a dummy video device
(consult ffmpeg and v4l2loopback for configuration for dummy video devices).
"""
from typing import Dict, Tuple, List
import cv2
import numpy as np
from pathlib import Path

from kenning.core.dataprovider import DataProvider
from kenning.utils.args_manager import get_parsed_json_dict


class CameraDataProvider(DataProvider):
    """
    Reads frames from the camera and passes them to Kenning nodes.
    """

    arguments_structure = {
        "video_file_path": {
            "argparse_name": "--video-file-path",
            "description": "Path to the camera device",
            "type": Path,
            "required": True,
        },
        "input_memory_layout": {
            "argparse_name": "--input-memory-layout",
            "description": "Layout of captured frames (NHWC or NCHW)",
            "type": str,
            "required": False,
            "default": "NCHW",
        },
        "input_color_format": {
            "argparse_name": "--input-color-format",
            "description": "Color format of captured frames (BGR or RGB)",
            "type": str,
            "required": False,
            "default": "BGR",
        },
        "input_width": {
            "argparse_name": "--input-width",
            "description": "Width of captured frame",
            "type": int,
            "required": False,
        },
        "input_height": {
            "argparse_name": "--input-height",
            "description": "Height of captured frame",
            "type": int,
            "required": False,
        },
    }

    def __init__(
        self,
        video_file_path: Path,
        input_memory_layout: str = "NCHW",
        input_color_format: str = "BGR",
        input_width: int = 416,
        input_height: int = 416,
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Creates the camera data provider.

        Parameters
        ----------
        video_file_path : Path
            Path to the video file.
        input_memory_layout : str
            Layout of the frame memory: NCHW or NHWC.
        input_color_format : str
            Color format of captured frames: RGB or BGR.
        input_width : int
            Width of the frame.
        input_height : int
            Height of the frame.
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        self.device_id = str(video_file_path)

        self.video_file_path = video_file_path
        self.input_memory_layout = input_memory_layout
        self.input_color_format = input_color_format
        self.input_width = input_width
        self.input_height = input_height

        self.device = None

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    def prepare(self):
        self.device = cv2.VideoCapture(self.device_id)

    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        data = cv2.resize(data, (self.input_width, self.input_height))
        if self.input_color_format == "RGB":
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if self.input_memory_layout == "NCHW":
            img = np.transpose(data, (2, 0, 1))
            return np.array(img, dtype=np.float32) / 255.0
        else:
            return np.array(data, dtype=np.float32) / 255.0

    def fetch_input(self):
        ret, frame = self.device.read()
        if ret:
            return frame
        else:
            raise VideoCaptureDeviceException(self.device_id)

    def detach_from_source(self):
        if self.device:
            self.device.release()

    @classmethod
    def _get_io_specification(
        cls, input_memory_layout, input_width, input_height, outputs={}
    ):
        """
        Creates runner IO specification from chosen parameters.

        Parameters
        ----------
        input_memory_layout : str
            Layout of the frame memory: NCHW or NHWC.
        input_width : int
            Width of the frame.
        input_height : int
            Height of the frame.
        outputs : Dict[str, str]
            Outputs of this Runner. If empty, the output name
            is set to 'frame'.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output layers specification.
        """
        if input_memory_layout == "NCHW":
            frame_shape = (1, 3, input_height, input_width)
        else:
            frame_shape = (1, input_height, input_width, 3)
        output_name = "frame"
        if outputs:
            output_name = list(outputs.keys())[0]
        return {
            "input": [],
            "output": [
                {"name": output_name, "shape": frame_shape, "dtype": "float32"}
            ],
        }

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(
            self.input_memory_layout,
            self.input_height,
            self.input_width,
            self.outputs,
        )

    @classmethod
    def parse_io_specification_from_json(cls, json_dict):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        return cls._get_io_specification(
            parsed_json_dict["input_memory_layout"],
            parsed_json_dict["input_width"],
            parsed_json_dict["input_height"],
        )

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        frame = self.fetch_input()
        frame = self.preprocess_input(frame)
        output_name = "frame"
        if self.outputs:
            output_name = list(self.outputs.keys())[0]
        return {output_name: np.expand_dims(frame, 0)}


class VideoCaptureDeviceException(Exception):
    """
    Exception to be raised when VideoCaptureDevice malfunctions
    during frame capture.
    """

    def __init__(self, device_id, message="Video device {} read error"):
        super().__init__(message.format(device_id))
