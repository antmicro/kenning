# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A DataProvider-derived class used to interface with a
camera, video file or a dummy video device
(consult ffmpeg and v4l2loopback for configuration for dummy video devices)
"""
from typing import Dict, Tuple, List
import cv2
import numpy as np
from pathlib import Path

from kenning.core.dataprovider import DataProvider
from kenning.utils.args_manager import get_parsed_json_dict


class CameraDataProvider(DataProvider):

    arguments_structure = {
        'video_file_path': {
            'argparse_name': '--video-file-path',
            'description': 'Path to the camera device',
            'type': Path,
            'required': True
        },
        'input_memory_layout': {
            'argparse_name': '--input-memory-layout',
            'description': 'Layout of captured frames (NHWC or NCHW)',
            'type': str,
            'required': False,
            'default': 'NCHW'
        },
        'input_color_format': {
            'argparse_name': '--input-color-format',
            'description': 'Color format of captured frames (BGR or RGB)',
            'type': str,
            'required': False,
            'default': 'BGR'
        },
        'input_width': {
            'argparse_name': '--input-width',
            'description': 'Width of captured frame',
            'type': int,
            'required': False
        },
        'input_height': {
            'argparse_name': '--input-height',
            'description': 'Height of captured frame',
            'type': int,
            'required': False
        }
    }

    def __init__(
            self,
            video_file_path: Path,
            input_memory_layout: str = 'NCHW',
            input_color_format: str = 'BGR',
            input_width: int = 416,
            input_height: int = 416,
            inputs_sources: Dict[str, Tuple[int, str]] = {},
            inputs_specs: Dict[str, Dict] = {},
            outputs: Dict[str, str] = {}):
        """
        Creates the camera data provider.

        Parameters
        ----------
        video_file_path: Path
            Path to the video file
        input_memory_layout: str
            Layout of the frame memory: NCHW or NHWC
        input_color_format: str
            Color format of captured frames: RGB or BGR
        input_width: int
            Width of the frame
        input_height: int
            Height of the frame
        inputs_sources: Dict[str, Tuple[int, str]]
            Input from where data is being retrieved
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs
        outputs: Dict[str, str]
            Outputs of this Runner
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
            outputs=outputs
        )

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.video_file_path,
            args.input_memory_layout,
            args.input_color_format,
            args.input_width,
            args.input_height
        )

    @classmethod
    def from_json(
            cls,
            json_dict: Dict,
            inputs_sources: Dict[str, Tuple[int, str]] = {},
            inputs_specs: Dict[str, Dict] = {},
            outputs: Dict[str, str] = {}):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            **parsed_json_dict,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs
        )

    def prepare(self):
        self.device = cv2.VideoCapture(self.device_id)

    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        data = cv2.resize(data, (self.input_width, self.input_height))
        if self.input_color_format == 'RGB':
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if self.input_memory_layout == 'NCHW':
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

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        if self.input_memory_layout == 'NCHW':
            frame_shape = (1, 3, self.input_height, self.input_width)
        else:
            frame_shape = (1, self.input_height, self.input_width, 3)
        return {
            'input': [],
            'output': [{
                'name': 'frame',
                'shape': frame_shape,
                'dtype': 'float32'
            }]
        }

    def run(
            self,
            inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        frame_global_name = None
        for local_name, global_name in self.outputs.items():
            if local_name == 'frame':
                frame_global_name = global_name

        frame = self.fetch_input()
        frame = self.preprocess_input(frame)
        return {frame_global_name: np.expand_dims(frame, 0)}


class VideoCaptureDeviceException(Exception):
    """
    Exception to be raised when VideoCaptureDevice malfunctions
    during frame capture
    """
    def __init__(self, device_id, message="Video device {} read error"):
        super().__init__(message.format(device_id))
