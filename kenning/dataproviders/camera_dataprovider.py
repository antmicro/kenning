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
            'argparse_name': '--video_file_path',
            'description': 'Path to the camera device',
            'type': Path,
            'required': True
        },
        'image_memory_layout': {
            'argparse_name': '--image_memory_layout',
            'description': 'Layout of capture frames (NHWC or NCHW)',
            'type': str,
            'required': False
        },
        'image_width': {
            'argparse_name': '--image_width',
            'description': 'Width of captured frame',
            'type': int,
            'required': False
        },
        'image_height': {
            'argparse_name': '--image_height',
            'description': 'Height of captured frame',
            'type': int,
            'required': False
        }
    }

    def __init__(
            self,
            video_file_path: Path,
            image_memory_layout: str = "NCHW",
            image_width: int = 416,
            image_height: int = 416,
            inputs_sources: Dict[str, Tuple[int, str]] = {},
            outputs: Dict[str, str] = {}):
        """
        Creates the camera data provider.

        Parameters
        ----------
        video_file_path: Path
            Path to the video file
        image_memory_layout: str
            Layout of the frame memory: NCHW or NHWC
        image_width: int
            Width of the frame
        image_height: int
            Height of the frame
        inputs_sources: Dict[str, Tuple[int, str]]
            Input from where data is being retrieved
        outputs: Dict[str, str]
            Outputs of this Runner
        """

        self.device_id = str(video_file_path)

        self.video_file_path = video_file_path
        self.image_memory_layout = image_memory_layout
        self.image_width = image_width
        self.image_height = image_height

        self.device = None

        super().__init__(
            inputs_sources=inputs_sources,
            outputs=outputs)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--video-file-path',
            help='Video file path (for cameras, use /dev/videoX where X is the device ID eg. /dev/video0)',  # noqa: E501
            type=Path,
            required=True
        )
        group.add_argument(
            '--image-memory-layout',
            help='Determines if images should be delivered in NHWC or NCHW format',  # noqa: E501
            choices=['NHWC', 'NCHW'],
            default='NCHW'
        )
        group.add_argument(
            '--image-width',
            help='Determines the width of the image for the model',
            type=int,
            default=416
        )
        group.add_argument(
            '--image-height',
            help='Determines the height of the image for the model',
            type=int,
            default=416
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.video_file_path,
            args.image_memory_layout,
            args.image_width,
            args.image_height
        )

    @classmethod
    def from_json(
            cls,
            json_dict: Dict,
            inputs_sources: Dict[str, Tuple[int, str]] = None,
            outputs: Dict[str, str] = None):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            **parsed_json_dict,
            inputs_sources=inputs_sources,
            outputs=outputs)

    def prepare(self):
        self.device = cv2.VideoCapture(self.device_id)

    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        data = cv2.resize(
            data,
            (self.image_width, self.image_height)
        )
        if self.image_memory_layout == "NCHW":
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
        if self.image_memory_layout == 'NCHW':
            frame_shape = (1, 3, self.image_height, self.image_width)
        else:
            frame_shape = (1, self.image_height, self.image_width, 3)
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
