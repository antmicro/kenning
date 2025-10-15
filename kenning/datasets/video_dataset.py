# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with Dataset generating data from video and saving it.
"""

from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.exceptions import CannotDownloadDatasetError
from kenning.core.measurements import Measurements
from kenning.datasets.helpers.depth_estimation import render_depth


class VideoDataset(Dataset):
    """
    Creates a dataset of images in series extracted from video.

    Video is passed via dataset_root argument.

    Dataset saves result as a video output, does not perform any evaluation.
    """

    arguments_structure = {
        "output_video_file_path": {
            "argparse_name": "--output-video-file-path",
            "description": "Path to video that where output will be saved.",
            "type": Path,
            "required": False,
            "default": "out.mp4",
        },
        "input_memory_layout": {
            "argparse_name": "--input-memory-layout",
            "description": "Layout of captured frames. (NHWC or NCHW)",
            "required": False,
            "default": "NCHW",
            "enum": ["NHWC", "NCHW"],
        },
        "input_color_format": {
            "argparse_name": "--input-color-format",
            "description": "Color format of captured frames. (BGR or RGB)",
            "required": False,
            "default": "BGR",
            "enum": ["BGR", "RGB"],
        },
        "input_width": {
            "argparse_name": "--input-width",
            "description": "Width of image output.",
            "type": int,
            "default": 416,
            "required": False,
        },
        "input_height": {
            "argparse_name": "--input-height",
            "description": "Height of image output.",
            "type": int,
            "default": 416,
            "required": False,
        },
        "preprocess_type": {
            "argparse_name": "--preprocess_type",
            "description": "Determines the preprocessing type.",
            "default": "none",
            "enum": ["caffe", "tf", "torch", "none"],
        },
        "postprocess_type": {
            "argparse_name": "--postprocess-type",
            "description": "Determines visualization postprocessing.",
            "default": "none",
            "enum": ["depth", "none"],
        },
        "output_framerate": {
            "argparse_name": "--output-framerate",
            "description": "What should be the output framerate of the video."
            "By default the framerate of input video will be assumed.",
            "type": int,
            "default": -1,
        },
        "video_codec": {
            "argparse_name": "--video-codec",
            "description": "What codec should be used when saving video.",
            "type": str,
            "default": "mp4v",
        },
    }

    def __init__(
        self,
        root: Path,
        output_video_file_path: Path = "out.mp4",
        input_memory_layout: str = "NCHW",
        input_color_format: str = "BGR",
        input_width: int = 416,
        input_height: int = 416,
        preprocess_type: str = "none",
        postprocess_type: str = "none",
        output_framerate: int = -1,
        video_codec: str = "mp4v",
        batch_size: int = 1,
        download_dataset: bool = False,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 1.0,
        split_fraction_val: float = None,
        split_seed: int = 42,
        dataset_percentage: float = 1,
    ):
        assert input_memory_layout in ["NHWC", "NCHW"]
        assert preprocess_type in ["caffe", "torch", "tf", "none"]
        assert postprocess_type in ["depth", "none"]
        assert input_color_format in ["RGB", "BGR"]
        self.video_file_path = root
        self.output_video_file_path = output_video_file_path
        self.input_memory_layout = input_memory_layout
        self.input_color_format = input_color_format
        self.input_width = input_width
        self.input_height = input_height
        self.video_codec = video_codec
        self.output_framerate = output_framerate
        self.preprocess_type = preprocess_type
        self.postprocess_type = postprocess_type
        self.vidcap = None

        self.video = None

        self.raw_images = []
        self.processed_images = []
        self.cur_img_idx = 0

        super().__init__(
            root,
            batch_size,
            force_download_dataset,
            download_dataset,
            split_fraction_test=split_fraction_test,
            split_seed=split_seed,
            dataset_percentage=1,
            shuffle_data=False,
        )

    def download_dataset_fun(self):
        raise CannotDownloadDatasetError(
            "VideoDataset cannot be downloaded\n"
            "Please provide your own video passing it as dataset_root and\n"
            "specify Video format using video_codec"
        )

    def get_class_names(self):
        return None

    def get_input_mean_std(self):
        return (0.0, 1.0)

    def postprocess_frame(self, frame: np.ndarray):
        data = cv2.resize(frame, (self.input_width, self.input_height))
        npimg = None
        if self.input_color_format == "RGB":
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if self.input_memory_layout == "NCHW":
            img = np.transpose(data, (2, 0, 1))
            npimg = np.array(img, dtype=np.float32) / 255.0
        else:
            npimg = np.array(data, dtype=np.float32) / 255.0

        if self.preprocess_type == "caffe":
            # convert to BGR
            npimg = npimg[:, :, ::-1]
        if self.preprocess_type == "tf":
            npimg /= 127.5
            npimg -= 1.0
        elif self.preprocess_type == "torch":
            npimg /= 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            npimg = (npimg - mean) / std
        elif self.preprocess_type == "caffe":
            mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
            npimg -= mean
        return np.array([npimg])

    def prepare(self):
        if not self.video_file_path.is_file():
            raise FileNotFoundError
        self.vidcap = cv2.VideoCapture(str(self.video_file_path))
        frame_count = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        default_framerate = int(self.vidcap.get(cv2.CAP_PROP_FPS))
        # to disable any capture limits in older cv versions
        self.vidcap.set(cv2.CAP_PROP_FPS, float("inf"))

        self.video = cv2.VideoWriter(
            str(self.output_video_file_path),
            cv2.VideoWriter_fourcc(*self.video_codec),
            self.output_framerate
            if self.output_framerate != -1
            else default_framerate,
            (self.input_width, self.input_height),
        )
        self.dataX = range(frame_count)
        self.dataY = [0] * frame_count

    def prepare_input_samples(self, samples: List[int]) -> List[np.ndarray]:
        success, image = self.vidcap.read()
        if success:
            image = self.postprocess_frame(image)
            return [image]
        else:
            self.vidcap.release()

    def prepare_output_samples(self, samples: List[Any]) -> List[np.ndarray]:
        return [np.array(samples)]

    def __del__(self):
        if self.video:
            self.video.release()

    def evaluate(self, predictions, truth):
        frame = predictions[0]
        if self.postprocess_type == "depth":
            frame = render_depth(frame)
            frame = cv2.resize(frame, (self.input_width, self.input_height))

        self.video.write(np.array(frame).astype("uint8"))
        return Measurements()
