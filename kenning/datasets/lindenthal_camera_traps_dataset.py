# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The Lindenthal Camera Traps wrapper.
"""

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple
from zipfile import ZipFile

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from kenning.core.measurements import Measurements
from kenning.datasets.helpers.detection_and_segmentation import (
    SegmObject,
)
from kenning.datasets.helpers.video_detection_and_segmentation import (
    VideoObjectDetectionSegmentationDataset,
)
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import ResourceManager, Resources

try:
    import rosbags
except ImportError:
    rosbags = None


class LindenthalCameraTrapsDataset(VideoObjectDetectionSegmentationDataset):
    """
    The Lindenthal Camera Traps dataset.

    https://lila.science/datasets/lindenthal-camera-traps/

    It's a dataset of 775 video sequences with duration from 15 to 45 seconds
    at 15 frames per second.
    Videos are captured in the wildlife park of Lindenthal.

    Dataset is annotated with:

    * image-level labels,
    * object bounding boxes,
    * object segmentation masks

    *License*: Community Data License Agreement (permissive variant).

    *Page*: `Lindenthal Camera Traps site
    <https://lila.science/datasets/lindenthal-camera-traps/>`_.
    """

    resources = Resources(
        {
            "images": "https://storage.googleapis.com/public-datasets-lila/lindenthal-camera-traps/lindenthal-camera-traps.zip",  # noqa: E501
        }
    )

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = False,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        task: str = "instance_segmentation",
        image_memory_layout: str = "NHWC",
        show_on_eval: bool = False,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        self.num_classes = 4
        super().__init__(
            root=root,
            batch_size=batch_size,
            download_dataset=download_dataset,
            force_download_dataset=force_download_dataset,
            external_calibration_dataset=external_calibration_dataset,
            split_fraction_test=split_fraction_test,
            split_fraction_val=split_fraction_val,
            split_seed=split_seed,
            task=task,
            image_memory_layout=image_memory_layout,
            show_on_eval=show_on_eval,
            image_width=image_width,
            image_height=image_height,
        )

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        topics = [
            "/device_0/sensor_0/Infrared_1/image/data",
            "/device_0/sensor_1/Color_0/image/data",
        ]

        # Set max cache size to 220 GB to allow dataset download
        ResourceManager().set_max_cache_size(220 * 10**9)

        with LoggerProgressBar() as logger_progress_bar, ZipFile(
            self.resources["images"],
            "r",
        ) as zip:
            iterable = tuple(
                filter(lambda x: x.endswith(".bag"), zip.namelist())
            )
            for f in tqdm(
                iterable=iterable,
                total=len(iterable),
                file=logger_progress_bar,
            ):
                zip.extract(member=f, path=self.root)
                self.convert_bag_to_images(
                    self.root / f, self.root / "images", topics
                )
                (self.root / f).unlink()
        shutil.rmtree((self.root / "lindenthal-camera-traps"))

    def prepare(self):
        annotationspath = self.root / "annotations.json"
        self.coco = COCO(annotationspath)

        self.classmap = {}
        self.classnames = []
        for classid in self.coco.cats.keys():
            self.classmap[classid] = self.coco.cats[classid]["name"]
            self.classnames.append(self.coco.cats[classid]["name"])

        coco_keys = list(self.coco.imgs.keys())
        sequences = defaultdict(list)
        keystoimgs = dict()
        self.imgstokeys = dict()

        for key, imgdata in zip(coco_keys, self.coco.loadImgs(coco_keys)):
            filepath = str(self.root / "images" / imgdata["file_name"])
            self.imgstokeys[filepath] = key
            keystoimgs[key] = filepath
            sequences[imgdata["seq_id"]].append(filepath)
        self.dataX = list(sequences.values())
        self.dataY = []

        annotations = defaultdict(list)
        for annkey, anndata in self.coco.anns.items():
            bbox = anndata["bbox"]
            width = self.coco.imgs[anndata["image_id"]]["width"]
            height = self.coco.imgs[anndata["image_id"]]["height"]
            annotations[keystoimgs[anndata["image_id"]]].append(
                SegmObject(
                    clsname=self.classmap[anndata["category_id"]],
                    maskpath=None,
                    xmin=bbox[0] / width,
                    ymin=bbox[1] / height,
                    xmax=(bbox[0] + bbox[2]) / width,
                    ymax=(bbox[1] + bbox[3]) / height,
                    mask=self.coco.annToMask(anndata) * 255,
                    score=1.0,
                    iscrowd=anndata["iscrowd"] == 1,
                )
            )

        for i, sequence in enumerate(self.dataX):
            self.dataY.append(
                [annotations[imgpath] for imgpath in self.dataX[i]]
            )

    def prepare_input_samples(
        self, samples: List[List[Path]]
    ) -> List[List[np.ndarray]]:
        def prepare_image(imgpath: Path) -> np.ndarray:
            """
            Loads and preprocesses the image.

            Parameters
            ----------
            imgpath : Path
                Path to the image file

            Returns
            -------
            np.ndarray
                Preprocessed image.
            """
            img = cv2.imread(str(imgpath))

            if all([self.image_width, self.image_height]):
                img = cv2.resize(img, (self.image_width, self.image_height))
            elif any([self.image_width, self.image_height]):
                KLogger.warning(
                    "Only one of image_width and image_height is set. "
                    "The image will not be resized."
                )

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            npimg = np.array(img, dtype=np.float32) / 255.0
            if self.image_memory_layout == "NCHW":
                npimg = np.transpose(npimg, (2, 0, 1))
            return npimg

        result = [
            [prepare_image(imgpath) for imgpath in sequence]
            for sequence in samples
        ]
        return result

    def evaluate(self, predictions, truth):
        measurements = super().evaluate(predictions, truth)
        currindex = self._dataindex - len(predictions)
        for sequence, groundtruth in zip(predictions, truth):
            seq_measurements = Measurements()
            for idx, frame in enumerate(sequence):
                cocoid = self.imgstokeys[
                    self.dataX[self._dataindices[currindex]][idx]
                ]
                width = self.coco.imgs[cocoid]["width"]
                height = self.coco.imgs[cocoid]["height"]
                for pred in frame:
                    xmin = max(min(pred.xmin * width, width), 0)
                    xmax = max(min(pred.xmax * width, width), 0)
                    ymin = max(min(pred.ymin * height, height), 0)
                    ymax = max(min(pred.ymax * height, height), 0)
                    w = xmax - xmin
                    h = ymax - ymin
                    image_name = "/".join(
                        self.dataX[self._dataindices[currindex]][idx].split(
                            "/"
                        )[-2:]
                    )
                    seq_measurements.add_measurement(
                        "predictions",
                        [
                            {
                                "image_name": image_name,
                                "category": pred.clsname,
                                "bbox": [xmin, ymin, w, h],
                                "score": pred.score,
                            }
                        ],
                    )
            # TODO: Add sequence-level metrics (e.g. mAP, IoU, etc.)
            currindex += 1
            measurements += seq_measurements
        return measurements

    def get_class_names(self):
        return self.classnames

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    @staticmethod
    def convert_bag_to_images(bagfile: Path, output: Path, topics: List[str]):
        """
        Convert `sensor_msgs/Image` messages from ROS1 bag file to images.
        Topics to save images from is specified in the `topics` parameter.

        Images are located in the directory named after stem of the bag file
        under the `output` directory.

        Images are named with 6-digit numbers starting from 000001 representing
        the order of the images in the bag file.

        Parameters
        ----------
        bagfile : Path
            Path to the bag file.
        output : Path
            Path to the output directory.
        topics : List[str]
            List of topics to convert to images.

        Raises
        ------
        ImportError
            If `rosbags` package is not installed.
        """
        if rosbags is None:
            error_message = (
                "rosbags package is not installed. "
                "Please install it with `pip install kenning[ros2]`"
            )
            KLogger.critical(error_message)
            raise ImportError(error_message)
        counter = 1
        rosbag_dir = output / bagfile.stem
        rosbag_dir.mkdir(parents=True, exist_ok=True)
        with rosbags.rosbag1.Reader(bagfile) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic in topics:
                    msg = rosbags.serde.deserialize_cdr(
                        rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype),
                        connection.msgtype,
                    )
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, -1
                    )
                    if img.shape[2] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(
                        str(rosbag_dir / f"{str(counter).zfill(6)}.png"), img
                    )
                    counter += 1
