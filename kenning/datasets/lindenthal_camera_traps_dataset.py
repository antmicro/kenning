# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The Lindenthal Camera Traps wrapper.
"""

import random
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from kenning.core.measurements import Measurements
from kenning.datasets.helpers.detection_and_segmentation import (
    ObjectDetectionSegmentationDataset,
    SegmObject,
)
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import (
    ResourceManager,
    Resources,
    ResourceURI,
    extract_zip,
)

try:
    import rosbags
except ImportError:
    rosbags = None


class LindenthalCameraTrapsAugmentation:
    """
    The Lindenthal Camera Traps data augmentation class.

    It's a simple class that provides augmentation methods for
    Lindenthal Camera Traps dataset.
    """

    def __init__(self):
        """
        Initializes the augmentation class.
        """
        # Replay transform used to apply the same transform to sequence
        # of images
        self.transform = None

    def prepare(self):
        """
        Prepares the augmentation transforms.

        Raises
        ------
        ImportError
            If the albumentations package is not installed.
        """
        try:
            import albumentations as A
        except ImportError:
            raise ImportError(
                "The albumentations package is required for "
                "augmentation transforms. Please install it using "
                "`pip install kenning[albumentations]`."
            )
        self.transform = A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.RandomCrop(width=512, height=480),
                        A.RandomCrop(width=256, height=256),
                        A.RandomCrop(width=420, height=240),
                        A.RandomCrop(width=640, height=360),
                        A.RandomCrop(width=590, height=259),
                        A.RandomCrop(width=384, height=186),
                    ],
                    p=0.5,
                ),
                A.RandomFog(fog_coef_lower=0, fog_coef_upper=0.3, p=0.5),
                A.RandomBrightnessContrast(brightness_by_max=False, p=0.5),
                A.Sharpen(p=0.5),
                A.ChannelShuffle(p=0.5),
                A.Downscale(scale_min=0.45, scale_max=0.85, p=0.5),
                A.RandomScale(scale_limit=[-0.2, 0.3], p=0.5),
                A.OneOf(
                    [
                        A.PixelDropout(dropout_prob=0.02),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.2)),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Rotate(limit=[-20, 20], border_mode=0),
                        A.ElasticTransform(
                            alpha=0.5,
                            sigma=20,
                            alpha_affine=12,
                            interpolation=1,
                            border_mode=0,
                            mask_value=0,
                            value=0,
                            approximate=True,
                        ),
                    ],
                    p=0.5,
                ),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["clsname", "maskpath", "score", "iscrowd"],
                min_visibility=0.2,
            ),
        )

    @staticmethod
    def get_dummy_data(height: int, width: int) -> Dict[str, List]:
        """
        Returns dummy data for replay transform.

        Parameters
        ----------
        height : int
            Height of the image.
        width : int
            Width of the image.

        Returns
        -------
        Dict[str, List]
            Dictionary of dummy data. Not includes image.
        """
        return {
            "image": np.zeros((height, width, 3), dtype=np.float32),
            "masks": [np.zeros((height, width), dtype=np.uint8)],
            "bboxes": [[0, 0, 100, 100]],
            "clsname": ["dummy"],
            "maskpath": ["dummy"],
            "score": [0.0],
            "iscrowd": [0],
        }

    def __call__(
        self,
        X: List[List[np.ndarray]],
        y: List[List[List[SegmObject]]],
        seed: Optional[int] = None,
    ) -> Tuple[List[List[np.ndarray]], List[List[List[SegmObject]]]]:
        """
        Augment the input sequences with random transformations.

        Parameters
        ----------
        X : List[List[np.ndarray]]
            List of preprocessed input sequences.
        y : List[List[List[SegmObject]]]
            List of ground truth sequences.
        seed : Optional[int]
            Random seed, by default None

        Returns
        -------
        Tuple[List[List[np.ndarray]], List[List[List[SegmObject]]]]
            Tuple of augmented input sequences and ground truth sequences.
        """
        random.seed(seed)

        # Iterate over all sequences
        for i, (video, labels) in enumerate(zip(X, y)):
            # Save replay data for the whole sequence
            height, width = video[0].shape[:2]
            replay_data = self.transform(**self.get_dummy_data(height, width))[
                "replay"
            ]

            # Iterate over all frames in the sequence
            for idx, (frame, gt) in enumerate(zip(video, labels)):
                is_dummy, frame_gt = self.extract_ground_truth(
                    gt, height, width
                )
                augmented = self.transform.replay(
                    replay_data,
                    image=frame,
                    **frame_gt,
                )
                video[idx] = augmented["image"]
                segmentations = []
                if not is_dummy:
                    segmentations = self.pack_augmentations(augmented)
                labels[idx] = segmentations
            X[i] = video
            y[i] = labels
        return X, y

    @staticmethod
    def extract_ground_truth(
        ground_truth: List[SegmObject], height: int, width: int
    ) -> Tuple[bool, Dict[str, List]]:
        """
        Extract ground truth values from the list of `SegmObject`s.

        Parameters
        ----------
        ground_truth : List[SegmObject]
            List of `SegmObject`s.
        height : int
            Image height.
        width : int
            Image width.

        Returns
        -------
        Tuple[bool, Dict[str, List]]
            Tuple of data from list of `SegmObject` and `bool` flag indicating
            whether the ground truth is empty.
            If yes, then the returned dictionary is filled with dummy data.
            Does not include image.
        """
        ret = {
            "masks": [],
            "bboxes": [],
            "clsname": [],
            "maskpath": [],
            "score": [],
            "iscrowd": [],
        }
        for obj in ground_truth:
            ret["clsname"].append(obj.clsname)
            ret["maskpath"].append(obj.maskpath)
            xmin = obj.xmin * width
            ymin = obj.ymin * height
            xmax = obj.xmax * width
            ymax = obj.ymax * height
            ret["bboxes"].append([xmin, ymin, xmax - xmin, ymax - ymin])
            ret["masks"].append(obj.mask)
            ret["score"].append(obj.score)
            ret["iscrowd"].append(obj.iscrowd)
        if len(ret["clsname"]) == 0:
            ret = LindenthalCameraTrapsAugmentation.get_dummy_data(
                height, width
            )
            ret.pop("image")
            return True, ret
        return False, ret

    @staticmethod
    def pack_augmentations(augmented_data: Dict[str, Any]) -> List[SegmObject]:
        """
        Pack augmented data into list of `SegmObject`s.

        Parameters
        ----------
        augmented_data : Dict[str, Any]
            Augmented data.

        Returns
        -------
        List[SegmObject]
            List of `SegmObject`s.
        """
        segmentations = []
        aug_height, aug_width = augmented_data["image"].shape[:2]
        for clsname, mask, bbox, maskpath, score, iscrowd in zip(
            augmented_data["clsname"],
            augmented_data["masks"],
            augmented_data["bboxes"],
            augmented_data["maskpath"],
            augmented_data["score"],
            augmented_data["iscrowd"],
        ):
            xmin, ymin, xmax, ymax = (
                bbox[0] / aug_width,
                bbox[1] / aug_height,
                (bbox[0] + bbox[2]) / aug_width,
                (bbox[1] + bbox[3]) / aug_height,
            )
            segmentations.append(
                SegmObject(
                    clsname=clsname,
                    mask=mask,
                    maskpath=maskpath,
                    score=score,
                    iscrowd=iscrowd,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                ),
            )
        return segmentations


class LindenthalCameraTrapsDataset(ObjectDetectionSegmentationDataset):
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
            "images": "https://storage.googleapis.com/public-datasets-lila/lindenthal-camera-traps/lindenthal-camera-traps.zip",
        }
    )

    demo_dataset_uri = ResourceURI(
        "kenning:///datasets/lindenthal-dataset-sample.zip"
    )

    arguments_structure = {
        "image_width": {
            "description": "Width of the input images",
            "type": int,
            "default": 0,
        },
        "image_height": {
            "description": "Height of the input images",
            "type": int,
            "default": 0,
        },
        "use_demonstration_dataset": {
            "description": "If set to True, then instead of downloading a full Lindenthal dataset, a smaller, demonstration dataset is downloaded",  # noqa: E501
            "type": bool,
            "default": False,
        },
    }

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
        image_width: int = 0,
        image_height: int = 0,
        min_iou: float = 0.5,
        max_preds: int = 100,
        augment: bool = False,
        use_demonstration_dataset: bool = False,
    ):
        self.num_classes = 4
        self.augment = augment
        self.use_demonstration_dataset = use_demonstration_dataset

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
            min_iou=min_iou,
            max_preds=max_preds,
        )

    def download_original_dataset(self):
        """
        Downloads the Lindenthal dataset.

        The dataset consists of ROS 1 rosbag files that contain data
        from various sensors - camera, depth camera, infrared camera, ...

        This method downloads the dataset, reads rosbag files, extracts
        the color and infrared images from it, and loads annotations for it.

        NOTE: At download, it reaches a size of around 220 GB.
        The downloaded file is later removed, but it is recommended
        to prepare a storage or configure a different one with
        `KENNING_CACHE_DIR`
        """
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
                **logger_progress_bar.kwargs,
            ):
                zip.extract(member=f, path=self.root)
                self.convert_bag_to_images(
                    self.root / f, self.root / "images", topics
                )
                (self.root / f).unlink()
        shutil.rmtree((self.root / "lindenthal-camera-traps"))

    def download_demonstration_dataset(self):
        """
        Downloads sample sequences from Lindenthal dataset.

        The sample sequences are of size around 1GB, providing
        ready-to-use images and annotations (without the need
        for ROS 1 rosbag unpacking).

        It is mostly for testing/demonstration purposes, it
        is not recommended for training due to size.

        It should be used with `split_fraction_test` set to 1.0.
        """
        extract_zip(self.root, self.demo_dataset_uri)

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.use_demonstration_dataset:
            self.download_demonstration_dataset()
        else:
            self.download_original_dataset()

    def __next__(self):
        X, y = super().__next__()
        if self.augment:
            X, y = self.augmentation(X, y)
        return X, y

    def prepare(self):
        if self.augment:
            self.augmentation = LindenthalCameraTrapsAugmentation()
            self.augmentation.prepare()

        annotationspath = self.root / "annotations.json"
        self.coco = COCO(annotationspath)
        coco_keys = list(self.coco.imgs.keys())

        # Load classes
        self.classmap = {}
        self.classnames = []
        self.imgstokeys = dict()
        for classid in self.coco.cats.keys():
            self.classmap[classid] = self.coco.cats[classid]["name"]
            self.classnames.append(self.coco.cats[classid]["name"])

        keystoimgs = dict()
        filepaths = []

        annotations = defaultdict(list)
        # Annotations classes per recording
        classes_by_recording = defaultdict(list)
        # Image idx per recording
        recording_to_idx = defaultdict(list)

        # Retrieve images
        for idx, (key, imgdata) in enumerate(
            zip(coco_keys, self.coco.loadImgs(coco_keys))
        ):
            filepath = str(self.root / "images" / imgdata["file_name"])
            self.imgstokeys[filepath] = key
            keystoimgs[key] = filepath
            filepaths.append(Path(filepath))
            recording_to_idx[imgdata["seq_id"]].append(idx)

        # Retrieve annotations
        for annkey, anndata in self.coco.anns.items():
            bbox = anndata["bbox"]
            width = self.coco.imgs[anndata["image_id"]]["width"]
            height = self.coco.imgs[anndata["image_id"]]["height"]
            classes_by_recording[
                self.coco.imgs[anndata["image_id"]]["seq_id"]
            ].append(self.classmap[anndata["category_id"]])
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

        # Get the majority class for each recording
        for key, value in classes_by_recording.items():
            classes_by_recording[key] = {
                "majority": max(set(value), key=value.count),
                "ids": recording_to_idx[key],
            }
        self.recording_to_idx = [
            x["ids"] for x in classes_by_recording.values()
        ]
        self.stratify_arg = [
            x["majority"] for x in classes_by_recording.values()
        ]

        self.dataX = [
            filepaths[idx]
            for recording in self.recording_to_idx
            for idx in recording
        ]
        self.dataY = [annotations[str(imgpath)] for imgpath in self.dataX]

    def prepare_input_samples(
        self, samples: List[Path]
    ) -> List[Dict[str, Any]]:
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
            {
                "data": prepare_image(imgpath),
                "video_id": imgpath.parent.name,
                "frame_id": int(imgpath.stem),
            }
            for imgpath in samples
        ]
        return result

    def train_test_split_representations(
        self,
        test_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        stratify: bool = True,
        append_index: bool = False,
    ) -> Tuple[List, ...]:
        from sklearn.model_selection import train_test_split

        if test_fraction is None:
            test_fraction = self.split_fraction_test
        if val_fraction is None:
            val_fraction = self.split_fraction_val
        if seed is None:
            seed = self.split_seed

        indices = list(range(len(self.recording_to_idx)))

        np.random.seed(seed)
        np.random.shuffle(indices)

        if not val_fraction:
            # All data is for testing
            if test_fraction == 1.0:
                indices = [
                    unroll
                    for idx in indices
                    for unroll in self.recording_to_idx[idx]
                ]
                testX = [self.dataX[idx] for idx in indices]
                testY = [self.dataY[idx] for idx in indices]
                ret = ([], testX, [], testY)
                if append_index:
                    ret += ([], indices)
                return ret

            # All data is for training
            elif test_fraction == 0.0:
                indices = [
                    unroll
                    for idx in indices
                    for unroll in self.recording_to_idx[idx]
                ]
                trainX = [self.dataX[idx] for idx in indices]
                trainY = [self.dataY[idx] for idx in indices]
                ret = (trainX, [], trainY, [])
                if append_index:
                    ret += (indices, [])
                return ret

            # Assert test_fraction is less than 1.0
            assert test_fraction < 1.0
        else:
            # Assert test_fraction + val_fraction is less than 1.0
            assert test_fraction + val_fraction < 1.0

        if stratify:
            stratify_arg = self.stratify_arg
        else:
            stratify_arg = None

        dataItrain, dataItest = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify_arg,
        )

        dataIval = None
        if val_fraction is not None and val_fraction != 0:
            if stratify:
                stratify_arg = [self.stratify_arg[idx] for idx in dataItrain]
            else:
                stratify_arg = None
            dataItrain, dataIval = train_test_split(
                dataItrain,
                test_size=val_fraction / (1 - test_fraction),
                random_state=seed,
                shuffle=True,
                stratify=stratify_arg,
            )

        dataItrain = [
            unroll
            for idx in dataItrain
            for unroll in self.recording_to_idx[idx]
        ]
        self.dataXtrain = [self.dataX[idx] for idx in dataItrain]
        self.dataYtrain = [self.dataY[idx] for idx in dataItrain]
        dataItest = [
            unroll
            for idx in dataItest
            for unroll in self.recording_to_idx[idx]
        ]
        self.dataXtest = [self.dataX[idx] for idx in dataItest]
        self.dataYtest = [self.dataY[idx] for idx in dataItest]
        indices = [dataItrain, dataItest]
        ret = (
            self.dataXtrain,
            self.dataXtest,
            self.dataYtrain,
            self.dataYtest,
        )
        if dataIval is not None:
            dataIval = [
                unroll
                for idx in dataIval
                for unroll in self.recording_to_idx[idx]
            ]
            self.dataXval = [self.dataX[idx] for idx in dataIval]
            self.dataYval = [self.dataY[idx] for idx in dataIval]
            indices.append(dataIval)
            ret += (self.dataXval, self.dataYval)

        if append_index:
            ret = ret + (*indices,)
        return ret

    def evaluate(self, predictions, truth, **kwargs):
        measurements = Measurements()
        self._dataindex -= len(truth) - 1
        for preds, groundtruths in zip(predictions, truth):
            frame_measurements = super().evaluate(
                [preds], [groundtruths], **kwargs
            )

            current_frame_idx = self._dataindices[self._dataindex - 1]
            recording_name = self.get_recording_name(current_frame_idx)

            self._dataindex += 1

            # Accamulate measurements per video
            video_measurements = Measurements()
            video_measurements.add_measurement(
                f"eval_video/{recording_name}",
                [
                    {
                        **deepcopy(frame_measurements.data),
                    }
                ],
                lambda: list(),
            )

            measurements += frame_measurements
            measurements += video_measurements
        return measurements

    def show_eval_images(self, predictions, truth):
        KLogger.debug(f"\ntruth\n{truth}")
        KLogger.debug(f"\npredictions\n{predictions}")
        for idx, (pred, gt) in enumerate(zip(predictions, truth)):
            img_idx = self._dataindices[self._dataindex - len(truth) + idx]
            img = self.prepare_input_samples([self.dataX[img_idx]])[0]["data"]
            if self.image_memory_layout == "NCHW":
                img = img.transpose(1, 2, 0)
            int_img = np.multiply(img, 255).astype("uint8")
            int_img = cv2.cvtColor(int_img, cv2.COLOR_BGR2GRAY)
            int_img = cv2.cvtColor(int_img, cv2.COLOR_GRAY2RGB)
            int_img = self.apply_predictions(int_img, pred, gt)

            cv2.imshow("evaluatedimage", int_img)
            while True:
                c = cv2.waitKey(100)
                is_alive = cv2.getWindowProperty(
                    "evaluated image", cv2.WND_PROP_VISIBLE
                )
                if is_alive < 1 or c != -1:
                    break

    def get_class_names(self):
        return self.classnames

    def get_recording_name(self, frame_index: int) -> str:
        """
        Returns the name of the recording that the frame belongs to.

        Parameters
        ----------
        frame_index : int
            The index of the frame.

        Returns
        -------
        str
            The name of the recording.
        """
        first_frame_key = self.imgstokeys[str(self.dataX[frame_index])]
        return self.coco.imgs[first_frame_key]["seq_id"]

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

    @staticmethod
    def form_subsequences(
        sequence: List[Any],
        policy: str = "subsequence",
        num_segments: int = 1,
        window_size: int = 1,
    ) -> List[List[Any]]:
        """
        Split a given sequence into subsequences according to defined `policy`.

        Parameters
        ----------
        sequence : List[Any]
            List of frames making up the sequence.
        policy : str
            The policy for splitting the sequence. Can be one of:
            * subsequence: splits the sequence into `num_segments` subsequences
            of equal length,
            * window: splits the sequence into subsequences
            of `window_size` length with stride equal to `1`. Next,
            `num_segments` subsequences are sampled from the list of
            windows on an equal stride basis.
            Defaults to subsequence.
        num_segments : int
            Number of segments to split the sequence into.
        window_size : int
            Size of the sliding window when using `sliding_window` policy.

        Returns
        -------
        List[List[Any]]
            List of subsequences.

        Raises
        ------
        ValueError
            If the policy is unknown.
        """
        if policy == "subsequence":
            segments = np.linspace(
                0,
                len(sequence),
                num_segments + 1,
                dtype=int,
                endpoint=True,
            )
            return [
                sequence[segments[i] : segments[i + 1]]
                for i in range(num_segments)
            ]
        elif policy == "window":
            windows = np.lib.stride_tricks.sliding_window_view(
                sequence, window_size
            )
            windows_indices = np.linspace(
                0,
                len(windows) - 1,
                num_segments,
                dtype=int,
                endpoint=True,
            )
            return windows[windows_indices].tolist()

        else:
            KLogger.error(f"Unknown policy: {policy}")
            raise ValueError

    @staticmethod
    def sample_items(
        segments: List[List[Any]],
        policy: str = "consecutive",
        items_per_segment: int = 1,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """
        Samples items from segments into a single list of items.
        In total, `len(num_segments) * items_per_segment` items are sampled.

        For the `consecutive` policy, from each segment, a random start-index
        is sampled from which `items_per_segment` consecutive items are
        returned.

        For the `step` policy, from each segment `items_per_segment` items
        are sampled with the equal stride.

        If the number of items in the segment is lower than
        `items_per_segment`, the whole segment is returned.

        Parameters
        ----------
        segments : List[List[Any]]
            List of segments, each containing a list of items.
        policy : str
            The policy for sampling the frames. Can be one of:
            * consecutive: samples `frames_per_segment` consecutive frames
            with random start index from each segment,
            * step: samples `frames_per_segment` frames with equal stride
            from each segment.
            Defaults to consecutive.
        items_per_segment : int
            Number of items to sample from each segment.
        seed : Optional[int]
            Seed for the random number generator.

        Returns
        -------
        List[Any]
            List of sampled items.

        Raises
        ------
        ValueError
            If the policy is unknown or the number of items per segment
            is lower than 1.
        """
        if items_per_segment < 1:
            KLogger.error(
                f"Number of items per segment must be greater than 0, "
                f"got {items_per_segment}"
            )
            raise ValueError
        sampled_items = []
        if policy == "consecutive":
            if seed is not None:
                np.random.seed(seed)
            for segment in segments:
                items_length = len(segment)
                if items_length < items_per_segment:
                    sampled_items.extend(segment)
                    continue

                start_index = np.random.randint(
                    0, items_length - items_per_segment + 1
                )
                sampled_items.extend(
                    segment[start_index : start_index + items_per_segment]
                )
        elif policy == "step":
            for segment in segments:
                items_length = len(segment)
                if items_length < items_per_segment:
                    sampled_items.extend(segment)
                    continue

                item_indices = np.linspace(
                    0,
                    items_length,
                    items_per_segment,
                    dtype=int,
                    endpoint=False,
                )
                sampled_items.extend(np.array(segment)[item_indices].tolist())
        else:
            KLogger.error(f"Unknown policy: {policy}")
            raise ValueError
        return sampled_items
