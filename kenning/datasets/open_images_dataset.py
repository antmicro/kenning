# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Open Images Dataset V6 wrapper.

The downloader part of the script is based on Open Images Dataset V6::

    https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
"""

# TODO things to consider:
# - provide images with no detectable objects.
# - add support for instance segmentation and other scenarios.

from math import floor, ceil
import cv2
import sys
import psutil
from concurrent import futures
import botocore
import tqdm
import boto3
import pandas as pd
import shutil
from pathlib import Path
import re
import numpy as np
from typing import Tuple, List, Optional
from collections import defaultdict

from kenning.utils.resource_manager import Resources

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.resources import coco_detection
from kenning.utils.logger import LoggerProgressBar, download_url

import zipfile

from kenning.datasets.helpers.detection_and_segmentation import (
    DetectObject,
    SegmObject,
    ObjectDetectionSegmentationDataset,
)

BUCKET_NAME = "open-images-dataset"
REGEX = r"(test|train|validation|challenge2018)/([a-fA-F0-9]*)"


def check_and_homogenize_one_image(image: str) -> Tuple[str, str]:
    """
    Subdivides download entry to split type and image ID.

    Parameters
    ----------
    image : str
        Image entry in format split/id.

    Yields
    ------
    Tuple[str, str]:
        Tuple containing split and image ID.
    """
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id


def check_and_homogenize_image_list(image_list: List[str]) -> Tuple[str, str]:
    """
    Converts download entries using check_and_homogenize_one_image.

    Parameters
    ----------
    image_list : list[str]
        List of download entries.

    Yields
    ------
    Tuple[str, str] :
        Download entries.
    """
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f"ERROR in line {line_number} of the image list."
                + f'The following image string is not recognized: "{image}".'
            )


def download_one_image(
    bucket, split: str, image_id: str, download_folder: Path
):
    """
    Downloads image from a bucket.

    Parameters
    ----------
    bucket : boto3 bucket
        Bucket to download from.
    split : str
        Dataset split.
    image_id : str
        Image id.
    download_folder : Path
        Target directory.
    """
    try:
        bucket.download_file(
            f"{split}/{image_id}.jpg", str(download_folder / f"{image_id}.jpg")
        )
    except botocore.exceptions.ClientError as exception:
        sys.exit(
            f"ERROR when downloading image `{split}/{image_id}`: "
            + f"{str(exception)}"
        )


def download_all_images(
    download_folder: Path, image_list: List[str], num_processes: int
):
    """
    Downloads all images specified in list of images.

    Parameters
    ----------
    download_folder : Path
        Path to the target directory.
    image_list : list[str]
        List of images.
    num_processes : int
        Number of threads to use for image download.
    """
    bucket = boto3.resource(
        "s3",
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    ).Bucket(BUCKET_NAME)

    download_folder.mkdir(parents=True, exist_ok=True)
    image_list = list(check_and_homogenize_image_list(image_list))

    with (
        LoggerProgressBar() as logger_progress_bar,
        tqdm.tqdm(
            total=len(image_list),
            desc="Downloading images",
            leave=True,
            file=logger_progress_bar,
        ) as progress_bar,
        futures.ThreadPoolExecutor(max_workers=num_processes) as executor,
    ):
        all_futures = [
            executor.submit(
                download_one_image, bucket, split, image_id, download_folder
            )
            for (split, image_id) in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)


def download_instance_segmentation_zip_file(zipdir: Path, url: str):
    """
    Downloads OpenImagesDatasetV6 segmentation mask zip files
    and extracts the contents.

    Parameters
    ----------
    zipdir : Path
        Directory to download and extract the zip file into.
    url : str
        Download URL.
    """
    download_url(url, zipdir)
    with zipfile.ZipFile(zipdir, "r") as zip_ref:
        zip_ref.extractall(zipdir.parent)


class OpenImagesDatasetV6(ObjectDetectionSegmentationDataset):
    """
    The Open Images Dataset V6.

    https://storage.googleapis.com/openimages/web/index.html

    It is a dataset of ~9M images annotated with:

    * image-level labels,
    * object bounding boxes,
    * object segmentation masks,
    * visual relationships,
    * localized narratives.

    *License*: Creative Commons Attribution 4.0 International License.

    *Page*: `Open Images Dataset V6 site
    <https://storage.googleapis.com/openimages/web/index.html>`_.
    """

    resources = Resources(
        {
            "class_names": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",  # noqa: E501,
            "train": {
                "object_detection": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",  # noqa: E501
                "instance_segmentation": "https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv",  # noqa: E501
            },
            "validation": {
                "object_detection": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",  # noqa: E501
                "instance_segmentation": "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv",  # noqa: E501
            },
            "test": {
                "object_detection": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",  # noqa: E501
                "instance_segmentation": "https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv",  # noqa: E501
            },
        }
    )
    arguments_structure = {
        "classes": {
            "argparse_name": "--classes",
            "description": "File containing Open Images class IDs and class names in CSV format to use (can be generated using kenning.scenarios.open_images_classes_extractor) or class type",  # noqa: E501
            "type": str,
            "default": "coco",
        },
        "download_annotations_type": {
            "argparse_name": "--download-annotations-type",
            "description": "Type of annotations to extract the images from",
            "default": "validation",
            "enum": ["train", "validation", "test"],
        },
        "download_num_bboxes_per_class": {
            "description": "Number of boxes per class to download",
            "default": 200,
            "type": int,
        },
        "crop_input_to_bboxes": {
            "argparse_name": "--crop-samples-to-bboxes",
            "description": "Crop input samples and masks to show only an area with ground truths",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "crop_input_margin_size": {
            "argparse_name": "--crop-margin",
            "description": "Crop margin",
            "type": float,
            "default": 0.1,
        },
        "download_seed": {
            "argparse_name": "--download-seed",
            "description": "Seed for image sampling",
            "type": int,
            "default": 12345,
        },
    }

    def __init__(
        self,
        root: Path,
        batch_size: int = 1,
        download_dataset: bool = True,
        force_download_dataset: bool = False,
        external_calibration_dataset: Optional[Path] = None,
        split_fraction_test: float = 0.2,
        split_fraction_val: Optional[float] = None,
        split_seed: int = 1234,
        task: str = "object_detection",
        classes: str = "coco",
        download_num_bboxes_per_class: int = 200,
        download_annotations_type: str = "validation",
        image_memory_layout: str = "NCHW",
        show_on_eval: bool = False,
        crop_input_to_bboxes: bool = False,
        crop_input_margin_size: float = 0.1,
        download_seed: int = 12345,
        image_width: int = 416,
        image_height: int = 416,
    ):
        assert image_memory_layout in ["NHWC", "NCHW"]
        self.classes = classes
        self.download_num_bboxes_per_class = download_num_bboxes_per_class
        if classes == "coco":
            with path(coco_detection, "cocov6.classes") as p:
                self.classes_path = Path(p)
        else:
            self.classes_path = Path(classes)
        self.download_annotations_type = download_annotations_type
        self.classmap = {}
        self.classnames = []
        self.crop_input_to_bboxes = crop_input_to_bboxes
        self.crop_input_margin_size = crop_input_margin_size
        if self.crop_input_to_bboxes:
            self.crop_dict = {}
        self.download_seed = download_seed
        super().__init__(
            root,
            batch_size,
            download_dataset,
            force_download_dataset,
            external_calibration_dataset,
            split_fraction_test,
            split_fraction_val,
            split_seed,
            task,
            image_memory_layout,
            show_on_eval,
            image_width,
            image_height,
        )

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)

        # prepare class files
        class_names_path = self.root / "classnames.csv"
        if self.classes_path:
            shutil.copy(self.classes_path, class_names_path)
        else:
            classes_path = self.resources["class_names"]
            shutil.copy(classes_path, class_names_path)

        # load classes
        self.classmap = {}
        with open(class_names_path, "r") as clsfile:
            for line in clsfile:
                clsid, clsname = line.strip().split(",")
                self.classmap[clsid] = clsname

        # prepare annotations
        annotations = pd.read_csv(
            self.resources[self.download_annotations_type, self.task]
        )

        # drop grouped bboxes (where one bbox covers multiple examples of a
        # class)
        if self.task == "object_detection":
            annotations = annotations[annotations.IsGroupOf == 0]

        # filter only entries with desired classes
        filtered = annotations[
            annotations.LabelName.isin(list(self.classmap.keys()))
        ]

        # sample image ids to get around download_num_bboxes_per_class bounding
        # boxes for each class
        sampleids = filtered.groupby(
            filtered.LabelName, group_keys=False
        ).apply(
            lambda grp: grp.sample(frac=1.0)
            .ImageID.drop_duplicates()
            .head(self.download_num_bboxes_per_class)
        )

        # get final annotations
        final_annotations = filtered[filtered.ImageID.isin(sampleids)]
        # sort by images
        final_annotations.sort_values("ImageID")

        # save annotations
        annotationspath = self.root / "annotations.csv"
        final_annotations.to_csv(annotationspath, index=False)

        # if the task is instance_segmentation download required masks
        if self.task == "instance_segmentation":
            imageidprefix = []
            maskdir = self.root / "masks"
            maskdir.mkdir(parents=True, exist_ok=True)

            # extract all first characters of ImageIDs into a set
            imageidprefix = set([i[0] for i in final_annotations.ImageID])

            # download the corresponding
            # zip and extract the needed masks from it
            # for each prefix in imageidprefix
            zip_url_template = {
                "train": "https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{}.zip",  # noqa: E501
                "validation": "https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-{}.zip",  # noqa: E501
                "test": "https://storage.googleapis.com/openimages/v5/test-masks/test-masks-{}.zip",  # noqa: E501
            }
            for i in tqdm.tqdm(
                sorted(imageidprefix), desc="Downloading zip files"
            ):
                zipdir = self.root / "zip/"
                zipdir.mkdir(parents=True, exist_ok=True)
                zipdir = self.root / "zip/file.zip"

                pattern = "^{}.*png".format(i)
                download_instance_segmentation_zip_file(
                    zipdir,
                    zip_url_template[self.download_annotations_type].format(i),
                )
                # for each file matching the current zip file's prefix
                # copy this file into mask directory
                for j in final_annotations.MaskPath:
                    if re.match(pattern, j):
                        shutil.copy(zipdir.parent / j, maskdir)
                shutil.rmtree(zipdir.parent)

        # prepare download entries
        download_entries = [
            f"{self.download_annotations_type}/{cid}"
            for cid in list(final_annotations.ImageID.unique())
        ]

        # download images
        imgdir = self.root / "img"
        imgdir.mkdir(parents=True, exist_ok=True)
        download_all_images(imgdir, download_entries, psutil.cpu_count())

    def prepare_instance_segmentation(self):
        annotations = defaultdict(list)
        annotationsfile = pd.read_csv(self.root / "annotations.csv")
        for _, row in annotationsfile.iterrows():
            annotations[row["ImageID"]].append(
                SegmObject(
                    clsname=self.classmap[row["LabelName"]],
                    maskpath=self.root / "masks" / row["MaskPath"],
                    xmin=row["BoxXMin"],
                    ymin=row["BoxYMin"],
                    xmax=row["BoxXMax"],
                    ymax=row["BoxYMax"],
                    mask=None,
                    score=1.0,
                    iscrowd=False,
                )
            )
        for k, v in annotations.items():
            self.dataX.append(k)
            self.dataY.append(v)
        if self.crop_input_to_bboxes:
            for x in range(len(self.dataX)):
                minx, miny = self.image_width + 1, self.image_height + 1
                maxx, maxy = 0, 0
                for sample in self.dataY[x]:
                    minx = (
                        sample.xmin * self.image_width - 1
                        if sample.xmin * self.image_width - 1 < minx
                        else minx
                    )
                    maxx = (
                        sample.xmax * self.image_width - 1
                        if sample.xmax * self.image_width - 1 > maxx
                        else maxx
                    )
                    miny = (
                        sample.ymin * self.image_height - 1
                        if sample.ymin * self.image_height - 1 < miny
                        else miny
                    )
                    maxy = (
                        sample.ymax * self.image_height - 1
                        if sample.ymax * self.image_height - 1 > maxy
                        else maxy
                    )
                span_x = maxx - minx
                span_y = maxy - miny
                minx = int(
                    floor(minx - span_x * self.crop_input_margin_size / 2)
                )
                minx = minx if minx > 0 else 0
                maxx = int(
                    ceil(maxx + span_x * self.crop_input_margin_size / 2)
                )
                maxx = (
                    maxx if maxx < self.image_width else self.image_width - 1
                )
                miny = int(
                    floor(miny - span_y * self.crop_input_margin_size / 2)
                )
                miny = miny if miny > 0 else 0
                maxy = int(
                    ceil(maxy + span_y * self.crop_input_margin_size / 2)
                )
                maxy = (
                    maxy if maxy < self.image_height else self.image_height - 1
                )
                self.crop_dict[self.dataX[x]] = [minx, miny, maxx, maxy]

    def prepare_object_detection(self):
        annotations = defaultdict(list)
        annotationsfile = pd.read_csv(self.root / "annotations.csv")
        for index, row in annotationsfile.iterrows():
            annotations[row["ImageID"]].append(
                DetectObject(
                    clsname=self.classmap[row["LabelName"]],
                    xmin=row["XMin"],
                    ymin=row["YMin"],
                    xmax=row["XMax"],
                    ymax=row["YMax"],
                    score=1.0,
                    iscrowd=False,
                )
            )
        for k, v in annotations.items():
            self.dataX.append(k)
            self.dataY.append(v)

    def prepare(self):
        class_names_path = self.root / "classnames.csv"
        self.classmap = {}
        with open(class_names_path, "r") as clsfile:
            for line in clsfile:
                clsid, clsname = line.strip().split(",")
                self.classmap[clsid] = clsname
                self.classnames.append(clsname)

        if self.task == "object_detection":
            self.prepare_object_detection()
        elif self.task == "instance_segmentation":
            self.prepare_instance_segmentation()
        self.numclasses = len(self.classmap)

    def get_sample_image_path(self, image_id: str) -> str:
        """
        Returns path to image of given id.

        Parameters
        ----------
        image_id : str
            Id of image.

        Returns
        -------
        str
            Path to the image.
        """
        return str(self.root / "img" / f"{image_id}.jpg")

    def prepare_input_samples(self, samples):
        result = []
        for sample in samples:
            img = cv2.imread(self.get_sample_image_path(sample))
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            npimg = np.array(img, dtype=np.float32) / 255.0
            if self.crop_input_to_bboxes:
                minx, miny, maxx, maxy = self.crop_dict[sample]
                npimg = npimg[miny:maxy, minx:maxx]
                npimg = cv2.resize(
                    npimg, (self.image_width, self.image_height)
                )
            if self.image_memory_layout == "NCHW":
                npimg = np.transpose(npimg, (2, 0, 1))

            result.append(npimg)
        return result

    def prepare_instance_segmentation_output_samples(self, samples):
        """
        Loads instance segmentation masks.

        Parameters
        ----------
        samples : list[list[SegmObject]]
            List of SegmObjects containing data about masks
            and their path.

        Returns
        -------
        list[list[SegmObject]]
            Prepared sample data.
        """
        result = []
        for sample in samples:
            result.append([])
            for subsample in sample:
                mask_img = cv2.imread(str(subsample[1]), cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.resize(
                    mask_img, (self.image_width, self.image_height)
                )
                if self.crop_input_to_bboxes:
                    img_id = str(subsample[1].name).split("_")[0]
                    minx, miny, maxx, maxy = self.crop_dict[img_id]
                    mask_img = mask_img[miny:maxy, minx:maxx]
                    mask_img = cv2.resize(
                        mask_img, (self.image_width, self.image_height)
                    )
                mask_img = np.array(mask_img, dtype=np.uint8)
                new_subsample = SegmObject(
                    clsname=subsample.clsname,
                    maskpath=subsample.maskpath,
                    xmin=subsample.xmin,
                    ymin=subsample.ymin,
                    xmax=subsample.xmax,
                    ymax=subsample.ymax,
                    mask=mask_img,
                    score=1.0,
                    iscrowd=False,
                )
                result[-1].append(new_subsample)
        return result

    def prepare_output_samples(self, samples):
        if self.task == "instance_segmentation":
            return self.prepare_instance_segmentation_output_samples(samples)
        else:
            return samples

    def get_class_names(self):
        return self.classnames
