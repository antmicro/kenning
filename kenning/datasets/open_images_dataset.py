"""
Open Images Dataset V6 wrapper.

The downloader part of the script is based on Open Images Dataset V6::

    https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
"""

# TODO things to consider:
# - provide images with no detectable objects.
# - add support for instance segmentation and other scenarios.

import os
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
from typing import Tuple, List, Dict
import re
from collections import namedtuple
import numpy as np
from collections import defaultdict
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.logger import download_url, get_logger
from kenning.resources import coco_detection

from matplotlib import pyplot as plt
from matplotlib import patches as patches

import zipfile

from typing import Union

BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'

DectObject = namedtuple(
    'DectObject',
    ['clsname', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
)
DectObject.__doc__ = """
Represents single detectable object in an image.

Attributes
----------
class : str
    class of the object
xmin, ymin, xmax, ymax : float
    coordinates of the bounding box
score : float
    the probability of correctness of the detected object
"""

SegmObject = namedtuple(
    'SegmObject',
    ['clsname', 'maskpath', 'xmin', 'ymin', 'xmax', 'ymax', 'mask', 'score']
)
SegmObject.__doc__ = """
Represents single segmentable mask in an image.

Attributes
----------
class : str
    class of the object
maskpath : Path
    path to mask file
xmin,ymin,xmax,ymax : float
    coordinates of the bounding box
mask : np.array
    loaded mask image
score : float
    the probability of correctness of the detected object
"""


def check_and_homogenize_one_image(image: str) -> Tuple[str, str]:
    """
    Subdivides download entry to split type and image ID.

    Parameters
    ----------
    image : str
        image entry in format split/id.

    Returns
    -------
    Tuple[str, str]: tuple containing split and image ID
    """
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id


def check_and_homogenize_image_list(image_list: List[str]) -> Tuple[str, str]:
    """
    Converts download entries using check_and_homogenize_one_image.

    Parameters
    ----------
    image_list : List[str]
        List of download entries

    Yields
    ------
    Tuple[str, str] : download entries
    """
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f'ERROR in line {line_number} of the image list.' +
                f'The following image string is not recognized: "{image}".'
            )


def download_one_image(
        bucket,
        split: str,
        image_id: str,
        download_folder: Path):
    """
    Downloads image from a bucket.

    Parameters
    ----------
    bucket : boto3 bucket
    split : str
        dataset split
    image_id : str
        image id
    download_folder : Path
        target directory
    """
    try:
        bucket.download_file(
            f'{split}/{image_id}.jpg',
            os.path.join(download_folder, f'{image_id}.jpg')
        )
    except botocore.exceptions.ClientError as exception:
        sys.exit(
            f'ERROR when downloading image `{split}/{image_id}`: ' +
            f'{str(exception)}'
        )


def download_all_images(
        download_folder: Path,
        image_list: List[str],
        num_processes: int):
    """
    Downloads all images specified in list of images.

    Parameters
    ----------
    download_folder : Path
        Path to the target directory
    image_list : List[str]
        List of images
    num_processes : int
        Number of threads to use for image download
    """
    bucket = boto3.resource(
        's3', config=botocore.config.Config(
            signature_version=botocore.UNSIGNED
        )
    ).Bucket(BUCKET_NAME)

    download_folder.mkdir(parents=True, exist_ok=True)
    image_list = list(check_and_homogenize_image_list(image_list))

    progress_bar = tqdm.tqdm(
        total=len(image_list),
        desc='Downloading images',
        leave=True
    )
    with futures.ThreadPoolExecutor(
            max_workers=num_processes) as executor:
        all_futures = [
            executor.submit(
                download_one_image,
                bucket,
                split,
                image_id,
                download_folder
            ) for (split, image_id) in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()


def download_instance_segmentation_zip_file(
        zipdir: Path,
        url: str):
    """
    Downloads OpenImagesDatasetV6 segmentation mask zip files
    and extracts the contents

    Parameters
    ----------
    zipdir: Path
        Directory to download and extract the zip file into
    url: str
        Download URL

    Returns
    -------
    """
    download_url(url, zipdir)
    with zipfile.ZipFile(zipdir, 'r') as zip_ref:
        zip_ref.extractall(zipdir.parent)


def compute_ap11(
        recall: List[float],
        precision: List[float]) -> float:
    """
    Computes 11-point Average Precision for a single class.

    Parameters
    ----------
    recall : List[float]
        List of recall values
    precision : List[float]
        List of precision values

    Returns
    -------
    float: 11-point interpolated average precision value
    """
    if len(recall) == 0:
        return 0
    return np.sum(np.interp(np.arange(0, 1.1, 0.1), recall, precision)) / 11


def get_recall_precision(
        measurementsdata: Dict,
        scorethresh: float = 0.5) -> List[Tuple[List[float], List[float]]]:
    """
    Computes recall and precision values at a given objectness threshold.

    Parameters
    ----------
    measurementsdata : Dict
        Data from Measurements object with eval_gcount/{cls} and eval_det/{cls}
        fields containing number of ground truths and per-class detections
    scorethresh : float
        Minimal objectness score threshold for detections

    Returns
    -------
    List[Tuple[List[float], List[float]]] : List with per-class lists of recall
    and precision values
    """
    lines = []
    for cls in measurementsdata['class_names']:
        gt_count = measurementsdata[f'eval_gtcount/{cls}'] if f'eval_gtcount/{cls}' in measurementsdata else 0  # noqa: E501
        dets = measurementsdata[f'eval_det/{cls}'] if f'eval_det/{cls}' in measurementsdata else []  # noqa: E501
        dets = [d for d in dets if d[0] >= scorethresh]
        dets.sort(reverse=True, key=lambda x: x[0])
        truepositives = np.array([entry[1] for entry in dets])
        cumsum = np.cumsum(truepositives)
        precision = cumsum / np.arange(1, len(cumsum) + 1)
        recall = cumsum / gt_count
        lines.append([recall, precision])
    return lines


def compute_map_per_threshold(
        measurementsdata: Dict,
        scorethresholds: List[float]) -> List[float]:
    """
    Computes mAP values depending on the objectness threshold.

    Parameters
    ----------
    measurementsdata : Dict
        Data from Measurements object with eval_gcount/{cls} and eval_det/{cls}
        fields containing number of ground truths and per-class detections
    scorethresholds : List[float]
        List of threshold values to verify the mAP for
    """
    maps = []
    for thresh in scorethresholds:
        lines = get_recall_precision(measurementsdata, thresh)
        aps = []
        for line in lines:
            aps.append(compute_ap11(line[0], line[1]))
        maps.append(np.mean(aps))

    return np.array(maps, dtype=np.float32)


class OpenImagesDatasetV6(Dataset):
    """
    The Open Images Dataset V6

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
    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            task: str = 'object_detection',
            classes: str = 'coco',
            download_num_bboxes_per_class: int = 200,
            download_annotations_type: str = 'validation',
            image_memory_layout: str = 'NCHW',
            show_on_eval: bool = False):
        assert image_memory_layout in ['NHWC', 'NCHW']
        self.task = task
        self.download_num_bboxes_per_class = download_num_bboxes_per_class
        if classes == 'coco':
            with path(coco_detection, 'cocov6.classes') as p:
                self.classes = Path(p)
        else:
            self.classes = Path(classes)
        self.download_annotations_type = download_annotations_type
        self.classmap = {}
        self.image_memory_layout = image_memory_layout
        self.classnames = []
        self.show_on_eval = show_on_eval
        super().__init__(root, batch_size, download_dataset)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--task',
            help='The task type',  # noqa: E501
            choices=['object_detection', 'instance_segmentation'],
            default='object_detection'
        )
        group.add_argument(
            '--download-num-bboxes-per-class',
            help='Number of images per object class (this is a preferred value, there may be less or more values)',  # noqa: E501
            type=int,
            default=200
        )
        group.add_argument(
            '--classes',
            help='File containing Open Images class IDs and class names in CSV format to use (can be generated using kenning.scenarios.open_images_classes_extractor) or class type',  # noqa: E501
            type=str,
            default='coco'
        )
        group.add_argument(
            '--download-annotations-type',
            help='Type of annotations to extract the images from',
            choices=['train', 'validation', 'test'],
            default='validation'
        )
        group.add_argument(
            '--download-seed',
            help='Seed for image sampling',
            type=int,
            default=12345
        )
        group.add_argument(
            '--image-memory-layout',
            help='Determines if images should be delivered in NHWC or NCHW format',  # noqa: E501
            choices=['NHWC', 'NCHW'],
            default='NCHW'
        )
        group.add_argument(
            '--show-predictions-on-eval',
            help='Show predictions during evaluation',
            action='store_true'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.download_dataset,
            args.task,
            args.classes,
            args.download_num_bboxes_per_class,
            args.download_annotations_type,
            args.image_memory_layout,
            args.show_predictions_on_eval
        )

    def download_dataset(self):
        self.root.mkdir(parents=True, exist_ok=True)

        # prepare class files
        classnamespath = self.root / 'classnames.csv'
        if self.classes:
            shutil.copy(self.classes, classnamespath)
        else:
            classnamesurl = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'  # noqa: E501
            download_url(classnamesurl, classnamespath)

        # prepare annotations
        annotationsurls = {
            'train': {
                'object_detection': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',  # noqa: E501
                'instance_segmentation': 'https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv'  # noqa: E501
            },
            'validation': {
                'object_detection': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',  # noqa: E501
                'instance_segmentation': 'https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv'  # noqa: E501
            },
            'test': {
                'object_detection': 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',  # noqa: E501
                'instance_segmentation': 'https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv'  # noqa: E501
            }
        }
        origannotationspath = self.root / 'original-annotations.csv'
        download_url(
            annotationsurls[self.download_annotations_type][self.task],
            origannotationspath
        )

        # load classes
        self.classmap = {}
        with open(classnamespath, 'r') as clsfile:
            for line in clsfile:
                clsid, clsname = line.strip().split(',')
                self.classmap[clsid] = clsname

        annotations = pd.read_csv(origannotationspath)

        # drop grouped bboxes (where one bbox covers multiple examples of a
        # class)
        if self.task == 'object_detection':
            annotations = annotations[annotations.IsGroupOf == 0]

        # filter only entries with desired classes
        filtered = annotations[annotations.LabelName.isin(
            list(self.classmap.keys())
        )]

        # sample image ids to get around download_num_bboxes_per_class bounding
        # boxes for each class
        sampleids = filtered.groupby(
            filtered.LabelName,
            group_keys=False
        ).apply(
            lambda grp: grp.sample(frac=1.0).ImageID.drop_duplicates().head(
                self.download_num_bboxes_per_class
            )
        )

        # get final annotations
        final_annotations = filtered[filtered.ImageID.isin(sampleids)]
        # sort by images
        final_annotations.sort_values('ImageID')

        # save annotations
        annotationspath = self.root / 'annotations.csv'
        final_annotations.to_csv(annotationspath, index=False)

        # if the task is instance_segmentation download required masks
        if self.task == 'instance_segmentation':
            imageidprefix = []
            maskdir = self.root / 'masks'
            maskdir.mkdir(parents=True, exist_ok=True)

            # extract all first characters of ImageIDs into a set
            imageidprefix = set([i[0] for i in final_annotations.ImageID])

            # download the corresponding
            # zip and extract the needed masks from it
            # for each prefix in imageidprefix
            zip_url_template = {
                'train': "https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{}.zip",  # noqa: E501
                'validation': "https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-{}.zip",  # noqa: E501
                'test': "https://storage.googleapis.com/openimages/v5/test-masks/test-masks-{}.zip"  # noqa: E501
            }
            for i in tqdm.tqdm(
                    sorted(imageidprefix),
                    desc="Downloading zip files"):
                zipdir = self.root / 'zip/'
                zipdir.mkdir(parents=True, exist_ok=True)
                zipdir = self.root / 'zip/file.zip'

                pattern = '^{}.*png'.format(i)
                download_instance_segmentation_zip_file(
                    zipdir,
                    zip_url_template[
                        self.download_annotations_type
                        ].format(i)
                )
                # for each file matching the current zip file's prefix
                # copy this file  into mask directory
                for j in final_annotations.MaskPath:
                    if re.match(pattern, j):
                        shutil.copy(zipdir.parent / j, maskdir)
                shutil.rmtree(zipdir.parent)

        # prepare download entries
        download_entries = [
            f'{self.download_annotations_type}/{cid}' for cid in list(
                final_annotations.ImageID.unique()
            )
        ]

        # download images
        imgdir = self.root / 'img'
        imgdir.mkdir(parents=True, exist_ok=True)
        download_all_images(imgdir, download_entries, psutil.cpu_count())

    def prepare_instance_segmentation(self):
        annotations = defaultdict(list)
        annotationsfile = pd.read_csv(self.root / 'annotations.csv')
        for index, row in annotationsfile.iterrows():
            annotations[row['ImageID']].append(SegmObject(
                clsname=self.classmap[row['LabelName']],
                maskpath=self.root / 'masks' / row['MaskPath'],
                xmin=row['BoxXMin'],
                ymin=row['BoxYMin'],
                xmax=row['BoxXMax'],
                ymax=row['BoxYMax'],
                mask=None,
                score=1.0
            ))
        for k, v in annotations.items():
            self.dataX.append(k)
            self.dataY.append(v)

    def prepare_object_detection(self):
        annotations = defaultdict(list)
        annotationsfile = pd.read_csv(self.root / 'annotations.csv')
        for index, row in annotationsfile.iterrows():
            annotations[row['ImageID']].append(DectObject(
                clsname=self.classmap[row['LabelName']],
                xmin=row['XMin'],
                ymin=row['YMin'],
                xmax=row['XMax'],
                ymax=row['YMax'],
                score=1.0
            ))
        for k, v in annotations.items():
            self.dataX.append(k)
            self.dataY.append(v)

    def prepare(self):
        classnamespath = self.root / 'classnames.csv'
        self.classmap = {}
        with open(classnamespath, 'r') as clsfile:
            for line in clsfile:
                clsid, clsname = line.strip().split(',')
                self.classmap[clsid] = clsname
                self.classnames.append(clsname)

        if self.task == 'object_detection':
            self.prepare_object_detection()
        elif self.task == 'instance_segmentation':
            self.prepare_instance_segmentation()
        self.numclasses = len(self.classmap)

    def prepare_input_samples(self, samples):
        result = []
        for sample in samples:
            img = cv2.imread(str(self.root / 'img' / f'{sample}.jpg'))
            img = cv2.resize(img, (416, 416))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            npimg = np.array(img).astype(np.float32) / 255.0
            if self.image_memory_layout == 'NCHW':
                npimg = np.transpose(npimg, (2, 0, 1))
            result.append(npimg)
        return result

    def compute_iou(self, b1: DectObject, b2: DectObject) -> float:
        """
        Computes the IoU between two bounding boxes.

        Parameters
        ----------
        b1 : DectObject
            First bounding box
        b2 : DectObject
            Second bounding box

        Returns
        -------
        float : IoU value
        """
        xmn = max(b1.xmin, b2.xmin)
        ymn = max(b1.ymin, b2.ymin)
        xmx = min(b1.xmax, b2.xmax)
        ymx = min(b1.ymax, b2.ymax)

        intersectarea = max(0, xmx - xmn) * max(0, ymx - ymn)

        b1area = (b1.xmax - b1.xmin) * (b1.ymax - b1.ymin)
        b2area = (b2.xmax - b2.xmin) * (b2.ymax - b2.ymin)

        iou = intersectarea / (b1area + b2area - intersectarea)

        return iou

    def compute_mask_iou(self, mask1: np.array, mask2: np.array) -> float:
        """
        Computes IoU between two masks

        Parameters
        ----------
        mask1 : np.array
            First mask
        mask2 : np.array
            Second mask

        Returns
        -------
        float : IoU value
        """

        mask_i = np.logical_and(mask1, mask2)
        mask_u = np.logical_or(mask1, mask2)

        mask_u_nonzero = np.count_nonzero(mask_u)
        # in empty masks union is zero
        if mask_u_nonzero != 0:
            align = np.count_nonzero(mask_i) / mask_u_nonzero
        else:
            align = 0.0
        return align

    def prepare_instance_segmentation_output_samples(self, samples):
        """
        Loads instance segmentation masks.

        Parameters
        ----------
        samples : List[List[SegmObject]]
            List of SegmObjects containing data about masks
            and their path

        Returns
        -------
        List[List[SegmObject]] : prepared sample data
        """
        result = []
        for sample in samples:
            result.append([])
            for subsample in sample:
                mask_img = cv2.imread(str(subsample[1]), cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.resize(mask_img, (416, 416))
                new_subsample = SegmObject(
                    clsname=subsample.clsname,
                    maskpath=subsample.maskpath,
                    xmin=subsample.xmin,
                    ymin=subsample.ymax,
                    xmax=subsample.xmax,
                    ymax=subsample.ymax,
                    mask=np.array(mask_img, dtype=np.uint8),
                    score=1.0
                )
                result[-1].append(new_subsample)
        return result

    def prepare_output_samples(self, samples):
        if self.task == "instance_segmentation":
            return self.prepare_instance_segmentation_output_samples(samples)
        else:
            return samples

    def get_hashable(
            self,
            unhashable: Union['DectObject', 'SegmObject']
            ) -> Union['DectObject', 'SegmObject']:

        """
        Returns hashable versions of objects depending on self.task

        Parameters
        ----------
        unhashable : Union['DectObject', 'SegmObject']
            Object to be made hashable

        Returns
        -------
        Union['DectObject', 'SegmObject'] : the hashable object
        """

        if self.task == 'object_detection':
            hashable = unhashable
        elif self.task == 'instance_segmentation':
            hashable = SegmObject(
                    clsname=unhashable.clsname,
                    maskpath=unhashable.maskpath,
                    xmin=unhashable.xmin,
                    ymin=unhashable.ymax,
                    xmax=unhashable.xmax,
                    ymax=unhashable.ymax,
                    mask=None,
                    score=1.0
                )
        return hashable

    def evaluate(self, predictions, truth):
        MIN_IOU = 0.5
        measurements = Measurements()

        # adding measurements:
        # - cls -> conf / tp-fp / iou

        for pred, groundtruth in zip(predictions, truth):
            used = set()
            for p in pred:
                foundgt = False
                maxiou = 0
                for gt in groundtruth:
                    if p.clsname == gt.clsname:
                        if self.task == 'object_detection':
                            iou = self.compute_iou(p, gt)
                        elif self.task == 'instance_segmentation':
                            iou = self.compute_mask_iou(p.mask, gt.mask)
                        maxiou = iou if iou > maxiou else maxiou
                        if iou > MIN_IOU and self.get_hashable(gt) not in used:
                            used.add(self.get_hashable(gt))
                            foundgt = True
                            measurements.add_measurement(
                                f'eval_det/{p.clsname}',
                                [[
                                    float(p.score),
                                    float(1),
                                    float(iou)
                                ]],
                                lambda: list()
                            )
                            break
                if not foundgt:
                    measurements.add_measurement(
                        f'eval_det/{p.clsname}',
                        [[
                            float(p.score),
                            float(0),
                            float(maxiou)
                        ]],
                        lambda: list()
                    )
            for gt in groundtruth:
                measurements.accumulate(
                    f'eval_gtcount/{gt.clsname}',
                    1,
                    lambda: 0
                )

        if self.show_on_eval and self.task == 'object_detection':
            log = get_logger()
            log.info(f'\ntruth\n{truth}')
            log.info(f'\npredictions\n{predictions}')
            for pred, gt in zip(predictions, truth):
                img = self.prepare_input_samples([self.dataX[self._dataindex - 1]])[0]  # noqa: E501
                fig, ax = plt.subplots()
                ax.imshow(img.transpose(1, 2, 0))
                for bbox in pred:
                    rect = patches.Rectangle(
                        (bbox.xmin * img.shape[1], bbox.ymin * img.shape[2]),
                        (bbox.xmax - bbox.xmin) * img.shape[1],
                        (bbox.ymax - bbox.ymin) * img.shape[2],
                        linewidth=3,
                        edgecolor='r',
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                for bbox in gt:
                    rect = patches.Rectangle(
                        (bbox.xmin * img.shape[1], bbox.ymin * img.shape[2]),
                        (bbox.xmax - bbox.xmin) * img.shape[1],
                        (bbox.ymax - bbox.ymin) * img.shape[2],
                        linewidth=2,
                        edgecolor='g',
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                plt.show()
        # if show_on_eval then use cv2 to apply truth and prediction masks
        # on different color channels
        elif self.show_on_eval and self.task == 'instance_segmentation':
            log = get_logger()
            log.info(f'\ntruth\n{truth}')
            log.info(f'\npredictions\n{predictions}')
            evaldir = self.root / 'eval'
            evaldir.mkdir(parents=True, exist_ok=True)
            for pred, gt in zip(predictions, truth):
                img = self.prepare_input_samples([self.dataX[self._dataindex - 1]])[0]  # noqa: E501
                if self.image_memory_layout == 'NCHW':
                    img = img.transpose(1,2,0)
                int_img = np.multiply(img, 255).astype('uint8')
                int_img = cv2.cvtColor(int_img, cv2.COLOR_BGR2GRAY)
                int_img = cv2.cvtColor(int_img, cv2.COLOR_GRAY2RGB)
                for i in gt:
                    mask_img = cv2.cvtColor(i.mask, cv2.COLOR_GRAY2RGB)
                    mask_img = mask_img.astype('float32') / 255.0
                    mask_img *= np.array([0.1, 0.1, 0.5])
                    mask_img = np.multiply(mask_img, 255).astype('uint8')
                    int_img = cv2.addWeighted(int_img, 1, mask_img, 0.7, 0)
                for i in pred:
                    mask_img = cv2.cvtColor(i.mask, cv2.COLOR_GRAY2RGB)
                    mask_img = mask_img.astype('float32') / 255.0
                    mask_img *= np.array([0.1, 0.5, 0.1])
                    mask_img = np.multiply(mask_img, 255).astype('uint8')
                    int_img = cv2.addWeighted(int_img, 1, mask_img, 0.7, 0)
                cv2.imwrite(
                    str(evaldir / self.dataX[self._dataindex - 1])+".jpg",
                    int_img
                )
        return measurements

    def get_class_names(self):
        return self.classnames
