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
from typing import Tuple, List
import re
from collections import namedtuple
import numpy as np
from collections import defaultdict
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path
from PIL import Image

from edge_ai_tester.core.dataset import Dataset
from edge_ai_tester.core.measurements import Measurements
from edge_ai_tester.utils.logger import download_url
from edge_ai_tester.resources import coco_detection

from matplotlib import pyplot as plt
from matplotlib import patches as patches


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

    *Page*: `Open Images Dataset V6 site <https://storage.googleapis.com/openimages/web/index.html>`_.
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
            choices=['object_detection'],
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
            help='File containing Open Images class IDs and class names in CSV format to use (can be generated using edge_ai_tester.scenarios.open_images_classes_extractor) or class type',
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
            'train': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',  # noqa: E501
            'validation': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',  # noqa: E501
            'test': 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'  # noqa: E501
        }
        origannotationspath = self.root / 'original-annotations.csv'
        download_url(
            annotationsurls[self.download_annotations_type],
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

    def prepare(self):
        classnamespath = self.root / 'classnames.csv'
        self.classmap = {}
        with open(classnamespath, 'r') as clsfile:
            for line in clsfile:
                clsid, clsname = line.strip().split(',')
                self.classmap[clsid] = clsname
                self.classnames.append(clsname)

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
        self.numclasses = len(self.classmap)
    
    def prepare_input_samples(self, samples):
        result = []
        for sample in samples:
            img = cv2.imread(str(self.root / 'img' / f'{sample}.jpg'))
            img = cv2.resize(img, (416, 416))  # this may be moved to specific model
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            npimg = np.array(img, dtype=np.float32) / 255.0
            if self.image_memory_layout == 'NCHW':
                npimg = np.transpose(npimg, (2, 0, 1))
            result.append(npimg)
        return result

    def compute_iou(self, b1: DectObject, b2: DectObject):
        xmn = max(b1.xmin, b2.xmin)
        ymn = max(b1.ymin, b2.ymin)
        xmx = min(b1.xmax, b2.xmax)
        ymx = min(b1.ymax, b2.ymax)

        intersectarea = max(0, xmx - xmn) * max(0, ymx - ymn)

        b1area = (b1.xmax - b1.xmin) * (b1.ymax - b1.ymin)
        b2area = (b2.xmax - b2.xmin) * (b2.ymax - b2.ymin)

        iou = intersectarea / (b1area + b2area - intersectarea)

        return iou

    def prepare_output_samples(self, samples):
        return samples

    def evaluate(self, predictions, truth):
        # TODO add metrics
        if self.show_on_eval:
            img = self.prepare_input_samples([self.dataX[self._dataindex - 1]])[0]
            fig, ax = plt.subplots()
            ax.imshow(img.transpose(1, 2, 0))
            for bbox in predictions[0]:
                rect = patches.Rectangle(
                    (bbox.xmin * img.shape[1], bbox.ymin * img.shape[2]),
                    (bbox.xmax - bbox.xmin) * img.shape[1],
                    (bbox.ymax - bbox.ymin) * img.shape[2],
                    linewidth=3,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
            for bbox in truth[0]:
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
        return Measurements()

    def get_class_names(self):
        return self.classnames
