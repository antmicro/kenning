# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from pathlib import Path
from collections import defaultdict
import tempfile
import numpy as np
import cv2
from pycocotools.coco import COCO

from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.logger import get_logger
from kenning.datasets.coco_dataset import download_and_extract


class VisualWakeWordsDataset(Dataset):

    dataset_urls = {
        'train2017': {
            'images': ['http://images.cocodataset.org/zips/train2017.zip'],
            'annotations': ['http://images.cocodataset.org/annotations/annotations_trainval2017.zip'],  # noqa: E501
        },
        'val2017': {
            'images': ['http://images.cocodataset.org/zips/val2017.zip'],
            'annotations': ['http://images.cocodataset.org/annotations/annotations_trainval2017.zip'],  # noqa: E501
        }
    }

    arguments_structure = {
        'dataset_type': {
            'argparse_name': '--dataset-type',
            'description': 'Type of dataset to download and use',
            'default': 'val2017',
            'enum': list(dataset_urls.keys())
        },
        'objects_class': {
            'argparse_name': '--objects_class',
            'description': 'Name of objects class to be filtered',
            'default': 'person',
            'type': str
        },
        'area_threshold': {
            'argparse_name': '--area-threshold',
            'description': 'Threshold of fraction of image area below which '
                           'objects are filtered',
            'default': .005,
            'type': float
        },
        'central_fraction': {
            'argparse_name': '--central-fraction',
            'description': 'Fraction used to crop images during preprocessing',
            'default': .875,
            'type': float
        },
        'image_memory_layout': {
            'argparse_name': '--image-memory-layout',
            'description': 'Determines if images should be delivered in NHWC '
                           'or NCHW format',
            'default': 'NHWC',
            'enum': ['NHWC', 'NCHW']
        },
        'image_width': {
            'description': 'Width of the input images',
            'type': int,
            'default': 96
        },
        'image_height': {
            'description': 'Height of the input images',
            'type': int,
            'default': 96
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            external_calibration_dataset: Optional[Path] = None,
            dataset_type: str = 'val2017',
            objects_class: str = 'person',
            area_threshold: float = .005,
            central_fraction: float = .875,
            image_memory_layout: str = 'NHWC',
            image_width: int = 96,
            image_height: int = 96):
        assert image_memory_layout in ['NHWC', 'NCHW']
        self.dataset_type = dataset_type
        self.objects_class = objects_class
        self.area_threshold = area_threshold
        self.central_fraction = central_fraction
        self.image_memory_layout = image_memory_layout
        self.image_width = image_width
        self.image_height = image_height
        self.numclasses = 2
        self.classnames = ['not-person', 'person']
        self.log = get_logger()
        super().__init__(
            root,
            batch_size,
            download_dataset,
            external_calibration_dataset
        )

    @classmethod
    def from_argparse(cls, args):
        return cls(
            root=args.dataset_root,
            batch_size=args.inference_batch_size,
            download_dataset=args.download_dataset,
            external_calibration_dataset=args.external_calibration_dataset,
            dataset_type=args.dataset_type,
            image_memory_layout=args.image_memory_layout,
            image_width=args.image_width,
            image_height=args.image_height
        )

    def get_class_names(self):
        return self.classnames

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            for url in self.dataset_urls[self.dataset_type]['images']:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')
            for url in self.dataset_urls[self.dataset_type]['annotations']:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')

    def prepare(self):
        annotationspath = \
            self.root / f'annotations/instances_{self.dataset_type}.json'
        self.coco = COCO(annotationspath)
        self.classmap = {}

        for classid in self.coco.cats.keys():
            self.classmap[classid] = self.coco.cats[classid]['name']

        cocokeys = list(self.coco.imgs.keys())
        self.keystoimgs = dict()

        for key, imgdata in zip(cocokeys, self.coco.loadImgs(cocokeys)):
            filepath = str(
                self.root / self.dataset_type / imgdata['file_name']
            )
            self.dataX.append(filepath)
            self.keystoimgs[key] = filepath

        classes = defaultdict(lambda: 0)
        for _, anndata in self.coco.anns.items():
            bbox = anndata['bbox']
            bbox_area = bbox[2]*bbox[3]
            img_area = self.image_width*self.image_height
            if (self.objects_class == self.classmap[anndata['category_id']] and
                    bbox_area > self.area_threshold*img_area):
                classes[self.keystoimgs[anndata['image_id']]] |= 1

        for inputid in self.dataX:
            self.dataY.append(classes[inputid])

    def prepare_input_samples(self, samples):
        result = []
        for imgpath in samples:
            img = cv2.imread(str(imgpath))
            w, h = img.shape[:2]
            img = img[int((w/2)*(1 - self.central_fraction)):
                      int((w/2)*(1 + self.central_fraction)),
                      int((h/2)*(1 - self.central_fraction)):
                      int((h/2)*(1 + self.central_fraction))]
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            img = img/127.5 - 1
            if self.image_memory_layout == 'NCHW':
                img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, -1)
            result.append(img)
        return result

    def evaluate(self, predictions, truth):
        confusion_matrix = np.zeros((self.numclasses, self.numclasses))

        for prediction, label in zip(predictions, truth):
            confusion_matrix[label, np.argmax(prediction)] += 1

        measurements = Measurements()
        measurements.accumulate(
            'eval_confusion_matrix',
            confusion_matrix,
            lambda: np.zeros((self.numclasses, self.numclasses))
        )
        measurements.accumulate('total', len(predictions), lambda: 0)
        return measurements