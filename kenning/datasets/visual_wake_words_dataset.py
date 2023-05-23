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

    annotationsurls = {
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
            'description': 'Type of dataset to download and use',  # noqa: E501
            'default': 'val2017',
            'enum': list(annotationsurls.keys())
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            external_calibration_dataset: Optional[Path] = None,
            dataset_type: str = 'val2017',
            image_memory_layout: str = 'NHWC',
            image_width: int = 96,
            image_height: int = 96):
        assert image_memory_layout in ['NHWC', 'NCHW']
        self.dataset_type = dataset_type
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
            for url in self.annotationsurls[self.dataset_type]['images']:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')
            for url in self.annotationsurls[self.dataset_type]['annotations']:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')

    def prepare(self):
        annotationspath = self.root / f'annotations/instances_{self.dataset_type}.json'  # noqa: E501
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
            if ('person' == self.classmap[anndata['category_id']] and
                    bbox[2]*bbox[3] > .005*self.image_width*self.image_height):
                classes[self.keystoimgs[anndata['image_id']]] |= 1

        for inputid in self.dataX:
            self.dataY.append(classes[inputid])

    def prepare_input_samples(self, samples):
        result = []
        for imgpath in samples:
            img = cv2.imread(
                str(imgpath)
            )
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, -1)
            npimg = np.array(img, dtype=np.float32) / 255.0
            if self.image_memory_layout == 'NCHW':
                npimg = np.transpose(npimg, (2, 0, 1))
            result.append(npimg)
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
