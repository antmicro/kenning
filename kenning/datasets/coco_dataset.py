import cv2
import numpy as np
from pathlib import Path
import tempfile
from zipfile import ZipFile
from tqdm import tqdm
from collections import defaultdict
from typing import Optional

from kenning.utils.logger import download_url, get_logger

from pycocotools.coco import COCO

from kenning.datasets.helpers.detection_and_segmentation import \
        DectObject, \
        ObjectDetectionSegmentationDataset


def download_and_extract(url: str, targetdir: Path, downloadpath: Path):
    """
    Downloads the ZIP file and extracts it to the provided target directory.

    Parameters
    ----------
    url : str
        URL to the ZIP file to download
    targetdir : Path
        Path to the target directory where extracted files will be saved
    downloadpath: Path
        Path where the ZIP file will be downloaded to
    """
    download_url(url, downloadpath)
    with ZipFile(downloadpath, 'r') as zip:
        for f in tqdm(iterable=zip.namelist(), total=len(zip.namelist())):
            zip.extract(member=f, path=targetdir)


class COCODataset2017(ObjectDetectionSegmentationDataset):
    """
    The COCO Dataset 2017

    https://cocodataset.org

    COCO is a large-scale object detection, segmentation and captioning
    dataset.

    For object detection, it provides 80 categories of objects.

    *License*: Creative Commons Attribution 4.0 License.

    *NOTE*: the license does not include the images, only annotations.

    *Page*: `COCO Dataset site <https://cocodataset.org>`_.
    """

    annotationsurls = {
        'train2017': {
            'images': ['http://images.cocodataset.org/zips/train2017.zip'],
            'object_detection': ['http://images.cocodataset.org/annotations/annotations_trainval2017.zip'],  # noqa: E501
        },
        'val2017': {
            'images': ['http://images.cocodataset.org/zips/val2017.zip'],
            'object_detection': ['http://images.cocodataset.org/annotations/annotations_trainval2017.zip'],  # noqa: E501
        }
    }

    arguments_structure = {
        'task': {
            'argparse_name': '--task',
            'description': 'The task type',
            'default': 'object_detection',
            'enum': list(set(
                key for dataset in annotationsurls.values()
                for key in dataset.keys() if key != 'images'
            ))
        },
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
            task: str = 'object_detection',
            dataset_type: str = 'val2017',
            image_memory_layout: str = 'NCHW',
            show_on_eval: bool = False,
            image_width: int = 416,
            image_height: int = 416):
        assert image_memory_layout in ['NHWC', 'NCHW']
        self.log = get_logger()
        self.dataset_type = dataset_type
        super().__init__(
            root,
            batch_size,
            download_dataset,
            external_calibration_dataset,
            task,
            image_memory_layout,
            show_on_eval,
            image_width,
            image_height
        )

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.download_dataset,
            args.external_calibration_dataset,
            args.task,
            args.dataset_type,
            args.image_memory_layout,
            args.show_predictions_on_eval,
            args.image_width,
            args.image_height
        )

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            for url in self.annotationsurls[self.dataset_type]['images']:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')
            for url in self.annotationsurls[self.dataset_type][self.task]:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')

    def prepare(self):
        annotationspath = self.root / f'annotations/instances_{self.dataset_type}.json'  # noqa: E501
        self.coco = COCO(annotationspath)
        self.classmap = {}
        self.classnames = []

        for classid in self.coco.cats.keys():
            self.classmap[classid] = self.coco.cats[classid]['name']
            self.classnames.append(self.coco.cats[classid]['name'])

        cocokeys = list(self.coco.imgs.keys())
        self.keystoimgs = dict()
        self.imgstokeys = dict()

        for key, imgdata in zip(cocokeys, self.coco.loadImgs(cocokeys)):
            filepath = str(
                self.root / self.dataset_type / imgdata['file_name']
            )
            self.dataX.append(filepath)
            self.keystoimgs[key] = filepath
            self.imgstokeys[filepath] = key

        annotations = defaultdict(list)
        for annkey, anndata in self.coco.anns.items():
            bbox = anndata['bbox']
            width = self.coco.imgs[anndata['image_id']]['width']
            height = self.coco.imgs[anndata['image_id']]['height']
            annotations[
                self.keystoimgs[anndata['image_id']]].append(DectObject(
                    clsname=self.classmap[anndata['category_id']],
                    xmin=bbox[0] / width,
                    ymin=bbox[1] / height,
                    xmax=(bbox[0] + bbox[2]) / width,
                    ymax=(bbox[1] + bbox[3]) / height,
                    score=1.0,
                    iscrowd=anndata['iscrowd'] == 1
                )
            )

        for inputid in self.dataX:
            self.dataY.append(annotations[inputid])

    def prepare_input_samples(self, samples):
        result = []
        for imgpath in samples:
            img = cv2.imread(
                str(imgpath)
            )
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            npimg = np.array(img, dtype=np.float32) / 255.0
            if self.image_memory_layout == 'NCHW':
                npimg = np.transpose(npimg, (2, 0, 1))
            result.append(npimg)
        return result

    def get_class_names(self):
        return self.classnames

    def evaluate(self, predictions, truth):
        measurements = super().evaluate(predictions, truth)
        currindex = self._dataindex - len(predictions)
        for pred, groundtruth in zip(predictions, truth):
            for p in pred:
                cocoid = self.imgstokeys[self.dataX[currindex]]
                width = self.coco.imgs[cocoid]['width']
                height = self.coco.imgs[cocoid]['height']
                xmin = max(min(p.xmin * width, width), 0)
                xmax = max(min(p.xmax * width, width), 0)
                ymin = max(min(p.ymin * height, height), 0)
                ymax = max(min(p.ymax * height, height), 0)
                w = xmax - xmin
                h = ymax - ymin
                measurements.add_measurement(
                    'predictions',
                    [{
                        'image_name': self.imgstokeys[self.dataX[currindex]],
                        'category': p.clsname,
                        'bbox': [xmin, ymin, w, h],
                        'score': p.score
                    }]
                )
        return measurements
