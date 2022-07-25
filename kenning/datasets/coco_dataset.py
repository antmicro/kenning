import cv2
import numpy as np
from pathlib import Path
import tempfile
from zipfile import ZipFile
from tqdm import tqdm
from collections import defaultdict

from kenning.core.dataset import Dataset
from kenning.utils.logger import download_url, get_logger

from pycocotools.coco import COCO

from kenning.datasets.helpers.detection_and_segmentation import \
        DectObject, \
        SegmObject, \
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
            'object_detection': ['http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip'],  # noqa: E501
        },
        'val2017': {
            'images': ['http://images.cocodataset.org/zips/val2017.zip'],
            'object_detection': ['http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip'],  # noqa: E501
        }
    }

    arguments_structure = {
        'task': {
            'argparse_name': '--task',
            'description': 'he task type',
            'default': 'object_detection',
            'enum': [
                key for dataset in annotationsurls.values()
                for key in dataset.keys() if key != 'images'
            ]
        },
        'dataset_type': {
            'argparse_name': '--dataset-type',
            'description': 'Type of dataset to download and use',  # noqa: E501
            'default': 'val2017',
            'enum': list(annotationsurls.keys())
        },
        'image_memory_layout': {
            'argparse_name': '--image-memory-layout',
            'description': 'Determines if images should be delivered in NHWC or NCHW format',  # noqa: E501
            'default': 'NCHW',
            'enum': ['NHWC', 'NCHW']
        },
        'show_on_eval': {
            'argparse_name': '--show-predictions-on-eval',
            'description': 'Show predictions during evaluation',
            'type': bool,
            'default': False
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            task: str = 'object_detection',
            dataset_type: str = 'val2017',
            image_memory_layout: str = 'NCHW',
            show_on_eval: bool = False):
        assert image_memory_layout in ['NHWC', 'NCHW']
        self.log = get_logger()
        self.task = task
        self.dataset_type = dataset_type
        self.image_memory_layout = image_memory_layout
        self.show_on_eval = show_on_eval
        super().__init__(root, batch_size, download_dataset)

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.download_dataset,
            args.task,
            args.dataset_type,
            args.image_memory_layout,
            args.show_on_eval
        )

    def download_dataset_fun(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            for url in self.annotationsurls[self.dataset_type]['images']:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')
            for url in self.annotationsurls[self.dataset_type][self.task]:
                download_and_extract(url, self.root, Path(tmpdir) / 'data.zip')

    def prepare(self):
        annotationspath = self.root / f'annotations/stuff_{self.dataset_type}.json'  # noqa: E501
        self.coco = COCO(annotationspath)
        self.classmap = {}

        for classid in self.coco.cats.keys():
            self.classmap[classid] = self.coco.cats[classid]['name']

        self.dataX = list(self.coco.imgs.keys())
        annotations = defaultdict(list)
        for annkey, anndata in self.coco.anns.items():
            bbox = anndata['bbox']
            annotations[anndata['image_id']].append(DectObject(
                clsname=self.classmap[anndata['category_id']],
                xmin=bbox[0],
                ymin=bbox[1],
                xmax=bbox[0] + bbox[2],
                ymax=bbox[1] + bbox[3],
                score=1.0
            ))

        for inputid in self.dataX:
            self.dataY.append(annotations[inputid])

    def prepare_input_samples(self, samples):
        result = []
        for imgdata in self.coco.loadImgs(samples):
            img = cv2.imread(
                str(self.root / self.dataset_type / imgdata['file_name'])
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
