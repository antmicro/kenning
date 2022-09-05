import cv2
import numpy as np
from collections import namedtuple
from typing import Union, List, Dict, Tuple
from pathlib import Path

from kenning.utils.logger import get_logger
from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements
from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument  # noqa: E501

from matplotlib import pyplot as plt
from matplotlib import patches as patches

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


def compute_iou(b1: DectObject, b2: DectObject) -> float:
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


def compute_mask_iou(mask1: np.array, mask2: np.array) -> float:
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


class ObjectDetectionSegmentationDataset(Dataset):
    arguments_structure = {
        'task': {
            'argparse_name': '--task',
            'description': 'The task type',
            'default': 'object_detection',
            'enum': ['object_detection', 'instance_segmentation']
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
        },
        'image_width': {
            'description': 'Width of the input images',
            'type': int,
            'default': 416
        },
        'image_height': {
            'description': 'Height of the input images',
            'type': int,
            'default': 416
        }
    }

    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            task: str = 'object_detection',
            image_memory_layout: str = 'NCHW',
            show_on_eval: bool = False,
            image_width: int = 416,
            image_height: int = 416):
        assert image_memory_layout in ['NHWC', 'NCHW']
        self.task = task
        self.classmap = {}
        self.image_memory_layout = image_memory_layout
        self.image_width = image_width
        self.image_height = image_height
        self.classnames = []
        self.show_on_eval = show_on_eval
        super().__init__(root, batch_size, download_dataset)

    @classmethod
    def form_parameterschema(cls):
        parameterschema = super(
            ObjectDetectionSegmentationDataset,
            ObjectDetectionSegmentationDataset
        ).form_parameterschema()

        if cls != ObjectDetectionSegmentationDataset:
            add_parameterschema_argument(
                parameterschema,
                cls.arguments_structure
            )
        return parameterschema

    @classmethod
    def form_argparse(cls):
        parser, group = super(
            ObjectDetectionSegmentationDataset,
            ObjectDetectionSegmentationDataset
        ).form_argparse()

        if cls != ObjectDetectionSegmentationDataset:
            add_argparse_argument(
                group,
                cls.arguments_structure
            )
        return parser, group

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
                            iou = compute_iou(p, gt)
                        elif self.task == 'instance_segmentation':
                            iou = compute_mask_iou(p.mask, gt.mask)
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
                    img = img.transpose(1, 2, 0)
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
