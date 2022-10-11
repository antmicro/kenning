import cv2
from typing import Optional
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
    [
        'clsname',
        'xmin',
        'ymin',
        'xmax',
        'ymax',
        'score',
        'iscrowd'
    ]
)
DectObject.__doc__ = """
Represents single detectable object in an image.

Attributes
----------
clsname : str
    class of the object
xmin, ymin, xmax, ymax : float
    coordinates of the bounding box
score : float
    the probability of correctness of the detected object
iscrowd : Optional[bool]
    tells if the bounding box is a crowd of objects.
    None or False if not crowd, True if bounding box represents
    crowd.
"""

SegmObject = namedtuple(
    'SegmObject',
    [
        'clsname',
        'maskpath',
        'xmin',
        'ymin',
        'xmax',
        'ymax',
        'mask',
        'score',
        'iscrowd'
    ]
)
SegmObject.__doc__ = """
Represents single segmentable mask in an image.

Attributes
----------
clsname : str
    class of the object
maskpath : Path
    path to mask file
xmin,ymin,xmax,ymax : float
    coordinates of the bounding box
mask : np.array
    loaded mask image
score : float
    the probability of correctness of the detected object
iscrowd : Optional[bool]
    tells if the bounding box is a crowd of objects.
    None or False if not crowd, True if bounding box represents
    crowd.
"""


def compute_ap(
        recall: List[float],
        precision: List[float],
        points: int = 101) -> float:
    """
    Computes N-point Average Precision for a single class.

    Parameters
    ----------
    recall : List[float]
        List of recall values
    precision : List[float]
        List of precision values
    points : int
        Number of points for Average Precision estimation

    Returns
    -------
    float: N-point interpolated average precision value
    """
    if len(recall) == 0:
        return 0
    return np.mean(
        np.interp(np.linspace(0, 1.0, num=points), recall, precision)
    )


def get_recall_precision(
        measurementsdata: Dict,
        scorethresh: float = 0.5,
        recallpoints: int = 101) -> List[Tuple[List[float], List[float]]]:
    """
    Computes recall and precision values at a given objectness threshold.

    Parameters
    ----------
    measurementsdata : Dict
        Data from Measurements object with eval_gcount/{cls} and eval_det/{cls}
        fields containing number of ground truths and per-class detections
    scorethresh : float
        Minimal objectness score threshold for detections
    recallpoints : int
        Number of points to use for recall-precision curves, default 101
        (as in COCO dataset evaluation)

    Returns
    -------
    List[Tuple[List[float], List[float]]] : List with per-class lists of recall
    and precision values
    """
    lines = []
    for cls in measurementsdata['class_names']:
        gt_count = measurementsdata[f'eval_gtcount/{cls}'] if f'eval_gtcount/{cls}' in measurementsdata else 0  # noqa: E501
        if gt_count == 0:
            continue
        dets = measurementsdata[f'eval_det/{cls}'] if f'eval_det/{cls}' in measurementsdata else []  # noqa: E501
        dets = [d for d in dets if d[0] >= scorethresh]
        dets.sort(reverse=True, key=lambda x: x[0])
        tps = np.array([entry[1] == 1 for entry in dets])
        fps = np.array([entry[1] == 0 for entry in dets])
        tpacc = np.cumsum(tps).astype(dtype=np.float)
        fpacc = np.cumsum(fps).astype(dtype=np.float)

        precisions = tpacc / (fpacc + tpacc + np.spacing(1))
        recalls = tpacc / gt_count

        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        recallthresholds = np.linspace(0.0, 1.0, num=recallpoints)
        inds = np.searchsorted(recalls, recallthresholds, side='left')
        newprecisions = np.array(len(recallthresholds))
        for oldid, newid in enumerate(inds):
            newprecisions[oldid] = precisions[newid]

        lines.append([recallthresholds, newprecisions])
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
            aps.append(compute_ap(line[0], line[1]))
        maps.append(np.mean(aps))

    return np.array(maps, dtype=np.float32)


def compute_dect_iou(b1: DectObject, b2: DectObject) -> float:
    """
    Computes the IoU between two bounding boxes.

    In evaluation, b1 is considered prediction, while
    b2 is the ground truth.

    If the ground truth has iscrowd parameter set to true,
    then the intersect area is divided by the area of the
    prediction bounding box instead of full area of the
    prediction and the ground truth

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

    if b2.iscrowd:
        iou = intersectarea / b1area
    else:
        iou = intersectarea / (b1area + b2area - intersectarea)

    return iou


def compute_segm_iou(mask1: np.array, mask2: np.array) -> float:
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
            external_calibration_dataset: Optional[Path] = None,
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
        super().__init__(
            root,
            batch_size,
            download_dataset,
            external_calibration_dataset
        )

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
                    score=1.0,
                    iscrowd=False
                )
        return hashable

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
        if self.task == 'object_detection':
            return compute_dect_iou(b1, b2)
        elif self.task == 'instance_segmentation':
            return compute_segm_iou(b1, b2)

    def show_dect_eval_images(self, predictions, truth):
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

    def show_segm_eval_images(self, predictions, truth):
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

    def show_eval_images(self, predictions, truth):
        if self.task == 'object_detection':
            self.show_dect_eval_images(predictions, truth)
        elif self.task == 'instance_segmentation':
            self.show_segm_eval_images(predictions, truth)

    def evaluate(self, predictions, truth):
        MIN_IOU = 0.5
        measurements = Measurements()

        # TODO add support for specifying ground truth area ranges
        # TODO add support for specifying IoU ranges

        for preds, groundtruths in zip(predictions, truth):
            # operate on a single image
            # first, let's sort predictions by score
            preds.sort(key=lambda x: -x.score)

            # store array of matched ground truth bounding boxes
            matchedgt = np.zeros([len(groundtruths)], dtype=np.int32)

            # for each prediction
            for predid, pred in enumerate(preds):
                # store index of best-matching ground truth
                bestiou = 0
                bestgt = -1
                # iterate over ground truth
                for gtid, gt in enumerate(groundtruths):
                    # skip mismatching classes
                    if pred.clsname != gt.clsname:
                        continue
                    # skip if ground truth is already matched and is not a
                    # crowd
                    if matchedgt[gtid] and not gt.iscrowd:
                        continue
                    iou = self.compute_iou(pred, gt)
                    if iou < bestiou:
                        continue
                    bestiou = iou
                    bestgt = gtid
                if bestgt == -1 or bestiou < MIN_IOU:
                    measurements.add_measurement(
                        f'eval_det/{pred.clsname}',
                        [[
                            float(pred.score),
                            float(0),
                            float(bestiou)
                        ]],
                        lambda: list()
                    )
                measurements.add_measurement(
                    f'eval_det/{pred.clsname}',
                    [[
                        float(pred.score),
                        float(1),
                        float(bestiou)
                    ]],
                    lambda: list()
                )
                matchedgt[bestgt] = predid

            for gt in groundtruths:
                measurements.accumulate(
                    f'eval_gtcount/{gt.clsname}',
                    1,
                    lambda: 0
                )

        if self.show_on_eval:
            self.show_eval_images(predictions, truth)

        return measurements
