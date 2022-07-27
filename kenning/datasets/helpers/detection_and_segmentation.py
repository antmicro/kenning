import numpy as np
from collections import namedtuple


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
        recall: list[float],
        precision: list[float]) -> float:
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
        measurementsdata: dict,
        scorethresh: float = 0.5) -> list[tuple[list[float], list[float]]]:
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
        measurementsdata: dict,
        scorethresholds: list[float]) -> list[float]:
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
