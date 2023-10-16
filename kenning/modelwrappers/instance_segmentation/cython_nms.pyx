# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b) nogil:
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b) nogil:
    return a if a <= b else b

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def nms(np.ndarray[np.float32_t, ndim=2] boxes,
        np.ndarray[np.float32_t, ndim=1] scores,
        np.float32_t thresh) -> np.array:
    """
    Performs non-maximum suppression on the bounding boxes

    Parameters
    ----------
    boxes : np.ndarray[np.float32_t, ndim=2]
        Array of shape (ndets, 4) with coordinates of the bounding boxes
    scores : np.ndarray[np.float32_t, ndim=1]
        Array of shape (ndets,) with scores of the detections
    thresh : float
        Threshold for NMS

    Returns
    -------
    np.array :
        Array with indices of detections to be kept
    """
    cdef int ndets = scores.shape[0]

    if ndets == 0:
        return np.array([], dtype=int)

    # Indices of detections to be suppressed
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=int)

    # Areas of detections
    cdef np.ndarray[np.float32_t, ndim=1] areas = np.array(
            [calc_area(box) for box in boxes],
            dtype=np.float32
    )

    # Order to iterate over detections (sorted by score)
    cdef np.ndarray[np.int64_t, ndim=1] order = np.argsort(
            scores,
    )[::-1]

    cdef np.float32_t iou

    cdef int i, j
    for i in range(ndets):
        if suppressed[order[i]] == 1:
            continue
        for j in range(i + 1, ndets):
            if suppressed[order[j]] == 1:
                continue
            iou = calc_iou(areas[order[i]], areas[order[j]], boxes[order[i]], boxes[order[j]])
            if iou >= thresh:
                suppressed[order[j]] = 1
    return np.where(suppressed == 0)[0]

@cython.boundscheck(False)
def calc_area(np.ndarray[np.float32_t, ndim=1] box) -> np.float32_t:
    """
    Calculates area of given bounding box

    Parameters
    ----------
    box : np.ndarray[np.float32_t, ndim=1]
        Array of shape (4,) with coordinates of the bounding box

    Returns
    -------
    np.float32_t :
        Area of the bounding box
    """
    cdef np.float32_t xmin = box[0]
    cdef np.float32_t ymin = box[1]
    cdef np.float32_t xmax = box[2]
    cdef np.float32_t ymax = box[3]
    return (xmax - xmin + 1) * (ymax - ymin + 1)

@cython.boundscheck(False)
def calc_iou(np.float32_t iarea, np.float32_t jarea, np.ndarray[np.float32_t, ndim=1] ibox,
             np.ndarray[np.float32_t, ndim=1] jbox) -> np.float32_t:
    """
    Calculates intersection over union of two detections

    Parameters
    ----------
    iarea : np.float32_t
        Area of the first detection
    jarea : np.float32_t
        Area of the second detection
    ibox : np.ndarray[np.float32_t, ndim=1]
        Array of shape (4,) with coordinates of the first bounding box
    jbox : np.ndarray[np.float32_t, ndim=1]
        Array of shape (4,) with coordinates of the second bounding box

    Returns
    -------
    np.float32_t :
        Intersection over union of the two detections
    """
    cdef np.float32_t xmin = max(ibox[0], jbox[0])
    cdef np.float32_t ymin = max(ibox[1], jbox[1])
    cdef np.float32_t xmax = min(ibox[2], jbox[2])
    cdef np.float32_t ymax = min(ibox[3], jbox[3])
    cdef np.float32_t intersection = max(0.0, xmax - xmin + 1) * max(0.0, ymax - ymin + 1)
    return intersection / (iarea + jarea - intersection)
