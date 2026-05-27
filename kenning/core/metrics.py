# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of methods for computing benchmark and quality metrics.
"""

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from kenning.utils.logger import KLogger

EPS = 1e-8


def accuracy(confusion_matrix: Union[List[List[int]], np.ndarray]) -> float:
    """
    Computes accuracy of the classifier based on confusion matrix.

    Parameters
    ----------
    confusion_matrix : Union[List[List[int]], np.ndarray]
        The Numpy nxn array or nxn list representing confusion matrix.

    Returns
    -------
    float
        Accuracy of the model
    """
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)


def mean_precision(
    confusion_matrix: Union[List[List[int]], np.ndarray],
) -> float:
    """
    Computes mean precision for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix : Union[List[List[int]], np.ndarray]
        The Numpy nxn array or nxn list representing confusion matrix.

    Returns
    -------
    float
        Precision of the model
    """
    return np.mean(
        np.array(confusion_matrix).diagonal()
        / (np.sum(confusion_matrix, axis=0) + EPS)
    )


def mean_sensitivity(
    confusion_matrix: Union[List[List[int]], np.ndarray],
) -> float:
    """
    Computes mean sensitivity for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix : Union[List[List[int]], np.ndarray]
        The Numpy nxn array or nxn list representing confusion matrix.

    Returns
    -------
    float
        Sensitivity of the model
    """
    return np.mean(
        np.array(confusion_matrix).diagonal()
        / (np.sum(confusion_matrix, axis=1) + EPS)
    )


def g_mean(confusion_matrix: Union[List[List[int]], np.ndarray]) -> float:
    """
    Computes g-mean metric for the confusion matrix.

    Parameters
    ----------
    confusion_matrix : Union[List[List[int]], np.ndarray]
        The Numpy nxn array or nxn list representing confusion matrix.

    Returns
    -------
    float
        G-Mean of the model
    """
    return np.float_power(
        np.prod(
            np.array(confusion_matrix).diagonal()
            / (np.sum(confusion_matrix, axis=1) + EPS)
        ),
        1.0 / np.array(confusion_matrix).shape[0],
    )


def mean_signed_difference(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Computes Mean Signed Difference.

    Parameters
    ----------
    x : np.ndarray
        Predictions, a set of 2D points with value and time.
    y : np.ndarray
        Ground truth, a set of 2D points with value and time.

    Returns
    -------
    float
        Mean signed difference for two tensors.

    Raises
    ------
    ValueError
        When inputs have mismatch length.
    """
    if not x.shape == y.shape and x.ndim == 1:
        raise ValueError("Shapes of input tensors are not equal")

    return float(np.mean(x - y))


def nab_metric_raw(
    x: np.ndarray,
    y: np.ndarray,
    atp: float = 2.0,
    atn: float = 0.0,
    afp: float = 0.25,
    afn: float = -0.25,
) -> float:
    """
    Computes raw NAB ( Numenta Anomaly Benchmark ) score.

    Found in https://arxiv.org/pdf/1510.03336

    Parameters
    ----------
    x : np.ndarray
        Predictions, a set of predicted anomalies.
    y : np.ndarray
        Ground truth, a set of ground truth anomalies.
    atp : float
        Weight for true positive, must greater than or equal 0.
    atn : float
        Weight for true negative, must be less than or equal 1.
    afp : float
        Weight for false positive, must be greater than or equal -1.
    afn : float
        Weight for false negative, must be lower than or equal 0.

    Returns
    -------
    float
        Raw Numenta Anomaly Benchmark score.
    """
    # 1. We use anomaly windows with a size equal of 10 % of data length
    # 2. We takes into account earliest detection within the window
    # 3. Each window is centered around ground truth anomaly label
    # 4. Scoring function gives higher positive scores to
    # True-positive detection earlier in the windows and negative
    # Scores to the detections outside the window ( false positive )
    # 5. We give weights for True-Positive, False-Positive,
    # False-Negative, True-Negative

    def scoring_function(y):
        return (atp - afp) * (1 / (1 + np.exp(5 * y))) - 1

    # We take a window of size which is 10 % of input data size
    window_size = int(len(y) * 0.1)

    # Found windows
    windows_centers = np.where(y == 1)[0].flatten()
    # Found indexes with true positive and false positives

    all_detections = np.where(x == 1)[0].flatten()

    score = 0.0

    # Count true negative
    true_negatives = np.where((y == x) & (y == 0))[0]
    score += len(true_negatives) * atn

    # Number of windows with zero detections
    fd = 0

    if len(windows_centers) == 0:
        return score + afp * len(all_detections)

    # Validate points inside windows
    for center in windows_centers:
        # Find earliest detection in the window and give it a positive score
        lower_index = max(center - window_size, 0)
        upper_index = min(center + window_size, len(y))

        window_ids = np.where(
            (all_detections >= lower_index) & (all_detections <= upper_index)
        )[0]
        window_ids = all_detections[window_ids]

        if len(window_ids) > 0:
            earliest_anomaly_index = np.min(window_ids)

            anomaly_relative_position = earliest_anomaly_index - upper_index

            score += scoring_function(anomaly_relative_position) * atp

            # For false positive outside the window on the left,
            # we assign coefficient of -1.0 to them

        else:
            fd += 1

    # Windows without detections are counted as false negatives
    score += fd * afn

    # Validate points outside detection windows
    # Between beginning of the set and first window
    lower_index = max(windows_centers[0] - window_size, 0)

    on_far_left = np.where(all_detections < lower_index)[0]
    on_far_left = all_detections[on_far_left]

    score -= len(on_far_left) * afp

    # Validate points between windows
    for c1, c2 in zip(windows_centers[:-1], windows_centers[1:]):
        window_left_boundary = min(c1 + window_size, len(y))
        window_right_boundary = max(c2 - window_size, 0)

        outside_window = np.where(
            (all_detections > window_left_boundary)
            & (all_detections < window_right_boundary)
        )[0]
        outside_window = all_detections[outside_window]

        score += (
            np.sum(scoring_function(outside_window - window_left_boundary))
            * afp
        )

    # Validate points between last window and end of the set
    upper_index = min(windows_centers[-1] + window_size, len(y))
    on_far_right = np.where(all_detections > upper_index)[0]
    on_far_right = all_detections[on_far_right]

    score += np.sum(scoring_function(on_far_right - upper_index)) * afp

    return score


def nab_metric(
    x: np.ndarray,
    y: np.ndarray,
    atp: float = 2.0,
    atn: float = 0.0,
    afp: float = 0.25,
    afn: float = -0.25,
) -> float:
    """
    Computes NAB ( Numenta Anomaly Benchmark ) score in range from 0 to 100.
    Found in https://arxiv.org/pdf/1510.03336.

    Parameters
    ----------
    x : np.ndarray
        Predictions, a set of predicted anomalies.
    y : np.ndarray
        Ground truth, a set of ground truth anomalies.
    atp : float
        Weight for true positive, must greater than or equal 0.
    atn : float
        Weight for true negative, must be less or equal 1.
    afp : float
        Weight for false positive, must be greater than or equal -1.
    afn : float
        Weight for false negative, must be lower than or equal 0.

    Returns
    -------
    float
        Numenta Anomaly Benchmark score.

    Raises
    ------
    ValueError
        When inputs have mismatch length, or weights parameters
        doesn't mean their conditions.
    """
    if not x.shape == y.shape and x.ndim == 1:
        raise ValueError("Shapes of input tensors are not equal")

    if atp < 0:
        raise ValueError(
            "True positive coefficient should be greater "
            f"or equal than 0, got {atp}"
        )
    if atn > 1:
        raise ValueError(
            "True negative coefficient should be less "
            f"or equal than 1, got {atn}"
        )
    if afp < -1:
        raise ValueError(
            "False positive coefficient should be greater "
            f"or equal than -1, got {afp}"
        )
    if afn > 0:
        raise ValueError(
            "False negative coefficient should be less "
            f"or equal 0, got {afn}"
        )

    # score from current detections
    score = nab_metric_raw(x, y, atp, atn, afp, afn)
    # score from perfect detections
    score_perfect_detector = nab_metric_raw(y, y, atp, atn, afp, afn)
    # score from no detections
    _x = np.zeros(x.shape)
    score_null_detector = nab_metric_raw(_x, y, atp, atn, afp, afn)

    # avoid division by zero
    if score_null_detector == score_perfect_detector:
        score_null_detector += 10e-36

    output = 100.0 * (
        (score - score_null_detector)
        / (score_perfect_detector - score_null_detector)
    )

    return np.floor(output)


def hausdorff_distance_metric(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Computes Hausdorff distance metrics.

    Parameters
    ----------
    x : np.ndarray
        Predictions, a set of 2D points.
    y : np.ndarray
        Ground truth, a set of 2D points.

    Returns
    -------
    float
        Hausdorff distance between two tensors.

    Raises
    ------
    ValueError
        When inputs have mismatch length.
    """
    if not x.ndim == 2 and y.ndim == 2:
        raise ValueError("Shapes of input tensors are not equal")

    def _d(a, b) -> float:
        d = np.sqrt(np.sum((b - a) ** 2, axis=1))
        return np.min(d)

    max_dxY = max(_d(_x, y) for _x in x)

    max_dyX = max(_d(_y, x) for _y in y)

    return float(max(max_dxY, max_dyX))


def compute_performance_metrics(measurementsdata: Dict[str, List]) -> Dict:
    """
    Computes performance metrics based on `measurementsdata` argument.
    If there is no performance metrics returns an empty dictionary.

    Computes mean, standard deviation and median for keys:

    * target_inference_step or protocol_inference_step
    * session_utilization_mem_percent
    * session_utilization_cpus_percent
    * session_utilization_gpu_mem_utilization
    * session_utilization_gpu_utilization

    Those metrics are stored as <name_of_key>_<mean|std|median>

    Additionally computes time of first inference step
    as `inferencetime_first` and average utilization
    of all cpus used as `session_utilization_cpus_percent_avg` key.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class.

    Returns
    -------
    Dict
        Gathered computed metrics.
    """
    computed_metrics = {}

    def compute_min_max(metric_name: str, metric_value: Optional[Dict] = None):
        if not metric_value:
            metric_value = measurementsdata[metric_name]
        operations = {
            "min": np.min,
            "max": np.max,
        }
        for op_name, op in operations.items():
            computed_metrics[f"{metric_name}_{op_name}"] = op(metric_value)

    def compute_metrics(metric_name: str, metric_value: Optional[Dict] = None):
        """
        Evaluates and saves metric in `operations` dictionary.

        Parameters
        ----------
        metric_name : str
            Name that is used to save metric evaluation.
        metric_value : Optional[Dict]
            Values that are used to evaluate the metric
            If it is none then `measurementsdata[metric_name]` is used.
        """
        if not metric_value:
            metric_value = measurementsdata[metric_name]
        operations = {
            "mean": np.mean,
            "std": np.std,
            "median": np.median,
        }
        for op_name, op in operations.items():
            computed_metrics[f"{metric_name}_{op_name}"] = op(metric_value)

    # inferencetime
    inference_step = None
    if "target_inference_step" in measurementsdata:
        inference_step = "target_inference_step"
    elif "protocol_inference_step" in measurementsdata:
        inference_step = "protocol_inference_step"

    if inference_step:
        compute_metrics("inferencetime", measurementsdata[inference_step])
        compute_min_max("inferencetime", measurementsdata[inference_step])

    # mem_percent
    if "session_utilization_mem_percent" in measurementsdata:
        compute_metrics(
            "session_utilization_mem_percent",
            measurementsdata["session_utilization_mem_percent"],
        )

    # cpus_percent
    if "session_utilization_cpus_percent" in measurementsdata:
        cpus_percent_avg = [
            np.mean(cpus)
            for cpus in measurementsdata["session_utilization_cpus_percent"]
        ]
        computed_metrics[
            "session_utilization_cpus_percent_avg"
        ] = cpus_percent_avg
        compute_metrics(
            "session_utilization_cpus_percent_avg", cpus_percent_avg
        )

    # gpu_mem
    if "session_utilization_gpu_mem_utilization" in measurementsdata:
        compute_metrics("session_utilization_gpu_mem_utilization")

    # gpu
    if "session_utilization_gpu_utilization" in measurementsdata:
        compute_metrics("session_utilization_gpu_utilization")

    return computed_metrics


class Metric(str, Enum):
    """
    The collection of available metrics.
    """

    ACC = "Accuracy"
    TOP_5 = "Top-5 accuracy"
    MEAN_PREC = "Mean precision"
    MEAN_SENS = "Mean sensitivity"
    G_MEAN = "G-mean"
    ROC_AUC = "ROC AUC"
    ROC_AUC_WEIGHTED = "ROC AUC weighted"
    ROC_AUC_CLASS = "ROC AUC"
    F1 = "F1 score"
    F1_WEIGHTED = "F1 score weighted"
    F1_CLASS = "F1 score"
    mAP = "mAP"
    MAX_mAP = "max_mAP"
    MAX_mAP_ID = "max_mAP_index"
    Hausdorff = "hausdorff"


# List of metrics used for classification
CLASSIFICATION_METRICS = [
    Metric.ACC,
    Metric.TOP_5,
    Metric.MEAN_PREC,
    Metric.MEAN_SENS,
    Metric.G_MEAN,
    Metric.ROC_AUC,
    Metric.ROC_AUC_WEIGHTED,
    Metric.F1,
    Metric.F1_WEIGHTED,
]

ANOMALY_DETECTION_METRICS = [Metric.Hausdorff]


def compute_classification_metrics(measurementsdata: Dict[str, List]) -> Dict:
    """
    Computes classification metrics based on `measurementsdata` argument.
    If there is no classification metrics returns an empty dictionary.

    Computes accuracy, top 5 accuracy, precision, sensitivity and g mean of
    passed confusion matrix stored as `eval_confusion_matrix`.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class.

    Returns
    -------
    Dict
        Gathered computed metrics.
    """
    # If confusion matrix is not present in the measurementsdata, then
    # classification metrics can not be calculated.
    if "eval_confusion_matrix" in measurementsdata:
        confusion_matrix = np.asarray(
            measurementsdata["eval_confusion_matrix"]
        )
        confusion_matrix[np.isnan(confusion_matrix)] = 0.0
        metrics = {
            Metric.ACC: accuracy(confusion_matrix),
            Metric.MEAN_PREC: mean_precision(confusion_matrix),
            Metric.MEAN_SENS: mean_sensitivity(confusion_matrix),
            Metric.G_MEAN: g_mean(confusion_matrix),
        }
        if "top_5_count" in measurementsdata.keys():
            metrics[Metric.TOP_5] = (
                measurementsdata["top_5_count"] / measurementsdata["total"]
            )
        if "predictions" in measurementsdata.keys():
            try:
                preds = [
                    int(v["prediction"])
                    for v in measurementsdata["predictions"]
                ]
                targets = [
                    int(v["target"]) for v in measurementsdata["predictions"]
                ]
            except ValueError:
                KLogger.warning(
                    "Cannot convert predictions to integers,"
                    "ROC AUC and F1 score will not be calculated"
                )
            else:
                class_num = len(set(targets))
                if class_num == 2:
                    metrics[Metric.ROC_AUC] = roc_auc_score(targets, preds)
                    metrics[Metric.F1] = f1_score(targets, preds)
                else:
                    metrics[Metric.ROC_AUC_WEIGHTED] = roc_auc_score(
                        targets,
                        [
                            [1.0 if p == i else 0.0 for i in range(class_num)]
                            for p in preds
                        ],
                        average="weighted",
                        multi_class="ovr",
                    )
                    metrics[Metric.ROC_AUC_CLASS] = roc_auc_score(
                        targets,
                        [
                            [1.0 if p == i else 0.0 for i in range(class_num)]
                            for p in preds
                        ],
                        average=None,
                        multi_class="ovr",
                    )
                    metrics[Metric.F1_WEIGHTED] = f1_score(
                        targets, preds, average="weighted"
                    )
                    metrics[Metric.F1_CLASS] = f1_score(
                        targets, preds, average=None
                    )
        return metrics
    return {}


def compute_detection_metrics(measurementsdata: Dict[str, List]) -> Dict:
    """
    Computes detection metrics based on `measurementsdata` argument.
    If there is no detection metrics returns an empty dictionary.

    Computes mAP values.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class.

    Returns
    -------
    Dict
        Gathered computed metrics.
    """
    from kenning.datasets.helpers.detection_and_segmentation import (
        compute_map_per_threshold,
    )

    # If ground truths count is not present in the measurementsdata, then
    # mAP metric can not be calculated.
    if any(
        [key.startswith("eval_gtcount") for key in measurementsdata.keys()]
    ):
        return {
            Metric.mAP: compute_map_per_threshold(measurementsdata, [0.0])[0]
        }
    return {}


def compute_renode_metrics(measurementsdata: List[Dict[str, List]]) -> Dict:
    """
    Computes Renode metrics based on `measurementsdata` argument.
    If there is no Renode metrics returns an empty dictionary.

    Computes instructions counter for all opcodes and for V-Extension opcodes.

    Parameters
    ----------
    measurementsdata : List[Dict[str, List]]
        Statistics from the Measurements class.

    Returns
    -------
    Dict
        Gathered computed metrics.
    """
    if not any(("opcode_counters" in data for data in measurementsdata)):
        return {}

    aggr_opcode_counters = []
    for data in measurementsdata:
        opcode_counters = defaultdict(int)

        for cpu_counters in data["opcode_counters"].values():
            for opcode, counter in cpu_counters.items():
                opcode_counters[opcode] += counter

        aggr_opcode_counters.append(opcode_counters)

    # retrieve all opcodes with nonzero counters
    all_opcodes = set()
    for opcode_counters in aggr_opcode_counters:
        for opcode, counter in opcode_counters.items():
            if counter > 0:
                all_opcodes.add(opcode)

    # retrieve counters
    opcode_ctrs = []

    for opcode in all_opcodes:
        counters = [opcode]
        for opcode_counters in aggr_opcode_counters:
            counters.append(opcode_counters.get(opcode, 0))
        opcode_ctrs.append(counters)

    opcode_ctrs.sort(key=lambda x: (sum(x[1:]), x[0]), reverse=True)

    v_opcode_ctrs = [
        counters for counters in opcode_ctrs if counters[0][0] == "v"
    ]

    # transpose
    t_opcode_ctrs = list(zip(*opcode_ctrs))
    t_v_opcode_ctrs = list(zip(*v_opcode_ctrs))

    ret = {}
    if len(opcode_ctrs):
        ret["sorted_opcode_counters"] = {}
        ret["sorted_opcode_counters"]["opcodes"] = t_opcode_ctrs[0]
        if len(measurementsdata) == 1:
            ret["sorted_opcode_counters"]["counters"] = {
                "counters": t_opcode_ctrs[1]
            }
        else:
            ret["sorted_opcode_counters"]["counters"] = {
                measurementsdata[i]["model_name"]: t_opcode_ctrs[i + 1]
                for i in range(len(measurementsdata))
            }

    if len(v_opcode_ctrs):
        ret["sorted_vector_opcode_counters"] = {}
        ret["sorted_vector_opcode_counters"]["opcodes"] = t_v_opcode_ctrs[0]
        if len(measurementsdata) == 1:
            ret["sorted_vector_opcode_counters"]["counters"] = {
                "counters": t_v_opcode_ctrs[1]
            }
        else:
            ret["sorted_vector_opcode_counters"]["counters"] = {
                data["model_name"]: t_v_opcode_ctrs[i + 1]
                for i, data in enumerate(measurementsdata)
            }

    ret["instructions_per_inference_pass"] = {
        data["model_name"]: int(sum(opcode_counters.values()) / data["total"])
        for data, opcode_counters in zip(
            measurementsdata, aggr_opcode_counters
        )
    }
    if len(v_opcode_ctrs):
        ret["vector_opcodes_fraction"] = {
            data["model_name"]: sum(t_v_opcode_ctrs[i + 1])
            / sum(t_opcode_ctrs[i + 1])
            for i, data in enumerate(measurementsdata)
        }

    ret["top_10_opcodes_per_inference_pass"] = {}
    for data, opcode_counters in zip(measurementsdata, aggr_opcode_counters):
        opcode_counters = list(map(list, opcode_counters.items()))
        opcode_counters.sort(key=lambda x: x[::-1], reverse=True)
        for i in range(len(opcode_counters)):
            opcode_counters[i][1] //= data["total"]
        top_10 = opcode_counters[:10]
        ret["top_10_opcodes_per_inference_pass"][data["model_name"]] = top_10

    if len(measurementsdata) == 1:
        ret["instructions_per_inference_pass"] = next(
            iter(ret["instructions_per_inference_pass"].values())
        )
        ret["top_10_opcodes_per_inference_pass"] = next(
            iter(ret["top_10_opcodes_per_inference_pass"].values())
        )
        if "vector_opcodes_fraction" in ret:
            ret["vector_opcodes_fraction"] = next(
                iter((ret["vector_opcodes_fraction"].values()))
            )

    return ret


def compute_text_summarization_metrics(
    measurementsdata: List[Dict[str, List]]
) -> Dict:
    """
    Computes text summarization metrics based on `measurementsdata` argument.
    If there is no text summarization metrics returns an empty dictionary.

    Computes rouge values.

    Parameters
    ----------
    measurementsdata : List[Dict[str, List]]
        Statistics from the Measurements class.

    Returns
    -------
    Dict
        Gathered computed metrics.
    """
    metrics = {}
    if "total" not in measurementsdata:
        return metrics

    rouge_metrics = [
        key for key in measurementsdata.keys() if key.startswith("rouge")
    ]

    for metric in rouge_metrics:
        metrics[metric] = (
            measurementsdata[metric] / measurementsdata["total"] * 100
        )
    return metrics
