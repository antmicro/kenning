# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of methods for computing benchmark and quality metrics.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np

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
            "accuracy": accuracy(confusion_matrix),
            "mean_precision": mean_precision(confusion_matrix),
            "mean_sensitivity": mean_sensitivity(confusion_matrix),
            "g_mean": g_mean(confusion_matrix),
        }
        if "top_5_count" in measurementsdata.keys():
            metrics["top_5_accuracy"] = (
                measurementsdata["top_5_count"] / measurementsdata["total"]
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
        return {"mAP": compute_map_per_threshold(measurementsdata, [0.0])[0]}
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
