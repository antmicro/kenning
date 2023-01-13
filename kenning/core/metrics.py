from typing import Dict, List, Optional, Union

import numpy as np

from kenning.utils import logger

log = logger.get_logger()


def accuracy(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes accuracy of the classifier based on confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)


def mean_precision(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes mean precision for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.mean(
        np.array(confusion_matrix).diagonal() /
        np.sum(confusion_matrix, axis=1)
    )


def mean_sensitivity(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes mean sensitivity for all classes in the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.mean(
        np.array(confusion_matrix).diagonal() /
        np.sum(confusion_matrix, axis=0)
    )


def g_mean(confusion_matrix: Union[List[List[int]], np.ndarray]):
    """
    Computes g-mean metric for the confusion matrix.

    Parameters
    ----------
    confusion_matrix: ArrayLike
        The Numpy nxn array or nxn list representing confusion matrix
    """
    return np.float_power(np.prod(
        np.array(confusion_matrix).diagonal() /
        np.sum(confusion_matrix, axis=0)
    ), 1.0 / np.array(confusion_matrix).shape[0])


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

    Additionaly computes time of first inference step
    as `inferencetime_first` and average utilization
    of all cpus used as `session_utilization_cpus_percent_avg` key.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class

    Returns
    -------
    Dict :
        Gathered computed metrics
    """
    computed_metrics = {}

    def compute_metrics(metric_name: str, metric_value: Optional[Dict] = None):
        """
        Evaluates and saves metric in `operations` dictionary.

        Parameters
        ----------
        metric_name : str
            Name that is used to save matric evaluation
        metric_value : Optional[Dict]
            Values that are used to evaluate the metric
            If it is none then `measurementsdata[metric_name]` is used.
        """
        if not metric_value:
            metric_value = measurementsdata[metric_name]
        operations = {
            'mean': np.mean,
            'std': np.std,
            'median': np.median,
        }
        for op_name, op in operations.items():
            computed_metrics[f'{metric_name}_{op_name}'] = op(metric_value)

    # inferencetime
    inference_step = None
    if 'target_inference_step' in measurementsdata:
        inference_step = 'target_inference_step'
    elif 'protocol_inference_step' in measurementsdata:
        inference_step = 'protocol_inference_step'

    if inference_step:
        compute_metrics('inferencetime', measurementsdata[inference_step])

    # mem_percent
    if 'session_utilization_mem_percent' in measurementsdata:
        compute_metrics(
            'session_utilization_mem_percent',
            measurementsdata['session_utilization_mem_percent']
        )

    # cpus_percent
    if 'session_utilization_cpus_percent' in measurementsdata:
        cpus_percent_avg = [
            np.mean(cpus) for cpus in
            measurementsdata['session_utilization_cpus_percent']
        ]
        computed_metrics['session_utilization_cpus_percent_avg'] = cpus_percent_avg  # noqa: E501
        compute_metrics('session_utilization_cpus_percent_avg', cpus_percent_avg)  # noqa: E501

    # gpu_mem
    if 'session_utilization_gpu_mem_utilization' in measurementsdata:
        compute_metrics('session_utilization_gpu_mem_utilization')

    # gpu
    if 'session_utilization_gpu_utilization' in measurementsdata:
        compute_metrics('session_utilization_gpu_utilization')

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
        Statistics from the Measurements class

    Returns
    -------
    Dict :
        Gathered computed metrics
    """

    if 'eval_confusion_matrix' in measurementsdata:
        return {
            'accuracy': accuracy(measurementsdata['eval_confusion_matrix']),
            'top_5_accuracy':
                measurementsdata['top_5_count'] / measurementsdata['total'],
            'mean_precision':
                mean_precision(measurementsdata['eval_confusion_matrix']),
            'mean_sensitivity':
                mean_sensitivity(measurementsdata['eval_confusion_matrix']),
            'g_mean': g_mean(measurementsdata['eval_confusion_matrix']),
        }
    return {}


def compute_detection_metrics(measurementsdata: Dict[str, List]) -> Dict:
    """
    Computes detection metrics based on `measurementsdata` argument.
    If there is no detection metrics returns an empty dictionary.

    Computes mAP values.

    Parameters
    ----------
    measurementsdata : Dict[str, List]
        Statistics from the Measurements class

    Returns
    -------
    Dict :
        Gathered computed metrics
    """
    from kenning.datasets.helpers.detection_and_segmentation import \
        compute_map_per_threshold

    if 'eval_gtcount' in measurementsdata:
        return {
            'mAP': compute_map_per_threshold(measurementsdata, [0.0])[0]
        }
    return {}
