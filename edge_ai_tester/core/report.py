"""
Functions to generate RST reports from templates and Measurements objects.
"""

from jinja2 import Template
from pathlib import Path
from typing import Dict, List, Union
import numpy as np


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


def create_report_from_measurements(
        template: Path,
        measurementsdata: Dict[str, List]):
    """
    Creates a report from template and measurements data.

    It additionaly provides methods for quality metrics.

    Parameters
    ----------
    template : Path
        Path to the Jinja template
    measurementsdata : Dict[str, List]
        dictionary describing measurements taken during benchmark
    """
    with open(template, 'r') as resourcetemplatefile:
        resourcetemplate = resourcetemplatefile.read()
        tm = Template(resourcetemplate)
        tm.globals['mean'] = np.mean
        tm.globals['std'] = np.std
        tm.globals['median'] = np.median
        tm.globals['accuracy'] = accuracy
        tm.globals['mean_precision'] = mean_precision
        tm.globals['mean_sensitivity'] = mean_sensitivity
        tm.globals['g_mean'] = g_mean

        content = tm.render(
            data=measurementsdata
        )

        return content
