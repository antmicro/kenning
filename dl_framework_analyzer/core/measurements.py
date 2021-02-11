"""
Module containing decorators for benchmark data gathering.
"""

from typing import List, Dict, Union
from collections import defaultdict
import time
from ..utils import logger

logger = logger.get_logger()


class Measurements(object):
    """
    Stores benchmark measurements for later processing.

    This is a dict-like object that wraps all processing results for later
    raport generation.

    The dictionary in Measurements has measurement type as a key, and list of
    values for given measurement type.

    Attributes
    ----------
    data : dict
        Dictionary storing lists of values
    """
    def __init__(self):
        self.data = defaultdict(list)

    def __iadd__(self, other: Union[Dict, 'Measurements']) -> 'Measurements':
        self.update_measurements(other)
        return self

    def update_measurements(self, other: Union[Dict, 'Measurements']):
        """
        Adds measurements of types given in the other object.

        It requires another Measurements object, or a dictionary that has
        string keys and values that are lists of values. The lists from the
        other object are appended to the lists in this object.

        Parameters
        ----------
        other : Union[Dict, 'Measurements']
            A dictionary or another Measurements object that contains lists in
            every entry.
        """
        assert isinstance(other, dict) or isinstance(other, Measurements)
        if isinstance(other, Measurements):
            for k, v in other.data.items():
                self.data[k] += other.data[k]
        else:
            for k, v in other.items():
                self.data[k] += other[k]

    def add_measurements(self, measurementtype: str, valueslist: List):
        """
        Adds new values to a given measurement type.

        Parameters
        ----------
        measurementtype : str
            the measurement type to be updated
        valueslist : List
            the list of values to add
        """
        assert isinstance(valueslist, list)
        assert isinstance(measurementtype, str)
        self.data[measurementtype] += valueslist

    def get_values(self, measurementtype: str) -> List:
        """
        Returns list of values for a given measurement type.

        Parameters
        ----------
        measurementtype : str
            The name of the measurement type

        Returns
        -------
        List : list of values for a given measurement type
        """
        return self.data[measurementtype]


def statistics(measurementname: str):
    """
    Decorator for measuring time of the function.

    The function wrapped by the decorator must return the Measurements object,
    since the decorator will append timing data to the returned Measurements
    object.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    """
    def statistics_decorator(function):
        def statistics_wrapper(*args):
            start = time.perf_counter_ns()
            measurements = function(*args)
            duration = time.perf_counter_ns() - start
            logger.debug(f'{function.__name__} time:  {duration / 1000000} ms')
            measurements += {'processing_time': [duration]}
            return measurements
        return statistics_wrapper
    return statistics_decorator
