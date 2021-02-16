"""
Module containing decorators for benchmark data gathering.
"""

from typing import List, Dict, Union, Any
from collections import defaultdict
import time
from dl_framework_analyzer.utils import logger
import psutil
# TODO add checking if NVIDIA is present. It may not be neccessary
from pynvml.smi import nvidia_smi
from threading import Thread

from functools import wraps

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

    def add_measurement(self, measurementtype: str, value: Any):
        """
        Add new value to a given measurement type.

        Parameters
        ----------
        measurementtype : str
            the measurement type to be updated
        value : Any
            the value to add
        """
        assert isinstance(measurementtype, str)
        self.data[measurementtype].append(value)

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


class MeasurementsCollector(object):
    measurements = Measurements()


def timemeasurements(measurementname: str):
    """
    Decorator for measuring time of the function.

    The duration is given in nanoseconds.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    """
    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            start = time.perf_counter_ns()
            returnvalue = function(*args)
            duration = time.perf_counter_ns() - start
            logger.debug(
                f'{function.__name__} time:  {duration / 1000000} ms'
            )
            MeasurementsCollector.measurements += {
                measurementname: [duration],
                f'{measurementname}_timestamp': [time.perf_counter_ns()]
            }
            return returnvalue
        return statistics_wrapper
    return statistics_decorator


class SystemStatsCollector(Thread):
    def __init__(self, prefix):
        Thread.__init__(self)
        self.measurements = Measurements()
        self.running = True
        self.prefix = prefix
        self.nvidia_smi = nvidia_smi.getInstance()

    def get_measurements(self):
        return self.measurements

    def run(self):
        while self.running:
            cpus = psutil.cpu_percent(interval=0.5, percpu=True)
            mem = psutil.virtual_memory()
            gpu = self.nvidia_smi.DeviceQuery(
                'memory.free, memory.total, utilization.gpu'
            )
            memtot = float(gpu['gpu'][0]['fb_memory_usage']['total'])
            memfree = float(gpu['gpu'][0]['fb_memory_usage']['free'])
            gpumemutilization = (memtot - memfree) / memtot * 100.0
            gpuutilization = float(gpu['gpu'][0]['utilization']['gpu_util'])
            self.measurements += {
                f'{self.prefix}_cpus_percent': [cpus],
                f'{self.prefix}_mem_percent': [mem.percent],
                f'{self.prefix}_gpu_utilization': [gpuutilization],
                f'{self.prefix}_gpu_mem_utilization': [gpumemutilization],
                f'{self.prefix}_timestamp': [time.perf_counter_ns()],
            }

    def stop(self):
        self.running = False


def systemstatsmeasurements(measurementname: str):
    """
    Decorator for measuring memory usage of the function.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    """

    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            measurementsthread = SystemStatsCollector(measurementname)
            measurementsthread.start()
            returnvalue = function(*args)
            measurementsthread.stop()
            measurementsthread.join()
            MeasurementsCollector.measurements += \
                measurementsthread.get_measurements()
            return returnvalue
        return statistics_wrapper
    return statistics_decorator
