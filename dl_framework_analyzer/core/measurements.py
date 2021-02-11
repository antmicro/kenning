import numpy as np
from collections import defaultdict
import time
from ..utils import logger

logger = logger.get_logger()


class Measurements(object):
    def __init__(self):
        self.data = defaultdict(list)

    def __iadd__(self, other):
        assert isinstance(other, dict) or isinstance(other, Measurements)
        if isinstance(other, Measurements):
            for k, v in other.data.items():
                self.data[k] += other.data[k]
        else:
            for k, v in other.items():
                self.data[k] += other[k]
        return self

def statistics(measurementname):
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
