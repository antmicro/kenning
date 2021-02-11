import numpy as np
from collections import defaultdict

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
