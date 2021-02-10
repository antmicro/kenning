import numpy as np
from collections import defaultdict

class Measurements(object):
    def __init__(self):
        self.data = defaultdict(list)

    def __iadd__(self, other):
        for k, v in other.data.items():
            self.data[k] += other.data[k]
        return self
