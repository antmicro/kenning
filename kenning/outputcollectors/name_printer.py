"""
A small, very basic OutputCollector-derived class used to test
handling of multiple OutputCollectors in inference_runner scenario
"""
from kenning.core.outputcollector import OutputCollector
from kenning.datasets.open_images_dataset import DectObject
from typing import Any


class NamePrinter(OutputCollector):
    def __init__(self):
        self.frame_counter = 0
        super().__init__()

    def detach_from_output(self):
        pass

    def should_close(self):
        return True

    def process_output(self, i: Any, o: DectObject):
        o = o[0]
        print("Frame", self.frame_counter, end=": ")
        self.frame_counter += 1
        for x in o:
            print(x.clsname, end=" ")
        print()
