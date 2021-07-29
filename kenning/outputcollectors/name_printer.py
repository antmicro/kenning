"""
A small, very basic OutputCollector-derived class used to test
handling of multiple OutputCollectors in inference_runner scenario
"""
from kenning.core.outputcollector import OutputCollector
from kenning.datasets.open_images_dataset import DectObject
from typing import Any, Union
from torch import FloatTensor


class NamePrinter(OutputCollector):
    def __init__(self, print_type: str = "detector"):
        self.frame_counter = 0
        self.print_type = print_type
        super().__init__()

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--print-type',
            help='What is the type of model that will input data to the NamePrinter',  # noqa: E501
            choices=['detector', 'classificator'],
            default='detector'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(args.print_type)

    def detach_from_output(self):
        pass

    def should_close(self):
        return False

    def process_output(self, i: Any, o: Union[DectObject, FloatTensor]):
        print("Frame", self.frame_counter, end=": ")
        if self.print_type == 'detector':
            o = o[0]
            self.frame_counter += 1
            for x in o:
                print(x.clsname, end=" ")
            print()
        elif self.print_type == 'classificator':
            for i in o:
                print(float(i), end=', ')
            print()
