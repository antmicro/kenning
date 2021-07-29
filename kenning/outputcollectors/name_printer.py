"""
A small, very basic OutputCollector-derived class used to test
handling of multiple OutputCollectors in inference_runner scenario
"""
from kenning.core.outputcollector import OutputCollector
from kenning.datasets.open_images_dataset import DectObject
from typing import Any, Union
import numpy as np
from pathlib import Path


class NamePrinter(OutputCollector):
    def __init__(self, print_type: str = "detector", file_path: Path = None):
        self.frame_counter = 0
        self.print_type = print_type
        self.classnames = []
        if file_path:
            with open(file_path, 'r') as f:
                for line in f:
                    self.classnames.append(line.strip())
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
        group.add_argument(
            '--classification-class-names',
            help='File with class names used to identify the output from classification models',  # noqa: E501
            type=Path
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(args.print_type, args.classification_class_names)

    def detach_from_output(self):
        pass

    def should_close(self):
        return False

    def process_output(self, i: Any, o: Union[DectObject, np.array]):
        print("Frame", self.frame_counter, end=": ")
        o = o[0]
        if self.print_type == 'detector':
            for x in o:
                print(x.clsname, end=" ")
            print()
        elif self.print_type == 'classificator':
            tuples = []
            if self.classnames:
                for i, j in zip(o, self.classnames):
                    tuples.append((i, j))
            else:
                it = 0
                for i in o:
                    tuples.append((i, "object {}".format(it)))
                    it += 1
            tuples.sort(key=lambda x: x[0], reverse=True)
            for i in range(min(5, len(tuples))):
                print(
                    "{}: {:.2f}".format(
                        tuples[i][1],
                        float(tuples[i][0])
                    ),
                    end=', '
                )
            print()
        self.frame_counter += 1
