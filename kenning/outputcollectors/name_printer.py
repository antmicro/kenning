# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A small, very basic OutputCollector-derived class used to test
handling of multiple OutputCollectors in inference_runner scenario.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

from kenning.core.outputcollector import OutputCollector
from kenning.datasets.helpers.detection_and_segmentation import DetectObject


class NamePrinter(OutputCollector):
    """
    Prints detected classes for classification task.
    """

    arguments_structure = {
        "print_type": {
            "argparse_name": "--print-name",
            "description": "What is the type of model that will input data to "
            "the NamePrinter",
            "enum": ["detector", "classificator"],
            "default": "detector",
        },
        "classification_class_names": {
            "argparse_name": "--classification-class-names",
            "description": "File with class names used to identify the output "
            "from classification models",
            "type": Path,
        },
    }

    def __init__(
        self,
        print_type: str = "detector",
        file_path: Path = None,
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        self.frame_counter = 0
        self.print_type = print_type
        self.classnames = []
        self.file_path = file_path
        if file_path:
            with open(file_path, "r") as f:
                for line in f:
                    self.classnames.append(line.strip())
        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    def detach_from_output(self):
        pass

    def should_close(self):
        return False

    def process_output(self, i: Any, o: Union[DetectObject, np.array]):
        print("Frame", self.frame_counter, end=": ")
        o = o[0]
        if self.print_type == "detector":
            for x in o:
                print(x.clsname, end=" ")
            print()
        elif self.print_type == "classificator":
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
                    "{}: {:.2f}".format(tuples[i][1], float(tuples[i][0])),
                    end=", ",
                )
            print()
        self.frame_counter += 1
