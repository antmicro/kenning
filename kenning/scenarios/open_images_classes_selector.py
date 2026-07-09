# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A tool for generating a file with class names for Open Images Dataset V6.

Usage:
uv run kenning/scenarios/open_images_classes_selector.py \
    build/wanted_classes build/OpenImagesClasses.csv

where build/wanted_classes is just a file with class names that you want:
```
cat
dog
car
```

and build/OpenImagesClasses.csv is a csv file ready to be used for example
in fine-tuining a model on OpenImagesDataset.
"""

import argparse
from pathlib import Path

from kenning.datasets.open_images_dataset import OpenImagesDatasetV6


def main():  # noqa: D103
    # print(f"path to openimages classes {}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "detectorclasses",
        help="Path to a file with desired class names",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Output path containing csv file with desired classes, ready to be used for OpenImagesDatasetV6",  # noqa: E501
        nargs="?",
        const="output.csv",
        default="output.csv",
        type=Path,
    )

    args = parser.parse_args()

    with open(args.detectorclasses, "r") as classnames:
        desirednames = classnames.read().split("\n")[:-1]

    open_images_path = OpenImagesDatasetV6.resources["class_names"]

    with open(open_images_path, "r") as classnames:
        v6names = classnames.read().split("\n")[:-1]

    clslst = []
    notfound = []
    for clsname in desirednames:
        anychoice = False
        for v6cls in v6names:
            v6entry = v6cls.split(",")
            if clsname.lower() == v6entry[1].lower():
                print(f"{clsname} => {v6cls}")
                clslst.append((clsname, v6entry[0], v6entry[1]))
                anychoice = True
        if not anychoice:
            notfound.append(clsname)
    for entry in notfound:
        print(f"{entry} => not found")

    with open(args.output, "w") as out:
        for entry in clslst:
            out.write(f"{entry[1]},{entry[0]}\n")

    if len(notfound) > 0:
        return 1
    return 0


if __name__ == "__main__":
    main()
