# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A tool for generating mapping from provided class names to
Open Images V6 names.
"""

import argparse
import json
from pathlib import Path


def main():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "v6classes",
        help="Path to the class-descriptions-boxable.csv for Open Images V6",
        type=Path,
    )
    parser.add_argument(
        "detectorclasses",
        help="Path to the file with class names recognizable by detector",
        type=Path,
    )
    parser.add_argument(
        "output",
        help="Output path containing file with the class name, V6 class id and V6 original name",  # noqa: E501
        type=Path,
    )
    parser.add_argument(
        "--remapper",
        help="Path to the JSON file with mapping from detector class name to v6 class name (lower case)",  # noqa: E501
        type=Path,
    )
    parser.add_argument(
        "--use-v6-class-names",
        help="Use Open Images class names instead of class names from detectorclasses file",  # noqa: E501
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.v6classes, "r") as classnames:
        v6names = classnames.read().split("\n")[:-1]

    with open(args.detectorclasses, "r") as classnames:
        coconames = classnames.read().split("\n")[:-1]

    with open(args.remapper, "r") as remapstruct:
        remapper = json.load(remapstruct)

    clslst = []
    notfound = []
    for clsname in coconames:
        anychoice = False
        for v6cls in v6names:
            v6entry = v6cls.split(",")
            if clsname.lower() == v6entry[1].lower():
                print(f"{clsname} => {v6cls}")
                clslst.append((clsname, v6entry[0], v6entry[1]))
                anychoice = True
            elif (
                clsname in remapper and remapper[clsname] == v6entry[1].lower()
            ):
                print(f"{clsname} => {v6cls}")
                clslst.append((clsname, v6entry[0], v6entry[1]))
                anychoice = True
        if not anychoice:
            notfound.append(clsname)
    for entry in notfound:
        print(f"{entry} => not found")

    with open(args.output, "w") as out:
        for entry in clslst:
            out.write(
                f"{entry[1]},{entry[2] if args.use_v6_class_names else entry[0]}\n"  # noqa: E501
            )

    if len(notfound) > 0:
        return 1
    return 0


if __name__ == "__main__":
    main()
