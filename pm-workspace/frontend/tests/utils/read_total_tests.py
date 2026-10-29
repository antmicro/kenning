#!/bin/bash

# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A python script used for getting total number of executed test
from playwright json results report.
"""

import json
import sys
from pathlib import Path


def main():
    """
    A main function of the program.
    It reads a first input argument,
    which is path to report summary
    in json format. Then it writes
    on the stdout total number
    of executed tests.
    """
    file_path = Path(sys.argv[1])

    with open(file_path, "r") as file:
        results = json.load(file)

        stats = results["stats"]
        expected = int(stats["expected"])
        skipped = int(stats["skipped"])
        unexpected = int(stats["unexpected"])
        flaky = int(stats["flaky"])

        total = expected + skipped + unexpected + flaky

        sys.stdout.write(str(total))


if __name__ == "__main__":
    main()
