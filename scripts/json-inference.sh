#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning test \
    --json-cfg "$1" \
    --measurements ./build/report-output.json \
    --verbosity INFO
