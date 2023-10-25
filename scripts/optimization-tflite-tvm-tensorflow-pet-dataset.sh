#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning test \
    --json-cfg scripts/optimizationconfigs/tvm-tflite-tensorflow-pet-dataset.json \
    --output build/tvm-tflite-pipeline-optimization-output.json \
    --verbosity INFO
