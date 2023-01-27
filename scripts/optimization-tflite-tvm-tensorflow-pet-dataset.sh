#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python3 -m kenning.scenarios.optimization_runner \
    scripts/optimizationconfigs/tvm-tflite-tensorflow-pet-dataset.json \
    build/tvm-tflite-pipeline-optimization-output.json \
    --verbosity INFO
