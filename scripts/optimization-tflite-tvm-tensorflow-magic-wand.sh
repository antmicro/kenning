#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning fine-tune-optimizers \
    --json-cfg scripts/optimizationconfigs/tvm-tflite-tensorflow-magic-wand.json \
    --output build/tvm-tflite-pipeline-optimization-output.json \
    --verbosity INFO
