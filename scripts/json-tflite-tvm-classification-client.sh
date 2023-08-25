#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning optimize test  \
    --json-cfg ./scripts/jsonconfigs/tflite-tvm-classification-client-server.json \
    --measurements ./build/tflite-tvm-classification.json \
    --verbosity INFO
