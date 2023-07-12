#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_tester \
    --json-cfg ./scripts/jsonconfigs/tflite-tvm-classification-client.json \
    --measurements ./build/tflite-tvm-classificationjson.json \
    --verbosity INFO
