#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.json_inference_tester \
    ./scripts/jsonconfigs/tflite-tvm-classification-client.json \
    ./build/tflite-tvm-classificationjson.json \
    --verbosity INFO
