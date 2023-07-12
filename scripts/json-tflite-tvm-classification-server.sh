#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_server \
    --json-cfg ./scripts/jsonconfigs/tflite-tvm-classification-server.json \
    --verbosity INFO
