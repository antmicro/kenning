#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

python -m kenning optimize test \
    --json-cfg ./scripts/jsonconfigs/yolact-tvm-cpu-detection.json \
    --measurements ./build/yolact-tvm.json \
    --verbosity INFO

python -m kenning test \
    --json-cfg ./scripts/jsonconfigs/yolact-tflite-detection.json \
    --measurements ./build/yolact-tflite.json \
    --verbosity INFO

python -m kenning report \
    --report-path build/yolact-report/report.md \
    --report-name "YOLACT detection report" \
    --root-dir build/yolact-report \
    --img-dir build/yolact-report/imgs \
    --report-types performance detection \
    --measurements build/yolact-tvm.json build/yolact-tflite.json
