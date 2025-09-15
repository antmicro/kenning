#!/bin/bash

# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

python -m kenning optimize test report \
    --cfg scripts/configs/yolact-onnx-detection.yml \
    --verbosity INFO