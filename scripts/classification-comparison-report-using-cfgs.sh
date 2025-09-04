#!/bin/bash

# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

python -m kenning optimize test --cfg scripts/configs/tensorflow-pet-dataset-mobilenet-tflite.yml --verbosity INFO

python -m kenning optimize test --cfg scripts/configs/tensorflow-pet-dataset-mobilenet-tvm.yml --verbosity INFO

python -m kenning report --cfg scripts/configs/pet-comparison-report.yml --verbosity INFO