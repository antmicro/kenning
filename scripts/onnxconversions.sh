#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

set -e

python -m kenning.scenarios.onnx_conversion \
    build/models-directory \
    build/onnx-support.md \
    --converters-list \
        kenning.onnxconverters.mxnet.MXNetONNXConversion \
        kenning.onnxconverters.pytorch.PyTorchONNXConversion \
        kenning.onnxconverters.tensorflow.TensorFlowONNXConversion

cat build/onnx-support.md
