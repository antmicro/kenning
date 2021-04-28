#!/bin/bash

set -e

python3 -m edge_ai_tester.scenarios.onnx_conversion \
    build/models-directory \
    build/onnx-support.rst \
    --converters-list \
        edge_ai_tester.onnxconverters.mxnet.MXNetONNXConversion \
        edge_ai_tester.onnxconverters.pytorch.PyTorchONNXConversion \
        edge_ai_tester.onnxconverters.tensorflow.TensorFlowONNXConversion

cat build/onnx-support.rst
