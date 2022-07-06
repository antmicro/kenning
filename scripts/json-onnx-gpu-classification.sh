#!/bin/bash

python3 -m kenning.scenarios.json_inference_tester \
    ./scripts/jsonconfigs/onnx-gpu-classification.json \
    ./build/report-output.json \
    --verbosity INFO \
    --convert-to-onnx ./build/converted_model.onnx
