#!/bin/bash

python -m kenning.scenarios.json_inference_tester \
    ./scripts/jsonconfigs/yolact-tvm-gpu-detection.json \
    ./build/yolact-tvm.json \
    --verbosity INFO

python -m kenning.scenarios.json_inference_tester \
    ./scripts/jsonconfigs/yolact-tflite-detection.json \
    ./build/yolact-tflite.json \
    --verbosity INFO

python -m kenning.scenarios.render_report \
    "YOLACT detection report" \
    build/yolact-report/report.rst \
    --root-dir build/yolact-report \
    --img-dir build/yolact-report/imgs \
    --report-types performance detection \
    --measurements build/yolact-tvm.json build/yolact-tflite.json
