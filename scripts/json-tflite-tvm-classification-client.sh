#!/bin/bash

python3 -m kenning.scenarios.json_inference_tester \
    ./scripts/jsonconfigs/tflite-tvm-classification-client.json \
    ./build/tflite-tvm-classificationjson.json \
    --verbosity INFO
