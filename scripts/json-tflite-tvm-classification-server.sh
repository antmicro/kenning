#!/bin/bash

python3 -m kenning.scenarios.json_inference_server \
    ./scripts/jsonconfigs/tflite-tvm-classification-server.json \
    --verbosity INFO
