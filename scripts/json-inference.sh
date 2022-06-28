#!/bin/bash

python3 -m kenning.scenarios.json_inference_tester \
    "$1" \
    ./build/report-output.json \
    --verbosity INFO
