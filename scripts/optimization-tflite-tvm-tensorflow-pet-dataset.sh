#!/bin/bash

python3 -m kenning.scenarios.optimization_runner \
    scripts/optimizationconfigs/tvm-tflite-tensorflow-pet-dataset.json \
    build/optimizations/output.json \
    --verbosity INFO
