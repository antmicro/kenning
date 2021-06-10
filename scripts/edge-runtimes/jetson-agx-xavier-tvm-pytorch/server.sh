#!/bin/bash

python3 -m kenning.scenarios.inference_server \
    kenning.runtimeprotocols.network.NetworkProtocol \
    kenning.runtimes.tvm.TVMRuntime \
    --host 0.0.0.0 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/nvidia/compiled-model.tar \
    --target-device-context cuda \
    --input-dtype float32 \
    --verbosity INFO
