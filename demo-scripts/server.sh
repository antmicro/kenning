#!/bin/bash

python3 -m dl_framework_analyzer.scenarios.inference_server \
    dl_framework_analyzer.runtimeprotocols.network.NetworkProtocol \
    dl_framework_analyzer.runtimes.tvm.TVMRuntime \
    --host 192.168.188.100 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/nvidia/compiled-model.tar \
    --target-device-context cuda \
    --input-dtype float32 \
    --verbosity INFO
