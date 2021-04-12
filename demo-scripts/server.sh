#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_server \
    edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
    edge_ai_tester.runtimes.tvm.TVMRuntime \
    --host 192.168.188.100 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/nvidia/compiled-model.tar \
    --target-device-context cuda \
    --input-dtype float32 \
    --verbosity INFO
