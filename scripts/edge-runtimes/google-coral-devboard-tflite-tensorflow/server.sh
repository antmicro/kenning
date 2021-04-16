#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_server \
    edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
    edge_ai_tester.runtimes.tflite.TFLiteRuntime \
    --host 0.0.0.0 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --delegates-list libedgetpu.so.1 \
    --verbosity INFO
