#!/bin/bash

python3 -m kenning.scenarios.inference_server \
    kenning.runtimeprotocols.network.NetworkProtocol \
    kenning.runtimes.tflite.TFLiteRuntime \
    --host 0.0.0.0 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --delegates-list libedgetpu.so.1 \
    --verbosity INFO
