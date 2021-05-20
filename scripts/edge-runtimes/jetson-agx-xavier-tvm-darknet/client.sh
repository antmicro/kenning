#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_tester \
    edge_ai_tester.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3 \
    edge_ai_tester.compilers.tvm.TVMCompiler \
    edge_ai_tester.runtimes.tvm.TVMRuntime \
    edge_ai_tester.datasets.open_images_dataset.OpenImagesDatasetV6 \
    ./build/jetson-agx-xavier-tvm-darknet.json \
    --protocol-cls edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
    --model-path ./edge_ai_tester/resources/models/detection/yolov3.weights \
    --model-framework darknet \
    --target "nvidia/jetson-agx-xavier" \
    --target-host "llvm -mtriple=aarch64-linux-gnu" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --host $1 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/nvidia/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/open-images-dataset \
    --inference-batch-size 1 \
    --libdarknet-path ./libdarknet.so \
    --verbosity INFO
