#!/bin/bash

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3 \
    kenning.compilers.tvm.TVMCompiler \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.datasets.open_images_dataset.OpenImagesDatasetV6 \
    ./build/jetson-agx-xavier-tvm-darknet.json \
    --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
    --model-path ./kenning/resources/models/detection/yolov3.weights \
    --model-framework darknet \
    --target "cuda -libs=cudnn,cublas -arch=sm_72" \
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
