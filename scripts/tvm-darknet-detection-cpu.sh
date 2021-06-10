#!/bin/bash

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3 \
    kenning.compilers.tvm.TVMCompiler \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.datasets.open_images_dataset.OpenImagesDatasetV6 \
    ./build/cpu-tvm-darknet.json \
    --model-path ./kenning/resources/models/detection/yolov3.weights \
    --model-framework darknet \
    --target "llvm" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cpu \
    --dataset-root ./build/open-images-dataset \
    --inference-batch-size 1 \
    --libdarknet-path ./libdarknet.so \
    --verbosity INFO
