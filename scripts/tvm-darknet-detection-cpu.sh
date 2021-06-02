#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_tester \
    edge_ai_tester.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3 \
    edge_ai_tester.compilers.tvm.TVMCompiler \
    edge_ai_tester.runtimes.tvm.TVMRuntime \
    edge_ai_tester.datasets.open_images_dataset.OpenImagesDatasetV6 \
    ./build/cpu-tvm-darknet.json \
    --model-path ./edge_ai_tester/resources/models/detection/yolov3.weights \
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
