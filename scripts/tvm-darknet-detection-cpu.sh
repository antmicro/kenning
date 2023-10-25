#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning optimize test \
    --modelwrapper-cls kenning.modelwrappers.object_detection.darknet_coco.TVMDarknetCOCOYOLOV3 \
    --dataset-cls kenning.datasets.open_images_dataset.OpenImagesDatasetV6 \
    --measurements ./build/cpu-tvm-darknet.json \
    --compiler-cls kenning.optimizers.tvm.TVMCompiler \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --model-path kenning:///models/object_detection/yolov3.weights \
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
