#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning optimize test \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/local-cpu-tvm-tensorflow-classification.json \
    --compiler-cls kenning.optimizers.tvm.TVMCompiler \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "llvm" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cpu \
    --dataset-root ./build/PetDataset/ \
    --inference-batch-size 1 \
    --verbosity INFO

python -m kenning report \
    --report-path docs/source/generated/local-cpu-tvm-tensorflow-classification.md \
    --report-name "Pet Dataset classification using TVM-compiled TensorFlow model" \
    --root-dir docs/source/ \
    --img-dir docs/source/generated/img \
    --report-types performance classification \
    --measurements build/local-cpu-tvm-tensorflow-classification.json \
    --smaller-header
