#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/pet-dataset-tvm.json \
    --compiler-cls kenning.optimizers.tvm.TVMCompiler \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target cuda \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --download-dataset \
    --verbosity INFO

python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/pet-dataset-tflite.json \
    --compiler-cls kenning.optimizers.tflite.TFLiteCompiler \
    --runtime-cls kenning.runtimes.tflite.TFLiteRuntime \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --compiled-model-path ./build/compiled-model.tflite \
    --save-model-path ./build/compiled-model.tflite \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO

python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/pet-dataset-iree.json \
    --compiler-cls kenning.optimizers.iree.IREECompiler \
    --runtime-cls kenning.runtimes.iree.IREERuntime \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --backend vulkan \
    --compiled-model-path ./build/compiled-model.vmbf \
    --save-model-path ./build/compiled-model.vmbf \
    --driver vulkan \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO

python -m kenning.scenarios.render_report \
    --report-path build/classification-report/report.md \
    --report-name "Classification comparison on Pet Dataset" \
    --root-dir build/classification-report \
    --img-dir build/classification-report/imgs \
    --report-types performance classification \
    --measurements build/pet-dataset-tvm.json build/pet-dataset-tflite.json build/pet-dataset-iree.json
