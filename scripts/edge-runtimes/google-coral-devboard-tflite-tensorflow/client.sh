#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/google-coral-devboard-tflite-tensorflow.json \
    --compiler-cls kenning.optimizers.tflite.TFLiteCompiler \
    --runtime-cls kenning.runtimes.tflite.TFLiteRuntime \
    --protocol-cls kenning.protocols.network.NetworkProtocol \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "edgetpu" \
    --compiled-model-path ./build/compiled-model.tflite \
    --inference-input-type int8 \
    --inference-output-type int8 \
    --host $1 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO
