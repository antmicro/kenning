#!/bin/bash

python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.compilers.tflite.TFLiteCompiler \
    kenning.runtimes.tflite.TFLiteRuntime \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/google-coral-devboard-tflite-tensorflow.json \
    --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
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
