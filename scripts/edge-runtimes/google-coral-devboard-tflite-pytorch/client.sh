#!/bin/bash

python3 -m edge_ai_tester.scenarios.inference_tester \
    edge_ai_tester.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2 \
    edge_ai_tester.compilers.tflite.TFLiteCompiler \
    edge_ai_tester.runtimes.tflite.TFLiteRuntime \
    edge_ai_tester.datasets.pet_dataset.PetDataset \
    ./build/google-coral-devboard-tflite-pytorch.json \
    --protocol-cls edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
    --model-path ./edge_ai_tester/resources/models/classification/pytorch_pet_dataset_mobilenetv2.pth \
    --convert-to-onnx ./build/pytorch_pet_dataset_mobilenetv2.onnx \
    --model-framework onnx \
    --target "edgetpu" \
    --compiled-model-path ./build/compiled-model.tflite \
    --inference-input-type int8 \
    --inference-output-type int8 \
    --host $1 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --dataset-root ./build/pet-dataset/ \
    --image-memory-layout NCHW \
    --inference-batch-size 1 \
    --verbosity INFO
