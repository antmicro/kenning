# Object detection optimizers comparison

This example demonstrates how to optimize computer vision models and compare different optimizers.

To carry out our experiment we will be using [YOLACT](https://github.com/dbolya/yolact?tab=readme-ov-file), short for "You Only Look At CoefficientTs" which is a fully convolutional model for real-time instance segmentation.

We will also need [`OpenImagesDatasetV6`](https://github.com/antmicro/kenning/blob/main/kenning/datasets/open_images_dataset.py).

Then to test GPU performance we will compare TVM and ONNX optimizations.
* [ONNXCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/onnx.py) - wrapper for ONNX deep learning compiler.
* [TVMCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/tvm.py) - wraps TVM compilation.

## Setup

To install all necessary dependencies run:

```bash
pip install "kenning[object_detection,tvm,onnxruntime,reports] @ git+https://github.com/antmicro/kenning.git"
```

## Experiments

### TVM optimization

When configuring `TVMCompiler` we have to remember to set the right `model_framework` - for this model it's `onnx`.

Since we want to utilize CPU in this example

We set level of optimizations in the field `opt_level`.

Full TVM configuration:
```{literalinclude} ../scripts/jsonconfigs/yolact-tvm-cpu-detection.json save-as=yolact-tvm-cpu-detection.json
:language: json
:emphasize-lines: 26-28
```
To run it, use this command:
```bash
python -m kenning optimize test \
    --json-cfg yolact-tvm-cpu-detection.json \
    --measurements ./build/yolact-tvm.json \
    --verbosity INFO
```

### ONX optimization

We just need to remember to set `execution_providers` to `"CPUExecutionProvider"` in the configuration file.

To optimize YOLACT using ONNX we will use the following pipeline:
```{literalinclude} ../scripts/jsonconfigs/yolact-onnx-cpu-detection.json save-as=yolact-onnx-cpu-detection.json
:language: json
:emphasize-lines: 38-38
```

To run it, use this command:
```bash
python -m kenning optimize test \
    --json-cfg yolact-onnx-cpu-detection.json \
    --measurements ./build/yolact-onnx.json \
    --verbosity INFO
```

## Comparison

To create a comparison report comparing performance and model quality for the above optimizers, run:
```bash
python -m kenning report \
    --report-path build/yolact-report/report.md \
    --report-name "YOLACT detection report" \
    --root-dir build/yolact-report \
    --img-dir build/yolact-report/imgs \
    --report-types performance detection \
    --measurements build/yolact-tvm.json build/yolact-onnx.json
```

## GPU Experiments

{{uses_gpu}}

### TVM optimization

To run TVM optimizer on GPU we need to change the pipeline configuration file and specify that we will be using CUDA.

For optimizer we need to set `target` to `cuda -libs=cudnn,cublas`. \
For runtime we need to set `target_device_context` to `cuda`. \

```{literalinclude} ../scripts/jsonconfigs/yolact-tvm-gpu-detection.json save-as=yolact-tvm-gpu-detection.json
:language: json
:emphasize-lines: 27-28
:emphasize-lines: 42-42
```
To run it:
```bash
python -m kenning optimize test \
    --json-cfg yolact-tvm-gpu-detection.json \
    --measurements ./build/yolact-gpu-tvm.json \
    --verbosity INFO
```

### ONNX optimization

We just have to specify that we want to prioritize using CUDA as execution provider.
```{literalinclude} ../scripts/jsonconfigs/yolact-onnx-gpu-detection.json save-as=yolact-onnx-gpu-detection.json
:language: json
:emphasize-lines: 36-36
```

Run it using:
```bash
python -m kenning optimize test \
    --json-cfg yolact-onnx-gpu-detection.json \
    --measurements ./build/yolact-gpu-onnx.json \
    --verbosity INFO
```

### Comparison
To create a comparison report comparing performance and model quality for the above optimizers, run:
```bash
python -m kenning report \
    --report-path build/yolact-report/report.md \
    --report-name "YOLACT detection report" \
    --root-dir build/yolact-report \
    --img-dir build/yolact-report/imgs \
    --report-types performance detection \
    --measurements build/yolact-gpu-tvm.json build/yolact-gpu-onnx.json
```
