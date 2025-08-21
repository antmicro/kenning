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

Brief summary of theses two models' performance:

:::{figure} ./img/yolact-cpu-mean-performance-comparison.*
---
name: yolact-cpu-mean-performance-comparison
alt: Mean performance comparison
align: center
---

Model size, speed and quality comparison for two YOLACT Optimizers
:::

| Model name             | Inference time [s]   | CPU usage [%]    | Memory usage [%]    |
|------------------------|----------------------|------------------|---------------------|
| build.yolact-tvm.json  | 1.077                | 52.910           | 26.487              |
| build.yolact-onnx.json | 0.226                | 52.087           | 23.545              |

## GPU Experiments

{{uses_gpu}}

### TVM optimization

To run TVM on GPU you will instead need to follow installation guide:
https://tvm.apache.org/docs/install/from_source.html in order to install TVM that is compatible with CUDA.

---

If you are using Linux follow these steps:
  1. Ensure installation of:
    - GCC >= 7.1 /  Clang >= 5.0
    - CMAKE >= 3.24
    - LLVM >= 15
    - Git
    - Python >= 3.8

  2. Clone tvm repository:
```bash test-skip
git clone --recursive https://github.com/apache/tvm tvm
git checkout -b v0.14.0
```

  3. Configure the build by setting the necessary flags in `config.cmake`:
```bash test-skip
cd tvm && mkdir build && cd build
cp ../cmake/config.cmake .

echo "set(USE_CUBLAS ON)">> config.cmake
echo "set(USE_CUDA ON)"  >> config.cmake
echo "set(USE_CUDNN ON)" >> config.cmake
echo "set(USE_MICRO ON)" >> config.cmake
echo "set(USE_LLVM ON)"  >> config.cmake
```

  4. Build it using:
```bash test-skip
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/ .. && cmake --build . --parallel $(nproc) && make install && ldconfig
```

  5. Add created libraries to `TVM_LIBRARY_PATH`:
```bash
export TVM_LIBRARY_PATH=/usr/lib/
```

  6. Remember to uninstall previously installed tvm version:
```bash
pip uninstall -y apache-tvm
```

  7. Finally install new version using `pip`:
```bash
pip install /tvm/python
```

---

  CUDA runtime is very strict about `gcc` versions so if you get an error: `unsupported GNU version! gcc versions later than 11 are not supported!`,
  then just set the necessary flag:
```bash
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'
```

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

If you want to run ONNX on GPU need to install onnxruntime that supports gpu usage:

First uninstall the cpu version of `onnxruntime`:
```bash
pip uninstall -y onnxruntime
```

Then install `onnxruntime` that uses gpu:
```bash
pip install "kenning[onnxruntime_gpu] @ git+https://github.com/antmicro/kenning.git"
```

And remember to add necessary CUDA libraries to path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```


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

A short rundown of report:

:::{figure} ./img/yolact-gpu-mean-performance-comparison.*
---
name: yolact-gpu-mean-performance-comparison
alt: Mean performance comparison
align: center
---

Model size, speed and quality comparison for two YOLACT Optimizers running on CUDA GPU
:::

| Model name               | Inference time [s]     | CPU usage [%]      | Memory usage [%]      |
|--------------------------|------------------------|--------------------|-----------------------|
| build.yolact-tvm.json    | 0.0113                 | 37.400             | 8.405                 |
| build.yolact-onnx.json   | 0.0344                 | 30.867             | 7.784                 |
