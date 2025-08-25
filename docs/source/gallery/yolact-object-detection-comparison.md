# Optimizing and comparing instance segmentation models

This example demonstrates how to optimize computer vision models and compare different optimizers.

Base model used for demonstration is going to be [YOLACT](https://github.com/dbolya/yolact?tab=readme-ov-file), short for "You Only Look At CoefficientTs" which is a fully convolutional model for real-time instance segmentation.

We will also need dataset for evaluation purposes - in this case it is going to be [`OpenImagesDatasetV6`](https://github.com/antmicro/kenning/blob/main/kenning/datasets/open_images_dataset.py).

Model will be deployed on CPU and GPU using following Kenning compilers:

* [ONNXCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/onnx.py) - wrapper for optimizing and converting models to format compliant for [ONNX Runtime](https://github.com/microsoft/onnxruntime).
* [TVMCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/tvm.py) - wrapper for [TVM deep neural network compiler](https://github.com/apache/tvm).

## Setup

To install all necessary dependencies run:

```bash
pip install "kenning[object_detection,tvm,onnxruntime,reports] @ git+https://github.com/antmicro/kenning.git"
```

## Experiments on CPU

### TVM optimization

The scenario for TVM compilation may look as follows:

```{literalinclude} ../scripts/jsonconfigs/yolact-tvm-cpu-detection.json save-as=yolact-tvm-cpu-detection.json
:language: json
:lineno-start: 1
```

In this scenario:

* `model_path` points to a location of the YOLACT model in ONNX format.
  It can be either local file or a remote URL.
  `kenning://` is a special schema for Kenning's demonstration models.
* `dataset` tells to use Open Images dataset.
  The model will be downloaded to `./build/OpenImagesDatasetV6`.
  The `task` field allows to specify whether the dataset is used for instance segmentation or object detection.
* `optimizers` contains only one element - `TVMCompiler`.
  In there we specify input model framework (`onnx`), and tell to use `llvm` target with `opt_level` equal to 3 (applying all possible optimizations not directly affecting model's output).
* `runtime` tells Kenning to use `TVMRuntime` for model execution, on CPU target.

To optimize and test the defined scenario, run:

```bash
python -m kenning optimize test \
    --cfg yolact-tvm-cpu-detection.json \
    --measurements ./build/yolact-tvm.json \
    --verbosity INFO
```

### ONNX Runtime execution of model

Switching to a different runtime is a matter of changing several lines in the scenario as shown below:

```{literalinclude} ../scripts/jsonconfigs/yolact-onnx-cpu-detection.json save-as=yolact-onnx-cpu-detection.json
:language: json
:lineno-start: 1
:emphasize-lines: 22-40
```

`ONNXRuntime` in this specific scenario acts as a passthrough for an existing ONNX model.
The main change in runtime lies in `ONNXRuntime`, where `execution_providers` holds a list of possible layer executors starting from the most preferred one.
For CPU-only execution `CPUExecutionProvider` is required.

This scenario can be executed with:

```bash
python -m kenning optimize test \
    --cfg yolact-onnx-cpu-detection.json \
    --measurements ./build/yolact-onnx.json \
    --verbosity INFO
```

## Comparison

:::{figure} ./img/yolact-cpu-mean-performance-comparison.*
---
name: yolact-cpu-mean-performance-comparison
alt: Mean performance comparison
align: center
---

Sample comparison plot demonstrating model size, speed and quality for two YOLACT Optimizers
:::

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

:::{admonition} Open for instructions on building TVM with CUDA enabled
:collapsible:

To run TVM on GPU you need to build it with necessary libraries included - follow [Building from source in TVM documentation](https://tvm.apache.org/docs/install/from_source.html).

On Linux-based distributions:
1. Uninstall previously installed TVM version (supports only CPU):
   ```bash
   pip uninstall -y apache-tvm
   ```
2. Ensure installation of:
   * GCC >= 7.1 /  Clang >= 5.0
   * CMAKE >= 3.24
   * LLVM >= 15
   * Git
   * Python >= 3.8
3. Clone tvm repository:
   ```bash test-skip
   git clone --recursive https://github.com/apache/tvm -b v0.14.0 tvm
   ```
4. Configure the build by setting the necessary flags in `config.cmake`:
   ```bash test-skip
   cd tvm && mkdir build && cd build
   cp ../cmake/config.cmake .
   echo "set(USE_CUBLAS ON)">> config.cmake
   echo "set(USE_CUDA ON)"  >> config.cmake
   echo "set(USE_CUDNN ON)" >> config.cmake
   echo "set(USE_MICRO ON)" >> config.cmake
   echo "set(USE_LLVM ON)"  >> config.cmake
   ```
5. Build TVM with the following command:
   ```bash test-skip
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/ .. && make -j $(nproc) && make install && ldconfig
   cd ../..
   ```
6. Add built libraries to `TVM_LIBRARY_PATH`:
   ```bash
   export TVM_LIBRARY_PATH=/usr/lib/
   ```
7. Install new TVM version using `pip`:
   ```bash test-skip
   pip install ./tvm/python
   ```

CUDA runtime is very strict about `gcc` versions so if you get an error `unsupported GNU version! gcc versions later than 11 are not supported!`, then set the necessary flag:
```bash
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'
```
:::

To run TVM optimization of YOLACT for GPU, the previous TVM scenario requires changes in:

* `TVMCompiler` - need to set `target` to `cuda -libs=cudnn,cublas`
* `TVMRuntime` - need to configure `target_device_context` to `cuda`

```{literalinclude} ../scripts/jsonconfigs/yolact-tvm-gpu-detection.json save-as=yolact-tvm-gpu-detection.json
:language: json
:lineno-start: 1
:emphasize-lines: 27-28,42-42
```

To run it:

```bash
python -m kenning optimize test \
    --cfg yolact-tvm-gpu-detection.json \
    --measurements ./build/yolact-gpu-tvm.json \
    --verbosity INFO
```

### ONNX optimization

:::{admonition} Open for instructions on installing ONNX Runtime with CUDA enabled
:collapsible:

If you want to run ONNX on GPU you need to install `onnxruntime-gpu`:

* Uninstall CPU-only variant with:
  ```bash
  pip uninstall -y onnxruntime
  ```
* Install GPU-enabled ONNX Runtime:
  ```bash
  pip install "kenning[onnxruntime_gpu] @ git+https://github.com/antmicro/kenning.git"
  ```
* Add necessary CUDA libraries to path:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  export PATH=/usr/local/cuda/bin:$PATH
  ```
:::

The only difference compared to CPU-only execution using ONNX Runtime lies in adding new executor in the `runtime`:

```{literalinclude} ../scripts/jsonconfigs/yolact-onnx-gpu-detection.json save-as=yolact-onnx-gpu-detection.json
:language: json
:lineno-start: 1
:emphasize-lines: 38
```

Run the scenario as follows:
```bash
python -m kenning optimize test \
    --cfg yolact-onnx-gpu-detection.json \
    --measurements ./build/yolact-gpu-onnx.json \
    --verbosity INFO
```

### Comparison of GPU runtimes

:::{figure} ./img/yolact-gpu-mean-performance-comparison.*
---
name: yolact-gpu-mean-performance-comparison
alt: Mean performance comparison
align: center
---

Model size, speed and quality comparison for two YOLACT Optimizers running on CUDA GPU
:::

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
