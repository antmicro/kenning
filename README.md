# Edge AI tester

Copyright (c) 2021 [Antmicro](https://www.antmicro.com)

This is a project for implementing testing pipelines for deploying deep learning models on edge devices.

## Deployment flow

The typical flow for the deep learning application deployment on edge is as follows:

* Prepare and analyse the dataset, data preprocessing and output postprocessing routines,
* Perform training (usually transfer learning) of the deep learning model, evaluate the model and improve until the model is sufficient for the task,
* Optimize the model, perform hardware-specific optimizations (e.g. quantization, pruning),
* Compile the model for a given target,
* Run the model on the target.

For nearly all of the above steps (training, optimization, compilation and runtime) there are different frameworks.
The cooperation between those frameworks differs and may provide different results.

This project:

* provides an API for wrapping model inference, compilation and running on edge devices,
* checks the compatibility between intermediate deployment steps and measures the quality and performance of the deep learning execution on target devices,
* returns JSON files with performance and quality metrics,
* converts above-mentioned JSON files to plots and RST files.

## Edge AI tester structure

The `edge_ai_tester` module consists of following submodules:

* `core` - provides interface APIs for datasets, models, compilers, runtimes and runtime protocols,
* `datasets` - provides implementations for datasets,
* `modelwrappers` - provides implementations for models,
* `compilers` - provides implementations for compilers of deep learning models,
* `runtimes` - provides implementations of runtime on target devices,
* `runtimeprotocols` - provides implementations for communication protocols between host and tested target,
* `onnxconverters` - provides ONNX conversions for a given framework along with a list of models to test the conversion on,
* `resources` - contains RST templates, trained models, and other resources,
* `scenarios` - contains executable scripts for running training, inference and benchmarks on target devices.
