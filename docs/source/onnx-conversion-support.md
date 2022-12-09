# ONNX support in deep learning frameworks

[ONNX](https://onnx.ai/) is an open format created to represent machine learning models.
The ONNX format is frequently updated with new operators that are present in state-of-the-art models.

Most of the frameworks for training, compiling and optimizing deep learning algorithms support ONNX format.
It allows conversion of models from one representation to another.

The `kenning.core.onnxconversion.ONNXConversion` class provides an API for writing compatibility tests between ONNX and deep learning frameworks.

It requires implementing:

* a method for ONNX model import for a given framework,
* a method for ONNX model export from a given framework,
* a list of models implemented in a given framework, where each model is exported to ONNX, and then imported back to the framework.

The `ONNXConversion` class implements a method for model conversion.
It catches exceptions and any issues in the import/export methods, and provides a report on conversion status per model.

See the [TensorFlowONNXConversion class](https://github.com/antmicro/kenning/blob/main/kenning/onnxconverters/tensorflow.py) for an example of API usage.

## ONNX support grid in deep learning frameworks

```{eval-rst}
.. include:: generated/onnx-support.rst
```
