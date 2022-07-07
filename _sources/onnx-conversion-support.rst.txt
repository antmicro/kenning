ONNX support in deep learning frameworks
========================================

`ONNX <https://onnx.ai/>`_ is an open format created to represent machine learning models.
The ONNX format is frequently updated with new operators that are present in the state-of-the-art models.

Most of the frameworks for training, compiling and optimizing deep learning algorithms support ONNX format.
It allows conversion of models from one representation to another.

The ``kenning.core.onnxconversion.ONNXConversion`` class provides an API for writing compatibility tests between ONNX and deep learning frameworks.

It requires implementing:

* method for importing ONNX model for a given framework,
* method for exporting ONNX model from a given framework,
* list of models implemented in a given framework, where each model will be exported to ONNX, and then imported back to the framework.

The ``ONNXConversion`` class implements a method for converting the models.
It catches exceptions and any issues in the import/export methods, and provides the report on conversion status per model.

Look at the `TensorFlowONNXConversion class <https://github.com/antmicro/kenning/blob/master/kenning/onnxconverters/tensorflow.py>`_ for an example of API usage.

ONNX support grid in deep learning frameworks
---------------------------------------------

.. include:: generated/onnx-support.rst
