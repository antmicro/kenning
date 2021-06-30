Kenning API
===========

.. _api-overview:

API overview
------------

.. figure:: img/class-flow.png
   :name: class-flow
   :alt: Kenning core classes and interactions between them
   :align: center

   Kenning core classes and interactions between them.

.. _dataset-api:

Dataset
-------

Dataset-based classes are responsible for:

* preparing the dataset - downloading the dataset and preparing the data required to work with the dataset,
* preparing the inputs and outpus - loading images, creating output one-hot vectors (classification) or lists of bounding boxes (detection) or masks (segmentation),
* evaluating the model.

The Dataset objects are used by:

* :ref:`modelwrapper-api` - for training purposes and model evaluation,
* :ref:`modelcompiler-api` - can be used i.e. for extracting calibration dataset for quantization purposes,
* :ref:`runtime-api` - is used for evaluating the model on target hardware.

The example subclasess:

* `PetDataset <https://github.com/antmicro/kenning/blob/master/kenning/datasets/pet_dataset.py>`_ for classification,
* `OpenImagesDatasetV6 <https://github.com/antmicro/kenning/blob/master/kenning/datasets/open_images_dataset.py>`_ for object detection,
* `RandomizedClassificationDataset <https://github.com/antmicro/kenning/blob/master/kenning/datasets/random_dataset.py>`_.

.. autoclass:: kenning.core.dataset.Dataset
   :members:

.. _modelwrapper-api:

ModelWrapper
------------

ModelWrapper-based objects wrap functions for:

* model preparation,
* model saving and loading,
* model inference in native framework,
* model-specific input and output processing,
* model conversion to the ONNX format.

Example model wrappers:

* `PyTorchWrapper <https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/frameworks/pytorch.py>`_ and `TensorFlowWrapper <https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/frameworks/tensorflow.py>`_ implement common methods for all models in PyTorch and TensorFlow frameworks,
* `PyTorchPetDatasetMobileNetV2 <https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/classification/pytorch_pet_dataset.py>`_ wraps the MobileNetV2 model for Pet classification implemented in PyTorch,
* `TensorFlowDatasetMobileNetV2 <https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/classification/tensorflow_pet_dataset.py>`_ wraps the MobileNetV2 model for Pet classification implemented in TensorFlow,
* `TVMDarknetCOCOYOLOV3 <https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/detectors/darknet_coco.py>`_ wraps the YOLOv3 model for COCO objet detection implemented in Darknet (without training and inference methods).

.. autoclass:: kenning.core.model.ModelWrapper
   :members:

.. _modelcompiler-api:

ModelCompiler
-------------

ModelCompiler objects wrap the deep learning compilation process.
They can perform the optimization of models (operation fusion, quantization) as well.

All ModelCompiler objects should provide methods for compiling models in ONNX format, but they can also provide support for other formats (like Keras .h5 files, or PyTorch .th files).

Example model compilers:

* `TFLiteCompiler <https://github.com/antmicro/kenning/blob/master/kenning/compilers/tflite.py>`_ - wraps TensorFlow Lite compilation,
* `TVMCompiler <https://github.com/antmicro/kenning/blob/master/kenning/compilers/tvm.py>`_ - wraps TVM compilation.

.. autoclass:: kenning.core.compiler.ModelCompiler
   :members:

.. _runtime-api:

Runtime
-------

The Runtime classes provide methods for running compiled models locally or remotely on target device.
Runtimes are usually compiler-specific (frameworks for deep learning compilers provide runtime libraries to run compiled models on a given hardware).

The client (host) side of the ``Runtime`` class utilizes the methods from :ref:`dataset-api`, :ref:`modelwrapper-api` and :ref:`runtimeprotocol-api` classes to run inference on the target device.
The server (target) side of the ``Runtime`` class requires implementing methods for:

* loading model delivered by the client,
* preparing inputs delivered by the client,
* running inference,
* preparing outputs to be delivered to the client,
* (optionally) sending inference statistics.

The examples for runtimes are:

* `TFLiteRuntime <https://github.com/antmicro/kenning/blob/master/kenning/runtimes/tflite.py>`_ for models compiled with TensorFlow Lite,
* `TVMRuntime <https://github.com/antmicro/kenning/blob/master/kenning/runtimes/tvm.py>`_ for models compiled with TVM.

.. autoclass:: kenning.core.runtime.Runtime
   :members:

.. _runtimeprotocol-api:

RuntimeProtocol
---------------

The RuntimeProtocol class conducts the communication between the client (host) and the server (target).

The RuntimeProtocol class requires implementing methods for:

* initializing the server and the client (communication-wise),
* waiting for the incoming data,
* sending the data,
* receiving the data,
* uploading the model inputs to the server,
* uploading the model to the server,
* requesting the inference on target,
* downloading the outputs from the server,
* (optionally) downloading the statistics from the server (i.e. performance speed, CPU/GPU utilization, power consumption),
* notifying of success or failure by the server,
* parsing messages.

The examples of RuntimeProtocol:

* `NetworkProtocol <https://github.com/antmicro/kenning/blob/master/kenning/runtimeprotocols/network.py>`_ - implements a TCP-based communication between the host and the client.

.. autoclass:: kenning.core.runtimeprotocol.RuntimeProtocol
   :members:

.. _measurements-api:

Measurements
------------

The ``kenning.core.measurements`` module contains ``Measurements`` and ``MeasurementsCollector`` classes for collecting performance and quality metrics.
``Measurements`` is a dict-like object that provides various methods for adding the performance metrics, adding values for time series, and updating existing values.

The dictionary held by ``Measurements`` needs to have serializable data, since most of the scripts save the performance results later in the JSON format for later report generation.

.. automodule:: kenning.core.measurements
   :members:

.. _onnxconversion-api:

ONNXConversion
--------------

ONNXConversion object contains methods for converting models in various frameworks to ONNX and vice versa.
It also provides methods for testing the conversion process empirically on a list of deep learning models implemented in tested frameworks.

.. autoclass:: kenning.core.onnxconversion.ONNXConversion
   :members:
