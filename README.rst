Kenning
=======

Copyright (c) 2020-2021 `Antmicro <https://www.antmicro.com>`_

.. image:: img/kenninglogo.png

This is a project for implementing and testing pipelines for deploying deep learning models on edge devices.

Deployment flow
---------------

Deploying deep learning models on edge devices usually involves the following steps:

* Preparation and analysis of the dataset, preparation of data preprocessing and output postprocessing routines,
* Model training (usually transfer learning), if necessary,
* Evaluation and improvement of the model until its quality is satisfactory,
* Model optimization, usually hardware-specific optimizations (e.g. operator fusion, quantization, neuron-wise or connection-wise pruning),
* Model compilation to a given target,
* Model execution on a given target.

There are different frameworks for most of the above steps (training, optimization, compilation and runtime). 
The cooperation between those frameworks differs and may provide different results.

This framework introduces interfaces for those above-mentioned steps that can be implemented using specific deep learning frameworks.

Based on the implemented interfaces, the framework can measure the inference duration and quality on a given target.
It also verifies the compatibility between various training, compilation and optimization frameworks.

Kenning structure
-----------------

The ``kenning`` module consists of the following submodules:

* ``core`` - provides interface APIs for datasets, models, compilers, runtimes and runtime protocols,
* ``datasets`` - provides implementations for datasets,
* ``modelwrappers`` - provides implementations for models for various problems implemented in various frameworks,
* ``compilers`` - provides implementations for compilers of deep learning models,
* ``runtimes`` - provides implementations of runtime on target devices,
* ``runtimeprotocols`` - provides implementations for communication protocols between host and tested target,
* ``onnxconverters`` - provides ONNX conversions for a given framework along with a list of models to test the conversion on,
* ``resources`` - contains project's resources, like RST templates, or trained models,
* ``scenarios`` - contains executable scripts for running training, inference, benchmarks and other tests on target devices.

Model preparation
-----------------

The ``kenning.core.dataset.Dataset`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes that implement the methods from ``kenning.core.dataset.Dataset`` are responsible for:

* preparing the dataset, including the download routines (use ``--download-dataset`` flag to download the dataset data),
* preprocessing the inputs into the format expected by most of the models for a given task,
* postprocessing the outputs for the evaluation process,
* evaluating a given model based on its predictions,
* subdividing the samples into training and validation datasets.

Based on the above methods, the ``Dataset`` class provides data to the model wrappers, compilers and runtimes to train and test the models.

The datasets are included in the ``kenning.datasets`` submodule.

Check out the `Pet Dataset wrapper <https://github.com/antmicro/kenning/blob/master/kenning/datasets/pet_dataset.py>`_ for an example of ``Dataset`` class implementation.

The ``kenning.core.model.ModelWrapper`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ModelWrapper`` class requires implementing methods for:

* model preparation,
* model saving and loading,
* model saving to the ONNX format,
* model-specific preprocessing of inputs and postprocessing of outputs, if neccessary,
* model inference,
* providing metadata (framework name and version),
* model training,
* input format specification,
* conversion of model inputs and outputs to bytes for the ``kenning.core.runtimeprotocol.RuntimeProtocol`` objects.

The ``ModelWrapper`` provides methods for running the inference in a loop from data from dataset and measures both the quality and inferenceperformance of the model.

The ``kenning.modelwrappers.frameworks`` submodule contains framework-wise specifications of ``ModelWrapper`` class - they implement all methods that are common for all the models implemented in this framework.

For the `Pet Dataset wrapper`_ object there is example classifier implemented in TensorFlow 2.x called `TensorFlowPetDatasetMobileNetV2 <https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/classification/tensorflow_pet_dataset.py>`_.

The ``kenning.core.compiler.ModelCompiler`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Objects of this class implement compilation and optional hardware-specific optimization.
For the latter, the ``ModelCompiler`` may require a dataset, for example to perform quantization or pruning.

The implementations for compiler wrappers are in ``kenning.compilers``.
For example, `TFLiteCompiler <https://github.com/antmicro/kenning/blob/master/kenning/compilers/tflite.py>`_ class wraps the TensorFlow Lite routines for compiling the model to a specified target.

Model deployment and benchmarking on target devices
---------------------------------------------------

Benchmarks of compiled models are performed in a client-server manner, where the target device acts as a server that accepts the compiled model and waits for the input data to infer, and the host device sends the input data and waits for the outputs to evaluate the quality of models.

The general communication protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The communication protocol is message-based.
There are:

* ``OK`` messages - indicate success, and may come with additional information,
* ``ERROR`` messages - indicate failure,
* ``DATA`` messages - provide input data for inference,
* ``MODEL`` messages - provide model to load for inference,
* ``PROCESS`` messages - request processing inputs delivered in ``DATA`` message,
* ``OUTPUT`` messages - request results of processing,
* ``STATS`` messages - request statistics from the target device.

The message types and enclosed data are encoded in format implemented in the ``kenning.core.runtimeprotocol.RuntimeProtocol``-based class.

The communication during inference benchmark session is as follows:

* The client (host) connects to the server (target),
* The client sends the ``MODEL`` request along with the compiled model,
* The server loads the model from request, prepares everything for running the model and sends the ``OK`` response,
* After receiving the ``OK`` response from the server, the client starts reading input samples from the dataset, preprocesses the inputs, and sends ``DATA`` request with the preprocessed input,
* Upon receiving the ``DATA`` request, the server stores the input for inference, and sends the ``OK`` message,
* Upon receiving confirmation, the client sends the ``PROCESS`` request,
* Just after receiving the ``PROCESS`` request, the server should send the ``OK`` message to confirm that it starts the inference, and just after finishing the inference the server should send another ``OK`` message to confirm that the inference is finished,
* After receiving the first ``OK`` message, the client starts measuring inference time until the second ``OK`` response is received,
* The client sends the ``OUTPUT`` request in order to receive the outputs from the server,
* Server sends the ``OK`` message along with the output data,
* The client parses the output and evaluates model performance,
* The client sends ``STATS`` request to obtain additional statistics (inference time, CPU/GPU/Memory utilization) from the server,
* If server provides any statistics, it sends the ``OK`` message with the data,
* The same process applies to the rest of input samples.

The way of determining the message type and sending data between the server and the client depends on the implementation of the ``kenning.core.runtimeprotocol.RuntimeProtocol`` class.
The implementation of running inference on the given target is implemented in the ``kenning.core.runtime.Runtime`` class.

The ``kenning.core.runtimeprotocol.RuntimeProtocol`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``RuntimeProtocol`` class conducts the communication between the client (host) and the server (target).

The ``RuntimeProtocol`` class requires implementing methods for:

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

Based on the above-mentioned methods, the ``kenning.core.runtime.Runtime`` connects the host with the target.

Look at the `TCP runtime protocol <https://github.com/antmicro/kenning/blob/master/kenning/runtimeprotocols/network.py>`_ for an example.

The ``kenning.core.runtime.Runtime`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Runtime`` objects provide an API for the host and (optionally) the target device.
If the target device does not support Python, the runtime needs to be implemented in a different language, and the host API needs to support it.

The client (host) side of the ``Runtime`` class utilizes the methods from ``Dataset``, ``ModelWrapper`` and ``RuntimeProtocol`` classes to run inference on the target device.
The server (target) side of the ``Runtime`` class requires implementing methods for:

* loading model delivered by the client,
* preparing inputs delivered by the client,
* running inference,
* preparing outputs to be delivered to the client,
* (optionally) sending inference statistics.

Look at the `TVM runtime <https://github.com/antmicro/kenning/blob/master/kenning/runtimes/tvm.py>`_ for an example.

ONNX conversion
---------------

Most of the frameworks for training, compiling and optimizing deep learning algorithms support ONNX format.
It allows conversion of models from one representation to another.

The ONNX API and format is constantly evolving, and there are more and more operators in new state-of-the-art models that need to be supported.

The ``kenning.core.onnxconversion.ONNXConversion`` class provides an API for writing compatibility tests between ONNX and deep learning frameworks.

It requires implementing:

* method for importing ONNX model for a given framework,
* method for exporting ONNX model from a given framework,
* list of models implemented in a given framework, where each model will be exported to ONNX, and then imported back to the framework.

The ``ONNXConversion`` class implements a method for converting the models.
It catches exceptions and any issues in the import/export methods, and provides the report on conversion status per model.

Look at the `TensorFlowONNXConversion class <https://github.com/antmicro/kenning/blob/master/kenning/onnxconverters/tensorflow.py>`_ for an example of API usage.

Running the benchmarks
----------------------

All executable Python scripts are available in the ``kenning.scenarios`` submodule.

Running model training on host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``kenning.scenarios.model_training`` script is run as follows::

    python -m kenning.scenarios.model_training \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.datasets.pet_dataset.PetDataset \
        --logdir build/logs \
        --dataset-root build/pet-dataset \
        --model-path build/trained-model.h5 \
        --batch-size 32 \
        --learning-rate 0.0001 \
        --num-epochs 50

By default, ``kenning.scenarios.model_training`` script requires two classes:

* ``ModelWrapper``-based class that describes model architecture and provides training routines,
* ``Dataset``-based class that provides training data for the model.

The remaining arguments are provided by the ``form_argparse`` class methods in each class, and may be different based on selected dataset and model.
In order to get full help for the training scenario for the above case, run::

    python -m kenning.scenarios.model_training \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.datasets.pet_dataset.PetDataset \
        -h

This will load all the available arguments for a given model and dataset.

The arguments in the above command are:

* ``--logdir`` - path to the directory where logs will be stored (this directory may be an argument for the TensorBoard software),
* ``--dataset-root`` - path to the dataset directory, required by the ``Dataset``-based class,
* ``--model-path`` - path where the trained model will be saved,
* ``--batch-size`` - training batch size,
* ``--learning-rate`` - training learning rate,
* ``--num-epochs`` - number of epochs.

If the dataset files are not present, use ``--download-dataset`` flag in order to let the Dataset API download the data.

Benchmarking trained model on host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``kenning.scenarios.inference_performance`` script runs the model using the deep learning framework used for training on a host device.
It runs the inference on a given dataset, computes model quality metrics and performance metrics.
The results from the script can be used as a reference point for benchmarking of the compiled models on target devices.

The example usage of the script is as follows::

    python -m kenning.scenarios.inference_performance \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.datasets.pet_dataset.PetDataset \
        build/result.json \
        --model-path kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
        --dataset-root build/pet-dataset

The obligatory arguments for the script are:

* ``ModelWrapper``-based class that implements the model loading, I/O processing and inference method,
* ``Dataset``-based class that implements fetching of data samples and evaluation of the model,
* ``build/result.json``, which is the path to the output JSON file with benchmark results.

The remaining parameters are specific to the ``ModelWrapper``-based class and ``Dataset``-based class.

Testing ONNX conversions
~~~~~~~~~~~~~~~~~~~~~~~~

The ``kenning.scenarios.onnx_conversion`` runs as follows::

    python -m kenning.scenarios.onnx_conversion \
        build/models-directory \
        build/onnx-support.rst \
        --converters-list \
            kenning.onnxconverters.pytorch.PyTorchONNXConversion \
            kenning.onnxconverters.tensorflow.TensorFlowONNXConversion \
            kenning.onnxconverters.mxnet.MXNetONNXConversion

The first argument is the directory, where the generated ONNX models will be stored.
The second argument is the RST file with import/export support table for each model for each framework.
The third argument is the list of ``ONNXConversion`` classes implementing list of models, import method and export method.

.. _compilation-and-deployment:

Running compilation and deployment of models on target hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two scripts - ``kenning.scenarios.inference_tester`` and ``kenning.scenarios.inference_server``.

The example call for the first script is following::

    python -m kenning.scenarios.inference_tester \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.compilers.tflite.TFLiteCompiler \
        kenning.runtimes.tflite.TFLiteRuntime \
        kenning.datasets.pet_dataset.PetDataset \
        ./build/google-coral-devboard-tflite-tensorflow.json \
        --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
        --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
        --model-framework keras \
        --target "edgetpu" \
        --compiled-model-path build/compiled-model.tflite \
        --inference-input-type int8 \
        --inference-output-type int8 \
        --host 192.168.188.35 \
        --port 12345 \
        --packet-size 32768 \
        --save-model-path /home/mendel/compiled-model.tflite \
        --dataset-root build/pet-dataset \
        --inference-batch-size 1 \
        --verbosity INFO

The script requires:

* ``ModelWrapper``-based class that implements model loading, I/O processing and optionally model conversion to ONNX format,
* ``ModelCompiler``-based class for compiling the model for a given target,
* ``Runtime``-based class that implements data processing and the inference method for the compiled model on the target hardware,
* ``Dataset``-based class that implements fetching of data samples and evaluation of the model,
* ``./build/google-coral-devboard-tflite-tensorflow.json``, which is the path to the output JSON file with performance and quality metrics.

In case of running inference on remote edge device, the ``--protocol-cls RuntimeProtocol`` also needs to be provided in order to provide communication protocol between the host and the target.
If ``--protocol-cls`` is not provided, the ``inference_tester`` will run inference on the host machine (which is useful for testing and comparison).

The remaining arguments come from the above-mentioned classes.
Their meaning is following:

* ``--model-path`` (``TensorFlowPetDatasetMobileNetV2`` argument) is the path to the trained model that will be compiled and executed on the target hardware,
* ``--model-framework`` (``TFLiteCompiler`` argument) tells the compiler what is the format of the file with the saved model (it tells which backend to use for parsing the model by the compiler),
* ``--target`` (``TFLiteCompiler`` argument) is the name of the target hardware for which the compiler generates optimized binaries,
* ``--compiled-model-path`` (``TFLiteCompiler`` argument) is the path where the compiled model will be stored on host,
* ``--inference-input-type`` (``TFLiteCompiler`` argument) tells TFLite compiler what will be the type of the input tensors,
* ``--inference-output-type`` (``TFLiteCompiler`` argument) tells TFLite compiler what will be the type of the output tensors,
* ``--host`` tells the ``NetworkProtocol`` what is the IP address of the target device,
* ``--port`` tells the ``NetworkProtocol`` on what port the server application is listening,
* ``--packet-size`` tells the ``NetworkProtocol`` what the packet size during communication should be,
* ``--save-model-path`` (``TFLiteRuntime`` argument) is the path where the compiled model will be stored on the target device,
* ``--dataset-root`` (``PetDataset`` argument) is the path to the dataset files,
* ``--inference-batch-size`` is the batch size for the inference on the target hardware,
* ``--verbosity`` is the verbosity of logs.

The example call for the second script is as follows::

    python -m kenning.scenarios.inference_server \
        kenning.runtimeprotocols.network.NetworkProtocol \
        kenning.runtimes.tflite.TFLiteRuntime \
        --host 0.0.0.0 \
        --port 12345 \
        --packet-size 32768 \
        --save-model-path /home/mendel/compiled-model.tflite \
        --delegates-list libedgetpu.so.1 \
        --verbosity INFO

This script only requires ``Runtime``-based class and ``RuntimeProtocol``-based class.
It waits for a client using a given protocol, and later runs inference based on the implementation from the ``Runtime`` class.

The additional arguments are as follows:

* ``--host`` (``NetworkProtocol`` argument) is the address where the server will listen,
* ``--port`` (``NetworkProtocol`` argument) is the port on which the server will listen,
* ``--packet-size`` (``NetworkProtocol`` argument) is the size of the packet,
* ``--save-model-path`` is the path where the received model will be saved,
* ``--delegates-list`` (``TFLiteRuntime`` argument) is a TFLite-specific list of libraries for delegating the inference to deep learning accelerators (``libedgetpu.so.1`` is the delegate for Google Coral TPUs).

First, the client compiles the model and sends it to the server using the runtime protocol.
Then, it sends next batches of data to process to the server.
In the end, it collects the benchmark metrics and saves them to JSON file.
In addition, it generates plots with performance changes over time.

Render report from benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``kenning.scenarios.inference_performance`` and ``kenning.scenarios.inference_tester`` create JSON files that contain:

* command string that was used to generate the JSON file,
* frameworks along with their versions used to train the model and compile the model,
* performance metrics, including:

    * CPU usage over time,
    * RAM usage over time,
    * GPU usage over time,
    * GPU memory usage over time,
* predictions and ground truth to compute quality metrics, i.e. in form of confusion matrix and top-5 accuracy for classification task.

The ``kenning.scenarios.render_report`` renders the report RST file along with plots for metrics for a given JSON file based on selected templates.

For example, for the file ``./build/google-coral-devboard-tflite-tensorflow.json`` created in :ref:`compilation-and-deployment` the report can be rendered as follows::

    python -m kenning.scenarios.render_report \
        build/google-coral-devboard-tflite-tensorflow.json \
        "Pet Dataset classification using TFLite-compiled TensorFlow model" \
        docs/source/generated/google-coral-devboard-tpu-tflite-tensorflow-classification.rst \
        --img-dir docs/source/generated/img/ \
        --root-dir docs/source/ \
        --report-types \
            performance \
            classification

Where:

* ``build/google-coral-devboard-tflite-tensorflow.json`` is the input JSON file with benchmark results
* ``"Pet Dataset classification using TFLite-compiled TensorFlow model"`` is the report name that will be used as title in generated plots,
* ``docs/source/generated/google-coral-devboard-tpu-tflite-tensorflow-classification.rst`` is the path to the output RST file,
* ``--img-dir docs/source/generated/img/`` is the path to the directory where generated plots will be stored,
* ``--root-dir docs/source`` is the root directory for documentation sources (it will be used to compute relative paths in the RST file),
* ``--report-types performance classification`` is the list of report types that will form the final RST file.

The ``performance`` type provides report sections for performance metrics, i.e.:

* Inference time changes over time,
* Mean CPU usage over time,
* RAM usage over time,
* GPU usage over time,
* GPU memory usage over time.

It also computes mean, standard deviation and median values for the above time series.

The ``classification`` type provides report section regarding quality metrics for classification task:

* Confusion matrics,
* Per-class precision,
* Per-class sensitivity,
* Accuracy,
* Top-5 accuracy,
* Mean precision,
* Mean sensitivity,
* G-Mean.

The above metrics can be used to determine any quality losses resulting from optimizations (i.e. pruning or quantization).

Adding new implementations
--------------------------

``Dataset``, ``ModelWrapper``, ``ModelCompiler``, ``RuntimeProtocol``, ``Runtime`` and other classes from ``kenning.core`` module have dedicated directories for their implementations.
Each method in base classes that requires implementation raises ``NotImplementedError`` exception.
Implemented methods can be also overriden, if neccessary.

Most of the base classes implement ``form_argparse`` and ``from_argparse`` methods.
The first one creates an argument parser and a group of arguments specific to the base class.
The second one creates an object of the class based on the arguments from argument parser.

Inheriting classes can modify ``form_argparse`` and ``from_argparse`` methods to provide better control over their processing, but they should always be based on the results of their base implementations.
