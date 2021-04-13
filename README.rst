Edge AI tester
==============

Copyright (c) 2021 `Antmicro <https://www.antmicro.com>`_.

This is a project for implementing testing pipelines for deploying deep learning models on edge devices.

Deployment flow
---------------

Deploying deep learning models on edge devices usually involves following steps:

* Preparation and analysis of the dataset, preparation of data preprocessing and output postprocessing routines,
* Model training (usually transfer learning), if necessary,
* Evaluation of the model and model improvement until its quality is satisfactory,
* Model optimization, usually hardware-specific optimizations (e.g. operator fusion, quantization, neuron-wise or connection-wise pruning),
* Model compilation to a given target,
* Model execution on a given target.

For nearly all of the above steps (training, optimization, compilation and runtime) there are different frameworks.
The cooperation between those frameworks differs and may provide different results.

This framework introduces interfaces for those above-mentioned steps that can be implemented using specific deep learning frameworks.

Based on the implemented interfaces, the framework can measure the inference duration and quality on a given target.
It also verifies the compatibility between various training, compilation and optimization frameworks.

Edge AI tester structure
------------------------

The ``edge_ai_tester`` module consists of following submodules:

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

The ``edge_ai_tester.core.dataset.Dataset`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes that implement the methods from ``edge_ai_tester.core.dataset.Dataset`` are responsible for:

* preparing the dataset, involving the download routines,
* preprocessing the inputs to format expected by most of the models for a given task,
* postprocessing the outputs for the evaluation process,
* evaluating a given model based on its predictions,
* subdividing the samples into training and validation datasets.

Based on the above methods the ``Dataset`` class provides data to the model wrappers, compilers and runtimes to train and test the models.

The datasets are present in the ``edge_ai_tester.datasets`` submodule.

Check out the `Pet Dataset wrapper <./edge_ai_tester/datasets/pet_dataset.py>`_ for an example of ``Dataset`` class implementation.

The ``edge_ai_tester.core.model.ModelWrapper`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ModelWrapper`` class requires implementing methods for:

* model preparation,
* model saving and loading,
* model saving to the ONNX format,
* model-specific preprocessing of inputs and postprocessing of outputs, if neccessary,
* model inference,
* providing metadata (framework name and version),
* model training,
* input format specification,
* conversion of model inputs and outputs to bytes for the ``edge_ai_tester.core.runtimeprotocol.RuntimeProtocol`` objects.

The ``ModelWrapper`` provides methods for running the inference in a loop from data from dataset and measures both the quality and inferenceperformance of the model.

The ``edge_ai_tester.modelwrappers.frameworks`` submodule contains framework-wise specifications of ``ModelWrapper`` class - they implement all methods that are common for all the models implemented in this framework.

For the `Pet Dataset wrapper`_ object there is example classifier implemented in TensorFlow 2.x called `TensorFlowPetDatasetMobileNetV2 <./edge_ai_tester/modelwrappers/classification/tensorflow_pet_dataset.py>`_.

The ``edge_ai_tester.core.compiler.ModelCompiler`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Objects of this class implement compilation and optional hardware-specific optimization.
For the latter, the ``ModelCompiler`` may require a dataset, for example to perform quantization or pruning.

The implementations for compiler wrappers are in ``edge_ai_tester.compilers``.
For example, `TFLiteCompiler <./edge_ai_tester/compilers/tvm.py>`_ class wraps the TVM routines for compiling the model to a specified target.

Model deployment and benchmarking on target devices
---------------------------------------------------

Benchmarks of compiled models are performed in a client-server manner, where the target device acts as a server that accepts the compiled model and waits for the input data to infer, and the host device sends the input data and waits for the outputs to evaluate the quality of models.

The general communication protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The communication protocol is message-based.
There are:

* ``OK`` messages - indicate success, and may come with additional data,
* ``ERROR`` messages - indicate failure,
* ``DATA`` messages - they provide input data for inference,
* ``MODEL`` messages - they provide model to load for inference,
* ``PROCESS`` messages - they request processing inputs delivered in ``DATA`` message,
* ``OUTPUT`` messages - they request results of processing,
* ``STATS`` messages - they request statistics from the target device.

The message types and enclosed data are encoded in format implemented in the ``edge_ai_tester.core.runtimeprotocol.RuntimeProtocol``-based class.

The communication during inference benchmark session is as follows:

* The client (host) connects to the server (target),
* The client sends the ``MODEL`` request along with the compiled model,
* The server loads the model from request, prepares everything for running the model and sends ``OK`` response,
* After receiving ``OK`` response from the server, clients starts reading input samples from the dataset, preprocesses the inputs, and sends ``DATA`` request with the preprocessed input,
* Upon receiving the ``DATA`` request, the server stores the input for inference, and sends ``OK`` message,
* Upon receiving confirmation, the client sends ``PROCESS`` request,
* Just after receiving the ``PROCESS`` request, the server should send the ``OK`` message to confirm that it starts the inference, and just after finishing the inference the server should send another ``OK`` message to confirm that the inference is finished,
* After receiving the first ``OK`` message, the client starts measuring inference time until the second ``OK`` response is received,
* The client sends ``OUTPUT`` request in order to receive the outputs from the server,
* Server sends ``OK`` message along with the output data,
* The client parses the output and evaluates model performance,
* The client sends ``STATS`` request to obtain additional statistics (inference time, CPU/GPU/Memory utilization) from the server,
* If server provides any statistics, it sends ``OK`` message with the data,
* The same process applies for the rest of input samples.

The way of determining the message type and sending data between the server and the client depends on the implementation of the ``edge_ai_tester.core.runtimeprotocol.RuntimeProtocol`` class.
The implementation of running inference on the given target is implemented in the ``edge_ai_tester.core.runtime.Runtime`` class.

The ``edge_ai_tester.core.runtimeprotocol.RuntimeProtocol`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
* notifying the success or failure by the server,
* parsing messages.

Based on the above-mentioned methods, the ``edge_ai_tester.core.runtime.Runtime`` connects the host with the target.

Look at the `TCP runtime protocol <./edge_ai_tester/runtimeprotocols/network.py>`_ for an example.

The ``edge_ai_tester.core.runtime.Runtime`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Runtime`` objects provide an API for the host and (optionally) the target device.
If the target device does not support Python, the runtime needs to be implemented in a different language, and the host API needs to support it.

The client (host) side of the ``Runtime`` class utilizes the methods from ``Dataset``, ``ModelWrapper`` and ``RuntimeProtocol`` classes to run inference on target device.
The server (target) side of the ``Runtime`` class requires implementing methods for:

* loading model delivered by the client,
* preparing inputs delivered by the client,
* running inference,
* preparing outputs to be delivered to the client,
* (optionally) sending inference statistics.

Look at the `TVM runtime <./edge_ai_tester/runtimes/tvm.py>`_ for an example.

ONNX conversion
---------------

Most of the frameworks for training, compiling and optimizing deep learning algorithms support ONNX format.
It allows conversion of models from one representation to another.

The ONNX API and format is constantly evolving, and there are more and more operators in new state-of-the-art models that need to be supported.

The ``edge_ai_tester.core.onnxconversion.ONNXConversion`` class provides an API for writing compatibility tests between ONNX and deep learning frameworks.

It requires implementing:

* method for importing ONNX model for a given framework,
* method for exporting ONNX model from a given framework,
* list of models implemented in a given framework, where each model will be exported to ONNX, and then imported back to the framework.

The ``ONNXConversion`` class implements method for converting the models.
It catches exceptions and any issues in the import/export methods, and provides the report on conversion status per model.

Look at the `TensorFlowONNXConversion class <./edge_ai_tester/onnxconverters/tensorflow.py>`_ for an example of API usage.

Running the benchmarks
----------------------

All executable Python scripts are available in the ``edge_ai_tester.scenarios`` submodule.

Running model training on host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``edge_ai_tester.scenarios.model_training`` script is run as follows::

    python -m edge_ai_tester.scenarios.model_training \
        edge_ai_tester.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        edge_ai_tester.datasets.pet_dataset.PetDataset \
        --logdir build/logs \
        --dataset-root build/pet-dataset \
        --model-path build/trained-model.h5 \
        --batch-size 32 \
        --learning-rate 0.0001 \
        --num-epochs 50

By default, ``edge_ai_tester.scenarios.model_training`` script requires two classes:

* ``ModelWrapper``-based class that describes model architecture and provides training routines,
* ``Dataset``-based class that provides training data for the model.

The remaining arguments are provided by the ``form_argparse`` class methods in each class, and may be different based on selected dataset and model.
In order to get the full help for the training scenario for the above case, run::

    python -m edge_ai_tester.scenarios.model_training \
        edge_ai_tester.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        edge_ai_tester.datasets.pet_dataset.PetDataset \
        -h

This will load all the available arguments for a given model and dataset.

The arguments in the above command are:

* ``--logdir`` - path to the directory where logs will be stored (this directory may be an argument for the TensorBoard software),
* ``--dataset-root`` - path to the dataset directory, required by the ``Dataset``-based class,
* ``--model-path`` - path where the trained model will be saved,
* ``--batch-size`` - training batch size,
* ``--learning-rate`` - training learning rate,
* ``--num-epochs`` - number of epochs.

Benchmarking trained model on host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``edge_ai_tester.scenarios.inference_performance`` script runs the model using its deep learning framework routines on a host device.
It runs the inference on a given dataset, computes model quality metrics and performance metrics.
The results from the script can be used as a reference point for the benchmarks of the compiled models on target devices.

The example usage of the script is following::

    python -m edge_ai_tester.scenarios.inference_performance \
        edge_ai_tester.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        edge_ai_tester.datasets.pet_dataset.PetDataset \
        build/report-directory \
        report-name \
        --model-path build/trained-model.h5 \
        --dataset-root build/pet-dataset

The obligatory arguments for the script are:

* ``ModelWrapper``-based class that implements the model loading, I/O processing and inference method,
* ``Dataset``-based class that implements fetching of data samples and evaluation of the model,
* ``build/report-directory``, which is the path where the JSON with benchmark results, along with plots for quality and performance metrics are stored,
* ``report-name``, which is the name of the report that will act as prefix for all files generated and saved in the ``build/report-directory``.

The remaining parameters are specific to the ``ModelWrapper``-based class and ``Dataset``-based class.

Testing ONNX conversions
~~~~~~~~~~~~~~~~~~~~~~~~

The ``edge_ai_tester.scenarios.onnx_conversion`` runs as follows::

    python -m edge_ai_tester.scenarios.onnx_conversion \
        ./onnx-models-directory \
        build/onnx-support-grid.rst \
        --converters-list \
            edge_ai_tester.onnxconverters.pytorch.PyTorchONNXConversion \
            edge_ai_tester.onnxconverters.tensorflow.TensorFlowONNXConversion

The first argument is the directory, where the generated ONNX models will be stored.
The second argument is the RST file with import/export support table for each model for each framework.
The third argument is the list of ``ONNXConversion`` classes implementing list of models, import method and export method.

Running compilation and deployment of models on target hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two scripts - ``edge_ai_tester.scenarios.inference_client`` and ``edge_ai_tester.scenarios.inference_server``.

The example call for the first script is following::

    python -m edge_ai_tester.scenarios.inference_client \
        edge_ai_tester.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        edge_ai_tester.compilers.tflite.TFLiteCompiler \
        edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
        edge_ai_tester.runtimes.tflite.TFLiteRuntime \
        edge_ai_tester.datasets.pet_dataset.PetDataset \
        build/report-directory \
        report-name \
        --model-path build/trained-model.h5 \
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
* ``RuntimeProtocol``-based class that implements communication between the host and the target hardware,
* ``Runtime``-based class that implements data processing and the inference method for the compiled model on the target hardware,
* ``Dataset``-based class that implements fetching of data samples and evaluation of the model,
* ``build/report-directory``, which is the path where JSON with benchmark results and benchmark plots will be saved,
* ``report-name``, which is the name of a given benchmark.

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
* ``--packet-size`` tells the ``NetworkProtocol`` what should be the packet size during communication,
* ``--save-model-path`` (``TFLiteRuntime`` argument) is the path where the compiled model will be stored on target device,
* ``--dataset-root`` (``PetDataset`` argument) is the path to the dataset files,
* ``--inference-batch-size`` is the batch size for the inference on the target hardware,
* ``--verbosity`` is the verbosity of logs.

The example call for the second script is following::

    python -m edge_ai_tester.scenarios.inference_server \
        edge_ai_tester.runtimeprotocols.network.NetworkProtocol \
        edge_ai_tester.runtimes.tflite.TFLiteRuntime \
        --host 0.0.0.0 \
        --port 12345 \
        --packet-size 32768 \
        --save-model-path /home/mendel/compiled-model.tflite \
        --delegates-list libedgetpu.so.1 \
        --verbosity INFO

This script only requires ``Runtime``-based class and ``RuntimeProtocol``-based class.
It waits for a client using a given protocol, and later runs inference based on the implementation from the ``Runtime`` class.

The additional arguments are following:

* ``--host`` (``NetworkProtocol`` argument) is the address where the server will listen,
* ``--port`` (``NetworkProtocol`` argument) is the port on which the server will listen,
* ``--packet-size`` (``NetworkProtocol`` argument) is the size of the packet,
* ``--save-model-path`` is the path where the received model will be saved,
* ``--delegates-list`` (``TFLiteRuntime`` argument) is a TFLite-specific list of libraries for delegating the inference to deep learning accelerators (``libedgetpu.so.1`` is the delegate for Google Coral TPUs).

First, the client compiles the model and sends it to the server using the runtime protocol.
Then, it sends next batches of data to process to the server.
In the end, it collects the benchmark metrics and saves them to JSON file.
In addition, it generates plots with performance changes over time.

Adding new implementations
--------------------------

``Dataset``, ``ModelWrapper``, ``ModelCompiler``, ``RuntimeProtocol``, ``Runtime`` and other classes from ``edge_ai_tester.core`` module have dedicated directories for their implementations.
Each method in base classes that requires implementation raises NotImplementedError.
Implemented methods can be also overriden, if neccessary.

Most of the base classes implement ``form_argparse`` and ``from_argparse`` methods.
The first one creates an argument parser and a group of arguments specific to the base class.
The second one creates an object of the class based on the arguments from argument parser.

Inheriting classes can modify ``form_argparse`` and ``from_argparse`` methods to provide better control over their processing, but they should always be based on the results of their base implementations.
