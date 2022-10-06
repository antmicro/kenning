# Kenning API

## Deployment API overview

```{figure} img/class-flow.png
---
name: class-flow
alt: Kenning core classes and interactions between them
align: center
---

Kenning core classes and interactions between them.
The green blocks represent the flow of the input data that is passed to the model for inference.
The orange blocks represent the flow of model deployment flow, from training to inference on target device.
The grey blocks represent the inference results and metrics flow.
```

{{projecturl}} provides:

* [](dataset-api) class - performs dataset downloading, preparation, input preprocessing, output postprocessing and model evaluation,
* [](modelwrapper-api) class - trains the model, prepares the model, performs model-specific input preprocessing and output postprocessing, runs inference on host using native framework,
* [](optimizer-api) class - optimizes and compiles the model,
* [](runtime-api) class - loads the model, performs inference on compiled model, runs target-specific processing of inputs and outputs, and runs performance benchmarks,
* [](runtimeprotocol-api) class - implements the communication protocol between the host and the target,
* [](dataprovider-api) class - implements providing data from such sources as camera, TCP connection or others for inference,
* [](outputcollector-api) class - implements parsing and utilizing data coming from inference (such as displaying the visualizations, sending the results to via TCP).

### Model processing

The orange blocks and arrows in the {figure:numref}`class-flow` represent the model life cycle:

* the model is designed, trained, evaluated and improved - the training is implemented in the [](modelwrapper-api).
  ```{note}
  This is an optional step - the already trained model can also be wrapped and used.
  ```
* the model is passed to the [](optimizer-api) where it is optimized for a given hardware and later compiled,
* during inference testing, the model is sent to the target using [](runtimeprotocol-api),
* the model is loaded on target side and used for inference using [](runtime-api).

Once the development of the model is complete, the optimized and compiled model can be used directly on target device using [](runtime-api).

### I/O data flow

The data flow is represented in the {figure:numref}`class-flow` with the green blocks.
The input data flow is depicted using green arrows, and the output data flow is depicted using grey arrows.

Firstly, the input and output data is loaded from dataset files and processed.
Later, since every model has its specific input preprocessing and output postprocessing routines, the data is passed to the [](modelwrapper-api) methods to apply those modifications.
During inference testing, the data is sent to and from the target using [](runtimeprotocol-api).

In the end, since [](runtime-api) runtimes also have their specific representations of data, the proper I/O processing is applied.

### Reporting data flow

The report rendering requires performance metrics and quality metrics.
The flow for this is presented with grey lines and blocks in {figure:numref}`class-flow`.

On target side, the performance metrics are computed and sent using [](runtimeprotocol-api) back to the host, and later passed to the report rendering.
After the output data goes through processing in the [](runtime-api) and [](modelwrapper-api), it is compared to the ground truth in the [](dataset-api) during model evaluation.
In the end, the results of model evaluation are passsed to the report rendering.

The final report is generated as an RST file with figures, as can be observed in the [](./sample-report).

(dataset-api)=
## Dataset

`kennning.core.dataset.Dataset`-based classes are responsible for:

* preparing the dataset, including the download routines (use `--download-dataset` flag to download the dataset data),
* preprocessing the inputs into the format expected by most of the models for a given task,
* postprocessing the outputs for the evaluation process,
* evaluating a given model based on its predictions,
* subdividing the samples into training and validation datasets.

The Dataset objects are used by:

* :ref:`modelwrapper-api` - for training purposes and model evaluation,
* :ref:`optimizer-api` - can be used i.e. for extracting calibration dataset for quantization purposes,
* :ref:`runtime-api` - is used for evaluating the model on target hardware.

The available implementations of datasets are included in the `kenning.datasets` submodule.
The example implementations:

* [PetDataset](https://github.com/antmicro/kenning/blob/master/kenning/datasets/pet_dataset.py) for classification,
* [OpenImagesDatasetV6](https://github.com/antmicro/kenning/blob/master/kenning/datasets/open_images_dataset.py) for object detection,
* [RandomizedClassificationDataset](https://github.com/antmicro/kenning/blob/master/kenning/datasets/random_dataset.py).

```{eval-rst}
.. autoclass:: kenning.core.dataset.Dataset
   :members:
```

(modelwrapper-api)=

## ModelWrapper

`kenning.core.model.ModelWrapper` base class requires implementing methods for:

* model preparation,
* model saving and loading,
* model saving to the ONNX format,
* model-specific preprocessing of inputs and postprocessing of outputs, if neccessary,
* model inference,
* providing metadata (framework name and version),
* model training,
* input format specification,
* conversion of model inputs and outputs to bytes for the `kenning.core.runtimeprotocol.RuntimeProtocol` objects.

The `ModelWrapper` provides methods for running the inference in a loop for data from dataset and measuring both the quality and inference performance of the model.

The `kenning.modelwrappers.frameworks` submodule contains framework-wise implementations of `ModelWrapper` class - they implement all methods that are common for given frameworks regardless of used model.

For the `Pet Dataset wrapper`_ object there is an example classifier implemented in TensorFlow 2.x called `TensorFlowPetDatasetMobileNetV2 <https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/tensorflow_pet_dataset.py>`_.

Examples of model wrappers:

* [PyTorchWrapper](https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/frameworks/pytorch.py) and [TensorFlowWrapper](https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/frameworks/tensorflow.py) implement common methods for all models in PyTorch and TensorFlow frameworks,
* [PyTorchPetDatasetMobileNetV2](https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/classification/pytorch_pet_dataset.py) wraps the MobileNetV2 model for Pet classification implemented in PyTorch,
* [TensorFlowDatasetMobileNetV2](https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/classification/tensorflow_pet_dataset.py) wraps the MobileNetV2 model for Pet classification implemented in TensorFlow,
* [TVMDarknetCOCOYOLOV3](https://github.com/antmicro/kenning/blob/master/kenning/modelwrappers/detectors/darknet_coco.py) wraps the YOLOv3 model for COCO objet detection implemented in Darknet (without training and inference methods).

```{eval-rst}
.. autoclass:: kenning.core.model.ModelWrapper
   :members:
```

(optimizer-api)=

## Optimizer

`kenning.core.optimizer.Optimizer` objects wrap the deep learning compilation process.
They can perform the optimization of models (operation fusion, quantization) as well.

All Optimizer objects should provide methods for compiling models in ONNX format, but they can also provide support for other formats (like Keras .h5 files, or PyTorch .th files).

Example model compilers:

* [TFLiteCompiler](https://github.com/antmicro/kenning/blob/master/kenning/compilers/tflite.py) - wraps TensorFlow Lite compilation,
* [TVMCompiler](https://github.com/antmicro/kenning/blob/master/kenning/compilers/tvm.py) - wraps TVM compilation.

```{eval-rst}
.. autoclass:: kenning.core.optimizer.Optimizer
   :members:
```

(runtime-api)=

## Runtime

`kenning.core.runtime.Runtime` class provides interfaces for methods for running compiled models locally or remotely on target device.
Runtimes are usually compiler-specific (frameworks for deep learning compilers provide runtime libraries to run compiled models on a given hardware).

The client (host) side of the `Runtime` class utilizes the methods from [](dataset-api), [](modelwrapper-api) and [](runtimeprotocol-api) classes to run inference on the target device.
The server (target) side of the `Runtime` class requires implementing methods for:

* loading model delivered by the client,
* preparing inputs delivered by the client,
* running inference,
* preparing outputs to be delivered to the client,
* (optionally) sending inference statistics.

The examples of runtimes are:

* [TFLiteRuntime](https://github.com/antmicro/kenning/blob/main/kenning/runtimes/tflite.py) for models compiled with TensorFlow Lite,
* [TVMRuntime](https://github.com/antmicro/kenning/blob/master/kenning/runtimes/tvm.py) for models compiled with TVM.

```{eval-rst}
.. autoclass:: kenning.core.runtime.Runtime
   :members:
```

(runtimeprotocol-api)=

## RuntimeProtocol

`kenning.core.runtimeprotocol.RuntimeProtocol` class conducts the communication between the client (host) and the server (target).

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

Based on the above-mentioned methods, the `kenning.core.runtime.Runtime` connects the host with the target.

The examples of RuntimeProtocol:

* [NetworkProtocol](https://github.com/antmicro/kenning/blob/master/kenning/runtimeprotocols/network.py) - implements a TCP-based communication between the host and the client.

(runtime-protocol-spec)=

### Runtime protocol specification

The communication protocol is message-based.
There are:

* `OK` messages - indicate success, and may come with additional information,
* `ERROR` messages - indicate failure,
* `DATA` messages - provide input data for inference,
* `MODEL` messages - provide model to load for inference,
* `PROCESS` messages - request processing inputs delivered in `DATA` message,
* `OUTPUT` messages - request results of processing,
* `STATS` messages - request statistics from the target device.

The message types and enclosed data are encoded in format implemented in the `kenning.core.runtimeprotocol.RuntimeProtocol`-based class.

The communication during inference benchmark session is as follows:

* The client (host) connects to the server (target),
* The client sends the `MODEL` request along with the compiled model,
* The server loads the model from request, prepares everything for running the model and sends the `OK` response,
* After receiving the `OK` response from the server, the client starts reading input samples from the dataset, preprocesses the inputs, and sends `DATA` request with the preprocessed input,
* Upon receiving the `DATA` request, the server stores the input for inference, and sends the `OK` message,
* Upon receiving confirmation, the client sends the `PROCESS` request,
* Just after receiving the `PROCESS` request, the server should send the `OK` message to confirm that it starts the inference, and just after finishing the inference the server should send another `OK` message to confirm that the inference is finished,
* After receiving the first `OK` message, the client starts measuring inference time until the second `OK` response is received,
* The client sends the `OUTPUT` request in order to receive the outputs from the server,
* Server sends the `OK` message along with the output data,
* The client parses the output and evaluates model performance,
* The client sends `STATS` request to obtain additional statistics (inference time, CPU/GPU/Memory utilization) from the server,
* If server provides any statistics, it sends the `OK` message with the data,
* The same process applies to the rest of input samples.

The way of determining the message type and sending data between the server and the client depends on the implementation of the `kenning.core.runtimeprotocol.RuntimeProtocol` class.
The implementation of running inference on the given target is implemented in the `kenning.core.runtime.Runtime` class.

### RuntimeProtocol API

`kenning.core.runtimeprotocol.RuntimeProtocol`-based classes implement the [](runtime-protocol-spec) in a given mean of transport, i.e. TCP connection, or UART.
It requires implementing methods for:

* initializing server (target hardware) and client (compiling host),
* sending and receiving data,
* connecting and disconnecting,
* uploading (host) and downloading (target hardware) the model,
* parsing and creating messages.

```{eval-rst}
.. autoclass:: kenning.core.runtimeprotocol.RuntimeProtocol
   :members:
```

(measurements-api)=

## Measurements

`kenning.core.measurements` module contains `Measurements` and `MeasurementsCollector` classes for collecting performance and quality metrics.
`Measurements` is a dict-like object that provides various methods for adding the performance metrics, adding values for time series, and updating existing values.

The dictionary held by `Measurements` needs to have serializable data, since most of the scripts save the performance results later in the JSON format for later report generation.

```{eval-rst}
.. automodule:: kenning.core.measurements
   :members:
```

(onnxconversion-api)=

## ONNXConversion

`ONNXConversion` object contains methods for converting models in various frameworks to ONNX and vice versa.
It also provides methods for testing the conversion process empirically on a list of deep learning models implemented in tested frameworks.

```{eval-rst}
.. autoclass:: kenning.core.onnxconversion.ONNXConversion
   :members:
```

(dataprovider-api)=

## DataProvider

The DataProvider classes are used during deployment for providing data to infer.
They can provide data from such sources as camera, video files, microphone data or TCP connection.

```{eval-rst}
.. autoclass:: kenning.core.dataprovider.DataProvider
   :members:
```

(outputcollector-api)=

## OutputCollector

The OutputCollector classes are used during deployment for receiving and processing inference results.
They can display the results, send them, or store them in a file.

```{eval-rst}
.. autoclass:: kenning.core.outputcollector.OutputCollector
   :members:
```
