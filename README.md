## Kenning

Copyright (c) 2020-2022 [Antmicro](https://www.antmicro.com>)

![Kenning](img/kenninglogo.png)

Kenning is a framework for creating deployment flows and runtimes for Deep Neural Network applications on various target hardware.

It aims towards providing modular execution blocks for:

* dataset management,
* model training,
* model compilation,
* model evaluation and performance reports,
* model runtime on target device,
* model input and output processing (i.e. fetching frames from camera and computing final predictions from model outputs).

that can be used seamlessly regardless of underlying frameworks for above-mentioned steps.

Kenning does not aim towards bringing yet another training or compilation framework for deep learning models - there are lots of mature and versatile frameworks that support certain models, training routines, optimization techniques, hardware platforms and other components crucial to the deployment flow.
Still, there is no framework that would support all of the models or target hardware devices - especially the support matrix between compilation frameworks and target hardware is extremely sparse.
This means that any change in the application, especially in hardware, may end up with a necessity to change the whole or significant part of the application flow.

Kenning addresses this issue by providing an unified API that focuses more on deployment tasks rather than their implementation - the developer decides which implementation for each task should be used, and Kenning allows to do it in a seamless way.
This way switching to another target platform results in most cases in very small change in the code, instead of reimplementing large parts of the project.
This way the Kenning can get the most out of the existing Deep Neural Network training and compilation frameworks.

For more details regarding the Kenning framework, Deep Neural Network deployment flow and its API check [Kenning documentation](https://antmicro.github.io/kenning/).

### Kenning installation

#### Module installation with pip

To install Kenning with its basic dependencies with pip, run:

```
pip install -U git+https://github.com/antmicro/kenning.git
```

Since Kenning can support various frameworks, and not all of them are required for user's use cases, the optional requirements are available as extra requirements.
The groups of extra requirements are following:

* `tensorflow` - modules for working with TensorFlow models (ONNX conversions, addons, and TensorFlow framework),
* `torch` - modules for working with PyTorch models,
* `mxnet` - modules for working with MXNet models,
* `nvidia_perf` - modules for performance measurements for NVIDIA GPUs,
* `object_detection` - modules for working with YOLOv3 object detection and Open Images Dataset V6 computer vision dataset.

To install the extra requirements, i.e. `tensorflow`, run:

```
sudo pip install git+https://github.com/antmicro/kenning.git#egg=kenning[tensorflow]
```

#### Working directly with the repository

For development purposes, and for usage of additional resources (as sample scripts or trained models), clone repository with:

```
git clone https://github.com/antmicro/kenning.git
```

To download model weights, install [Git Large File Storage](https://git-lfs.github.com>) (if not installed) and run::

```
cd kenning/
git lfs pull
```

### Kenning structure

The `kenning` module consists of the following submodules:

* `core` - provides interface APIs for datasets, models, optimizers, runtimes and runtime protocols,
* `datasets` - provides implementations for datasets,
* `modelwrappers` - provides implementations for models for various problems implemented in various frameworks,
* `optimizers` - provides implementations for compilers of deep learning models,
* `runtimes` - provides implementations of runtime on target devices,
* `runtimeprotocols` - provides implementations for communication protocols between host and tested target,
* `dataproviders` - provides implementations for reading input data from various sources, such as camera, directories or TCP connections,
* `outputcollectors` - provides implementations for processing outputs from models, i.e. saving results to file, or displaying predictions on screen.
* `onnxconverters` - provides ONNX conversions for a given framework along with a list of models to test the conversion on,
* `resources` - contains project's resources, like RST templates, or trained models,
* `scenarios` - contains executable scripts for running training, inference, benchmarks and other tests on target devices,
* `utils` - various functions and classes used in all above-mentioned submodules.

### Using Kenning as a library in Python scripts

Kenning is a regular Python module - after pip installation it can be used in Python scripts.
The example compilation of the model can look as follows:

```python
from kenning.datasets.pet_dataset import PetDataset
from kenning.modelwrappers.classification.tensorflow_pet_dataset import TensorFlowPetDatasetMobileNetV2
from kenning.compilers.tflite import TFLiteCompiler
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.core.measurements import MeasurementsCollector

dataset = PetDataset(
    root='./build/pet-dataset/',
    download_dataset=True
)
model = TensorFlowPetDatasetMobileNetV2(
    modelpath='./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5',
    dataset=dataset
)
compiler = TFLiteCompiler(
    dataset=dataset,
    compiled_model_path='./build/compiled-model.tflite',
    modelframework='keras',
    target='default',
    inferenceinputtype='float32',
    inferenceoutputtype='float32'
)
compiler.compile(
    inputmodelpath='./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5',
    inputshapes=model.get_input_spec()[0],
    dtype=model.get_input_spec()[1]
)
```

The above script downloads the dataset and compiles the model using TensorFlow Lite to the model with FP32 inputs and outputs.

To get a quantized model, replace `target`, `inferenceinputtype` and `inferenceoutputtype` to `int8`:

```python
...

compiler = TFLiteCompiler(
    dataset=dataset,
    compiled_model_path='./build/compiled-model.tflite',
    modelframework='keras',
    target='int8',
    inferenceinputtype='int8',
    inferenceoutputtype='int8',
    dataset_percentage=0.3
)
compiler.compile(
    inputmodelpath='./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5',
    inputshapes=model.get_input_spec()[0],
    dtype=model.get_input_spec()[1]
)
```

To check how the compiled model is performing, create `TFLiteRuntime` object and run local model evaluation:

```python
...

runtime = TFLiteRuntime(
    protocol=None,
    modelpath='./build/compiled-model.tflite'
)

runtime.run_locally(
    dataset,
    model,
    './build/compiled-model.tflite'
)
MeasurementsCollector.save_measurements('out.json')
```

Method `runtime.run_locally` runs benchmarks of the model on the current device.
The `MeasurementsCollector` class collects all benchmarks' data for the model inference and saves it in JSON format that can be later used to render results as described in [section on rendering report from benchmarks](#render-report-from-benchmarks).

### Using Kenning scenarios

One can also use ready-to-use Kenning scenarios.
All executable Python scripts are available in the `kenning.scenarios` submodule.

#### Running model training on host

The `kenning.scenarios.model_training` script is run as follows:

```
python -m kenning.scenarios.model_training \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    --logdir build/logs \
    --dataset-root build/pet-dataset \
    --model-path build/trained-model.h5 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --num-epochs 50
```

By default, `kenning.scenarios.model_training` script requires two classes:

* `ModelWrapper`-based class that describes model architecture and provides training routines,
* `Dataset`-based class that provides training data for the model.

The remaining arguments are provided by the `form_argparse` class methods in each class, and may be different based on selected dataset and model.
In order to get full help for the training scenario for the above case, run:

```
python -m kenning.scenarios.model_training \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    -h
```

This will load all the available arguments for a given model and dataset.

The arguments in the above command are:

* `--logdir` - path to the directory where logs will be stored (this directory may be an argument for the TensorBoard software),
* `--dataset-root` - path to the dataset directory, required by the `Dataset`-based class,
* `--model-path` - path where the trained model will be saved,
* `--batch-size` - training batch size,
* `--learning-rate` - training learning rate,
* `--num-epochs` - number of epochs.

If the dataset files are not present, use `--download-dataset` flag in order to let the Dataset API download the data.

#### Benchmarking trained model on host

The `kenning.scenarios.inference_performance` script runs the model using the deep learning framework used for training on a host device.
It runs the inference on a given dataset, computes model quality metrics and performance metrics.
The results from the script can be used as a reference point for benchmarking of the compiled models on target devices.

The example usage of the script is as follows:

```
python -m kenning.scenarios.inference_performance \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    build/result.json \
    --model-path kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --dataset-root build/pet-dataset
```

The obligatory arguments for the script are:

* `ModelWrapper`-based class that implements the model loading, I/O processing and inference method,
* `Dataset`-based class that implements fetching of data samples and evaluation of the model,
* `build/result.json`, which is the path to the output JSON file with benchmark results.

The remaining parameters are specific to the `ModelWrapper`-based class and `Dataset`-based class.

#### Testing ONNX conversions

The `kenning.scenarios.onnx_conversion` runs as follows:

```
python -m kenning.scenarios.onnx_conversion \
    build/models-directory \
    build/onnx-support.rst \
    --converters-list \
        kenning.onnxconverters.pytorch.PyTorchONNXConversion \
        kenning.onnxconverters.tensorflow.TensorFlowONNXConversion \
        kenning.onnxconverters.mxnet.MXNetONNXConversion
```

The first argument is the directory, where the generated ONNX models will be stored.
The second argument is the RST file with import/export support table for each model for each framework.
The third argument is the list of `ONNXConversion` classes implementing list of models, import method and export method.

#### Running compilation and deployment of models on target hardware

There are two scripts - `kenning.scenarios.inference_tester` and `kenning.scenarios.inference_server`.

The example call for the first script is following:

```
python -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.runtimes.tflite.TFLiteRuntime \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/google-coral-devboard-tflite-tensorflow.json \
    --modelcompiler-cls kenning.compilers.tflite.TFLiteCompiler \
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
```

The script requires:

* `ModelWrapper`-based class that implements model loading, I/O processing and optionally model conversion to ONNX format,
* `Runtime`-based class that implements data processing and the inference method for the compiled model on the target hardware,
* `Dataset`-based class that implements fetching of data samples and evaluation of the model,
* `./build/google-coral-devboard-tflite-tensorflow.json`, which is the path to the output JSON file with performance and quality metrics.

`--modelcompiler-cls Optimizer` can be additionaly provided to compile the model for a given target. If it is not provided, the `inference_tester` will run the model loaded by `ModelWrapper`.

In case of running inference on remote edge device, the `--protocol-cls RuntimeProtocol` also needs to be provided in order to provide communication protocol between the host and the target.
If `--protocol-cls` is not provided, the `inference_tester` will run inference on the host machine (which is useful for testing and comparison).

The remaining arguments come from the above-mentioned classes.
Their meaning is following:

* `--model-path` (`TensorFlowPetDatasetMobileNetV2` argument) is the path to the trained model that will be compiled and executed on the target hardware,
* `--model-framework` (`TFLiteCompiler` argument) tells the compiler what is the format of the file with the saved model (it tells which backend to use for parsing the model by the compiler),
* `--target` (`TFLiteCompiler` argument) is the name of the target hardware for which the compiler generates optimized binaries,
* `--compiled-model-path` (`TFLiteCompiler` argument) is the path where the compiled model will be stored on host,
* `--inference-input-type` (`TFLiteCompiler` argument) tells TFLite compiler what will be the type of the input tensors,
* `--inference-output-type` (`TFLiteCompiler` argument) tells TFLite compiler what will be the type of the output tensors,
* `--host` tells the `NetworkProtocol` what is the IP address of the target device,
* `--port` tells the `NetworkProtocol` on what port the server application is listening,
* `--packet-size` tells the `NetworkProtocol` what the packet size during communication should be,
* `--save-model-path` (`TFLiteRuntime` argument) is the path where the compiled model will be stored on the target device,
* `--dataset-root` (`PetDataset` argument) is the path to the dataset files,
* `--inference-batch-size` is the batch size for the inference on the target hardware,
* `--verbosity` is the verbosity of logs.

The example call for the second script is as follows:

```
python -m kenning.scenarios.inference_server \
    kenning.runtimeprotocols.network.NetworkProtocol \
    kenning.runtimes.tflite.TFLiteRuntime \
    --host 0.0.0.0 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --delegates-list libedgetpu.so.1 \
    --verbosity INFO
```

This script only requires `Runtime`-based class and `RuntimeProtocol`-based class.
It waits for a client using a given protocol, and later runs inference based on the implementation from the `Runtime` class.

The additional arguments are as follows:

* `--host` (`NetworkProtocol` argument) is the address where the server will listen,
* `--port` (`NetworkProtocol` argument) is the port on which the server will listen,
* `--packet-size` (`NetworkProtocol` argument) is the size of the packet,
* `--save-model-path` is the path where the received model will be saved,
* `--delegates-list` (`TFLiteRuntime` argument) is a TFLite-specific list of libraries for delegating the inference to deep learning accelerators (`libedgetpu.so.1` is the delegate for Google Coral TPUs).

First, the client compiles the model and sends it to the server using the runtime protocol.
Then, it sends next batches of data to process to the server.
In the end, it collects the benchmark metrics and saves them to JSON file.
In addition, it generates plots with performance changes over time.

#### Render report from benchmarks

The `kenning.scenarios.inference_performance` and `kenning.scenarios.inference_tester` create JSON files that contain:

* command string that was used to generate the JSON file,
* frameworks along with their versions used to train the model and compile the model,
* performance metrics, including:

  * CPU usage over time,
  * RAM usage over time,
  * GPU usage over time,
  * GPU memory usage over time,

* predictions and ground truth to compute quality metrics, i.e. in form of confusion matrix and top-5 accuracy for classification task.

The `kenning.scenarios.render_report` renders the report RST file along with plots for metrics for a given JSON file based on selected templates.

For example, for the file `./build/google-coral-devboard-tflite-tensorflow.json` created in [Running compilation and deployment of models on target hardware](#running-compilation-and-deployment-of-models-on-target-hardware) the report can be rendered as follows:

```
python -m kenning.scenarios.render_report \
    build/google-coral-devboard-tflite-tensorflow.json \
    "Pet Dataset classification using TFLite-compiled TensorFlow model" \
    docs/source/generated/google-coral-devboard-tpu-tflite-tensorflow-classification.rst \
    --img-dir docs/source/generated/img/ \
    --root-dir docs/source/ \
    --report-types \
        performance \
        classification
```

Where:

* `build/google-coral-devboard-tflite-tensorflow.json` is the input JSON file with benchmark results
* `"Pet Dataset classification using TFLite-compiled TensorFlow model"` is the report name that will be used as title in generated plots,
* `docs/source/generated/google-coral-devboard-tpu-tflite-tensorflow-classification.rst` is the path to the output RST file,
* `--img-dir docs/source/generated/img/` is the path to the directory where generated plots will be stored,
* `--root-dir docs/source` is the root directory for documentation sources (it will be used to compute relative paths in the RST file),
* `--report-types performance classification` is the list of report types that will form the final RST file.

The `performance` type provides report sections for performance metrics, i.e.:

* Inference time changes over time,
* Mean CPU usage over time,
* RAM usage over time,
* GPU usage over time,
* GPU memory usage over time.

It also computes mean, standard deviation and median values for the above time series.

The `classification` type provides report section regarding quality metrics for classification task:

* Confusion matrics,
* Per-class precision,
* Per-class sensitivity,
* Accuracy,
* Top-5 accuracy,
* Mean precision,
* Mean sensitivity,
* G-Mean.

The above metrics can be used to determine any quality losses resulting from optimizations (i.e. pruning or quantization).

### Adding new implementations

`Dataset`, `ModelWrapper`, `Optimizer`, `RuntimeProtocol`, `Runtime` and other classes from `kenning.core` module have dedicated directories for their implementations.
Each method in base classes that requires implementation raises `NotImplementedError` exception.
Implemented methods can be also overriden, if neccessary.

Most of the base classes implement `form_argparse` and `from_argparse` methods.
The first one creates an argument parser and a group of arguments specific to the base class.
The second one creates an object of the class based on the arguments from argument parser.

Inheriting classes can modify `form_argparse` and `from_argparse` methods to provide better control over their processing, but they should always be based on the results of their base implementations.
