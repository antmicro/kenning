## Kenning

Copyright (c) 2020-2022 [Antmicro](https://www.antmicro.com>)

![Kenning](img/kenninglogo.png)

Kenning is a framework for creating deployment flows and runtimes for Deep Neural Network applications on various target hardware.

[Kenning documentation](https://antmicro.github.io/kenning/) | [Core API](https://antmicro.github.io/kenning/kenning-api.html) | [kenning.ai](https://opensource.antmicro.com/projects/kenning)

Kenning aims towards providing modular execution blocks for:

* dataset management
* model training
* model optimization and compilation for a given target hardware
* model evaluation and performance reports
* running models using efficient runtimes on target device
* model input and output processing (i.e. normalizing frames or computing final predictions from model outputs)

that can be used seamlessly regardless of underlying frameworks for above-mentioned steps.

Kenning does not aim towards bringing yet another training or compilation framework for deep learning models - there are lots of mature and versatile frameworks that support certain models, training routines, optimization techniques, hardware platforms and other components crucial to the deployment flow.
Still, there is no framework that would support all of the models or target hardware devices - especially the support matrix between compilation frameworks and target hardware is extremely sparse.
This means that any change in the application, especially in hardware, may end up with a necessity to change the whole or significant part of the application flow.

Kenning addresses this issue by providing a unified API that focuses more on deployment tasks rather than their implementation - the developer decides which implementation for each task should be used, and Kenning allows to do it in a seamless way.
This way switching to another target platform results in most cases in a very small change in the code, instead of reimplementing larger parts of the project.
This way the Kenning can get the most out of the existing Deep Neural Network training and compilation frameworks.

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
* `object_detection` - modules for working with YOLOv3 object detection and Open Images Dataset V6 computer vision dataset,
* `iree` - modules for IREE compilation and runtime,
* `tvm` - modules for Apache TVM compilation and runtime,
* `onnxruntime` - modules for ONNX Runtime,
* `docs` - modules for generating documentation,
* `test` - modules for testing the Kenning framework.

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
* `compilers` - provides implementations for compilers and optimizers for deep learning models,
* `runtimes` - provides implementations of runtime on target devices,
* `runtimeprotocols` - provides implementations for communication protocols between host and tested target,
* `dataproviders` - provides implementations for reading input data from various sources, such as camera, directories or TCP connections,
* `outputcollectors` - provides implementations for processing outputs from models, i.e. saving results to file, or displaying predictions on screen.
* `onnxconverters` - provides ONNX conversions for a given framework along with a list of models to test the conversion on,
* `report` - provides methods for rendering reports,
* `drawing` - provides methods for rendering plots for reports,
* `resources` - contains project's resources, like RST templates, or trained models,
* `scenarios` - contains executable scripts for running training, inference, benchmarks and other tests on target devices,
* `utils` - various functions and classes used in all above-mentioned submodules,
* `tests` - submodules for framework testing.

The `core` classes that are used throughout the whole Kenning framework are:

* `Dataset` class - performs dataset downloading, preparation, dataset-specific input preprocessing (i.e. input file opening, normalization), output postprocessing and model evaluation,
* `ModelWrapper` class - trains the model, prepares the model, performs model-specific input preprocessing and output postprocessing, runs inference on host using native framework,
* `Optimizer` class - optimizes and compiles the model,
* `Runtime` class - loads the model, performs inference on compiled model, runs target-specific processing of inputs and outputs, and runs performance benchmarks,
* `RuntimeProtocol` class - implements the communication protocol between the host and the target,
* `DataProvider` class - implements providing data from such sources as camera, TCP connection or others for inference,
* `OutputCollector` class - implements parsing and utilizing data coming from inference (such as displaying the visualizations, sending the results to via TCP).

### Kenning usage

There are several ways to use Kenning:

* Using executable scripts from the `scenarios` submodule, configurable via JSON files (recommended approach)
* Using executable scripts from the `scenarios` submodule, configurable via command-line arguments
* Using Kenning as a Python module.

Kenning scenarios are executable scripts for:

* Training and benchmarking the model using the native framework (`model_training`),
* Optimizing and compiling the model for a target hardware (`json_inference_tester` and `inference_tester`),
* Benchmarking the model on the target hardware (`json_inference_tester`, `json_inference_server`, `inference_tester`, `inference_server`),
* Rendering performance and quality reports from benchmark data (`render_report`).

For more details on each of the above scenarios, check the [Kenning documentation](https://antmicro.github.io/kenning/).

### Example use case of Kenning - optimizing a classifier

Let's consider a simple scenario, where we want to optimize the inference time and memory usage of the classification model executed on x86 CPU.

For this we are going to use [`PetDataset`](https://github.com/antmicro/kenning/blob/main/kenning/datasets/pet_dataset.py) Dataset and [`TensorFlowPetDatasetMobileNetV2`](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/tensorflow_pet_dataset.py) ModelWrapper.

We will skip the training process The already trained model can be found in `kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5`.

However, the training of the above model can be performed using the following command:

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

#### Benchmarking model using the native framework

First of all, we want to check how the trained model performs using the native framework on CPU.

For this, we will use `json_inference_tester` scenario.
Scenarios are configured using the files written in JSON format.
In our case, the JSON file (called `native.json`) will look like this:

```json
{
    "model_wrapper":
    {
        "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
        "parameters":
        {
            "model_path": "./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5"
        }
    },
    "dataset":
    {
        "type": "kenning.datasets.pet_dataset.PetDataset",
        "parameters":
        {
            "dataset_root": "./build/pet-dataset",
            "download_dataset": true
        }
    }
}
```

This JSON provides a configuration for running the model natively and evaluating it against the defined Dataset.

For every class in the above JSON file there are two keys required: `type` which is a module path of our class and `parameters` which is used to provide arguments that are used to create the instances of our classes.

In `model_wrapper`, we specify the model used for evaluation - here it is MobileNetV2 trained on Pet Dataset. The `model_path` is a path to the saved model.
The `TensorFlowPetDatasetMobileNetV2` model wrapper provides methods for loading the model, preprocessing the inputs, postprocessing the outputs and running inference using native framework (TensorFlow in this case).

The dataset provided for evaluation is Pet Dataset - here we specify that we want to download the dataset (`download_dataset`) to the `./build/pet-dataset` directory (`dataset_root`).
The `PetDataset` class can download the dataset (if necessary), load it, read the inputs and outputs from files and process them, and implement the evaluation methods for the model.

Having the above config saved in the `native.json` file, run the `json_inference_tester` scenario:

```bash
python -m kenning.scenarios.json_inference_tester native.json build/native-out.json --verbosity INFO
```

This module runs the inference based on given configuration, evaluates the model and stores the quality and performance metrics in a JSON format, saved to `build/native-out.json` file.
All below configurations can be executed with this command.

To visualize the evaluation and benchmark results, run the `render_report` module:

```bash
python -m kenning.scenarios.render_report build/native-out.json 'native' build/benchmarks/native.rst --root-dir build/benchmarks --img-dir build/benchmarks/imgs --verbosity INFO --report-types performance classification
```

This module takes the output JSON file generated by the `json_inference_tester` module, and creates a report titled `native`, which is saved in the `build/benchmarks/native.rst` directory.
As specified in the `--report-types` flag, we create the sections in the report for the `performance` and `classification` metrics (for example, there is also a `detection` report type for object detection tasks).

In the `build/benchmarks/imgs` there will be images with the `native_*` prefix visualizing the confusion matrix, CPU and memory usage, as well as inference time.

The `build/benchmarks/native.rst` file is a ReStructuredText document containing the full report for the model - apart from linking to the generated visualizations, it provides aggregated information about the CPU and memory usage, as well as classification quality metrics, such as accuracy, sensitivity, precision, G-Mean.
Such file can be included in a larger, Sphinx-based documentation, which allows easy, automated report generation, using e.g. CI, as can be seen in the [Kenning documentation](https://antmicro.github.io/kenning/sample-report.html).

While native frameworks are great for training and inference, model design, training on GPUs and distributing training across many devices, e.g. in a cloud environment, for pure inference for production purposes we have a quite large variety of inference-focused frameworks that focus on getting the most out of hardware in order to get the results as fast as possible.

#### Optimizing the model using TensorFlow Lite

One of such frameworks is TensorFlow Lite - a lightweight library for inferring networks on edge - it has a small binary size, which can be even more reduced (by disabling unused operators), and a very optimized format of input models, called FlatBuffers.

Before the TensorFlow Lite Interpreter (runtime for the TensorFlow Lite library) can be used, the model first needs to be optimized and compiled to the `.tflite` format.

Let's add a TensorFlow Lite Optimizer that will convert our MobileNetV2 model to a FlatBuffer format, as well as TensorFlow Lite Runtime that will execute the model:

```{code-block} json
---
emphasize-lines: 16, 19-31, 32-39
---
{
    "model_wrapper":
    {
        "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
        "parameters":
        {
            "model_path": "./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5"
        }
    },
    "dataset":
    {
        "type": "kenning.datasets.pet_dataset.PetDataset",
        "parameters":
        {
            "dataset_root": "./build/pet-dataset",
            "download_dataset": false,
        }
    },
    "optimizers":
    [
        {
            "type": "kenning.compilers.tflite.TFLiteCompiler",
            "parameters":
            {
                "target": "default",
                "compiled_model_path": "./build/fp32.tflite",
                "inference_input_type": "float32",
                "inference_output_type": "float32"
            }
        }
    ],
    "runtime":
    {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters":
        {
            "save_model_path": "./build/fp32.tflite"
        }
    }
}
```

In the already existing blocks we only disable downloading the dataset - the `download_dataset` parameter can be also removed, since the dataset is not downloaded by default.

The first newly added thing is the presence of  the `optimizers` list - it allows us to add one or more objects inheriting from the `kenning.core.optimizer.Optimizer` class.
Optimizers read the model from the input file, apply various optimizations on them, and then save the optimized model to a new file.

In our current scenario we will use the `TFLiteCompiler` class - it reads the model in Keras-specific format, optimizes the model and saves it to the `./build/fp32.tflite` file.
Worth-noting here are the parameters of this particular Optimizer (each Optimizer usually has a different set of parameters):

* `target` - indicates what the desired target device (or model type) is, `default` is the regular CPU. Another example here could be `edgetpu`, which can compile models for Google Coral platform.
* `compiled_model_path` - indicates where the model should be saved.
* `inference_input_type` and `inference_output_type` - indicate what the input and output type of the model should be. Usually, all trained models use FP32 weights (32-bit floating point) and activations - using `float32` here keeps the weights unchanged.

The second thing that is added to the previous flow is the `runtime` block - it provides a class inheriting from the `kenning.core.runtime.Runtime` class that is able to load the final model and run inference on target hardware. Usually, each `Optimizer` has a corresponding `Runtime` that is able to run its results.

To compile the scenario (called `tflite-fp32.json`), run:

```bash
python -m kenning.scenarios.json_inference_tester tflite-fp32.json build/tflite-fp32-out.json --verbosity INFO

python -m kenning.scenarios.render_report build/tflite-fp32-out.json 'tflite-fp32' build/benchmarks/tflite-fp32.rst --root-dir build/benchmarks --img-dir build/benchmarks/imgs --verbosity INFO --report-types performance classification
```

While it depends on the used platform, there should be a significant improvement in both inference time (around 10-15 times faster model compared to the native model) and memory usage (around 2 times smaller output model).
What’s worth noting is that we get a significant improvement with no harm in the quality of the model - the outputs stay the same.

#### Quantizing the model using TensorFlow Lite

To further reduce the memory usage we can quantize the model - it is a process where all weights and activations in the model are calibrated to work with the `INT8` precision, instead of `FP32` precision.
While it may severely harm the quality of the predictions, with proper calibration the quality reduction can be negligible.

The model can be quantized during the compilation process in TensorFlow Lite.
With Kenning, it can be achieved with the following simple additions:

```{code-block} json
---
emphasize-lines: 24-27,36
---
{
    "model_wrapper":
    {
        "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
        "parameters":
        {
            "model_path": "./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5"
        }
    },
    "dataset":
    {
        "type": "kenning.datasets.pet_dataset.PetDataset",
        "parameters":
        {
            "dataset_root": "./build/pet-dataset"
        }
    },
    "optimizers":
    [
        {
            "type": "kenning.compilers.tflite.TFLiteCompiler",
            "parameters":
            {
                "target": "int8",
                "compiled_model_path": "./build/int8.tflite",
                "inference_input_type": "int8",
                "inference_output_type": "int8"
            }
        }
    ],
    "runtime":
    {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters":
        {
            "save_model_path": "./build/int8.tflite"
        }
    }
}
```

The only changes here in comparison to the previous configuration appear in the `TFLiteCompiler` configuration - we change `target`, `inference_input_type` and `inference_output_type` to `int8`.
What `TFLiteCompiler` does in the background is fetches the subset of images from the `PetDataset` object to calibrate the model, so the entire model calibration process happens automatically.

Let's run the above scenario (`tflite-int8.json`):

```bash
python -m kenning.scenarios.json_inference_tester tflite-int8.json build/tflite-int8-out.json --verbosity INFO

python -m kenning.scenarios.render_report build/tflite-int8-out.json 'tflite-int8' build/benchmarks/tflite-int8.rst --root-dir build/benchmarks --img-dir build/benchmarks/imgs --verbosity INFO --report-types performance classification
```

This results in over 7 times smaller model compared to the native model without the significant loss of accuracy, but there is no speed improvement.

#### Speeding up the inference with Apache TVM

To speed up the inference of the quantized model, we can utilize vector extensions in the x86 CPUs, more specifically AVX2.
For this, let's use the Apache TVM framework for compiling efficient runtimes for various hardware platforms.

The scenario looks like this:

```{code-block} json
---
emphasize-lines: 29-38,41-47
---
{
    "model_wrapper":
    {
        "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
        "parameters":
        {
            "model_path": "./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5"
        }
    },
    "dataset":
    {
        "type": "kenning.datasets.pet_dataset.PetDataset",
        "parameters":
        {
            "dataset_root": "./build/pet-dataset"
        }
    },
    "optimizers":
    [
        {
            "type": "kenning.compilers.tflite.TFLiteCompiler",
            "parameters":
            {
                "target": "int8",
                "compiled_model_path": "./build/int8.tflite",
                "inference_input_type": "int8",
                "inference_output_type": "int8"
            }
        },
        {
            "type": "kenning.compilers.tvm.TVMCompiler",
            "parameters": {
                "target": "llvm -mcpu=core-avx2",
                "opt_level": 3,
                "conv2d_data_layout": "NCHW",
                "compiled_model_path": "./build/int8_tvm.tar"
            }
        }
    ],
    "runtime":
    {
        "type": "kenning.runtimes.tvm.TVMRuntime",
        "parameters":
        {
            "save_model_path": "./build/int8_tvm.tar"
        }
    }
}

```

As it can be observed, addition of a new framework is just a matter of simply adding and configuring another optimizer and using the corresponding `Runtime` to the final `Optimizer`.

The `TVMCompiler`, with `llvm -mcpu=core-avx2` as the target optimizes and compiles the model to use vector extensions. The final result is a `.tar` file containing the shared object that implements the whole model.

Let's compile the scenario (`tvm-avx2-int8.json`):

```bash
python -m kenning.scenarios.json_inference_tester tvm-avx2-int8.json build/tvm-avx2-int8-out.json --verbosity INFO

python -m kenning.scenarios.render_report build/tvm-avx2-int8-out.json 'tvm-avx2-int8' build/benchmarks/tvm-avx2-int8.rst --root-dir build/benchmarks --img-dir build/benchmarks/imgs --verbosity INFO --report-types performance classification
```

This results in an over 40 times faster model compared to native implementation, with 3 times smaller size.

This shows how easily we can interconnect various frameworks and get the most out of the hardware using Kenning, while performing just minor alterations to the configuration file.

The summary of passes can be seen below:

|               | Speed boost | Accuracy     | Size reduction  |
|---------------|-------------|--------------|-----------------|
| native        |           1 | 0.9572730984 |               1 |
| tflite-fp32   | 15.79405698 | 0.9572730984 |     1.965973551 |
| tflite-int8   | 1.683232669 | 0.9519662539 |      7.02033412 |
| tvm-avx2-int8 | 41.61514549 | 0.9487005035 |     3.229375069 |

### Using Kenning as a library in Python scripts

Kenning is also a regular Python module - after pip installation it can be used in Python scripts.
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
model.save_io_specification(model.modelpath)
compiler = TFLiteCompiler(
    dataset=dataset,
    compiled_model_path='./build/compiled-model.tflite',
    modelframework='keras',
    target='default',
    inferenceinputtype='float32',
    inferenceoutputtype='float32'
)
compiler.compile(
    inputmodelpath='./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5'
)
```

The above script downloads the dataset and compiles the model using TensorFlow Lite to the model with FP32 inputs and outputs.

To get a quantized model, replace `target`, `inferenceinputtype` and `inferenceoutputtype` to `int8`:

```python
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
    inputmodelpath='./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5'
)
```

To check how the compiled model is performing, create `TFLiteRuntime` object and run local model evaluation:

```python
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

As it can be observed, all classes accessible from JSON files in the scenarios have the same configuration as the classes in above Python scripts.

### Adding new implementations

`Dataset`, `ModelWrapper`, `Optimizer`, `RuntimeProtocol`, `Runtime` and other classes from `kenning.core` module have dedicated directories for their implementations.
Each method in base classes that requires implementation raises `NotImplementedError` exception.
They can be easily implemented or extended, but they need to conform to certain rules, usually described in the source documentation.

For more details and examples on how Kenning framework can be adjusted and enhanced, follow the [Kenning documentation](https://antmicro.github.io/kenning/).
Implemented methods can be also overriden, if neccessary.

Most of the base classes implement `form_argparse` and `from_argparse` methods.
The first one creates an argument parser and a group of arguments specific to the base class.
The second one creates an object of the class based on the arguments from argument parser.

Inheriting classes can modify `form_argparse` and `from_argparse` methods to provide better control over their processing, but they should always be based on the results of their base implementations.
