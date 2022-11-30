# Defining optimization pipelines in Kenning

{{project}} blocks (specified in the [](kenning-api) can be configured either via command line (see [](cmd-usage)), or via configuration files, specified in JSON format.
The latter approach allows the user to create more advance and easy-to-reproduce scenarios for model deployment.
Most of all, various optimizers available through Kenning can be chained to utilize various optimizations to get better performing models.

One of the most commonly used scenarios in Kenning is optimizing and compiling the model.
It can be done using {{json_compilation_script}}.

## JSON specification

{{json_compilation_script}} takes the specification of optimization and testing flow in a JSON format.
The root element of the JSON file is a dictionary that can have the following keys:

* `model_wrapper` - **mandatory field**, accepts dictionary as a value that defines the [](modelwrapper-api) object for the deployed model (provides I/O processing, optionally model).
* `dataset` - **mandatory field**, accepts dictionary as a value that defines the [](dataset-api) object for model optimization and evaluation.
* `optimizers` - *optional field*, accepts list of dictionaries specifying sequence of [](optimizer-api)-based optimizations applied on the model.
* `runtime_protocol` - *optional field*, defines the [](runtimeprotocol-api) object used to communicate with the remote target platform.
* `runtime` - *optional field* (**required** when `optimizers` are provided), defines the [](runtime-api)-based object that will infer the model on target device.

Each dictionary in above fields consists of:

* `type` - appropriate class for the key,
* `parameters` - `type`-specific arguments for underlying class (see [](defining-arguments-for-core-classes)).

## Evaluating model using its native framework

The simplest JSON configuration looks as follows:

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

It only takes `model_wrapper` and `dataset`.
With this, the model will be loaded and evaluated using its native framework.

The used [](modelwrapper-api) is `TensorFlowPetDatasetMobileNetV2`, which is a MobileNetV2 model trained to classify 37 breeds of cats and dogs.
In `type` field we specify the full "path" to the class by specifying the module it is implemented in (`kenning.modelwrappers.classification.tensorflow_pet_dataset`) and the name of the class (`TensorFlowPetDatasetMobileNetV2`), in a Python-like format (dot-separated).

In `parameters`, the arguments specific to `TensorFlowPetDatasetMobileNetV2` are provided.
Looking at the arguments' specification, the available parameters are:

```python
# this argument structure is taken from kenning.core.model - it is inherited by child classes
arguments_structure = {
    'modelpath': {
        'argparse_name': '--model-path',
        'description': 'Path to the model',
        'type': Path,
        'required': True
    }
}
```

The only mandatory parameter here is `model_path`, which points to the file with the model.
It is a required argument.

The `dataset` used here is `PetDataset` - as previously, it is provided in a module-like format (`kenning.datasets.pet_dataset.PetDataset`). The parameters here are specified in `kenning.core.dataset.Dataset` (inherited) and `kenning.core.dataset.PetDataset`:

```python
arguments_structure = {
    # coming from kenning.core.dataset.Dataset
    'root': {
        'argparse_name': '--dataset-root',
        'description': 'Path to the dataset directory',
        'type': Path,
        'required': True
    },
    'batch_size': {
        'argparse_name': '--inference-batch-size',
        'description': 'The batch size for providing the input data',
        'type': int,
        'default': 1
    },
    'download_dataset': {
        'description': 'Downloads the dataset before taking any action',
        'type': bool,
        'default': False
    },
    # coming from kenning.datasets.pet_dataset.PetDataset
    'classify_by': {
        'argparse_name': '--classify-by',
        'description': 'Determines if classification should be performed by species or by breeds',
        'default': 'breeds',
        'enum': ['species', 'breeds']
    },
    'image_memory_layout': {
        'argparse_name': '--image-memory-layout',
        'description': 'Determines if images should be delivered in NHWC or NCHW format',
        'default': 'NHWC',
        'enum': ['NHWC', 'NCHW']
    }
}
```

As it can be observed, the parameters allow the user to:

* specify the dataset's location,
* download the dataset,
* configure data layout and batch size,
* configure anything specific for the dataset.

```{note}
For more details on defining parameters for Kenning core classes, check [](defining-arguments-for-core-classes).
```

If no `optimizers` or `runtime` is specified, the model is executed using the [](modelwrapper-api)'s `run_inference` method.
The dataset test data is passed through the model and evaluation metrics are collected.

To run the defined pipeline (assuming that the JSON file is under `pipeline.json`), run:

```bash
python -m kenning.scenarios.json_inference_tester pipeline.json measurements.json --verbosity INFO
```

The `measurements.json` file is the output of the {{json_compilation_script}} providing the measurements data.
It contains such information as:

* defined above JSON configuration,
* versions of used packages from core classes (such as `tensorflow`, `torch`, `tvm`, ...),
* available resource usage readings (CPU usag, GPU usage, memory usage, ...),
* data necessary for evaluation, such as predictions, confusion matrix, ...

Which can be later used in [](report-generation).

```{note}
Check {doc}`measurements` for more information.
```

## Optimizing the model and running it on the same device

The model optimization and deployment can be performed directly on target device, if device is able to perform the optimization steps.
It can be also used to check the outcome of certain optimization on desktop platform before deployment.

Optimizations and compilers used in the scenario are defined in the `optimizers` field.
This field accepts a list of optimizers - they are applied on the model in the same order as they are defined in the `optimizers` field.

For example, a model can be subjected to the following optimizations:

* Quantization of weights and activations using TensorFlow Lite.
* Conversion of data layout from NHWC to NCHW format using Apache TVM
* Compilation to x86 runtime with AVX2 vector extensions using Apache TVM.

The scenario in such case will look like this:

```{code-block} json
---
emphasize-lines: 18-47
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

As emphasized, the `optimizers` list is added, with two entries:

* block of type `kenning.compilers.tflite.TFLiteCompiler`, quantizing the model,
* block of type `kenning.compilers.tvm.TVMCompiler`, performing the rest of the optimization steps.

In `runtime` field, the TVM-specific runtime of type `kenning.runtimes.tvm.TVMRuntime` is used.

The first optimizer in the list reads the input model path from the [](modelwrapper-api)'s `model_path` field.
Each consecutive [](optimizer-api) reads the model from file saved by the previous [](optimizer-api).
In the simplest scenario, the model is saved to `compiled_model_path` path in each optimizer, and is fetched by the next [](optimizer-api).

In case the default output file type of the previous [](optimizer-api) is not supported by the next [](optimizer-api), the first common supported model format is determined and used to pass the model between optimizers.

In case no such format exists, the {{json_compilation_script}} returns an error.

```{note}
More details on input/output formats between [](optimizer-api) objects can be found in [](kenning-development).
```

The scenario can be executed as follows:

```bash
python -m kenning.scenarios.json_inference_tester scenario.json output.json
```

## Compiling the model and running it remotely

For some platforms, we cannot run Python script to evaluate or run the model to check its quality - the dataset is too large to fit in the storage, no libraries or compilation tools are available for the target platform, or the device does not have an operating system to run Python on.

In such case, it is possible to evaluate the system remotely using the [](runtimeprotocol-api) and ``kenning.scenarios.json_inference_server`` scenario.

For this use case we need two JSON files - one for configuring the inference server, and the second one for configuring ``kenning.scenarios.json_inference_tester``, which acts as a runtime client.

The client and the server may communicate via different means, protocols and interfaces - we can use TCP communication, UART communication and other.
It depends on the used [](runtimeprotocol-api).

Let's start with the client configuration by adding `runtime_protocol` entry:

```{code-block} json
---
emphasize-lines: 48-57
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
    },
    "runtime_protocol":
    {
        "type": "kenning.runtimeprotocols.network.NetworkProtocol",
        "parameters":
        {
            "host": "10.9.8.7",
            "port": 12345,
            "packet_size": 32768
        }
    }
}
```

In `runtime_protocol` entry, we specify a `kenning.runtimeprotocols.network.NetworkProtocol` and provide server's address (`host`), application port (`port`) ad packet size (`packet_size`).
The `runtime` block is still needed to perform runtime-specific data preprocessing and postprocessing in the client application (the server only infers data).

The configuration for the server looks as follows:

```{code-block} json
{
    "runtime":
    {
        "type": "kenning.runtimes.tvm.TVMRuntime",
        "parameters":
        {
            "save_model_path": "./build/compiled_model_server.tar"
        }
    },
    "runtime_protocol":
    {
        "type": "kenning.runtimeprotocols.network.NetworkProtocol",
        "parameters":
        {
            "host": "0.0.0.0",
            "port": 12345,
            "packet_size": 32768
        }
    }
}
```

In server only `runtime` and `runtime_protocol` need to be specified.
It uses `runtime_protocol` to receive requests from clients and `runtime` to run the tested models.

The remaining things are provided by the client - input data and model.
Direct outputs from the model are sent as is to the client so it can postprocess them and evaluate the model using the dataset.
Server also sends the measurements from its sensors in JSON format if it's able to collect and send them.

First, run the server so it will be available for the client:

```bash
python3 -m kenning.scenarios.json_inference_server \
    ./scripts/jsonconfigs/tflite-tvm-classification-server.json \
    --verbosity INFO
```

Secondly, run the client:

```bash
python3 -m kenning.scenarios.json_inference_tester \
    ./scripts/jsonconfigs/tflite-tvm-classification-client.json \
    ./build/tflite-tvm-classificationjson.json \
    --verbosity INFO
```

The rest of the flow is automated.
