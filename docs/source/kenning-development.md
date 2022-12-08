# Developing Kenning blocks

This chapter describes the development process of Kenning components.

## Model metadata

Since not all model formats supported by Kenning provide information about inputs and outputs or other data required for compilation or runtime purposes, each model processed by Kenning comes with a JSON file describing an I/O specification (and other useful metadata).

The JSON file with metadata for file `<model-name>.<ext>` is saved as `<model-name>.<ext>.json` in the same directory.

The example metadata file looks as follows (for ResNet50 for the ImageNet classification problem):

```json
{
    "input": [
        {
            "name": "input_0",
            "shape": [1, 224, 224, 3],
            "dtype": "int8",
            "order": 0,
            "scale": 1.0774157047271729,
            "zero_point": -13,
            "prequantized_dtype": "float32"
        }
    ],
    "output": [
        {
            "name": "output_0",
            "shape": [1, 1000],
            "dtype": "int8",
            "order": 0,
            "scale": 0.00390625,
            "zero_point": -128,
            "prequantized_dtype": "float32"
        }
    ]
}
```

The metadata file consists of:

* `input` list of structures describing inputs for the model,
* `output` list of structures describing outputs for the model.

Each input and output is described by the following parameters:

* `name` - input/output name.
* `shape` - input/output tensor shape.
* `dtype` - output type.
* `order` - some of the runtimes/compilers allow accessing inputs and outputs by id.
  This field describes the id of the current input/output.
* `scale` - scale parameter for the quantization purposes.
  Present only if the input/output requires quantization/dequantization.
* `zero_point` - zero point parameter for the quantization purposes.
  Present only if the input/output requires quantization/dequantization.
* `prequantized_dtype` - input/output data type before quantization.

The model metadata is used by all classes in Kenning for understanding the format of the inputs and outputs.
```{warning}
The [](optimizer-api) objects can affect the format of inputs and outputs, e.g. quantize the network. It is crucial to update the I/O specification when the current block modifies it.
```

## Implementing a new Kenning component

Firstly, check the [](kenning-api) for available building blocks for Kenning and their documentation.
Adding a new block to Kenning is a matter of creating a new class inheriting from one of `kenning.core` classes, providing configuration parameters, and implementing methods - at least the unimplemented ones.

For example purposes, let's create a sample [](optimizer-api)-based class, which will convert an input model to the TensorFlow Lite format.

First, let's create minimal code with the empty class:

```python
from kenning.core.optimizer import Optimizer


class TensorFlowLiteCompiler(Optimizer):
    pass
```

(defining-arguments-for-core-classes)=

### Defining arguments for core classes

Kenning classes can be created in three ways:

* Using constructor in Python,
* Using argparse from command-line (see [](cmd-usage)),
* Using JSON-based dictionaries from JSON scenarios (see [](json-scenarios))

To support all three methods, the newly implemented class requires creating a dictionary called `arguments_structure` that holds all configurable parameters of the class, along with their description, type and additional information.

This structure is used to create:

* an `argparse` group, to configure class parameters from terminal level (via the `form_argparse` method).
  Later, a class can be created with the `from_argparse` method.
* JSON schema to configure the class from the JSON file (via the `form_parameterschema` method).
  Later, a class can be created with the `from_parameterschema` method.

`arguments_structure` is a dictionary of form:

```python
arguments_structure = {
    'argument_name': {
        'description': 'Help for the argument',
        'type': str,
        'required': True
    }
}
```

The `argument_name` is a name used in:

* Python constructor,
* Argparse argument (in a form of `--argument-name`),
* JSON argument.

The fields describing the argument are following:

* `argparse_name` - if there is a need for a different flag in argparse, it can be provided here,
* `description` - description of the node, displayed during parsing error for JSON, or in help in case of command-line access
* `type` - type of argument, can be:

    * `Path` from `pathlib` module,
    * `str`
    * `float`
    * `int`
    * `bool`
* `default` - default value for the argument,
* `required` - boolean, tells if argument is required or not
* `enum` - a list of possible values for the argument,
* `is_list` - tells if argument is a list of objects of types given in `type` field,
* `nullable` - tells if argument can be empty (`None`).

Let's add parameters to the example class:

```python
from kenning.core.optimizer import Optimizer


class TensorFlowLiteCompiler(Optimizer):
    arguments_structure = {
        'inferenceinputtype': {
            'argparse_name': '--inference-input-type',
            'description': 'Data type of the input layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'inferenceoutputtype': {
            'argparse_name': '--inference-output-type',
            'description': 'Data type of the output layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'quantize_model': {
            'argparse_name': '--quantize-model',
            'description': 'Tells if model should be quantized',
            'type': bool,
            'default': False
        },
        'dataset_percentage': {
            'description': 'Tells how much data from dataset (from 0.0 to 1.0) will be used for calibration dataset',  # noqa: E501
            'type': float,
            'default': 0.25
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            inferenceinputtype: str = 'float32',
            inferenceoutputtype: str = 'float32',
            dataset_percentage: float = 0.25,
            quantize_model: bool = False):
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.dataset_percentage = dataset_percentage
        self.quantize_model = quantize_model
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
            args.quantize_model
        )
```

In addition to defined arguments, there are also default [](optimizer-api) arguments - the [](dataset-api) object and path to save the model (`compiled_model_path`).

Also, a `from_argparse` object creator is implemented, since there are additional parameters (`dataset`) to handle. Function `from_parameterschema` is created automatically.

The above implementation of arguments is common for all core classes.

### Defining supported output and input types

The Kenning classes for consecutive steps are meant to work in a seamless manner, which means providing various ways to pass the model from one class to another.

Usually, each class can accept multiple model input formats and provides at least one output format that can be accepted in other classes (except for `terminal` compilers, such as [Apache TVM](https://tvm.apache.org/) that compile model to runtime library).

The list of supported output formats is represented in a class with an `outputtypes` list:

```python
    outputtypes = [
        'tflite'
    ]
```

The supported input formats are delivered in a form of a dictionary, mapping the supported input type name to the function used to load a model:

```python
    inputtypes = {
        'keras': kerasconversion,
        'tensorflow': tensorflowconversion
    }
```

Let's update the code with supported types:

```python
from kenning.core.optimizer import Optimizer
import tensorflow as tf
from pathlib import Path


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter


def tensorflowconversion(modelpath: Path):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelpath)
    return converter


class TensorFlowLiteCompiler(Optimizer):
    arguments_structure = {
        'inferenceinputtype': {
            'argparse_name': '--inference-input-type',
            'description': 'Data type of the input layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'inferenceoutputtype': {
            'argparse_name': '--inference-output-type',
            'description': 'Data type of the output layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'quantize_model': {
            'argparse_name': '--quantize-model',
            'description': 'Tells if model should be quantized',
            'type': bool,
            'default': False
        },
        'dataset_percentage': {
            'description': 'Tells how much data from dataset (from 0.0 to 1.0) will be used for calibration dataset',  # noqa: E501
            'type': float,
            'default': 0.25
        }
    }

    outputtypes = [
        'tflite'
    ]

    inputtypes = {
        'keras': kerasconversion,
        'tensorflow': tensorflowconversion
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            inferenceinputtype: str = 'float32',
            inferenceoutputtype: str = 'float32',
            dataset_percentage: float = 0.25,
            quantize_model: bool = False):
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.dataset_percentage = dataset_percentage
        self.quantize_model = quantize_model
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
            args.quantize_model
        )
```

### Implementing unimplemented methods

The remaining aspect of developing Kenning classes is implementing unimplemented methods.
Unimplemented methods raise the `NotImplementedError`.

In case of the [](optimizer-api) class, it is the `compile` method and `get_framework_and_version`.

When implementing unimplemented methods, it is crucial to follow type hints both for inputs and outputs - more details can be found in the documentation for the method.
Sticking to the type hints ensures compatibility between blocks, which is required for a seamless connection between compilation components.

Let's finish the implementation of the [](optimizer-api)-based class:

```python
from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, List


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter


def tensorflowconversion(modelpath: Path):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelpath)
    return converter


class TensorFlowLiteCompiler(Optimizer):
    arguments_structure = {
        'inferenceinputtype': {
            'argparse_name': '--inference-input-type',
            'description': 'Data type of the input layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'inferenceoutputtype': {
            'argparse_name': '--inference-output-type',
            'description': 'Data type of the output layer',
            'default': 'float32',
            'enum': ['float32', 'int8', 'uint8']
        },
        'quantize_model': {
            'argparse_name': '--quantize-model',
            'description': 'Tells if model should be quantized',
            'type': bool,
            'default': False
        },
        'dataset_percentage': {
            'description': 'Tells how much data from dataset (from 0.0 to 1.0) will be used for calibration dataset',  # noqa: E501
            'type': float,
            'default': 0.25
        }
    }

    outputtypes = [
        'tflite'
    ]

    inputtypes = {
        'keras': kerasconversion,
        'tensorflow': tensorflowconversion
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            inferenceinputtype: str = 'float32',
            inferenceoutputtype: str = 'float32',
            dataset_percentage: float = 0.25,
            quantize_model: bool = False):
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.dataset_percentage = dataset_percentage
        self.quantize_model = quantize_model
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
            args.quantize_model
        )

    def compile(
            self,
            inputmodelpath: Path,
            io_spec: Optional[Dict[str, List[Dict]]] = None):

        # load I/O specification for the model
        if io_spec is None:
            io_spec = self.load_io_specification(inputmodelpath)

        # load the model using chosen input type
        converter = self.inputtypes[self.inputtype](inputmodelpath)

        # preparing model compilation using class arguments
        if self.quantize_model:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
        converter.inference_input_type = tf.as_dtype(self.inferenceinputtype)
        converter.inference_output_type = tf.as_dtype(self.inferenceoutputtype)

        # dataset can be used during compilation i.e. for calibration
        # purposes
        if self.dataset and self.quantize_model:
            def generator():
                for entry in self.dataset.calibration_dataset_generator(
                        self.dataset_percentage):
                    yield [np.array(entry, dtype=np.float32)]
            converter.representative_dataset = generator

        # compile and save the model
        tflite_model = converter.convert()
        with open(self.compiled_model_path, 'wb') as f:
            f.write(tflite_model)

        # update the I/O specification
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        signature = interpreter.get_signature_runner()

        def update_io_spec(sig_det, int_det, key):
            for order, spec in enumerate(io_spec[key]):
                old_name = spec['name']
                new_name = sig_det[old_name]['name']
                spec['name'] = new_name
                spec['order'] = order

            quantized = any([det['quantization'][0] != 0 for det in int_det])
            new_spec = []
            for det in int_det:
                spec = [
                    spec for spec in io_spec[key]
                    if det['name'] == spec['name']
                ][0]

                if quantized:
                    scale, zero_point = det['quantization']
                    spec['scale'] = scale
                    spec['zero_point'] = zero_point
                    spec['prequantized_dtype'] = spec['dtype']
                    spec['dtype'] = np.dtype(det['dtype']).name
                new_spec.append(spec)
            io_spec[key] = new_spec

        update_io_spec(
            signature.get_input_details(),
            interpreter.get_input_details(),
            'input'
        )
        update_io_spec(
            signature.get_output_details(),
            interpreter.get_output_details(),
            'output'
        )

        # save updated I/O specification
        self.save_io_specification(inputmodelpath, io_spec)

    def get_framework_and_version(self):
        return 'tensorflow', tf.__version__
```

There are several important things regarding the above code snippet:

* The information regarding inputs and outputs can be collected with the `self.load_io_specification` method, present in all classes.
* The information about the input format to use is delivered in the `self.inputtype` field - it is updated automatically by the function consulting the best supported format for previous and current block.
* If the I/O metadata is affected by the current block, it needs to be updated and saved along with the compiled model using the `self.save_io_specification` method.

### Using the implemented block

In the Python script, the above class can be used with other classes from the [](kenning-api) as is - it is a regular Python code.

To use the implemented block in the JSON scenario (as described in [](json-scenarios), the module implementing the class needs to be available from the current directory, or the path to the module needs to be added to the `PYTHONPATH` variable.

Let's assume that the class was implemented in `my_optimizer.py` file.
The scenario can look as follow:

```{code-block} json
---
emphasize-lines: 18-29
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
            "type": "my_optimizer.TensorFlowLiteCompiler",
            "parameters":
            {
                "compiled_model_path": "./build/compiled-model.tflite",
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
            "save_model_path": "./build/compiled-model.tflite"
        }
    }
}
```

The emphasized line demonstrates usage of implemented `TensorFlowLiteCompiler` from the `my_optimizer.py` script.

This sums up the Kenning development process.
