# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TensorFlow Lite deep learning compiler.
"""

import subprocess
from pathlib import Path
from shutil import which
from typing import Dict, List, Literal, Optional

import numpy as np
import tensorflow as tf

from kenning.core.dataset import Dataset
from kenning.core.optimizer import IOSpecificationNotFoundError
from kenning.optimizers.tensorflow_optimizers import TensorFlowOptimizer
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class EdgeTPUCompilerError(Exception):
    """
    Exception occurs when edgetpu_compiler fails to compile the model.
    """

    pass


def kerasconversion(model_path: PathOrURI) -> tf.lite.TFLiteConverter:
    """
    Converts Keras file to TFLite format.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert

    Returns
    -------
    tf.lite.TFLiteConverter
        TFLite converter for model
    """
    model = tf.keras.models.load_model(str(model_path), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter


def tensorflowconversion(model_path: PathOrURI) -> tf.lite.TFLiteConverter:
    """
    Converts TensorFlow file to TFLite format.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert

    Returns
    -------
    tf.lite.TFLiteConverter
        TFLite converter for model
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    return converter


def onnxconversion(model_path: PathOrURI) -> tf.lite.TFLiteConverter:
    """
    Converts ONNX file to TFLite format.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert

    Returns
    -------
    tf.lite.TFLiteConverter
        TFLite converter for model
    """
    from datetime import datetime

    import onnx
    from onnx_tf.backend import prepare

    onnxmodel = onnx.load(str(model_path))
    model = prepare(onnxmodel)
    convertedpath = model_path.with_suffix(
        f'.{datetime.now().strftime("%Y%m%d-%H%M%S")}.pb'
    )
    model.export_graph(str(convertedpath))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(convertedpath))
    return converter


class TFLiteCompiler(TensorFlowOptimizer):
    """
    The TFLite and EdgeTPU compiler.
    """

    outputtypes = ["tflite"]

    inputtypes = {
        "keras": kerasconversion,
        "tensorflow": tensorflowconversion,
        "onnx": onnxconversion,
    }

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "onnx",
            "enum": list(inputtypes.keys()),
        },
        "target": {
            "description": "The TFLite target device scenario",
            "default": "default",
            "enum": ["default", "int8", "float16", "edgetpu"],
        },
        "inferenceinputtype": {
            "argparse_name": "--inference-input-type",
            "description": "Data type of the input layer",
            "default": "float32",
            "enum": ["float32", "int8", "uint8"],
        },
        "inferenceoutputtype": {
            "argparse_name": "--inference-output-type",
            "description": "Data type of the output layer",
            "default": "float32",
            "enum": ["float32", "int8", "uint8"],
        },
        "dataset_percentage": {
            "description": "Tells how much data from dataset (from 0.0 to 1.0) will be used for calibration dataset",  # noqa: E501
            "type": float,
            "default": 0.25,
        },
        "quantization_aware_training": {
            "description": "Enable quantization aware training",
            "type": bool,
            "default": False,
        },
        "use_tf_select_ops": {
            "description": "Enable Tensorflow ops in model conversion (via SELECT_TF_OPS)",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "resolver_template_path": {
            "description": """
            Path to the custom template for creating C-based file with `tflite::MicroMutableOpResolver` with used ops for a given model.

            When provided, it will be used instead of the default template located in `kenning.resources.templates` module
            (as `tflite_ops_resolver.h.template`).

            The template is a Jinja2 template, where names of used ops are provided via `opcode_names` list.

            Number of used ops can be established with `{{opcode_names|length}}`.
            Ops can be added to resolver like so:
            ```
            // ...
            tflite::MicroMutableOpResolver<{{opcode_names|length}}> resolver;
            // ...
            {%- for opcode in opcode_names %}
                g_tflite_resolver.Add{{opcode}}();
            {%- endfor %}
            // ...
            ```
            """,  # noqa: E501
            "type": ResourceURI,
            "default": None,
            "nullable": True,
        },
        "resolver_output_path": {
            "description": "Path where a C-based file with tflite::MicroMutableOpResolver with used ops for a given model should be saved",  # noqa: E501
            "type": ResourceURI,
            "default": None,
            "nullable": True,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        target: str = "default",
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = "adam",
        disable_from_logits: bool = False,
        save_to_zip: bool = False,
        model_framework: str = "onnx",
        inferenceinputtype: str = "float32",
        inferenceoutputtype: str = "float32",
        dataset_percentage: float = 0.25,
        quantization_aware_training: bool = False,
        use_tf_select_ops: bool = False,
        resolver_template_path: Optional[ResourceURI] = None,
        resolver_output_path: Optional[ResourceURI] = None,
    ):
        """
        The TFLite and EdgeTPU compiler.

        Compiler converts input models to the .tflite format.
        It also can adapt .tflite models to work with the EdgeTPU devices, i.e.
        Google Coral devboard.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        target : str
            Target accelerator on which the model will be executed.
        epochs : int
            Number of epochs used for quantization aware training.
        batch_size : int
            The size of a batch used for quantization aware training.
        optimizer : str
            Optimizer used during the training.
        disable_from_logits : bool
            Determines whether output of the model is normalized.
        save_to_zip : bool
            Determines whether optimized model should additionally be saved
            in ZIP format.
        model_framework : str
            Framework of the input model, used to select a proper backend. If
            set to "any", then the optimizer will try to derive model framework
            from file extension.
        inferenceinputtype : str
            Data type of the input layer.
        inferenceoutputtype : str
            Data type of the output layer.
        dataset_percentage : float
            If the dataset is used for optimization (quantization), the
            dataset_percentage determines how much of data samples is going
            to be used.
        quantization_aware_training : bool
            Enables quantization aware training instead of a post-training
            quantization. If enabled the model has to be retrained.
        use_tf_select_ops : bool
            Enables adding SELECT_TF_OPS to the set of converter
            supported ops.
        resolver_template_path: Optional[ResourceURI]
            When provided, points to a template with a template for creating
            C/header file with tflite::MicroMutableOpResolver
            with all ops necessary to run a given model
        resolver_output_path: Optional[ResourceURI]
            When provided, a C-based file with tflite::MicroMutableOpResolver
            with ops specific to the created model is saved to a specified
            path.
            If `resolver_template_path` is provided, it will be used as a
            template for the C file generation. Otherwise, default template
            will be used.
        """
        self.target = target
        self.model_framework = model_framework
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.set_input_type(model_framework)
        self.dataset_percentage = dataset_percentage
        self.quantization_aware_training = quantization_aware_training
        self.use_tf_select_ops = use_tf_select_ops
        self.resolver_template_path = resolver_template_path
        self.resolver_output_path = resolver_output_path
        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            disable_from_logits=disable_from_logits,
            save_to_zip=save_to_zip,
        )

    @staticmethod
    def create_resolver_file_from_ops_list(
        opcode_names: List[str],
        output_path: PathOrURI,
        resolver_template: Optional[PathOrURI] = None,
    ):
        """
        Creates a C file with tflite::MicroMutableOpResolver
        containing ops used by model delivered in model_obj.

        Parameters
        ----------
        opcode_names: List[str]
            Names of ops in pascal case, for example names check
            https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h
            (names without prefix `Add`)
        output_path: PathOrURI
            Path where the C file with resolver should be saved
        resolver_template: Optional[PathOrURI]
            Path to custom template for resolver file.
            When provided, it will be used instead of the default template
            located in `kenning.resources.templates` module
            (as `tflite_ops_resolver.h.template`).

            The template is a Jinja2 template, where names of used ops are
            provided via `opcode_names` list.

            Number of used ops can be established with
            `{{opcode_names|length}}`.
            Ops can be added to resolver like so:
            ```
            // ...
            tflite::MicroMutableOpResolver<{{opcode_names|length}}> resolver;
            // ...
            {%- for opcode in opcode_names %}
                g_tflite_resolver.Add{{opcode}}();
            {%- endfor %}
            // ...
            ```
        """
        from jinja2 import Template

        templatecontent = None

        if resolver_template is None:
            import sys

            from kenning.resources import templates

            if sys.version_info.minor < 9:
                from importlib_resources import path
            else:
                from importlib.resources import path

            with path(templates, "tflite_ops_resolver.h.template") as rpath:
                resolver_template = rpath

        with open(resolver_template, "r") as resolvertemplatefile:
            templatecontent = resolvertemplatefile.read()

        template = Template(templatecontent)

        content = template.render(opcode_names=opcode_names)

        with open(output_path, "w") as output:
            output.write(content)

    @staticmethod
    def create_resolver_file_from_model(
        modeldata: bytes,
        output_path: PathOrURI,
        resolver_template: Optional[PathOrURI] = None,
        additional_ops: Optional[List[str]] = None,
    ):
        """
        Creates a C file with tflite::MicroMutableOpResolver
        containing ops used by model delivered in model_obj.

        Parameters
        ----------
        modeldata: bytes
            TensorFlow Lite Flatbuffer data, read directly
            from file
        output_path: PathOrURI
            Path where the C file with resolver should be saved
        resolver_template: Optional[PathOrURI]
            Path to custom template for resolver file.
            When provided, it will be used instead of the default template
            located in `kenning.resources.templates` module
            (as `tflite_ops_resolver.h.template`).

            The template is a Jinja2 template, where names of used ops are
            provided via `opcode_names` list.

            Number of used ops can be established with
            `{{opcode_names|length}}`.
            Ops can be added to resolver like so:
            ```
            // ...
            tflite::MicroMutableOpResolver<{{opcode_names|length}}> resolver;
            // ...
            {%- for opcode in opcode_names %}
                g_tflite_resolver.Add{{opcode}}();
            {%- endfor %}
            // ...
            ```
        additional_ops: Optional[List[str]]
            Names of additional ops to ones in the model, in pascal case, for example names check
            https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h
            (names without prefix `Add`)
        """  # noqa: E501
        from tensorflow.lite.python import schema_py_generated

        def convert_to_pascal_case(opname: str):
            opstring = ""
            for part in str(opname).split("_"):
                if len(part) > 1:
                    if part[0].isalpha():
                        opstring += part.lower().capitalize()
                    else:
                        opstring += part.upper()
            opstring = opstring.replace("Lstm", "LSTM")
            opstring = opstring.replace("BatchMatmul", "BatchMatMul")
            return opstring

        modelobj = schema_py_generated.Model.GetRootAsModel(modeldata)
        model = schema_py_generated.ModelT.InitFromObj(modelobj)

        opcodenames = {
            opid: opname
            for opname, opid in schema_py_generated.BuiltinOperator.__dict__.items()  # noqa: E501
        }

        opcode_names = []
        for entry in sorted(model.operatorCodes, key=lambda c: c.builtinCode):
            opcode_names.append(
                convert_to_pascal_case(opcodenames[entry.builtinCode])
            )

        if additional_ops is not None:
            for op in additional_ops:
                if op not in opcode_names:
                    opcode_names.append(op)

        TFLiteCompiler.create_resolver_file_from_ops_list(
            opcode_names=opcode_names,
            output_path=output_path,
            resolver_template=resolver_template,
        )

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        import tensorflow_model_optimization as tfmot

        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        if not io_spec or not io_spec["output"] or not io_spec["input"]:
            raise IOSpecificationNotFoundError(
                "No input/ouput specification found"
            )

        from copy import deepcopy

        io_spec = deepcopy(io_spec)

        if self.quantization_aware_training:
            assert self.inputtype == "keras"
            model = tf.keras.models.load_model(str(input_model_path))

            def annotate_model(layer):
                if isinstance(layer, tf.keras.layers.Dense):
                    return tfmot.quantization.keras.quantize_annotate_layer(
                        layer
                    )
                return layer

            quant_aware_annotate_model = tf.keras.models.clone_model(
                model, clone_function=annotate_model
            )

            pcqat_model = tfmot.quantization.keras.quantize_apply(
                quant_aware_annotate_model,
                tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(
                    preserve_sparsity=True
                ),
            )

            pcqat_model = self.train_model(pcqat_model)
            converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
        else:
            input_type = self.get_input_type(input_model_path)

            converter = self.inputtypes[input_type](input_model_path)

        if self.target in ["int8", "edgetpu"]:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if self.inferenceinputtype in [
                "int8",
                "uint8",
            ] and self.inferenceinputtype in ["int8", "uint8"]:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
        elif self.target == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        else:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
        if self.use_tf_select_ops:
            converter.target_spec.supported_ops.append(
                tf.lite.OpsSet.SELECT_TF_OPS
            )
        converter.inference_input_type = tf.as_dtype(self.inferenceinputtype)
        converter.inference_output_type = tf.as_dtype(self.inferenceoutputtype)

        if self.dataset is not None and self.target != "default":

            def generator():
                for entry in self.dataset.calibration_dataset_generator(
                    self.dataset_percentage
                ):
                    yield [
                        np.array(entry, dtype=np.float32).reshape(
                            io_spec.get("processed_input", io_spec["input"])[
                                0
                            ]["shape"]
                        )
                    ]

            converter.representative_dataset = generator

        tflite_model = converter.convert()

        self.compiled_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.compiled_model_path, "wb") as f:
            f.write(tflite_model)
        if self.save_to_zip:
            self.compress_model_to_zip()

        if self.resolver_output_path is not None:
            TFLiteCompiler.create_resolver_file_from_model(
                modeldata=tflite_model,
                output_path=self.resolver_output_path,
                resolver_template=self.resolver_template_path,
            )

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        signature = interpreter.get_signature_runner()

        def update_io_spec(sig_det, int_det, key):
            for order, spec in enumerate(io_spec[key]):
                old_name = spec["name"]
                new_name = sig_det[old_name]["name"]
                spec["name"] = new_name
                spec["order"] = order

            quantized = any([det["quantization"][0] != 0 for det in int_det])
            new_spec = []
            for det in int_det:
                spec = [
                    spec
                    for spec in io_spec[key]
                    if det["name"] == spec["name"]
                ][0]

                if quantized:
                    scale, zero_point = det["quantization"]
                    spec["scale"] = scale
                    spec["zero_point"] = zero_point
                    spec["prequantized_dtype"] = spec["dtype"]
                    spec["dtype"] = np.dtype(det["dtype"]).name
                new_spec.append(spec)
            io_spec[key] = new_spec

        if "processed_input" not in io_spec:
            io_spec["processed_input"] = deepcopy(io_spec["input"])
        if "processed_output" not in io_spec:
            io_spec["processed_output"] = deepcopy(io_spec["output"])

        update_io_spec(
            signature.get_input_details(),
            interpreter.get_input_details(),
            "processed_input",
        )
        update_io_spec(
            signature.get_output_details(),
            interpreter.get_output_details(),
            "output",
        )

        self.save_io_specification(input_model_path, io_spec)

        if self.target == "edgetpu":
            edgetpu_compiler = which("edgetpu_compiler")
            if edgetpu_compiler is None:
                raise EdgeTPUCompilerError(
                    "edgetpu_compiler missing - check https://coral.ai/docs/edgetpu/compiler on how to install edgetpu_compiler"  # noqa: E501
                )
            returncode = subprocess.call(
                f"{edgetpu_compiler} {self.compiled_model_path}".split()
            )
            edgetpupath = Path(
                f"{Path(self.compiled_model_path).stem}_edgetpu.tflite"
            )
            if not edgetpupath.is_file():
                raise EdgeTPUCompilerError(
                    f"{self.compiled_model_path}_edgetpu.tflite not created"
                )

            edgetpupath.rename(self.compiled_model_path)

            return returncode
        return 0
