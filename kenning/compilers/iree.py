# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for IREE compiler
"""
from pathlib import Path
from typing import List, Optional, Dict
from iree.compiler import tools as ireecmp
from iree.compiler import version
import re

from kenning.core.optimizer import Optimizer, IOSpecificationNotFoundError
from kenning.core.dataset import Dataset


def input_shapes_dict_to_list(inputshapes):
    """
    Turn the dictionary of 'name':'shape' of every input layer to ordered list.
    The order of input layers is inferred from names. It is assumed that the
    name of every input layer contains single ID number, and the order of the
    inputs are according to their IDs.

    Parameters
    ----------
    inputshapes : Dict[str, Tuple[int, ...]]
        inputshapes argument of IREECompiler.compile method

    Returns
    -------
    List[Tuple[int, ...]] :
        Shapes of each input layer in order
    """

    layer_order = {}
    for name in inputshapes.keys():
        layer_id = int(re.search(r"\d+", name).group(0))
        layer_order[name] = layer_id
    ordered_layers = sorted(list(inputshapes.keys()), key=layer_order.get)
    return [inputshapes[layer] for layer in ordered_layers]


def kerasconversion(model_path, input_spec):
    import tensorflow as tf
    from iree.compiler import tf as ireetf

    # Calling the .fit() method of keras model taints the state of the model,
    # breaking the IREE compiler. Because of that, the workaround is needed.
    original_model = tf.keras.models.load_model(model_path)
    model = tf.keras.models.clone_model(original_model)
    model.set_weights(original_model.get_weights())
    del original_model

    inputspec = [tf.TensorSpec(
        spec['shape'],
        spec['dtype']
    ) for spec in input_spec]

    class WrapperModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.m = model
            self.m.main = lambda *args: self.m(*args, training=False)
            self.main = tf.function(
                input_signature=inputspec
            )(self.m.main)

    return ireetf.compile_module(
        WrapperModule(), exported_names=['main'], import_only=True)


def tfconversion(model_path, input_spec):
    import tensorflow as tf
    from iree.compiler import tf as ireetf
    model = tf.saved_model.load(model_path)

    inputspec = [tf.TensorSpec(
        spec['shape'],
        spec['dtype']
    ) for spec in input_spec]

    model.main = tf.function(
        input_signature=inputspec
    )(lambda *args: model(*args))
    return ireetf.compile_module(
        model, exported_names=['main'], import_only=True)


def tfliteconversion(model_path, input_spec):
    from iree.compiler import tflite as ireetflite

    return ireetflite.compile_file(model_path, import_only=True)


backend_convert = {
    # CPU backends
    'dylib': 'dylib-llvm-aot',
    'vmvx': 'vmvx',
    # GPU backends
    'vulkan': 'vulkan-spirv',
    'cuda': 'cuda'
}


class IREECompiler(Optimizer):
    """
    IREE compiler
    """

    inputtypes = {
        'keras': kerasconversion,
        'tf': tfconversion,
        'tflite': tfliteconversion
    }

    outputtypes = ['iree']

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'keras',
            'enum': list(inputtypes.keys())
        },
        'backend': {
            'argparse_name': '--backend',
            'description': 'Name of the backend that will run the compiled module',  # noqa: E501
            'required': True,
            'enum': list(backend_convert.keys())
        },
        'compiler_args': {
            'argparse_name': '--compiler-args',
            'description': 'Additional options that are passed to compiler',
            'default': None,
            'is_list': True,
            'nullable': True
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            backend: str,
            modelframework: str = 'keras',
            compiler_args: Optional[List[str]] = None):
        """
        Wrapper for IREE compiler

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage
        compiled_model_path : Path
            Path where compiled model will be saved
        backend : str
            Backend on which the model will be executed
        modelframework : str
            Framework of the input model
        compiler_args : List[str]
            Additional arguments for the compiler. Every options should be in a
            separate string, which should be formatted like this:
            <option>=<value>, or <option> for flags (example:
            'iree-cuda-llvm-target-arch=sm_60'). Full list of options can be
            listed by running 'iree-compile -h'.
        """

        self.modelframework = modelframework
        self.set_input_type(modelframework)
        self.backend = backend
        self.compiler_args = compiler_args

        self.converted_backend = backend_convert.get(backend, backend)
        if compiler_args is not None:
            self.parsed_compiler_args = [
                f"--{option}" for option in compiler_args
            ]
        else:
            self.parsed_compiler_args = []

        if modelframework in ("keras", "tf"):
            self.compiler_input_type = "mhlo"
        elif modelframework == "tflite":
            self.compiler_input_type = "tosa"

        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.backend,
            args.model_framework,
            args.compiler_args
        )

    def compile(
            self,
            inputmodelpath: Path,
            io_spec: Optional[Dict[str, List[Dict]]] = None):
        if io_spec is None:
            io_spec = self.load_io_specification(inputmodelpath)

        try:
            input_spec = io_spec['input']
        except (TypeError, KeyError):
            raise IOSpecificationNotFoundError('No input specification found')

        self.model_load = self.inputtypes[self.inputtype]
        imported_model = self.model_load(inputmodelpath, input_spec)
        compiled_buffer = ireecmp.compile_str(
            imported_model,
            input_type=self.compiler_input_type,
            extra_args=self.parsed_compiler_args,
            target_backends=[self.converted_backend]
        )

        with open(self.compiled_model_path, "wb") as f:
            f.write(compiled_buffer)
        self.save_io_specification(inputmodelpath, io_spec)

    def get_framework_and_version(self):
        return "iree", version.VERSION
