"""
Wrapper for IREE compiler
"""
from pathlib import Path
from typing import List, Optional
# from iree.compiler import tools as ireecmp
from iree.compiler import version
import re

from kenning.core.optimizer import Optimizer
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


def kerasconversion(model_path, input_shapes, dtype):
    import tensorflow as tf
    from iree.compiler import tf as ireetf

    # Calling the .fit() method of keras model taints the state of the model,
    # breaking the IREE compiler. Because of that, the workaround is needed.
    original_model = tf.keras.models.load_model(model_path)
    model = tf.keras.models.clone_model(original_model)
    model.set_weights(original_model.get_weights())
    del original_model

    inputspec = []
    for input_layer in model.inputs:
        inputspec.append(tf.TensorSpec(input_shapes[input_layer.name], dtype))

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


def tfconversion(model_path, input_shapes, dtype):
    import tensorflow as tf
    from iree.compiler import tf as ireetf
    model = tf.saved_model.load(model_path)

    ordered_shapes = input_shapes_dict_to_list(input_shapes)

    inputspec = []
    for shape in ordered_shapes:
        inputspec.append(tf.TensorSpec(shape, dtype))

    model.main = tf.function(
        input_signature=inputspec
    )(lambda *args: model(*args))
    return ireetf.compile_module(
        model, exported_names=['main'], import_only=True)


def tfliteconversion(model_path, input_shape, dtype):
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

    outputtypes = []

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
        'compiler-args': {
            'argaprse_name': '--compiler-args',
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
            modelframework: str,
            backend: str,
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
        modelframework : str
            Framework of the input model
        backend : str
            Backend on which the model will be executed
        compiled_args : List[str]
            Additional arguments for the compiler. Every options should be in a
            separate string, which should be formatted like this:
            <option>=<value>, or <option> for flags (example:
            'iree-cuda-llvm-target-arch=sm_60'). Full list of options can be
            listed by running 'iree-compile -h'.
        """

        self.model_load = self.inputtypes[modelframework]
        self.model_framework = modelframework
        self.backend = backend_convert.get(backend, backend)
        if compiler_args is not None:
            self.compiler_args = [f"--{option}" for option in compiler_args]
        else:
            self.compiler_args = []

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
            args.model_framework,
            args.backend
        )

    def compile(
            self,
            inputmodelpath: Path,
            io_specs: Optional[dict[list[dict]]] = None):

        # TODO: adapt it to the new serialization pipeline

        # imported_model = self.model_load(inputmodelpath, inputshapes, dtype)
        # compiled_buffer = ireecmp.compile_str(
        #     imported_model,
        #     input_type=self.compiler_input_type,
        #     extra_args=self.compiler_args,
        #     target_backends=[self.backend]
        # )

        # # When compiling TFLite model, IREE does not provide information
        # # regarding input signature from Python API. Manual passing of input
        # # shapes and dtype to the runtime is required.
        # shapes_list = input_shapes_dict_to_list(inputshapes)
        # model_dict = {
        #     'model': compiled_buffer,
        #     'shapes': shapes_list,
        #     'dtype': dtype
        # }
        # with open(self.compiled_model_path, "wb") as f:
        #     f.write(str(model_dict).encode("utf-8"))

    def get_framework_and_version(self):
        return "iree", version.VERSION
