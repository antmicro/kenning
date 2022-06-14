"""
Wrapper for IREE compiler
"""
from pathlib import Path
from typing import Dict, Tuple, List
import re

from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset


# TODO: Add support for tflite models
# TODO: check for dtypes other than float32 (possibly tied to tflite)

def tf_model_parse(model_path, input_shape, dtype):
    import tensorflow as tf
    try:
        model = tf.saved_model.load(model_path)
    except OSError:
        model = tf.keras.models.load_model(model_path)
    # TODO: adapt predict signature for multi-input models
    input_shape = list(input_shape.values())[0]

    class WrapperModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        @tf.function(input_signature=[tf.TensorSpec(input_shape, dtype)])
        def predict(self, x):
            return self.m(x, training=False)

    return WrapperModule()


def tflite_model_parse(model, input_shape, dtype):
    raise NotImplementedError  # TODO


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
        'tf': tf_model_parse,
        'tflite': tflite_model_parse
    }

    outputtypes = []

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'tf',
            'enum': list(inputtypes.keys())
        },
        'backend': {
            'argparse_name': '--backend',
            'description': 'Name of the backend that will run the compiled module',
            'required': True,
            'enum': list(backend_convert.keys())
        },
        'compiler-args': {
            'argaprse_name': '--compiler-args',
            'description': 'Additional options that are passed to compiler',
            'default': None,
            'is_list': True
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str,
            backend: str,
            compiler_args: List[str] = None):
        """
        IREE compiler

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
            separate string, which should be formatted like this: <option>=<value>,
            or <option> for flags (example: 'iree-cuda-llvm-target-arch=sm_60').
            Full list of options can be listed by running 'iree-compile -h'.
        """
        if modelframework == "tf":
            from iree.compiler import tf as ireecmp
            self.model_load = tf_model_parse
        elif modelframework == "tflite":
            from iree.compiler import tflite as ireecmp
            self.model_load = tflite_model_parse
        else:
            raise RuntimeError(f"Unsupported model_framework. Choose from {list(self.inputtypes.keys())}.")

        self.ireecmp = ireecmp
        self.model_framework = modelframework
        self.backend = backend_convert.get(backend, backend)
        if compiler_args is not None:
            self.compiler_args = [f"--{option}" for option in compiler_args]
        else:
            self.compiler_args = []
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
            inputshapes: Dict[str, Tuple[int, ...]],
            dtype: str = 'float32'):

        model = self.model_load(inputmodelpath, inputshapes, dtype)
        self.ireecmp.compile_module(
            model,
            output_file=self.compiled_model_path,
            extra_args=self.compiler_args,
            exported_names=['predict'],
            target_backends=[self.backend]
        )

    def get_framework_and_version(self):
        module_path = Path(self.ireecmp.__file__)
        version_text = (module_path.parents[1] / "version.py").read_text()
        version = re.search(r'VERSION = "[\d.]+"', version_text)
        return version.group(0).split()[-1].strip('"')
