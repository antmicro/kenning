"""
Wrapper for ONNX deep learning compiler.
"""

from pathlib import Path
import tensorflow as tf
import tf2onnx
import torch
import onnx

from kenning.core.dataset import Dataset
from kenning.core.optimizer import Optimizer, CompilationError


def kerasconversion(model_path, input_shapes, dtype):
    model = tf.keras.models.load_model(model_path)

    input_spec = [tf.TensorSpec(
        input_shapes[list(input_shapes.keys())[0]],
        dtype,
        name=list(input_shapes.keys())[0]
    )]

    modelproto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec
    )

    return modelproto


def torchconversion(model_path, input_shapes, dtype):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=dev)

    if not isinstance(model, torch.nn.Module):
        raise CompilationError(
            f'TVM compiler expects the input data of type: torch.nn.Module, but got: {type(model).__name__}'  # noqa: E501
        )

    input = torch.randn(
        input_shapes[list(input_shapes.keys())[0]],
        device=dev
    )
    traced_module = torch.jit.trace(model, input)

    import io
    mem_buffer = io.BytesIO()
    torch.onnx.export(traced_module, input, mem_buffer)
    onnx_model = onnx.load_model_from_string(mem_buffer.getvalue())
    return onnx_model


def tfliteconversion(model_path, input_shapes, dtype):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    modelproto, _ = tf2onnx.convert.from_tflite(
        str(model_path),
        input_names=[input['name'] for input in input_details],
        output_names=[output['name'] for output in output_details]
    )

    return modelproto


class ONNXCompiler(Optimizer):
    """
    The ONNX compiler.
    """

    outputtypes = [
        'onnx'
    ]

    inputtypes = {
        'keras': kerasconversion,
        'torch': torchconversion,
        'tflite': tfliteconversion
    }

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'keras',
            'enum': list(inputtypes.keys())
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str = 'keras'):
        """
        The ONNX compiler.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model
        compiled_model_path : Path
            Path where compiled model will be saved
        modelframework : str
            Framework of the input model, used to select a proper backend
        """
        self.modelframework = modelframework
        self.set_input_type(modelframework)
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.model_framework
        )

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes,
            dtype: str = 'float32'):
        self.inputdtype = dtype

        model = self.inputtypes[self.inputtype](
            inputmodelpath,
            inputshapes,
            dtype
        )

        onnx.save(model, self.compiled_model_path)

        return 0

    def get_framework_and_version(self):
        return ('onnx', onnx.__version__)
