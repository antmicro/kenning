"""
Wrapper for ONNX deep learning compiler.
"""

from pathlib import Path
import tensorflow as tf
import tf2onnx
import torch
import onnx
from typing import Optional

from kenning.core.dataset import Dataset
from kenning.core.optimizer import Optimizer, CompilationError


def kerasconversion(model_path, input_spec, output_names):
    model = tf.keras.models.load_model(model_path)

    input_spec = [tf.TensorSpec(
        spec['shape'],
        spec['dtype'],
        name=spec['name']
    ) for spec in input_spec]
    modelproto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec
    )

    return modelproto


def torchconversion(model_path, input_spec, output_names):
    dev = 'cpu'
    model = torch.load(model_path, map_location=dev)

    if not isinstance(model, torch.nn.Module):
        raise CompilationError(
            f'ONNX compiler expects the input data of type: torch.nn.Module, but got: {type(model).__name__}'  # noqa: E501
        )

    model.eval()

    input = tuple(torch.randn(
        spec['shape'],
        device=dev
    ) for spec in input_spec)

    import io
    mem_buffer = io.BytesIO()
    torch.onnx.export(
        model,
        input,
        mem_buffer,
        opset_version=11,
        input_names=[spec['name'] for spec in input_spec],
        output_names=output_names
    )
    onnx_model = onnx.load_model_from_string(mem_buffer.getvalue())
    return onnx_model


def tfliteconversion(model_path, input_spec, output_names):
    modelproto, _ = tf2onnx.convert.from_tflite(
        str(model_path),
        input_names=[input['name'] for input in input_spec],
        output_names=output_names
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
            io_specs: Optional[dict[list[dict]]] = None):

        if not io_specs:
            io_specs = self.load_io_specification(inputmodelpath)

        from copy import deepcopy
        io_specs = deepcopy(io_specs)

        input_spec = io_specs['input']
        output_spec = io_specs['output']

        try:
            output_names = [spec['name'] for spec in output_spec]
        except KeyError:
            output_names = None

        model = self.inputtypes[self.inputtype](
            inputmodelpath,
            input_spec,
            output_names
        )

        onnx.save(model, self.compiled_model_path)

        # update the io specification with names
        for spec, input in zip(input_spec, model.graph.input):
            spec['name'] = input.name

        for spec, output in zip(output_spec, model.graph.output):
            spec['name'] = output.name

        self.save_io_specification(
            inputmodelpath,
            {
                'input': input_spec,
                'output': output_spec
            }
        )
        return 0

    def get_framework_and_version(self):
        return ('onnx', onnx.__version__)
