"""
Wrapper for TensorFlow Lite deep learning compiler.
"""

import tensorflow as tf
from shutil import which
import subprocess
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset


class EdgeTPUCompilerError(Exception):
    """
    Exception occurs when edgetpu_compiler fails to compile the model.
    """
    pass


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter


def tensorflowconversion(modelpath: Path):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelpath)
    return converter


def onnxconversion(modelpath: Path):
    from onnx_tf.backend import prepare
    import onnx
    from datetime import datetime
    onnxmodel = onnx.load(modelpath)
    model = prepare(onnxmodel)
    convertedpath = str(Path(modelpath).with_suffix(f'.{datetime.now().strftime("%Y%m%d-%H%M%S")}.pb'))  # noqa: E501
    model.export_graph(convertedpath)
    converter = tf.lite.TFLiteConverter.from_saved_model(convertedpath)
    return converter


class TFLiteCompiler(Optimizer):
    """
    The TFLite and EdgeTPU compiler.
    """

    outputtypes = [
        'tflite'
    ]

    inputtypes = {
        'tensorflow': tensorflowconversion,
        'keras': kerasconversion,
        'onnx': onnxconversion,
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str,
            target: str,
            inferenceinputtype: str,
            inferenceoutputtype: str,
            dataset_percentage: float = 1.0):
        """
        The TFLite and EdgeTPU compiler.

        This model wrapper converts input models to the .tflite models.
        It also can adapt .tflite models to work with the EdgeTPU devices, i.e.
        Google Coral devboard.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage
        compiled_model_path : Path
            Path where compiled model will be saved
        modelframework : str
            Framework of the input model, used to select a proper backend
        target : str
            Target accelerator on which the model will be executed
        inferenceinputtype : str
            Data type of the input layer
        inferenceoutputtype : str
            Data type of the output layer
        dataset_percentage : float
            If the dataset is used for optimization (quantization), the
            dataset_percentage determines how much of data samples is going
            to be used
        """
        self.set_input_type(modelframework)
        self.target = target
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        super().__init__(dataset, compiled_model_path, dataset_percentage)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse(quantizes_model=True)
        group.add_argument(
            '--model-framework',
            help='The input type of the model, framework-wise',
            choices=cls.inputtypes.keys(),
            default='onnx'
        )
        group.add_argument(
            '--target',
            help='The TFLite target device scenario',
            # TODO this may require some alterations
            choices=['default', 'int8', 'edgetpu'],
            required=True
        )
        group.add_argument(
            '--inference-input-type',
            help='Data type of the input layer',
            choices=['float32', 'int8', 'uint8'],
            default='float32'
        )
        group.add_argument(
            '--inference-output-type',
            help='Data type of the output layer',
            choices=['float32', 'int8', 'uint8'],
            default='float32'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.model_framework,
            args.target,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
        )

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes: Dict[str, Tuple[int, ...]],
            dtype: str = 'float32'):
        converter = self.inputtypes[self.inputtype](inputmodelpath)
        self.inputdtype = dtype

        if self.target in ['int8', 'edgetpu']:
            self.inputdtype = 'int8'
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_opts = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
        else:
            converter.target_spec.supported_opts = [
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
        converter.inference_input_type = tf.as_dtype(self.inferenceinputtype)
        converter.inference_output_type = tf.as_dtype(self.inferenceoutputtype)

        if self.dataset is not None and self.target != 'default':
            def generator():
                for entry in self.dataset.calibration_dataset_generator(
                        self.dataset_percentage):
                    yield [np.array(entry, dtype=np.float32)]
            converter.representative_dataset = generator

        tflite_model = converter.convert()

        with open(self.compiled_model_path, 'wb') as f:
            f.write(tflite_model)

        if self.target == 'edgetpu':
            edgetpu_compiler = which('edgetpu_compiler')
            if edgetpu_compiler is None:
                raise EdgeTPUCompilerError(
                    'edgetpu_compiler missing - check https://coral.ai/docs/edgetpu/compiler on how to install edgetpu_compiler'  # noqa: E501
                )
            returncode = subprocess.call(
                f'{edgetpu_compiler} {self.compiled_model_path}'.split()
            )
            edgetpupath = Path(
                f'{Path(self.compiled_model_path).stem}_edgetpu.tflite'
            )
            if not edgetpupath.is_file():
                raise EdgeTPUCompilerError(
                    f'{self.compiled_model_path}_edgetpu.tflite not created'
                )

            edgetpupath.rename(self.compiled_model_path)

            return returncode
        return 0

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)

    def get_inputdtype(self) -> str:
        return self.inputdtype
