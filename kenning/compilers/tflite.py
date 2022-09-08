"""
Wrapper for TensorFlow Lite deep learning compiler.
"""

import tensorflow as tf
from shutil import which
import subprocess
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List
import tensorflow_model_optimization as tfmot

from kenning.core.optimizer import IOSpecificationNotFoundError
from kenning.compilers.tensorflow_optimizers import TensorFlowOptimizer
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


class TFLiteCompiler(TensorFlowOptimizer):
    """
    The TFLite and EdgeTPU compiler.
    """

    outputtypes = [
        'tflite'
    ]

    inputtypes = {
        'keras': kerasconversion,
        'tensorflow': tensorflowconversion,
        'onnx': onnxconversion,
    }

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'onnx',
            'enum': list(inputtypes.keys())
        },
        'target': {
            'description': 'The TFLite target device scenario',
            'default': 'default',
            'enum': ['default', 'int8', 'float16', 'edgetpu']
        },
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
        'dataset_percentage': {
            'description': 'Tells how much data from dataset (from 0.0 to 1.0) will be used for calibration dataset',  # noqa: E501
            'type': float,
            'default': 0.25
        },
        'quantization_aware_training': {
            'description': 'Enable quantization aware training',
            'type': bool,
            'default': False
        },
        'use_tf_select_ops': {
            'description': 'Enable Tensorflow ops in model conversion (via SELECT_TF_OPS)',  # noqa: E501
            'type': bool,
            'default': False
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            target: str = 'default',
            epochs: int = 10,
            batch_size: int = 32,
            optimizer: str = 'adam',
            disable_from_logits: bool = False,
            modelframework: str = 'onnx',
            inferenceinputtype: str = 'float32',
            inferenceoutputtype: str = 'float32',
            dataset_percentage: float = 0.25,
            quantization_aware_training: bool = False,
            use_tf_select_ops: bool = False):
        """
        The TFLite and EdgeTPU compiler.

        Compiler converts input models to the .tflite format.
        It also can adapt .tflite models to work with the EdgeTPU devices, i.e.
        Google Coral devboard.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage
        compiled_model_path : Path
            Path where compiled model will be saved
        target : str
            Target accelerator on which the model will be executed
        epochs : int
            Number of epochs used for quantization aware training
        batch_size : int
            The size of a batch used for quantization aware training
        optimizer : str
            Optimizer used during the training
        disable_from_logits
            Determines whether output of the model is normalized
        modelframework : str
            Framework of the input model, used to select a proper backend
        inferenceinputtype : str
            Data type of the input layer
        inferenceoutputtype : str
            Data type of the output layer
        dataset_percentage : float
            If the dataset is used for optimization (quantization), the
            dataset_percentage determines how much of data samples is going
            to be used
        quantization_aware_training : bool
            Enables quantization aware training instead of a post-training
            quantization. If enabled the model has to be retrained.
        use_tf_select_ops : bool
            Enables adding SELECT_TF_OPS to the set of converter
            supported ops
        """
        self.target = target
        self.modelframework = modelframework
        self.inferenceinputtype = inferenceinputtype
        self.inferenceoutputtype = inferenceoutputtype
        self.set_input_type(modelframework)
        self.dataset_percentage = dataset_percentage
        self.quantization_aware_training = quantization_aware_training
        self.use_tf_select_ops = use_tf_select_ops
        super().__init__(dataset, compiled_model_path, epochs, batch_size, optimizer, disable_from_logits)  # noqa: E501

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.target,
            args.epochs,
            args.batch_size,
            args.optimizer,
            args.disable_from_logits,
            args.model_framework,
            args.inference_input_type,
            args.inference_output_type,
            args.dataset_percentage,
            args.quantization_aware_training,
            args.use_tf_select_ops
        )

    def compile(
            self,
            inputmodelpath: Path,
            io_spec: Optional[Dict[str, List[Dict]]] = None):

        if io_spec is None:
            io_spec = self.load_io_specification(inputmodelpath)

        if not io_spec or not io_spec['output'] or not io_spec['input']:
            raise IOSpecificationNotFoundError('No input/ouput specification found')  # noqa: E501

        from copy import deepcopy
        io_spec = deepcopy(io_spec)

        if self.quantization_aware_training:
            assert self.inputtype == 'keras'
            model = tf.keras.models.load_model(inputmodelpath)

            def annotate_model(layer):
                if isinstance(layer, tf.keras.layers.Dense):
                    return tfmot.quantization.keras.quantize_annotate_layer(
                        layer
                    )
                return layer

            quant_aware_annotate_model = tf.keras.models.clone_model(
                model,
                clone_function=annotate_model
            )

            pcqat_model = tfmot.quantization.keras.quantize_apply(
                quant_aware_annotate_model,
                tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True)  # noqa: E501
            )

            pcqat_model = self.train_model(pcqat_model)
            converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
        else:
            converter = self.inputtypes[self.inputtype](inputmodelpath)

        if self.target in ['int8', 'edgetpu']:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
        elif self.target == 'float16':
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

        if self.dataset is not None and self.target != 'default':
            def generator():
                for entry in self.dataset.calibration_dataset_generator(
                        self.dataset_percentage):
                    yield [np.array(entry, dtype=np.float32)]
            converter.representative_dataset = generator

        tflite_model = converter.convert()

        with open(self.compiled_model_path, 'wb') as f:
            f.write(tflite_model)

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

        update_io_spec(signature.get_input_details(), interpreter.get_input_details(), 'input')  # noqa: E501
        update_io_spec(signature.get_output_details(), interpreter.get_output_details(), 'output')  # noqa: E501

        self.save_io_specification(inputmodelpath, io_spec)

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
