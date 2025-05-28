# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ONNXConversion for TensorFlow models.
"""

from pathlib import Path

import onnx2tf
import tensorflow as tf
import tensorflow.keras.applications as apps
import tf2onnx

from kenning.core.onnxconversion import (
    ModelEntry,
    ONNXConversion,
    SupportStatus,
)


class TensorFlowONNXConversion(ONNXConversion):
    """
    Provides methods and test models for TensorFlow-ONNX support.
    """

    def __init__(self):
        super().__init__("tensorflow", tf.__version__)

    def prepare(self):
        self.add_entry(
            "DenseNet201",
            lambda: apps.DenseNet201(),
            tensor_spec=(
                tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),
            ),
        )
        self.add_entry(
            "MobileNetV2",
            lambda: apps.MobileNetV2(),
            tensor_spec=(
                tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),
            ),
        )
        self.add_entry(
            "ResNet50",
            lambda: apps.ResNet50(),
            tensor_spec=(
                tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),
            ),
        )
        # self.add_entry(
        #     'VGG16',
        #     lambda: apps.VGG16(),
        #     tensor_spec=(tf.TensorSpec(
        #         (None, 224, 224, 3),
        #         tf.float32,
        #         name="input"
        #     ),))

    def onnx_export(
        self, modelentry: ModelEntry, exportpath: Path
    ) -> SupportStatus:
        model = modelentry.modelgenerator()
        spec = modelentry.parameters["tensor_spec"]
        modelproto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, output_path=exportpath
        )
        del model
        return SupportStatus.SUPPORTED

    def onnx_import(
        self, modelentry: ModelEntry, importpath: Path
    ) -> SupportStatus:
        model = onnx2tf.convert(
            input_onnx_file_path=str(importpath),
            non_verbose=True,
        )

        spec = modelentry.parameters["tensor_spec"][0]
        inp = tf.random.normal(
            [s if s is not None else 1 for s in spec.shape], dtype=spec.dtype
        )
        model.run(inp)
        del model
        return SupportStatus.SUPPORTED
