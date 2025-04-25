# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains simple Convolution Neural Network (CNN) model wrapper.

Compatible with AnomalyDetectionDataset.
"""

import os
from argparse import Namespace
from pathlib import Path
from typing import List, Literal, Optional

from sklearn import metrics

from kenning.cli.command_template import TRAIN
from kenning.core.platform import Platform
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.modelwrappers.anomaly_detection.cnn import (
    PyTorchAnomalyDetectionCNN,
)
from kenning.utils.class_loader import append_to_sys_path
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class Ai8xUnsupportedDevice(Exception):
    """
    Raised if platform, unsupported by ai8x-training framework, is used.
    """


class Ai8xAnomalyDetectionCNN(PyTorchAnomalyDetectionCNN):
    """
    Model wrapper for anomaly detection with CNN.

    It is compatible with AutoML flow.
    """

    # default_dataset
    arguments_structure = {
        "quantize_activation": {
            "description": "Whether activation should be quantized",
            "type": bool,
            "default": True,
            "subcommands": [TRAIN],
        },
        "qat_start_epoch": {
            "description": "From which epoch the quantization-aware training should begin",  # noqa: E501
            "type": int,
            "nullable": 0,
            "subcommands": [TRAIN],
        },
        "qat_weight_bits": {
            "description": "The number of bits, weights (and optionally activation) should be quantized to",  # noqa: E501
            "type": int,
            "enum": [1, 2, 4, 8],
            "default": 8,
            "subcommands": [TRAIN],
        },
        "qat_outlier_removal_z_score": {
            "description": "The z-score threshold for outlier removal during activation range calculation",  # noqa: E501
            "type": float,
            "default": 8.0,
            "subcommands": [TRAIN],
        },
        "ai8x_training_path": {
            "description": "Path to the ai8x-training tool",
            "type": Path,
            "nullable": True,
            "default": None,
        },
    }
    # model_class = "kenning.modelwrappers.anomaly_detection.models.cnn.AnomalyDetectionCNN"  # noqa: E501

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: AnomalyDetectionDataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        filters: List[int] = [8, 16],
        kernel_size: int = 3,
        conv_padding: int = 0,
        conv_stride: int = 1,
        conv_dilation: int = 1,
        conv_activation: Optional[Literal["ReLU", "Abs"]] = None,
        conv_batch_norm: Optional[Literal["Affine", "NoAffine"]] = None,
        pooling: Optional[Literal["Max", "Avg"]] = None,
        pool_size: int = 1,
        pool_stride: int = 2,
        pool_dilation: int = 1,
        fc_neurons: List[int] = [8],
        fc_activation: Optional[Literal["ReLU", "Abs"]] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        evaluate: bool = True,
        logdir: Optional[Path] = None,
        quantize_activation: bool = True,
        qat_start_epoch: int = 0,
        qat_weight_bits: int = 8,
        qat_outlier_removal_z_score: float = 8.0,
        ai8x_training_path: Optional[Path] = None,
    ):
        super().__init__(
            model_path=model_path,
            dataset=dataset,
            from_file=from_file,
            model_name=model_name,
            filters=filters,
            kernel_size=kernel_size,
            conv_padding=conv_padding,
            conv_stride=conv_stride,
            conv_dilation=conv_dilation,
            conv_activation=conv_activation,
            conv_batch_norm=conv_batch_norm,
            pooling=pooling,
            pool_size=pool_size,
            pool_stride=pool_stride,
            pool_dilation=pool_dilation,
            fc_neurons=fc_neurons,
            fc_activation=fc_activation,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            evaluate=evaluate,
            logdir=logdir,
        )

        self.quantize_activation = quantize_activation
        self.qat_start_epoch = qat_start_epoch
        self.qat_weight_bits = qat_weight_bits
        self.qat_outlier_removal_z_score = qat_outlier_removal_z_score
        self.qat_config = {
            "start_epoch": qat_start_epoch,
            "weight_bits": qat_weight_bits,
            "outlier_removal_z_score": qat_outlier_removal_z_score,
        }

        if ai8x_training_path is None and "AI8X_TRAINING_PATH" in os.environ:
            ai8x_training_path = Path(os.environ["AI8X_TRAINING_PATH"])

        if not ai8x_training_path:
            raise ValueError("ai8x_training_path not specified")
        if not ai8x_training_path.exists():
            raise FileNotFoundError(f"{ai8x_training_path} not found")

        self.ai8x_training_path = ai8x_training_path

    @staticmethod
    def _setup_device(platform: Platform, ai8x_training_path: Path):
        die_type = None
        if "max78000" in platform.name:
            die_type = 85
        elif "max78002" in platform.name:
            die_type = 87
        if die_type is None:
            raise Ai8xUnsupportedDevice(
                f"Platform {platform.name} is not supported"
            )

        with append_to_sys_path([ai8x_training_path]):
            import ai8x

        # Setup device type for ai8x-training framework
        ai8x.set_device(die_type, simulate=False, round_avg=False)

    def read_platform(self, platform: Platform):
        self._setup_device(platform, self.ai8x_training_path)

    @classmethod
    def model_params_from_context(
        cls,
        dataset: AnomalyDetectionDataset,
        platform: Optional[Platform] = None,
    ):
        if platform is not None:
            cls._setup_device(
                platform,
                cls.arguments_structure["ai8x_training_path"]["default"],
            )

        return {
            **super().model_params_from_context(dataset),
            "qat": True,
        } | {
            # Use default values as they can be overridden
            # in scenario definition
            key: cls.arguments_structure[key]["default"]
            for key in (
                "quantize_activation",
                "qat_weight_bits",
                "ai8x_training_path",
            )
        }

    def create_model_structure(self):
        super().create_model_structure(
            qat=True,
            quantize_activation=self.quantize_activation,
            qat_weight_bits=self.qat_weight_bits,
            ai8x_training_path=self.ai8x_training_path,
        )

    def prepare_model(self):
        if self.model_prepared:
            return None
        if self.from_file:
            with append_to_sys_path([self.ai8x_training_path]):
                self.load_model(self.model_path)
            self.model_prepared = True
        else:
            self.create_model_structure()
            self.model_prepared = True
            self.save_model(self.model_path)
        self.model.to(self.device)

    def save_model(
        self, model_path: PathOrURI, export_dict: Optional[bool] = None
    ):
        import torch

        self.prepare_model()
        if export_dict is None:
            export_dict = self.DEFAULT_SAVE_MODEL_EXPORT_DICT
        if export_dict:
            torch.save(self.model.to_pure_torch().state_dict(), model_path)
        else:
            torch.save(self.model.to_pure_torch(), model_path)

    def train_model(self):
        (
            train_loader,
            test_loader,
            criterion,
            opt,
            postprocess,
        ) = self._prepare_training()

        def epoch_start_hook(epoch: int):
            nonlocal opt
            if self.qat_start_epoch == epoch:  # TODO maybe move to start
                with append_to_sys_path([self.ai8x_training_path]):
                    import ai8x
                    from utils import model_wrapper

                KLogger.info("Starting QAT - adjusting the model")
                self.model, _, _ = model_wrapper.unwrap(self.model)
                ai8x.fuse_bn_layers(self.model)
                ai8x.pre_qat(
                    self.model,
                    train_loader,
                    Namespace(device=self.device),
                    self.qat_config,
                )
                opt = ai8x.update_optimizer(self.model, opt)
                ai8x.initiate_qat(self.model, self.qat_config)
                self.model.to(self.device)
                KLogger.info("Prepared the model for QAT")

        self._train_model(
            train_loader,
            test_loader,
            opt,
            criterion,
            postprocess,
            metrics.f1_score,
            epoch_start_hook=epoch_start_hook,
        )

    @classmethod
    def get_output_formats(cls):
        return ["ai8x"]
