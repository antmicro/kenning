# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for YOLOv4 model.

Allows for transfer learning of models based on 'OpenImageDatasetV6' for
selected classes only.

This wrapper uses PyTorch implementation of YOLOv4:
kenning/modelwrappers/frameworks/helpers/pytorch_yolov4_impl.py
which is based on implementation in this repository:
https://github.com/Tianxiaomo/pytorch-YOLOv4
"""

from __future__ import annotations

import copy
import re
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    TrainingParametersMissingError,
)
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.modelwrappers.frameworks.helpers.pytorch_yolov4_impl import Yolov4
from kenning.modelwrappers.object_detection.yolov4 import ONNXYOLOV4
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI, ResourceURI

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


def _freeze_bn(module):
    import torch.nn as nn

    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        module.weight.requires_grad_(False)
        module.bias.requires_grad_(False)


class YOLOV4TL(ONNXYOLOV4):
    """
    Model wrapper for YOLOv4 that supports transfer learning.
    """

    onnx_pretrained_model_uri = "kenning:///models/detection/yolov4.onnx"
    pretrained_model_uri = "kenning:///models/detection/yolov4.pth"
    default_dataset = COCODataset2017
    arguments_structure = {
        "save_model_path": {
            "argparse_name": "--save-model-path",
            "description": "During training, path where the fine-tuned model will be saved.",  # noqa: E501
            "type": ResourceURI,
            "default": None,
            "subcommands": [TRAIN],
        },
        "reset_last_layers": {
            "argparse_name": "--reset-last-layers",
            "description": "Resets last layers of the model for fine-tuning.",
            "type": bool,
            "default": False,
            "subcommands": [TRAIN],
        },
        "backbone_layers_to_freeze": {
            "argparse_name": "--backbone_layers_to_freeze",
            "description": "Regex names of layers that will be freezen during the first part of fine-tuning.",  # noqa: E501
            "type": List[str],
            "default": [],
            "subcommands": [TRAIN],
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used.",  # noqa: E501
            "type": int,
            "default": None,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "description": "Learning rate for training.",
            "type": float,
            "default": None,
            "subcommands": [TRAIN],
        },
        "head_num_epochs": {
            "argparse_name": "--head-num-epochs",
            "description": "Number of epochs for training only the head of Yolov4 with backbone being freezen.",  # noqa: E501
            "type": int,
            "default": 5,
            "subcommands": [TRAIN],
        },
        "full_num_epochs": {
            "argparse_name": "--full-num-epochs",
            "description": "Number of epochs to train the full model for.",
            "type": int,
            "default": 10,
            "subcommands": [TRAIN],
        },
        "logdir": {
            "argparse_name": "--logdir",
            "description": "Path to the logging directory.",
            "type": Path,
            "default": None,
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file=True,
        model_name: Optional[str] = None,
        class_names: str = "coco",
        save_model_path: Optional[PathOrURI] = None,
        backbone_layers_to_freeze: List[str] = [],
        reset_last_layers: Optional[bool] = None,
        batch_size: int = 1,
        learning_rate: Optional[float] = None,
        head_num_epochs: int = 1,
        full_num_epochs: int = 1,
        logdir: Optional[Path] = None,
    ):
        self._device = None
        if save_model_path:
            self.save_model_path = save_model_path
        else:
            self.save_model_path = model_path
        self.reset_last_layers = reset_last_layers

        super().__init__(
            model_path,
            dataset,
            from_file,
            model_name,
            class_names,
        )

        self.backbone_layers_to_freeze = backbone_layers_to_freeze
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.head_num_epochs = head_num_epochs
        self.full_num_epochs = full_num_epochs
        self.logdir = logdir

    @property
    def device(self):
        import torch

        if self._device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._device

    def freeze_backbone(self):
        """
        Freezes model layers specified by the regex patterns, which are
        provided in a scenario/config.
        """
        for name, sub_module in self.model.named_modules():
            if any(
                re.search(layer_re, name)
                for layer_re in self.backbone_layers_to_freeze
            ):
                KLogger.debug(f"Freezing {name}")
                for param in sub_module.parameters():
                    param.requires_grad = False

        self.model.apply(_freeze_bn)

    def unfreeze(self):
        """
        Unfreezes models weights.
        """
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def create_model_structure(self, model_path: Optional[ResourceURI] = None):
        self.model = Yolov4(n_classes=self.numclasses)

    def load_pretrained_torch_model(self, model_path: PathOrURI):
        import torch

        weights = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        dst_sd = self.model.state_dict()
        new_sd = dst_sd.copy()

        used = set()

        for dst_name, dst_tensor in dst_sd.items():
            for src_name, src_tensor in weights.items():
                if src_name in used:
                    continue

                if src_tensor.shape == dst_tensor.shape:
                    new_sd[dst_name] = src_tensor
                    used.add(src_name)
                    break

        self.model.load_state_dict(new_sd, strict=False)

    def load_torch_model(self, model_path: PathOrURI):
        import torch

        weights = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        if weights:
            self.model.load_state_dict(copy.deepcopy(weights))
        self.model.eval()

    def prepare_model(self):
        if self.model_prepared:
            return None
        # Load parameters from ONNX model, still needed for data processing
        super().load_model(ResourceURI(self.onnx_pretrained_model_uri))
        self.create_model_structure()

        if self.reset_last_layers:
            self.load_pretrained_torch_model(
                ResourceURI(self.pretrained_model_uri)
            )
            self.save_model(self.save_model_path)
        else:
            try:
                self.load_torch_model(self.model_path)
            except FileNotFoundError:
                # TODO: add downloading pretrained pytorch weights
                self.load_pretrained_torch_model(self.model_path)
        self.model.to(self.device)
        self.model_prepared = True

    def preprocess_input(self, X: List[np.array]) -> List[np.array]:
        return [x[:, ::-1, :, :] for x in X]

    def save_model(
        self, model_path: PathOrURI, export_dict: Optional[bool] = None
    ):
        import torch

        model = self.model
        KLogger.info(f"saving model to {model_path}")

        torch.save(model.state_dict(), model_path)

    def _train_model(
        self,
        num_epochs: int,
        opt: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr_scheduler: Optional[torch.optim.lr_scheduler],
        log_train: Optional[TextIOWrapper],
        log_eval: Optional[TextIOWrapper],
        best_loss: float = float("inf"),
    ) -> float:
        """
        General training loop for YOLOv4.

        Parameters
        ----------
        num_epochs : int
            Number of training epochs.
        opt : Optimizer
            The model optimizer.
        criterion : Callable[[torch.Tensor], torch.Tensor]
            The function calculation loss.
        train_loader : DataLoader
            Train dataset DataLoader
        val_loader : DataLoader
            Validation dataset DataLoader
        lr_scheduler : Optional[torch.optim.lr_scheduler]
            Scheduler for learning rate.
        log_train : Optional[TextIOWrapper]
            File for logging training statistics
        log_eval : Optional[TextIOWrapper]
            File for logging validation statistics
        best_loss : float
            Best loss that was achieved previously.

        Returns
        -------
        float
            Value of the best loss
        """
        import torch
        import torch.nn as nn

        for epoch in range(num_epochs):
            self.model.train()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            epoch_loss = 0
            loss_count = 0

            with LoggerProgressBar() as logger_progress_bar:
                bar = tqdm(train_loader, **logger_progress_bar.kwargs)
                for input, labels in bar:
                    opt.zero_grad()

                    outputs = self.model(input)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                    opt.step()

                    epoch_loss += loss.item()
                    loss_count += 1
                    bar.set_description(
                        f"train epoch: {epoch:3d} loss: "
                        f"{epoch_loss / loss_count:.3f}"
                    )
            if log_train:
                log_train.write(f"{epoch_loss / loss_count:.4f}\n")
                log_train.flush()

            if lr_scheduler:
                lr_scheduler.step()

            # Evaluate model on validation set
            eval_loss = 0
            eval_loss_count = 0
            with torch.no_grad():
                self.model.eval()
                with LoggerProgressBar() as logger_progress_bar:
                    bar = tqdm(val_loader, **logger_progress_bar.kwargs)
                    for input, labels in bar:
                        outputs = self.model(input)
                        loss = criterion(outputs, labels)
                        eval_loss += loss.item()
                        eval_loss_count += 1
                        # TODO: add mAP, currently this is impractical due to
                        # postprocess_outputs() taking too much time
                        # outputs = [x.detach().cpu().numpy() for x in outputs]
                        # preds = self.postprocess_outputs(outputs)
                        # measurements = self.dataset.evaluate(preds, labels)
                        # thresholds = np.arange(0.2, 1.05, 0.05)
                        # maps = compute_map_per_threshold(
                        #     measurements, thresholds
                        # )

                        bar.set_description(
                            f"Evaluating model after epoch: {epoch:3d} loss: "
                            f"{eval_loss / eval_loss_count:.3f}"
                        )
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_model(self.save_model_path)
            if log_eval:
                log_eval.write(f"{eval_loss / eval_loss_count:.4f}\n")
                log_eval.flush()
        return best_loss

    def train_model(self):
        """
        Fine-tunes the YOLOv4 model.
        """
        import torch
        import torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader
        from torchvision.transforms import v2

        from kenning.modelwrappers.object_detection.pytorch_yolo_dataset import (  # noqa: E501
            YoloDataset,
        )

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.save_model_path:
            missing_params.append("save_model_path")

        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.head_num_epochs:
            missing_params.append("head_num_epochs")

        if not self.full_num_epochs:
            missing_params.append("full_num_epochs")

        if not self.logdir:
            missing_params.append("logdir")
        else:
            self.logdir.mkdir(exist_ok=True, parents=True)

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations(0.25)
        train_transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomPhotometricDistort(),
                v2.SanitizeBoundingBoxes(),
            ]
        )
        train_dataset = YoloDataset(
            Xt, Yt, self.dataset, self, transforms=train_transforms
        )

        def detection_collate(batch):
            images = torch.stack([item[0] for item in batch])
            targets = [item[1][0] for item in batch]
            return images, targets

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=detection_collate,
        )
        val_dataset = YoloDataset(Xv, Yv, self.dataset, self)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=detection_collate,
        )

        opt = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            momentum=0.9,
        )
        self.freeze_backbone()
        lr_scheduler = CosineAnnealingLR(opt, T_max=self.full_num_epochs)

        # Logging
        train_path = self.logdir / Path("train_loss")
        log_train = open(train_path, mode="w")
        eval_path = self.logdir / Path("eval_loss")
        log_eval = open(eval_path, mode="w")

        KLogger.info("Freezing backbone")
        best_loss = self._train_model(
            self.head_num_epochs,
            opt,
            self.loss_torch,
            train_loader,
            val_loader,
            log_train=log_train,
            log_eval=log_eval,
            lr_scheduler=None,
        )

        KLogger.info("Training the whole model")
        # Reduce learning rate by factor of 10
        for g in opt.param_groups:
            g["lr"] *= 0.1
        self.unfreeze()
        self.model.apply(_freeze_bn)
        self._train_model(
            self.full_num_epochs,
            opt,
            self.loss_torch,
            train_loader,
            val_loader,
            lr_scheduler=lr_scheduler,
            log_train=log_train,
            log_eval=log_eval,
            best_loss=best_loss,
        )

        if log_train:
            log_train.close()
        if log_eval:
            log_eval.close()

    def _get_io_specification(self, keyparams, batch_size):
        filters = (len(self.classnames) + 5) * 3
        return {
            "input": [
                {
                    "name": "input",
                    "shape": (
                        batch_size,
                        3,
                        keyparams["width"],
                        keyparams["height"],
                    ),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "output",
                    "shape": (
                        batch_size,
                        filters,
                        keyparams["width"] // (8 * 2**0),
                        keyparams["height"] // (8 * 2**0),
                    ),
                    "dtype": "float32",
                },
                {
                    "name": "output.3",
                    "shape": (
                        batch_size,
                        filters,
                        keyparams["width"] // (8 * 2**1),
                        keyparams["height"] // (8 * 2**1),
                    ),
                    "dtype": "float32",
                },
                {
                    "name": "output.7",
                    "shape": (
                        batch_size,
                        filters,
                        keyparams["width"] // (8 * 2**2),
                        keyparams["height"] // (8 * 2**2),
                    ),
                    "dtype": "float32",
                },
            ],
            "processed_output": [
                {
                    "name": "detection_output",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers.detection_and_segmentation.DetectObject",  # noqa: E501
                    },
                }
            ],
        }

    @classmethod
    def get_framework(cls) -> str:
        return "torch"

    @classmethod
    def get_framework_version(cls) -> str:
        return str(torch.__version__)

    @classmethod
    def get_output_formats(cls):
        return ["torch"]
