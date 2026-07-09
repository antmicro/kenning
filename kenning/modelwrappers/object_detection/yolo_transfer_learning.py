# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
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

If you want to use pretrained model weights you have to download
yolov4.pth from repository above.
"""

import copy
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kenning.cli.command_template import TEST, TRAIN
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    TrainingParametersMissingError,
)
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.open_images_dataset import OpenImagesDatasetV6
from kenning.modelwrappers.frameworks.helpers.pytorch_yolov4_impl import Yolov4
from kenning.modelwrappers.object_detection.yolov4 import ONNXYOLOV4
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI, ResourceURI

Optimizer = TypeVar("torch.optim.Optimizer")


class YOLOV4TL(ONNXYOLOV4):
    """
    Model wrapper for YOLOv4 that supports transfer learning.
    """

    pretrained_model_uri = "kenning:///models/detection/yolov4.onnx"
    default_dataset = COCODataset2017
    arguments_structure = {
        "save_model_path": {
            "argparse_name": "--save-model-path",
            "description": "During training, path where the fine-tuned model will be saved.",  # noqa: E501
            "type": ResourceURI,
            "default": None,
            "subcommands": [TRAIN],
        },
        "pretrained": {
            "argparse_name": "--pretrained",
            "description": "dupa",
            "type": bool,
            "default": False,
            "subcommands": [TRAIN, TEST],
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used.",  # noqa: E501
            "type": int,
            "default": None,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "description": "Learning rate for training",
            "type": float,
            "default": None,
            "subcommands": [TRAIN],
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs to train for",
            "type": int,
            "default": None,
            "subcommands": [TRAIN],
        },
        "logdir": {
            "argparse_name": "--logdir",
            "description": "Path to the logging directory",
            "type": Path,
            "default": None,
            "subcommands": [TRAIN],
        },
    }

    layer_names_to_change = [
        "head.conv2",
        "head.conv10",
        "head.conv18",
    ]

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file=True,
        model_name: Optional[str] = None,
        class_names: str = "coco",
        save_model_path: Optional[PathOrURI] = None,
        pretrained: Optional[bool] = None,
        batch_size: int = 1,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        logdir: Optional[Path] = None,
    ):
        self._device = None
        if save_model_path:
            self.save_model_path = save_model_path
        else:
            self.save_model_path = model_path
        self.pretrained = pretrained

        super().__init__(
            model_path,
            dataset,
            from_file,
            model_name,
            class_names,
        )

        if isinstance(self.dataset, OpenImagesDatasetV6):
            self.classnames = list(
                map(lambda n: n.split(",")[1], self.classnames)
            )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.logdir = logdir

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._device

    def freeze_backbone(self):
        """
        Freezes backbone of the model.
        """
        backbone_layers = [f"down{i}" for i in range(1, 6)]
        for layer_name in backbone_layers:
            layer = getattr(self.model, layer_name)
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes models weights.
        """
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def create_model_structure(self, model_path: Optional[ResourceURI] = None):
        self.model = Yolov4(n_classes=self.numclasses)

    def load_pretrained_torch_model(self, model_path: PathOrURI):
        weights = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        dst_sd = self.model.state_dict()
        new_sd = dst_sd.copy()

        used = set()

        for dst_name, dst_tensor in dst_sd.items():
            for src_name, src_tensor in weights.items():
                if src_name in used or any(
                    src_name.startswith(layer)
                    for layer in self.layer_names_to_change
                ):
                    continue

                if src_tensor.shape == dst_tensor.shape:
                    new_sd[dst_name] = src_tensor
                    used.add(src_name)
                    break

        self.model.load_state_dict(new_sd, strict=False)

    def load_torch_model(self, model_path: PathOrURI):
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
        super().load_model(ResourceURI(self.pretrained_model_uri))
        self.create_model_structure()

        if self.pretrained:
            self.load_pretrained_torch_model(self.model_path)
        else:
            try:
                self.load_torch_model(self.model_path)
            except FileNotFoundError:
                # todo: add downloading pretrained pytorch weights
                self.load_pretrained_torch_model(self.model_path)
        self.model.to(self.device)
        self.model_prepared = True

    def preprocess_input(self, X: List[Any]) -> List[np.array]:
        def _preprocess(img):
            """
            Conversion between BGR and RGB.
            """
            img = (img.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            img = img[..., ::-1]
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (0, 3, 1, 2))
            return img

        return [_preprocess(x) for x in X]

    def run_inference(self, X: List) -> Any:
        res = self.model(*X)
        return [r.detach().cpu().numpy() for r in res]

    def save_model(
        self, model_path: PathOrURI, export_dict: Optional[bool] = None
    ):
        model = self.model
        KLogger.info(f"saving model to {model_path}")

        torch.save(model.state_dict(), model_path)
        json_cfg = Path(f"{self.save_model_path}.json")
        with json_cfg.open(mode="w") as f:
            import json

            json.dump(
                self._get_io_specification(self.keyparams, self.batch_size), f
            )

    def visualize_image(self, visualizer, input, outputs):
        outputs_np = [o.detach().numpy() for o in outputs]
        postPreds = self.postprocess_outputs(outputs_np)

        objects = postPreds[0][0]

        img = input[0].detach().numpy()
        KLogger.info(f"{img.shape}")
        img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = visualizer.visualize_data(img, objects)

        cv2.imshow("OpenImages", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        KLogger.info(f"{img.shape}")

    def _train_model(
        self,
        num_epochs: int,
        opt: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr_scheduler: Optional[torch.optim.lr_scheduler],
    ):
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
        lr_scheduler: Optional[torch.optim.lr_scheduler]
            Scheduler for learning rate.
        """
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            loss_sum = torch.zeros(1).to(self.device)
            loss_count = 0

            with LoggerProgressBar() as logger_progress_bar:
                bar = tqdm(train_loader, **logger_progress_bar.kwargs)
                for input, labels in bar:
                    opt.zero_grad()

                    outputs = self.model(input)
                    loss = criterion(outputs, labels[0])
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 10.0
                    )

                    opt.step()
                    if lr_scheduler:
                        lr_scheduler.step()

                    loss_sum += loss
                    loss_count += 1
                    bar.set_description(
                        f"train epoch: {epoch:3d} loss: "
                        f"{loss_sum.data.cpu().numpy().sum() / loss_count:.3f}"
                    )

            # Evaluate model on validation set
            with torch.no_grad():
                eval_loss = 0
                eval_loss_count = 0
                self.model.eval()
                with LoggerProgressBar() as logger_progress_bar:
                    bar = tqdm(val_loader, **logger_progress_bar.kwargs)
                    for input, labels in bar:
                        outputs = self.model(input)
                        eval_loss += float(criterion(outputs, labels[0]))
                        eval_loss_count += 1
                        bar.set_description(
                            f"Evaluating model after epoch: {epoch:3d} loss: "
                            f"{eval_loss / eval_loss_count:.3f}"
                        )
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_model(self.save_model_path)

        self.model.eval()

    def train_model(self):
        """
        Fine-tunes the YOLOv4 model.
        """
        import torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import Dataset

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.save_model_path:
            missing_params.append("save_model_path")

        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.num_epochs:
            missing_params.append("num_epochs")

        if not self.logdir:
            missing_params.append("logdir")
        else:
            self.logdir.mkdir(exist_ok=True, parents=True)

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

        class _YoloDataset(Dataset):
            def __init__(
                self,
                inputs,
                labels,
                dataset,
                wrapper,
            ):
                self.inputs = inputs
                self.labels = labels
                self.dataset = dataset
                self.wrapper = wrapper
                self.device = wrapper.device

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                batch_x = [self.inputs[idx]]
                data = self.dataset.prepare_input_samples(batch_x)
                data = self.wrapper._preprocess_input(data)
                if isinstance(data[0], torch.Tensor):
                    data = torch.stack(data)
                batch_y = [self.labels[idx]]
                label = self.dataset.prepare_output_samples(batch_y)

                data = torch.Tensor(data).to(self.device)
                try:
                    label = [
                        torch.from_numpy(np.asarray(_l)).to(self.device)
                        for _l in label
                    ]
                except (ValueError, TypeError):
                    pass

                data = data[0].squeeze(0)
                return data, label

        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations(0.25)
        train_dataset = _YoloDataset(Xt, Yt, self.dataset, self)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )
        val_dataset = _YoloDataset(Xv, Yv, self.dataset, self)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )

        detector_epochs = 10

        opt = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            momentum=0.9,
        )
        self.freeze_backbone()
        lr_scheduler = CosineAnnealingLR(opt, T_max=self.num_epochs)

        KLogger.info("Train only neck and head")
        self._train_model(
            detector_epochs,
            opt,
            self.loss_torch,
            train_loader,
            val_loader,
            # lr_scheduler=lr_scheduler,
        )

        KLogger.info("Training the whole model")
        # Reduce learning rate by factor of 10
        for g in opt.param_groups:
            g["lr"] *= 0.1
        self.unfreeze()
        self._train_model(
            self.num_epochs,
            opt,
            self.loss_torch,
            train_loader,
            val_loader,
            lr_scheduler=lr_scheduler,
        )

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
