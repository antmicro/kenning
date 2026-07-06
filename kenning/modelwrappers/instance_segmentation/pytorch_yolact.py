# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for YOLACT model for instance segmentation.

Pretrained on COCO dataset. Supported input size: 550x550.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch
import numpy as np

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.datasets.helpers.detection_and_segmentation import SegmObject
from kenning.datasets.open_images_dataset import OpenImagesDatasetV6
from kenning.interfaces.io_interface import IOInterface
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


def _build_yolact_model(
    num_classes: int, resnet_layers: Tuple[int, ...]
) -> Any:
    """
    Build a YOLACT model with the given class count and backbone depth.

    The YOLACT model is not available in a trusted external library such as
    torchvision, therefore the architecture is implemented manually. This is
    required for correctly loading the pre-trained model and
    replacing the prediction head.

    Parameters
    ----------
    num_classes : int
        Number of classes for the model to detect.
    resnet_layers : Tuple[int, ...]
        Configuration of the ResNet backbone layers (e.g., (3, 4, 6, 3)).

    Returns
    -------
    Any
        Constructed YOLACT model instance.

    References
    ----------
    YOLACT Repo: https://github.com/dbolya/yolact
    """
    from kenning.modelwrappers.instance_segmentation.model_yolact.model import (  # noqa: E501
        _YolactModel,
    )

    return _YolactModel(num_classes, resnet_layers)


class PyTorchYOLACT(PyTorchWrapper, ABC):
    """
    Abstract class for the YOLACT model.

    The inheriting classes depend on specific datasets.
    """

    default_dataset = OpenImagesDatasetV6
    pretrained_model_uri = (
        "kenning:///models/instance_segmentation/yolact_resnet50.pth"
    )
    SCORE_THRESHOLD = 0.05
    MAX_DETECTIONS = 200
    NMS_IOU_THRESHOLD = 0.5
    MASK_THRESHOLD = 0.5
    NUM_CLASSES = 81
    arguments_structure = {
        "num_classes": {
            "argparse_name": "--num-classes",
            "description": "Number of classes that the model can classify.",
            "type": int,
            "default": NUM_CLASSES,
        },
        "score_threshold": {
            "argparse_name": "--score-threshold",
            "description": "Score threshold for filtering detections.",
            "type": float,
            "default": SCORE_THRESHOLD,
        },
        "max_detections": {
            "argparse_name": "--max-detections",
            "description": "Maximum number of detections to return.",
            "type": int,
            "default": MAX_DETECTIONS,
        },
        "nms_iou_threshold": {
            "argparse_name": "--nms-iou-threshold",
            "description": "IoU threshold for Non-Maximum Suppression.",
            "type": float,
            "default": NMS_IOU_THRESHOLD,
        },
        "mask_threshold": {
            "argparse_name": "--mask-threshold",
            "description": "Threshold for binarizing predicted masks.",
            "type": float,
            "default": MASK_THRESHOLD,
        },
        "freeze_backbone": {
            "argparse_name": "--freeze-backbone",
            "description": """Freeze backbone and FPN during training,
                              train only the head.""",
            "type": bool,
            "default": True,
            "subcommands": [TRAIN],
        },
        "pretrained_weights_path": {
            "argparse_name": "--pretrained-weights-path",
            "description": """Path to pretrained weights to load before
                              training. If not provided and from_file=False,
                              model is trained from scratch.""",
            "type": ResourceURI,
            "default": "kenning:///models/instance_segmentation/yolact_resnet50.pth",
            "subcommands": [TRAIN],
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": """Batch size for training. If not assigned,
                              dataset batch size will be used.""",
            "type": int,
            "default": 32,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "description": "Learning rate for training.",
            "type": float,
            "default": 0.0004,
            "subcommands": [TRAIN],
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs to train for.",
            "type": int,
            "default": 10,
            "subcommands": [TRAIN],
        },
        "logdir": {
            "argparse_name": "--logdir",
            "description": "Path to the logging directory.",
            "type": Path,
            "default": "./build/logs",
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool,
        model_name: Optional[str] = None,
        num_classes: int = NUM_CLASSES,
        score_threshold: float = SCORE_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
        nms_iou_threshold: float = NMS_IOU_THRESHOLD,
        mask_threshold: float = MASK_THRESHOLD,
        freeze_backbone: Optional[bool] = True,
        pretrained_weights_path: Optional[
            ResourceURI
        ] = "kenning:///models/instance_segmentation/yolact_resnet50.pth",
        batch_size: Optional[int] = 32,
        learning_rate: Optional[float] = 0.0004,
        num_epochs: Optional[int] = 10,
        logdir: Optional[Path] = "./build/logs",
        export_dict: bool = True,
    ):
        super().__init__(
            model_path,
            dataset,
            from_file,
            model_name=model_name,
            export_dict=export_dict,
        )
        if num_classes is not None:
            self.num_classes = num_classes
        elif dataset is not None:
            self.num_classes = len(dataset.classnames) + 1

        if dataset is not None:
            self.class_names = dataset.get_class_names()
        else:
            io_spec = self.load_io_specification(self.model_path)
            segmentation_output = IOInterface.find_spec(
                io_spec, "processed_output", "segmentation_output"
            )
            self.class_names = segmentation_output.get("class_names", [])

        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.nms_iou_threshold = nms_iou_threshold
        self.mask_threshold = mask_threshold
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.logdir = logdir
        self.freeze_backbone = freeze_backbone
        self.pretrained_weights_path = pretrained_weights_path

    @abstractmethod
    def preprocess_targets(
        self, y: List[List[SegmObject]]
    ) -> Tuple[List["torch.Tensor"], List["torch.Tensor"], List[int]]:
        """
        Convert raw dataset targets to YOLACT training format.

        Parameters
        ----------
        y : List[List[SegmObject]]
            A batch of target objects.

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]
            Preprocessed targets containing bounding boxes,
            masks, and crowd counts.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def create_model_structure(self):
        self.model = _build_yolact_model(
            num_classes=self.num_classes, resnet_layers=(3, 4, 6, 3)
        )

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess input images for the YOLACT model.

        Converts the input batch to a PyTorch tensor, dynamically resizes
        images to 550x550 using bilinear interpolation, and applies ImageNet
        standardization (mean and standard deviation). The input pixels are
        expected to be in the [0.0, 1.0] range. Image size must be 550x550.

        Parameters
        ----------
        X : List[np.ndarray]
            A list containing the input image batch. Expected to have the shape
            (B, C, H, W) with float values ranging from 0.0 to 1.0.

        Returns
        -------
        List[np.ndarray]
            A list containing a single NumPy array with the preprocessed,
            resized, and normalized image batch.
        """
        import torch
        import torch.nn.functional as F

        img = torch.from_numpy(np.array(X[0], dtype=np.float32))

        # Dynamically resize the image to 550x550 (standard YOLACT input size)
        # using bilinear interpolation if it does not already match.
        if img.shape[-2:] != (550, 550):
            img = F.interpolate(
                img, size=(550, 550), mode="bilinear", align_corners=False
            )

        # Define ImageNet statistics (Mean and Std) for normalization.
        # The .view(1, 3, 1, 1) ensures proper broadcasting
        # across (Batch, Channel, Height, Width).
        mean = torch.tensor([123.68, 116.78, 103.94]).view(1, 3, 1, 1)
        std = torch.tensor([58.40, 57.12, 57.38]).view(1, 3, 1, 1)

        img = (img * 255.0 - mean) / std

        return [img.numpy()]

    def postprocess_outputs(
        self, y: List["torch.Tensor"]
    ) -> List[List[List[SegmObject]]]:
        """
        Postprocess raw model outputs into structured segmentation objects.

        Iterates over the batch dimension of the raw predictions and processes
        each image individually. Tensors are kept on their original device
        (typically GPU) to maximize processing speed during Non-Maximum
        Suppression.

        Parameters
        ----------
        y : List[torch.Tensor]
            A list of 4 output tensors from the YOLACT model, representing
            a batch of predictions:
            - y[0]: Bounding box offsets, shape (B, N, 4).
            - y[1]: Class confidences, shape (B, N, num_classes).
            - y[2]: Mask coefficients, shape (B, N, mask_dim).
            - y[3]: Prototype masks, shape (B, proto_h, proto_w, mask_dim).

        Returns
        -------
        List[List[List[SegmObject]]]
            A list containing the model's unified output. This single output
            is a list of length B (batch size), where each element is
            a list of SegmObject instances representing the surviving
            detections and their corresponding masks for that specific image.
        """
        import torch

        boxes = torch.as_tensor(y[0])
        confs = torch.as_tensor(y[1])
        coefs = torch.as_tensor(y[2])
        proto = torch.as_tensor(y[3])

        # If data came without a batch dimension, add batch dimension (B=1)
        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(0)
            confs = confs.unsqueeze(0)
            coefs = coefs.unsqueeze(0)
            # proto.ndim (proto_h, proto_w, mask_dim)
            if proto.ndim == 3:
                proto = proto.unsqueeze(0)

        results = [
            self._process_image(boxes[b], confs[b], coefs[b], proto[b])
            for b in range(boxes.shape[0])
        ]

        return [results]

    def _process_image(
        self,
        b_boxes: "torch.Tensor",
        b_confs: "torch.Tensor",
        b_coefs: "torch.Tensor",
        b_proto: "torch.Tensor",
    ) -> List[SegmObject]:
        """
        Turn one image's raw model outputs into a list of SegmObject.

        Parameters
        ----------
        b_boxes : torch.Tensor
            Bounding box predictions.
        b_confs : torch.Tensor
            Confidence predictions.
        b_coefs : torch.Tensor
            Mask coefficient predictions.
        b_proto : torch.Tensor
            Prototype masks.

        Returns
        -------
        List[SegmObject]
            List of segmented objects for the image.
        """
        from kenning.modelwrappers.instance_segmentation.model_yolact.utils import (  # noqa: E501
            _assemble_masks,
            _select_detections,
        )

        # Filter raw predictions using score thresholding and NMS to get
        # the final bounding boxes, scores, class IDs, and mask coefficients
        dets, scores, classes, coef = _select_detections(
            b_boxes,
            b_confs,
            b_coefs,
            self.score_threshold,
            self.max_detections,
            self.nms_iou_threshold,
        )
        # If no detections survived the filtering process,
        # return an empty list of objects for this image
        if dets is None:
            return []

        # Clamp the bounding box coordinates to the [0.0, 1.0] range
        # to ensure they do not exceed the image boundaries
        dets_clamped = dets.clamp(0.0, 1.0)

        # Define target dimensions. Note: Dimensions are hardcoded to 550x550
        # because the model weights and the priors generation logic
        # (generate_priors) in utils strictly support
        # only this input resolution.
        target_h = 550
        target_w = 550
        # Combine prototype masks and instance coefficients,
        # then crop and resize them to the standard 550x550 YOLACT dimensions
        masks = _assemble_masks(
            b_proto, coef, dets_clamped, target_h, target_w
        )

        # Convert the processed tensors into a list of
        # standardized SegmObject instances
        return self._to_segm_objects(dets_clamped, classes, scores, masks)

    def _to_segm_objects(
        self,
        dets: "torch.Tensor",
        classes: "torch.Tensor",
        scores: "torch.Tensor",
        masks: "torch.Tensor",
    ) -> List[SegmObject]:
        """
        Build SegmObject entries from surviving detections and their masks.

        Parameters
        ----------
        dets : torch.Tensor
            Clamped bounding box detections.
        classes : torch.Tensor
            Predicted class indices.
        scores : torch.Tensor
            Detection scores.
        masks : torch.Tensor
            Assembled masks for the detections.

        Returns
        -------
        List[SegmObject]
            List of constructed segmentation objects.
        """
        objects = []
        # Iterate through all surviving detections that
        # passed the filtering and NMS
        for i in range(dets.shape[0]):
            xmin, ymin, xmax, ymax = dets[i].cpu().tolist()
            cls_id = classes[i].item()

            # Resolve the class name using class_names array if available,
            # otherwise fall back to string representation of the class ID
            cls_name = (
                self.class_names[cls_id]
                if hasattr(self, "class_names")
                and 0 <= cls_id < len(self.class_names)
                else str(cls_id)
            )

            # Construct and append the standardized SegmObject instance
            objects.append(
                SegmObject(
                    clsname=cls_name,
                    maskpath=None,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    mask=(masks[i].cpu().numpy() > self.mask_threshold).astype(
                        np.uint8
                    ),
                    score=scores[i].item(),
                    iscrowd=False,
                )
            )

        return objects

    def prepare_model(self) -> None:
        """
        Selects the appropriate initialization strategy based on from_file,
        pretrained_weights_path and freeze_backbone.

        - Inference (from_file=True): loads weights and sets eval mode.
        - Finetune head(pretrained path + freeze_backbone=True): loads weights,
          freezes backbone and FPN, reinitializes the heads.
        - Finetune full (pretrained path, backbone not frozen): loads weights
          and sets train mode on all parameters.
        - From scratch (no pretrained path): Xavier-initializes all modules.
        """
        if self.model_prepared:
            return
        self.create_model_structure()

        # Inference
        if self.from_file:
            self.load_model(self.model_path)
            self.model.eval()

        # Finetune head
        elif self.pretrained_weights_path and self.freeze_backbone:
            self.load_model(self.pretrained_weights_path)
            self.model._freeze_backbone()
            self.model._freeze_fpn()
            self.model._init_heads_only()
            self.model.train()

        # Finetune full
        elif self.pretrained_weights_path:
            self.load_model(self.pretrained_weights_path)
            self.model.train()

        # Train from scratch
        else:
            self.model.prepare_for_training(head_only=False)
            self.model.train()

        self.model.to(self.device)
        self.model_prepared = True

    def load_weights(self, weights: Dict[str, "torch.Tensor"]) -> None:
        """
        Load a state-dict into the model, skipping shape-mismatched keys.

        Parameters
        ----------
        weights : Dict[str, torch.Tensor]
            Dictionary containing model weights.
        """
        import copy

        model_sd = self.model.state_dict()

        filtered = {
            k: v
            for k, v in weights.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        self.model.load_state_dict(copy.deepcopy(filtered), strict=False)
        self.model.eval()

    def load_model(self, model_path: PathOrURI) -> None:
        """
        Load model weights from a file into the already-constructed model.

        Overrides the base load_model which calls create_model_structure
        internally, which would reset self.model to fresh weights and destroy
        already loaded ones.

        Parameters
        ----------
        model_path : PathOrURI
            Path to the saved model file.
        """
        import torch

        model_path = ResourceURI(model_path)

        input_data = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        if isinstance(input_data, torch.nn.Module):
            input_data = input_data.state_dict()

        self.load_weights(input_data)

    def save_model(
        self, model_path: PathOrURI, export_dict: Optional[bool] = None
    ) -> None:
        """
        Serialize the full model or only dict to disk using dill.

        Parameters
        ----------
        model_path : PathOrURI
            Path where the model will be saved.
        export_dict : Optional[bool], optional
            If True, only state_dict is saved.
            Otherwise, the whole model is pickled.
        """
        import torch

        if export_dict is None:
            export_dict = self.export_dict
        if export_dict:
            torch.save(self.model.state_dict(), str(model_path))
        else:
            import dill

            torch.save(self.model, str(model_path), pickle_module=dill)

    def _train_step(
        self,
        images: "torch.Tensor",
        targets: List["torch.Tensor"],
        masks: List["torch.Tensor"],
        num_crowds: List[int],
        optimizer: "torch.optim.Optimizer",
        params: List["torch.nn.Parameter"],
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """
        Run one optimization step and return the loss for a non-empty batch.

        Performs forward pass, loss computation, backward pass, gradient
        clipping and (if the loss is finite) an optimizer step. The LR
        scheduler is advanced by the caller.

        Parameters
        ----------
        images : torch.Tensor
            Batch of input images.
        targets : List[torch.Tensor]
            Ground truth bounding box targets.
        masks : List[torch.Tensor]
            Ground truth instance masks.
        num_crowds : List[int]
            Number of crowd objects in each image.
        optimizer : torch.optim.Optimizer
            Optimizer used for training.
        params : List[torch.nn.Parameter]
            List of model parameters requiring gradients.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Total scalar loss and dictionary
            containing specific loss components.
        """
        import torch

        optimizer.zero_grad()
        preds = self.model(images)
        losses = self.model.compute_loss(preds, targets, masks, num_crowds)
        loss = sum(losses.values())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        if torch.isfinite(loss).item():
            optimizer.step()
        else:
            KLogger.warning(
                f"Infinite or NaN loss detected: {loss.item()}. "
                "Skipping optimizer step."
            )

        return loss, losses

    def train_model(self) -> Any:
        """
        Train the model for the configured number of epochs.

        Sets up AdamW optimizer with cosine-warmup LR scheduling, iterates
        over the training split, logs per-iteration and per-epoch losses to
        logdir/train_log.txt (if configured), and saves the final model
        to model_path.

        Returns
        -------
        Any
            Trained PyTorch model instance.
        """
        import torch
        import torch.optim as optim
        from tqdm import tqdm

        from kenning.modelwrappers.instance_segmentation.model_yolact.utils import (  # noqa: E501
            _make_cosine_warmup_lambda,
        )

        if self.batch_size is not None:
            self.dataset.set_batch_size(self.batch_size)
        self.prepare_model()
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        KLogger.info(
            "Number of trainable parameters: "
            f"{sum(p.numel() for p in params):,}"
        )

        lr = self.learning_rate if self.learning_rate is not None else 1e-4
        num_epochs = self.num_epochs if self.num_epochs is not None else 1

        try:
            iters_per_epoch = len(self.dataset.iter_train())
        except (TypeError, AttributeError):
            iters_per_epoch = 100
        total_iters = num_epochs * iters_per_epoch
        warmup_iters = max(1, int(total_iters * 0.05))

        optimizer = optim.AdamW(params, lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, _make_cosine_warmup_lambda(warmup_iters, total_iters)
        )

        if self.logdir:
            self.logdir.mkdir(parents=True, exist_ok=True)
            log_file = open(self.logdir / "train_log.txt", "w")
        else:
            log_file = None

        self.model.train()
        iteration = 0

        try:
            epoch_pbar = tqdm(range(num_epochs), desc="Epochs", unit="epoch")
            for epoch in epoch_pbar:
                epoch_losses = defaultdict(list)

                batch_pbar = tqdm(
                    self.dataset.iter_train(),
                    total=iters_per_epoch,
                    desc=f"Epoch {epoch}",
                    leave=False,
                    unit="batch",
                )
                for X, y in batch_pbar:
                    if self.should_cancel:
                        break

                    images = torch.as_tensor(self.preprocess_input(X)[0]).to(
                        self.device
                    )
                    targets, masks, num_crowds = self.preprocess_targets(y)

                    keep = [i for i, t in enumerate(targets) if t.shape[0] > 0]
                    if len(keep) == 0:
                        iteration += 1
                        scheduler.step()
                        continue
                    if len(keep) < len(targets):
                        images = images[keep]
                        targets = [targets[i] for i in keep]
                        masks = [masks[i] for i in keep]
                        num_crowds = [num_crowds[i] for i in keep]

                    loss, losses = self._train_step(
                        images, targets, masks, num_crowds, optimizer, params
                    )
                    scheduler.step()

                    current_lr = optimizer.param_groups[0]["lr"]

                    for k, v in losses.items():
                        epoch_losses[k].append(v.item())
                    epoch_losses["total"].append(loss.item())

                    parts = " ".join(
                        f"{k}={v.item():.3f}" for k, v in losses.items()
                    )
                    batch_pbar.set_postfix_str(
                        f"total={loss.item():.3f} | {parts} | "
                        f"lr={current_lr:.2e}"
                    )
                    if log_file and iteration % 10 == 0:
                        log_file.write(
                            f"epoch={epoch} iter={iteration} "
                            f"lr={current_lr:.2e} "
                            f"total={loss.item():.3f} | {parts}\n"
                        )
                        log_file.flush()

                    iteration += 1

                if self.should_cancel:
                    break

                means = {
                    k: sum(v) / len(v) for k, v in epoch_losses.items() if v
                }
                summary = " ".join(f"{k}={v:.3f}" for k, v in means.items())
                epoch_pbar.set_postfix_str(summary)
                if log_file:
                    log_file.write(f"Epoch {epoch} | {summary} ==\n\n")
                    log_file.flush()
        finally:
            if log_file:
                log_file.close()

        self.model.eval()
        self.save_model(self.model_path)
        return self.model

    @classmethod
    def _get_io_specification(
        cls, num_classes: int, batch_size: int = 1
    ) -> Dict[str, List[Dict]]:
        return {
            "input": [
                {
                    "name": "input",
                    "shape": (batch_size, 3, -1, -1),
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input",
                    "shape": (batch_size, 3, 550, 550),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "box",
                    "shape": (batch_size, 19248, 4),
                    "dtype": "float32",
                },
                {
                    "name": "class",
                    "shape": (batch_size, 19248, num_classes),
                    "dtype": "float32",
                },
                {
                    "name": "coef",
                    "shape": (batch_size, 19248, 32),
                    "dtype": "float32",
                },
                {
                    "name": "proto",
                    "shape": (batch_size, 138, 138, 32),
                    "dtype": "float32",
                },
                {
                    "name": "seg",
                    "shape": (batch_size, num_classes - 1, 69, 69),
                    "dtype": "float32",
                },
                {"name": "priors", "shape": (19248, 4), "dtype": "float32"},
            ],
            "processed_output": [
                {
                    "name": "segmentation_output",
                    "type": "List",
                    "dtype": {
                        "type": "List",
                        "dtype": "kenning.datasets.helpers."
                        "detection_and_segmentation.SegmObject",
                    },
                }
            ],
        }

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        batch_size = self.dataset.batch_size if self.dataset else 1
        io_spec = self._get_io_specification(self.num_classes, batch_size)
        io_spec["processed_output"][0]["class_names"] = self.class_names

        return io_spec

    @classmethod
    def derive_io_spec_from_json_params(
        cls, json_dict: Dict
    ) -> Dict[str, List[Dict]]:
        return cls._get_io_specification(json_dict["num_classes"])


class YOLACTOpenImages(PyTorchYOLACT):
    """
    YOLACT model wrapper for the Open Images V6 dataset.
    """

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        return super().preprocess_input(X)

    def preprocess_targets(
        self, y: List[List[SegmObject]]
    ) -> Tuple[List["torch.Tensor"], List["torch.Tensor"], List[int]]:
        """
        Convert Open Images annotations to YOLACT training targets.

        Separates crowd from non-crowd objects, encodes bounding boxes and
        class indices into tensors, and stacks GT masks.

        Parameters
        ----------
        y : List[List[SegmObject]]
            List of objects containing raw annotation data for a batch.

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]
            Tuple consisting of bounding box targets, segmentation masks,
            and the number of crowd targets per image.
        """
        import torch

        batch = y[0]
        device = self.device

        targets, masks, num_crowds = [], [], []
        # Iterate through each sample/image in the batch
        for sample in batch:
            normal, crowd = [], []

            # Process each annotated object in the sample
            for obj in sample:
                # Map class name to its integer ID using class_names,
                # or default to 0 if not found
                if (
                    hasattr(self, "class_names")
                    and obj.clsname in self.class_names
                ):
                    cls_id = self.class_names.index(obj.clsname)
                else:
                    cls_id = 0

                # Package bounding box coordinates and class
                # ID alongside its segmentation mask
                entry = (
                    [obj.xmin, obj.ymin, obj.xmax, obj.ymax, float(cls_id)],
                    obj.mask,
                )
                # Separate standard objects from crowd instances
                # (crowds are typically appended at the end)
                (crowd if getattr(obj, "iscrowd", False) else normal).append(
                    entry
                )

            ordered = normal + crowd
            # Handle edge case: if an image contains no objects,
            # append empty tensors and zero crowd count
            if len(ordered) == 0:
                targets.append(torch.zeros((0, 5), device=device))
                masks.append(torch.zeros((0, 1, 1), device=device))
                num_crowds.append(0)
                continue

            # Convert bounding boxes and class IDs into a
            # single PyTorch tensor on the correct device
            boxes = torch.tensor(
                [e[0] for e in ordered], dtype=torch.float32, device=device
            )

            # Process masks for the current image
            mask_stack = []
            for _, m in ordered:
                mt = torch.as_tensor(np.asarray(m), dtype=torch.float32)
                if mt.numel() and mt.max() > 1:
                    mt = mt / 255.0
                mask_stack.append(mt)

            # Stack all masks into a single tensor,
            # binarize them using a 0.5 threshold, and move to device
            masks_t = torch.stack(mask_stack, 0).gt(0.5).float().to(device)

            # Store processed targets, masks, and the count of
            # crowd elements for this sample
            targets.append(boxes)
            masks.append(masks_t)
            num_crowds.append(len(crowd))

        return targets, masks, num_crowds
