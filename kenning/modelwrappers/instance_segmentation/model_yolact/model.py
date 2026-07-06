# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
The YOLACT model implemented in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kenning.modelwrappers.instance_segmentation.model_yolact.loss_function import (  # noqa: E501
    _YolactLoss,
)
from kenning.modelwrappers.instance_segmentation.model_yolact.utils import (
    _generate_priors,
)


class _Bottleneck(nn.Module):
    """
    ResNet bottleneck residual block (1x1 > 3x3 > 1x1, expansion 4).
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        """
        Parameters
        ----------
        inplanes : int
            Number of input channels.
        planes : int
            Base channel width; the final output has planes * expansion
            channels.
        stride : int, optional
            Stride applied to the 3x3 convolution. Defaults to 1.
        downsample : nn.Module | None, optional
            Optional projection shortcut (conv + BN) to match spatial and
            channel dimensions when stride > 1 or
            inplanes != planes * expansion. Defaults to None.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, inplanes, H, W).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (B, planes * expansion, H', W'),
            where H' and W' are reduced by stride.
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class _ResNetBackbone(nn.Module):
    """
    ResNet feature extractor producing the C2-C5 pyramid stages.
    """

    def __init__(self, layers: list[int]) -> None:
        """
        Parameters
        ----------
        layers : list[int]
            List of four integers specifying the number of bottleneck blocks
            in stages C2-C5 (e.g. [3, 4, 6, 3] for ResNet-50).
        """
        super().__init__()
        self.inplanes = 64
        self.layers = nn.ModuleList()
        self.channels = []

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._make_layer(64, layers[0], stride=1)
        self._make_layer(128, layers[1], stride=2)
        self._make_layer(256, layers[2], stride=2)
        self._make_layer(512, layers[3], stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> None:
        """
        Create one residual stage and append it to the backbone.

        Also records the stage's output channel count in self.channels.

        Parameters
        ----------
        planes : int
            Base channel width for the stage; the output has
            planes * _Bottleneck.expansion channels.
        blocks : int
            Number of bottleneck blocks in the stage.
        stride : int, optional
            Stride for the first block's 3x3 convolution and its shortcut
            projection. Defaults to 1.

        Returns
        -------
        None
            The stage is appended to self.layers and its output channel count
            to self.channels in place.
        """
        downsample = nn.Sequential(
            nn.Conv2d(
                self.inplanes,
                planes * _Bottleneck.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes * _Bottleneck.expansion),
        )
        layers = [_Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes))
        self.channels.append(self.inplanes)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 3, H, W).

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of four feature maps (C2, C3, C4, C5), one per residual
            stage, with progressively halved spatial dimensions and doubled
            channel counts.
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        outs = []

        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return tuple(outs)


class _FPN(nn.Module):
    """
    Feature Pyramid Network building 5 same-channel prediction levels.
    """

    def __init__(self, in_channels: list[int]) -> None:
        """
        Parameters
        ----------
        in_channels : list[int]
            List of channel counts for the backbone feature maps fed into the
            FPN (e.g. [512, 1024, 2048] for C3-C5 of ResNet-50).
        """
        super().__init__()
        self.lat_layers = nn.ModuleList(
            [nn.Conv2d(x, 256, kernel_size=1) for x in reversed(in_channels)]
        )
        self.pred_layers = nn.ModuleList(
            [
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
                for _ in in_channels
            ]
        )
        self.downsample_layers = nn.ModuleList(
            [
                nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            ]
        )

    def forward(self, backbone_outs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        backbone_outs : list[torch.Tensor]
            List of backbone feature maps (e.g. C3-C5), each of shape
            (B, C_i, H_i, W_i), ordered from largest to smallest spatial
            resolution.

        Returns
        -------
        list[torch.Tensor]
            List of five 256-channel feature maps: the three FPN levels
            corresponding to the input stages plus two additional downsampled
            levels appended at the end.
        """
        out = [torch.zeros(1, device=backbone_outs[0].device)] * len(
            backbone_outs
        )
        x = torch.zeros(1, device=backbone_outs[0].device)
        j = len(backbone_outs)

        for lat_layer in self.lat_layers:
            j -= 1
            if j < len(backbone_outs) - 1:
                _, _, h, w = backbone_outs[j].size()
                x = F.interpolate(
                    x, size=(h, w), mode="bilinear", align_corners=False
                )
            x = x + lat_layer(backbone_outs[j])
            out[j] = x

        j = len(backbone_outs)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        for ds in self.downsample_layers:
            out.append(ds(out[-1]))

        return out


class _PredictionModule(nn.Module):
    """
    Shared head predicting box offsets, class scores and mask coefficients.
    """

    def __init__(
        self,
        num_classes: int = 81,
        coef_dim: int = 32,
        num_priors: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int, optional
            Total number of classes including background. Defaults to 81
            (COCO).
        coef_dim : int, optional
            Number of mask prototype coefficients per detection.
            Defaults to 32.
        num_priors : int, optional
            Number of anchor priors per spatial location. Defaults to 3.
        """
        super().__init__()
        self.num_classes = num_classes
        self.coef_dim = coef_dim
        self.num_priors = num_priors
        self.upfeature = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bbox_layer = nn.Conv2d(
            256, num_priors * 4, kernel_size=3, padding=1
        )
        self.conf_layer = nn.Conv2d(
            256, num_priors * num_classes, kernel_size=3, padding=1
        )
        self.mask_layer = nn.Conv2d(
            256, num_priors * coef_dim, kernel_size=3, padding=1
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            FPN feature map of shape (B, 256, H, W) for a single prediction
            level.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Three tensors:

            - bbox: box offsets of shape (B, H*W*num_priors, 4).
            - conf: class logits of shape (B, H*W*num_priors, num_classes).
            - coef: mask coefficients in [-1, 1] of shape
              (B, H*W*num_priors, coef_dim), passed through tanh.
        """
        x = self.upfeature(x)
        b = x.size(0)
        bbox = (
            self.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
        )
        conf = (
            self.conf_layer(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(b, -1, self.num_classes)
        )
        coef = torch.tanh(
            self.mask_layer(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(b, -1, self.coef_dim)
        )

        return bbox, conf, coef


class _YolactModel(nn.Module):
    """
    Full YOLACT model: backbone + FPN + protonet + shared prediction head.
    """

    def __init__(
        self,
        num_classes: int,
        resnet_layers: list[int],
        coef_dim: int = 32,
        max_size: int = 550,
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Total number of classes including background.
        resnet_layers : list[int]
            List of four integers specifying bottleneck block counts per stage,
            e.g. [3, 4, 6, 3] for ResNet-50.
        coef_dim : int, optional
            Number of mask prototype coefficients. Defaults to 32.
        max_size : int, optional
            Expected square input resolution in pixels, used to generate anchor
            priors. Defaults to 550.
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone = _ResNetBackbone(resnet_layers)
        self.fpn = _FPN([512, 1024, 2048])
        # Prototype network producing the mask prototype bank.
        self.proto_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, coef_dim, kernel_size=1),
        )
        self.prediction_layers = nn.ModuleList(
            [_PredictionModule(num_classes, coef_dim)]
        )
        self.semantic_seg_conv = nn.Conv2d(256, num_classes - 1, kernel_size=1)

        self.criterion = _YolactLoss(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 3, H, W), where H and W are
            typically 550.

        Returns
        -------
        list[torch.Tensor]
            List of six tensors:

            0. boxes: raw box offsets, shape (B, P, 4).
            1. confs: class logits, shape (B, P, num_classes).
            2. coefs: mask coefficients, shape (B, P, coef_dim).
            3. proto: prototype feature maps, shape (B, mask_h, mask_w,
               coef_dim).
            4. seg: semantic segmentation logits, shape (B, num_classes-1,
               mask_h, mask_w).
            5. priors: anchor priors in center form, shape (P, 4).
        """
        backbone_outs = self.backbone(x)
        fpn_outs = self.fpn(
            [
                backbone_outs[1],
                backbone_outs[2],
                backbone_outs[3],
            ]
        )
        fpn_outs = fpn_outs[:5]

        proto = F.relu(self.proto_net(fpn_outs[0]))
        proto = proto.permute(0, 2, 3, 1).contiguous()

        boxes, confs, coefs = [], [], []
        for feat in fpn_outs:
            b, c, co = self.prediction_layers[0](feat)
            boxes.append(b)
            confs.append(c)
            coefs.append(co)

        priors = _generate_priors().to(x.device)

        return [
            torch.cat(boxes, dim=1),
            torch.cat(confs, dim=1),
            torch.cat(coefs, dim=1),
            proto,
            self.semantic_seg_conv(fpn_outs[0]),
            priors,
        ]

    def compute_loss(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        masks: list[torch.Tensor],
        num_crowds: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Compute the multi-task loss from raw forward outputs.

        Repackages the ordered forward outputs into the dict expected by
        the loss criterion and evaluates it.

        Parameters
        ----------
        predictions : list[torch.Tensor]
            List returned by forward — six tensors in the order
            [boxes, confs, coefs, proto, seg, priors].
        targets : list[torch.Tensor]
            List of length B. Each element is a float tensor of shape (M_i, 5)
            with [xmin, ymin, xmax, ymax, class] rows for the M_i GT objects
            in image i.
        masks : list[torch.Tensor]
            List of length B. Each element is a float tensor of shape
            (M_i, H, W) with binary ground-truth instance masks.
        num_crowds : list[int]
            List of length B indicating how many of the last entries in
            targets[i] / masks[i] are crowd annotations.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with scalar loss tensors {B, C, M, S} as produced by
            _YolactLoss.
        """
        pred_dict = {
            "box": predictions[0],
            "class": predictions[1],
            "coef": predictions[2],
            "proto": predictions[3],
            "seg": predictions[4],
            "priors": predictions[5],
        }

        return self.criterion(pred_dict, targets, masks, num_crowds)

    def prepare_for_training(self, head_only: bool) -> None:
        """
        Xavier/BN-initialize the model for training from scratch or head-only.

        Parameters
        ----------
        head_only : bool
            If True, only the protonet, prediction layers and
            semantic-segmentation head are initialized and the backbone and
            FPN are frozen. If False, all modules are initialized.

        Returns
        -------
        None
            Module weights are updated in place and the backbone / FPN are
            frozen when head_only is True.
        """
        if head_only:
            modules_to_init = [
                self.proto_net,
                self.prediction_layers,
                self.semantic_seg_conv,
            ]
        else:
            modules_to_init = [
                self.backbone,
                self.fpn,
                self.proto_net,
                self.prediction_layers,
                self.semantic_seg_conv,
            ]

        for module_group in modules_to_init:
            for m in module_group.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        if head_only:
            self._freeze_backbone()
            self._freeze_fpn()

    def _freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters and put its BatchNorm layers in eval.

        Sets requires_grad=False on every backbone parameter and switches all
        backbone BatchNorm2d modules to eval mode so their running statistics
        are not updated during training.

        Returns
        -------
        None
            Backbone state is modified in place.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True) -> "_YolactModel":
        """
        Set training mode while keeping frozen sub-nets BatchNorm in eval.

        Overrides nn.Module.train so that if the backbone or FPN has been
        frozen, their BatchNorm layers stay in eval mode (frozen running
        stats) even when the model is switched to train.

        Parameters
        ----------
        mode : bool, optional
            If True, set the model to training mode; if False, set it to
            evaluation mode (equivalent to model.eval()). Defaults to True.

        Returns
        -------
        _YolactModel
            self, following the nn.Module.train convention.
        """
        super().train(mode)
        if mode:
            for param in self.backbone.parameters():
                if not param.requires_grad:
                    for module in self.backbone.modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            module.eval()
                    break
            for param in self.fpn.parameters():
                if not param.requires_grad:
                    for module in self.fpn.modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            module.eval()
                    break
        return self

    def _freeze_fpn(self) -> None:
        """
        Freeze all FPN parameters and put its BatchNorm layers in eval.

        Sets requires_grad=False on every FPN parameter and switches all FPN
        BatchNorm2d modules to eval mode so their running statistics are not
        updated during training.

        Returns
        -------
        None
            FPN state is modified in place.
        """
        for param in self.fpn.parameters():
            param.requires_grad = False
        for module in self.fpn.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def _init_heads_only(self) -> None:
        """
        Xavier/BN-initialize only the head modules,
        leaving backbone/FPN intact.

        Used when fine-tuning a pretrained backbone with a fresh prediction
        head. Applies Xavier uniform initialization to all Conv2d weights and
        sets BatchNorm2d weight/bias to 1/0.

        Returns
        -------
        None
            Head module weights are updated in place.
        """
        for module_group in [
            self.proto_net,
            self.prediction_layers,
            self.semantic_seg_conv,
        ]:
            for m in module_group.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
