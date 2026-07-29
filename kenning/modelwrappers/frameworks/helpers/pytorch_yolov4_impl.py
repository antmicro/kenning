# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv4 implementation in pytorch based on https://github.com/Tianxiaomo/pytorch-YOLOv4.

You can explore this structure by opening yolov4.onnx (you can get it
from kenning-resources) in netron.
"""

from enum import Enum
from functools import partial

import torch
from torch import nn


class _Activation(Enum):
    """
    Available activation functions with pre-defined arguments.
    """

    Mish = partial(nn.Mish)
    Relu = partial(nn.ReLU, inplace=True)
    Leaky = partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)

    def __call__(self):
        return self.value()


class _ConvByActivation(nn.Module):
    """
    Simple wrapper for convolutional layer with optional batch norm layer and
    activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: _Activation | None,
        bn: bool = True,
        bias: bool = False,
    ):
        """
        Creates `_ConvByActivation`.

        Parameters
        ----------
        in_channels : int
        out_channels : int
        kernel_size : int
        stride : int
        activation : _Activation | None
        bn : bool
            Include batch norm layer.
        bias : bool
            Include bias as a learnable parameter.
        """
        super().__init__()
        pad = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                pad,
                bias=bias,
            )
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation())
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of two convolution
    layers.
    """

    def __init__(self, ch: int, nblocks: int = 1, shortcut: bool = True):
        """
        Creates a residual block used in Yolov4.

        Parameters
        ----------
        ch : int
            number of input and output channels.
        nblocks : int
            number of residual blocks.
        shortcut : bool
            if True, residual tensor addition is enabled.
        """
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock = [
                _ConvByActivation(ch, ch, 1, 1, _Activation.Mish),
                _ConvByActivation(ch, ch, 3, 1, _Activation.Mish),
            ]
            self.module_list.append(nn.Sequential(*resblock))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.module_list:
            h = module(x)
            x = x + h if self.shortcut else h
        return x


class _DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvByActivation(3, 32, 3, 1, _Activation.Mish)

        self.conv2 = _ConvByActivation(32, 64, 3, 2, _Activation.Mish)
        self.conv3 = _ConvByActivation(64, 64, 1, 1, _Activation.Mish)
        self.conv4 = _ConvByActivation(64, 64, 1, 1, _Activation.Mish)

        self.conv5 = _ConvByActivation(64, 32, 1, 1, _Activation.Mish)
        self.conv6 = _ConvByActivation(32, 64, 3, 1, _Activation.Mish)

        self.conv7 = _ConvByActivation(64, 64, 1, 1, _Activation.Mish)
        self.conv8 = _ConvByActivation(128, 64, 1, 1, _Activation.Mish)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4

        x7 = self.conv7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class _DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvByActivation(64, 128, 3, 2, _Activation.Mish)
        self.conv2 = _ConvByActivation(128, 64, 1, 1, _Activation.Mish)
        self.conv3 = _ConvByActivation(128, 64, 1, 1, _Activation.Mish)

        self.resblock = _ResBlock(ch=64, nblocks=2)

        self.conv4 = _ConvByActivation(64, 64, 1, 1, _Activation.Mish)
        self.conv5 = _ConvByActivation(128, 128, 1, 1, _Activation.Mish)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class _DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvByActivation(128, 256, 3, 2, _Activation.Mish)
        self.conv2 = _ConvByActivation(256, 128, 1, 1, _Activation.Mish)
        self.conv3 = _ConvByActivation(256, 128, 1, 1, _Activation.Mish)

        self.resblock = _ResBlock(ch=128, nblocks=8)
        self.conv4 = _ConvByActivation(128, 128, 1, 1, _Activation.Mish)
        self.conv5 = _ConvByActivation(256, 256, 1, 1, _Activation.Mish)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class _DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvByActivation(256, 512, 3, 2, _Activation.Mish)
        self.conv2 = _ConvByActivation(512, 256, 1, 1, _Activation.Mish)
        self.conv3 = _ConvByActivation(512, 256, 1, 1, _Activation.Mish)

        self.resblock = _ResBlock(ch=256, nblocks=8)
        self.conv4 = _ConvByActivation(256, 256, 1, 1, _Activation.Mish)
        self.conv5 = _ConvByActivation(512, 512, 1, 1, _Activation.Mish)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class _DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvByActivation(512, 1024, 3, 2, _Activation.Mish)
        self.conv2 = _ConvByActivation(1024, 512, 1, 1, _Activation.Mish)
        self.conv3 = _ConvByActivation(1024, 512, 1, 1, _Activation.Mish)

        self.resblock = _ResBlock(ch=512, nblocks=4)
        self.conv4 = _ConvByActivation(512, 512, 1, 1, _Activation.Mish)
        self.conv5 = _ConvByActivation(1024, 1024, 1, 1, _Activation.Mish)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class _Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvByActivation(1024, 512, 1, 1, _Activation.Leaky)
        self.conv2 = _ConvByActivation(512, 1024, 3, 1, _Activation.Leaky)
        self.conv3 = _ConvByActivation(1024, 512, 1, 1, _Activation.Leaky)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        self.conv4 = _ConvByActivation(2048, 512, 1, 1, _Activation.Leaky)
        self.conv5 = _ConvByActivation(512, 1024, 3, 1, _Activation.Leaky)
        self.conv6 = _ConvByActivation(1024, 512, 1, 1, _Activation.Leaky)
        self.conv7 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)

        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv8 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)

        self.conv9 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)
        self.conv10 = _ConvByActivation(256, 512, 3, 1, _Activation.Leaky)
        self.conv11 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)
        self.conv12 = _ConvByActivation(256, 512, 3, 1, _Activation.Leaky)
        self.conv13 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)
        self.conv14 = _ConvByActivation(256, 128, 1, 1, _Activation.Leaky)

        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv15 = _ConvByActivation(256, 128, 1, 1, _Activation.Leaky)

        self.conv16 = _ConvByActivation(256, 128, 1, 1, _Activation.Leaky)
        self.conv17 = _ConvByActivation(128, 256, 3, 1, _Activation.Leaky)
        self.conv18 = _ConvByActivation(256, 128, 1, 1, _Activation.Leaky)
        self.conv19 = _ConvByActivation(128, 256, 3, 1, _Activation.Leaky)
        self.conv20 = _ConvByActivation(256, 128, 1, 1, _Activation.Leaky)

    def forward(
        self,
        downsample5: torch.Tensor,
        downsample4: torch.Tensor,
        downsample3: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.conv1(downsample5)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)

        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        up = self.upsample1(x7)

        x8 = self.conv8(downsample4)

        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        up = self.upsample2(x14)

        x15 = self.conv15(downsample3)

        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class _Yolov4Head(nn.Module):
    def __init__(self, output_ch: int, n_classes: int):
        """
        Creates the head of the Yolo model.

        Parameters
        ----------
        output_ch : int
            Number of the output channels.
        n_classes : int
            Number of classes in the dataset.
        """
        super().__init__()
        self.conv1 = _ConvByActivation(128, 256, 3, 1, _Activation.Leaky)
        self.conv2 = _ConvByActivation(
            256, output_ch, 1, 1, activation=None, bn=False, bias=True
        )

        self.conv3 = _ConvByActivation(128, 256, 3, 2, _Activation.Leaky)

        self.conv4 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)
        self.conv5 = _ConvByActivation(256, 512, 3, 1, _Activation.Leaky)
        self.conv6 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)
        self.conv7 = _ConvByActivation(256, 512, 3, 1, _Activation.Leaky)
        self.conv8 = _ConvByActivation(512, 256, 1, 1, _Activation.Leaky)
        self.conv9 = _ConvByActivation(256, 512, 3, 1, _Activation.Leaky)
        self.conv10 = _ConvByActivation(
            512, output_ch, 1, 1, activation=None, bn=False, bias=True
        )

        self.conv11 = _ConvByActivation(256, 512, 3, 2, _Activation.Leaky)

        self.conv12 = _ConvByActivation(1024, 512, 1, 1, _Activation.Leaky)
        self.conv13 = _ConvByActivation(512, 1024, 3, 1, _Activation.Leaky)
        self.conv14 = _ConvByActivation(1024, 512, 1, 1, _Activation.Leaky)
        self.conv15 = _ConvByActivation(512, 1024, 3, 1, _Activation.Leaky)
        self.conv16 = _ConvByActivation(1024, 512, 1, 1, _Activation.Leaky)
        self.conv17 = _ConvByActivation(512, 1024, 3, 1, _Activation.Leaky)
        self.conv18 = _ConvByActivation(
            1024, output_ch, 1, 1, activation=None, bn=False, bias=True
        )

    def forward(
        self, x: torch.Tensor, input2: torch.Tensor, input3: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        x3 = self.conv3(x)
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        x11 = self.conv11(x8)
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        return [x2, x10, x18]


class Yolov4(nn.Module):
    """
    Yolov4 model.

    It produces three output layers for small, medium and large objects:
    - `(batch_size, output_ch, 76, 76)`
    - `(batch_size, output_ch, 38, 38)`
    - `(batch_size, output_ch, 19, 19)`
    where `output_ch = n_boxes * (4 + 1 + n_classes)`.

    At each spatial location, the model predicts `n_boxes` detection vectors.
    Each vector consists of:
    - 4 bounding-box regression values (offsets for center, width, height)
    - objectness score
    - `n_classes` class scores

    For the COCO dataset, where `n_boxes = 3` and `n_classes = 80`:
    `output_ch = 3 * (4 + 1 + 80) = 255`.
    """

    def __init__(self, n_classes: int = 80):
        """
        Creates the Yolov4 model.

        Parameters
        ----------
        n_classes : int
            Number of classes in the dataset.
        """
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = _DownSample1()
        self.down2 = _DownSample2()
        self.down3 = _DownSample3()
        self.down4 = _DownSample4()
        self.down5 = _DownSample5()
        # neek
        self.neek = _Neck()
        # head
        self.head = _Yolov4Head(output_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output
