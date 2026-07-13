# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv4 implementation in pytorch based on https://github.com/Tianxiaomo/pytorch-YOLOv4.
"""

import torch
import torch.nn.functional as F
from torch import nn

from kenning.utils.logger import KLogger


class _Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class _Upsample(nn.Module):
    def __init__(self):
        super(_Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert x.data.dim() == 4
        # _, _, tH, tW = target_size

        if inference:
            """
            B = x.data.size(0)
            C = x.data.size(1)
            H = x.data.size(2)
            W = x.data.size(3)
            """

            return (
                x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1)
                .expand(
                    x.size(0),
                    x.size(1),
                    x.size(2),
                    target_size[2] // x.size(2),
                    x.size(3),
                    target_size[3] // x.size(3),
                )
                .contiguous()
                .view(x.size(0), x.size(1), target_size[2], target_size[3])
            )
        else:
            return F.interpolate(
                x, size=(target_size[2], target_size[3]), mode="nearest"
            )


class _Conv_Bn_Activation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation,
        bn=True,
        bias=False,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
            )
        else:
            self.conv.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    pad,
                    bias=False,
                )
            )
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(_Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            KLogger.error(f"Activation {activation} is not supported.")

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x


class _ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(_Conv_Bn_Activation(ch, ch, 1, 1, "mish"))
            resblock_one.append(_Conv_Bn_Activation(ch, ch, 3, 1, "mish"))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class _DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _Conv_Bn_Activation(3, 32, 3, 1, "mish")

        self.conv2 = _Conv_Bn_Activation(32, 64, 3, 2, "mish")
        self.conv3 = _Conv_Bn_Activation(64, 64, 1, 1, "mish")
        # [route]
        # layers = -2
        self.conv4 = _Conv_Bn_Activation(64, 64, 1, 1, "mish")

        self.conv5 = _Conv_Bn_Activation(64, 32, 1, 1, "mish")
        self.conv6 = _Conv_Bn_Activation(32, 64, 3, 1, "mish")
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = _Conv_Bn_Activation(64, 64, 1, 1, "mish")
        # [route]
        # layers = -1, -7
        self.conv8 = _Conv_Bn_Activation(128, 64, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class _DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _Conv_Bn_Activation(64, 128, 3, 2, "mish")
        self.conv2 = _Conv_Bn_Activation(128, 64, 1, 1, "mish")
        # r -2
        self.conv3 = _Conv_Bn_Activation(128, 64, 1, 1, "mish")

        self.resblock = _ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = _Conv_Bn_Activation(64, 64, 1, 1, "mish")
        # r -1 -10
        self.conv5 = _Conv_Bn_Activation(128, 128, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
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
        self.conv1 = _Conv_Bn_Activation(128, 256, 3, 2, "mish")
        self.conv2 = _Conv_Bn_Activation(256, 128, 1, 1, "mish")
        self.conv3 = _Conv_Bn_Activation(256, 128, 1, 1, "mish")

        self.resblock = _ResBlock(ch=128, nblocks=8)
        self.conv4 = _Conv_Bn_Activation(128, 128, 1, 1, "mish")
        self.conv5 = _Conv_Bn_Activation(256, 256, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
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
        self.conv1 = _Conv_Bn_Activation(256, 512, 3, 2, "mish")
        self.conv2 = _Conv_Bn_Activation(512, 256, 1, 1, "mish")
        self.conv3 = _Conv_Bn_Activation(512, 256, 1, 1, "mish")

        self.resblock = _ResBlock(ch=256, nblocks=8)
        self.conv4 = _Conv_Bn_Activation(256, 256, 1, 1, "mish")
        self.conv5 = _Conv_Bn_Activation(512, 512, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
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
        self.conv1 = _Conv_Bn_Activation(512, 1024, 3, 2, "mish")
        self.conv2 = _Conv_Bn_Activation(1024, 512, 1, 1, "mish")
        self.conv3 = _Conv_Bn_Activation(1024, 512, 1, 1, "mish")

        self.resblock = _ResBlock(ch=512, nblocks=4)
        self.conv4 = _Conv_Bn_Activation(512, 512, 1, 1, "mish")
        self.conv5 = _Conv_Bn_Activation(1024, 1024, 1, 1, "mish")

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class _neek(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = _Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv2 = _Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv3 = _Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = _Conv_Bn_Activation(2048, 512, 1, 1, "leaky")
        self.conv5 = _Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv6 = _Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv7 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        # UP
        self.upsample1 = _Upsample()
        # R 85
        self.conv8 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        # R -1 -3
        self.conv9 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv10 = _Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv11 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv12 = _Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv13 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv14 = _Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        # UP
        self.upsample2 = _Upsample()
        # R 54
        self.conv15 = _Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        # R -1 -3
        self.conv16 = _Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.conv17 = _Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv18 = _Conv_Bn_Activation(256, 128, 1, 1, "leaky")
        self.conv19 = _Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv20 = _Conv_Bn_Activation(256, 128, 1, 1, "leaky")

    def forward(self, input, downsample4, downsample3, inference=False):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class _Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = _Conv_Bn_Activation(128, 256, 3, 1, "leaky")
        self.conv2 = _Conv_Bn_Activation(
            256, output_ch, 1, 1, "linear", bn=False, bias=True
        )

        # R -4
        self.conv3 = _Conv_Bn_Activation(128, 256, 3, 2, "leaky")

        # R -1 -16
        self.conv4 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv5 = _Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv6 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv7 = _Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv8 = _Conv_Bn_Activation(512, 256, 1, 1, "leaky")
        self.conv9 = _Conv_Bn_Activation(256, 512, 3, 1, "leaky")
        self.conv10 = _Conv_Bn_Activation(
            512, output_ch, 1, 1, "linear", bn=False, bias=True
        )

        # R -4
        self.conv11 = _Conv_Bn_Activation(256, 512, 3, 2, "leaky")

        # R -1 -37
        self.conv12 = _Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv13 = _Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv14 = _Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv15 = _Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv16 = _Conv_Bn_Activation(1024, 512, 1, 1, "leaky")
        self.conv17 = _Conv_Bn_Activation(512, 1024, 3, 1, "leaky")
        self.conv18 = _Conv_Bn_Activation(
            1024, output_ch, 1, 1, "linear", bn=False, bias=True
        )

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
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
    """

    def __init__(
        self, yolov4conv137weight=None, n_classes=80, inference=False
    ):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = _DownSample1()
        self.down2 = _DownSample2()
        self.down3 = _DownSample3()
        self.down4 = _DownSample4()
        self.down5 = _DownSample5()
        # neek
        self.neek = _neek(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(
                self.down1,
                self.down2,
                self.down3,
                self.down4,
                self.down5,
                self.neek,
            )
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {
                k1: v
                for (k, v), k1 in zip(pretrained_dict.items(), model_dict)
            }
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)

        # head
        self.head = _Yolov4Head(output_ch, n_classes, inference)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output
