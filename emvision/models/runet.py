"""
Residual U-Net with same or valid convolutions
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from . import utils


__all__ = ["RUNet"]


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = False,
    mode: str = "valid",
):
    """Convolution paired with the required padding."""
    padding = utils.pad_size(kernel_size, mode)
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


class INReLU(nn.Sequential):
    def __init__(self, in_channels, norm=nn.InstanceNorm3d, activation=nn.ReLU):
        super(INReLU, self).__init__()
        if norm:
            self.add_module("norm", norm(in_channels, affine=True))
        self.add_module("relu", activation(inplace=True))


class INReLUConv(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, mode="valid", norm=None
    ):
        super(INReLUConv, self).__init__()
        self.add_module("norm_relu", INReLU(in_channels, norm=norm))
        self.add_module(
            "conv", conv(in_channels, out_channels, kernel_size=kernel_size, mode=mode, bias=not norm)
        )

        self.mode = mode
        self.crop_margin = self.compute_crop()

    def compute_crop(self):
        return utils.crop_margin(self.conv.kernel_size, self.mode)


class ResBlock(nn.Module):
    def __init__(self, channels, mode="valid", norm=None):
        super(ResBlock, self).__init__()
        self.conv1 = INReLUConv(channels, channels, mode=mode, norm=norm)
        self.conv2 = INReLUConv(channels, channels, mode=mode, norm=norm)
        self.inner_crop_margin = utils.sum3(
            self.conv1.crop_margin, self.conv2.crop_margin
        )

        self.crop_margin = self.compute_crop()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += utils.crop3d(residual, self.inner_crop_margin)
        return x

    def compute_crop(self):
        return tuple(map(sum, zip(*(m.crop_margin for n, m in self.named_children()))))


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, mode="valid", norm=None):
        super(ConvBlock, self).__init__()
        self.add_module(
            "pre", INReLUConv(in_channels, out_channels, mode=mode, norm=norm)
        )
        self.add_module("res", ResBlock(out_channels, mode=mode, norm=norm))
        self.add_module(
            "post", INReLUConv(out_channels, out_channels, mode=mode, norm=norm)
        )

        self.crop_margin = self.compute_crop()

    def compute_crop(self):
        return tuple(map(sum, zip(*(m.crop_margin for n, m in self.named_children()))))


class DownConvBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, scale_factor, mode="valid", norm=None
    ):
        super(DownConvBlock, self).__init__()
        self.add_module("maxpool", nn.MaxPool3d((2, 2, 2)))
        self.add_module(
            "conv", ConvBlock(in_channels, out_channels, mode=mode, norm=norm)
        )

        self.crop_margin = self.compute_crop()

    def compute_crop(self):
        return self.conv.crop_margin


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(1, 2, 2), norm=None):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="trilinear"),
            conv(in_channels, out_channels, kernel_size=1, bias=not norm),
        )

    def forward(self, x, skip, margin):
        return self.up(x) + utils.crop3d(skip, margin)


class UpConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, scale_factor=(1, 2, 2), mode="valid", norm=None
    ):
        super(UpConvBlock, self).__init__()
        self.up = UpBlock(in_channels, out_channels, scale_factor=scale_factor, norm=norm)
        self.conv = ConvBlock(out_channels, out_channels, mode=mode, norm=norm)

        self.crop_margin = self.compute_crop()
        self.scale_factor = scale_factor

    def forward(self, x, skip, margin):
        x = self.up(x, skip, margin)
        return self.conv(x)

    def compute_crop(self):
        return self.conv.crop_margin


class RUNet(nn.Module):
    def __init__(
        self,
        width: list[int] = [16, 32, 64, 128, 256, 512],
        scale_factor: tuple[int, int, int] = (2, 2, 2),
        mode: str = "valid",
        norm: Optional[nn.Module] = nn.InstanceNorm3d,
    ):
        super(RUNet, self).__init__()
        assert len(width) > 1
        depth = len(width) - 1

        self.iconv = ConvBlock(width[0], width[0], norm=norm)

        self.dconvs = nn.ModuleList()
        for d in range(depth):
            self.dconvs.append(
                DownConvBlock(
                    width[d], width[d + 1], scale_factor, mode=mode, norm=norm
                )
            )

        self.uconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            self.uconvs.append(
                UpConvBlock(width[d + 1], width[d], scale_factor, mode=mode, norm=norm)
            )

        self.final = INReLU(width[0], norm=norm)

        self.scale_factor = scale_factor
        self.inner_crop_margins = self.compute_inner_crops()
        self.init_weights()

        self.crop_margin = self.compute_crop()

    def forward(self, x):
        x = self.iconv(x)

        skip = list()
        for dconv in self.dconvs:
            skip.append(x)
            x = dconv(x)

        for uconv, margin in zip(self.uconvs, self.inner_crop_margins):
            x = uconv(x, skip.pop(), margin)

        return self.final(x)

    def compute_inner_crops(self):
        crops = list()
        prev_up_crop = (0, 0, 0)
        prev_total_crop = (0, 0, 0)

        for uconv, dconv in zip(self.uconvs, reversed(self.dconvs)):
            # total_crop = (prev + dconv + prev_uconv) * scale_factor

            # new crop from this level of the hierarchy
            new_crop = utils.sum3(dconv.crop_margin, prev_up_crop)
            total_crop = utils.mul3(
                utils.sum3(new_crop, prev_total_crop), uconv.scale_factor
            )

            crops.append(total_crop)

            prev_total_crop = total_crop
            prev_up_crop = uconv.crop_margin

        return crops

    def compute_crop(self):
        level0_crops = utils.sum3(self.iconv.crop_margin, self.uconvs[-1].crop_margin)

        return utils.sum3(level0_crops, self.inner_crop_margins[-1])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def input_patch_sizes(self, num_shapes):
        """Computes acceptable patch sizes for valid convs"""
        mintopsz = (0, 0, 0)
        for uconv in reversed(self.uconvs):
            crop = utils.mul3(uconv.crop_margin, (2, 2, 2))
            mintopsz = utils.sum3(mintopsz, crop)
            mintopsz = utils.div3(mintopsz, (2, 2, 2))

        mintopsz = utils.sum3(mintopsz, (1, 1, 1))  # need at least one extra voxel

        def input_size(topsz):
            sz = topsz
            for dconv in self.dconvs:
                crop = utils.mul3(dconv.crop_margin, (2, 2, 2))
                sz = utils.sum3(sz, crop)
                sz = utils.mul3(sz, self.scale_factor)
            return utils.sum3(sz, utils.mul3(self.iconv.crop_margin, (2, 2, 2)))

        return [input_size(utils.sum3(mintopsz, (i, i, i))) for i in range(num_shapes)]

    def output_patch_sizes(self, num_shapes):
        """Computes output patch sizes for the valid input patch sizes"""
        return [
            utils.sub3(inputsz, utils.mul3(self.crop_margin, (2, 2, 2)))
            for inputsz in self.input_patch_sizes(num_shapes)
        ]
