"""NAFNet architecture for the external-backbone replication.

Copied without behavioral changes from
https://github.com/megvii-research/NAFNet at commit
2b4af71ebe098a92a75910c233a3965a3e93ede4.
Copyright (c) 2022 megvii-model. Licensed under the MIT license.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value):
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).square().mean(1, keepdim=True)
        normalized = (value - mean) / torch.sqrt(variance + self.eps)
        return (
            self.weight.view(1, -1, 1, 1) * normalized
            + self.bias.view(1, -1, 1, 1)
        )


class SimpleGate(nn.Module):
    def forward(self, value):
        first, second = value.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channels // 2,
                dw_channels // 2,
                kernel_size=1,
                bias=True,
            ),
        )
        self.sg = SimpleGate()
        ffn_channels = ffn_expand * channels
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1, bias=True)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        )
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp):
        value = self.norm1(inp)
        value = self.conv1(value)
        value = self.conv2(value)
        value = self.sg(value)
        value = value * self.sca(value)
        value = self.conv3(value)
        value = self.dropout1(value)
        first = inp + value * self.beta
        value = self.conv4(self.norm2(first))
        value = self.sg(value)
        value = self.conv5(value)
        value = self.dropout2(value)
        return first + value * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=(1, 1, 1, 28),
        dec_blk_nums=(1, 1, 1, 1),
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width
        for blocks in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(blocks)])
            )
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, 2))
            channels *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(middle_blk_num)]
        )

        for blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, 2 * channels, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(blocks)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, value):
        height, width = value.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(value, (0, pad_w, 0, pad_h))

    def forward(self, inp):
        height, width = inp.shape[-2:]
        inp = self.check_image_size(inp)
        value = inp
        value = self.intro(value)
        encoders = []
        for encoder, down in zip(self.encoders, self.downs):
            value = encoder(value)
            encoders.append(value)
            value = down(value)
        value = self.middle_blks(value)
        for decoder, up, skip in zip(
            self.decoders, self.ups, encoders[::-1]
        ):
            value = up(value)
            value = value + skip
            value = decoder(value)
        value = self.ending(value)
        value = value + inp
        return value[:, :, :height, :width]
