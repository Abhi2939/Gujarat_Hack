from __future__ import annotations

import torch
import torch.nn as nn


class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class EnhanceNetNoPool(nn.Module):
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor) if scale_factor != 1 else None

        number_f = 32
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def enhance(self, x, x_r):
        for _ in range(4):
            x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x
        for _ in range(4):
            x = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image_1, x

    def forward(self, x):
        if self.upsample is not None:
            x_down = nn.functional.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        else:
            x_down = x

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.upsample is not None:
            x_r = self.upsample(x_r)

        _, enhanced = self.enhance(x, x_r)
        return enhanced