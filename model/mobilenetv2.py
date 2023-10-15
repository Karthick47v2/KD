"""MobileNetV2 in pytorch

Reference:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
MobileNetV2: Inverted Residuals and Linear Bottlenecks
"""
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion=6):
        super().__init__()

        hidden_dim = in_channels * expansion

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):

    def __init__(self, class_num=100, input_channels=3):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, kernel_size=1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, expansion):

        layers = [LinearBottleNeck(
            in_channels, out_channels, stride, expansion)]

        for _ in range(repeat - 1):
            layers.append(LinearBottleNeck(
                out_channels, out_channels, 1, expansion))

        return nn.Sequential(*layers)


def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)
