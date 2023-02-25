from typing import List

import torch
from torch.nn import Module, Sequential, ELU, Conv2d, Linear, BatchNorm2d
from torch.nn.functional import avg_pool2d

from utils import zero_initialize_layer

"""
The base of this code was taken from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = Conv2d(in_channels=planes, out_channels=self.expansion * planes, kernel_size=1)
        self.bn3 = BatchNorm2d(self.expansion * planes)
        self.act = ELU(inplace=True)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv2d(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride),
                BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNet(Module):
    def __init__(self, block, num_blocks: List[int], num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.cnn1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fnn1 = Linear(512 * block.expansion, num_classes)
        self.act = ELU()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def zero_initialization(self, mode: str, factor: float = 1.):
        for layer in [self.cnn1, self.fnn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            if isinstance(layer, Linear) or isinstance(layer, Conv2d):
                zero_initialize_layer(layer, mode, factor)
            elif isinstance(layer, Sequential):
                for sublayer in layer:
                    if isinstance(layer, Linear) or isinstance(layer, Conv2d):
                        zero_initialize_layer(sublayer, mode, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.cnn1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fnn1(x)
        return x


def ResNet50() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3])
