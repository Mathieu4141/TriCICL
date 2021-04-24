"""
This implementation is largely inspired by
https://github.com/akamaster/pytorch_resnet_cifar10
"""
from typing import Callable

from torch import Tensor, relu
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, Sequential, init
from torch.nn.functional import avg_pool2d

from tricicl.models.feature_based_module import FeatureBasedModule


class ResNet32(FeatureBasedModule):
    def __init__(self, n_classes: int):
        super().__init__(n_classes)

        self.in_planes = 16

        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 5, stride=1)
        self.layer2 = self._make_layer(32, 5, stride=2)
        self.layer3 = self._make_layer(64, 5, stride=2)

        self.classifier = Linear(64, n_classes)

        self.apply(_weights_init)

    def featurize(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

    def classify(self, features: Tensor) -> Tensor:
        return self.classifier(features)

    def _make_layer(self, planes: int, num_blocks: int, stride) -> Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.EXPANSION

        return Sequential(*layers)


def _weights_init(m: Module):
    if isinstance(m, Linear) or isinstance(m, Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(Module):
    def __init__(self, lambda_: Callable[[Tensor], Tensor]):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return self.lambda_(x)


class BasicBlock(Module):
    EXPANSION = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, self.EXPANSION * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.EXPANSION * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out
