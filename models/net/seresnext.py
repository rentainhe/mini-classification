import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class SEResNeXtBottleneck(nn.Module):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(
            inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se_module(out) + shortcut
        out = self.relu(out)

        return out

class SEResNext(nn.Module):

    def __init__(self, block, block_num, groups=32, reduction=16, class_num=100):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1, groups, reduction)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2, groups, reduction)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2, groups, reduction)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2, groups, reduction)

        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x


    def _make_stage(self, block, num, out_channels, stride, groups, reduction):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, groups=groups, reduction=reduction))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, stride=1, groups=groups, reduction=reduction))
            num -= 1

        return nn.Sequential(*layers)


def seresnext26():
    return SEResNext(SEResNeXtBottleneck, [2, 2, 2, 2], groups=32, reduction=16)

def seresnext50():
    return SEResNext(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16)

def seresnext101():
    return SEResNext(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16)