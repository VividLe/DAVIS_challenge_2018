import torch.nn as nn

import SegNet_resnet_utils as uitls


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=None):
        super(Bottleneck, self).__init__()
        self.conv1 = uitls.conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if dilation is None:
            self.conv2 = uitls.conv3x3(planes, planes, stride)
        else:
            self.conv2 = uitls.dila_conv3x3(planes, planes, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = uitls.conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out