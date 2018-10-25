import torch.nn as nn

import SegNet_utils as uitls

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=None):
        super(BasicBlock, self).__init__()
        if dilation is None:
            self.conv1 = uitls.conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = uitls.dila_conv3x3(inplanes, planes, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if dilation is None:
            self.conv2 = uitls.conv3x3(planes, planes)
        else:
            self.dial_conv2 = uitls.dila_conv3x3(planes, planes, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out