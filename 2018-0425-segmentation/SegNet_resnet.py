import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import BasicBlock
import Bottleneck
import SkipConnection
import DeepMask
import SegNet_resnet_utils as uitls


class ResNet(nn.Module):

    def __init__(self, block, layers, skip_connection, deep_mask=None):
        super(ResNet, self).__init__()
        self.deep_mask = deep_mask
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # remove the pooling operation, use dilation convolution
        self.layer3 = self._make_layer(block, 256, layers[2], dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dilation=4)

        # bottleneck
        self.bottleneck = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)

        # skip connection and transition convolution
        # for ResNet 50,101 etc.
        self.skip3 = self._make_connection(skip_connection, in_planes=1024+1024, out_planes=512)
        self.skip2 = self._make_connection(skip_connection, in_planes=512+512, out_planes=256)
        self.skip1 = self._make_connection(skip_connection, in_planes=256+256, out_planes=64, transition_up=True)
        self.skip0 = self._make_connection(skip_connection, in_planes=64+64, out_planes=64, transition_up=True)

        self.conv_end_1 = nn.Conv2d(64, 64, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        self.bn_end_1 = nn.BatchNorm2d(64)
        self.salmap = nn.Conv2d(64, 2, kernel_size=1, bias=False)
        self.LSoftmax = nn.LogSoftmax(dim=1)

        self.mask0 = uitls.conv1x1(64, 2)
        if deep_mask is not None:
            self.mask4 = uitls.conv1x1(1024, 2)
            self.mask3 = uitls.conv1x1(512, 2)
            self.mask2 = uitls.conv1x1(256, 2)
            self.mask1 = uitls.conv1x1(64, 2)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _make_connection(self, skip_connection, in_planes, out_planes, transition_up=False):
        layers = []
        layers.append(skip_connection(in_planes, out_planes, transition_up))

        return nn.Sequential(*layers)


    def forward(self, x):
        skip_connect = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 112
        skip_connect.append(x)
        # 56
        x = self.maxpool(x)

        # 56
        x = self.layer1(x)
        skip_connect.append(x)
        # 28
        x = self.layer2(x)
        skip_connect.append(x)
        x = self.layer3(x)
        skip_connect.append(x)
        x = self.layer4(x)

        # bottleneck
        fea_map = self.bottleneck(x)
        mask4 = self.mask4(fea_map)
        mask4 = self.LSoftmax(mask4)

        skip = skip_connect.pop()
        # 28
        x = self.skip3((fea_map, skip))
        mask3 = self.mask3(x)
        mask3 = self.LSoftmax(mask3)

        skip = skip_connect.pop()
        # 28
        x = self.skip2((x, skip))
        mask2 = self.mask2(x)
        mask2 = self.LSoftmax(mask2)

        skip = skip_connect.pop()
        # 56
        x = self.skip1((x, skip))
        mask1 = self.mask1(x)
        mask1 = self.LSoftmax(mask1)

        skip = skip_connect.pop()
        # 112
        x = self.skip0((x, skip))
        x = self.mask0(x)
        mask0 = self.LSoftmax(x)

        if self.deep_mask is None:
            return [mask0]
        else:
            return [mask4, mask3, mask2, mask1, mask0]


def resnet18():
    model = ResNet(BasicBlock.BasicBlock, [2, 2, 2, 2], SkipConnection.SkipConnection)
    return model

def resnet34():
    model = ResNet(BasicBlock.BasicBlock, [3, 4, 6, 3], SkipConnection.SkipConnection)
    return model

def resnet50():
    model = ResNet(Bottleneck.Bottleneck, [3, 4, 6, 3], SkipConnection.SkipConnection, DeepMask.DeepMask)
    return model

def resnet101():
    model = ResNet(Bottleneck.Bottleneck, [3, 4, 23, 3], SkipConnection.SkipConnection)
    return model

# image = torch.randn((1, 3, 224, 224))
# image = Variable(image).cuda()
# model = resnet50().cuda()
# result = model(image)


