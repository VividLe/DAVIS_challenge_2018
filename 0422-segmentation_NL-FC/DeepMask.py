import torch.nn as nn

import SegNet_resnet_utils as uitls


class DeepMask(nn.Module):

    def __init__(self, in_planes, mid_planes, edge):
        super(DeepMask, self).__init__()
        self.edge = edge
        self.bn = nn.BatchNorm2d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.conv1 = uitls.conv1x1(in_planes, mid_planes)
        self.conv2 = uitls.conv3x3(mid_planes, mid_planes)
        self.conv3 = uitls.conv1x1(mid_planes, 2)
        self.LSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = uitls.center_crop(x, self.edge, self.edge)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.LSoftmax(x)

        return x



