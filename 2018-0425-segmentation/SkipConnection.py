import torch
import torch.nn as nn
import torch.nn.functional as F

import SegNet_resnet_utils as uitls


class SkipConnection(nn.Module):

    def __init__(self, in_planes, out_planes, transition_up=False):
        super(SkipConnection, self).__init__()
        self.transition_up = transition_up
        self.conv_cat = uitls.conv3x3(in_planes, out_planes)
        self.bn_cat = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, data):
        x_in = data[0]
        x_skip = data[1]
        if self.transition_up:
            x_in = F.upsample(x_in, scale_factor=2, mode='bilinear')

        # center crop x_in to facilitate the cat operation
        x_in = uitls.center_crop(x_in, x_skip.size(2), x_skip.size(3))
        out = torch.cat((x_in, x_skip), dim=1)
        out = self.conv_cat(out)
        out = self.bn_cat(out)
        out = self.relu(out)

        return out