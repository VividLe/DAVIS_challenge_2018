import torch.nn as nn
import torch.nn.functional as F

import SamePad2d


class TopDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.padding2 = SamePad2d.SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x, y):
        y = F.upsample(y, scale_factor=2)
        x = self.conv1(x)
        return self.conv2(self.padding2(x+y))