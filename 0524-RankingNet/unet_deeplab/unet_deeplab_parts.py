#!/usr/bin/python
#coding:utf-8
# sub-parts of the U-Net_deeplab model

import torch
import torch.nn as nn
import torch.nn.functional as F


class oper_conv(nn.Module):
    '''(conv => BN => ReLU) '''
    def __init__(self, in_ch, out_ch):
        super(oper_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        nn.init.xavier_uniform(self.conv[0].weight),
        nn.init.constant(self.conv[0].bias, 0),

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv2 = oper_conv(in_ch*2, out_ch)
        self.bn_for_x2 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
       
        x1 = self.up(x1)
        x2 = self.bn_for_x2(x2)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
 
        x = self.conv2(x)
        return x

class up_second(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_second, self).__init__()
        # in_ch*2 is 1024 because of concatenation
        self.conv = oper_conv(in_ch*2, out_ch)
        self.bn_for_x2 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x2 = self.bn_for_x2(x2)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
    
        x = self.conv(x)
        return x


