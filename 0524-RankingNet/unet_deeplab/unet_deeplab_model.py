#!/usr/bin/python
# full assembly of the sub-parts to form the complete net
# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


from .unet_deeplab_parts import *


class UNet_deeplab(nn.Module):
    def __init__(self, in_channels, feature_length):
        super(UNet_deeplab, self).__init__()
        '''(conv => BN => ReLU) '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        nn.init.xavier_uniform(self.conv1[0].weight),
        nn.init.constant(self.conv1[0].bias, 0),
        nn.init.xavier_uniform(self.conv1[0].weight),
        nn.init.constant(self.conv1[0].bias, 0),

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.fc7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=7),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=feature_length, kernel_size=1)
        )
        nn.init.xavier_uniform(self.fc7_1[0].weight),
        nn.init.constant(self.fc7_1[0].bias, 0),

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.fc6(x5)
        x7 = self.fc7(x6)
        x7 = self.fc7_1(x7)

        return x7

    def init_parameters(self, pretrain_vgg16_1024):
        ##### init parameter using pretrain deeplab model ###########
        conv_blocks = [self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5,
                       self.fc6]
        listkey = [['conv2_1', 'conv2_2'], ['conv3_1', 'conv3_2', 'conv3_3'],
                   ['conv4_1', 'conv4_2', 'conv4_3'], ['conv5_1', 'conv5_2', 'conv5_3'], ['fc6_2']]
        for idx, conv_block in enumerate(conv_blocks):
            num_conv = 0
            for l2 in conv_block:
                if isinstance(l2, nn.Conv2d):
                    num_conv += 1
                    l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.weight']
                    l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.bias']
                    # print(l2.bias.data)
        return self


