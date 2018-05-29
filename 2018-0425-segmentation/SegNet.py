import torch.nn as nn

import SegNet_utils as utils
import SegNet_resnet_utils as uitls
import SkipConnection
import DeepMask


class SegNet(nn.Module):

    def __init__(self, skip_connection, deep_mask=None):
        super(SegNet, self).__init__()
        self.deep_mask = deep_mask
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(num_features=256)  # 1/4

        # conv4
        self.conv4_1 = utils.dila_conv3x3(in_planes=256, out_planes=512, dilation=2)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = utils.dila_conv3x3(512, 512, 2)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = utils.dila_conv3x3(512, 512, 2)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(num_features=512)  # 1/4

        # conv5
        self.conv5_1 = utils.dila_conv3x3(in_planes=512, out_planes=512, dilation=4)
        self.relu5_1 = nn.ReLU(inplace=True)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = utils.dila_conv3x3(512, 512, 4)
        self.relu5_2 = nn.ReLU(inplace=True)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = utils.dila_conv3x3(512, 512, 4)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(num_features=512)  # 1/4

        # bottleneck
        self.bottleneck = utils.conv3x3(in_planes=512, out_planes=512)
        self.bn_bot = nn.BatchNorm2d(num_features=512)
        self.relu_bot = nn.ReLU(inplace=True)

        # skip connection and transition convolution
        # for ResNet 50,101 etc.
        self.skip4 = self._make_connection(skip_connection, 512, 512, 128)
        self.skip3 = self._make_connection(skip_connection, 256, 256, 64, transition_up=True)
        self.skip2 = self._make_connection(skip_connection, 128, 128, 32, transition_up=True)
        self.skip1 = self._make_connection(skip_connection, 64, 64, 16)

        self.last_conv = uitls.conv1x1(32, 2)
        self.map = nn.LogSoftmax(dim=1)

        if deep_mask is not None:
            self.mask5 = self._gen_mask(deep_mask, 512, 64, edge=56)
            self.mask4 = self._gen_mask(deep_mask, 256, 32, edge=56)
            self.mask3 = self._gen_mask(deep_mask, 128, 16, edge=56)
            self.mask2 = self._gen_mask(deep_mask, 64, 8, edge=112)
            self.mask1 = self._gen_mask(deep_mask, 32, 8, edge=224)

        # self._initialize_weights()

    def _make_connection(self, skip_connection, in_planes, skip_planes, out_planes_half, transition_up=False):
        layers = []
        layers.append(skip_connection(in_planes, skip_planes, out_planes_half, transition_up))
        return nn.Sequential(*layers)

    def _gen_mask(self, deep_mask, in_planes, mid_planes, edge):
        layers = []
        layers.append(deep_mask(in_planes, mid_planes, edge))
        return nn.Sequential(*layers)


    def forward(self, x):
        skip_connect = []

        h = self.relu1_1(self.conv1_1(x))
        h = self.relu1_2(self.conv1_2(h))
        skip_connect.append(h)
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        skip_connect.append(h)
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        skip_connect.append(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        skip_connect.append(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))

        fea_map = self.relu_bot(self.bn_bot(self.bottleneck(h)))
        # fea_map = self.bottleneck(h)
        if self.deep_mask is not None:
            mask5 = self.mask5(fea_map)
        # return [mask5]

        # skip connection and transition convolution
        skip = skip_connect.pop()
        x = self.skip4((fea_map, skip)) # 56
        if self.deep_mask is not None:
            mask4 = self.mask4(x)

        skip = skip_connect.pop()
        x = self.skip3((x, skip)) # 112
        if self.deep_mask is not None:
            mask3 = self.mask3(x)

        skip = skip_connect.pop()
        x = self.skip2((x, skip)) # 224
        if self.deep_mask is not None:
            mask2 = self.mask2(x)

        skip = skip_connect.pop()
        x = self.skip1((x, skip)) # 224
        if self.deep_mask is not None:
            mask1 = self.mask1(x)

        if self.deep_mask is not None:
            return [mask5, mask4, mask3, mask2, mask1]
        else:
            x = self.last_conv(x)
            mask = self.map(x)
            return  [mask]


def segnet():
    # model = SegNet(skip_connection=SkipConnection.SkipConnection, deep_mask=DeepMask.DeepMask)
    model = SegNet(skip_connection=SkipConnection.SkipConnection)
    return model
