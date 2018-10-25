import torch.nn as nn

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                     kernel_size=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def dila_conv3x3(in_planes, out_planes, dilation):
    return  nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3,
                      dilation=dilation, padding=dilation, bias=False)

def center_crop(layer, max_height, max_width):
    #https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    #Author does a center crop which crops both inputs (skip and upsample) to size of minimum dimension on both w/h
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]