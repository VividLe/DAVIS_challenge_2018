import torch
from torch.autograd import Variable

import SegNet
import Vgg_FCN

model_VggFcn = Vgg_FCN.FCN32s()
VggFcn_weight_file = 'fcn32s_from_caffe.pth'
VggFcn_state_dict = torch.load(VggFcn_weight_file)

model_SegNet = SegNet.segnet()
SegNet_dict = model_SegNet.state_dict()

pretrained_dict = {k: v for k, v in VggFcn_state_dict.items() if k in SegNet_dict}
SegNet_dict.update(pretrained_dict)
model_SegNet.load_state_dict(SegNet_dict)

# for k in SegNet_dict:
#     print(k)

model = model_SegNet.cuda()
img = Variable(torch.randn((2, 3, 224, 224))).cuda()
mask = model(img)
print(len(mask))
for i in mask:
    print(i.size())
