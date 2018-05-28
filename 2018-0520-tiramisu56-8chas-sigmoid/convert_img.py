import os
from PIL import Image
import numpy as np

ori_img_dir = '/disk2/zhangni/davis/saliency/train_jpg/'
res_img_dir = '/disk2/zhangni/davis/saliency/train/'

img_set = os.listdir(ori_img_dir)
for name in img_set:
    img = Image.open(ori_img_dir + name)
    res_name = name[:-3] + 'png'
    img.save(res_img_dir + res_name)

