import os
from PIL import Image
import numpy as np

ori_img_dir = '/disk5/yangle/DAVIS/dataset/saliency/train/'
res_img_dir = '/disk5/yangle/DAVIS/dataset/saliency/train_jpg/'

# img_set = os.listdir(res_img_dir)
# for name in img_set:
#     img = Image.open(res_img_dir + name)
#     img = np.asarray(img)
#     shape = img.shape
#     if len(shape) != 3:
#         print(name)

img_set = os.listdir(ori_img_dir)
for name in img_set:
    img = Image.open(ori_img_dir + name)
    res_name = name[:-3] + 'jpg'
    img.save(res_img_dir + res_name)

