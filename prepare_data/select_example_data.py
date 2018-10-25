import os
import random
import shutil

ori_img_rp = '/disk5/yangle/DAVIS/dataset/saliency/train/'
ori_mask_rp = '/disk5/yangle/DAVIS/dataset/saliency/trainannot/'
res_img_rp = '/disk5/yangle/DAVIS/dataset/saliency/val/'
res_mask_rp = '/disk5/yangle/DAVIS/dataset/saliency/valannot/'

if not os.path.exists(res_img_rp):
    os.makedirs(res_img_rp)
if not os.path.exists(res_mask_rp):
    os.makedirs(res_mask_rp)

img_name_set = os.listdir(ori_img_rp)
img_name_set.sort()

val_img_set = random.sample(img_name_set, 2557)

for img_name in val_img_set:
    print(img_name)
    res_name = img_name
    shutil.move(ori_img_rp+img_name, res_img_rp+res_name)
    shutil.move(ori_mask_rp+img_name, res_mask_rp+res_name)
    # shutil.copyfile(ori_img_rp+img_name, res_img_rp+res_name)
    # shutil.copyfile(ori_mask_rp+img_name, res_mask_rp+res_name)

