import os
from PIL import Image
import numpy as np
import shutil

# ## Run Object Detection
ref_cate_mask_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/Annotations/480p/'
ori_cate_img_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/Images/'
ori_mask_rp = '/disk5/yangle/DAVIS/result/DAVIS-maskrcnn/0510-0856-several_objects/'
res_img_rp = '/disk5/yangle/DAVIS/dataset/sig_obj/test/'
res_cont_rp = '/disk5/yangle/DAVIS/dataset/sig_obj/testcont/'

cate_set = os.listdir(ori_cate_img_rp)
cate_set.sort()

for cate_name in cate_set:

    ann_mask_file = ref_cate_mask_rp + cate_name + '/00000.png'
    mask = Image.open(ann_mask_file)
    mask = np.asarray(mask)
    num_m = mask.max()
    if num_m != 1:
        print('skip category %s'.format(cate_name))
        continue

    ori_img_rp = ori_cate_img_rp + cate_name + '/'
    img_set = os.listdir(ori_img_rp)
    img_set.sort()

    for img_name in img_set:
        print(img_name)
        cate_img_name = cate_name+img_name
        shutil.copyfile(ori_img_rp+img_name, res_img_rp+cate_img_name)
        mask = Image.open(ori_mask_rp+cate_img_name)
        mask = mask.point(lambda i: i*255)
        mask.save(res_cont_rp+cate_img_name)

