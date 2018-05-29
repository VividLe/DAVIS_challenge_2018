import os
from PIL import Image
import shutil

ori_cla_rp = '/home/zhangdong/database/DAVIS/Annotations/480p/'
res_rp = '/home/zhangdong/database/DAVIS/mAnnotations/'

cla_name_set = os.listdir(ori_cla_rp)
for cla_name in cla_name_set:
    print(cla_name)
    ori_img_rp = ori_cla_rp + cla_name + '/'

    img_name_set = os.listdir(ori_img_rp)
    for img_name in img_name_set:
        res_img_name = cla_name + img_name[:-4] + '.png'
        shutil.copyfile(ori_img_rp + img_name, res_rp + res_img_name)

# cla_name_set = os.listdir(ori_cla_rp)
# for cla_name in cla_name_set:
#     print(cla_name)
#     ori_img_rp = ori_cla_rp + cla_name + '/'
#
#     img_name_set = os.listdir(ori_img_rp)
#     for img_name in img_name_set:
#         img = Image.open(ori_img_rp + img_name)
#         res_img_name = cla_name + img_name[:-4] + '.png'
#         img.save(res_rp + res_img_name)
