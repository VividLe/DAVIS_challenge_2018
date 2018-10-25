import os
import shutil
from PIL import Image

img_rp = '/disk5/zhangdong/database/VOC2012_SEG_AUG/newsegmentations/'
img_set = os.listdir(img_rp)
img_set.sort()

# fol_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/MergeFiles/DAVIS/Annotations/480p/'
# fol_set = os.listdir(fol_rp)
# fol_set.sort()

num_max = 0
for img_name in img_set:
    mask = Image.open(img_rp+img_name)
    _, obj_num = mask.getextrema()
    if obj_num > num_max:
        num_max = obj_num
        name_record = img_name
print(name_record, num_max)

# for fol_name in fol_set:
#     img_rp = fol_rp + fol_name + '/'
#     img_set = os.listdir(img_rp)
#     if len(img_set) < 10:
#         print(fol_name, 'contains', len(img_set), 'images')
#         rm_rp = fol_rp + fol_name + '/'
#         shutil.rmtree(rm_rp)

# def concat_name(rp):
#     name_set = os.listdir(rp)
#     col_name = ' '
#     for iname in name_set:
#         col_name += iname
#     return col_name
#
# # for fol in range(1):
# #     fol_name = fol_set[fol]
# for fol_name in fol_set:
#     base_rp = fol_rp + fol_name + '/'
#     name_list = ['mask', 'gt', 'box', 'boxC']
#     name_img = concat_name(base_rp + 'image/')
#     for cla_name in name_list:
#         name = concat_name(base_rp + cla_name + '/')
#         if name != name_img:
#             print('image name not equal at path', base_rp + cla_name + '/')
#
# print('check finished')

# # check number equal
# def count_num(rp):
#     name_set = os.listdir(rp)
#     num = len(name_set)
#     return num
#
#
# for fol_name in fol_set:
#     base_rp = fol_rp + fol_name + '/'
#     name_list = ['mask', 'gt', 'box', 'boxC']
#     num_img = count_num(base_rp + 'image/')
#     if num_img == 0:
#         print('empty folder', fol_name)
#         continue
#     for cla_name in name_list:
#         num = count_num(base_rp + cla_name + '/')
#         if num != num_img:
#             print('image number not equal at path', base_rp + cla_name + '/')
#
# print('check finished')
