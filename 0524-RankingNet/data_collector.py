import os
import shutil
import random
from PIL import Image
import numpy as np

img_rp = '/disk5/yangle/DAVIS/dataset/Ranking/Val/img_cont/img/'
cont_rp = '/disk5/yangle/DAVIS/dataset/Ranking/Val/img_cont/cont/'
data_rp = '/disk5/yangle/DAVIS/dataset/Ranking/processing/'


def copy_img_cont(ori_file, res_file):
    # image
    shutil.copyfile(ori_file, res_file)
    # context
    ori_file_cont = ori_file.replace('/img/', '/cont/').replace('.jpg', '.png')
    res_file_cont = res_file.replace('.jpg', '.png')
    shutil.copyfile(ori_file_cont, res_file_cont)


def random_coord(edge, ratio_min=0.2):
    while True:
        data = random.sample(range(0, edge), 2)
        data.sort()
        edge_min = round(edge * ratio_min)
        if data[1] - data[0] >= edge_min:
            break
    return data[0], data[1]

folder = ['val', 'valpos', 'valneg']
for fol_name in folder:
    dir_rp = data_rp + fol_name
    if not os.path.exists(dir_rp):
        os.makedirs(dir_rp)

img_set = os.listdir(img_rp)
img_set.sort()

for img_name in img_set:
    img_name_base = img_name[:-6]

    img_obj_ord = img_name[-6:-4]
    order_next = int(img_name_base[-3:]) + 1
    order_next = str(order_next)
    order_next = order_next.zfill(3)
    img_name_next = img_name_base[:-3] + order_next + img_obj_ord + '.jpg'
    img_file_next = img_rp + img_name_next
    # print(img_file_next)
    if not os.path.exists(img_file_next):
        # print('image %s has no subsequence, skip' % img_name)
        continue
    # copy image
    copy_img_cont(img_rp + img_name, data_rp + 'val/' + img_name)
    # positive
    copy_img_cont(img_file_next, data_rp + 'valpos/' + img_name)
    # negative
    imgs_next_prefix = img_name_base[:-3] + order_next
    prefixed_files = [filename for filename in os.listdir(img_rp) if filename.startswith(imgs_next_prefix)]
    if len(prefixed_files) > 1:
        while True:
            img_name_neg = random.sample(prefixed_files, 1)
            img_name_neg = img_name_neg[0]
            img_neg_ord = img_name_neg[-6:-4]
            if img_obj_ord != img_neg_ord:
                break
        copy_img_cont(img_rp + img_name_neg, data_rp + 'valneg/' + img_name)
    else:
        '''
        if only one object occurs in the image, 
        randomly crop a patch and resize it to 224x224, serving as negative exemplars
        Possible, the cropped patch is similar with the object
        '''
        # only contain single object, random crop
        # print('single object')
        col_min, col_max = random_coord(224, 0.2)
        row_min, row_max = random_coord(224, 0.2)
        # crop image
        img = Image.open(img_rp + img_name)
        img_np = np.asarray(img, dtype=np.uint8)
        img_crop_np = img_np[col_min:col_max, row_min:row_max, :]
        img_crop = Image.fromarray(img_crop_np, mode='RGB')
        img_crop = img_crop.resize((224, 224))
        img_crop.save(data_rp + 'valneg/' + img_name)
        # crop mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_rp = img_rp.replace('/img/', '/cont/')
        mask = Image.open(mask_rp + mask_name)
        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_crop_np = mask_np[col_min:col_max, row_min:row_max]
        mask_crop = Image.fromarray(mask_crop_np, mode='L')
        mask_crop = mask_crop.resize((224, 224))
        mask_crop.save(data_rp + 'valneg/' + mask_name)

