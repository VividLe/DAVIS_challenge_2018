import os
import shutil
from PIL import Image
import numpy as np
from multiprocessing import Pool

import utils

mask_rp = '/disk5/yangle/DAVIS/dataset/Ranking/sep_object/'
image_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/mImages/'
cont_rp = '/disk5/yangle/DAVIS/dataset/Ranking/img_cont/'

PADDING = 100
MIN_EDGE = 10

mask_set = os.listdir(mask_rp)
mask_set.sort()

# def prep_cont_date(mask_name):
for iord, mask_name in enumerate(mask_set[2877:]):
    print(iord)
    print(mask_name)
    mask = Image.open(mask_rp + mask_name)
    print(mask_rp + mask_name)

    # coordinate: enlarge the object box with 0.4
    mask_np = np.asarray(mask, dtype='uint8')
    # skip frame lacking object
    if mask_np.max() == 0:
        continue

    col_num, row_num = mask_np.shape
    mask_pad = np.zeros((col_num+PADDING*2, row_num+PADDING*2), dtype=np.uint8)
    img_pad = np.zeros((col_num+PADDING*2, row_num+PADDING*2, 3), dtype=np.uint8)

    cols, rows = np.where(mask_np == 255)
    col_min = np.min(cols)
    col_max = np.max(cols)
    row_min = np.min(rows)
    row_max = np.max(rows)
    if col_max-col_min < MIN_EDGE or row_max-row_min < MIN_EDGE:
        continue

    hei_top, hei_bot = utils.scale_box(col_min, col_max, col_num, factor=0.4)
    wid_lef, wid_rig = utils.scale_box(row_min, row_max, row_num, factor=0.4)
    hei_top += PADDING
    hei_bot += PADDING
    wid_lef += PADDING
    if wid_lef < 0:
        wid_lef = 0
    wid_rig += PADDING

    # crop gt save as .png
    mask_pad[PADDING:col_num+PADDING, PADDING:row_num+PADDING] = mask_np
    mask_crop_np = mask_pad[hei_top:hei_bot, wid_lef:wid_rig]
    mask_crop = Image.fromarray(mask_crop_np, mode='L')
    mask_crop = mask_crop.resize((224, 224))
    mask_crop.save(cont_rp + mask_name)

    # crop image save as .jpg
    img_name = mask_name[:-6] + '.png'
    img = Image.open(image_rp + img_name)
    img_np = np.asarray(img, dtype=np.uint8)
    img_pad[PADDING:col_num+PADDING, PADDING:row_num+PADDING, :] = img_np
    img_crop_np = img_pad[hei_top:hei_bot, wid_lef:wid_rig, :]
    img_crop = Image.fromarray(img_crop_np, mode='RGB')
    img_crop = img_crop.resize((224, 224))
    img_name_save = mask_name[:-4] + '.jpg'
    img_crop.save(cont_rp + img_name_save)

# if __name__ == '__main__':
#     num_processor = 60
#     fol_set = os.listdir(mask_rp)
#     fol_set.sort()
#     pool = Pool(num_processor)
#     run = pool.map(prep_cont_date, fol_set)
#     pool.close()
#     pool.join()
