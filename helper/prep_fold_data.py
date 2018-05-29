import os
import shutil
from PIL import Image
import numpy as np

fol_sys_mask_rp = '/disk5/yangle/DAVIS/dataset/saliency/train_syn_formask/'
ori_img_rp = '/disk5/yangle/DAVIS/dataset/saliency/train/'
ori_mask_rp = '/disk5/yangle/DAVIS/dataset/saliency/trainannot/'
res_root_rp = '/disk5/yangle/DAVIS/dataset/cat_128/TrainPatch/'


def scale_box(col_min, col_max, cols, factor=0.4):
    hei_half = round((col_max - col_min + 1) * 0.5 * factor * 0.5)
    if (col_min - hei_half >= 1) and (col_max + hei_half <= cols):
        hei_top = col_min - hei_half
        hei_bot = col_max + hei_half
    elif (col_min - hei_half < 1) and (col_max + hei_half <= cols):
        hei_top = 1
        hei_bot = 224
    elif (col_min - hei_half >= 1) and (col_max + hei_half > cols):
        hei_bot = cols
        hei_top = cols - 223
    else:
        hei_top = 1
        hei_bot = cols
    return int(hei_top), int(hei_bot)


fol_set = os.listdir(fol_sys_mask_rp)
fol_set.sort()

for fol_name in fol_set:
    print(fol_name)
    res_rp = res_root_rp + fol_name + '/'
    if not os.path.exists(res_rp):
        os.makedirs(res_rp)
    
    img_name = fol_name+'.png'
    # context image
    img_cont = Image.open(ori_img_rp+img_name)
    img_cont = img_cont.resize((224, 224))
    img_cont.save(res_rp+img_name)
    # gt
    gt = Image.open(ori_mask_rp+img_name)
    gt = gt.resize((224, 224))
    gt = gt.point(lambda i: i / 255)
    gt_name = fol_name + '_gt.png'
    gt.save(res_rp+gt_name)


    # coarse mask
    # former mask, crop, resize to 28x28, resize to 224x224
    res_comask = res_rp + 'comask' + '/'
    if not os.path.exists(res_comask):
        os.makedirs(res_comask)
    # systhetic former mask
    res_fomask = res_rp + 'fomask' + '/'
    if not os.path.exists(res_fomask):
        os.makedirs(res_fomask)
    # cropped image
    res_image = res_rp + 'image' + '/'
    if not os.path.exists(res_image):
        os.makedirs(res_image)

    syn_mask_rp = fol_sys_mask_rp + fol_name + '/'
    mask_set = os.listdir(syn_mask_rp)
    mask_set.sort()

    for mask_name in mask_set:
        mask = Image.open(syn_mask_rp+mask_name)
        # coarse mask
        mask_small = mask.resize((28, 28))
        mask_coarse = mask_small.resize((224, 224))
        mask_coarse.save(res_comask+mask_name)

        # coordinate: enlarge the object box with 0.4
        mask_np = np.asarray(mask, dtype='uint8')
        col_num, row_num = mask_np.shape
        cols, rows = np.where(mask_np == 255)
        col_min = np.min(cols)
        col_max = np.max(cols)
        row_min = np.min(rows)
        row_max = np.max(rows)
        hei_top, hei_bot = scale_box(col_min, col_max, col_num, factor=0.4)
        wid_lef, wid_rig = scale_box(row_min, row_max, row_num, factor=0.4)

        # crop image mask, resize to 224x224 and save
        mask_crop_np = mask_np[hei_top:hei_bot, wid_lef:wid_rig]
        mask_crop = Image.fromarray(mask_crop_np, mode='L')
        mask_crop = mask_crop.resize((224, 224))
        mask_crop.save(res_fomask+mask_name)

        image = Image.open(ori_img_rp + img_name)
        image_np = np.asarray(image, dtype='uint8')
        image_crop_np = image_np[hei_top:hei_bot, wid_lef:wid_rig, :]
        image_crop = Image.fromarray(image_crop_np, mode='RGB')
        image_crop = image_crop.resize((224, 224))
        image_crop.save(res_image+mask_name)
