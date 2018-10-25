import os
from PIL import Image
import numpy as np
from multiprocessing import Pool

# fol_sys_mask_rp = '/disk2/zhangni/davis/saliency/val_syn_formask/'
# ori_img_rp = '/disk2/zhangni/davis/saliency/val/'
# ori_mask_rp = '/disk2/zhangni/davis/saliency/valannot/'
# res_root_rp = '/disk2/zhangni/davis/cat_128/ValPatch/'
fol_sys_mask_rp = '/disk5/yangle/DAVIS/dataset/saliency/val_syn_formask/'
ori_img_rp = '/disk5/yangle/DAVIS/dataset/saliency/val/'
ori_mask_rp = '/disk5/yangle/DAVIS/dataset/saliency/valannot/'
res_root_rp = '/disk5/yangle/DAVIS/dataset/cat_128/cat_128/ValPatch/'

PADDING = 100
MIN_EDGE = 10


def scale_box(col_min, col_max, factor=0.4):
    hei_half = round((col_max - col_min + 1) * 0.5 * factor)
    hei_top = col_min - hei_half
    hei_bot = col_max + hei_half

    return int(hei_top), int(hei_bot)


def prepare_data(fol_name):
    print(fol_name)
    res_rp = res_root_rp + fol_name + '/'
    if not os.path.exists(res_rp):
        os.makedirs(res_rp)

    img_name = fol_name + '.png'
    # context image
    img_cont = Image.open(ori_img_rp + img_name)
    img_cont = img_cont.resize((224, 224))
    img_cont.save(res_rp + img_name)
    # # gt
    gt = Image.open(ori_mask_rp + img_name)
    gt_save = gt.resize((224, 224))
    gt_save = gt_save.point(lambda i: i / 255)
    gt_name = fol_name + '_gt.png'
    gt_save.save(res_rp + gt_name)

    # coarse mask
    # former mask, crop, resize to 28x28, resize to 224x224
    res_comask = res_rp + 'comask' + '/'
    if not os.path.exists(res_comask):
        os.makedirs(res_comask)
    # # systhetic former mask
    res_fomask = res_rp + 'fomask' + '/'
    if not os.path.exists(res_fomask):
        os.makedirs(res_fomask)
    # cropped image
    res_image = res_rp + 'image' + '/'
    if not os.path.exists(res_image):
        os.makedirs(res_image)
    # cropped image
    res_gt = res_rp + 'cropgt' + '/'
    if not os.path.exists(res_gt):
        os.makedirs(res_gt)

    syn_mask_rp = fol_sys_mask_rp + fol_name + '/'
    mask_set = os.listdir(syn_mask_rp)
    mask_set.sort()

    for mask_name in mask_set:
        mask = Image.open(syn_mask_rp + mask_name)
        # coarse mask
        mask_small = mask.resize((28, 28))
        mask_coarse = mask_small.resize((224, 224))
        mask_coarse.save(res_comask + mask_name)

        # coordinate: enlarge the object box with 0.4
        mask_np = np.asarray(mask, dtype='uint8')
        col_num, row_num = mask_np.shape
        cols, rows = np.where(mask_np == 255)
        col_min = np.min(cols)
        col_max = np.max(cols)
        row_min = np.min(rows)
        row_max = np.max(rows)
        if col_max - col_min < MIN_EDGE or row_max - row_min < MIN_EDGE:
            continue
        hei_top, hei_bot = scale_box(col_min, col_max, factor=0.4)
        wid_lef, wid_rig = scale_box(row_min, row_max, factor=0.4)
        hei_top += PADDING
        if hei_top < 0:
            hei_top = 0
        hei_bot += PADDING
        if hei_bot > col_num + PADDING * 2:
            hei_bot = col_num + PADDING * 2 - 1
        wid_lef += PADDING
        if wid_lef < 0:
            wid_lef = 0
        wid_rig += PADDING
        if wid_rig > row_num + PADDING * 2:
            wid_rig = row_num + PADDING * 2 - 1

        # crop gt
        gt_np = np.asarray(gt, dtype='uint8')
        gt_pad = np.zeros((col_num + PADDING * 2, row_num + PADDING * 2), dtype=np.uint8)
        gt_pad[PADDING:col_num + PADDING, PADDING:row_num + PADDING] = gt_np
        gt_crop_np = gt_pad[hei_top:hei_bot, wid_lef:wid_rig]
        gt_crop = Image.fromarray(gt_crop_np, mode='L')
        gt_crop = gt_crop.resize((224, 224))
        gt_crop.save(res_gt + mask_name)

        # crop image mask, resize to 224x224 and save
        mask_pad = np.zeros((col_num + PADDING * 2, row_num + PADDING * 2), dtype=np.uint8)
        mask_pad[PADDING:col_num + PADDING, PADDING:row_num + PADDING] = mask_np
        mask_crop_np = mask_pad[hei_top:hei_bot, wid_lef:wid_rig]
        mask_crop = Image.fromarray(mask_crop_np, mode='L')
        mask_crop = mask_crop.resize((224, 224))
        mask_crop.save(res_fomask + mask_name)

        # crop image
        image = Image.open(ori_img_rp + img_name)
        img_np = np.asarray(image, dtype='uint8')
        col_num, row_num, _ = img_np.shape
        img_pad = np.zeros((col_num + PADDING * 2, row_num + PADDING * 2, 3), dtype=np.uint8)
        img_pad[PADDING:col_num + PADDING, PADDING:row_num + PADDING, :] = img_np
        img_crop_np = img_pad[hei_top:hei_bot, wid_lef:wid_rig, :]
        image_crop = Image.fromarray(img_crop_np, mode='RGB')
        image_crop = image_crop.resize((224, 224))
        image_crop.save(res_image + mask_name)


if __name__ == '__main__':
    num_processor = 60
    fol_set = os.listdir(fol_sys_mask_rp)
    fol_set.sort()
    for fol_name in fol_set:
        prepare_data(fol_name)
    # pool = Pool(num_processor)
    # run = pool.map(prepare_data, fol_set)
    # pool.close()
    # pool.join()
