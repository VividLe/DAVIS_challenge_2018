import os
import numpy as np
from PIL import Image

ori_mask_rp = '/disk5/yangle/DAVIS/result/mask/step4_refine/'
cate_rp = '/disk5/yangle/DAVIS/result/mask/test-cha/'
ref_cate_img_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/ZIP_files/DAVIS-cha/JPEGImages/480p/'
ref_cate_gt_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/ZIP_files/DAVIS-cha/Annotations/480p/'

cate_set = os.listdir(ref_cate_img_rp)
cate_set.sort()

for cate_name in cate_set:
    print(cate_name)
    ref_gt = Image.open(ref_cate_gt_rp + cate_name + '/00000.png')
    _, obj_num = ref_gt.getextrema()
    cols, rows = ref_gt.size
    cate_mask_rp = ori_mask_rp + cate_name + '/'

    res_mask_rp = cate_rp + cate_name + '/'
    if not os.path.exists(res_mask_rp):
        os.makedirs(res_mask_rp)

    ref_img_rp = ref_cate_img_rp + cate_name + '/'
    img_set = os.listdir(ref_img_rp)
    img_set.sort()
    for img_name in img_set:
        img_fol_name = img_name[:-4]
        mask_rp = cate_mask_rp + img_fol_name + '/'
        if not os.path.exists(mask_rp):
            print('empty image',mask_rp)
            pred = np.zeros((rows, cols), dtype=np.uint8)
            pred_mask = Image.fromarray(pred, mode='L')
            save_file = res_mask_rp + img_fol_name + '.png'
            pred_mask.save(save_file)
            continue
        # possibly not existing
        mask_set = os.listdir(mask_rp)
        mask_set.sort()
        if len(mask_set) == 1:
            mask = Image.open(mask_rp + mask_set[0])
            mask = mask.point(lambda i: i/255)
            save_file = res_mask_rp + img_fol_name + '.png'
            mask.save(save_file)
        else:
            # contain multiple objects
            pixel_num = np.zeros((len(mask_set)))
            mask_col = []
            for imask, mask_name in enumerate(mask_set):
                mask = Image.open(mask_rp + mask_name)
                if mask.size != ref_gt.size:
                    mask = mask.resize(ref_gt.size)
                mask_np = np.asarray(mask, dtype=np.uint8)
                mask_col.append(mask_np)
                rows_c, _ = np.where(mask_np == 255)
                pixel_num[imask] = len(rows_c)
            # descend order
            index = pixel_num.argsort()[::-1]
            pred = np.zeros((rows, cols), dtype=np.uint8)
            for ind in index:
                ind_mask = index[ind]
                mask_name = mask_set[ind_mask]
                mask_np = mask_col[ind_mask]
                obj_num = int(mask_name[:-4])
                pred = np.where(mask_np == 255, obj_num, pred)

            pred_mask = Image.fromarray(pred, mode='L')
            save_file = res_mask_rp + img_fol_name + '.png'
            pred_mask.save(save_file)


