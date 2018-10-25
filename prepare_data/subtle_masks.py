import os
from PIL import Image
import numpy as np
from multiprocessing import Pool


DISCARD_TH = 500

base_mask_rp = '/disk5/yangle/DAVIS/result/mask/maskrcnn_Caffe2/SegMask/'
base_cate_res_rp = '/disk5/yangle/DAVIS/result/mask/maskrcnn_Caffe2/SegMaskSep/'


def save_obj_mask(mask_name):
# for mask_name in mask_set:
    cate_name = mask_name[:-9]
    img_base_name = mask_name[-9:-4]
    img_res_rp = cate_res_rp + cate_name + '/' + img_base_name + '/'
    if not os.path.exists(img_res_rp):
        os.makedirs(img_res_rp)

    mask_ori = Image.open(mask_rp+mask_name)
    mask_ori_np = np.asarray(mask_ori, dtype=np.uint8)
    [rows, cols] = mask_ori_np.shape
    obj_num = mask_ori_np.max()
    for iobj in range(1, obj_num+1):
        mask_obj_np = np.zeros((rows, cols), dtype=np.uint8)
        num_c, _ = np.where(mask_ori_np == iobj)
        # discard subtle object
        if len(num_c) < DISCARD_TH:
            continue
        mask_obj_np = np.where(mask_ori_np == iobj, 255, mask_obj_np)
        order = str(iobj)
        order = order.zfill(2)
        save_file = img_res_rp + order + '.png'
        mask_obj = Image.fromarray(mask_obj_np, mode='L')
        mask_obj.save(save_file)


if __name__ == '__main__':
    meth_set = os.listdir(base_mask_rp)
    meth_set.sort()
    for meth_name in meth_set:
        print(meth_name)
        global mask_rp
        mask_rp = base_mask_rp + meth_name + '/'
        global cate_res_rp
        cate_res_rp = base_cate_res_rp + meth_name + '/'
        if not os.path.exists(cate_res_rp):
            os.makedirs(cate_res_rp)
        num_processor = 120
        fol_set = os.listdir(mask_rp)
        fol_set.sort()
        pool = Pool(num_processor)
        run = pool.map(save_obj_mask, fol_set)
        pool.close()
        pool.join()

