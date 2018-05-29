import os
from PIL import Image
import shutil
from multiprocessing import Pool
import numpy as np

ori_mask_rp = '/disk5/yangle/DAVIS/result/mask/step1_select/'
sep_mask_rp = '/disk5/yangle/DAVIS/result/RankNet_googleM/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/'
res_mask_rp = '/disk5/yangle/DAVIS/result/RankNet_googleF/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/'
ref_cate_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/MergeFiles/DAVIS/Annotations/480p/'


def sepa_masks(mask_name):
    if mask_name[-5] != '0':
        if mask_name[-12] == '0':
            cate_name = mask_name[:-12]
            img_name = mask_name[-12:-7]
        else:
            cate_name = mask_name[:-11]
            img_name = mask_name[-11:-6]
    else:
        if mask_name[-13] == '0':
            cate_name = mask_name[:-13]
            img_name = mask_name[-13:-8]
        else:
            cate_name = mask_name[:-12]
            img_name = mask_name[-12:-7]
    cate_rp = sep_mask_rp + cate_name + '/'

    img_rp = cate_rp + img_name + '/'
    if not os.path.exists(img_rp):
        os.makedirs(img_rp)
    shutil.copyfile(ori_mask_rp+mask_name, img_rp+mask_name)


def merge_masks(cate_name):
# for cate_name in cate_set:
    ann = Image.open(ref_cate_rp + cate_name + '/00000.png')
    ann_np = np.asarray(ann, dtype=np.uint8)
    rows, cols = ann_np.shape
    save_rp = res_mask_rp + cate_name + '/'
    if not os.path.exists(save_rp):
        os.makedirs(save_rp)
    img_fol_rp = sep_mask_rp + cate_name + '/'
    img_fol_set = os.listdir(img_fol_rp)
    for img_pre_name in img_fol_set:
        seg_rp = img_fol_rp + img_pre_name + '/'
        seg_set = os.listdir(seg_rp)
        mask_merg = np.zeros((rows, cols), dtype=np.uint8)
        for seg_name in seg_set:
            obj_ord = int(seg_name[-5])
            if obj_ord == 0:
                obj_ord = 10
            seg = Image.open(seg_rp + seg_name)
            if seg.size != ann.size:
                seg = seg.resize((ann.size), Image.NEAREST)
            seg_np = np.asarray(seg, dtype=np.uint8)
            mask_merg = np.where(seg_np == 255, obj_ord, mask_merg)
        mask_img = Image.fromarray(mask_merg, mode='L')
        save_file = save_rp + img_pre_name + '.png'
        mask_img.save(save_file)


def step1():
    num_processor = 60
    mask_set = os.listdir(ori_mask_rp)
    mask_set.sort()
    # for mask_name in mask_set:
    #     sepa_masks(mask_name)
    pool = Pool(num_processor)
    run = pool.map(sepa_masks, mask_set)
    pool.close()
    pool.join()


def step2():
    num_processor = 60
    cate_set = os.listdir(sep_mask_rp)
    cate_set.sort()
    # for cate_name in cate_set:
    #     merge_masks(cate_name)
    pool = Pool(num_processor)
    run = pool.map(merge_masks, cate_set)
    pool.close()
    pool.join()

if __name__ == '__main__':
    step1()
    step2()
    # num_processor = 60
    # mask_set = os.listdir(ori_mask_rp)
    # mask_set.sort()
    # pool = Pool(num_processor)
    # run = pool.map(sepa_masks, mask_set)
    # pool.close()
    # pool.join()
    # cate_set = os.listdir(sep_mask_rp)
    # cate_set.sort()
    # pool = Pool(num_processor)
    # run = pool.map(merge_masks, cate_set)
    # pool.close()
    # pool.join()

