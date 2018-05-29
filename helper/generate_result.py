import os
import numpy as np
from PIL import Image

mask_rp = '/disk5/yangle/DAVIS/result/fuse_sel/'
cate_rp = '/disk5/yangle/DAVIS/result/test_cha/'
ref_cate_img_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/ZIP_files/DAVIS-cha/JPEGImages/480p/'
ref_cate_gt_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/ZIP_files/DAVIS-cha/Annotations/480p/'


cate_set = os.listdir(ref_cate_img_rp)
cate_set.sort()

for cate_name in cate_set:
    print(cate_name)
    ref_gt = Image.open(ref_cate_gt_rp + cate_name + '/00000.png')
    _, obj_num = ref_gt.getextrema()
    cols, rows = ref_gt.size

    res_mask_rp = cate_rp + cate_name + '/'
    if not os.path.exists(res_mask_rp):
        os.makedirs(res_mask_rp)

    ref_img_rp = ref_cate_img_rp + cate_name + '/'
    img_set = os.listdir(ref_img_rp)
    img_set.sort()
    for img_name in img_set:
        pred = np.zeros((rows, cols), dtype=np.uint8)
        for iobj in range(1, obj_num+1):
            mask_name = cate_name + img_name[:-4] + '_' + str(iobj)
            mask_file = mask_rp + mask_name + '.png'
            if os.path.exists(mask_file):
                mask = Image.open(mask_file)
                mask_np = np.asarray(mask, dtype=np.uint8)
                # fill the pred
                pred = np.where(mask_np == 255, iobj, pred)
            else:
                print('file not exists: %s' % mask_file)
        pred_mask = Image.fromarray(pred, mode='L')
        save_file = res_mask_rp + img_name[:-4] + '.png'
        pred_mask.save(save_file)




# mask_set = os.listdir(mask_rp)
# mask_set.sort()
# for mask_name in mask_set:
#     cate_name = mask_name[:-11]
#     img_name = mask_name[-11:-6]
#     obj_ord = int(mask_name[-5])


