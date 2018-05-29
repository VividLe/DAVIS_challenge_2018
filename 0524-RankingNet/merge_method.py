import os
import shutil

methods_rp = '/disk5/yangle/DAVIS/result/mask/step1_select/maskrcnn-TF/'
res_cate_rp = '/disk5/yangle/DAVIS/result/mask/step2_rearrange/maskrcnn-TF/'

met_set = os.listdir(methods_rp)
met_set.sort()

for iord, met_name in enumerate(met_set):
    iord_str = str(iord+1)
    iord_str = iord_str.zfill(2)
    cate_rp = methods_rp + met_name + '/'
    cate_set = os.listdir(cate_rp)
    cate_set.sort()

    for cate_name in cate_set:
        print(cate_name)
        each_cate_rp = res_cate_rp + cate_name + '/'

        mask_fol_rp = cate_rp + cate_name + '/'
        mask_fol_set = os.listdir(mask_fol_rp)
        mask_fol_set.sort()

        for mask_fol_name in mask_fol_set:
            fuse_img_rp = each_cate_rp + mask_fol_name + '/'
            if not os.path.exists(fuse_img_rp):
                os.makedirs(fuse_img_rp)
            mask_rp = mask_fol_rp + mask_fol_name + '/'
            mask_set = os.listdir(mask_rp)
            mask_set.sort()

            for mask_name in mask_set:
                obj_ord = mask_name[-5]
                obj_ord = str(obj_ord)
                obj_ord = obj_ord.zfill(2)
                save_file = fuse_img_rp + iord_str + '_' + obj_ord + '.png'
                shutil.copyfile(mask_rp+mask_name, save_file)







