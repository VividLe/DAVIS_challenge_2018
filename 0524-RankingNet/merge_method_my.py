import os
import shutil

methods_rp = '/disk5/yangle/DAVIS/result/mask/step1_select/'
res_met_rp = '/disk5/yangle/DAVIS/result/mask/step2_rearrange/'

met_set = os.listdir(methods_rp)
met_set.sort()

for iord, met_name in enumerate(met_set):
    iord_str = str(iord+1)
    iord_str = iord_str.zfill(2)
    masks_rp = methods_rp + met_name + '/'
    mask_set = os.listdir(masks_rp)
    mask_set.sort()

    for mask_name in mask_set:
        print(mask_name)
        cate_name = mask_name[:-12]
        img_fol_name = mask_name[-12:-7]
        obj_str = mask_name[-6:-4]
        fuse_img_rp = res_met_rp + cate_name + '/' + img_fol_name + '/'
        if not os.path.exists(fuse_img_rp):
            os.makedirs(fuse_img_rp)
        save_file = fuse_img_rp + iord_str + '_' + obj_str + '.png'
        # shutil.move(masks_rp + mask_name, save_file)
        # forbidden use move
        shutil.copyfile(masks_rp + mask_name, save_file)


