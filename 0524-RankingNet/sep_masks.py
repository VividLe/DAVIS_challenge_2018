import os
import shutil

res_cate_rp = '/disk5/yangle/DAVIS/result/fuse/'
masks_rp = '/disk5/yangle/DAVIS/result/mask/DAVIS-maskrcnn/MaskRCNN_TF/big_merge/'

prefix = '09_'

mask_set = os.listdir(masks_rp)
mask_set.sort()

for mask_name in mask_set:
    cate_name = mask_name[:-12]
    img_fol_name = mask_name[-12:-7]
    obj_str = mask_name[-5]
    if obj_str != '0':
        obj_num_str = '0' + obj_str
    else:
        obj_num_str = '10'
    fuse_img_rp = res_cate_rp + cate_name + '/' + img_fol_name + '/'
    if not os.path.exists(fuse_img_rp):
        os.makedirs(fuse_img_rp)
    save_file = fuse_img_rp + prefix + obj_num_str + '.png'
    print(save_file)
    shutil.move(masks_rp + mask_name, save_file)


# mask_set = os.listdir(masks_rp)
# mask_set.sort()
#
# cate_set = os.listdir(res_cate_rp)
# cate_set.sort()
# for cate_name in cate_set:
#     print(cate_name)
#     mask_fol_rp = res_cate_rp + cate_name + '/'
#     mask_fol_set = os.listdir(mask_fol_rp)
#     mask_fol_set.sort()
#
#     for mask_fol_name in mask_fol_set:
#         fuse_img_rp = mask_fol_rp + mask_fol_name + '/'
#
#         mask_name_fore = cate_name + mask_fol_name
#         # search for the target mask
#         prefixed_files = [filename for filename in os.listdir(masks_rp) if filename.startswith(mask_name_fore)]
#         for file_name in prefixed_files:
#             obj_str = file_name[-5]
#             if obj_str != '0':
#                 obj_num_str = '0' + obj_str
#             else:
#                 obj_num_str = '10'
#             save_file = fuse_img_rp + prefix + obj_num_str + '.png'
#             shutil.move(masks_rp+file_name, save_file)
