import os
import shutil

ref_mask_rp = '/disk5/yangle/DAVIS/dataset/Objectness/train/'
ori_fol_rp = '/disk5/yangle/DAVIS/dataset/Objectness/synthe_mask/'
res_fol_rp = '/disk5/yangle/DAVIS/dataset/Objectness/train_synthe_mask/'

mask_set = os.listdir(ref_mask_rp)
mask_set.sort()

for mask_name in mask_set:
    fol_name = mask_name[:-4]
    sys_rp = ori_fol_rp + fol_name + '_1/'
    if os.path.exists(sys_rp):
        sys_set = os.listdir(sys_rp)
        if len(sys_set) > 0:
            res_rp = res_fol_rp + fol_name + '/'
            if not os.path.exists(res_rp):
                os.makedirs(res_rp)
            for iord in range(1, 6):
                sys_rp_c = ori_fol_rp + fol_name + '_' + str(iord) + '/'
                if not os.path.exists(sys_rp_c):
                    break
                else:
                    shutil.move(sys_rp_c, res_rp)
        else:
            mask_file = ref_mask_rp + mask_name
            print('delete file %s', mask_file)
            os.remove(mask_file)
    else:
        mask_file = ref_mask_rp+mask_name
        print('delete file %s', mask_file)
        os.remove(mask_file)

