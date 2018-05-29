import os

val_fol_rp = '/disk5/yangle/DAVIS/dataset/Objectness/train_synthe_mask/'
val_fol_set = os.listdir(val_fol_rp)
val_fol_set.sort()

for val_fol_name in val_fol_set:
    img_fol_rp = val_fol_rp + val_fol_name + '/'
    img_fol_set = os.listdir(img_fol_rp)
    img_fol_set.sort()

    for fol_name in img_fol_set:
        img_rp = img_fol_rp + fol_name + '/'
        img_set = os.listdir(img_rp)
        if len(img_set) == 0:
            print('delete empty folder %s', img_rp)
            os.rmdir(img_rp)

