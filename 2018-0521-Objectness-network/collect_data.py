import os
from PIL import Image
import random
import numpy as np

val_img_rp = '/disk5/yangle/DAVIS/dataset/Objectness/val/'
syn_rp = '/disk5/yangle/DAVIS/dataset/Objectness/val_synthe_mask/'
res_rp = '/disk5/yangle/DAVIS/dataset/Objectness/valcont/'
sys_set = os.listdir(syn_rp)
sys_set.sort()

def file_copy(fol_name):
# for fol_name in sys_set:
    img_name = fol_name + '.png'
    val_img = Image.open(val_img_rp + img_name)
    val_img_np = np.asarray(val_img, dtype=np.uint8)
    rows, cols, _ = val_img_np.shape
    cont_np = np.zeros((rows, cols),dtype=np.uint8)

    img_fol_rp = syn_rp + fol_name + '/'
    img_fol_set = os.listdir(img_fol_rp)
    img_fol_set.sort()

    for iord, img_fol_name in enumerate(img_fol_set):
        img_rp = img_fol_rp + img_fol_name + '/'
        img_set = os.listdir(img_rp)

        img_sel = random.sample(img_set, 1)
        img_name_sel = img_sel[0]
        seg = Image.open(img_rp + img_name_sel)
        seg_np = np.asarray(seg, dtype=np.uint8)
        cont_np = np.where(seg_np==255, iord+1, cont_np)
    cont_img = Image.fromarray(cont_np, mode='L')
    cont_img.save(res_rp+img_name)



