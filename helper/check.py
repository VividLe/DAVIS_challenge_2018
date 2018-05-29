import os
from PIL import Image
import shutil

img_rp = '/disk5/yangle/DAVIS/dataset/sig_obj/testcont/'

img_set = os.listdir(img_rp)
img_set.sort()

for iord, img_name in enumerate(img_set):
    img = Image.open(img_rp+img_name)
    wid, hei = img.size
    if wid == 28:
        name_former = img_set[iord-1]
        shutil.copyfile(img_rp+name_former, img_rp+img_name)
        print('replace %s with %s' % (img_name, name_former))


