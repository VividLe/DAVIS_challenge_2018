import os
from PIL import Image

ori_annot_path = '/home/yangle/DAVIS/dataset/DUTS/trainannot_255/'
res_annot_path = '/home/yangle/DAVIS/dataset/DUTS/trainannot/'

if not os.path.exists(res_annot_path):
    os.makedirs(res_annot_path)

img_set = os.listdir(ori_annot_path)
for img_name in img_set:
    print(img_name)
    img = Image.open(ori_annot_path + img_name)
    img = img.point(lambda i: i/255)
    img.save(res_annot_path + img_name)

# a simple implement
# for img_name in img_set:
#     print(img_name)


