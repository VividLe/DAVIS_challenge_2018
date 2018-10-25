import os
from PIL import Image
import math
from multiprocessing import Pool

ori_img_rp = '/home/zhangdong/database/DUTS/train/'
res_img_rp = '/home/yangle/DAVIS/dataset/data/'
if not os.path.exists(res_img_rp):
    os.makedirs(res_img_rp)


def enlarge_img(img_name):
    print(img_name)
    ori_img = Image.open(ori_img_rp + img_name)
    res_img = ori_img.resize((224, 224))
    res_img.save(res_img_rp + img_name)

if __name__ == '__main__':
    num_proc = 30
    pool = Pool(num_proc)
    img_set = os.listdir(ori_img_rp)
    run = pool.map(enlarge_img, img_set)
    pool.close()
    pool.join()

# img_set = os.listdir(ori_img_rp)
# for img_name in img_set[1:3]:
#     ori_img = Image.open(ori_img_rp + img_name)
#     ho, wo = ori_img.size
#     factor = math.sqrt(all_num / (wo * ho))
#     hr, wr = int(round(ho * factor)), int(round(wo * factor))
#     res_img = ori_img.resize((hr, wr))
#     res_img.save(res_img_rp + img_name)
#     ori_mask = Image.open(ori_mask_rp + img_name)
#     res_mask = ori_mask.resize((hr, wr))
#     res_mask.save(res_mask_rp + img_name)
