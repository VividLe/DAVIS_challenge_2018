import os
import shutil

val_cate_list = ['libby', 'loading', 'mbike-trick',
                 'motocross-jump', 'paragliding-launch',
                 'parkour', 'pigs', 'scooter-black',
                 'shooting', 'soapbox']

mix_img_rp = '/disk5/yangle/DAVIS/dataset/Ranking/img_cont/img/'
mix_cont_rp = '/disk5/yangle/DAVIS/dataset/Ranking/img_cont/cont/'
val_img_rp = '/disk5/yangle/DAVIS/dataset/Ranking/Val/img_cont/img/'
val_cont_rp = '/disk5/yangle/DAVIS/dataset/Ranking/Val/img_cont/cont/'

val_data_set = []
for val_cate in val_cate_list:
    prefixed_files = [filename for filename in os.listdir(mix_img_rp) if filename.startswith(val_cate)]
    print(len(prefixed_files))
    val_data_set += prefixed_files
print(len(val_data_set))

for img_name in val_data_set:
    shutil.move(mix_img_rp+img_name, val_img_rp+img_name)
    mask_name = img_name.replace('.jpg', '.png')
    shutil.move(mix_cont_rp+mask_name, val_cont_rp+mask_name)


