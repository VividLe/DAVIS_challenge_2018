from PIL import Image
import numpy as np


def scale_box(col_min, col_max, cols, factor=0.4):
    hei_half = round((col_max - col_min + 1) * 0.5 * factor * 0.5)
    if (col_min - hei_half >= 1) and (col_max + hei_half <= cols):
        hei_top = col_min - hei_half
        hei_bot = col_max + hei_half
    elif (col_min - hei_half < 1) and (col_max + hei_half <= cols):
        hei_top = 1
        hei_bot = 224
    elif (col_min - hei_half >= 1) and (col_max + hei_half > cols):
        hei_bot = cols
        hei_top = cols - 223
    else:
        hei_top = 1
        hei_bot = cols
    return int(hei_top), int(hei_bot)

mask = Image.open('DUTS00170_1_0.png')
image = Image.open('DUTS00170.png')
mask_np = np.asarray(mask, dtype='uint8')
col_num, row_num = mask_np.shape
cols, rows = np.where(mask_np == 255)
col_min = np.min(cols)
col_max = np.max(cols)
row_min = np.min(rows)
row_max = np.max(rows)

hei_top, hei_bot = scale_box(col_min, col_max, col_num, factor=0.4)
wid_lef, wid_rig = scale_box(row_min, row_max, row_num, factor=0.4)

mask_crop_np = mask_np[hei_top:hei_bot, wid_lef:wid_rig]
mask_crop = Image.fromarray(mask_crop_np, mode='L')
mask_crop = mask_crop.resize((224, 224))
mask_crop.save('mask_crop.png')

image_np = np.asarray(image, dtype='uint8')
image_crop_np = image_np[hei_top:hei_bot, wid_lef:wid_rig, :]
image_crop = Image.fromarray(image_crop_np, mode='RGB')
image_crop = image_crop.resize((224, 224))
image_crop.save('image_crop.png')

