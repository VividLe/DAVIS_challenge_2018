import os
import numpy as np
import skimage.io
from PIL import Image

import coco
import model as modellib

import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

ori_img_rp = '/home/yangle/DAVIS/code/mask-rcnn/images/'
res_img_rp = './'

img_set = os.listdir(ori_img_rp)
img_set.sort()

for img_name in img_set:
    print(img_name)
    image = skimage.io.imread(ori_img_rp + img_name)

    res_file = res_img_rp + img_name
    # if os.path.exists(res_file):
    #     continue

    try:
        # # Run detection
        results = model.detect([image])

        prediction = results[0]
        mask_set = prediction['masks']
        # class_ids = prediction['class_ids']

        hei_m, wid_m, num_m = mask_set.shape

        mask_pred = np.zeros((hei_m, wid_m), dtype=np.uint8)
        for inum in range(num_m):
            order = inum + 1
            mask_cur = mask_set[:, :, inum]
            mask_pred = np.where(mask_cur == 1, order, mask_pred)

        mask_save = Image.fromarray(mask_pred)
        mask_save.save(res_img_rp + img_name)
    except RuntimeError:
        print('Error occurs')
    print('continuing')
