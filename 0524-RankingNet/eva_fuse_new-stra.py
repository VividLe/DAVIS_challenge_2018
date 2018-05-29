# coding: utf-8

import argparse
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import shutil

import dataset_8ch as saliency
from unet_deeplab import UNet_deeplab
import experiment
import utils as utils
from parameter import *


def net_prediction(img, mask, model):
    if img.size != mask.size:
        img = img.resize(mask.size)
    utils.prepare_test_data(img, mask)
    normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
    transform = transforms.Compose([transforms.ToTensor(), normalize, ])
    img_ref, mask_ref = utils.test_data_loader(transform)
    inputs = torch.cat((img_ref, mask_ref), 0)
    inputs = torch.unsqueeze(inputs, dim=0)
    inputs = Variable(inputs.cuda(), volatile=True)
    feature = model(inputs)
    # remove dimision 2,3
    feature.squeeze_(dim=2)
    feature.squeeze_(dim=2)
    return feature


def main():
    cate_fol_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/MergeFiles/DAVIS/JPEGImages/480p/'
    cate_gt_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/MergeFiles/DAVIS/Annotations/480p/'
    sep_cate_rp = '/disk5/yangle/DAVIS/result/mask/step2_rearrange/'
    res_rp = '/disk5/yangle/DAVIS/result/mask/step3_objectmask/'

    if not os.path.exists(res_rp):
        os.makedirs(res_rp)

    # load model
    model = UNet_deeplab(in_channels=4, feature_length=512)
    model = model.cuda()
    weights_fpath = 'ranking-weights-62-ZD.pth'
    state = torch.load(weights_fpath)
    model.load_state_dict(state['state_dict'])
    model.eval()

    cate_set = os.listdir(sep_cate_rp)
    cate_set.sort()
    for cate_name in cate_set[54:]:
        # only dispose test data
        gt_rp = cate_gt_rp + cate_name + '/'
        if not os.path.exists(gt_rp):
            continue
        # compare with the first annotated image
        img_ref = Image.open(cate_fol_rp + cate_name + '/00000.jpg')
        mask_begin = Image.open(cate_gt_rp + cate_name + '/00000.png')

        _, object_num = mask_begin.getextrema()
        for iobj in range(1, object_num+1):
            obj_str = str(iobj)
            obj_str = obj_str.zfill(2)
            post_fix = '_' + obj_str + '.png'
            # only consider the i th object
            mask_ref = mask_begin.point(lambda i: 255 if i == iobj else 0)
            feature_ref = net_prediction(img_ref, mask_ref, model)

            img_fol_rp = sep_cate_rp + cate_name + '/'
            img_fol_set = os.listdir(img_fol_rp)
            img_fol_set.sort()

            for img_ord, img_fol_name in enumerate(img_fol_set):
                img = Image.open(cate_fol_rp + cate_name + '/' + img_fol_name + '.jpg')
                mask_rp = img_fol_rp + img_fol_name + '/'
                # select mask_set, only dispose current object
                mask_set = os.listdir(mask_rp)
                mask_set_s = [f for f in mask_set if f.endswith(post_fix)]
                # this object contains no masks
                if len(mask_set_s) == 0:
                    continue
                mask_set_s.sort()
                distance_min = 100
                for mask_name in mask_set_s:
                    print(mask_rp + mask_name)
                    mask = Image.open(mask_rp + mask_name)
                    # skip too tiny object
                    if not utils.HasObject(mask):
                        continue
                    feature_c = net_prediction(img, mask, model)
                    distance = torch.dist(feature_c, feature_ref, p=2)
                    distance = distance.data[0]
                    print('distance between two vectors is %f' % distance)
                    if distance < distance_min:
                        distance_min = distance
                        mask_sel = mask
                        feature_sel = feature_c
                # do not select the object
                if distance_min == 100:
                    continue
                save_rp = res_rp + cate_name + '/' + img_fol_name + '/'
                if not os.path.exists(save_rp):
                    os.makedirs(save_rp)
                mask_save_file = save_rp + obj_str + '.png'
                mask_sel.save(mask_save_file)
                # # do not update feature
                # order = img_ord + 2.0
                # feature_ref = (order-1)/order*feature_ref + 1/order * feature_sel

if __name__=='__main__':
    main()
