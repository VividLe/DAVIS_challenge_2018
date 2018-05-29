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
    sep_cate_rp = '/disk5/yangle/DAVIS/result/mask/try/all-cate-e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/'
    res_rp = '/disk5/yangle/DAVIS/result/mask/try/sel/'

    if not os.path.exists(res_rp):
        os.makedirs(res_rp)

    # load model
    model = UNet_deeplab(in_channels=4, feature_length=512)
    model = model.cuda()
    weights_fpath = '/disk5/yangle/DAVIS/result/TrainNet/ranking-0518/weights/ranking-test-weights-10-0.071-0.000-0.022-0.000.pth'
    state = torch.load(weights_fpath)
    model.load_state_dict(state['state_dict'])
    model.eval()

    cate_set = os.listdir(sep_cate_rp)
    cate_set.sort()
    for cate_name in cate_set[3:4]:
        # compare with the first annotated image
        img_ref = Image.open(cate_fol_rp + cate_name + '/00000.jpg')
        mask_begin = Image.open(cate_gt_rp + cate_name + '/00000.png')
        # only dispose test data
        gt_set = os.listdir(cate_gt_rp + cate_name + '/')
        if len(gt_set) > 1:
            continue
        _, object_num = mask_begin.getextrema()
        for iobj in range(1, object_num+1):
            # only consider the i th object
            mask_ref = mask_begin.point(lambda i: 255 if i == iobj else 0)
            feature_ref = net_prediction(img_ref, mask_ref, model)

            img_fol_rp = sep_cate_rp + cate_name + '/'
            img_fol_set = os.listdir(img_fol_rp)
            img_fol_set.sort()
            # error occurs here #
            for img_ord, img_fol_name in enumerate(img_fol_set):
                img = Image.open(cate_fol_rp + cate_name + '/' + img_fol_name + '.jpg')
                mask_rp = img_fol_rp + img_fol_name + '/'
                mask_set = os.listdir(mask_rp)
                mask_set.sort()
                distance_min = 100
                for mask_name in mask_set:
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
                        name_sel = mask_name
                        feature_sel = feature_c
                mask_save_file = res_rp + cate_name + img_fol_name + name_sel[:-4] + str(iobj) + '.png'
                mask_sel.save(mask_save_file)

                feature_ref = feature_sel
                # update feature
                order = img_ord + 2.0
                feature_ref = (order-1)/order*feature_ref + 1/order * feature_sel

if __name__=='__main__':
    main()
