import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import pickle

import dataset_8ch as saliency
from unet_deeplab import UNet_deeplab
import utils as utils


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
    cate_fol_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/JPEGImages/480p/'
    cate_gt_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/Annotations/480p/'
    sep_cate_rp = '/disk5/yangle/DAVIS/result/RankNet/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/'
    res_rp = '/disk5/yangle/DAVIS/result/ZD/RankingNet/e2e_mask_rcnn_X-101-64x4d-FPN_1xdemo/'

    if not os.path.exists(res_rp):
        os.makedirs(res_rp)

    # load model
    model = UNet_deeplab(in_channels=4, feature_length=512)
    model = model.cuda()
    weights_fpath = '/disk5/zhangdong/DAVIS/result/TrainNet2/ranking-test/weights/ranking-test-weights-62-0.135-0.000-0.384-0.000.pth'
    # weights_fpath = '/disk5/yangle/DAVIS/result/TrainNet/ranking-0518/weights/ranking-test-weights-10-0.071-0.000-0.022-0.000.pth'
    state = torch.load(weights_fpath)
    model.load_state_dict(state['state_dict'])
    model.eval()

    distance_set = []
    ratio_set = []
    cate_set = os.listdir(sep_cate_rp)
    cate_set.sort()
    for cate_name in cate_set:
        # only dispose train val data
        gt_set = os.listdir(cate_gt_rp + cate_name + '/')
        if len(gt_set) == 1:
            continue
        img_fol_rp = sep_cate_rp + cate_name + '/'
        img_fol_set = os.listdir(img_fol_rp)
        img_fol_set.sort()

        for img_ord, img_fol_name in enumerate(img_fol_set):
            # sample every ten frames
            if img_ord % 10 != 0:
                continue
            img = Image.open(cate_fol_rp + cate_name + '/' + img_fol_name + '.jpg')
            gt = Image.open(cate_gt_rp + cate_name + '/' + img_fol_name + '.png')
            _, object_num = gt.getextrema()

            for iobj in range(1, object_num+1):
                obj_str = str(iobj)
                obj_str = obj_str.zfill(2)
                # only consider the i th object
                mask_ref = gt.point(lambda i: 255 if i == iobj else 0)
                if not utils.HasObject(mask_ref):
                    continue
                feature_ref = net_prediction(img, mask_ref, model)

                mask_rp = img_fol_rp + img_fol_name + '/'
                mask_set = os.listdir(mask_rp)
                mask_set.sort()
                if len(mask_set) == 0:
                    print('encounter empty folder')
                    continue

                for iord, mask_name in enumerate(mask_set):
                    print(mask_rp + mask_name)
                    mask = Image.open(mask_rp + mask_name)
                    # skip too tiny object
                    if not utils.HasObject(mask):
                        # print('skip tiny mask %s', mask_name)
                        mask_set[iord] = []
                        continue
                    # feature vector diatance
                    feature_c = net_prediction(img, mask, model)
                    distance = torch.dist(feature_c, feature_ref, p=2)
                    distance = distance.data[0]
                    distance_set.append(distance)
                    # segmentation ratio
                    mask_ref_np = np.asarray(mask_ref, dtype=np.uint8)
                    ground_truth_bool = mask_ref_np == 255
                    mask_np = np.asarray(mask, dtype=np.uint8)
                    mask_bool = mask_np == 255
                    obj_value = np.sum((mask_bool & ground_truth_bool)) / np.sum(mask_bool, dtype=np.float32)
                    ratio_set.append(obj_value)
    out = open('data_zd.pkl', 'wb')
    pickle.dump([distance_set, ratio_set], out)
    out.close()

if __name__=='__main__':
    main()
