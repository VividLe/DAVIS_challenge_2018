import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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

    SEL_DIST_TH = 20

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

    cate_set = os.listdir(sep_cate_rp)
    cate_set.sort()
    for cate_name in cate_set:
        # compare with the first annotated image
        img_ref = Image.open(cate_fol_rp + cate_name + '/00000.jpg')
        mask_begin = Image.open(cate_gt_rp + cate_name + '/00000.png')

        # only dispose test data
        gt_set = os.listdir(cate_gt_rp + cate_name + '/')
        if len(gt_set) > 1:
            continue
        _, object_num = mask_begin.getextrema()
        for iobj in range(1, object_num+1):
            obj_str = str(iobj)
            obj_str = obj_str.zfill(2)
            # only consider the i th object
            mask_ref = mask_begin.point(lambda i: 255 if i == iobj else 0)
            # print(img_ref.mode)
            feature_ref = net_prediction(img_ref, mask_ref, model)

            img_fol_rp = sep_cate_rp + cate_name + '/'
            img_fol_set = os.listdir(img_fol_rp)
            img_fol_set.sort()
            for img_ord, img_fol_name in enumerate(img_fol_set):
                img = Image.open(cate_fol_rp + cate_name + '/' + img_fol_name + '.jpg')
                cols_img, rows_img = img.size
                mask_rp = img_fol_rp + img_fol_name + '/'
                mask_set = os.listdir(mask_rp)
                mask_set.sort()
                if len(mask_set) == 0:
                    print('encounter empty folder')
                    continue
                distance_set = []
                for iord, mask_name in enumerate(mask_set):
                    print(mask_rp + mask_name)
                    mask = Image.open(mask_rp + mask_name)
                    # skip too tiny object
                    if not utils.HasObject(mask):
                        # print('skip tiny mask %s', mask_name)
                        mask_set[iord] = []
                        continue
                    feature_c = net_prediction(img, mask, model)
                    distance = torch.dist(feature_c, feature_ref, p=2)
                    distance = distance.data[0]
                    print('distance between two vectors is %f' % distance)


                    distance_set.append(distance)
                # sort in descend order index
                distance_set_np = np.asarray(distance_set)
                index = np.argsort(-distance_set_np)
                if len(index) < 3:
                    sel_num = len(index)
                else:
                    sel_num = round(len(index) * 0.4)

                mask_save = np.zeros((rows_img, cols_img), dtype=np.uint8)
                for ind in range(int(sel_num)):
                    mask_name = mask_set[index[ind]]
                    mask = Image.open(mask_rp + mask_name)
                    mask_np = np.asarray(mask, dtype=np.uint8)
                    mask_save = np.where(mask_np == 255, 255, mask_save)
                mask_save_file = res_rp + cate_name + img_fol_name + mask_name[:-4] + '_' + obj_str + '.png'
                mask_mer = Image.fromarray(mask_save, mode='L')
                mask_mer.save(mask_save_file)
                feature_sel = net_prediction(img, mask_mer, model)

                order = img_ord + 2.0
                feature_ref = (order-1)/order*feature_ref + 1/order * feature_sel

if __name__=='__main__':
    main()
