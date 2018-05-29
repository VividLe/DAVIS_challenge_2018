# coding: utf-8

import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch.nn.functional as F

import dataset_8ch as saliency
import tiramisu_enlarge as tiramisu
import utils as utils
from parameter import *


def net_prediction(img, comask, fomask, model):
    utils.prepare_data(img, comask, fomask)
    normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
    transform = transforms.Compose([transforms.ToTensor(), normalize, ])
    img, comask, cont, fomask = utils.test_data_loader(transform)
    inputs = torch.cat((img, comask, cont, fomask), 0)
    inputs = torch.unsqueeze(inputs, dim=0)
    inputs = Variable(inputs.cuda(), volatile=True)
    mask = model(inputs)
    return mask


def main():
    cate_fol_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/ZIP_files/DAVIS-cha/JPEGImages/480p/'
    cate_gt_rp = '/disk5/yangle/BasicDataset/dataset/DAVIS-Semantic/ZIP_files/DAVIS-cha/Annotations/480p/'
    sep_cate_rp = '/disk5/yangle/DAVIS/result/fuse/'
    res_rp = '/disk5/yangle/DAVIS/result/fuse_sig/'

    if not os.path.exists(res_rp):
        os.makedirs(res_rp)

    # load model
    model = tiramisu.FCDenseNet57(in_channels=8, n_classes=N_CLASSES)

    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    weights_fpath = 'cat_sig-weights-36.pth'
    state = torch.load(weights_fpath)
    model.load_state_dict(state['state_dict'])
    model.eval()

    cate_set = os.listdir(sep_cate_rp)
    cate_set.sort()
    for cate_name in cate_set:
        # skip test or dev
        ann_rp = cate_gt_rp + cate_name
        if not os.path.exists(ann_rp):
            continue
        mask_begin = Image.open(ann_rp + '/00000.png')
        _, object_num = mask_begin.getextrema()
        for iobj in range(1, object_num+1):
            # only consider the i th object
            mask_former = mask_begin.point(lambda i: 255 if i == iobj else 0)

            img_fol_rp = sep_cate_rp + cate_name + '/'
            img_fol_set = os.listdir(img_fol_rp)
            img_fol_set.sort()
            # start from image 00001.jpg
            for ord_fol, img_fol_name in enumerate(img_fol_set[1:]):
                img = Image.open(cate_fol_rp + cate_name + '/' + img_fol_name + '.jpg')
                mask_rp = img_fol_rp + img_fol_name + '/'
                mask_set = os.listdir(mask_rp)
                mask_set.sort()
                comask_set = []
                for mask_name in mask_set:
                    # only consider the i th object
                    if iobj != int(mask_name[-6:-4]):
                        continue
                    comask = Image.open(mask_rp + mask_name)
                    # skip too tiny object
                    if not utils.HasObject(comask):
                        continue
                    comask_set.append(comask)
                    output = net_prediction(img, comask, mask_former, model)
                    pred_mask = output[4]
                    pred_mask = pred_mask.squeeze()
                    pred_mask = F.sigmoid(pred_mask)
                    pred_mask = pred_mask.data.cpu()
                    save_path = res_rp + cate_name + img_fol_name + '_' + mask_name
                    print(save_path)
                    torchvision.utils.save_image(pred_mask, save_path)
                # collect former mask (as union) for next iteration
                if len(comask_set) > 0:
                    mask_former = utils.collect_former_mask(comask_set)



if __name__=='__main__':
    main()
