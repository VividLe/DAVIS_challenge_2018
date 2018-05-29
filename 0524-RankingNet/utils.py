# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import os
import random
import shutil
from multiprocessing import Pool
from PIL import Image
import numpy as np

import dataset_8ch as saliency
import joint_transforms
from parameter import *

# global
ori_base_rp = None
res_img_rp = None
res_gt_rp = None
res_cont_rp = None
res_box_rp = None
res_boxC_rp = None

# def predict(model, input_loader, n_batches=1):
#     input_loader.batch_size = 233
#     #Takes input_loader and returns array of prediction tensors
#     predictions = []
#     model.eval()
#     for input, target in input_loader:
#         data, label = Variable(input.cuda(), volatile=True), Variable(target.cuda())
#         output = model(data)
#         pred = get_predictions(output)
#         predictions.append([input,target,pred])
#     return predictions


def get_predictions(output_batch):
	# Variables(Tensors) of size (bs,12,224,224)
	bs, c, h, w = output_batch.size()
	tensor = output_batch.data
	# Argmax along channel axis (softmax probabilities)
	values, indices = tensor.cpu().max(1)
	indices = indices.view(bs, h, w)
	return indices


def error(preds, targets):
	assert preds.size() == targets.size()
	bs, h, w = preds.size()
	n_pixels = bs * h * w
	incorrect = preds.ne(targets).cpu().sum()
	err = 100. * incorrect / n_pixels
	return round(err, 5)


def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    for batch_idx, (img, img_cont, pos, pos_cont, neg, neg_cont) in enumerate(trn_loader):
        img = torch.cat((img, img_cont), dim=1)
        pos = torch.cat((pos, pos_cont), dim=1)
        neg = torch.cat((neg, neg_cont), dim=1)
        img, pos, neg = Variable(img.cuda()), Variable(pos.cuda()), Variable(neg.cuda())
        query = model(img)
        postive = model(pos)
        negative = model(neg)
        loss = criterion(query, postive, negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tmp = loss.data[0]
        print(loss_tmp)
        trn_loss += loss_tmp
    trn_loss /= len(trn_loader)
    return trn_loss


def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    for img, img_cont, pos, pos_cont, neg, neg_cont in test_loader:
        img = torch.cat((img, img_cont), dim=1)
        pos = torch.cat((pos, pos_cont), dim=1)
        neg = torch.cat((neg, neg_cont), dim=1)
        img, pos, neg = Variable(img.cuda()), Variable(pos.cuda()), Variable(neg.cuda())
        query = model(img)
        postive = model(pos)
        negative = model(neg)
        loss = criterion(query, postive, negative)
        loss_tmp = loss.data[0]
        print(loss_tmp)
        test_loss += loss_tmp
    test_loss /= len(test_loader)
    return test_loss


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
	"""Sets the learning rate to the initially 
		configured `lr` decayed by `decay` every `n_epochs`"""
	new_lr = lr * (decay ** (cur_epoch // n_epochs))
	for param_group in optimizer.param_groups:
		param_group['lr'] *= 0.5


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		# kaiming is first name of author whose last name is 'He' lol
		init.kaiming_uniform(m.weight)
		m.bias.data.zero_()


# def file_copy(fol_name):
# 	base_file_rp = ori_base_rp + fol_name + '/'
# 	ori_pos_rp = base_file_rp + 'pos/'
# 	ori_neg_rp = base_file_rp + 'neg/'
#
# 	img_set = os.listdir(ori_img_rp)
# 	sel_order = random.randint(0, len(img_set) - 1)
# 	sel_name = img_set[sel_order]
#
# 	shutil.copyfile(ori_img_rp + sel_name, res_img_rp + sel_name)
# 	shutil.copyfile(ori_comask_rp + sel_name, res_comask_rp + sel_name)
# 	shutil.copyfile(ori_fomask_rp + sel_name, res_fomask_rp + sel_name)
# 	shutil.copyfile(base_file_rp + fol_name + '_gt.png', res_gt_rp + sel_name)
# 	shutil.copyfile(base_file_rp + fol_name + '.png', res_cont_rp + sel_name)


def copy_img_cont(ori_file, res_file):
    # image
    shutil.copyfile(ori_file, res_file)
    # context
    ori_file_cont = ori_file.replace('/img/', '/cont/').replace('.jpg', '.png')
    res_file_cont = res_file.replace('.jpg', '.png')
    shutil.copyfile(ori_file_cont, res_file_cont)


def random_coord(edge, ratio_min=0.2):
    while True:
        data = random.sample(range(0, edge), 2)
        data.sort()
        edge_min = round(edge * ratio_min)
        if data[1] - data[0] >= edge_min:
            break
    return data[0], data[1]


def prepare_data(img_name):
    img_rp = ori_base_rp + 'img/'

    img_name_base = img_name[:-6]

    img_obj_ord = img_name[-6:-4]
    order_next = int(img_name_base[-3:]) + 1
    order_next = str(order_next)
    order_next = order_next.zfill(3)
    img_name_next = img_name_base[:-3] + order_next + img_obj_ord + '.jpg'
    img_file_next = img_rp + img_name_next
    # print(img_file_next)
    if os.path.exists(img_file_next):
        # print('image %s has no subsequence, skip' % img_name)
        # continue
        # copy image
        copy_img_cont(img_rp + img_name, res_img_rp + img_name)
        # positive
        copy_img_cont(img_file_next, res_pos_rp + img_name)
        # negative
        imgs_next_prefix = img_name_base[:-3] + order_next
        prefixed_files = [filename for filename in os.listdir(img_rp) if filename.startswith(imgs_next_prefix)]
        if len(prefixed_files) > 1:
            while True:
                img_name_neg = random.sample(prefixed_files, 1)
                img_name_neg = img_name_neg[0]
                img_neg_ord = img_name_neg[-6:-4]
                if img_obj_ord != img_neg_ord:
                    break
            copy_img_cont(img_rp + img_name_neg, res_neg_rp + img_name)
        else:
            '''
            if only one object occurs in the image, 
            randomly crop a patch and resize it to 224x224, serving as negative exemplars
            Possible, the cropped patch is similar with the object
            '''
            # only contain single object, random crop
            # print('single object')
            col_min, col_max = random_coord(224, 0.2)
            row_min, row_max = random_coord(224, 0.2)
            # crop image
            img = Image.open(img_rp + img_name)
            img_np = np.asarray(img, dtype=np.uint8)
            img_crop_np = img_np[col_min:col_max, row_min:row_max, :]
            img_crop = Image.fromarray(img_crop_np, mode='RGB')
            img_crop = img_crop.resize((224, 224))
            img_crop.save(res_neg_rp + img_name)
            # crop mask
            mask_name = img_name.replace('.jpg', '.png')
            mask_rp = img_rp.replace('/img/', '/cont/')
            mask = Image.open(mask_rp + mask_name)
            mask_np = np.asarray(mask, dtype=np.uint8)
            mask_crop_np = mask_np[col_min:col_max, row_min:row_max]
            mask_crop = Image.fromarray(mask_crop_np, mode='L')
            mask_crop = mask_crop.resize((224, 224))
            mask_crop.save(res_neg_rp + mask_name)


def collect_data(pass_base_rp, res_base_rp):
    # # parallel
	# change a series of global variables
	global ori_base_rp
	ori_base_rp = pass_base_rp
	global res_img_rp
	res_img_rp = res_base_rp + '/'
	global res_pos_rp
	res_pos_rp = res_base_rp + 'pos/'
	global res_neg_rp
	res_neg_rp = res_base_rp + 'neg/'

	os.makedirs(res_img_rp)
	os.makedirs(res_pos_rp)
	os.makedirs(res_neg_rp)

	fol_name_set = os.listdir(ori_base_rp+'img/')
	pool = Pool(processor_num)
	pool.map(prepare_data, fol_name_set)
	pool.close()
	pool.join()
	print('data collected')


def data_loader(dataset_path):
	# dataset
	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])

	train_dset = saliency.Saliency(
		dataset_path, 'TRain', joint_transform=train_joint_transformer,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	train_loader = torch.utils.data.DataLoader(
		train_dset, batch_size=batch_size, shuffle=True)

	val_dset = saliency.Saliency(
		dataset_path, 'val',
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	# decrease the validation batchsize
	val_loader = torch.utils.data.DataLoader(
		val_dset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader


def scale_box(col_min, col_max, factor=0.4):
	hei_half = round((col_max - col_min + 1) * 0.5 * factor)
	hei_top = col_min - hei_half
	hei_bot = col_max + hei_half

	return int(hei_top), int(hei_bot)


def prepare_test_data(img, mask):
    PADDING = 100
    # mask
    mask_np = np.asarray(mask, dtype=np.uint8)
    col_num, row_num = mask_np.shape
    mask_pad = np.zeros((col_num + PADDING * 2, row_num + PADDING * 2), dtype=np.uint8)
    img_pad = np.zeros((col_num + PADDING * 2, row_num + PADDING * 2, 3), dtype=np.uint8)

    cols, rows = np.where(mask_np == 255)
    col_min = np.min(cols)
    col_max = np.max(cols)
    row_min = np.min(rows)
    row_max = np.max(rows)

    hei_top, hei_bot = scale_box(col_min, col_max, factor=0.4)
    wid_lef, wid_rig = scale_box(row_min, row_max, factor=0.4)
    hei_top += PADDING
    if hei_top < 0:
        hei_top = 0
    hei_bot += PADDING
    if hei_bot > col_num + PADDING * 2:
        hei_bot = col_num + PADDING * 2 - 1
    wid_lef += PADDING
    if wid_lef < 0:
        wid_lef = 0
    wid_rig += PADDING
    if wid_rig > row_num + PADDING * 2:
        wid_rig = row_num + PADDING * 2 - 1

    mask_pad[PADDING:col_num + PADDING, PADDING:row_num + PADDING] = mask_np
    mask_crop_np = mask_pad[hei_top:hei_bot, wid_lef:wid_rig]
    mask_crop = Image.fromarray(mask_crop_np, mode='L')
    # mask_crop = mask_crop.point(lambda i: i * 255)
    mask_crop = mask_crop.resize((224, 224))
    mask_crop.save(test_root_path + 'mask.png')

    img_np = np.asarray(img, dtype=np.uint8)
    img_pad[PADDING:col_num + PADDING, PADDING:row_num + PADDING, :] = img_np
    img_crop_np = img_pad[hei_top:hei_bot, wid_lef:wid_rig, :]
    img_crop = Image.fromarray(img_crop_np, mode='RGB')
    img_crop = img_crop.resize((224, 224))
    img_crop.save(test_root_path + 'image.png')


def test_data_loader(transform):
	img = Image.open(test_root_path + 'image.png')
	img = transform(img)
	mask = Image.open(test_root_path + 'mask.png')
	mask = transform(mask)
	return img, mask


def HasObject(fomask):
	fomask_np = np.asarray(fomask, dtype=np.uint8)
	cols, rows = np.where(fomask_np == 255)
	if len(cols) < 100 or (cols.max()-cols.min()) < 10 or (rows.max()-rows.min()) < 10:
		return False
	else:
		return True

