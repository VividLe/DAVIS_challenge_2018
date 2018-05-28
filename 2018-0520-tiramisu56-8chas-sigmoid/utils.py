# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import os
import random
import shutil
from multiprocessing import Pool

import dataset_sigmoid as saliency
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


def compute_BCE_loss(input, label):
	criterion = nn.BCELoss()
	probs = F.sigmoid(input)
	probs_flat = probs.view(-1)
	y_flat = label.view(-1)
	loss = criterion(probs_flat, y_flat.float())
	return loss


def train(model, trn_loader, optimizer, criterion, epoch):
	model.train()
	trn_loss = 0
	trn_error = 0
	for batch_idx, (img, targets, img_cont, fomask, comask) in enumerate(trn_loader):
		inputs = torch.cat((img, comask, img_cont, fomask), 1)
		inputs = Variable(inputs.cuda())
		targets = Variable(targets.cuda())
		optimizer.zero_grad()
		output = model(inputs)
		loss = criterion(output, targets)
		# loss = compute_BCE_loss(output, targets)
		loss.backward()
		optimizer.step()
		# trn_loss += loss.data[0]
		batch_loss = loss.data[0]
		trn_loss += batch_loss
		print('batchsize loss: %f' % (batch_loss))
		# pred = get_predictions(output)
		# trn_error += error(pred, targets.data.cpu())
	trn_loss /= len(trn_loader)  # n_batches
	trn_error /= len(trn_loader)
	return trn_loss, trn_error


def train_sig(model, trn_loader, optimizer, criterion, epoch):
	model.train()
	trn_loss = 0
	trn_error = 0
	for batch_idx, (img, targets, img_cont, fomask, comask) in enumerate(trn_loader):
		inputs = torch.cat((img, comask, img_cont, fomask), 1)
		inputs = Variable(inputs.cuda())
		optimizer.zero_grad()
		outputs = model(inputs)

		pred_mask = outputs[0]
		target = Variable(targets[0]).cuda()
		loss = compute_BCE_loss(pred_mask, target)
		loss = loss * loss_weight[0]
		for idx in range(1, 5):
			pred_mask = outputs[idx]
			target = Variable(targets[idx]).cuda()
			loss_c = compute_BCE_loss(pred_mask, target)
			loss += loss_c * loss_weight[idx]
			# pred = get_predictions(pred_mask)
			# trn_error += error(pred, target.data.cpu()) * 0.2

		loss.backward()
		optimizer.step()
		loss_value = loss.data[0]
		print(loss_value)
		trn_loss += loss_value

	trn_loss /= len(trn_loader)  # n_batches
	trn_error /= len(trn_loader)
	return trn_loss, trn_error


def test(model, test_loader, criterion, epoch=1):
	model.eval()
	test_loss = 0
	test_error = 0
	for img, targets, img_cont, fomask, comask in test_loader:
		inputs = torch.cat((img, comask, img_cont, fomask), 1)
		inputs = Variable(inputs.cuda(), volatile=True)
		outputs = model(inputs)
		print(type(outputs))
		print(len(outputs))
		pred_mask = outputs[4]
		target = Variable(targets[4]).cuda()
		test_loss += compute_BCE_loss(pred_mask, target)
	test_loss /= len(test_loader)  # n_batches
	test_error /= len(test_loader)
	return test_loss, test_error


def test_sig(model, test_loader, criterion, epoch=1):
	model.eval()
	test_loss = 0
	test_error = 0
	for img, targets, img_cont, fomask, comask in test_loader:
		inputs = torch.cat((img, comask, img_cont, fomask), 1)
		inputs = Variable(inputs.cuda(), volatile=True)
		outputs = model(inputs)
		pred_mask = outputs[0]
		target = Variable(targets[0]).cuda()
		loss = compute_BCE_loss(pred_mask, target)
		print(loss.data[0])
		test_loss += loss

	test_loss /= len(test_loader)  # n_batches
	test_error /= len(test_loader)
	return test_loss.data[0], test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
	"""Sets the learning rate to the initially 
		configured `lr` decayed by `decay` every `n_epochs`"""
	new_lr = lr * (decay ** (cur_epoch // n_epochs))
	for param_group in optimizer.param_groups:
		# param_group['lr'] = new_lr
		param_group['lr'] *= 0.5


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		# kaiming is first name of author whose last name is 'He' lol
		init.kaiming_uniform(m.weight)
		# m.bias.data.zero_()


def file_copy(fol_name):
	base_file_rp = ori_base_rp + fol_name + '/'
	ori_img_rp = base_file_rp + 'image/'
	ori_comask_rp = base_file_rp + 'comask/'
	ori_fomask_rp = base_file_rp + 'fomask/'

	img_set = os.listdir(ori_img_rp)
	sel_order = random.randint(0, len(img_set) - 1)
	sel_name = img_set[sel_order]

	shutil.copyfile(ori_img_rp + sel_name, res_img_rp + sel_name)
	shutil.copyfile(ori_comask_rp + sel_name, res_comask_rp + sel_name)
	shutil.copyfile(ori_fomask_rp + sel_name, res_fomask_rp + sel_name)
	shutil.copyfile(base_file_rp + fol_name + '_gt.png', res_gt_rp + sel_name)
	shutil.copyfile(base_file_rp + fol_name + '.png', res_cont_rp + sel_name)


def collect_data(pass_base_rp, res_base_rp):
	# # parallel
	# change a series of global variables
	global ori_base_rp
	ori_base_rp = pass_base_rp
	global res_img_rp
	res_img_rp = res_base_rp + '/'
	global res_gt_rp
	res_gt_rp = res_base_rp + 'annot/'
	global res_cont_rp
	res_cont_rp = res_base_rp + 'cont/'
	global res_comask_rp
	res_comask_rp = res_base_rp + 'comask/'
	global res_fomask_rp
	res_fomask_rp = res_base_rp + 'fomask/'

	os.makedirs(res_img_rp)
	os.makedirs(res_gt_rp)
	os.makedirs(res_cont_rp)
	os.makedirs(res_comask_rp)
	os.makedirs(res_fomask_rp)

	fol_name_set = os.listdir(ori_base_rp)
	pool = Pool(processor_num)
	pool.map(file_copy, fol_name_set)
	pool.close()
	pool.join()
	print('data collected')


def data_loader(dataset_path):
	# dataset
	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])

	train_dset = saliency.Saliency(
		dataset_path, 'train', joint_transform=train_joint_transformer,
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


def data_loader_sig(dataset_path):
	# dataset
	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])
	img_transform = transforms.Compose([transforms.ToTensor(), normalize])
	target_transform = transforms.Compose([transforms.ToTensor()])
	mask_size_list = [14, 28, 56, 112, 224]

	train_dset = saliency.Saliency(dataset_path, 'train', joint_transform=train_joint_transformer,
								   transform=img_transform, target_transform=target_transform,
								   mask_size_list=mask_size_list)
	train_loader = torch.utils.data.DataLoader(
		train_dset, batch_size=batch_size, shuffle=True)

	val_dset = saliency.Saliency(dataset_path, 'val',
								 transform=img_transform, target_transform=target_transform,
								 mask_size_list=mask_size_list)
	# decrease the validation batchsize
	val_loader = torch.utils.data.DataLoader(
		val_dset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader
