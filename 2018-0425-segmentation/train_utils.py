# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from distutils.version import LooseVersion


def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 233
    #Takes input_loader and returns array of prediction tensors
    predictions = []
    model.eval()
    for input, target in input_loader:
        data, label = Variable(input.cuda(), volatile=True), Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions


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

############################
# loss function is correct #
############################
# def cross_entropy2d(input, target, weight=None, size_average=True):
#     # input: (n, c, h, w), target: (n, h, w)
#     n, c, h, w = input.size()
#     # log_p: (n, c, h, w)
#     if LooseVersion(torch.__version__) < LooseVersion('0.3'):
#         # ==0.2.X
#         log_p = F.log_softmax(input)
#     else:
#         # >=0.3
#         log_p = F.log_softmax(input, dim=1)
#     # log_p: (n*h*w, c)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
#     log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     log_p = log_p.view(-1, c)
#     # target: (n*h*w,)
#     mask = target >= 0
#     target = target[mask]
#     loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
#     if size_average:
#         loss /= mask.data.sum()
#     return loss


def train(model, trn_loader, optimizer, criterion, epoch):
	model.train()
	trn_loss = 0
	trn_error = 0
	for batch_idx, (inputs, targets) in enumerate(trn_loader):
		inputs = Variable(inputs.cuda())
		optimizer.zero_grad()
		outputs = model(inputs)

		pred_mask = outputs[0]
		target = Variable(targets[0]).cuda()
		loss = criterion(pred_mask, target)
		loss = loss * 0.2
		for idx in range(1, 5):
			pred_mask = outputs[idx]
			target = Variable(targets[idx]).cuda()
			loss_c = criterion(pred_mask, target)
			loss += loss_c * 0.2
			pred = get_predictions(pred_mask)
			trn_error += error(pred, target.data.cpu()) * 0.2

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
	for inputs, targets in test_loader:
		inputs = Variable(inputs.cuda(), volatile=True)
		outputs = model(inputs)
		# inputs = Variable(inputs.cuda())
		# with torch.no_grad():
		# 	outputs = model(inputs)

		pred_mask = outputs[4]
		target = Variable(targets[4]).cuda()
		test_loss += criterion(pred_mask, target)
		pred = get_predictions(pred_mask)
		test_error += error(pred, target.data.cpu())

	test_loss /= len(test_loader)  # n_batches
	test_error /= len(test_loader)
	return test_loss.data[0], test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
	"""Sets the learning rate to the initially 
		configured `lr` decayed by `decay` every `n_epochs`"""
	new_lr = lr * (decay ** (cur_epoch // n_epochs))
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		# kaiming is first name of author whose last name is 'He' lol
		init.kaiming_uniform(m.weight)
		# m.bias.data.zero_()

