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
	bs, _, h, w = preds.size()
	n_pixels = bs * h * w
	incorrect = preds.ne(targets).cpu().sum()
	err = 100. * incorrect / n_pixels
	return round(err, 5)


def compute_loss(criterion, input, label):
    # convert y_pred -> [0,1]
    probs = F.sigmoid(input)
    probs_flat = probs.view(-1)
    y_flat = label.view(-1)
    loss = criterion(probs_flat, y_flat.float())
    return loss


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
		loss = compute_loss(criterion, pred_mask, target)
		loss = loss * 0.2
		for idx in range(1, 5):
			pred_mask = outputs[idx]
			target = Variable(targets[idx]).cuda()
			loss_c = compute_loss(criterion, pred_mask, target)
			loss += loss_c * 0.2
			# pred = get_predictions(pred_mask)
			# pred = torch.unsqueeze(pred, 1)
			# trn_error += error(pred, target.data.cpu()) * 0.2

		loss.backward()
		optimizer.step()
		loss_value = loss.data[0]
		print(loss_value)
		trn_loss += loss_value
	trn_loss /= len(trn_loader)  # n_batches
	# trn_error /= len(trn_loader)
	trn_error = 0
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
		# pred = get_predictions(pred_mask)
		# test_error += error(pred, target.data.cpu())

	test_loss /= len(test_loader)  # n_batches
	# test_error /= len(test_loader)
	test_error = 0
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


def adjust_learning_rate_SGD(optimizer, decay_rate=0.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        param_group['lr'] = param_group['lr'] * decay_rate
    return optimizer

