# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import time
import os
import shutil

import experiment
import utils as utils
from parameter import *
import DenseNet


def main():
	cudnn.benchmark = True

	densenet201 = torchvision.models.densenet201(pretrained=True)
	dict_densenet201 = densenet201.state_dict()
	model = DenseNet.densenet201(vector_len=512)
	# # initialize
	DenseNet_dict = model.state_dict()

	pretrained_dict = {k: v for k, v in dict_densenet201.items() if k in DenseNet_dict}
	# for k in pretrained_dict:
	# 	print(k)
	DenseNet_dict.update(pretrained_dict)
	model.load_state_dict(DenseNet_dict)

	# seperate layers, to set different lr
	param_exist = []
	param_add = []
	for k, (name, module) in enumerate(model.named_children()):
		# existing layers including: self.features
		if k == 1:
			for param in module.parameters():
				param_exist.append(param)
		# adding layers including: self.classifier
		else:
			for param in module.parameters():
				param_add.append(param)
	model = model.cuda()
	# model = torch.nn.DataParallel(model).cuda()

	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))
	optimizer = optim.SGD([{'params': param_exist, 'lr': LEARNING_RATE*0.1},
						   {'params': param_add}],
						  lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

	# use margin=2
	criterion = nn.TripletMarginLoss(margin=2, p=2).cuda()

	exp_dir = EXPERIMENT + 'rankingVGG'
	if os.path.exists(exp_dir):
		shutil.rmtree(exp_dir)
	exp = experiment.Experiment('rankingVGG', EXPERIMENT)
	exp.init()

	START_EPOCH = exp.epoch
	END_EPOCH = START_EPOCH + N_EPOCHS

	for epoch in range(1, END_EPOCH):

		since = time.time()

		# # ### Collect data ###
		# # delete existing folder and old data
		if os.path.exists(res_root_path):
			shutil.rmtree(res_root_path)
		utils.collect_data(ori_train_base_rp, res_train_base_rp)
		utils.collect_data(ori_val_base_rp, res_val_base_rp)
		# data loader
		train_loader, val_loader = utils.data_loader(res_root_path)

		# # ### Train ###
		trn_loss = utils.train(model, train_loader, optimizer, criterion, epoch)
		trn_err = 0
		print('Epoch {:d}: Train - Loss: {:.4f}\tErr: {:.4f}'.format(epoch, trn_loss, trn_err))
		time_elapsed = time.time() - since
		print('Train Time {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))

		### Test ###
		val_loss = utils.test(model, val_loader, criterion, epoch)
		val_err = 0
		print('Val - Loss: {:.4f}, Error: {:.4f}'.format(val_loss, val_err))
		time_elapsed = time.time() - since
		print('Total Time {:.0f}m {:.0f}s\n'.format(
			time_elapsed // 60, time_elapsed % 60))

		### Save Metrics ###
		exp.save_history('train', trn_loss, trn_err)
		exp.save_history('val', val_loss, val_err)

		### Checkpoint ###
		exp.save_weights(model, trn_loss, val_loss, trn_err, val_err)
		exp.save_optimizer(optimizer, val_loss)

		## Early Stopping ##
		if (epoch - exp.best_val_loss_epoch) > MAX_PATIENCE:
			print(("Early stopping at epoch %d since no "
				   +"better loss found since epoch %.3").format(epoch, exp.best_val_loss))
			break

		# Adjust Lr ###
		if epoch % 4 == 0:
			utils.adjust_learning_rate(LEARNING_RATE, LR_DECAY, optimizer,
								 epoch, DECAY_LR_EVERY_N_EPOCHS)

		exp.epoch += 1


if __name__=='__main__':
    main()

