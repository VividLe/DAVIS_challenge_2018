# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os
import shutil

import experiment
import utils as utils
from parameter import *
from unet_deeplab import UNet_deeplab


def main():
	cudnn.benchmark = True

	deeplab_caffe2pytorch = 'train_iter_20000.caffemodel.pth'
	print('load model:', deeplab_caffe2pytorch)
	pretrained_model = torch.load(deeplab_caffe2pytorch)
	model = UNet_deeplab(in_channels=4, feature_length=512)
	model = model.init_parameters(pretrained_model)

	# seperate layers, to set different lr
	param_exist = []
	param_add = []
	for k, (name, module) in enumerate(model.named_children()):
		# existing layers including: conv1~conv5, fc6, fc7
		if k < 7:
			for param in module.parameters():
				param_exist.append(param)
		# adding layers including: fc7_1
		else:
			for param in module.parameters():
				param_add.append(param)
	model = model.cuda()

	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))
	optimizer = optim.RMSprop([{'params': param_exist, 'lr': LEARNING_RATE*0.1},
						   {'params': param_add}], lr=LEARNING_RATE,
							  weight_decay=WEIGHT_DECAY, eps=1e-12)

	# use margin=2
	criterion = nn.TripletMarginLoss(margin=2, p=2).cuda()

	exp_dir = EXPERIMENT + 'ranking-test'
	if os.path.exists(exp_dir):
		shutil.rmtree(exp_dir)
	exp = experiment.Experiment('ranking-test', EXPERIMENT)
	exp.init()

	START_EPOCH = exp.epoch
	END_EPOCH = START_EPOCH + N_EPOCHS

	for epoch in range(START_EPOCH, END_EPOCH):

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

		# Adjust Lr ###--old method
		utils.adjust_learning_rate(LEARNING_RATE, LR_DECAY, optimizer,
							 epoch, DECAY_LR_EVERY_N_EPOCHS)

		exp.epoch += 1


if __name__=='__main__':
    main()

