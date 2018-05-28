# coding: utf-8

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import time
import os
import shutil

import saliency_dataset as saliency
import joint_transforms
import tiramisu
import experiment
import utils
from parameters import *


def main():

	torch.cuda.manual_seed(seed)
	cudnn.benchmark = CUDNN

	model = tiramisu.FCDenseNet57(n_classes=N_CLASSES)
	#model = model.cuda()
	model = torch.nn.DataParallel(model).cuda()
	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))
	model.apply(utils.weights_init)
	optimizer = optim.SGD(model.parameters(),
						  lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
	criterion = nn.NLLLoss2d().cuda()

	exp_dir = EXPERIMENT + 'Objectness'
	if os.path.exists(exp_dir):
		shutil.rmtree(exp_dir)

	exp = experiment.Experiment('Objectness', EXPERIMENT)
	exp.init()

	START_EPOCH = exp.epoch
	END_EPOCH = START_EPOCH + N_EPOCHS

	for epoch in range(1, END_EPOCH):

		since = time.time()
		# # ### Collect data ###
		# # # delete existing folder and old data
		cont_rp = data_root_path+'traincont/'
		if os.path.exists(cont_rp):
			shutil.rmtree(cont_rp)
		utils.collect_data(data_root_path, 'train')
		cont_rp = data_root_path + 'valcont/'
		if os.path.exists(cont_rp):
			shutil.rmtree(cont_rp)
		utils.collect_data(data_root_path, 'val')
		# data loader
		train_loader, val_loader = utils.data_loader(data_root_path)

		### Train ###
		trn_loss, trn_err = utils.train(model, train_loader, optimizer,criterion, epoch)
		print('Epoch {:d}: Train - Loss: {:.4f}\tErr: {:.4f}'.format(epoch, trn_loss, trn_err))
		time_elapsed = time.time() - since
		print('Train Time {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))

		### Test ###
		val_loss, val_err = utils.test(model, val_loader, criterion, epoch)
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
		if epoch % 4 == 0:
			utils.adjust_learning_rate(LEARNING_RATE, LR_DECAY, optimizer,
								 epoch, DECAY_LR_EVERY_N_EPOCHS)

		exp.epoch += 1


if __name__=='__main__':
    main()

