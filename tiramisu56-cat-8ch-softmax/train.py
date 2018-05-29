# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os
import shutil

import tiramisu as tiramisu
import experiment
import utils as utils
from parameter import *

def main():

	torch.cuda.manual_seed(seed)
	cudnn.benchmark = CUDNN

	# model
	model = tiramisu.FCDenseNet67(in_channels=8, n_classes=N_CLASSES)
	model = model.cuda()
	# model = torch.nn.DataParallel(model).cuda()
	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))
	model.apply(utils.weights_init)
	optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE,
							  weight_decay=WEIGHT_DECAY, eps=1e-12)
	criterion = nn.NLLLoss2d().cuda()

	exp = experiment.Experiment(EXPNAME, EXPERIMENT)
	exp.init()

	START_EPOCH = exp.epoch
	END_EPOCH = START_EPOCH + N_EPOCHS

	# for epoch in range(1):
	for epoch in range(START_EPOCH, END_EPOCH):

		since = time.time()

		# ### Collect data ###
		# delete existing folder and old data
		if os.path.exists(res_root_path):
			shutil.rmtree(res_root_path)
		utils.collect_data(ori_train_base_rp, res_train_base_rp)
		utils.collect_data(ori_val_base_rp, res_val_base_rp)
		# data loader
		train_loader, val_loader = utils.data_loader(res_root_path)

		### Train ###
		trn_loss, trn_err = utils.train(model, train_loader, optimizer, criterion, epoch)
		print('Epoch {:d}: Train - Loss: {:.4f}\tErr: {:.4f}'.format(epoch, trn_loss, trn_err))
		time_elapsed = time.time() - since
		print('Train Time {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))

		### val ###
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

		utils.adjust_learning_rate(LEARNING_RATE, LR_DECAY, optimizer,
							 epoch, DECAY_LR_EVERY_N_EPOCHS)

		exp.epoch += 1


if __name__=='__main__':
    main()

