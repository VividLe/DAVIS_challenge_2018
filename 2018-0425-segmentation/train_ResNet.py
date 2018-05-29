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
import torchvision

import dataset as dataset
import joint_transforms
import experiment
import train_utils as utils
import SegNet_resnet as SegNet


def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--DATASET_PATH', type=str, default='/data/davis/dataset/DUTS/')
	# parser.add_argument('--EXPERIMENT', type=str, default='/data/davis/result/TrainNet/')
	parser.add_argument('--DATASET_PATH', type=str, default='/disk5/zhangdong/database/DUTS/')
	parser.add_argument('--EXPERIMENT', type=str, default='/home/yangle/DAVIS/result/TrainNet/')
	parser.add_argument('--N_EPOCHS', type=int, default=200)
	parser.add_argument('--MAX_PATIENCE', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--N_CLASSES', type=int, default=2)
	parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
	parser.add_argument('--LR_DECAY', type=float, default=0.995)
	parser.add_argument('--DECAY_LR_EVERY_N_EPOCHS', type=int, default=1)
	parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
	parser.add_argument('--CUDNN', type=bool, default=True)
	args = parser.parse_args()

	torch.cuda.manual_seed(args.seed)
	cudnn.benchmark = args.CUDNN

	normalize = transforms.Normalize(mean=dataset.mean, std=dataset.std)
	train_joint_transformer = transforms.Compose([
        joint_transforms.JointResize(256),
        joint_transforms.JointRandomCrop(224),
        joint_transforms.JointRandomHorizontalFlip()])
	mask_size_list = [28, 28, 28, 56, 112]

	train_dset = dataset.Saliency(
		args.DATASET_PATH, 'train', train_joint_transformer, mask_size_list,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	train_loader = torch.utils.data.DataLoader(
		train_dset, batch_size=args.batch_size, shuffle=True)

	test_joint_transforms_img = transforms.Compose([joint_transforms.JointResize(224)])
	val_dset = dataset.Saliency(
		args.DATASET_PATH, 'val', test_joint_transforms_img, mask_size_list,
		transform=transforms.Compose([transforms.ToTensor(), normalize]))
	val_loader = torch.utils.data.DataLoader(
		val_dset, batch_size=args.batch_size, shuffle=False)

	print("TrainImages: %d" % len(train_loader.dataset.imgs))
	print("ValImages: %d" % len(val_loader.dataset.imgs))

	example_inputs, example_targets = next(iter(train_loader))
	print("InputsBatchSize: ", example_inputs.size())
	print("TargetsBatchSize: ", len(example_targets))
	print("\nInput (size, max, min) ---")
	# input
	i = example_inputs[0]
	print(i.size())
	print(i.max())
	print(i.min())
	print("Target (size, max, min) ---")
	# target
	for mask in example_targets:
		print(mask.size())
		print(mask.max())
		print(mask.min())

	resnet50 = torchvision.models.resnet50(pretrained=True)
	dict_resnet50 = resnet50.state_dict()
	model = SegNet.resnet50()
	# # initialize
	model.apply(utils.weights_init)
	SegNet_dict = model.state_dict()

	pretrained_dict = {k: v for k, v in dict_resnet50.items() if k in SegNet_dict}
	# for k in pretrained_dict:
	# 	print(k)
	SegNet_dict.update(pretrained_dict)
	model.load_state_dict(SegNet_dict)

	# seperate layers, to set different lr
	param_exist = []
	param_add = []
	for k, (name, module) in enumerate(model.named_children()):
		# existing layers including: conv1 bn1 relu maxpool
		# layer1 layer2 layer3 layer4
		if k < 8:
			for param in module.parameters():
				param_exist.append(param)
		# adding layers including: bottleneck skip3 skip2 skip1 skip0
		# conv_end_1 bn_end_1 salmap Sigmoid mask0 mask4 mask3 mask2 mask1
		else:
			for param in module.parameters():
				param_add.append(param)
	model = torch.nn.DataParallel(model).cuda()

	# model = model.cuda()
	# model = torch.nn.DataParallel(model).cuda()

	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))
	optimizer = optim.RMSprop([{'params': param_exist, 'lr': args.LEARNING_RATE*0.1},
						   {'params': param_add}], lr=args.LEARNING_RATE,
							  weight_decay=args.WEIGHT_DECAY, eps=1e-12)
	criterion = nn.NLLLoss2d().cuda()

	exp_dir = args.EXPERIMENT + 'ResNet50NL'
	if os.path.exists(exp_dir):
		shutil.rmtree(exp_dir)

	exp = experiment.Experiment('ResNet50NL', args.EXPERIMENT)
	exp.init()

	START_EPOCH = exp.epoch
	END_EPOCH = START_EPOCH + args.N_EPOCHS

	for epoch in range(START_EPOCH, END_EPOCH):

		since = time.time()

		# ### Train ###
		trn_loss, trn_err = utils.train(model, train_loader, optimizer, criterion, epoch)
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
		if (epoch - exp.best_val_loss_epoch) > args.MAX_PATIENCE:
			print(("Early stopping at epoch %d since no "
				   +"better loss found since epoch %.3").format(epoch, exp.best_val_loss))
			break

		# Adjust Lr ###--old method
		utils.adjust_learning_rate(args.LEARNING_RATE, args.LR_DECAY, optimizer,
							 epoch, args.DECAY_LR_EVERY_N_EPOCHS)

		exp.epoch += 1


if __name__=='__main__':
    main()

