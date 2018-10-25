# coding: utf-8

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import camvid_dataset as camvid
import tiramisu
import experiment
import utils


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--EXP_NAME', type=str, default='tiramisu12')
	parser.add_argument('--EXP_DIR', type=str, default='/disk3/yangle/code/benchmark/tiramisu/')
	parser.add_argument('--CAMVID_PATH', type=str, default='data/CamVid/')
	parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
	parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
	args = parser.parse_args()

	normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
	test_dset = camvid.CamVid(
		args.CAMVID_PATH, 'test', joint_transform=None,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize
		]))
	test_loader = torch.utils.data.DataLoader(
		test_dset, batch_size=2, shuffle=False)

	model = tiramisu.FCDenseNet103(n_classes=12)
	model = model.cuda()
	optimizer = optim.RMSprop(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
	experiment = experiment.Experiment(args.EXP_NAME, args.EXP_DIR)
	experiment.resume(model, optimizer)

	criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda()).cuda()
	test_loss, test_error = utils.test(model, test_loader, criterion)
	print(test_loss)
	print(test_error)

if __name__=='__main__':
    main()



