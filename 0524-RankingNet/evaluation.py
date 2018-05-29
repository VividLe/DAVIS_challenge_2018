# coding: utf-8

import argparse
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

import dataset_8ch as saliency
import tiramisu as tiramisu
import experiment
import utils as utils
from parameter import *


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--DATASET_PATH', type=str, default='/disk5/yangle/DAVIS/dataset/cat_128/process/')
	parser.add_argument('--SAVE_DIR', type=str, default='/disk5/yangle/DAVIS/result/tiramisu/tiramisu57-8chs/')
	parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
	parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0001)
	args = parser.parse_args()

	if not os.path.exists(args.SAVE_DIR):
		os.makedirs(args.SAVE_DIR)

	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	test_dset = saliency.Saliency(
		args.DATASET_PATH, 'val', joint_transform=None,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)

	# model = tiramisu.FCDenseNet67(in_channels=8, n_classes=N_CLASSES)
	model = tiramisu.FcDnSubtle(in_channels=8, n_classes=N_CLASSES)

	# model = model.cuda()
	model = torch.nn.DataParallel(model).cuda()
	weights_fpath = 'cat_8ch-111.pth'
	state = torch.load(weights_fpath)
	model.load_state_dict(state['state_dict'])
	model.eval()

	count = 1
	for iord, (img, target, img_cont, fomask, comask) in enumerate(test_loader):
		inputs = torch.cat((img, comask, img_cont, fomask), 1)
		inputs = Variable(inputs.cuda(), volatile=True)
		output = model(inputs)
		pred = utils.get_predictions(output)
		pred = pred[0]
		img_name = str(iord) + '.png'
		save_path = args.SAVE_DIR + img_name
		torchvision.utils.save_image(pred, save_path)
		print(count)
		count += 1


if __name__=='__main__':
    main()



