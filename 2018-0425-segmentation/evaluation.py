# coding: utf-8

import argparse
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

import dataset as saliency
import SegNet_resnet as SegNet
import joint_transforms
import experiment
import train_utils as utils


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--DATASET_PATH', type=str, default='/home/zhangdong/database/DUTS/')
	parser.add_argument('--SAVE_DIR', type=str, default='/home/yangle/DAVIS/result/DUTS/')
	args = parser.parse_args()

	normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
	test_dset = saliency.TestImage(
		args.DATASET_PATH, 'val', joint_transform=None,
		transform=transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			normalize
		]))

	model = SegNet.resnet50()
	# model = model.cuda()
	model = torch.nn.DataParallel(model).cuda()
	weight_path = 'ResNet50-weights-18.pth'
	state = torch.load(weight_path)
	model.load_state_dict(state['state_dict'])
	model = model.module

	test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)
	count = 1
	for data, name in test_loader:
		# data = Variable(data.cuda(), volatile=True)
		# output = model(data)
		data = Variable(data.cuda())
		with torch.no_grad():
			output = model(data)
		pred = utils.get_predictions(output[4])
		pred = pred[0]
		# transforms_size = torchvision.transforms.Resize((hei, wid))
		# mask = transforms_size([pred])
		name = name[0]
		img_name = str(name)
		save_path = args.SAVE_DIR + img_name
		torchvision.utils.save_image(pred, save_path)
		print(count)
		count += 1


if __name__=='__main__':
    main()



