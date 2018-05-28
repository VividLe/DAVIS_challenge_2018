# coding: utf-8

import argparse
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image

# import dataset_8ch as saliency
import dataset_sigmoid as DataSet
import tiramisu_enlarge as tiramisu
import experiment
import utils as utils
from parameter import *


def main():
	# utils.collect_data(ori_val_base_rp, res_val_base_rp)
	parser = argparse.ArgumentParser()
	parser.add_argument('--DATASET_PATH', type=str, default='/disk2/zhangni/davis/cat_128/process_sig/')
	parser.add_argument('--SAVE_DIR', type=str, default='/disk2/zhangni/davis/result/mask/tiramisu57-8chs-sig/')
	args = parser.parse_args()

	if not os.path.exists(args.SAVE_DIR):
		os.makedirs(args.SAVE_DIR)

	normalize = transforms.Normalize(mean=DataSet.mean, std=DataSet.std)
	test_dset = DataSet.TestData(
		args.DATASET_PATH, 'val', joint_transform=None,
		transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
	test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)

	model = tiramisu.FCDenseNet57(in_channels=8, n_classes=N_CLASSES)

	model = model.cuda()
	#model = torch.nn.DataParallel(model).cuda()
	weights_fpath = 'cat_sig-weights-36.pth'
	state = torch.load(weights_fpath)
	model.load_state_dict(state['state_dict'])
	model.eval()

	count = 1
	for iord, (img, nlist, img_cont, fomask, comask) in enumerate(test_loader):
		inputs = torch.cat((img, comask, img_cont, fomask), 1)
		inputs = Variable(inputs.cuda(), volatile=True)
		output = model(inputs)
		pred_mask = output[4]
		pred_mask = pred_mask.squeeze()
		pred_mask = F.sigmoid(pred_mask)
		pred_mask = pred_mask.data.cpu()
		# pred_mask *= 255
		mask_name = nlist[0]
		save_path = args.SAVE_DIR + mask_name
		torchvision.utils.save_image(pred_mask, save_path)
		print(count)
		count += 1


if __name__=='__main__':
    main()



