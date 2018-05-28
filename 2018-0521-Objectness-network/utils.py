# coding: utf-8

import torch.nn as nn
import torch
import torch.nn.init as init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool
import os
from PIL import Image
import random

import joint_transforms
import saliency_dataset as saliency
from parameters import *


def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 233
    #Takes input_loader and returns array of prediction tensors
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
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
    bs, h, w = preds.size()
    n_pixels = bs * h * w
    incorrect = preds.ne(targets).cpu().sum()
    err = 100. * incorrect / n_pixels
    return round(err, 5)


def compute_BCE_loss(input, label):
	criterion = nn.BCELoss()
	probs = F.sigmoid(input)
	probs_flat = probs.view(-1)
	y_flat = label.view(-1)
	loss = criterion(probs_flat, y_flat.float())
	return loss


def calculate_sample_loss(pred, target, num_object):
    loss = 0
    for inum in range(num_object):
        pred_c = pred[inum, :, :]
        tar_c = target[inum, :, :]
        loss += compute_BCE_loss(pred_c, tar_c)
    loss /= num_object
    return loss


def calculate_loss(inputs, targets, num_max):
    bat, _, _, _ = inputs.size()
    num_bat = bat
    loss = 0
    for ibat in range(bat):
        num_object = num_max[ibat]
        if num_object != 0:
            data_pred = inputs[ibat, :, :, :]
            data_tar = targets[ibat, :, :, :]
            loss += calculate_sample_loss(data_pred, data_tar, num_object)
        else:
            num_bat -= 1
    loss /= num_bat
    return loss


def train(model, trn_loader, optimizer,criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for batch_idx, (inputs, targets, num_max) in enumerate(trn_loader):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        optimizer.zero_grad()
        output = model(inputs)
        criterion = torch.nn.BCELoss()
        m = torch.nn.Sigmoid()
        bat, _, _, _ = inputs.size()
        num_bat = bat
        loss = 0
        for ibat in range(bat):
            num_object = num_max[ibat]
            if num_object != 0:
                pred = output[ibat, :, :, :]
                pred = pred[:num_object, :, :]
                targ = targets[ibat, :, :, :]
                targ = targ[:num_object, :, :]
                loss += criterion(m(pred), targ)
            else:
                num_bat -= 1
        if num_bat != 0:
            loss /= num_bat
            loss.backward()
            optimizer.step()
            loss_c = loss.data[0]
            trn_loss += loss_c
            print(loss_c)
    trn_loss /= len(trn_loader)  # n_batches
    trn_error /= len(trn_loader)
    return trn_loss, trn_error


def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_error = 0
    for data, target, num_max in test_loader:
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        output = model(data)
        criterion = torch.nn.BCELoss()
        m = torch.nn.Sigmoid()
        bat, _, _, _ = data.size()
        num_bat = bat
        loss = 0
        for ibat in range(bat):
            num_object = num_max[ibat]
            if num_object != 0:
                pred = output[ibat, :, :, :]
                pred = pred[:num_object, :, :]
                targ = target[ibat, :, :, :]
                targ = targ[:num_object, :, :]
                loss += criterion(m(pred), targ)
            else:
                num_bat -= 1
        if num_bat != 0:
            loss /= num_bat
            loss_c = loss.data[0]
            test_loss += loss_c
            print(loss_c)
    test_loss /= len(test_loader)  # n_batches
    test_error /= len(test_loader)
    return test_loss, test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        # param_group['lr'] = new_lr
        param_group['lr'] *= 0.5


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # kaiming is first name of author whose last name is 'He' lol
        init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def file_copy(fol_name):
    img_name = fol_name + '.png'
    val_img = Image.open(val_img_rp + img_name)
    val_img_np = np.asarray(val_img, dtype=np.uint8)
    rows, cols, _ = val_img_np.shape
    cont_np = np.zeros((rows, cols),dtype=np.uint8)

    img_fol_rp = syn_rp + fol_name + '/'
    img_fol_set = os.listdir(img_fol_rp)
    img_fol_set.sort()

    for iord, img_fol_name in enumerate(img_fol_set):
        img_rp = img_fol_rp + img_fol_name + '/'
        img_set = os.listdir(img_rp)

        img_sel = random.sample(img_set, 1)
        img_name_sel = img_sel[0]
        seg = Image.open(img_rp + img_name_sel)
        seg_np = np.asarray(seg, dtype=np.uint8)
        cont_np = np.where(seg_np==255, iord+1, cont_np)
    cont_img = Image.fromarray(cont_np, mode='L')
    # cont_img = cont_img.resize((224, 224), Image.NEAREST)
    cont_img.save(res_mask_rp+img_name)


def collect_data(data_root_path, phase):
    # # parallel
    # change a series of global variables
    global res_mask_rp
    res_mask_rp = data_root_path + phase + 'cont/'
    global val_img_rp
    val_img_rp = data_root_path + phase + '/'
    global syn_rp
    syn_rp = data_root_path + phase + '_synthe_mask/'

    os.makedirs(res_mask_rp)

    fol_set = os.listdir(data_root_path + phase + '_synthe_mask/')
    fol_set.sort()

    pool = Pool(processor_num)
    pool.map(file_copy, fol_set)
    pool.close()
    pool.join()
    print('data collected')


def data_loader(dataset_path):
    normalize = transforms.Normalize(mean=saliency.mean, std=saliency.std)
    train_joint_transformer = transforms.Compose([
        joint_transforms.JointResize((224)),
        joint_transforms.JointRandomHorizontalFlip()])

    train_dset = saliency.Saliency(
        dataset_path, 'train', joint_transform=train_joint_transformer,
        transform=transforms.Compose([transforms.ToTensor(), normalize, ]))
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True)

    test_joint_transforms = transforms.Compose([joint_transforms.JointResize(224)])
    val_dset = saliency.Saliency(
        dataset_path, 'val', joint_transform=test_joint_transforms,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

