# coding: utf-8

import torch.nn as nn
import torch
import torch.nn.init as init
from torch.autograd import Variable
import torchvision
import numpy as np
import torch.nn.functional as F


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


# def criterion_new(output, targets):
#     loss = 0.00
#     for i in range(output.shape[0]):
#         loss_b = 0.00
#         targets_c = targets[i,:,:].cpu().data.numpy()
#         class_n = targets_c.max()
#         #print("targets", targets.size())
#         #print("output", output.size())
#         #print("class_n",class_n)
#         i_class = 1
#         while(i_class <= 8):
#             # print("target_c",targets_c.shape)
#             mask_i = [targets_c == i_class]
#             mask_ii = np.sum(mask_i)
#             # print(mask_ii)
#             if mask_ii == 0:
#                 targets_ps = np.zeros([output.shape[2],output.shape[3]])
#             else:
#                 targets_ps = np.zeros([output.shape[2], output.shape[3]])
#                 targets_ps = np.where(targets_c == i_class, 1, 0)
#             # targets_save = torch.from_numpy(targets_ps*255)
#             # save_path = str('../'+str(i)+'_'+str(i_class)+'.png')
#             # torchvision.utils.save_image(targets_save, save_path)
#             targets_ps = Variable(torch.from_numpy(targets_ps))
#             targets_new = targets_ps.cuda().float()
#             output_n = (output[i,i_class-1,:,:]).squeeze(0)
#             output_new = output_n.squeeze(1)
#             loss_b += compute_BCE_loss(output_new,targets_new)
#             i_class += 1
#     loss += loss_b/8
#     return loss


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

