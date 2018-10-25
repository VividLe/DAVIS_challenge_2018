import os
import sys
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from dataset import get_loader
from unet_deeplab import UNet_deeplab
import math


def compute_loss(input, label):
    criterion = nn.BCELoss()
    # convert y_pred -> [0,1]
    probs = F.sigmoid(input)
    probs_flat = probs.view(-1)
    y_flat = label.view(-1)
    loss = criterion(probs_flat, y_flat.float())
    return loss


def save_image(save_dir, contents, i):
    fh = open(save_dir, 'a')
    l = len(contents)
    fh.write(str(i) + '\n')
    for n in range(l):
        fh.write(contents[n])
        fh.write('\n')
    fh.close()


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    # encode part
    # update_lr_group = optimizer.param_groups[0:7]
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer
def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    constant_lr_group = optimizer.param_groups[9]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(constant_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()

def train_net(net, epochs=5, batch_size=10, lr=0.01, val_percent=0.05,
              cp=False, gpu=False):
    train_dir_img = '/home/zhangni/dataset/DUTS/DUTS-TR/DUTS-TR-Image/'
    train_dir_mask = '/home/zhangni/dataset/DUTS/DUTS-TR/DUTS-TR-Mask/'

    # train_dir_img = './test_image/'
    # train_dir_mask = './test_mask/'

    # train_dir_img = '/home/zhangni/research/Pytorch-UNet-master-DUTS/error_image/img/'
    # train_dir_mask = '/home/zhangni/research/Pytorch-UNet-master-DUTS/error_image/mask/'
    dir_checkpoint = './checkpoint/'
    dir_model_loss = './model_epoch_loss/'
    img_size = 256
    train_loader = get_loader(train_dir_img, train_dir_mask, img_size, batch_size, mode='train',
                              num_thread=1)
    # ids = get_ids(dir_img)
    # ids = split_ids(ids)

    # iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_loader.dataset),
               str(cp), str(gpu)))

    # N_train = len(iddataset['train'])

    # train_dir_img = '/home/zhangni/dataset/ECSSD/train/images/'
    # train_dir_mask = '/home/zhangni/dataset/ECSSD/train/gt/'
    # save_img_and_mask(iddataset['train'], dir_img, dir_mask, train_dir_img, train_dir_mask)

    # val_dir_img = '/home/zhangni/dataset/ECSSD/val/images/'
    # val_dir_mask = '/home/zhangni/dataset/ECSSD/val/gt/'
    # save_img_and_mask(iddataset['val'], dir_img, dir_mask, val_dir_img, val_dir_mask)



    N_train = len(train_loader) * batch_size
    # val_loader = get_loader(val_dir_img, val_dir_mask, img_size, batch_size, mode='test',
    #                          num_thread=1)

    # optimizer = optim.SGD(net.parameters(),
                          # lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.SGD([
        # encode part (update lr)
        {'params': net.conv1.parameters(), 'lr': 0.001},
        {'params': net.conv2.parameters(), 'lr': 0.001},
        {'params': net.conv3.parameters(), 'lr': 0.001},
        {'params': net.conv4.parameters(), 'lr': 0.001},
        {'params': net.conv5.parameters(), 'lr': 0.001},
        {'params': net.fc6.parameters(), 'lr': 0.001},
        {'params': net.fc7.parameters(), 'lr': 0.001},
        # decode part
        {'params': net.fc7_1.parameters()},
        {'params': net.conv_loss1.parameters()},
        {'params': net.conv_loss2.parameters()},
        {'params': net.conv_loss3.parameters()},
        {'params': net.conv_loss4.parameters()},
        {'params': net.conv_loss5.parameters()},
        {'params': net.conv_loss.parameters()},
        {'params': net.up1.parameters()},
        {'params': net.up2.parameters()},
        {'params': net.up3.parameters()},
        {'params': net.up4.parameters()},
        {'params': net.up5.parameters()},

    ], lr=lr, momentum=0.9, weight_decay=0.0005)

    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)
    for epoch in range(epochs):
        # if (epoch+1) % 4 == 0:
            # lr *= 0.5
            # optimizer = optim.SGD(net.parameters(),
            #                       lr=lr, momentum=0.9, weight_decay=0.0005)
        # if whole_iter_num != 0:
        #     optimizer = adjust_learning_rate(optimizer, decay_rate=.9)
        #     save_dir = './model_epoch_loss/loss.txt'
        #     save_lr(save_dir, optimizer)
        #     print('have updated lr!!')
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, lr))

        epoch_total_loss = 0
        epoch_loss = 0

        # if 1:
        #    val_dice = eval_net(net, val_loader, gpu, batch_size, n_color=3, img_size=256)
        #    print('Validation Dice Coeff: {}'.format(val_dice))
        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break
            images, label_256, label_32, label_64, label_128, filename = data_batch
            images, label_256 = Variable(images.cuda()), Variable(label_256.cuda())
            label_32, label_64, label_128 = Variable(label_32.cuda()), Variable(label_64.cuda()), \
                                            Variable(label_128.cuda())
            # print(filename)
            # save_image('imagename.txt', filename, i)
            for_loss1, for_loss2, for_loss3, for_loss4, for_loss5, y_pred = net(images)
            # print("y_pred", y_pred)
            # print("label_32", label_32)
            # loss
            loss = compute_loss(y_pred, label_256)
            # loss1  after fc7
            loss1 = compute_loss(for_loss1, label_32)
            # loss2 after up1
            loss2 = compute_loss(for_loss2, label_32)
            # loss3 after up2
            loss3 = compute_loss(for_loss3, label_32)
            # loss4 after up3
            loss4 = compute_loss(for_loss4, label_64)
            # loss after up4
            loss5 = compute_loss(for_loss5, label_128)

            total_loss = loss + loss1 * 0.1 + loss2 * 0.3 + loss3 * 0.3 \
                         + loss4 * 0.5 + loss5 * 0.5
            epoch_total_loss += total_loss.data[0]
            epoch_loss += loss.data[0]


            print('whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- loss: {3:.6f}'.format((whole_iter_num+1), (i + 1) * batch_size / N_train,
                                                     total_loss.data[0], loss.data[0]))
            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1
            # print('whole_iter_num:', whole_iter_num)
            # if whole_iter_num > 30000:
            #     break
            if whole_iter_num % 7000 == 0:
                optimizer = adjust_learning_rate(optimizer, decay_rate=.1)
                save_dir = './model_epoch_loss/loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './model_epoch_loss/loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss / iter_num, epoch+1)
        # if epoch == 0 or epoch == 50 or epoch == 75 or epoch == 95 or epoch == 99:
        torch.save(net.state_dict(),
                   dir_model_loss + 'MODEL_EPOCH{}_LOSS{}.pth'.format(epoch + 1, epoch_total_loss / iter_num))
        print('Saved')

        if cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))

            print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=35, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    # parser.add_option('-c', '--load', dest='load',
    #                   default=False, help='load file model')

    (options, args) = parser.parse_args()

    deeplab_caffe2pytorch = 'train_iter_20000.caffemodel.pth'
    # vgg16_model = 'vgg16-397923af.pth'
    net = UNet_deeplab(3, 1)
    net.cuda()
    model = torch.load(deeplab_caffe2pytorch)
    # vgg16_model = torch.load(vgg16_model)
    print('load model:', deeplab_caffe2pytorch)
    net = net.init_parameters(model)
    # net = net.init_parameters_vgg16(vgg16_model)
    # print(net.conv1[0].bias.data)

    load_model = 'model_epoch_loss/MODEL_EPOCH10_LOSS1.0056836081342109.pth'
    # print(net.conv1[0].bias.data)
    load = False
    if load:
        net.load_state_dict(torch.load(load_model))
        print('Model loaded from {}'.format(load_model))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
