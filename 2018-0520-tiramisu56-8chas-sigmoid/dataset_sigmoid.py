import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader
import torchvision
import torchvision.transforms as transforms
import joint_transforms

# mean and std
mean = [0.4669773, 0.42179972, 0.37274405]
std = [0.29709259, 0.28449976, 0.29766443]


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


def _make_image_namelist(dir):
    images = []
    namelist = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                item_name = fname
                namelist.append(item_name)
                item_path = os.path.join(root, fname)
                images.append(item_path)
    return images, namelist


class Saliency(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=None, mask_size_list = None,
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.mask_size_list = mask_size_list
        self.joint_transform = joint_transform
        self.loader = loader
        self.mean = mean
        self.std = std
        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        # print(path)
        img = Image.open(path)
        # gt
        ann_path = path.replace(self.split, self.split + 'annot')
        target = Image.open(ann_path)
        target = target.point(lambda i: i*255)
        # context image
        cont_path = path.replace(self.split, self.split + 'cont')
        img_cont = Image.open(cont_path)
        # box
        fomask_path = path.replace(self.split, self.split + 'fomask')
        fomask = Image.open(fomask_path)
        # box context
        comask_path = path.replace(self.split, self.split + 'comask')
        comask = Image.open(comask_path)

        if self.joint_transform is not None:
            img, target, img_cont, fomask, comask = self.joint_transform([img, target, img_cont, fomask, comask])

        if self.transform is not None:
            img = self.transform(img)
            img_cont = self.transform(img_cont)
            fomask = self.transform(fomask)
            comask = self.transform(comask)

        targets = []
        if self.mask_size_list is not None:
            for size in self.mask_size_list:
                transform_maks = transforms.Compose([joint_transforms.MaskResize(size)])
                [target_c] = transform_maks([target])
                targets.append(self.target_transform(target_c))
        return img, targets, img_cont, fomask, comask

    def __len__(self):
        return len(self.imgs)


class TestData(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=None, mask_size_list = None,
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.mask_size_list = mask_size_list
        self.joint_transform = joint_transform
        self.loader = loader
        self.mean = mean
        self.std = std
        self.imgs, self.nlist = _make_image_namelist(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        # print(path)
        img = Image.open(path)
        # name list
        nlist = self.nlist[index]
        # # gt
        # ann_path = path.replace(self.split, self.split + 'annot')
        # target = Image.open(ann_path)
        # target = target.point(lambda i: i*255)
        # context image
        cont_path = path.replace(self.split, self.split + 'cont')
        img_cont = Image.open(cont_path)
        # box
        fomask_path = path.replace(self.split, self.split + 'fomask')
        fomask = Image.open(fomask_path)
        # box context
        comask_path = path.replace(self.split, self.split + 'comask')
        comask = Image.open(comask_path)

        if self.joint_transform is not None:
            img, img_cont, fomask, comask = self.joint_transform([img, img_cont, fomask, comask])

        if self.transform is not None:
            img = self.transform(img)
            img_cont = self.transform(img_cont)
            fomask = self.transform(fomask)
            comask = self.transform(comask)

        # targets = []
        # if self.mask_size_list is not None:
        #     for size in self.mask_size_list:
        #         transform_maks = transforms.Compose([joint_transforms.MaskResize(size)])
        #         [target_c] = transform_maks([target])
        #         targets.append(self.target_transform(target_c))
        return img, nlist, img_cont, fomask, comask

    def __len__(self):
        return len(self.imgs)


