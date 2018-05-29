import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader

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


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


class Saliency(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.mean = mean
        self.std = std
        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        # print(path)
        img = Image.open(path)
        # mask
        ann_path = path.replace(self.split, self.split + 'annot')
        target = Image.open(ann_path)
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

        target = self.target_transform(target)
        return img, target, img_cont, fomask, comask

    def __len__(self):
        return len(self.imgs)


class TestImage(data.Dataset):

    def __init__(self, root, split='val', joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.mean = mean
        self.std = std
        self.imgs, self.nlist = _make_image_namelist(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        # img = self.loader(path)
        img = Image.open(path)
        nlist = self.nlist[index]
        # target = Image.open(path.replace(self.split, self.split + 'annot'))
        # context image
        cont_path = path.replace(self.split, self.split + 'cont')
        img_cont = Image.open(cont_path)
        # box
        box_path = path.replace(self.split, self.split + 'box')
        img_box = Image.open(box_path)
        # box context
        boxC_path = path.replace(self.split, self.split + 'boxC')
        img_boxC = Image.open(boxC_path)

        # if self.joint_transform is not None:
        #     img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)
            img_cont = self.transform(img_cont)
            img_box = self.transform(img_box)
            img_boxC = self.transform(img_boxC)

        # target = self.target_transform(target)
        return img, nlist, img_cont, img_box, img_boxC

    def __len__(self):
        return len(self.imgs)

