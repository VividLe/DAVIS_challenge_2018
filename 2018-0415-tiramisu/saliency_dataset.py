import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader

# mean and std
mean = [0.485, 0.456, 0.406, 0.5]
std = [0.229, 0.224, 0.225, 1]


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
        # img = self.loader(path)
        img = Image.open(path)
        # print(img)
        ann_path = path.replace(self.split, self.split + 'annot')
        # ann_path = ann_path.replace('tif', 'png')
        target = Image.open(ann_path)
        # print(target)

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)

        target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class TestImage(data.Dataset):

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
        self.imgs, self.nlist = _make_image_namelist(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        # img = self.loader(path)
        img = Image.open(path)
        nlist = self.nlist[index]
        # target = Image.open(path.replace(self.split, self.split + 'annot'))

        if self.joint_transform is not None:
            img = self.joint_transform([img])

        if self.transform is not None:
            img = self.transform(img)

        # target = self.target_transform(target)
        return img, nlist

    def __len__(self):
        return len(self.imgs)

