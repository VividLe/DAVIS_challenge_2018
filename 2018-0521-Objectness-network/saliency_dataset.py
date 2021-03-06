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


class ConvertTarget(object):
    def __call__(self, mask):
        mask = mask.resize((224, 224), Image.NEAREST)
        mask_np = np.asarray(mask, dtype=np.uint8)
        target_np = np.zeros((10, 224, 224), dtype=np.float)
        for inp in range(mask_np.max()):
            target_np[inp, :, :] = np.where(mask_np == inp + 1, 1, target_np)
        target_d = torch.from_numpy(target_np)
        target_t = target_d.float()
        num_max = mask_np.max()
        return target_t, num_max


class ConvertCont(object):
    def __call__(self, mask):
        mask = mask.resize((224, 224), Image.NEAREST)
        mask_np = np.asarray(mask, dtype=np.uint8)
        target_d = torch.from_numpy(mask_np)
        target_t = target_d.float()
        return target_t


class Saliency(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=ConvertTarget(),
                 cont_transform=ConvertCont()):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.joint_transform = joint_transform
        self.target_transform = target_transform
        self.cont_transform = cont_transform
        self.mean = mean
        self.std = std
        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        ann_path = path.replace(self.split, self.split + 'annot')
        target = Image.open(ann_path)
        cont_path = path.replace(self.split, self.split + 'cont')
        cont = Image.open(cont_path)

        if self.joint_transform is not None:
            img, target, cont = self.joint_transform([img, target, cont])

        if self.transform is not None:
            img = self.transform(img)

        cont = self.cont_transform(cont)
        cont = torch.unsqueeze(cont, dim=0)

        target, num_max = self.target_transform(target)
        img = torch.cat((img, cont), dim=0)
        return img, target, num_max

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
        img = self.loader(path)
        nlist = self.nlist[index]
        # target = Image.open(path.replace(self.split, self.split + 'annot'))

        # if self.joint_transform is not None:
        #     img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)

        # target = self.target_transform(target)
        return img, nlist

    def __len__(self):
        return len(self.imgs)

