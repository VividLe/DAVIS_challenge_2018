import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader
import torchvision.transforms as transforms

import joint_transforms

mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

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


def img_transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


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


class Dataset(data.Dataset):

    def __init__(self, root, split='train', joint_transform_img=None,
                 mask_size_list = None,
                 transform=False, target_transform=LabelToLongTensor()):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform_img = joint_transform_img
        self.mask_size_list = mask_size_list
        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        if self.joint_transform_img is not None:
            [img] = self.joint_transform_img([img])
        img = np.array(img, dtype=np.uint8)
        if self.transform:
            img = img_transform(img)

        ann_path = path.replace(self.split, self.split + 'annot')
        target = Image.open(ann_path)
        targets = []
        if self.mask_size_list is not None:
            for size in self.mask_size_list:
                transform_maks = transforms.Compose([joint_transforms.MaskResize(size)])
                [target_c] = transform_maks([target])
                targets.append(self.target_transform(target_c))

        return img, targets

    def __len__(self):
        return len(self.imgs)


