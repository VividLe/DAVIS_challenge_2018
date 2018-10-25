import os
from PIL import Image
import torch
from torch.utils import data
import transforms as trans
from torchvision import transforms

# for DUTE dataset
class ImageData(data.Dataset):
    def __init__(self, img_root, label_root, transform, t_transform, label_32_transform, label_64_transform, label_128_transform):
        self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
        self.label_path = list(map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        self.transform = transform
        self.t_transform = t_transform
        self.label_32_transform = label_32_transform
        self.label_64_transform = label_64_transform
        self.label_128_transform = label_128_transform

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')
        l = len(fn)
        filename = fn[l-1]
        image = Image.open(self.image_path[item]).convert('RGB')
        image_w = int(image.size[0])
        image_h = int(image.size[1])
        # convert('L')  convert image to gray
        label = Image.open(self.label_path[item]).convert('L')
        mask_w = int(label.size[0])
        mask_h = int(label.size[1])
        # if self.transform is not None:
        image = self.transform(image)
        # if self.t_transform is not None:
        label_256 = self.t_transform(label)
        if self.label_32_transform is not None and self.label_64_transform is not None and self.label_128_transform is\
                not None:
            label_32 = self.label_32_transform(label)
            label_64 = self.label_64_transform(label)
            label_128 = self.label_128_transform(label)
            return image, label_256, label_32, label_64, label_128, filename

        else:
            return image, label_256, filename, mask_w, mask_h, image_w, image_h


    def __len__(self):
        return len(self.image_path)


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, mode='train', num_thread=1):
    shuffle = False
    mean = torch.Tensor(3, 256, 256)
    mean[0, :, :] = 125.5325
    mean[1, :, :] = 118.1743
    mean[2, :, :] = 101.3507
    # mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255
    if mode == 'train':
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            # trans.ToTensor  image -> [0,255]
            trans.ToTensor(),
            trans.Lambda(lambda x: x - mean)
        ])
        t_transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            # transform.ToTensor  label -> [0,1]
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        label_32_transform = trans.Compose([
            trans.Scale((32, 32)),
            transforms.ToTensor(),
        ])
        label_64_transform = trans.Compose([
            trans.Scale((64, 64)),
            transforms.ToTensor(),
        ])
        label_128_transform = trans.Compose([
            trans.Scale((128, 128)),
            transforms.ToTensor(),
        ])
        shuffle = True
    else:
        # define transform to images
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            trans.ToTensor(),
            trans.Lambda(lambda x: x - mean)
        ])

        # define transform to ground truth
        t_transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
    if mode == 'train':
        dataset = ImageData(img_root, label_root, transform, t_transform, label_32_transform, label_64_transform, label_128_transform)
        # print(dataset.image_path)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
        return data_loader
    else:
        dataset = ImageData(img_root, label_root, transform, t_transform, label_32_transform=None, label_64_transform=None, label_128_transform=None)
        # print(dataset.image_path)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
        return data_loader

