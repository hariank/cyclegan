import glob
from os.path import join

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class UnpairedI2IDataset(Dataset):
    def __init__(self, A_dir, B_dir, transform=None):
        self.imgs_A = self._get_images(A_dir)
        self.imgs_B = self._get_images(B_dir)
        self.transform = transform

    def _get_images(self, im_dir):
        imgs = []
        for img_name in glob.glob(join(im_dir, '*.jpg')):
            imgs.append(img_name)
        return imgs

    def __len__(self):
        return max(len(self.imgs_A), len(self.imgs_B))

    def __getitem__(self, idx):
        img_A = Image.open(self.imgs_A[idx])
        img_B = Image.open(self.imgs_B[np.random.randint(len(self.imgs_B))])
        return self.transform(img_A), self.transform(img_B)


def get_loaders(args, transform):
    train_dataset = UnpairedI2IDataset(join(args.data_dir, 'trainA'),
                                       join(args.data_dir, 'trainB'),
                                       transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              **args.loader_kwargs)
    test_dataset = UnpairedI2IDataset(join(args.data_dir, 'testA'),
                                      join(args.data_dir, 'testB'),
                                      transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             **args.loader_kwargs)

    return train_loader, test_loader
