import sys
import os
import re
import copy
import os.path as osp
import logging
import glob
import json
import numpy as np
import torchvision
import torch
import torch_geometric
import cv2
import random
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torchvision import transforms as Trans
from torchvision import datasets
from PIL import Image
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_off

def to_list(x):
    if not isinstance(x, (tuple, list)):
        x = [x]
    return x


def files_exist(files):
    return len(files) != 0 and all(osp.exists(f) for f in files)


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


transform = Trans.Compose([
    Trans.Resize(800, 600),
    Trans.ToTensor(),
    Trans.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


class ImageLoad(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = torch.from_numpy(array)
        return data

    def __len__(self):
        return len(self.imgs)

class myGeometricDataSet(torch_geometric.data.Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None,
                 pre_filter=None):
        super(myGeometricDataSet, self).__init__()

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.__indices__ = None

        # 执行 self._download() 方法
        if 'download' in self.__class__.__dict__.keys():
            self._download()
            # 执行 self._process() 方法
        if 'process' in self.__class__.__dict__.keys():
            self._process()

    def _download(self):
        if os.path.isfile(self.raw_paths):  # pragma: no cover
            return

        os.makedirs(self.raw_dir)
        self.download()

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            logging.warnings.warn(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            logging.warnings.warn(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        if os.path.isfile(self.processed_paths):  # pragma: no cover
            return

        print('Processing...')

        os.makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)