"""
Title: fmnist_loader.py
Description: The loader classes for the FashionMNIST datasets
Author: Lek'Sai Ye, University of Chicago
"""

from PIL import Image
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import FashionMNIST

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms


# #########################################################################
# 1. Base Dataset
# #########################################################################
class BaseDataset(ABC):
    def __init__(self, root: str):
        super().__init__()

        self.root = root
        self.label_normal = ()
        self.label_abnormal = ()
        self.train_set = None
        self.test_set = None

    @abstractmethod
    def loaders(self,
                batch_size: int,
                shuffle_train=True,
                shuffle_test=False,
                num_workers: int = 0):
        pass

    def __repr__(self):
        return self.__class__.__name__


# #########################################################################
# 2. FashionMNIST Dataset
# #########################################################################
class FashionMNISTDataset(FashionMNIST):
    """
    Add an index to get item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        transform = transforms.ToTensor()
        img = transform(img)
        return img, int(target), index


# #########################################################################
# 2. FashionMNIST Loader for Training
# #########################################################################
class FashionMNISTLoader(BaseDataset):
    def __init__(self,
                 root: str='/net/leksai/data/FashionMNIST',
                 label_normal: tuple=(0,),
                 label_abnormal: tuple=(),  # If unsupervised, do not specify
                 ratio_abnormal: float=0.1):
        super().__init__(root)

        # Initialization
        self.root = root
        self.label_normal = label_normal
        self.label_abnormal = label_abnormal
        self.ratio_abnormal = ratio_abnormal

        # Read in initial Full Set
        # Add in download=True if you haven't downloaded yet
        print('Loading dataset for you!')
        train_set = FashionMNISTDataset(root=root, train=True, transform=transforms.ToTensor(), download=True)
        test_set = FashionMNISTDataset(root=root, train=False, transform=transforms.ToTensor(), download=True)
        print('Almost loaded!')

        # Get the labels for classes intended to use
        y_train = train_set.targets.cpu().data.numpy()
        y_test = test_set.targets.cpu().data.numpy()

        # Get the indices for classes intended to use
        train_idx = self.get_idx(y_train, label_normal, label_abnormal, ratio_abnormal, True)
        test_idx = self.get_idx(y_test, label_normal, label_abnormal, ratio_abnormal, False)

        # Get the subset
        self.train_set = Subset(train_set, train_idx)
        self.test_set = Subset(test_set, test_idx)

    def get_idx(self, y, label_normal, label_abnormal, ratio_abnormal, train):
        """
        Creat a numpy list of indices of label_ in labels.
        Inputs:
            y (np.array): dataset.targets.cpu().data.numpy()
            label_normal (tuple): e.g. (0,)
            label_abnormal (tuple): e.g. (1,)
            ratio_abnormal (float): e.g. 0.1
            train (bool): True / False
        """
        idx_normal = np.argwhere(np.isin(y, label_normal)).flatten()

        if label_abnormal:
            idx_abnormal = np.argwhere(np.isin(y, label_abnormal)).flatten()
            np.random.shuffle(idx_abnormal)
            if train:
                idx_abnormal = idx_abnormal[:int(len(idx_abnormal) * ratio_abnormal)]
            idx_all = np.hstack((idx_normal, idx_abnormal))
        else:
            idx_all = idx_normal
        return idx_all

    def loaders(self,
                batch_size: int,
                shuffle_train=True,
                shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  num_workers=num_workers,
                                  drop_last=True)
        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle_test,
                                 num_workers=num_workers,
                                 drop_last=False)
        return train_loader, test_loader


# #########################################################################
# 2. FashionMNIST Loader for Evaluation
# #########################################################################
class FashionMNISTLoaderEval(BaseDataset):
    def __init__(self,
                 root: str='/net/leksai/data/FashionMNIST',
                 label: tuple=(),
                 test_eval: bool=False):
        super().__init__(root)

        # Initialization
        self.root = root
        self.label = label

        # Read in initial Full Set
        # Add in download=True if you haven't downloaded yet
        train_set = FashionMNISTDataset(root=root, train=True, transform=transforms.ToTensor(), download=True)
        test_set = FashionMNISTDataset(root=root, train=False, transform=transforms.ToTensor(), download=True)

        # Get the labels for classes intended to use
        y_train = train_set.targets.cpu().data.numpy()
        y_test = test_set.targets.cpu().data.numpy()

        # Get the indices for classes intended to use
        train_idx = self.get_idx(y_train, label)
        test_idx = self.get_idx(y_test, label)

        # Get the subset
        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)
        if test_eval:
            self.all_set = test_set
        else:
            self.all_set = ConcatDataset((train_set, test_set))

    def get_idx(self, y, label):
        """
        Creat a numpy list of indices of label_ in labels.
        Inputs:
            y (np.array): dataset.targets.cpu().data.numpy()
            label (tuple): e.g. (0,)
        """
        return np.argwhere(np.isin(y, label)).flatten()

    def loaders(self,
                batch_size: int,
                shuffle=False,
                num_workers: int = 0):
        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)
        return all_loader
