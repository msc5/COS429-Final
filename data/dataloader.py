# System Modules
import time
import json
import os

# Torch
import torch
from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader

# Modules
import util
from .fewshot import FewShotDataset, FewShotSampler, emitFewShotLoader
from .tests import iterate_dataset


class Siamese:

    def __init__(
            self,
            *dataloaders,
    ):
        self.dls = [d for d in dataloaders]
        self.iters = [iter(d) for d in self.dls]
        self.prime = self.dls[0]
        self.__name__ = ','.join([d.dataset.__name__ for d in dataloaders])
        self.batch_size = self.prime.batch_size
        self.sampler = self.prime.sampler

    def __len__(self):
        return len(self.prime)

    def __iter__(self):
        return self

    def __next__(self):
        step = [next(dl, None) for dl in self.iters]
        for i, s in enumerate(step):
            if not s:
                if i == 0:
                    raise StopIteration
                self.iters[i] = iter(self.dls[i])
                step[i] = next(self.iters[i])
        return step


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    bs = 20
    k = 5
    n = 5
    m = 15

    train_dl = emitFewShotLoader('omniglot', device, 'train', bs, k, n, m)
    test_dl = emitFewShotLoader('omniglot', device, 'test', bs, k, n, m)
    dl = Siamese(test_dl, train_dl)
    ds = dl.dls[0].dataset

    iterate_dataset(dl, ds)
