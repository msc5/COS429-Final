import torch

from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, RandomSampler
from torchvision.transforms import ToTensor

from PIL import Image

import os
import json
import random
import math
import itertools
import time

import util
from .tests import iterate_dataset


class FewShotDataset(Dataset):

    def __init__(
            self,
            path,
            split='train',
            structure=['Class'],
            class_level=0,
            device='cpu',
    ):
        self.dir = os.path.join('datasets', path)
        self.split = split
        self.structure = structure
        self.class_level = class_level
        self.device = device
        self.transform = ToTensor()
        self.__name__ = 'FewShotDataset [' + path + ']'
        self.build()

    def build(self):
        """
        Builds a dictionary from the file tree structure, with each lowest node
        representing whether that file has been accessed
        """

        # Create dictionary from file tree structure
        dict = {}
        for root, dirs, files in os.walk(self.dir):
            d = dict
            for x in root.split(os.sep):
                d = d.setdefault(x, {})
            if files:
                for f in files:
                    d.setdefault(f, False)

        # Traverse dictionary back down to self.dir
        for d in self.dir.split(os.sep):
            dict = dict[d]
        self.dict = dict

    def __str__(self):
        classes = self.classes()
        num_classes = len(classes)
        num_examples = 0
        for c in classes:
            num_examples += len(self.examples(c))
        avg_examples = int(num_examples / num_classes)
        msg = (
            f'{"Number of Classes: ":>40}{num_classes}\n'
            f'{"Number of Examples: ":>40}{num_examples}\n'
            f'{"Average Number of Examples per Class: ":>40}{avg_examples}'
        )
        return msg

    def __item(self, key=None):
        d = self.dict
        if not key:
            return d[self.split]
        # TODO: Implement error handling here
        for p in key.split(os.sep):
            d = d[p]
        return d

    def __items(self, root=None):
        q = [(self.split if root is None else root, self.__item(root), 0)]
        items = []
        while q:
            key, val, level = q.pop(0)
            if len(items) < level + 1:
                items.append([])
            items[level].append(key)
            if val:
                for k, v in val.items():
                    q.append((os.path.join(key, k), v, level + 1))
        return items

    def classes(self):
        """
        Returns a list of all classes in the dataset
        """
        return self.__items()[self.class_level + 1]

    def examples(self, key):
        """
        Returns a list of all examples in the dataset for a given class
        """
        return self.__items(key)[-1]

    def __get_images(self, keys):
        images = [
            torch.cat([
                self.transform(
                    Image.open(
                        os.path.join(self.dir, p)
                    )).unsqueeze(0) for p in k
            ]).unsqueeze(0) for k in keys
        ]
        return torch.cat(images, dim=0).to(self.device)

    def __getitem__(self, keys):
        s, q = keys
        support = self.__get_images(s)
        query = self.__get_images(q)
        return support, query


class FewShotSampler(Sampler):

    def __init__(self, dataset, k=5, n=1, m=1):
        self.ds = dataset
        self.k = k
        self.n = n
        self.m = m

    def __emit_c_iter(self):
        c_iter = iter(BatchSampler(RandomSampler(self.classes), self.k, True))
        return c_iter

    def __emit_e_iter(self):
        e_iter = [
            iter(BatchSampler(RandomSampler(e), self.n + self.m, True))
            for e in self.examples
        ]
        return e_iter

    def __step_e_iter(self):
        e = [next(e_iter, None) for e_iter in self.e_iter]
        if not e[0]:
            self.c = next(self.c_iter, None)
            if not self.c:
                raise StopIteration
            self.c = [self.classes[i] for i in self.c]
            self.examples = [self.ds.examples(c) for c in self.c]
            self.e_iter = self.__emit_e_iter()
            return self.__step_e_iter()
        e = [[self.examples[j][i] for i in examples]
             for j, examples in enumerate(e)]
        s = [examples[:(self.n)] for examples in e]
        q = [examples[(self.n):] for examples in e]
        return s, q

    def __iter__(self):
        self.dict = self.ds.dict.copy()
        self.classes = self.ds.classes()
        self.c_iter = self.__emit_c_iter()
        self.c = [self.classes[i] for i in next(self.c_iter)]
        self.examples = [
            self.ds.examples(c) for c in self.c
        ]
        self.e_iter = self.__emit_e_iter()
        return self

    def __next__(self):
        return self.__step_e_iter()

    def __len__(self):
        pass


def emitFewShotLoader(data, device, split, bs, k, n, m):
    try:
        path = os.path.join('datasets', 'datasets.json')
        config = json.load(open(path))[data]
        structure = config['structure']
        class_level = config['class_level']
    except:
        print('Error Opening datasets.json')
    ds = FewShotDataset(
        data,
        split=split,
        structure=structure,
        class_level=class_level,
        device=device,
    )
    s = FewShotSampler(ds, k=k, n=n, m=m)
    dl = DataLoader(ds, sampler=s, batch_size=bs, drop_last=True)
    return dl


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = 'omniglot'
    bs = 20
    k = 5
    n = 5
    m = 1

    dl = emitFewShotLoader(data, device, 'train', bs, k, n, m)
    ds = dl.dataset

    iterate_dataset(dl, ds)
