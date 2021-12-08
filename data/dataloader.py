import torch

import matplotlib.pyplot as plt

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor, Resize

from torch.utils.data import Dataset, DataLoader


def omniglot_DataLoader():
    dataset = Omniglot(
        'datasets/omniglot',
        True,
        download=True,
        transform=ToTensor(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
    )
    return dataset, dataloader


class OmniglotDataset(Dataset):

    def __init__(self, device):
        self.device = device
        self.batch_size = 20
        self.ds = Omniglot(
            'datasets/omniglot',
            background=True,
            download=True,
            transform=self.transform
        )

    def transform(self, x):
        x = ToTensor()(x)
        x = Resize(28)(x)
        return x

    def __len__(self):
        return int(len(self.ds) / self.batch_size)

    def __getitem__(self, i):
        a = i * self.batch_size
        b = a + self.batch_size
        index = torch.arange(a, b, 1).tolist()
        x = torch.cat([self.ds[j][0].unsqueeze(0) for j in index])
        return x.to(self.device), i


class OmniglotDataLoader(DataLoader):

    def __init__(self, device, batch_size=1):
        self.device = device
        self.ds = OmniglotDataset(self.device)
        self.batch_size = batch_size
        self.i = 0

    def __len__(self):
        rem = 0 if len(self.ds) % self.batch_size == 0 else 1
        return int(len(self.ds) / self.batch_size) + rem

    def __getitem__(self, i):
        a = i * self.batch_size
        b = a + self.batch_size
        index = torch.arange(a, b if b < len(self.ds) else len(self.ds), 1)
        x = torch.cat([self.ds[j][0].unsqueeze(0) for j in index.tolist()])
        y = torch.tensor([self.ds[j][1] for j in index.tolist()]).unsqueeze(1)
        return x.to(self.device), y.to(self.device)

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        self.i += 1
        if self.i > len(self):
            self.i = 0
            raise StopIteration
        return self[i]


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dl = OmniglotDataLoader(device, 10)
    print(len(dl.ds), dl.ds.batch_size)
    print(len(dl), dl.batch_size)
    for i, a in enumerate(dl):
        print(i, a[1].shape, a[1].shape)
