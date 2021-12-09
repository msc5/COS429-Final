import torch

import matplotlib.pyplot as plt

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor, Resize, Compose

from torch.utils.data import Dataset, DataLoader, ConcatDataset


class OmniglotDataset(Dataset):

    def __init__(
            self,
            device,
            background=True,
    ):
        super(OmniglotDataset).__init__()
        self.device = device
        self.n = 20
        self.ds = Omniglot(
            'datasets/omniglot',
            background=background,
            download=True,
            transform=Compose([Resize(28), ToTensor()])
        )

    def __len__(self):
        return int(len(self.ds) / self.n)

    def __getitem__(self, i):
        a = i * self.n
        b = a + self.n
        index = torch.arange(a, b, 1).tolist()
        x = torch.cat([self.ds[j][0].unsqueeze(0) for j in index])
        return x.to(self.device), i


class Siamese(Dataset):

    def __init__(
            self,
            *datasets,
    ):
        super(Siamese).__init__()
        self.ds = [d for d in datasets]

    def __len__(self):
        return len(self.ds[0])

    def __getitem__(self, i):
        return [ds[i % len(ds)] for ds in self.ds]


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = OmniglotDataset(background=True, device=device)
    test_ds = OmniglotDataset(background=False, device=device)

    ds = Siamese(train_ds, test_ds)

    dl = DataLoader(ds, batch_size=8, shuffle=True)

    print("Train Dataset Length: ", len(train_ds))
    print("Test Dataset Length: ", len(test_ds))
    print("Dataloader Length: ", len(dl))

    for i, a in enumerate(dl):
        print(f'{i:<5}', a[0][0].shape, a[0]
              [1].shape, a[1][0].shape, a[1][1].shape)
