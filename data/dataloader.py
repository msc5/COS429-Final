import torch

import matplotlib.pyplot as plt

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor, Resize, Compose

from torch.utils.data import Dataset, DataLoader, ConcatDataset


class OmniglotDataset(Dataset):

    def __init__(
            self,
            shots=1,
            device='cpu',
            background=True,
    ):
        super(OmniglotDataset).__init__()
        self.device = device
        self.n = shots
        self.ds = Omniglot(
            'datasets/omniglot',
            background=background,
            download=True,
            transform=Compose([Resize(28), ToTensor()])
        )

    def __len__(self):
        return int(len(self.ds) / 20)

    def __getitem__(self, i):
        a = i * 20
        b = a + 20
        x = torch.cat([self.ds[j][0].unsqueeze(0) for j in range(a, b)])
        x = x.unsqueeze(0).to(self.device)
        mask = torch.randperm(20).to(self.device)
        return (x[mask[0:self.n]], x[mask[self.n:self.n * 2]]), i


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

    train_ds = OmniglotDataset(shots=1, background=True, device=device)
    test_ds = OmniglotDataset(shots=1, background=False, device=device)

    ds = Siamese(train_ds, test_ds)

    dl = DataLoader(ds, batch_size=20, shuffle=True, drop_last=True)

    print("Train Dataset Length: ", len(train_ds))
    print("Test Dataset Length: ", len(test_ds))
    print("Dataloader Length: ", len(dl))

    for i, a in enumerate(dl):
        print(
            f'{i:<5}',
            tuple(a[0][0][0].shape),
            tuple(a[0][0][1].shape),
            tuple(a[1][0][0].shape),
            tuple(a[1][0][1].shape),
        )
