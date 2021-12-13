import torch

import matplotlib.pyplot as plt

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor, Resize, Compose

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from miniimagenettools.mini_imagenet_dataloader import MiniImageNetDataLoader


class OmniglotDataset(Dataset):

    def __init__(
            self,
            n=1,
            m=1,
            device='cpu',
            background=True,
    ):
        super(OmniglotDataset).__init__()
        self.device = device
        self.n = n
        self.m = m
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
        x = torch.cat([self.ds[j][0].unsqueeze(0)
                      for j in range(a, b)]).to(self.device)
        mask = torch.randperm(20).to(self.device)
        return (x[mask[0:self.n]], x[mask[self.n:self.n + self.m]]), i


class ImageNetDataLoader:

    def __init__(
        self,
        k=20,
        n=1,
        m=1,
        phase='train'
    ):
        self.phase = phase
        self.dl = MiniImageNetDataLoader(
            shot_num=n,
            way_num=k,
            episode_test_sample_num=m
        )
        self.dl.generate_data_list(phase=phase)
        self.dl.load_list(phase=phase)

    def __getitem__(self, i):
        s, q = self.dl.get_batch(phase=self.phase, idx=i)
        return s, q


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # train_data = OmniglotDataset(background=True, device=device)
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # # print(len(train_dataloader.ds), train_dataloader.ds.batch_size)
    # print(f'Batch Size: {train_dataloader.batch_size}')
    # print(
    #     f'Dataloader Length = 964/{train_dataloader.batch_size}: {len(train_dataloader)}')
    # for X, y in train_dataloader:
    #     print(f'Shape of X and dtype: {X.shape}, {X.dtype}')
    #     print(f'Shape of y and dtype: {y.shape}, {y.dtype}')
    #     break

    train_ds = OmniglotDataset(shots=3, background=True, device=device)
    test_ds = OmniglotDataset(shots=1, background=False, device=device)

    ds = Siamese(train_ds, test_ds)

    dl = DataLoader(ds, batch_size=20, shuffle=True, drop_last=True)

    print("Train Dataset Length: ", len(train_ds))
    print("Test Dataset Length: ", len(test_ds))
    print("Dataloader Length: ", len(dl))

    for i, a in enumerate(dl):
        # a[0] is the training dataset with a length of 2
        # a[0][0] is a list containing the support and query set of this batch, a[0][1] are the classes only in this batch
        # a[1] is the testing dataset

        if i == 0:
            print(f'a[0][0]')
            # grabs a list of length 2 containing the support and query set
            print(a[0][0])
            print()
            print(f'a[0][0][0]')  # grabs the support set
            print(a[0][0][0])
            # print(a[0][1]) # grabs the tensor array containing all classes only in this batch
            # print(a[0][1][0]) # grabs first class of this batch
            print()

        print(
            f'{i:<5}',
            tuple(a[0][0][0].shape),
            tuple(a[0][0][1].shape),
            tuple(a[1][0][0].shape),
            tuple(a[1][0][1].shape),
        )
