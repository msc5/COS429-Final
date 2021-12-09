import torch

import matplotlib.pyplot as plt

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor, Resize, Compose

from torch.utils.data import Dataset, DataLoader


# transform data outside this class
class OmniglotDataset(Dataset):

    def __init__(self, background: bool, device):
        '''
        background: True = use background set, otherwise evaluation set
        '''
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

    # each item is all images of a character (a class): there are 20 images per character and each image is (channel, height, width), so each item is (20, channel, height, width). Since all the images are the same character, the label is an integer, namely the index associated with this item.
    def __getitem__(self, i):
        a = i * self.n
        b = a + self.n
        index = torch.arange(a, b, 1).tolist()
        x = torch.cat([self.ds[j][0].unsqueeze(0) for j in index])
        return x.to(self.device), i


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = OmniglotDataset(background=True, device=device)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    print(len(ds))
    print(len(dl))
    for i, a in enumerate(dl):
        print(i, a[0].shape, a[1].shape)
