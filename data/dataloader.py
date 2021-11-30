import torch

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor

def omniglot_DataLoader(path):
    dataset = Omniglot(
            '../omniglot',
            True,
            download=True,
            transform=ToTensor(),
        )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
        )
    return dataloader

if __name__ == '__main__':
    pass
