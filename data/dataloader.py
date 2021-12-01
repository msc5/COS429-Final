import torch

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor

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
        )
    return dataloader

if __name__ == '__main__':
    pass
