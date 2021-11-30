import torch

from torchvision.datasets import ImageNet, Omniglot


def omniglot_DataLoader(path):
    dataset = Omniglot(
            '../omniglot',
            True,
            download=True,
        )
    dataloader = torch.utils.data.DataLoader(dataset)
    # return dataloader
    return dataset

if __name__ == '__main__':
    pass
