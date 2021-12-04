import torch

import matplotlib.pyplot as plt

from torchvision.datasets import ImageNet, Omniglot
from torchvision.transforms import ToTensor


def omniglot_DataLoader():
    dataset = Omniglot(
        'datasets/omniglot',
        False,
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


if __name__ == '__main__':
    training_data, dl = omniglot_DataLoader()
    # im, lb = ds[30]
    # plt.imshow(im.view(105, 105), cmap='gray')
    # plt.show()
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 10, 10
    print(len(training_data))
    print('characters = ', len(training_data) / 20)
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
