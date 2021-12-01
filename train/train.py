import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from tqdm import tqdm

# System modules
import os
import io

# Our modules
import arch
import data


def header(f):
    msg = (
        f'{"":10}{"Batch":10}{"Progress":10}'
    )
    f.write(msg)
    print(msg)


def write_ep(f, i):
    msg = (
        f'{"":10}{i:10}'
    )


def write_it(f, i, j, loss, n_seen, n_tot):
    msg = (
        f'{"":10}'
        f'{"iter:":10}{j:10n}'
        f'{"loss":10}{loss:.6f}'
        f'{"prog":10}{n_seen:7}/ {n_tot:7}'
    )
    f.write(msg)
    print(msg)


def train(
        model,
        dataloader,
        optim,
        loss_fn,
        epochs,
        device,
):

    size = len(dataloader)
    batch_size = dataloader.batch_size

    # Name of model, save location, and log file stream
    name = model.__name__
    path = os.path.join('models', name)
    f = io.open(os.path.join('logs', name + '_log'), 'a')

    # Use GPU or CPU to train model
    model = model.to(device)
    model.train()

    header(f)

    for i in range(epochs):

        for (j, data) in tqdm(enumerate(dataloader)):

            inputs, labels = data
            optim.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optim.step()

            # # Print details 4 times per epoch
            # if j % (size // 4) == 0:
            #     n_seen=j * batch_size
            #     n_tot=size * batch_size
            #     write_it(f, i, j, loss, n_seen, n_tot)

        write_ep(f, i, loss)

    torch.save(model.state_dict(), path)


def test(
        arch,
        path,
        dataloader
):

    # Load Model from state_dict
    model = arch()
    model.load_state_dict(torch.load(path))
    model.eval()

    # Test Model on specified data


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = data.omniglot_DataLoader()
    model = arch.ResNetwork(1, 105, 1623)
    optim = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optim, loss_fn, 10, device)
