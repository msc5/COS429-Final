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
            f'{"":10}{"Epoch":10}{"Loss":15}{"Accuracy":15}{"Elapsed Time":15}\n'
    )
    f.write(msg)
    print(msg)


def write_ep(f, i, epochs, loss):
    msg = (
            f'{"":10}{i:<2}/ {epochs:<6}{loss:10.6f}'
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

    print(device)
    size = len(dataloader)
    batch_size = dataloader.batch_size

    # Name of model and save location
    name = model.__name__
    path = os.path.join('models', name)

    # Write file stream    
    if not os.path.exists('logs'): os.makedirs('logs')
    f = io.open(os.path.join('logs', name + '_log'), 'a')

    # Use GPU or CPU to train model
    model = model.to(device)
    model.train()
    torch.cuda.empty_cache()

    header(f)

    for i in range(1, epochs):

        for data in tqdm(
                dataloader,
                desc=f'{"":10}{i:<10}',
                colour='green',
                bar_format='{desc}|{bar:30}| {rate_fmt}',
                leave=False,
        ):

            inputs, labels = data
            optim.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optim.step()

        write_ep(f, i, epochs, loss.item())

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

    # Test Model on dataloader


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = data.omniglot_DataLoader()
    model = arch.ResNetwork(1, 105, 1623)
    # moldel = arch.relation_network()
    optim = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optim, loss_fn, 10, device)
