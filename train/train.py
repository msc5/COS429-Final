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
        f'{"":30}{"Train":20}{"Test":20}\n'
        f'{"":10}{"Epoch":8}{"Batch":12}'
        f'{"Loss":10}{"Accuracy":10}'
        f'{"Loss":10}{"Accuracy":10}'
        f'{"Elapsed Time":15}\n'
    )
    f.write(msg)
    print(msg)


def write_it(i, j, size, loss, stream=None):
    msg = (
        f'{"":10}{i:<8}{j:<3} / {size:<6}{loss:<10.4f}{"":30}'
    )
    if stream is not None:
        stream.write(msg)
    return msg


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
    if not os.path.exists('logs'):
        os.makedirs('logs')
    f = io.open(os.path.join('logs', name + '_log'), 'a')

    # Use GPU or CPU to train model
    model = model.to(device)
    model.train()
    torch.cuda.empty_cache()

    header(f)

    l = 0
    for i in range(1, epochs):

        t = tqdm(
            dataloader,
            desc=write_it(i, 0, size, l),
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, data in enumerate(t):

            inputs, labels = data

            optim.zero_grad()
            outputs = model(inputs.to(device))

            loss = loss_fn(outputs, labels.to(device))
            l = loss.item()
            loss.backward()
            optim.step()

            t.set_description(write_it(i, j, size, l))

        print(write_it(i, size, size, l))

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
    ds, dataloader = data.omniglot_DataLoader()
    model = arch.ResNetwork(1, 105, 1623)
    # model = arch.relation_network()
    # model = arch.MatchingNets(1, 105, 10)
    optim = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optim, loss_fn, 10, device)
