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


def write_it(i, j, size, loss, acc, stream=None):
    msg = (
        f'{"":10}{i:<8}{j:<3} / {size:<6}{loss:<10.4f}{acc:<10.4f}{"":20}'
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
    acc = 0
    for i in range(1, epochs):

        t = tqdm(
            dataloader,
            desc=write_it(i, 0, size, l, acc),
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, data in enumerate(t):

            inputs, labels = data

            outputs = model(inputs, inputs, labels)
            lab = labels.repeat_interleave(20)
            lab_oh = torch.zeros(100, 964).to(device).scatter(
                1, lab.unsqueeze(1), 1)
            out = outputs.float()

            loss = loss_fn(out, lab_oh)
            l = loss.item()
            loss.backward()
            optim.step()

            # Compute Accuracy
            pred = out.argmax(dim=1)
            correct = torch.sum(lab == pred)
            acc = correct / 100

            t.set_description(write_it(i, j, size, l, acc))

        print(write_it(i, size, size, l, acc))

    torch.save(model.state_dict(), path)


def (outputs):
    outputs = model(inputs, inputs, labels)
    lab = labels.repeat_interleave(20)
    lab_oh = torch.zeros(100, 964).to(device).scatter(
        1, lab.unsqueeze(1), 1)
    out = outputs.float()


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
    # device = torch.device('cpu')
    dataloader = data.OmniglotDataLoader(device, 5)
    model = arch.MatchingNets(device, 1, 64)
    optim = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.NLLLoss()

    train(model, dataloader, optim, loss_fn, 10, device)
