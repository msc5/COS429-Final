import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

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
        f'{"":40}{"Train":20}{"Test":20}\n'
        f'{"":10}{"Epoch":8}{"Batch":12}'
        f'{"Loss":10}{"Accuracy":10}'
        f'{"Loss":10}{"Accuracy":10}'
        f'{"Elapsed Time":15}\n'
    )
    f.write(msg)
    print(msg)


def write_it(i, j, size, loss, acc, test_loss, test_acc, stream=None):
    msg = (
        f'{"":10}{i:<8}{j:<3} / {size:<6}'
        f'{loss:<10.4f}{acc:<10.4f}'
        f'{test_loss:<10.4f}{test_acc:<10.4f}'
    )
    if stream is not None:
        stream.write(msg)
    return msg


def train(
        model,
        dataloader,
        callbacks,
        optim,
        loss_fn,
        epochs,
        device,
):

    size = len(dataloader)
    batch_size = dataloader.batch_size

    print(device)
    print(size, batch_size)

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
    model.zero_grad()

    # Print header
    header(f)

    for i in range(1, epochs):

        t = tqdm(
            dataloader,
            desc=write_it(i, 0, size, 0, 0, 0, 0),
            colour='green',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, (train_dl, test_dl) in enumerate(t):

            inputs, _ = train_dl

            loss, acc = callbacks[0](
                model, inputs, loss_fn, batch_size, device, train=True)

            test_inputs, _ = test_dl

            test_loss, test_acc = callbacks[0](
                model, test_inputs, loss_fn, batch_size, device, train=False)

            t.set_description(
                write_it(i, j, size, loss, acc, test_loss, test_acc))

        print(write_it(i, size, size, loss, acc, test_loss, test_acc))

    torch.save(model.state_dict(), path)


def omniglotCallBack(model, inputs, loss_fn, batch_size, device, train=True):

    if train:
        model.train()
        optim.zero_grad()
    else:
        model.eval()

    outputs = model(inputs)

    # Compute Loss
    loss_tr = loss_fn(outputs[1], outputs[0])
    loss = loss_tr.item()

    # Compute Accuracy
    pred = outputs[0].argmax(dim=1)
    lab = outputs[1].argmax(dim=1)
    correct = torch.sum(lab == pred).to(device)
    acc = correct / pred.shape[0]

    if train:
        loss_tr.backward()
        optim.step()

    return loss, acc


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

    train_ds = data.OmniglotDataset(background=True, device=device)
    test_ds = data.OmniglotDataset(background=False, device=device)
    ds = data.Siamese(train_ds, test_ds)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    model = arch.MatchingNets(device, 1, 64)
    optim = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    callbacks = [omniglotCallBack]

    train(model, dl, callbacks, optim, loss_fn, 64, device)
