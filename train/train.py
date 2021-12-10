import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torchvision import transforms

from tqdm import tqdm

# System modules
import os
import io

# Our modules
import arch
import data


class Logger:

    def __init__(self, epochs, batches, device):
        self.epochs = epochs
        self.batches = batches
        self.e = 0      # Epoch Count
        self.b = 0      # Batch Count
        self.device = device
        self.base = torch.zeros(epochs, batches, 5).to(device)
        self.data = [self.base.clone()]

    def log(
        self,
        train_loss,
        train_acc,
        test_loss,
        test_acc,
        elapsed_time
    ):
        if self.b == self.batches:
            self.data.append(self.base.clone())
            self.b = 0
            self.e += 1
        self.data[-1][self.e, self.b, :] = torch.tensor([
            train_loss, train_acc, test_loss, test_acc, elapsed_time
        ])
        self.b += 1

    def __getitem__(self, i):
        means = self.data[-1][self.e, 0:self.b, :].mean(dim=0)
        return means

    def __str__(self):
        train_loss, train_acc, test_loss, test_acc, _ = self[-1]
        msg = (
            f'{"":10}{self.e:<8}{self.b:<3} / {self.batches:<6}'
            f'{train_loss:<10.4f}{train_acc:<10.4f}'
            f'{test_loss:<10.4f}{test_acc:<10.4f}'
        )
        return msg


def header(f):
    msg = (
        f'{"":35}{"Train":20}{"Test":20}\n'
        f'{"":10}{"Epoch":8}{"Batch":12}'
        f'{"Loss":10}{"Accuracy":10}'
        f'{"Loss":10}{"Accuracy":10}'
        f'{"Elapsed Time":15}\n'
    )
    f.write(msg)
    print(msg)


def write_it(i, j, size, l, a, test_l, test_a, stream=None):
    msg = (
        f'{"":10}{i:<8}{j:<3} / {size:<6}'
        f'{l:<10.4f}{a:<10.4f}'
        f'{test_l:<10.4f}{test_a:<10.4f}'
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
    print(epochs, size, batch_size)

    # Initialize Logger
    logger = Logger(epochs, size, device)

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

        run_l = 0
        run_a = 0
        t = tqdm(
            dataloader,
            desc=write_it(i, 0, size, 0, 0, 0, 0),
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, (train_dl, test_dl) in enumerate(t):

            (s_in, t_in), _ = train_dl

            train_loss, train_acc = callbacks[0](
                model, (s_in, t_in), loss_fn, batch_size, device, train=True)

            (test_s_in, test_t_in), _ = test_dl

            test_loss, test_acc = callbacks[0](
                model, (test_s_in, test_t_in), loss_fn, batch_size, device, train=False)

            logger.log(train_loss, train_acc, test_loss, test_acc, 1)

            t.set_description(str(logger))

        print(logger)

    torch.save(model.state_dict(), path)


def omniglotCallBack(model, inputs, loss_fn, batch_size, device, train=True):

    if train:
        model.train()
        optim.zero_grad()
    else:
        model.eval()

    outputs = model(inputs[0], inputs[1])

    # Compute Loss
    loss = loss_fn(outputs[1], outputs[0])
    l = loss.item()

    # Compute Accuracy
    pred = outputs[0].argmax(dim=1)
    lab = outputs[1].argmax(dim=1)
    correct = torch.sum(lab == pred).data
    a = (correct / pred.shape[0]).item()

    if train:
        loss.backward()
        # clip_grad_norm_(model.parameters(), 1)
        optim.step()

    return l, a


def test(
        model,
        path,
        dataloader,
        loss_fn,
        callbacks,
        device
):

    # Load Model from state_dict
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()

    batches = len(dataloader)

    logger = Logger(1, batches, device)

    # Test Model on dataloader
    for j, data in enumerate(dataloader):

        test_inputs, _ = data
        print(test_inputs.shape)

        test_loss, test_acc = callbacks[0](
            model, test_inputs, loss_fn, batches, device, train=False)

        logger.log(0, 0, test_loss, test_acc, 0)

    print(logger)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    train_ds = data.OmniglotDataset(shots=1, background=True, device=device)
    test_ds = data.OmniglotDataset(shots=1, background=False, device=device)
    ds = data.Siamese(train_ds, test_ds)
    dl = DataLoader(ds, batch_size=20, shuffle=True, drop_last=True)

    model = arch.MatchingNets(device, 1, 64)
    optim = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    callbacks = [omniglotCallBack]

    train(model, dl, callbacks, optim, loss_fn, 2**13, device)

    # test_dl = DataLoader(test_ds, batch_size=20, shuffle=True, drop_last=True)
    # test(model, 'models/MatchingNets',
    # test_dl, loss_fn, [omniglotCallBack], device)
