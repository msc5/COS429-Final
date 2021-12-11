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

    def __init__(self, epochs, batches, path):
        self.epochs = epochs        # Total Epochs
        self.batches = batches      # Total Batches
        self.e = 0                  # Current Epoch
        self.b = 0                  # Current Batch
        self.data = torch.zeros(epochs, batches, 5)
        self.path = path

    def log(
        self,
        results,
        elapsed_time
    ):
        # train_loss, train_acc = train
        # test_loss, test_acc = test
        data = torch.tensor((*results, elapsed_time))
        self.data[self.e, self.b, :] = data
        means = self.data[self.e, 0:self.b + 1, :].mean(dim=0)
        msg = self.msg(means)
        self.b += 1
        if self.b == self.batches:
            self.b = 0
            self.e += 1
            torch.save(self.data, self.path)
        return msg

    def msg(self, data):
        train_loss, train_acc, test_loss, test_acc, _ = data
        msg = (
            f'{"":10}{self.e:<8}{self.b + 1:<3} / {self.batches:<6}'
            f'{train_loss:<10.4f}{train_acc:<10.4f}'
            f'{test_loss:<10.4f}{test_acc:<10.4f}'
        )
        return msg

    # def __getitem__(self, i):
    #     rad = self.b if self.b != self.batches else self.batches - 1
    #     means = self.data[self.e, 0:rad, :].mean(dim=0)
    #     return means

    # def __str__(self):
    #     return self.msg(self[-1])

    def header(self):
        msg = (
            f'{"":35}{"Train":20}{"Test":20}\n'
            f'{"":10}{"Epoch":8}{"Batch":12}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Elapsed Time":15}\n'
        )
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

    batches = len(dataloader)
    batch_size = dataloader.batch_size

    print(device)
    print(epochs, batches, batch_size)

    # Name of model and save location
    name = model.__name__
    path = os.path.join('models', name + '3')
    if not os.path.exists('models'):
        os.makedirs('models')

    # Log save location
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_path = os.path.join('logs', name + '_log')

    # Initialize Logger
    logger = Logger(epochs, batches, log_path)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.train()
    model.zero_grad()

    # Print header
    print(logger.header())

    for i in range(1, epochs):

        t = tqdm(
            dataloader,
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, (train, test) in enumerate(t):

            results = (
                *callbacks[0](model, data[0], loss_fn, train=True),
                *callbacks[0](model, data[1], loss_fn, train=False)
            )

            log = logger.log(results, 1)

            t.set_description(log)

        print(log)
        torch.save(model.state_dict(), path)


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
    print(batches)

    logger = Logger(1, batches)

    # Test Model on dataloader
    for j, data in enumerate(dataloader):

        test_res = callbacks[0](model, data, loss_fn, train=False)

        logger.log((0, 0), test_res, 0)
        print(logger)

    print(logger)


def omniglotCallBack(model, inputs, loss_fn, train=True):

    if train:
        model.train()
        optim.zero_grad()
    else:
        model.eval()

    (s, q), _ = inputs
    (pred, lab) = model(s, q)

    # Compute Loss
    loss_t = loss_fn(pred, lab)
    loss = loss_t.item()

    # Compute Accuracy
    correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
    acc = correct / pred.shape[0]

    if train:
        loss_t.backward()
        clip_grad_norm_(model.parameters(), 1)
        optim.step()

    return loss, acc


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    train_ds = data.OmniglotDataset(
        shots=1, background=True, device=device)
    test_ds = data.OmniglotDataset(
        shots=1, background=False, device=device)
    ds = data.Siamese(train_ds, test_ds)
    dl = DataLoader(ds, batch_size=20, shuffle=True, drop_last=True)

    model = arch.MatchingNets(device, 1, 64)
    optim = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    callbacks = [omniglotCallBack]

    train(
        model,
        dl,
        callbacks,
        optim,
        loss_fn,
        2**13,
        device
    )

    # test_dl = DataLoader(test_ds, batch_size=20,
    #                      shuffle=True, drop_last=True)
    # test(model, 'models/MatchingNets',
    #      test_dl, loss_fn, [omniglotCallBack], device)
