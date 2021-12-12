import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torchvision import transforms

from tqdm import tqdm

import os
import io
import time

# Our modules
import arch
import data


class Logger:

    def __init__(self, epochs, batches, path=None):
        self.epochs = epochs        # Total Epochs
        self.batches = batches      # Total Batches
        self.e = 0                  # Current Epoch
        self.b = 0                  # Current Batch
        self.path = path
        self.data = torch.zeros(epochs, batches, 5)

    def log(
        self,
        results,
        elapsed_time
    ):
        data = torch.tensor((*results, elapsed_time))
        self.data[self.e, self.b, :] = data
        means = self.data[self.e, 0:self.b + 1, :].mean(dim=0)
        msg = self.msg(means)
        self.b += 1
        if self.b == self.batches:
            self.b = 0
            self.e += 1
            if self.path is not None:
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
        callback,
        optimizer,
        scheduler,
        loss_fn,
        epochs,
        device,
):

    batches = len(dataloader)
    batch_size = dataloader.batch_size

    print(device)
    print(epochs, batches, batch_size)

    # Setup files
    name = model.__name__
    dir = os.path.join('saves', name)
    model_path = os.path.join(dir, 'model')
    log_path = os.path.join(dir, 'log')
    if not os.path.exists('saves'):
        os.makedirs('saves')
    if os.path.exists(dir):
        ans = input(
            f'{name} has already been trained. Overwrite save files? (y/n)\n',
        )
        if ans == 'y' or ans == 'Y':
            pass
        else:
            return
    else:
        os.makedirs(dir)

    # Initialize Logger
    logger = Logger(epochs, batches, log_path)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.zero_grad()

    # Print header
    print(logger.header())
    tic = time.perf_counter()

    for i in range(1, epochs):

        t = tqdm(
            dataloader,
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, (train, test) in enumerate(t):
            results = (
                *callback(model, train, optimizer, loss_fn, train=True),
                *callback(model, test, None, loss_fn, train=False)
            )
            toc = time.perf_counter()
            log = logger.log(results, toc - tic)
            t.set_description(log)

        print(log)
        torch.save(model.state_dict(), model_path)
        scheduler.step()


def test(
        model,
        name,
        dataloader,
        loss_fn,
        callback,
        device
):

    # Load Model from state_dict
    dir = os.path.join('saves', name)
    if not os.path.exists(dir):
        print(f'{name} does not exist')
    model_path = os.path.join(dir, 'model')
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    batches = len(dataloader)
    print(batches)

    log_path = os.path.join(dir, 'log_test')
    logger = Logger(1, batches, log_path)

    # Test Model on dataloader
    for j, data in enumerate(dataloader):

        results = callback(model, data, None, loss_fn, train=False)

        log = logger.log((0, 0, *results), 1)
        print(log)

    print(log)


def omniglotCallBack(
        model,
        inputs,
        optimizer,
        loss_fn,
        train=True
):

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    (s, q), _ = inputs
    (pred, lab) = model(s, q)

    # Compute Loss
    loss_t = loss_fn(pred, lab.argmax(dim=1))
    loss = loss_t.item()

    # Compute Accuracy
    correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
    acc = correct / pred.shape[0]

    if train:
        loss_t.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return loss, acc


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    train_ds = data.OmniglotDataset(
        shots=1, background=True, device=device)
    test_ds = data.OmniglotDataset(
        shots=1, background=False, device=device)
    ds = data.Siamese(train_ds, test_ds)
    dataloader = DataLoader(ds, batch_size=20, shuffle=True, drop_last=True)

    model = arch.MatchingNets(device, 1, 64)
    model.__name__ = model.__name__ + input('Model Name:\n')
    print(f'Training {model.__name__}')
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 250, 1000])
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.NLLLoss()

    print('=' * 120)

    train(
        model,
        dataloader,
        omniglotCallBack,
        optimizer,
        scheduler,
        loss_fn,
        2**13,
        device
    )

    # test_dl = DataLoader(test_ds, batch_size=20,
    #                      shuffle=True, drop_last=True)
    # test(model, model.__name__,
    #      test_dl, loss_fn, omniglotCallBack, device)
