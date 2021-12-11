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

    def __init__(self, epochs, batches, stream=None):
        self.epochs = epochs        # Total Epochs
        self.batches = batches      # Total Batches
        self.e = 0                  # Current Epoch
        self.b = 0                  # Current Batch
        self.stream = stream
        self.data = torch.zeros(epochs, batches, 5)

    def log(
        self,
        train,
        test,
        elapsed_time
    ):
        train_loss, train_acc = train
        test_loss, test_acc = test
        # if self.b == self.batches:
        #     self.b = 0
        #     self.e += 1
        data = torch.tensor([
            train_loss,
            train_acc / self.batches,
            test_loss,
            test_acc / self.batches,
            elapsed_time
        ])
        self.data[self.e, self.b, :] = data
        if self.stream is not None:
            self.stream.write(self.msg(data))
        self.step()
        # self.b += 1

    def step(self):
        # print(self.b)
        self.b += 1
        if self.b == self.batches:
            self.b = 0
            self.e += 1

    def msg(self, data):
        train_loss, train_acc, test_loss, test_acc, _ = data
        msg = (
            f'{"":10}{self.e:<8}{self.b:<3} / {self.batches:<6}'
            f'{train_loss:<10.4f}{train_acc:<10.4f}'
            f'{test_loss:<10.4f}{test_acc:<10.4f}'
        )
        return msg

    def __getitem__(self, i):
        means = self.data[self.e, 0:self.b, :].mean(dim=0)
        return means

    def __str__(self):
        return self.msg(self[-1])

    def header(self):
        msg = (
            f'{"":35}{"Train":20}{"Test":20}\n'
            f'{"":10}{"Epoch":8}{"Batch":12}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Elapsed Time":15}\n'
        )
        if self.stream is not None:
            self.stream.write(self.msg(data))
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
    path = os.path.join('models', name + '2')

    # Write file stream
    if not os.path.exists('logs'):
        os.makedirs('logs')
    f = io.open(os.path.join('logs', name + '_log'), 'a')

    # Initialize Logger
    logger = Logger(epochs, batches, stream=f)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.train()
    model.zero_grad()

    # Print header
    print(logger.header())

    for i in range(1, epochs):

        run_l = 0
        run_a = 0
        t = tqdm(
            dataloader,
            desc=str(logger),
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, data in enumerate(t):

            train = callbacks[0](model, data[0], loss_fn, train=True)
            test = callbacks[0](model, data[1], loss_fn, train=False)

            logger.log(train, test, 1)

            t.set_description(str(logger))

        print(logger)
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

    logger = Logger(1, batches, device)

    # Test Model on dataloader
    for j, data in enumerate(dataloader):

        test = callbacks[0](model, data, loss_fn, train=False)

        logger.log((0, 0), test, 0)

    print(logger)


def omniglotCallBack(model, inputs, loss_fn, train=True):

    if train:
        model.train()
        optim.zero_grad()
    else:
        model.eval()

    (s, q), _ = inputs
    outputs = model(s, q)

    # Compute Loss
    loss_t = loss_fn(outputs[1], outputs[0])
    loss = loss_t.item()

    # Compute Accuracy
    pred = outputs[0].argmax(dim=1)
    lab = outputs[1].argmax(dim=1)
    correct = torch.sum(lab == pred).data

    if train:
        loss_t.backward()
        # clip_grad_norm_(model.parameters(), 1)
        optim.step()

    return loss, correct


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

    # train(model, dl, callbacks, optim, loss_fn, 2**13, device)

    test_dl = DataLoader(test_ds, batch_size=20,
                         shuffle=True, drop_last=True)
    test(model, 'models/MatchingNets',
         test_dl, loss_fn, [omniglotCallBack], device)
