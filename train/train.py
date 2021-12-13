import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader

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

    for i in range(epochs):

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


# def test(
#         model,
#         name,
#         dataloader,
#         loss_fn,
#         callback,
#         device
# ):

#     # Load Model from state_dict
#     dir = os.path.join('saves', name)
#     if not os.path.exists(dir):
#         print(f'{name} does not exist')
#     model_path = os.path.join(dir, 'model')
#     model.load_state_dict(torch.load(model_path))
#     model = model.to(device)

#     batches = len(dataloader)
#     print(batches)

#     log_path = os.path.join(dir, 'log_test')
#     logger = Logger(1, batches, log_path)

#             # Compute predictions
#             if name == "RelationNetwork":
#                 pred = model(sup_set, query_set)
#                 target_labels = torch.eye(num_classes).repeat_interleave(query_num_examples_per_class, dim=0).to(device)
#             elif name == "MatchingNets":
#                 outputs = model(sup_set, query_set)
#                 pred = outputs[1]
#                 target_labels = outputs[0]

#         results = callback(model, data, None, loss_fn, train=False)

#         log = logger.log((0, 0, *results), 1)
#         print(log)

#     print(log)


def omniglotCallBack(
        model,
        inputs,
        optimizer,
        loss_fn,
        device,
        train=True
):

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    (s, t), _ = inputs
    pred = model(s, t)

    k = int(pred.shape[1])
    n = int(pred.shape[0] / k)

    lab = torch.eye(k).repeat_interleave(n, dim=0).to(device)

    # Compute Loss
    loss_t = loss_fn(pred, lab)
    loss = loss_t.item()

    # Compute Accuracy
    correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
    acc = correct / pred.shape[0]

    if train:
        loss_t.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return loss, acc


def imagenetCallBack(
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

    (ss, sl), (ts, tl) = inputs

    ss = torch.tensor(ss)
    sl = torch.tensor(sl)
    ts = torch.tensor(ts)
    tl = torch.tensor(tl)

    k = sl.shape[1]
    n = int(sl.shape[0] / k)
    q = int(tl.shape[0] / n)

    # lab = (
    #     sl.argmax(dim=1).unsqueeze(0).expand(k * n, k)
    #     tl.argmax(dim=1).unsqueeze(1).expand()
    # ).int()
    lab = tl

    pred = model(ss, ts)

    # Compute Loss
    loss_t = loss_fn(pred, lab)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 20
    num_examples_per_class = 1
    # train_dataset = data.OmniglotDataset(background=True, device=device)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # train_ds = data.OmniglotDataset(
    #     shots=num_examples_per_class, background=True, device=device)
    # test_ds = data.OmniglotDataset(
    #     shots=num_examples_per_class, background=False, device=device)

    train_ds = data.ImageNetDataLoader(20, 1, 1, phase='train')
    test_ds = data.ImageNetDataLoader(20, 1, 1, phase='test')
    ds = data.Siamese(train_ds, test_ds)
    dataloader = DataLoader(ds, batch_size=20, shuffle=True, drop_last=True)

    # model = arch.MatchingNets(device, 1, 64)
    # model = arch.RelationNetwork(
    #     1, 64, 128, 64, 64, 20, 1
    # )
    model = arch.CustomNetwork(20, 1, 64, 3, device).to(device)
    model.__name__ = model.__name__ + input('Model Name:\n')
    print(f'Training {model.__name__}')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [40, 250, 1000], gamma=0.5)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.NLLLoss()
    loss_fn = nn.MSELoss()

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
