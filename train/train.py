import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

import os
import time
import shutil

import arch
import data

from .logger import Logger


def train(
        name,
        model,
        dataloader,
        callback,
        optimizer,
        scheduler,
        loss_fn,
        epochs,
        device,
):

    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    print(device)
    print(epochs, num_batches, batch_size)

    # Setup files
    model_type = model.__name__
    type_dir = os.path.join('saves', model_type)
    model_dir = os.path.join(type_dir, name)
    model_path = os.path.join(model_dir, 'model')
    log_path = os.path.join(model_dir, 'log')
    if not os.path.exists('saves'):
        os.makedirs('saves')
    if not os.path.exists(type_dir):
        os.makedirs(type_dir)
    if os.path.exists(model_dir):
        ans = input(
            f'{model_type + ": " + name} has already been trained. Overwrite save files? (y/n)\n',
        )
        if ans == 'y' or ans == 'Y':
            pass
        else:
            return
    else:
        os.makedirs(model_dir)
    config_path = os.path.join(model_dir, 'config.json')
    shutil.copyfile('config.json', config_path)

    # Initialize Logger
    logger = Logger(epochs, num_batches, log_path)

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
        for j, (train_ds, test_ds) in enumerate(t):
            train_results = callback(
                model,
                train_ds,
                optimizer,
                loss_fn,
                device,
                train=True
            )
            with torch.no_grad():
                test_results = callback(
                    model,
                    test_ds,
                    None,
                    loss_fn,
                    device,
                    train=False
                )
            toc = time.perf_counter()
            log = logger.log((*train_results, *test_results), toc - tic)
            t.set_description(log)

        print(log)
        torch.save(model.state_dict(), model_path)
        scheduler.step()


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

    (s, t), classes = inputs

    pred = model(s, t)

    k = int(pred.shape[1])
    m = int(pred.shape[0] / k)

    lab = torch.eye(k).repeat_interleave(m, dim=0).to(device)

    # Compute Loss
    loss_t = loss_fn(pred, lab)
    loss = loss_t.item()

    # Compute Accuracy
    correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
    acc = correct / pred.shape[0]

    if train:
        optimizer.zero_grad()
        loss_t.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return loss, acc


def build_model(model_config):
    if model_config['arch'] == 'RelationNetwork':
        in_feat_rel = 64 if config['dataset'] == 'Omniglot' else 576
        model = arch.RelationNetwork(64, 64, in_feat_rel, k, n, m)
    elif model_config['arch'] == 'MatchingNetwork':
        model = arch.MatchingNets(device, 64, 64)
    elif model_config['arch'] == 'CustomNetwork':
        model = arch.CustomNetwork(3, 28, 64, 16, device)


if __name__ == '__main__':

    config = json.load(open('config.json'))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = config['models']
    experiments = config['experiments']

    for n, m in models.items():
        for e, p in experiments.items():
            train(
                n,
                build_model(m),
                data.fewshot.emitFewShotLoader(
                    p['dataset'],
                    device,
                    p['todo'],
                    p['batch_size'],
                    p['k'],
                    p['n'],
                    p['m']
                ),
                optimizer,
                scheduler,
                m['loss_fn'],
                p['epochs'],
                device
            )

    # # Task setup
    # k = config['k']           # Number of classes
    # n = config['n']           # Number of examples per support class
    # m = config['m']           # Number of examples per query class

    # batch_size = config['batch_size']

    # dataloader = data.emitFewShotLoader(config['dataset'], batch_size, k, n, m)

    # filters_in = 64

    # if config['arch'] == 'RelationNetwork':
    #     in_feat_rel = 64 if config['dataset'] == 'Omniglot' else 576
    #     model = arch.RelationNetwork(filters_in, 64, in_feat_rel, k, n, m)
    # elif config['arch'] == 'MatchingNetwork':
    #     model = arch.MatchingNets(device, filters_in, 64)
    # elif config['arch'] == 'CustomNetwork':
    #     model = arch.CustomNetwork(3, 28, filters_in, 16, k, n, m, device)

    # if config['loss_fn'] == 'MSE':
    #     loss_fn = nn.MSELoss()
    # elif config['loss_fn'] == 'NLL':
    #     loss_fn = nn.NLLLoss()
    # elif config['loss_fn'] == 'CrossEntropy':
    #     loss_fn = nn.CrossEntropy()

    # model_name = config['name']
    # model_arch = config['arch']
    # lr = config['learning_rate']
    # schedule = config['schedule']
    # epochs = config['epochs']

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, schedule, gamma=0.5)

    # print(
    #     f'Training {model_arch} {model_name} on {k}-way {n}-shot {m}-query-shot')

    # if config['train']:
    #     train(
    #         model_name,
    #         model,
    #         dataloader,
    #         callback,
    #         optimizer,
    #         scheduler,
    #         loss_fn,
    #         epochs,
    #         device
    #     )
    # if config['test']:
    #     test(
    #         model_name,
    #         model,
    #         test_dataloader,
    #         callback,
    #         loss_fn,
    #         device,
    #     )
