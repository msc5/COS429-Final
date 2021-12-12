import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# System modules
import os

# Our modules
import arch
import data

# Train the model


def train(dataloader, model, loss_fn, optimizer, query_num_examples_per_class, num_classes, device):

    model = model.to(device)
    model.train()

    # Name of model and save location
    name = model.__name__
    path = os.path.join('models', name)

    num_batches = len(dataloader)

    # zero gradient per batch or per epoch? usually, zero gradient per batch
    # optimizer.zero_grad()
    for batch, data in enumerate(dataloader):
        (sup_set, query_set), _ = data[0]

        sup_set, query_set = sup_set.to(device), query_set.to(device)

        # Compute predictions
        if name == "RelationNetwork":
            pred = model(sup_set, query_set)
            target_labels = torch.eye(num_classes).repeat_interleave(
                query_num_examples_per_class, dim=0).to(device)
        elif name == "MatchingNets":
            outputs = model(sup_set, query_set)
            pred = outputs[1]
            target_labels = outputs[0]

        loss = loss_fn(pred, target_labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if (batch + 1) % 6 == 0:
            loss = loss.item()
            # Compute Accuracy
            pred = pred.argmax(dim=1)
            target_labels = target_labels.argmax(dim=1)
            correct = torch.sum(target_labels == pred).data
            accuracy = correct/pred.shape[0]

            print(
                f"Batch {batch + 1}/{num_batches} - Training Loss: {loss:>4f}  Training Accuracy: {(100*accuracy):>0.1f}%")

    torch.save(model.state_dict(), path)


# Test the model
def test(dataloader, model, loss_fn, query_num_examples_per_class, num_classes):
    # size is no longer total guess bc we drop the last batch as it is not batch_size=20 long; must calculate total guesses
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    name = model.__name__

    model.eval()

    total_guesses = 0
    test_loss, correct = 0, 0

    with torch.no_grad():
        for test_data in dataloader:
            (sup_set, query_set), _ = test_data

            sup_set, query_set = sup_set.to(device), query_set.to(device)

            # Compute predictions
            if name == "RelationNetwork":
                pred = model(sup_set, query_set)
                target_labels = torch.eye(num_classes).repeat_interleave(
                    query_num_examples_per_class, dim=0).to(device)
            elif name == "MatchingNets":
                outputs = model(sup_set, query_set)
                pred = outputs[1]
                target_labels = outputs[0]

            test_loss += loss_fn(pred, target_labels).item()

            # Compute Accuracy
            pred = pred.argmax(dim=1)
            target_labels = target_labels.argmax(dim=1)
            correct += torch.sum(target_labels == pred).data.type(torch.float)
            total_guesses += pred.shape[0]

    test_loss /= num_batches
    correct /= total_guesses
    print(
        f"Test Error: \n  Avg Test Loss: {test_loss:>8f},   Test Accuracy: {(100*correct):>0.1f}% \n")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    num_classes = 20
    num_examples_per_class = 1
    # train_dataset = data.OmniglotDataset(background=True, device=device)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_ds = data.OmniglotDataset(
        shots=num_examples_per_class, background=True, device=device)
    test_ds = data.OmniglotDataset(
        shots=num_examples_per_class, background=False, device=device)
    ds = data.Siamese(train_ds, test_ds)
    # batch size is the number of classes in the support set
    dl = DataLoader(ds, batch_size=num_classes, shuffle=True, drop_last=True)

    test_dataloader = DataLoader(
        test_ds, batch_size=num_classes, shuffle=False, drop_last=True)

    # model = arch.MatchingNets(device, 1, 64).to(device)
    model = arch.RelationNetwork(
        device, 1, 64, 128, 64, 64, num_classes, num_examples_per_class)

    if model.__name__ == "RelationNetwork":
        optimizer = optim.Adam(model.parameters(), lr=10e-3)
        loss_fn = nn.MSELoss()
    elif model.__name__ == "MatchingNets":
        # was lr=0.0005
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [100, 200, 300, 400])
        loss_fn = nn.CrossEntropyLoss()

    # Train model for 10 epochs
    epochs = 1000

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dl, model, loss_fn, optimizer,
              num_examples_per_class, num_classes, device)
        print()
        test(test_dataloader, model, loss_fn,
             num_examples_per_class, num_classes)
        # scheduler.step()

    print("Done!")
