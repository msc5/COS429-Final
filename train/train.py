
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from arch.test_conv import ResNetwork

from data.dataloader import omniglot_DataLoader


def train(model, dataloader, optim, loss_fn, epochs):
    
    it = 1
    model.train()
    for i in range(epochs):
        for (j, data) in enumerate(dataloader):
            inputs, labels = data
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            if j % it== 0:
                print('it: ', j, ' loss: ', loss.item())


if __name__ == '__main__': 
    
    dataloader = omniglot_DataLoader('../omniglot')
    model = ResNetwork(1, 105, 1623)
    
    optim = optim.Adam(model.parameters(), lr=0.001)
    
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optim, loss_fn, 1)
