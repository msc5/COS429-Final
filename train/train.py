
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from arch.test_conv import StackedRes

from data.dataloader import omniglot_DataLoader


def train(model, dataloader, optim, loss_fn, epochs):
    
    rl = 0
    for i in range(epochs):
        for j, data in enumerate(dataloader, 0):
            inputs, labels = data
            convert = transforms.ToTensor()
            inputs = convert(inputs)
            inputs = torch.unsqueeze(inputs, 0)
            print(inputs.shape)
            optim.zero_grad()
            outputs = model(inputs)
            print(outputs.shape)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            rl += loss.item()
            if j % 2000 == 0:
                print('loss: ', rl / 2000)


if __name__ == '__main__': 
    
    dataloader = omniglot_DataLoader('../omniglot')
    model = StackedRes(1)
    
    optim = optim.Adam(model.parameters(), lr=0.001)
    
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optim, loss_fn, 1)


