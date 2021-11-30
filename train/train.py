import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

from arch.test_conv import ResNetwork

from data.dataloader import omniglot_DataLoader


def train(model, dataloader, optim, loss_fn, epochs):
    
    print(len(dataloader))
    print(enumerate(dataloader))
    it = 1
    rl = 0
    for i in range(epochs):
        for (j, data) in enumerate(dataloader):
            inputs, labels = data
            #print(inputs.shape, labels.shape)
            #print(labels)
            convert = transforms.ToTensor()
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            rl += loss.item()
            if j % it== 0:
                print('it: ', j, ' loss: ', loss.item())


if __name__ == '__main__': 
    
    dataloader = omniglot_DataLoader('../omniglot')
    model = ResNetwork(1, 105, 1623)
    
    optim = optim.Adam(model.parameters(), lr=0.001)
    
    loss_fn = nn.CrossEntropyLoss()

    train(model, dataloader, optim, loss_fn, 1)
