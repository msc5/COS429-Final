import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

import os

# Our modules
import arch
import data


def train(
        model,
        dataloader, 
        optim, 
        loss_fn, 
        epochs,
        device  
        ):
    
    size = len(dataloader)
    batch_size = dataloader.batch_size

    # Set name and save location for model
    name = model.__name__
    path = os.path.join('models', name)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.train()
   
    for i in range(epochs):

        print('epoch: ', i)

        for (j, data) in enumerate(dataloader):
            
            inputs, labels = data
            optim.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optim.step()
            
            # Print details 4 times per epoch
            if j % (size // 4) == 0:
                n_seen = j * batch_size
                n_tot = size * batch_size
                print(f'{"iter:":10}{j:<10}{"loss:":<10}{loss.item():.6f}')

    torch.save(model.state_dict(), path)


def test(
        arch,
        path
        ):

    model = arch()
    model.load_state_dict(torch.load(path))
    model.eval()
    

if __name__ == '__main__': 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = data.omniglot_DataLoader('../omniglot')
    model = arch.ResNetwork(1, 105, 1623)
    optim = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    train(model, dataloader, optim, loss_fn, 10, device)
