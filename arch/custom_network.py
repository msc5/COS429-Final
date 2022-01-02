import torch
import torch.nn as nn

from torchinfo import summary


class ConvBlock(nn.Module):

    def __init__(self, fi, fo, res=True):
        """
        Convolutional Block
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv (64)
        """
        super(ConvBlock, self).__init__()
        self.res = res
        self.seq = nn.Sequential(
            nn.Conv2d(fi, fo, 3, padding='same'),
            nn.BatchNorm2d(fo),
            nn.ReLU()
        )

    def forward(self, x):
        res = x
        x = self.seq(x)
        if self.res:
            x = x + res
        return x


class PoolBlock(nn.Module):

    def __init__(self, fi, fo):
        """
        Pool Block
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv (64)
        """
        super(PoolBlock, self).__init__()
        self.seq = nn.Sequential(
            ConvBlock(fi, fo, res=False),
            nn.MaxPool2d(2),
            ConvBlock(fo, fo),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.seq(x)


class MultiBlock(nn.Module):

    def __init__(self, f, g):
        super(MultiBlock, self).__init__()
        self.f = f
        self.g = g

    def forward(self, x, y):
        x = self.f(x)
        y = self.g(y)
        return x, y


class Meta(nn.Module):

    def __init__(self, fi, fo):
        super(Meta, self).__init__()
        self.seq = nn.Sequential(
            PoolBlock(fi, fo),
            # PoolBlock(fo, fo),
            ConvBlock(fo, fo),
            ConvBlock(fo, fo),
            ConvBlock(fo, fo)
        )

    def forward(self, x):
        return self.seq(x)


class Distance(nn.Module):

    def forward(x):
        pass


class CustomNetwork(nn.Module):

    def __init__(self, layers, fi, fo, device='cpu'):
        super(CustomNetwork, self).__init__()
        self.device = device
        self.__name__ = 'CustomNetwork'
        self.fi = fi
        self.fo = fo
        self.pool = MultiBlock(
            PoolBlock(fi, fo),
            PoolBlock(fi, fo)
        )
        self.list = nn.ModuleList([
            MultiBlock(
                ConvBlock(fo, fo),
                ConvBlock(fo, fo)
            ) for _ in range(layers)
        ])
        self.meta = Meta(self.fo, self.fo)

    def forward(self, s, t):
        k, n, c, h, w = s.shape
        q, m, _, _, _ = t.shape
        s = s.view(-1, c, h, w)
        t = t.view(-1, c, h, w)
        s, t = self.pool(s, t)
        x = []
        for i, layer in enumerate(self.list):
            s, t = layer(s, t)
            z = torch.cat((
                t.unsqueeze(1).expand(q * m, k * n, -1, -1, -1),
                s.unsqueeze(0).expand(q * m, k * n, -1, -1, -1)
            ), dim=2)
            x.append(z)
        x = torch.cat(x, dim=2)
        print(x.shape)
        x = x.flatten(start_dim=0, end_dim=1)
        x = self.meta(x)
        print(x.shape)
        x = x.view(q * m, k * n, -1)
        x = x.view(q * m * k, self.li * n)
        # print(x.shape)
        x = self.dec(x)
        x = x.view(q * m, k).softmax(dim=1)
        # print(x.shape)
        return x


if __name__ == '__main__':

    k = 20
    n = 1
    m = 1
    size = 84
    channels = 3
    meta_layers = 3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CustomNetwork(meta_layers, channels, 16, device).to(device)
    summary(model, input_size=[
        (k, n, channels, size, size),
        (k, m, channels, size, size)
    ], device=device)
