import torch
import torch.nn as nn

from torchinfo import summary

from .relation_network import ConvBlock, PoolBlock


class MultiBlock(nn.Module):

    def __init__(self, op):
        super(MultiBlock, self).__init__()
        self.op = op

    def forward(self, x, y):
        x = self.op(x)
        y = self.op(y)
        return x, y


class Meta(nn.Module):

    def __init__(self, fi, fo):
        super(Meta, self).__init__()
        self.seq = nn.Sequential(
            ConvBlock(fi, fo),
            ConvBlock(fo, fo)
        )


class Decoder(nn.Module):

    def __init__(self, fo, lo):
        """
        Decoder
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv
        """
        super(Decoder, self).__init__()
        self.pool = PoolBlock(2 * fo, fo)

    def forward(self, x, y):
        z = torch.cat((x, y), 1)
        z = self.pool(z)
        z = z.view(z.shape[0], -1)
        li = z.shape[1]
        z = nn.Sequential(
            nn.Linear(li, self.lo),
            nn.ReLU(),
            nn.Linear(self.lo, self.lo),
            nn.Sigmoid()
        )(z)
        return z


class CustomNetwork(nn.Module):

    def __init__(self, fi, fo):
        super(CustomNetwork, self).__init__()
        self.list = nn.ModuleList([
            MultiBlock(ConvBlock(fi, fo)),
            MultiBlock(ConvBlock(fo, fo)),
            MultiBlock(ConvBlock(fo, fo)),
        ])

    def forward(self, x, y):
        k, n, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        y = y.view(-1, c, h, w)
        maps = torch.empty()
        for i, m in enumerate(self.list):
            x, y = m(x, y)
            z = torch.cat((x, y), dim=1)
            maps.append(z)
        maps = torch.cat(maps, dim=1)
        print(maps.shape)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CustomNetwork(1, 64).to(device)
    summary(model, input_size=[(20, 5, 1, 105, 105), (20, 5, 1, 105, 105)])
