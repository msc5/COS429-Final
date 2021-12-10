import torch
import torch.nn as nn

from torchinfo import summary

from .relation_network import ConvBlock, PoolBlock

# This model implements the theory that similarity
# should be derived from all levels of encoder feature maps, not just one


class MultiBlock(nn.Module):

    def __init__(self, op):
        super(MultiBlock, self).__init__()
        self.op = op

    def forward(self, x, y):
        x = self.op(x)
        y = self.op(y)
        return x, y


class Decoder(nn.Module):

    def __init__(self, fo, lo):
        """
        Decoder
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv
            li: Number of input features in linear layer
            lo: Number of output features in linear layer
        """
        super(Decoder, self).__init__()
        self.pool = PoolBlock(2 * fo, fo)
        self.lo = lo

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

    def __init__(self, fi):
        super(CustomNetwork, self).__init__()
        fo = 64
        self.pool = MultiBlock(PoolBlock(fi, fo))
        self.conv = MultiBlock(ConvBlock(fo, fo))
        self.dec = Decoder(fo)

    def forward(self, x, y):
        x, y = self.pool(x, y)
        z = self.dec(x, y)
        x, y = self.conv(x, y)
        w = self.dec(x, y)


if __name__ == '__main__':
    model = CustomNetwork(1)
    summary(model, input_size=[(1, 105, 105), (1, 105, 105)])
