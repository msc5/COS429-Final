import torch
import torch.nn as nn

from torchsummary import summary

from relation_network import ConvBlock, PoolBlock, Decoder

# This model implements the theory that similarity
# should be derived from all levels of encoder feature maps, not just one


class MultiBlock(nn.Module):

    def __init__(self, op, fi, fo):
        super(MultiBlock, self).__init__()
        self.op = op

    def forward(self, x, y):
        x = self.op(x)
        y = self.op(y)
        return x, y


class CustomNetwork(nn.Module):

    def __init__(self, fi, fo):
        super(CustomNetwork, self).__init__()
        self.pool = MultiBlock(PoolBlock(fi, fo), fi, fo)
        self.conv = MultiBlock(ConvBlock(fi, fo), fi, fo)
        self.dec = Decoder(2 * fi, fo)

    def forward(self, x, y):
        x, y = self.pool(x, y)
        z = 1
        x, y = self.conv(x, y)
