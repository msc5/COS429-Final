import torch
import torch.nn as nn

from torchsummary import summary


# The following Classes are implemented as described
# in the paper: Learning to Compare: Relation Network for Few-Shot Learning


class ConvBlock(nn.Module):

    def __init__(self, fi, fo):
        """
        Convolutional Block
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv (64)
        """
        super(ConvBlock, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(fi, fo, 3),
            nn.BatchNorm2d(fo),
            nn.ReLU()
        )

    def forward(self, x):
        return self.stack(x)


class PoolBlock(nn.Module):

    def __init__(self, fi, fo):
        """
        Pool Block
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv (64)
        """
        super(PoolBlock, self).__init__()
        self.stack = nn.Sequential(
            ConvBlock(fi, fo),
            nn.MaxPool2d(2),
            ConvBlock(fo, fo),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.stack(x)


class Encoder(nn.Module):

    def __init__(self, fi, fo):
        """
        Encoder
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv
        """
        super(Encoder, self).__init__()
        self.stack = nn.Sequential(
            PoolBlock(fi, fo),
            ConvBlock(fo, fo),
            ConvBlock(fo, fo)
        )

    def forward(self, x):
        return self.stack(x)


class Decoder(nn.Module):

    def __init__(self, fi, fo, li, lo):
        """
        Decoder
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv
            li: Number of input features in linear layer
            lo: Number of output features in linear layer
        """
        super(Decoder, self).__init__()
        self.pool = PoolBlock(fi, fo)
        self.stack = nn.Sequential(
            nn.Linear(li, lo),
            nn.ReLU(),
            nn.Linear(lo, lo),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.stack(x)
        return x


class RelationNetwork(nn.Module):

    def __init__(self, H):
        """
        Decoder
        Arguments:
        """
        super(RelationNetwork, self).__init__()
        fi = 3
        fo = 64
        li = fo * 3**2
        lo = 8
        self.enc = Encoder(fi, fo)
        self.dec = Decoder(2 * fo, fo, li, lo)

    def forward(self, x, y):
        x = self.enc(x)
        y = self.enc(y)
        z = torch.cat((x, y), 1)
        z = self.dec(z)
        return z


if __name__ == '__main__':
    # model = Encoder(3, 64)
    model = RelationNetwork(105)
    summary(model, input_size=[(3, 105, 105), (3, 105, 105)])