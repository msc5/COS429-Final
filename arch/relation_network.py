import torch
import torch.nn as nn

from torchinfo import summary


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
        self.seq = nn.Sequential(
            nn.Conv2d(fi, fo, 3, padding='same'),
            nn.BatchNorm2d(fo),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


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
            ConvBlock(fi, fo),
            nn.MaxPool2d(2),
            ConvBlock(fo, fo),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.seq(x)


class Encoder(nn.Module):

    def __init__(self, fi, fo):
        """
        Encoder
        Arguments:
            fi: Number of channels in input
            fo: Number of filters in conv
        """
        super(Encoder, self).__init__()
        self.seq = nn.Sequential(
            PoolBlock(fi, fo),
            ConvBlock(fo, fo),
            ConvBlock(fo, fo)
        )

    def forward(self, x):
        return self.seq(x)


class Decoder(nn.Module):

    def __init__(self, fo, li, lo):
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
        self.seq = nn.Sequential(
            nn.Linear(li, lo),
            nn.ReLU(),
            nn.Linear(lo, lo),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        z = torch.cat((x, y), 1)
        z = self.pool(z)
        z = z.view(z.shape[0], -1)
        z = self.seq(z)
        return z


class RelationNetwork(nn.Module):

    def __init__(self, fi):
        """
        Relation Network
        Arguments:
        """
        super(RelationNetwork, self).__init__()
        fo = 64
        li = fo * 3**2
        lo = 8
        self.enc = Encoder(fi, fo)
        self.dec = Decoder(fo, li, lo)

    def forward(self, x, y):
        x = self.enc(x)
        y = self.enc(y)
        z = self.dec(x, y)
        return z


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RelationNetwork(1).to(device)
    summary(model, input_size=[(1, 105, 105), (1, 105, 105)])
