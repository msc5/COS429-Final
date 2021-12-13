import torch
import torch.nn as nn

from torchinfo import summary


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
        res = x
        x = self.seq(x) + res
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
            ConvBlock(fi, fo),
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


class Decoder(nn.Module):

    def __init__(self, li, lm, lo):
        super(Decoder, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(li, lm),
            # nn.ReLU(),
            nn.Linear(lm, lo),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class CustomNetwork(nn.Module):

    def __init__(self,  k, fi, fo, l, device='cpu'):
        super(CustomNetwork, self).__init__()
        self.device = device
        self.__name__ = 'CustomNetwork'
        self.fo = fo
        self.li = fo * 2 * l
        self.pool = MultiBlock(
            PoolBlock(fi, fo),
            PoolBlock(fi, fo)
        )
        self.list = nn.ModuleList([
            MultiBlock(
                ConvBlock(fo, fo),
                ConvBlock(fo, fo)
            ) for _ in range(l)
        ])
        # for p in self.pool.parameters():
        #     p.requires_grad = False
        # for p in self.list.parameters():
        #     p.requires_grad = False
        # self.meta = Meta(fo * 2 * l, fo * 2 * l)
        self.meta = Meta(self.li, self.li)
        self.dec = Decoder(self.li, 200, 1)

    def forward(self, s, t):
        k, n, c, h, w = s.shape
        q, m, _, _, _ = t.shape
        s = s.view(-1, c, h, w)
        t = t.view(-1, c, h, w)
        s, t = self.pool(s, t)
        x = []
        # print(s.shape, t.shape)
        for i, layer in enumerate(self.list):
            s, t = layer(s, t)
            z = torch.cat((
                t.unsqueeze(1).expand((k * n, q * m, -1, -1, -1)),
                s.unsqueeze(0).expand((k * n, q * m, -1, -1, -1))
            ), dim=2)
            x.append(z)
        x = torch.cat(x, dim=2)
        x = x.view(-1, self.li, int(h / 2 / 2), int(w / 2 / 2))
        # print(x.max(), x.min())
        x = self.meta(x)
        # print(x.shape)
        # print(x.max(), x.min())
        x = x.view(k * n * q * m, -1)
        # print(x.max(), x.min())
        x = self.dec(x)
        # print(x.max(), x.min())
        # x = x.view(q * m, k * n)
        # print(x.max(), x.min())
        # x = x.sum(dim=1)
        # print(x.max(), x.min())
        x = x.view(q * m, k * n).softmax(dim=1)
        # print(x.max(), x.min())
        return x


if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CustomNetwork(20, 1, 64, 3, device).to(device)
    summary(model, input_size=[(20, 1, 1, 28, 28), (20, 1, 1, 28, 28)])
