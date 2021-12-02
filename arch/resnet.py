import torch
import torch.nn as nn

from torchsummary import summary


class ConvBlock(nn.Module):

    def __init__(self, fi, fo, fs, skip):
        super(ConvBlock, self).__init__()
        self.skip = skip
        self.stack = nn.Sequential(
            nn.Conv2d(fi, fo, fs, padding='same'),
            nn.ReLU(),
            nn.Conv2d(fo, fo, fs, padding='same')
        )

    def forward(self, x):
        res = x
        x = self.stack(x)
        if not self.skip:
            x = x + res
        return x


class ConvStack(nn.Module):

    def __init__(self, fi, fo, nb, skip):
        super(ConvStack, self).__init__()
        self.ent = ConvBlock(fi, fo, 3, skip)
        list = nn.ModuleList([ConvBlock(fo, fo, 3, False) for _ in range(nb)])
        self.stack = nn.Sequential(*list)

    def forward(self, x):
        x = self.ent(x)
        x = self.stack(x)
        return x


class ResNet(nn.Module):

    def __init__(self, fi, hi, wi, lo):
        super(ResNet, self).__init__()
        fo = 64
        # ResNet34
        self.stack = nn.Sequential(
            nn.Conv2d(fi, fo // 2, 7, padding='same'),
            nn.MaxPool2d(2),
            ConvStack(fo // 2, fo * 2**0, 3, True),
            ConvStack(fo * 2**0, fo * 2**1, 4, True),
            ConvStack(fo * 2**1, fo * 2**2, 6, True),
            ConvStack(fo * 2**2, fo * 2**3, 3, True),
            nn.AvgPool2d(2)
        )
        ho = hi // 2 // 2
        wo = wi // 2 // 2
        self.li = fo * 2**3 * ho * wo
        self.dec = nn.Linear(self.li, lo)

    def forward(self, x):
        x = self.stack(x)
        x = x.view(x.shape[0], -1)
        x = self.dec(x)
        return x


if __name__ == '__main__':
    hi = 105
    wi = 105
    model = ResNet(3, hi, wi, 1000)
    summary(model, input_size=(3, hi, wi))
