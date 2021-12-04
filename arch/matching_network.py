
import torch
import torch.nn as nn

from torchinfo import summary


class Conv(nn.Module):

    def __init__(self, fi, fo):
        super(Conv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(fi, fo, 3, padding='same'),
            nn.BatchNorm2d(fo),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.seq(x)


class Embed(nn.Module):

    def __init__(self, fi, fo):
        super(Embed, self).__init__()
        self.fo = fo
        self.seq = nn.Sequential(
            Conv(fi, fo),
            Conv(fo, fo),
            Conv(fo, fo),
            Conv(fo, fo),
        )

    def forward(self, x):
        x = self.seq(x)
        x = x.view(-1, self.fo)
        return x


class Classifier(nn.Module):

    def __init__(self, li, lo):
        super(Classifier, self).__init__()
        self.lin = nn.Linear(li, lo)

    def forward(self, x):
        x = x.view(1, -1)
        return self.lin(x)


class Distance(nn.Module):

    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, s, t):
        # L2 distance
        n, q = s.shape[0], t.shape[0]
        dist = (
            s.unsqueeze(1).expand(n, q, -1) -
            t.unsqueeze(0).expand(n, q, -1)
        ).pow(2).sum(dim=2)
        return dist


class MatchingNets(nn.Module):

    def __init__(self, fi, fo, lo):
        super(MatchingNets, self).__init__()
        self.embed = Embed(fi, fo)
        self.classify = Classifier(lo, lo)
        self.distance = Distance()

    def forward(self, s, t):
        s = self.embed(s)
        t = self.embed(t)
        dist = self.distance(s, t)
        pred = self.classify(dist)
        return pred


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.has_cuda else 'cpu')
    model = MatchingNets(1, 64, 10).to(device)
    summary(model, input_size=[(10, 1, 28, 28), (1, 1, 28, 28)])
