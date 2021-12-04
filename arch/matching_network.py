
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
        # x: (kq, nm, c, h, w)
        kq, nm, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.seq(x)
        x = x.view(-1, self.fo)
        # x: (kq * nm, fo)
        return x


class Classifier(nn.Module):

    def __init__(self, li, lo):
        super(Classifier, self).__init__()

    def forward(self, x):
        return x


class Distance(nn.Module):

    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, s, t):
        # L2 distance
        n, q = s.shape[0], t.shape[0]
        dist = (
            - s.unsqueeze(1).expand(n, q, -1)
            + t.unsqueeze(0).expand(n, q, -1)
        ).pow(2).sum(dim=2)
        return dist


class MatchingNets(nn.Module):

    def __init__(self, fi, fo, lo):
        super(MatchingNets, self).__init__()
        self.f = Embed(fi, fo)
        self.g = Embed(fi, fo)
        self.classify = Classifier(lo, lo)
        self.distance = Distance()

    def forward(self, s, t, y):
        # n-shot k-way m-shot q-way task
        # s: (k, n, c, h, w)
        # t: (q, m, c, h, w)
        k, n, _, _, _ = s.shape
        q, m, _, _, _ = t.shape
        s = self.f(s)
        t = self.g(t)
        dist = self.distance(s, t).T
        y_oh = torch.zeros(k * n, k).cuda().scatter(1,
                                                    y.type(torch.int64), 1)
        pred = torch.mm(dist, y_oh)
        return pred


if __name__ == '__main__':
    n = 10
    k = 8
    m = 5
    q = 12
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MatchingNets(1, 64, 10).to(device)
    summary(model, input_size=[(k, n, 1, 28, 28),
            (q, m, 1, 28, 28), (k * n, 1)])
